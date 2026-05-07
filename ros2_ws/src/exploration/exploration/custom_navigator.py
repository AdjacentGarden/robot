#!/usr/bin/env python3
"""
Lightweight custom navigator — replaces the entire Nav2 stack.

Global planner: A* with distance-weighted cost (paths stay centered in corridors)
Local planner:  Pure-pursuit path follower with costmap safety cascade
Recovery:       quick backup + curved escape

Inputs:  /map, /scan, TF(map→base_footprint)
Outputs: /cmd_vel, /navigation/path
"""

import heapq
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String, ColorRGBA, Bool
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


class CustomNavigator(Node):

    def __init__(self):
        super().__init__('custom_navigator')

        self._last_recovery_time = 0.0
        self._recovery_tier = 0

        # ── Robot ──
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('max_vel_x', 0.22)
        self.declare_parameter('min_vel_x', -0.10)
        self.declare_parameter('max_vel_theta', 0.7)
        self.declare_parameter('max_accel_x', 0.18)
        self.declare_parameter('max_decel_x', 0.28)
        self.declare_parameter('max_accel_theta', 1.0)
        self.declare_parameter('max_lateral_accel', 0.10)
        self.declare_parameter('cmd_smoothing_alpha', 0.35)
        self.declare_parameter('forward_angular_trim', 0.0)
        self.declare_parameter('min_cmd_linear_speed', 0.0)
        self.declare_parameter('min_cmd_angular_speed', 0.0)

        # ── Local planner ──
        self.declare_parameter('control_freq', 15.0)
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('lookahead_dist', 0.45)
        self.declare_parameter('safety_horizon', 0.5)
        self.declare_parameter('heading_kp', 1.4)
        self.declare_parameter('steering_deadband_angle', 0.08)
        self.declare_parameter('allow_in_place_rotation', False)
        self.declare_parameter('use_ackermann_model', False)
        self.declare_parameter('wheelbase', 0.32)
        self.declare_parameter('max_steer_angle', 0.45)
        self.declare_parameter('enable_fallback_turn_search', True)
        self.declare_parameter('min_turning_speed', 0.05)
        self.declare_parameter('fallback_turn_speed', 0.12)
        self.declare_parameter('turning_hysteresis_angle', 0.35)
        self.declare_parameter('sharp_turn_angle', 1.05)
        self.declare_parameter('very_sharp_turn_angle', 1.57)
        self.declare_parameter('sharp_turn_speed', 0.06)
        self.declare_parameter('very_sharp_turn_speed', 0.03)
        self.declare_parameter('large_heading_deg', 60.0)
        self.declare_parameter('behind_heading_deg', 120.0)
        self.declare_parameter('large_heading_release_deg', 20.0)
        self.declare_parameter('behind_heading_release_deg', 35.0)
        self.declare_parameter('moderate_lateral_error', 0.18)
        self.declare_parameter('recovery_lateral_error', 0.18)
        self.declare_parameter('fallback_cycles_before_active', 3)
        self.declare_parameter('state_min_hold_time', 0.35)
        self.declare_parameter('large_heading_recovery_angular', 1.05)
        self.declare_parameter('behind_heading_recovery_angular', 1.20)
        self.declare_parameter('large_heading_recovery_linear', 0.0)
        self.declare_parameter('fallback_in_place_angular', 0.85)
        self.declare_parameter('fallback_hold_cycles', 5)
        self.declare_parameter('fallback_recovery_guard_time', 1.2)

        # ── A* ──
        self.declare_parameter('astar_limit', 100000)
        self.declare_parameter('inflation_mult', 1.5)    # hard inflation = mult * robot_radius
        self.declare_parameter('proximity_weight', 3.0)   # cost weight for being near walls
        self.declare_parameter('goal_search_radius', 0.8)
        self.declare_parameter('goal_search_step', 0.1)

        # ── Recovery ──
        self.declare_parameter('stuck_timeout', 3.5)
        self.declare_parameter('stuck_radius', 0.12)
        self.declare_parameter('stuck_heading_progress_deg', 10.0)
        self.declare_parameter('enable_stuck_recovery', True)
        self.declare_parameter('max_recoveries', 6)
        self.declare_parameter('backup_vel', -0.10)
        self.declare_parameter('backup_time', 0.8)
        self.declare_parameter('spin_vel', 0.6)
        self.declare_parameter('spin_duration', 1.5)
        self.declare_parameter('recovery_curve_speed', 0.08)
        self.declare_parameter('recovery_curve_angular', 0.28)
        self.declare_parameter('recovery_curve_duration', 1.0)
        self.declare_parameter('enable_recovery_head_turn', True)
        self.declare_parameter('recovery_head_turn_angle_deg', 90.0)
        self.declare_parameter('recovery_head_turn_speed', 0.20)
        self.declare_parameter('recovery_head_turn_timeout', 8.0)
        self.declare_parameter('recovery_head_turn_sector_deg', 70.0)
        self.declare_parameter('recovery_head_turn_min_valid_points', 8)
        self.declare_parameter('recovery_head_turn_min_eval_range', 0.12)
        self.declare_parameter('recovery_head_turn_max_eval_range', 1.50)
        self.declare_parameter('recovery_head_turn_bias', 0.05)

        # ── Goal ──
        self.declare_parameter('goal_timeout', 90.0)
        self.declare_parameter('replan_interval', 5.0)
        self.declare_parameter('fallback_replan_count', 6)
        self.declare_parameter('enforce_goal_yaw', True)
        self.declare_parameter('goal_yaw_tolerance', 0.08)
        self.declare_parameter('goal_yaw_kp', 1.2)
        self.declare_parameter('goal_yaw_max_vel_theta', 0.35)
        self.declare_parameter('goal_yaw_timeout', 10.0)
        self.declare_parameter('goal_yaw_settle_time', 0.4)
        self.declare_parameter('debug_tracking', True)

        # ── TF fallback (optional open-loop pose integration) ──
        self.declare_parameter('use_open_loop_pose', False)
        self.declare_parameter('open_loop_tf_stall_timeout', 0.8)
        self.declare_parameter('open_loop_min_cmd_linear', 0.04)
        self.declare_parameter('open_loop_min_cmd_angular', 0.08)

        # ── Topics ──
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        p = self.get_parameter
        self.robot_r = float(p('robot_radius').value)
        self.max_vx = float(p('max_vel_x').value)
        self.min_vx = float(p('min_vel_x').value)
        self.max_vth = float(p('max_vel_theta').value)
        self.max_accel_x = float(p('max_accel_x').value)
        self.max_decel_x = float(p('max_decel_x').value)
        self.max_accel_theta = float(p('max_accel_theta').value)
        self.max_lat_accel = float(p('max_lateral_accel').value)
        self.cmd_smoothing_alpha = float(p('cmd_smoothing_alpha').value)
        self.forward_angular_trim = float(p('forward_angular_trim').value)
        self.min_cmd_linear_speed = float(p('min_cmd_linear_speed').value)
        self.min_cmd_angular_speed = float(p('min_cmd_angular_speed').value)

        self.ctrl_freq = float(p('control_freq').value)
        self.goal_tol = float(p('goal_tolerance').value)
        self.lookahead = float(p('lookahead_dist').value)
        self.safety_hz = float(p('safety_horizon').value)
        self.heading_kp = float(p('heading_kp').value)
        self.steering_deadband_angle = float(p('steering_deadband_angle').value)
        self.allow_in_place_rotation = bool(p('allow_in_place_rotation').value)
        self.use_ackermann_model = bool(p('use_ackermann_model').value)
        self.wheelbase = max(1e-3, float(p('wheelbase').value))
        self.max_steer_angle = max(1e-3, float(p('max_steer_angle').value))
        self.enable_fallback_turn_search = bool(p('enable_fallback_turn_search').value)
        self.min_turning_speed = float(p('min_turning_speed').value)
        self.fallback_turn_speed = float(p('fallback_turn_speed').value)
        self.turning_hysteresis_angle = float(p('turning_hysteresis_angle').value)
        self.sharp_turn_angle = float(p('sharp_turn_angle').value)
        self.very_sharp_turn_angle = float(p('very_sharp_turn_angle').value)
        self.sharp_turn_speed = float(p('sharp_turn_speed').value)
        self.very_sharp_turn_speed = float(p('very_sharp_turn_speed').value)
        self.large_heading_rad = math.radians(float(p('large_heading_deg').value))
        self.behind_heading_rad = math.radians(float(p('behind_heading_deg').value))
        self.large_heading_release_rad = math.radians(float(p('large_heading_release_deg').value))
        self.behind_heading_release_rad = math.radians(float(p('behind_heading_release_deg').value))
        self.moderate_lateral_error = float(p('moderate_lateral_error').value)
        self.recovery_lateral_error = float(p('recovery_lateral_error').value)
        self.fallback_cycles_before_active = max(1, int(p('fallback_cycles_before_active').value))
        self.state_min_hold_time = float(p('state_min_hold_time').value)
        self.large_heading_recovery_angular = float(p('large_heading_recovery_angular').value)
        self.behind_heading_recovery_angular = float(p('behind_heading_recovery_angular').value)
        self.large_heading_recovery_linear = float(p('large_heading_recovery_linear').value)
        self.fallback_in_place_angular = float(p('fallback_in_place_angular').value)
        self.fallback_hold_cycles = max(1, int(p('fallback_hold_cycles').value))
        self.fallback_recovery_guard_time = max(0.0, float(p('fallback_recovery_guard_time').value))

        self.astar_limit = int(p('astar_limit').value)
        self.inflate_mult = float(p('inflation_mult').value)
        self.prox_weight = float(p('proximity_weight').value)
        self.goal_search_radius = float(p('goal_search_radius').value)
        self.goal_search_step = float(p('goal_search_step').value)

        self.stuck_timeout = float(p('stuck_timeout').value)
        self.stuck_radius = float(p('stuck_radius').value)
        self.stuck_heading_progress_rad = math.radians(float(p('stuck_heading_progress_deg').value))
        self.enable_stuck_recovery = bool(p('enable_stuck_recovery').value)
        self.max_recover = int(p('max_recoveries').value)
        self.backup_vel = float(p('backup_vel').value)
        self.backup_time = float(p('backup_time').value)
        self.spin_vel = float(p('spin_vel').value)
        self.spin_dur = float(p('spin_duration').value)
        self.recovery_curve_speed = float(p('recovery_curve_speed').value)
        self.recovery_curve_angular = float(p('recovery_curve_angular').value)
        self.recovery_curve_dur = float(p('recovery_curve_duration').value)
        self.enable_recovery_head_turn = bool(p('enable_recovery_head_turn').value)
        self.recovery_head_turn_angle_deg = float(p('recovery_head_turn_angle_deg').value)
        self.recovery_head_turn_speed = float(p('recovery_head_turn_speed').value)
        self.recovery_head_turn_timeout = float(p('recovery_head_turn_timeout').value)
        self.recovery_head_turn_sector_deg = float(p('recovery_head_turn_sector_deg').value)
        self.recovery_head_turn_min_valid_points = int(p('recovery_head_turn_min_valid_points').value)
        self.recovery_head_turn_min_eval_range = float(p('recovery_head_turn_min_eval_range').value)
        self.recovery_head_turn_max_eval_range = float(p('recovery_head_turn_max_eval_range').value)
        self.recovery_head_turn_bias = float(p('recovery_head_turn_bias').value)

        self.goal_timeout = float(p('goal_timeout').value)
        self.replan_sec = float(p('replan_interval').value)
        self.fallback_replan_count = int(p('fallback_replan_count').value)
        self.enforce_goal_yaw = bool(p('enforce_goal_yaw').value)
        self.goal_yaw_tol = float(p('goal_yaw_tolerance').value)
        self.goal_yaw_kp = float(p('goal_yaw_kp').value)
        self.goal_yaw_max_w = float(p('goal_yaw_max_vel_theta').value)
        self.goal_yaw_timeout = float(p('goal_yaw_timeout').value)
        self.goal_yaw_settle_time = float(p('goal_yaw_settle_time').value)
        self.debug_tracking = bool(p('debug_tracking').value)
        self.use_open_loop_pose = bool(p('use_open_loop_pose').value)
        self.open_loop_tf_stall_timeout = float(p('open_loop_tf_stall_timeout').value)
        self.open_loop_min_cmd_linear = float(p('open_loop_min_cmd_linear').value)
        self.open_loop_min_cmd_angular = float(p('open_loop_min_cmd_angular').value)

        cmd_topic = str(p('cmd_vel_topic').value)

        # ── State ──
        self.map_msg: OccupancyGrid | None = None
        self.scan_msg: LaserScan | None = None
        self.cur_v = 0.0
        self.cur_w = 0.0
        self._nav_lock = threading.Lock()
        self._preempt = threading.Event()
        self._last_cmd_time = time.time()
        self._last_turn_sign = 0.0
        self._last_path_idx = 0
        self._last_pose_for_debug = None
        self._last_pose_time_for_debug = None
        self._recovery_turn_sign = -1.0
        self._open_loop_active = False
        self._open_loop_pose = None
        self._tf_stall_since = None
        self._last_tf_pose = None
        self._last_loop_time = time.time()
        self._control_state = 'NORMAL_FOLLOW'
        self._control_state_reason = 'startup'
        self._control_state_since = time.time()
        self._fallback_cycle_counter = 0
        self._fallback_guard_until = 0.0

        # ── Distance-field costmap (updated from /map) ──
        self._dist_field = None
        self._dist_res = 0.05
        self._dist_ox = 0.0
        self._dist_oy = 0.0
        self._dist_w = 0
        self._dist_h = 0
        self._last_dist_update = 0.0

        # ── TF ──
        self.tf_buf = Buffer()
        self.tf_lis = TransformListener(self.tf_buf, self)

        from rclpy.qos import QoSProfile, QoSDurabilityPolicy
        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # ── Subscriptions ──
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._on_map, latched_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)

        # ── Publishers ──
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, '/navigation/path', 10)
        self.local_traj_pub = self.create_publisher(
            Marker, '/navigation/local_traj', 10)
        self.local_goal_pub = self.create_publisher(
            Marker, '/navigation/local_goal', 10)
        self.status_pub = self.create_publisher(
            String, '/navigation/status', 10)

        # ── Subscribers ──
        self.is_forced_rest = False
        self.rest_sub = self.create_subscription(
            Bool, '/is_forced_rest', self._on_forced_rest, 10)

        # ── Action Server ──
        self._action_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            execute_callback=self._execute_nav,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=self._cancel_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            f'CustomNavigator ready | v={self.max_vx} w={self.max_vth} '
            f'inflate={self.inflate_mult}x prox_w={self.prox_weight} '
            f'cv2={_CV2} cmd_topic={cmd_topic}')

    # ═══════════════════ Callbacks ═══════════════════

    def _on_map(self, msg):
        self.map_msg = msg
        now = time.time()
        if _CV2 and now - self._last_dist_update > 0.5:
            self._update_dist_field(msg)
            self._last_dist_update = now

    def _on_scan(self, msg):
        self.scan_msg = msg

    def _on_forced_rest(self, msg):
        self.is_forced_rest = msg.data

    def _cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    def _pose(self):
        try:
            tf = self.tf_buf.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z))
            return (t.x, t.y, yaw)
        except TransformException:
            return None

    # ═══════════════════ Navigation ═══════════════════

    def _execute_nav(self, goal_handle):
        self._preempt.set()
        with self._nav_lock:
            self._preempt.clear()
            return self._run_nav(goal_handle)

    def _run_nav(self, goal_handle):
        gx = goal_handle.request.pose.pose.position.x
        gy = goal_handle.request.pose.pose.position.y
        q = goal_handle.request.pose.pose.orientation
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        goal_has_yaw = q_norm > 1e-6
        goal_yaw = 0.0
        if goal_has_yaw:
            qx = q.x / q_norm
            qy = q.y / q_norm
            qz = q.z / q_norm
            qw = q.w / q_norm
            goal_yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz))

        if goal_has_yaw:
            self.get_logger().info(
                f'Goal: ({gx:.2f}, {gy:.2f}, yaw={goal_yaw:.3f})')
        else:
            self.get_logger().warn(
                f'Goal: ({gx:.2f}, {gy:.2f}) without valid orientation quaternion; '
                f'skipping final yaw alignment')

        self.cur_v = self.cur_w = 0.0
        self._last_turn_sign = 0.0
        self._last_path_idx = 0
        self._last_pose_for_debug = None
        self._last_pose_time_for_debug = None
        self._recovery_turn_sign = -1.0
        pose = self._pose()
        if not pose or not self.map_msg:
            if not pose:
                self.get_logger().warn('Navigation aborted: missing TF pose.')
            if not self.map_msg:
                self.get_logger().warn('Navigation aborted: missing map_msg.')
            goal_handle.abort(); self._stop()
            return NavigateToPose.Result()

        self._open_loop_active = False
        self._open_loop_pose = pose
        self._tf_stall_since = None
        self._last_tf_pose = pose
        self._last_loop_time = time.time()
        self._fallback_guard_until = 0.0

        path = self._plan_astar(pose[0], pose[1], gx, gy)
        if not path:
            self.get_logger().warn('A* failed')
            goal_handle.abort(); self._stop()
            return NavigateToPose.Result()
        self._pub_path(path)
        nav_gx, nav_gy = path[-1]
        snap_delta = math.hypot(nav_gx - gx, nav_gy - gy)
        if snap_delta > 0.05:
            self.get_logger().warn(
                f'Planning goal adjusted by {snap_delta:.2f}m '
                f'({gx:.2f},{gy:.2f}) -> ({nav_gx:.2f},{nav_gy:.2f})')

        dt = 1.0 / self.ctrl_freq
        t0 = time.time()
        last_replan = t0
        n_rec = 0
        prog_pos = (pose[0], pose[1])
        prog_yaw = pose[2]
        prog_t = t0
        fallback_count = 0

        while rclpy.ok():
            if self._preempt.is_set():
                goal_handle.abort(); self._stop()
                return NavigateToPose.Result()
            if goal_handle.is_cancel_requested:
                goal_handle.canceled(); self._stop()
                return NavigateToPose.Result()
            if time.time() - t0 > self.goal_timeout:
                self.get_logger().warn('Goal timeout')
                goal_handle.abort(); self._stop()
                return NavigateToPose.Result()

            now = time.time()
            dt_loop = max(1e-3, now - self._last_loop_time)
            self._last_loop_time = now

            pose = self._resolve_pose(dt_loop)
            if not pose:
                time.sleep(dt); continue
            rx, ry, ryaw = pose

            d2g_req = math.hypot(gx - rx, gy - ry)
            d2g_nav = math.hypot(nav_gx - rx, nav_gy - ry)
            d2g = min(d2g_req, d2g_nav)
            if d2g < self.goal_tol:
                if goal_has_yaw:
                    align_state = self._final_align_to_goal_yaw(goal_handle, goal_yaw)
                    if align_state == 'canceled':
                        goal_handle.canceled()
                        self._stop()
                        return NavigateToPose.Result()
                    if align_state == 'preempted':
                        goal_handle.abort()
                        self._stop()
                        return NavigateToPose.Result()
                    if align_state == 'timeout' and self.enforce_goal_yaw:
                        self.get_logger().warn(
                            'Goal position reached but final yaw alignment timed out')
                        goal_handle.abort()
                        self._stop()
                        return NavigateToPose.Result()
                    if align_state == 'timeout' and not self.enforce_goal_yaw:
                        self.get_logger().warn(
                            'Goal position reached; final yaw alignment timed out but '
                            'enforce_goal_yaw is false, finishing goal')

                self.get_logger().info('Goal reached')
                self._stop(); goal_handle.succeed()
                with open(r'/home/test/code/final_0418/function/pet_navigation_result.txt', 'w') as f:
                    f.write(f"success")
                return NavigateToPose.Result()

            # Stuck: translation OR rotation counts as progress
            dp = math.hypot(rx - prog_pos[0], ry - prog_pos[1])
            da = abs(self._angle_diff(prog_yaw, ryaw))
            
            # Reset stuck timer if making progress OR if forced rest is active
            fallback_guard_active = time.time() < self._fallback_guard_until
            if (dp > self.stuck_radius or
                    da > self.stuck_heading_progress_rad or
                    self.is_forced_rest or fallback_guard_active):
                prog_pos, prog_yaw, prog_t = (rx, ry), ryaw, time.time()
            elif time.time() - prog_t > self.stuck_timeout:
                if not self.enable_stuck_recovery:
                    self.get_logger().warn('Stuck detected, stopping because stuck recovery is disabled')
                    goal_handle.abort(); self._stop()
                    return NavigateToPose.Result()
                n_rec += 1
                if n_rec > self.max_recover:
                    self.get_logger().warn('Max recoveries')
                    goal_handle.abort(); self._stop()
                    return NavigateToPose.Result()
                recovery_reason = (
                    f'stuck_timeout dt={time.time() - prog_t:.2f}s '
                    f'dp={dp:.3f}m da={math.degrees(da):.1f}deg'
                )
                lateral_error, _ = self._path_metrics(rx, ry, path)
                self.get_logger().warn(
                    f'[RECOVERY] Triggered {n_rec}/{self.max_recover} reason={recovery_reason}')
                self._recovery(goal_handle, reason=recovery_reason, lateral_error=lateral_error)
                p2 = self._pose()
                if p2:
                    prog_pos, prog_yaw = (p2[0], p2[1]), p2[2]
                prog_t = time.time()
                new_path = self._plan_astar(
                    p2[0] if p2 else rx, p2[1] if p2 else ry, gx, gy)
                if new_path:
                    self._last_path_idx = 0
                    path = new_path; self._pub_path(path)
                    nav_gx, nav_gy = path[-1]
                last_replan = time.time()
                continue

            # Periodic replan
            now = time.time()
            if now - last_replan > self.replan_sec:
                new_path = self._plan_astar(rx, ry, gx, gy)
                if new_path:
                    self._last_path_idx = 0
                    path = new_path; self._pub_path(path)
                    nav_gx, nav_gy = path[-1]
                last_replan = now

            decision = self._local_plan(rx, ry, ryaw, path, d2g_nav)
            v = float(decision['linear_x'])
            w = float(decision['angular_z'])
            mode = str(decision['state'])

            if mode == 'FALLBACK':
                fallback_count += 1
                self._fallback_guard_until = time.time() + self.fallback_recovery_guard_time
            else:
                fallback_count = 0

            if fallback_count >= self.fallback_replan_count:
                self.get_logger().warn(
                    f'[FALLBACK] persisted {fallback_count} cycles -> force replan '
                    f'reason={decision["reason"]}')
                new_path = self._plan_astar(rx, ry, gx, gy)
                if new_path:
                    self._last_path_idx = 0
                    path = new_path
                    self._pub_path(path)
                    nav_gx, nav_gy = path[-1]
                    last_replan = time.time()
                    fallback_count = 0
                    continue

            v, w = self._apply_dynamics_limits(v, w)
            v, w = self._apply_min_speed(v, w)
            cmd = Twist()
            cmd.linear.x = v; cmd.angular.z = w
            self.cmd_pub.publish(cmd)
            self.cur_v, self.cur_w = v, w

            self._log_motion_command(decision, v, w)

            # Publish local trajectory visualization
            self._pub_local_traj(rx, ry, ryaw, v, w)

            fb = NavigateToPose.Feedback()
            fb.current_pose.header.frame_id = 'map'
            fb.current_pose.pose.position.x = rx
            fb.current_pose.pose.position.y = ry
            fb.distance_remaining = float(d2g)
            fb.number_of_recoveries = n_rec
            goal_handle.publish_feedback(fb)

            time.sleep(dt)

        self._stop(); goal_handle.abort()
        return NavigateToPose.Result()

    def _resolve_pose(self, dt_loop):
        tf_pose = self._pose()
        if not self.use_open_loop_pose:
            return tf_pose

        cmd_active = (
            abs(self.cur_v) >= self.open_loop_min_cmd_linear or
            abs(self.cur_w) >= self.open_loop_min_cmd_angular
        )

        if tf_pose:
            if self._last_tf_pose is not None:
                dx = tf_pose[0] - self._last_tf_pose[0]
                dy = tf_pose[1] - self._last_tf_pose[1]
                dyaw = self._angle_diff(self._last_tf_pose[2], tf_pose[2])
                moved = math.hypot(dx, dy) > 1e-3 or abs(dyaw) > 0.01
                if cmd_active and not moved:
                    if self._tf_stall_since is None:
                        self._tf_stall_since = time.time()
                    elif (time.time() - self._tf_stall_since) > self.open_loop_tf_stall_timeout:
                        if not self._open_loop_active:
                            self._open_loop_active = True
                            if self._open_loop_pose is None:
                                self._open_loop_pose = tf_pose
                            self.get_logger().warn(
                                'TF pose appears stalled during motion, switching to open-loop pose fallback')
                else:
                    self._tf_stall_since = None
                    if moved and self._open_loop_active:
                        self._open_loop_active = False
                        self._open_loop_pose = tf_pose
                        self.get_logger().info('TF pose recovered, leaving open-loop pose fallback')
            self._last_tf_pose = tf_pose
            if not self._open_loop_active:
                self._open_loop_pose = tf_pose
                return tf_pose

        if self._open_loop_pose is None:
            return tf_pose

        self._open_loop_pose = self._integrate_pose(self._open_loop_pose, self.cur_v, self.cur_w, dt_loop)
        return self._open_loop_pose

    @staticmethod
    def _integrate_pose(pose, v, w, dt):
        x, y, yaw = pose
        if abs(w) > 1e-6:
            x += v / w * (math.sin(yaw + w * dt) - math.sin(yaw))
            y += v / w * (-math.cos(yaw + w * dt) + math.cos(yaw))
            yaw += w * dt
        else:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        return (x, y, yaw)

    def _apply_min_speed(self, v, w):
        if v != 0.0 and self.min_cmd_linear_speed > 0.0:
            v = math.copysign(max(abs(v), self.min_cmd_linear_speed), v)
        if w != 0.0 and self.min_cmd_angular_speed > 0.0:
            w = math.copysign(max(abs(w), self.min_cmd_angular_speed), w)
        return v, w

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.cur_v = self.cur_w = 0.0
        self._last_cmd_time = time.time()

    def _final_align_to_goal_yaw(self, goal_handle, goal_yaw):
        dt = 1.0 / self.ctrl_freq
        start_t = time.time()
        settle_start = None
        max_w = min(self.max_vth, max(0.0, self.goal_yaw_max_w))

        while rclpy.ok():
            if self._preempt.is_set():
                self._stop()
                return 'preempted'
            if goal_handle.is_cancel_requested:
                self._stop()
                return 'canceled'

            pose = self._resolve_pose(dt)
            if not pose:
                time.sleep(dt)
                continue

            yaw_err = self._angle_diff(pose[2], goal_yaw)
            abs_err = abs(yaw_err)
            if abs_err <= self.goal_yaw_tol:
                if settle_start is None:
                    settle_start = time.time()
                self._stop()
                if time.time() - settle_start >= self.goal_yaw_settle_time:
                    self.get_logger().info(
                        f'Final yaw aligned: err={yaw_err:.3f} rad '
                        f'(tol={self.goal_yaw_tol:.3f})')
                    return 'aligned'
                time.sleep(dt)
                continue

            settle_start = None
            if time.time() - start_t > self.goal_yaw_timeout:
                self._stop()
                return 'timeout'

            w = self.goal_yaw_kp * yaw_err
            if max_w > 0.0:
                w = max(-max_w, min(max_w, w))
            _, w = self._apply_dynamics_limits(0.0, w)
            _, w = self._apply_min_speed(0.0, w)

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = w
            self.cmd_pub.publish(cmd)
            self.cur_v = 0.0
            self.cur_w = w

            self.get_logger().info(
                f'final_yaw_align err={yaw_err:.3f} cmd_w={w:.3f}',
                throttle_duration_sec=0.5)
            time.sleep(dt)

        self._stop()
        return 'timeout'

    # ═══════════════════ Distance-Field Costmap ═══════════════════

    def _update_dist_field(self, map_msg=None):
        if map_msg is None:
            map_msg = self.map_msg
        if not map_msg:
            return
        info = map_msg.info
        w, h, res = info.width, info.height, info.resolution
        grid = np.array(map_msg.data, dtype=np.int8).reshape(h, w)
        occ = (grid > 65).astype(np.uint8)
        free = ((1 - occ) * 255).astype(np.uint8)
        dist_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        self._dist_field = dist_px.astype(np.float32) * res
        self._dist_res = res
        self._dist_ox = info.origin.position.x
        self._dist_oy = info.origin.position.y
        self._dist_w = w
        self._dist_h = h

    def _dist_at(self, x, y):
        """Distance to nearest obstacle at world (x, y)."""
        if self._dist_field is None:
            return float('inf')
        gx = int((x - self._dist_ox) / self._dist_res)
        gy = int((y - self._dist_oy) / self._dist_res)
        if 0 <= gx < self._dist_w and 0 <= gy < self._dist_h:
            return float(self._dist_field[gy, gx])
        return 0.0

    def _check_cmd(self, rx, ry, ryaw, v, w):
        """Simulate (v, w), return min distance to obstacles along trajectory."""
        n = max(3, int(self.safety_hz / 0.05))
        x, y, yaw = rx, ry, ryaw
        dt = self.safety_hz / n
        min_d = float('inf')
        for _ in range(n):
            if abs(w) > 1e-6:
                x += v / w * (math.sin(yaw + w * dt) - math.sin(yaw))
                y += v / w * (-math.cos(yaw + w * dt) + math.cos(yaw))
                yaw += w * dt
            else:
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
            d = self._dist_at(x, y)
            min_d = min(min_d, d)
        return min_d

    # ═══════════════════ Pure-Pursuit Local Planner ═══════════════════

    def _find_local_goal(self, rx, ry, ryaw, path, lookahead=None):
        if lookahead is None:
            lookahead = self.lookahead
        if not path:
            return None
        start_idx = min(self._last_path_idx, max(0, len(path) - 1))
        min_d, closest = float('inf'), start_idx
        for i in range(start_idx, len(path)):
            px, py = path[i]
            d = math.hypot(px - rx, py - ry)
            if d < min_d:
                min_d = d
                closest = i
        self._last_path_idx = max(self._last_path_idx, closest)
        accum = 0.0
        for i in range(closest, len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            accum += math.hypot(dx, dy)
            if accum >= lookahead:
                return self._prefer_forward_point(rx, ry, ryaw, path, i + 1)
        return self._prefer_forward_point(rx, ry, ryaw, path, len(path) - 1)

    def _prefer_forward_point(self, rx, ry, ryaw, path, idx):
        for j in range(idx, len(path)):
            px, py = path[j]
            if ((px - rx) * math.cos(ryaw) + (py - ry) * math.sin(ryaw)) >= -0.02:
                return path[j]
        return path[idx]

    def _stabilize_turn_error(self, err):
        if abs(err) < self.steering_deadband_angle:
            self._last_turn_sign = 0.0
            return 0.0
        if abs(abs(err) - math.pi) < self.turning_hysteresis_angle and self._last_turn_sign != 0.0:
            err = math.copysign(abs(err), self._last_turn_sign)
        sign = math.copysign(1.0, err) if abs(err) > 1e-3 else self._last_turn_sign
        self._last_turn_sign = sign if abs(err) > 0.05 else 0.0
        return err

    def _curve_speed(self, raw_v, err, d2g):
        v = raw_v
        if abs(err) > 0.15:
            v = max(v, self.min_turning_speed)
        if abs(err) > self.sharp_turn_angle:
            v = min(v, max(self.sharp_turn_speed, self.min_turning_speed))
        if abs(err) > self.very_sharp_turn_angle:
            v = min(v, max(self.very_sharp_turn_speed, self.min_turning_speed))
        if d2g < 0.4:
            v = min(v, self.max_vx * max(0.04, d2g / 0.4))
            if abs(err) > 0.15 and d2g > self.goal_tol:
                v = max(v, self.min_turning_speed)
        return v

    def _pursuit_vw(self, ryaw, lg, ry, rx, d2g):
        """Compute (v, w) toward local goal point."""
        a2g = math.atan2(lg[1] - ry, lg[0] - rx)
        err = self._stabilize_turn_error(self._angle_diff(ryaw, a2g))
        align = max(0.0, math.cos(err))
        v = self._curve_speed(self.max_vx * max(0.04, align), err, d2g)
        if self.use_ackermann_model:
            ld = max(0.10, math.hypot(lg[0] - rx, lg[1] - ry))
            curvature = 2.0 * math.sin(err) / ld
            max_curvature = math.tan(self.max_steer_angle) / self.wheelbase
            curvature = max(-max_curvature, min(max_curvature, curvature))
            w = v * curvature
        else:
            w = self.heading_kp * err
        w = max(-self.max_vth, min(self.max_vth, w))
        return v, w

    def _limit_for_curvature(self, v, w):
        if self.max_lat_accel <= 0.0 or abs(v) < 1e-6 or abs(w) < 1e-6:
            return v
        max_v = self.max_lat_accel / abs(w)
        if v >= 0.0:
            return min(v, max_v)
        return max(v, -max_v)

    def _apply_dynamics_limits(self, v, w):
        now = time.time()
        dt = max(1.0 / max(self.ctrl_freq, 1.0), now - self._last_cmd_time)
        self._last_cmd_time = now

        v = max(self.min_vx, min(self.max_vx, v))
        w = max(-self.max_vth, min(self.max_vth, w))
        v = self._limit_for_curvature(v, w)

        dv = v - self.cur_v
        max_dv = (self.max_accel_x if dv >= 0.0 else self.max_decel_x) * dt
        if abs(dv) > max_dv:
            v = self.cur_v + math.copysign(max_dv, dv)

        dw = w - self.cur_w
        max_dw = self.max_accel_theta * dt
        if abs(dw) > max_dw:
            w = self.cur_w + math.copysign(max_dw, dw)

        if abs(v) > 0.04:
            w += self.forward_angular_trim

        alpha = max(0.0, min(1.0, self.cmd_smoothing_alpha))
        if alpha > 0.0:
            v = (1.0 - alpha) * self.cur_v + alpha * v
            w = (1.0 - alpha) * self.cur_w + alpha * w

        if abs(w) < 1e-3:
            w = 0.0
        return v, w

    def _publish_nav_status(self, state, reason):
        msg = String()
        msg.data = f'{state}:{reason}'
        self.status_pub.publish(msg)

    def _log_motion_command(self, decision, cmd_v, cmd_w):
        state = decision['state']
        reason = decision['reason']
        self._publish_nav_status(state, reason)
        if not self.debug_tracking:
            return
        self.get_logger().info(
            f'[{state}] reason={reason} '
            f'angle_error={math.degrees(decision["angle_error"]):.1f}deg '
            f'heading_error={math.degrees(decision["heading_error"]):.1f}deg '
            f'lateral_error={decision["lateral_error"]:.3f}m '
            f'target_direction={math.degrees(decision["target_direction"]):.1f}deg '
            f'linear_x={cmd_v:.3f} angular_z={cmd_w:.3f} '
            f'allow_backward={str(decision["allow_backward"]).lower()}',
            throttle_duration_sec=0.20)

    def _build_decision(self, state, reason, angle_error, lateral_error, target_direction,
                        linear_x, angular_z, allow_backward, lookahead=None):
        return {
            'state': state,
            'reason': reason,
            'angle_error': angle_error,
            'heading_error': angle_error,
            'lateral_error': lateral_error,
            'distance_to_path': lateral_error,
            'target_direction': target_direction,
            'linear_x': linear_x,
            'angular_z': angular_z,
            'allow_backward': allow_backward,
            'lookahead': lookahead,
        }

    def _path_metrics(self, rx, ry, path):
        if not path:
            return 0.0, 0
        start_idx = min(self._last_path_idx, max(0, len(path) - 1))
        min_d = float('inf')
        closest = start_idx
        for i in range(start_idx, len(path)):
            px, py = path[i]
            d = math.hypot(px - rx, py - ry)
            if d < min_d:
                min_d = d
                closest = i
        return min_d, closest

    def _apply_control_state_hysteresis(self, desired_state, reason, abs_err, target_behind):
        now = time.time()
        if self._control_state == 'LARGE_HEADING_RECOVERY' and desired_state != 'LARGE_HEADING_RECOVERY':
            release_err = self.behind_heading_release_rad if target_behind else self.large_heading_release_rad
            if abs_err > release_err or (now - self._control_state_since) < self.state_min_hold_time:
                return self._control_state, f'hold_{self._control_state_reason}'

        if self._control_state == 'FALLBACK' and desired_state != 'FALLBACK':
            if self._fallback_cycle_counter < self.fallback_hold_cycles and (
                    now - self._control_state_since) < self.state_min_hold_time:
                return self._control_state, f'hold_{self._control_state_reason}'

        if desired_state != self._control_state:
            self._control_state = desired_state
            self._control_state_reason = reason
            self._control_state_since = now
        else:
            self._control_state_reason = reason

        return self._control_state, reason

    def _classify_follow_state(self, heading_err, lateral_error, forward_projection):
        abs_err = abs(heading_err)
        target_behind = abs_err >= self.behind_heading_rad or forward_projection < -0.05
        severe_path_error = lateral_error >= self.moderate_lateral_error
        recovery_path_error = lateral_error >= self.recovery_lateral_error
        severe_heading_error = abs_err >= self.large_heading_rad

        if target_behind and recovery_path_error:
            desired_state = 'LARGE_HEADING_RECOVERY'
            reason = 'target_behind_off_path'
        elif severe_heading_error and recovery_path_error:
            desired_state = 'LARGE_HEADING_RECOVERY'
            reason = 'large_heading_off_path'
        else:
            desired_state = 'NORMAL_FOLLOW'
            reason = 'path_tracking_stable' if not severe_path_error else 'path_error_drive_through'

        desired_state, reason = self._apply_control_state_hysteresis(
            desired_state, reason, abs_err, target_behind)
        return desired_state, reason, target_behind

    def _command_from_state(self, state, heading_err, d2g, target_behind):
        abs_err = abs(heading_err)
        sign = math.copysign(1.0, heading_err) if abs_err > 1e-6 else 0.0

        if state == 'LARGE_HEADING_RECOVERY':
            max_w = self.behind_heading_recovery_angular if target_behind else self.large_heading_recovery_angular
            w = sign * min(max_w, self.max_vth)
            if self.allow_in_place_rotation:
                v = self.large_heading_recovery_linear
            else:
                v = max(self.min_turning_speed, self.max_vx * 0.10)
            return v, w

        align = max(0.0, math.cos(heading_err))
        base_v = self._curve_speed(self.max_vx * max(0.08, align), heading_err, d2g)
        v = base_v
        w = self.heading_kp * heading_err
        return v, max(-self.max_vth, min(self.max_vth, w))

    def _candidate_speed_scales(self, state):
        if state == 'NORMAL_FOLLOW':
            return [1.0, 0.85, 0.70]
        return [1.0]

    def _ordered_turn_trials(self, preferred_sign):
        sign = preferred_sign if preferred_sign in (-1.0, 1.0) else self._recovery_turn_sign
        return [
            sign * 0.45, -sign * 0.45,
            sign * 0.70, -sign * 0.70,
            sign * min(self.fallback_in_place_angular, self.max_vth),
            -sign * min(self.fallback_in_place_angular, self.max_vth),
        ]

    def _local_plan(self, rx, ry, ryaw, path, d2g):
        """Pure pursuit with large-error intervention and conservative fallback."""
        if not path:
            return self._build_decision(
                'FALLBACK', 'path_missing', 0.0, 0.0, ryaw, 0.0, 0.0, False)

        col_r = self.robot_r * 0.6
        lateral_error, _ = self._path_metrics(rx, ry, path)

        if d2g < self.goal_tol * 1.8 and path:
            gx, gy = path[-1]
            self._pub_local_goal(gx, gy)
            target_direction = math.atan2(gy - ry, gx - rx)
            heading_err = self._stabilize_turn_error(self._angle_diff(ryaw, target_direction))
            state, reason, target_behind = self._classify_follow_state(
                heading_err, lateral_error, (gx - rx) * math.cos(ryaw) + (gy - ry) * math.sin(ryaw))
            v, w = self._command_from_state(state, heading_err, d2g, target_behind)
            return self._build_decision(
                state, f'goal_zone_{reason}', heading_err, lateral_error, target_direction, v, w, False)

        lookaheads = [self.lookahead, max(0.18, self.lookahead * 0.7), max(0.10, self.lookahead * 0.4)]
        best_blocked = None

        for la in lookaheads:
            lg = self._find_local_goal(rx, ry, ryaw, path, la)
            if not lg:
                continue
            self._pub_local_goal(lg[0], lg[1])
            target_direction = math.atan2(lg[1] - ry, lg[0] - rx)
            heading_err = self._stabilize_turn_error(self._angle_diff(ryaw, target_direction))
            forward_projection = (lg[0] - rx) * math.cos(ryaw) + (lg[1] - ry) * math.sin(ryaw)
            state, reason, target_behind = self._classify_follow_state(
                heading_err, lateral_error, forward_projection)
            v, w = self._command_from_state(state, heading_err, d2g, target_behind)

            if self._dist_field is None:
                self._fallback_cycle_counter = 0
                return self._build_decision(
                    state, f'no_costmap_{reason}', heading_err, lateral_error,
                    target_direction, v, w, False, la)

            safe = False
            if state == 'LARGE_HEADING_RECOVERY':
                safe = self._check_cmd(rx, ry, ryaw, v, w) >= col_r
            else:
                for scale in self._candidate_speed_scales(state):
                    v_try = v * scale
                    if scale < 1.0:
                        v_try = max(0.0, v_try)
                    if self._check_cmd(rx, ry, ryaw, v_try, w) >= col_r:
                        self._fallback_cycle_counter = 0
                        return self._build_decision(
                            state, reason, heading_err, lateral_error,
                            target_direction, v_try, w, False, la)

            if safe:
                self._fallback_cycle_counter = 0
                return self._build_decision(
                    state, reason, heading_err, lateral_error,
                    target_direction, v, w, False, la)

            best_blocked = self._build_decision(
                state, f'blocked_{reason}_la_{la:.2f}', heading_err, lateral_error,
                target_direction, 0.0, 0.0, False, la)

        if best_blocked is None:
            return self._build_decision(
                'FALLBACK', 'no_local_goal', 0.0, lateral_error, ryaw, 0.0, 0.0, False)

        self._fallback_cycle_counter += 1
        if self._fallback_cycle_counter < self.fallback_cycles_before_active:
            wait_reason = (
                f'blocked_hysteresis_{self._fallback_cycle_counter}/'
                f'{self.fallback_cycles_before_active}_{best_blocked["reason"]}'
            )
            return self._build_decision(
                best_blocked['state'], wait_reason, best_blocked['angle_error'],
                best_blocked['lateral_error'], best_blocked['target_direction'],
                0.0, 0.0, False, best_blocked['lookahead'])

        if not self.enable_fallback_turn_search:
            return self._build_decision(
                'FALLBACK', f'fallback_disabled_{best_blocked["reason"]}',
                best_blocked['angle_error'], best_blocked['lateral_error'],
                best_blocked['target_direction'], 0.0, 0.0, False, best_blocked['lookahead'])

        preferred_sign = self._last_turn_sign if self._last_turn_sign != 0.0 else (
            math.copysign(1.0, self.cur_w) if abs(self.cur_w) > 1e-3 else self._recovery_turn_sign)
        for w_try in self._ordered_turn_trials(preferred_sign):
            if self.allow_in_place_rotation and self._check_cmd(rx, ry, ryaw, 0.0, w_try) >= col_r:
                self._control_state, reason = self._apply_control_state_hysteresis(
                    'FALLBACK', 'fallback_in_place_turn', abs(best_blocked['angle_error']), False)
                return self._build_decision(
                    'FALLBACK', reason, best_blocked['angle_error'], best_blocked['lateral_error'],
                    best_blocked['target_direction'], 0.0, w_try, False, best_blocked['lookahead'])

            turn_v = max(0.0, min(self.fallback_turn_speed, self.max_vx * 0.30))
            if self._check_cmd(rx, ry, ryaw, turn_v, w_try) >= col_r:
                self._control_state, reason = self._apply_control_state_hysteresis(
                    'FALLBACK', 'fallback_drive_turn', abs(best_blocked['angle_error']), False)
                return self._build_decision(
                    'FALLBACK', reason, best_blocked['angle_error'], best_blocked['lateral_error'],
                    best_blocked['target_direction'], turn_v, w_try, False, best_blocked['lookahead'])

        self._control_state, reason = self._apply_control_state_hysteresis(
            'FALLBACK', 'fallback_no_safe_cmd', abs(best_blocked['angle_error']), False)
        return self._build_decision(
            'FALLBACK', reason, best_blocked['angle_error'], best_blocked['lateral_error'],
            best_blocked['target_direction'], 0.0, 0.0, False, best_blocked['lookahead'])

    @staticmethod
    def _angle_diff(a, b):
        d = b - a
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _score_side_clearance(self, values):
        if len(values) < self.recovery_head_turn_min_valid_points:
            return None
        mn = min(values)
        avg = sum(values) / float(len(values))
        return 0.6 * avg + 0.4 * mn

    def _scan_side_clearance(self):
        msg = self.scan_msg
        if msg is None or not msg.ranges:
            return None, None, 0, 0

        sector = math.radians(max(5.0, min(85.0, self.recovery_head_turn_sector_deg)))
        min_eval = max(float(msg.range_min), self.recovery_head_turn_min_eval_range)
        max_eval = min(float(msg.range_max), self.recovery_head_turn_max_eval_range)
        if max_eval <= min_eval:
            max_eval = float(msg.range_max)

        left_vals = []
        right_vals = []
        ang = float(msg.angle_min)
        inc = float(msg.angle_increment)

        for r in msg.ranges:
            rr = float(r)
            if math.isfinite(rr) and min_eval <= rr <= max_eval:
                if 0.0 < ang <= sector:
                    left_vals.append(rr)
                elif -sector <= ang < 0.0:
                    right_vals.append(rr)
            ang += inc

        left_score = self._score_side_clearance(left_vals)
        right_score = self._score_side_clearance(right_vals)
        return left_score, right_score, len(left_vals), len(right_vals)

    def _default_recovery_turn_sign(self):
        if self._last_turn_sign != 0.0:
            return self._last_turn_sign
        if abs(self.cur_w) > 1e-3:
            return math.copysign(1.0, self.cur_w)
        self._recovery_turn_sign *= -1.0
        return self._recovery_turn_sign

    def _choose_recovery_turn_sign(self):
        turn_sign = self._default_recovery_turn_sign()
        left_score, right_score, left_n, right_n = self._scan_side_clearance()

        if left_score is None or right_score is None:
            self.get_logger().info(
                f'Recovery head-turn fallback sign={turn_sign:+.0f} '
                f'(scan points left={left_n}, right={right_n})')
            self._recovery_turn_sign = turn_sign
            return turn_sign

        if left_score > right_score + self.recovery_head_turn_bias:
            turn_sign = 1.0
        elif right_score > left_score + self.recovery_head_turn_bias:
            turn_sign = -1.0

        self.get_logger().info(
            f'Recovery head-turn choose {"left" if turn_sign > 0 else "right"} '
            f'left_score={left_score:.3f} right_score={right_score:.3f} '
            f'left_n={left_n} right_n={right_n}')
        self._recovery_turn_sign = turn_sign
        return turn_sign

    def _perform_recovery_head_turn(self, goal_handle, turn_sign):
        dt = 1.0 / self.ctrl_freq
        target = math.radians(max(10.0, min(180.0, self.recovery_head_turn_angle_deg)))
        raw_w = turn_sign * min(abs(self.recovery_head_turn_speed), self.max_vth)
        _, w_cmd = self._apply_min_speed(0.0, raw_w)
        if abs(w_cmd) < 1e-3:
            w_cmd = turn_sign * min(max(0.10, self.min_cmd_angular_speed), self.max_vth)

        accum = 0.0
        pose = self._pose()
        prev_yaw = pose[2] if pose else None
        t0 = time.time()

        while rclpy.ok():
            if goal_handle.is_cancel_requested or self._preempt.is_set():
                self._stop(); return
            if accum >= target:
                break
            if time.time() - t0 > self.recovery_head_turn_timeout:
                self.get_logger().warn(
                    f'Recovery head-turn timeout after {self.recovery_head_turn_timeout:.1f}s '
                    f'(accum={math.degrees(accum):.1f}deg/{self.recovery_head_turn_angle_deg:.1f}deg)')
                break

            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)
            self.cur_v = 0.0
            self.cur_w = w_cmd
            self._publish_nav_status('LARGE_HEADING_RECOVERY', 'recovery_head_turn')
            if self.debug_tracking:
                self.get_logger().info(
                    f'[LARGE_HEADING_RECOVERY] reason=recovery_head_turn '
                    f'angle_error={math.degrees(target - accum):.1f}deg '
                    f'heading_error={math.degrees(target - accum):.1f}deg '
                    f'lateral_error=0.000m target_direction={math.degrees(turn_sign * target):.1f}deg '
                    f'linear_x=0.000 angular_z={w_cmd:.3f} allow_backward=false',
                    throttle_duration_sec=0.20)
            time.sleep(dt)

            p_now = self._pose()
            if p_now and prev_yaw is not None:
                accum += abs(self._angle_diff(prev_yaw, p_now[2]))
                prev_yaw = p_now[2]
            else:
                accum += abs(w_cmd) * dt

        self._stop()

    # ═══════════════════ Recovery ═══════════════════

    def _recovery(self, goal_handle, reason='stuck', lateral_error=0.0):
        self.get_logger().warn(f'[RECOVERY] Entering recovery reason={reason}')
        self._fallback_cycle_counter = 0
        self._fallback_guard_until = 0.0

        # --- Dynamic backup distance logic ---
        now = time.time()
        # If triggered again within 10 seconds, increment the tier
        if now - self._last_recovery_time < 10.0:
            self._recovery_tier += 1
            self.get_logger().info(f'Repeated recovery within 10s! Tier: {self._recovery_tier}, distance += {12*self._recovery_tier} cm')
        else:
            self._recovery_tier = 0
            
        self._last_recovery_time = now

        spd = abs(self.backup_vel)
        if spd < 0.01:
            spd = 0.08 
        extra_distance = 0.12 * self._recovery_tier
        actual_backup_time = self.backup_time + (extra_distance / spd)
        # -----------------------------------

        dt = 1.0 / self.ctrl_freq
        t0 = time.time()
        while time.time() - t0 < actual_backup_time and rclpy.ok():
            if goal_handle.is_cancel_requested or self._preempt.is_set():
                self._stop(); return
            bv, _ = self._apply_min_speed(self.backup_vel, 0.0)
            cmd = Twist()
            cmd.linear.x = bv
            self.cmd_pub.publish(cmd)
            self.cur_v = bv
            self.cur_w = 0.0
            self._publish_nav_status('BACKWARD_RECOVERY', reason)
            if self.debug_tracking:
                self.get_logger().info(
                    f'[BACKWARD_RECOVERY] reason={reason} '
                    f'angle_error=0.0deg heading_error=0.0deg lateral_error=0.000m '
                    f'target_direction=0.0deg linear_x={bv:.3f} angular_z=0.000 '
                    f'allow_backward=true',
                    throttle_duration_sec=0.20)
            time.sleep(dt)
        turn_sign = self._choose_recovery_turn_sign()

        allow_head_turn = (
            self.enable_recovery_head_turn and
            lateral_error >= self.recovery_lateral_error
        )
        if self.enable_recovery_head_turn and not allow_head_turn:
            self.get_logger().info(
                'Skip recovery head-turn: '
                f'lateral_error={lateral_error:.3f}m < '
                f'recovery_lateral_error={self.recovery_lateral_error:.3f}m')

        if allow_head_turn:
            self._perform_recovery_head_turn(goal_handle, turn_sign)
            return

        t0 = time.time()
        while time.time() - t0 < self.recovery_curve_dur and rclpy.ok():
            if goal_handle.is_cancel_requested or self._preempt.is_set():
                self._stop(); return
            cmd = Twist()
            rv = max(self.min_turning_speed, self.recovery_curve_speed)
            rw = turn_sign * min(abs(self.recovery_curve_angular), self.max_vth * 0.6)
            cmd.linear.x, cmd.angular.z = self._apply_min_speed(rv, rw)
            self.cmd_pub.publish(cmd)
            self.cur_v = cmd.linear.x
            self.cur_w = cmd.angular.z
            self._publish_nav_status('RECOVERY', reason)
            if self.debug_tracking:
                self.get_logger().info(
                    f'[RECOVERY] reason={reason}_curve_exit '
                    f'angle_error=0.0deg heading_error=0.0deg lateral_error=0.000m '
                    f'target_direction=0.0deg '
                    f'linear_x={cmd.linear.x:.3f} angular_z={cmd.angular.z:.3f} '
                    f'allow_backward=false',
                    throttle_duration_sec=0.20)
            time.sleep(dt)

        self._stop()

    # ═══════════════════ A* Global Planner ═══════════════════

    def _plan_astar(self, sx, sy, gx, gy):
        """A* with distance-weighted cost — produces centered, wall-distant paths.
        Falls back to reduced inflation if path not found."""
        goal_candidates = [(gx, gy)]
        snapped_goal = self._find_nearby_goal(gx, gy)
        if snapped_goal and math.hypot(snapped_goal[0] - gx, snapped_goal[1] - gy) > 1e-3:
            goal_candidates.append(snapped_goal)
            self.get_logger().info(
                f'A* snapped goal ({gx:.2f},{gy:.2f}) -> '
                f'({snapped_goal[0]:.2f},{snapped_goal[1]:.2f})')

        for tx, ty in goal_candidates:
            for inflate_mult in (self.inflate_mult, 1.2, 1.0):
                path = self._plan_astar_impl(sx, sy, tx, ty, inflate_mult)
                if path:
                    return path
                self.get_logger().info(
                    f'A* retry failed inflate={inflate_mult:.2f} '
                    f'goal=({tx:.2f},{ty:.2f})')
        return []

    def _find_nearby_goal(self, gx, gy):
        """Snap a blocked/unknown goal to the nearest safe explored cell."""
        if not self.map_msg:
            return None
        info = self.map_msg.info
        w, h, res = info.width, info.height, info.resolution
        grid = np.array(self.map_msg.data, dtype=np.int8).reshape(h, w)
        free = (grid >= 0) & (grid <= 50)
        occ = grid > 65

        goal_gx, goal_gy = self._w2g(gx, gy, info)
        if not (0 <= goal_gx < w and 0 <= goal_gy < h):
            return None

        search_cells = max(1, int(math.ceil(self.goal_search_radius / res)))
        step_cells = max(1, int(math.ceil(self.goal_search_step / res)))
        safe_c = max(1, int(math.ceil(self.robot_r / res)))

        best = None
        for r in range(0, search_cells + 1, step_cells):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nx, ny = goal_gx + dx, goal_gy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if not free[ny, nx] or occ[ny, nx]:
                        continue
                    if not self._grid_cell_safe(nx, ny, free, occ, safe_c, w, h):
                        continue
                    cand = self._g2w(nx, ny, info)
                    score = math.hypot(cand[0] - gx, cand[1] - gy)
                    if best is None or score < best[0]:
                        best = (score, cand)
            if best is not None:
                return best[1]
        return None

    @staticmethod
    def _grid_cell_safe(gx, gy, free, occ, safe_c, w, h):
        for dy in range(-safe_c, safe_c + 1):
            for dx in range(-safe_c, safe_c + 1):
                if dx * dx + dy * dy > safe_c * safe_c:
                    continue
                nx, ny = gx + dx, gy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    return False
                if occ[ny, nx] or not free[ny, nx]:
                    return False
        return True

    def _plan_astar_impl(self, sx, sy, gx, gy, inflate_mult):
        """A* implementation with configurable inflation."""
        if not self.map_msg:
            return []

        info = self.map_msg.info
        w, h, res = info.width, info.height, info.resolution
        grid = np.array(self.map_msg.data, dtype=np.int8).reshape(h, w)

        s = self._w2g(sx, sy, info)
        g = self._w2g(gx, gy, info)
        if not (0 <= s[0] < w and 0 <= s[1] < h and
                0 <= g[0] < w and 0 <= g[1] < h):
            self.get_logger().warn('A* fail: out of bounds')
            return []

        occ = grid > 65

        # ── Distance field for this exact grid ──
        if _CV2:
            free = ((1 - occ.astype(np.uint8)) * 255).astype(np.uint8)
            dist_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
            dist_m = dist_px.astype(np.float32) * res
        else:
            dist_m = None

        # ── Hard inflation: inflate_mult × robot_radius ──
        inflate_r = self.robot_r * inflate_mult
        safe_c = max(2, int(math.ceil(inflate_r / res)))
        if _CV2:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * safe_c + 1, 2 * safe_c + 1))
            blocked = cv2.dilate(occ.astype(np.uint8), k).astype(bool)
        else:
            blocked = self._inflate_manual(occ, safe_c, h, w)

        # Do not allow planning through unknown space (avoids shortcuts through unmapped walls)
        blocked[grid < 0] = True

        # Clear start neighborhood (robot might be near a wall or slightly in unknown)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = s[1] + dy, s[0] + dx
                if 0 <= ny < h and 0 <= nx < w:
                    blocked[ny, nx] = False

        n_free = int(np.sum((grid >= 0) & (grid <= 50)))
        self.get_logger().info(
            f'A* ({sx:.1f},{sy:.1f})→({gx:.1f},{gy:.1f}) '
            f'free={n_free} inflate={safe_c}c',
            throttle_duration_sec=1.0)

        # ── Proximity cost zone ──
        prox_limit = self.robot_r * 5  # cells within this get extra cost

        # ── A* search ──
        heap = [(0.0, s[0], s[1])]
        came = {}
        gs = {(s[0], s[1]): 0.0}
        closed = set()
        tol = max(2, int(self.goal_tol / res))

        while heap:
            _, cx, cy = heapq.heappop(heap)
            if (cx, cy) in closed:
                continue
            closed.add((cx, cy))

            if abs(cx - g[0]) <= tol and abs(cy - g[1]) <= tol:
                # Reconstruct path
                path_g = [(cx, cy)]
                while (cx, cy) in came:
                    cx, cy = came[(cx, cy)]
                    path_g.append((cx, cy))
                path_g.reverse()
                return self._grid_to_world(path_g, info, res)

            for ddx, ddy, base_c in (
                (0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1),
                (1, 1, 1.414), (1, -1, 1.414),
                (-1, 1, 1.414), (-1, -1, 1.414),
            ):
                nx, ny = cx + ddx, cy + ddy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in closed or blocked[ny, nx]:
                    continue

                # Distance-weighted cost: penalize cells near obstacles
                cost = base_c
                if dist_m is not None:
                    d = dist_m[ny, nx]
                    if d < prox_limit:
                        # Quadratic falloff: max penalty near walls, zero at prox_limit
                        ratio = (prox_limit - d) / prox_limit
                        cost += base_c * ratio * ratio * self.prox_weight

                ng = gs[(cx, cy)] + cost
                if ng < gs.get((nx, ny), 1e18):
                    came[(nx, ny)] = (cx, cy)
                    gs[(nx, ny)] = ng
                    hv = math.hypot(g[0] - nx, g[1] - ny)
                    heapq.heappush(heap, (ng + hv, nx, ny))

            if len(closed) > self.astar_limit:
                self.get_logger().warn('A* hit limit')
                break

        self.get_logger().warn(f'A* exhausted ({len(closed)} cells)')
        return []

    @staticmethod
    def _w2g(wx, wy, info):
        gx = int((wx - info.origin.position.x) / info.resolution)
        gy = int((wy - info.origin.position.y) / info.resolution)
        return gx, gy

    @staticmethod
    def _g2w(gx, gy, info):
        wx = info.origin.position.x + (gx + 0.5) * info.resolution
        wy = info.origin.position.y + (gy + 0.5) * info.resolution
        return wx, wy

    def _grid_to_world(self, path_g, info, res):
        """Convert grid path to world coords with subsampling."""
        wp = []
        for px, py in path_g:
            wx, wy = self._g2w(px, py, info)
            if not wp or math.hypot(wx - wp[-1][0], wy - wp[-1][1]) >= res * 3:
                wp.append((wx, wy))
        if path_g:
            lx, ly = self._g2w(path_g[-1][0], path_g[-1][1], info)
            if wp and math.hypot(lx - wp[-1][0], ly - wp[-1][1]) > res:
                wp.append((lx, ly))
        return wp

    @staticmethod
    def _inflate_manual(occ, r, h, w):
        """Fallback inflation without cv2."""
        inf = np.zeros((h, w), dtype=bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r:
                    continue
                sr = (max(0, dy), h + min(0, dy))
                sc = (max(0, dx), w + min(0, dx))
                tr = (max(0, -dy), h + min(0, -dy))
                tc = (max(0, -dx), w + min(0, -dx))
                inf[tr[0]:tr[1], tc[0]:tc[1]] |= occ[sr[0]:sr[1], sc[0]:sc[1]]
        return inf

    # ═══════════════════ Path & Viz Publishing ═══════════════════

    def _pub_path(self, path):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for wx, wy in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _pub_local_goal(self, x, y):
        """Publish lookahead target point as a sphere marker."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)
        m.lifetime.sec = 0
        m.lifetime.nanosec = 300_000_000
        self.local_goal_pub.publish(m)

    def _pub_local_traj(self, rx, ry, ryaw, v, w):
        """Publish the simulated trajectory as a line strip marker."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03  # line width
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.5, a=0.9)
        m.lifetime.sec = 0
        m.lifetime.nanosec = 300_000_000

        x, y, yaw = rx, ry, ryaw
        dt = 0.1
        for _ in range(15):  # 1.5 seconds ahead
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.05
            m.points.append(pt)
            if abs(w) > 1e-6:
                x += v / w * (math.sin(yaw + w * dt) - math.sin(yaw))
                y += v / w * (-math.cos(yaw + w * dt) + math.cos(yaw))
                yaw += w * dt
            else:
                x += v * math.cos(yaw) * dt
                y += v * math.sin(yaw) * dt
        self.local_traj_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = CustomNavigator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
