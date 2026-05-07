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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String, ColorRGBA
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

        # ── Robot ──
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('max_vel_x', 0.22)
        self.declare_parameter('min_vel_x', -0.10)
        self.declare_parameter('max_vel_theta', 0.7)
        self.declare_parameter('max_accel_x', 0.18)
        self.declare_parameter('max_decel_x', 0.28)
        self.declare_parameter('max_accel_theta', 1.0)
        self.declare_parameter('max_lateral_accel', 0.10)

        # ── Local planner ──
        self.declare_parameter('control_freq', 15.0)
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('lookahead_dist', 0.45)
        self.declare_parameter('safety_horizon', 0.5)
        self.declare_parameter('allow_in_place_rotation', False)
        self.declare_parameter('min_turning_speed', 0.05)
        self.declare_parameter('sharp_turn_angle', 1.05)
        self.declare_parameter('very_sharp_turn_angle', 1.57)
        self.declare_parameter('sharp_turn_speed', 0.06)
        self.declare_parameter('very_sharp_turn_speed', 0.03)

        # ── A* ──
        self.declare_parameter('astar_limit', 100000)
        self.declare_parameter('inflation_mult', 1.5)    # hard inflation = mult * robot_radius
        self.declare_parameter('proximity_weight', 3.0)   # cost weight for being near walls
        self.declare_parameter('goal_search_radius', 0.8)
        self.declare_parameter('goal_search_step', 0.1)

        # ── Recovery ──
        self.declare_parameter('stuck_timeout', 3.5)
        self.declare_parameter('stuck_radius', 0.12)
        self.declare_parameter('max_recoveries', 5)
        self.declare_parameter('backup_vel', -0.10)
        self.declare_parameter('backup_time', 0.8)
        self.declare_parameter('spin_vel', 0.6)
        self.declare_parameter('spin_duration', 1.5)

        # ── Goal ──
        self.declare_parameter('goal_timeout', 90.0)
        self.declare_parameter('replan_interval', 5.0)

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

        self.ctrl_freq = float(p('control_freq').value)
        self.goal_tol = float(p('goal_tolerance').value)
        self.lookahead = float(p('lookahead_dist').value)
        self.safety_hz = float(p('safety_horizon').value)
        self.allow_in_place_rotation = bool(p('allow_in_place_rotation').value)
        self.min_turning_speed = float(p('min_turning_speed').value)
        self.sharp_turn_angle = float(p('sharp_turn_angle').value)
        self.very_sharp_turn_angle = float(p('very_sharp_turn_angle').value)
        self.sharp_turn_speed = float(p('sharp_turn_speed').value)
        self.very_sharp_turn_speed = float(p('very_sharp_turn_speed').value)

        self.astar_limit = int(p('astar_limit').value)
        self.inflate_mult = float(p('inflation_mult').value)
        self.prox_weight = float(p('proximity_weight').value)
        self.goal_search_radius = float(p('goal_search_radius').value)
        self.goal_search_step = float(p('goal_search_step').value)

        self.stuck_timeout = float(p('stuck_timeout').value)
        self.stuck_radius = float(p('stuck_radius').value)
        self.max_recover = int(p('max_recoveries').value)
        self.backup_vel = float(p('backup_vel').value)
        self.backup_time = float(p('backup_time').value)
        self.spin_vel = float(p('spin_vel').value)
        self.spin_dur = float(p('spin_duration').value)

        self.goal_timeout = float(p('goal_timeout').value)
        self.replan_sec = float(p('replan_interval').value)

        cmd_topic = str(p('cmd_vel_topic').value)

        # ── State ──
        self.map_msg: OccupancyGrid | None = None
        self.scan_msg: LaserScan | None = None
        self.cur_v = 0.0
        self.cur_w = 0.0
        self._nav_lock = threading.Lock()
        self._preempt = threading.Event()
        self._last_cmd_time = time.time()

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

        # ── Subscriptions ──
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._on_map, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._on_scan, 10)

        # ── Publishers ──
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, '/navigation/path', 10)
        self.local_traj_pub = self.create_publisher(
            Marker, '/navigation/local_traj', 10)
        self.local_goal_pub = self.create_publisher(
            Marker, '/navigation/local_goal', 10)
        self.status_pub = self.create_publisher(
            String, '/navigation/status', 10)

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
            f'cv2={_CV2}')

    # ═══════════════════ Callbacks ═══════════════════

    def _on_map(self, msg):
        self.map_msg = msg
        now = time.time()
        if _CV2 and now - self._last_dist_update > 0.5:
            self._update_dist_field(msg)
            self._last_dist_update = now

    def _on_scan(self, msg):
        self.scan_msg = msg

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
        self.get_logger().info(f'Goal: ({gx:.2f}, {gy:.2f})')

        self.cur_v = self.cur_w = 0.0
        pose = self._pose()
        if not pose or not self.map_msg:
            goal_handle.abort(); self._stop()
            return NavigateToPose.Result()

        path = self._plan_astar(pose[0], pose[1], gx, gy)
        if not path:
            self.get_logger().warn('A* failed')
            goal_handle.abort(); self._stop()
            return NavigateToPose.Result()
        self._pub_path(path)

        dt = 1.0 / self.ctrl_freq
        t0 = time.time()
        last_replan = t0
        n_rec = 0
        prog_pos = (pose[0], pose[1])
        prog_yaw = pose[2]
        prog_t = t0

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

            pose = self._pose()
            if not pose:
                time.sleep(dt); continue
            rx, ry, ryaw = pose

            d2g = math.hypot(gx - rx, gy - ry)
            if d2g < self.goal_tol:
                self.get_logger().info('Goal reached')
                self._stop(); goal_handle.succeed()
                return NavigateToPose.Result()

            # Stuck: translation OR rotation counts as progress
            dp = math.hypot(rx - prog_pos[0], ry - prog_pos[1])
            da = abs(self._angle_diff(prog_yaw, ryaw))
            if dp > self.stuck_radius or da > 0.3:
                prog_pos, prog_yaw, prog_t = (rx, ry), ryaw, time.time()
            elif time.time() - prog_t > self.stuck_timeout:
                n_rec += 1
                if n_rec > self.max_recover:
                    self.get_logger().warn('Max recoveries')
                    goal_handle.abort(); self._stop()
                    return NavigateToPose.Result()
                self.get_logger().info(f'Stuck → recovery {n_rec}/{self.max_recover}')
                self._recovery(goal_handle)
                p2 = self._pose()
                if p2:
                    prog_pos, prog_yaw = (p2[0], p2[1]), p2[2]
                prog_t = time.time()
                new_path = self._plan_astar(
                    p2[0] if p2 else rx, p2[1] if p2 else ry, gx, gy)
                if new_path:
                    path = new_path; self._pub_path(path)
                last_replan = time.time()
                continue

            # Periodic replan
            now = time.time()
            if now - last_replan > self.replan_sec:
                new_path = self._plan_astar(rx, ry, gx, gy)
                if new_path:
                    path = new_path; self._pub_path(path)
                last_replan = now

            v, w = self._local_plan(rx, ry, ryaw, path, d2g)

            self.get_logger().info(
                f'v={v:.3f} w={w:.3f} d={d2g:.2f}',
                throttle_duration_sec=2.0)

            v, w = self._apply_dynamics_limits(v, w)
            cmd = Twist()
            cmd.linear.x = v; cmd.angular.z = w
            self.cmd_pub.publish(cmd)
            self.cur_v, self.cur_w = v, w

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

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.cur_v = self.cur_w = 0.0
        self._last_cmd_time = time.time()

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

    def _find_local_goal(self, rx, ry, path, lookahead=None):
        if lookahead is None:
            lookahead = self.lookahead
        if not path:
            return None
        min_d, closest = float('inf'), 0
        for i, (px, py) in enumerate(path):
            d = math.hypot(px - rx, py - ry)
            if d < min_d:
                min_d = d
                closest = i
        accum = 0.0
        for i in range(closest, len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            accum += math.hypot(dx, dy)
            if accum >= lookahead:
                return path[i + 1]
        return path[-1]

    def _pursuit_vw(self, ryaw, lg, ry, rx, d2g):
        """Compute (v, w) toward local goal point."""
        a2g = math.atan2(lg[1] - ry, lg[0] - rx)
        err = self._angle_diff(ryaw, a2g)
        w = max(-self.max_vth, min(self.max_vth, 2.0 * err))
        align = max(0.0, math.cos(err))
        v = self.max_vx * max(0.04, align)
        if abs(err) > self.sharp_turn_angle:
            v = min(v, self.sharp_turn_speed)
        if abs(err) > self.very_sharp_turn_angle:
            v = min(v, self.very_sharp_turn_speed)
        if d2g < 0.4:
            v = min(v, self.max_vx * max(0.04, d2g / 0.4))
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

        return v, w

    def _local_plan(self, rx, ry, ryaw, path, d2g):
        """Pure pursuit with costmap safety. Always returns nonzero command."""
        if not path:
            return 0.0, 0.0

        # Collision threshold: generous because A* paths are already far from walls
        col_r = self.robot_r * 0.6

        # Close to goal: aim directly at goal, skip path following
        if d2g < self.goal_tol * 1.8 and path:
            gx, gy = path[-1]
            self._pub_local_goal(gx, gy)
            a2g = math.atan2(gy - ry, gx - rx)
            err = self._angle_diff(ryaw, a2g)
            w = max(-self.max_vth, min(self.max_vth, 2.0 * err))
            v = min(0.10, self.max_vx * max(0.04, d2g / 0.4))
            if abs(err) > self.sharp_turn_angle:
                v = min(v, self.sharp_turn_speed)
            if abs(err) > self.very_sharp_turn_angle:
                v = min(v, self.very_sharp_turn_speed)
            return v, w

        # No costmap → direct pursuit
        if self._dist_field is None:
            lg = self._find_local_goal(rx, ry, path)
            if lg:
                self._pub_local_goal(lg[0], lg[1])
                return self._pursuit_vw(ryaw, lg, ry, rx, d2g)
            return 0.0, 0.0

        # Try different lookahead distances (long → short)
        for la in [self.lookahead, 0.20, 0.08]:
            lg = self._find_local_goal(rx, ry, path, la)
            if not lg:
                continue
            self._pub_local_goal(lg[0], lg[1])
            v, w = self._pursuit_vw(ryaw, lg, ry, rx, d2g)
            # Speed cascade: full → 60% → crawl → ackermann-safe turning crawl
            for v_try in [v, v * 0.6, self.very_sharp_turn_speed]:
                if self._check_cmd(rx, ry, ryaw, v_try, w) >= col_r:
                    return v_try, w
            if abs(w) > 0.05:
                turn_v = 0.0 if self.allow_in_place_rotation else self.min_turning_speed
                if self._check_cmd(rx, ry, ryaw, turn_v, w) >= col_r:
                    return turn_v, w

        # All lookaheads blocked → angular search
        for w_try in [0.5, -0.5, 0.7, -0.7, 0.3, -0.3]:
            turn_v = max(0.04, self.min_turning_speed)
            if self._check_cmd(rx, ry, ryaw, turn_v, w_try) >= col_r:
                return turn_v, w_try
            if self.allow_in_place_rotation and self._check_cmd(rx, ry, ryaw, 0.0, w_try) >= col_r:
                return 0.0, w_try

        # Last resort: stop and let stuck recovery handle it.
        return 0.0, 0.0

    @staticmethod
    def _angle_diff(a, b):
        d = b - a
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    # ═══════════════════ Recovery ═══════════════════

    def _recovery(self, goal_handle):
        dt = 1.0 / self.ctrl_freq
        t0 = time.time()
        while time.time() - t0 < self.backup_time and rclpy.ok():
            if goal_handle.is_cancel_requested or self._preempt.is_set():
                self._stop(); return
            cmd = Twist(); cmd.linear.x = self.backup_vel
            self.cmd_pub.publish(cmd); time.sleep(dt)
        t0 = time.time()
        while time.time() - t0 < self.spin_dur and rclpy.ok():
            if goal_handle.is_cancel_requested or self._preempt.is_set():
                self._stop(); return
            cmd = Twist()
            cmd.angular.z = self.spin_vel
            if not self.allow_in_place_rotation:
                cmd.linear.x = max(self.very_sharp_turn_speed, self.min_turning_speed)
            self.cmd_pub.publish(cmd); time.sleep(dt)
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

        # Clear start neighborhood (robot might be near a wall)
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
