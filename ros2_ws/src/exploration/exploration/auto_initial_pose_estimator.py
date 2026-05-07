#!/usr/bin/env python3

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

try:
    import cv2
except ImportError:
    cv2 = None


class AutoInitialPoseEstimator(Node):
    def __init__(self):
        super().__init__('auto_initial_pose_estimator')
        self.map_data = None
        self.scan_data = None
        self.scan_samples = []
        self.amcl_pose_msg = None
        self.amcl_pose_seq = 0
        self.amcl_pose_wall_time = 0.0
        self.published = False
        self.estimation_in_progress = False
        self.estimation_thread = None

        self.dist_field = None
        self.startup_motion_started = False
        self.startup_motion_deadline = 0.0
        self.startup_motion_origin = None

        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)
        self.declare_parameter('search_x_range', 0.20)
        self.declare_parameter('search_y_range', 0.20)
        self.declare_parameter('search_yaw_range_deg', 30.0)
        self.declare_parameter('search_xy_step', 0.02)
        self.declare_parameter('search_yaw_step_deg', 2.0)
        self.declare_parameter('coarse_xy_step', 0.04)
        self.declare_parameter('coarse_yaw_step_deg', 4.0)
        self.declare_parameter('downsample_beams', 180)
        self.declare_parameter('hit_threshold', 65)
        self.declare_parameter('endpoint_tolerance', 0.12)
        self.declare_parameter('free_space_penalty', 1.5)
        self.declare_parameter('sample_points_per_beam', 3)
        self.declare_parameter('scan_sample_count', 6)
        self.declare_parameter('enable_startup_scan_motion', True)
        self.declare_parameter('startup_scan_duration', 3.5)
        self.declare_parameter('startup_scan_angular_speed', 0.15)
        self.declare_parameter('startup_scan_linear_speed', 0.06)
        self.declare_parameter('startup_min_translation', 0.12)
        self.declare_parameter('startup_min_rotation_deg', 8.0)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_repeats', 3)
        self.declare_parameter('publish_interval', 0.15)
        self.declare_parameter('covariance_xy', 0.05)
        self.declare_parameter('covariance_yaw', 0.10)
        self.declare_parameter('candidate_count', 4)
        self.declare_parameter('candidate_refine_radius', 0.04)
        self.declare_parameter('candidate_refine_yaw_deg', 4.0)
        self.declare_parameter('candidate_amcl_wait', 0.45)
        self.declare_parameter('candidate_nomotion_updates', 2)
        self.declare_parameter('candidate_match_weight', 1.0)
        self.declare_parameter('candidate_cov_weight', 8.0)
        self.declare_parameter('candidate_pose_delta_weight', 6.0)
        self.declare_parameter('candidate_yaw_delta_weight', 2.0)
        self.declare_parameter('candidate_min_amcl_age', 0.15)
        self.declare_parameter('candidate_min_amcl_msgs', 1)
        self.declare_parameter('min_match_ratio', 0.10)
        self.declare_parameter('center_bias_xy_weight', 6.0)
        self.declare_parameter('center_bias_yaw_weight', 1.5)
        self.declare_parameter('unknown_space_penalty', 0.7)

        qos_profile = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 20)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_pub = self.create_publisher(
            Twist, str(self.get_parameter('cmd_vel_topic').value), 10)
        self.nomotion_cli = self.create_client(Empty, '/request_nomotion_update')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self._startup_motion_tick)
        self.create_timer(0.1, self._estimation_tick)

        self.get_logger().info('Auto Initial Pose Estimator started, waiting for map and scan...')

    def map_callback(self, msg):
        if self.map_data is None:
            self.map_data = msg
            self._build_distance_field(msg)
            self.get_logger().info('Map received.')

    def scan_callback(self, msg):
        self.scan_data = msg
        self._record_scan_sample(msg)
        if len(self.scan_samples) == 1:
            self.get_logger().info('First LaserScan received.')

    def amcl_pose_callback(self, msg):
        self.amcl_pose_msg = msg
        self.amcl_pose_seq += 1
        self.amcl_pose_wall_time = time.time()

    def try_estimate_and_publish(self):
        if self.estimation_in_progress:
            return
        if self.map_data is None or self.scan_data is None or self.published:
            return
        if self._startup_motion_active():
            return
        if len(self.scan_samples) < max(1, int(self.get_parameter('scan_sample_count').value)):
            return
        if not self._startup_motion_sufficient():
            return

        self.estimation_in_progress = True
        self.estimation_thread = threading.Thread(
            target=self._run_estimation,
            name='auto_initial_pose_estimation',
            daemon=True,
        )
        self.estimation_thread.start()

    def _estimation_tick(self):
        self.try_estimate_and_publish()

    def _run_estimation(self):
        try:
            self.get_logger().info('Starting AMCL-assisted initial pose evaluation...')
            best_pose, best_score, base_count = self.find_best_pose()
            match_ratio = 0.0 if base_count == 0 else max(0.0, best_score) / float(base_count)
            if match_ratio < float(self.get_parameter('min_match_ratio').value):
                self.get_logger().warn(
                    f'Best candidate match ratio too low ({match_ratio:.3f}), '
                    'falling back to configured initial pose.')
                best_pose = self._configured_pose()

            self.log_pose_delta(best_pose)
            self.publish_initial_pose(best_pose)
            self.published = True
            self._stop_robot()
            self.get_logger().info('Initial pose published successfully! Exiting logic loop.')

            self.destroy_subscription(self.map_sub)
            self.destroy_subscription(self.scan_sub)
        finally:
            self.estimation_in_progress = False
            self.estimation_thread = None

    def find_best_pose(self):
        init_pose = self._initial_search_pose()
        base_pts = self._build_aggregated_points()
        if not base_pts:
            self.get_logger().warn('No valid scan points. Falling back to default start pose.')
            return self._configured_pose(), 0.0, 0

        target_beams = max(30, int(self.get_parameter('downsample_beams').value))
        if len(base_pts) > target_beams:
            step = max(1, len(base_pts) // target_beams)
            base_pts = base_pts[::step]

        map_info = self._map_info()

        coarse_candidates = self._search_candidates(
            center_pose=init_pose,
            range_pose=(
                float(self.get_parameter('search_x_range').value),
                float(self.get_parameter('search_y_range').value),
                math.radians(float(self.get_parameter('search_yaw_range_deg').value)),
            ),
            step_pose=(
                max(float(self.get_parameter('search_xy_step').value),
                    float(self.get_parameter('coarse_xy_step').value)),
                max(float(self.get_parameter('search_xy_step').value),
                    float(self.get_parameter('coarse_xy_step').value)),
                math.radians(max(float(self.get_parameter('search_yaw_step_deg').value),
                                 float(self.get_parameter('coarse_yaw_step_deg').value))),
            ),
            base_pts=base_pts,
            map_info=map_info,
            limit=max(1, int(self.get_parameter('candidate_count').value)),
        )

        refine_radius = float(self.get_parameter('candidate_refine_radius').value)
        refine_yaw = math.radians(float(self.get_parameter('candidate_refine_yaw_deg').value))
        final_candidates = []
        for coarse in coarse_candidates:
            refined = self._search_candidates(
                center_pose=coarse['pose'],
                range_pose=(refine_radius, refine_radius, refine_yaw),
                step_pose=(
                    float(self.get_parameter('search_xy_step').value),
                    float(self.get_parameter('search_xy_step').value),
                    math.radians(float(self.get_parameter('search_yaw_step_deg').value)),
                ),
                base_pts=base_pts,
                map_info=map_info,
                limit=2,
            )
            final_candidates.extend(refined)

        deduped = self._dedupe_candidates(final_candidates)
        if not deduped:
            return self._configured_pose(), 0.0, len(base_pts)

        best_entry = None
        for idx, candidate in enumerate(deduped, start=1):
            amcl_xy_std, amcl_yaw_std = self._evaluate_candidate(candidate['pose'])
            candidate['amcl_xy_std'] = amcl_xy_std
            candidate['amcl_yaw_std'] = amcl_yaw_std
            pose_delta = self._candidate_pose_delta(candidate['pose'])
            yaw_delta = self._candidate_yaw_delta(candidate['pose'])
            candidate['pose_delta'] = pose_delta
            candidate['yaw_delta'] = yaw_delta
            candidate['combined_score'] = (
                float(self.get_parameter('candidate_match_weight').value) * candidate['score']
                - float(self.get_parameter('candidate_cov_weight').value)
                * (amcl_xy_std + 0.5 * amcl_yaw_std)
                - float(self.get_parameter('candidate_pose_delta_weight').value) * pose_delta
                - float(self.get_parameter('candidate_yaw_delta_weight').value) * yaw_delta
            )
            self.get_logger().info(
                f'Candidate {idx}: pose=({candidate["pose"][0]:.3f}, {candidate["pose"][1]:.3f}, '
                f'{math.degrees(candidate["pose"][2]):.1f} deg) '
                f'match={candidate["score"]:.2f} '
                f'amcl_xy_std={amcl_xy_std:.3f} amcl_yaw_std={amcl_yaw_std:.3f} '
                f'pose_delta={pose_delta:.3f} yaw_delta={math.degrees(yaw_delta):.1f} deg '
                f'combined={candidate["combined_score"]:.2f}')
            if best_entry is None or candidate['combined_score'] > best_entry['combined_score']:
                best_entry = candidate

        best_pose = best_entry['pose']
        match_ratio = max(0.0, best_entry['score']) / len(base_pts) * 100.0
        self.get_logger().info(
            f'Local Match Result: X={best_pose[0]:.3f}, Y={best_pose[1]:.3f}, '
            f'Yaw={math.degrees(best_pose[2]):.1f} deg | Score: {best_entry["score"]:.2f}/{len(base_pts)} '
            f'({match_ratio:.1f}%)')
        return best_pose, best_entry['score'], len(base_pts)

    def _search_candidates(self, center_pose, range_pose, step_pose, base_pts, map_info, limit):
        cx, cy, cyaw = center_pose
        rx, ry, ryaw = range_pose
        sx, sy, syaw = step_pose
        x_steps = self._build_axis_steps(cx, rx, sx)
        y_steps = self._build_axis_steps(cy, ry, sy)
        yaw_steps = self._build_axis_steps(cyaw, ryaw, syaw)

        scored = []
        for yaw in yaw_steps:
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            rot_pts = [(bx * cos_yaw - by * sin_yaw, bx * sin_yaw + by * cos_yaw) for bx, by in base_pts]
            for tx in x_steps:
                for ty in y_steps:
                    score = self._score_pose(tx, ty, rot_pts, map_info)
                    score -= self._center_bias_penalty(center_pose, (tx, ty, yaw))
                    scored.append({'pose': (tx, ty, yaw), 'score': score})
        scored.sort(key=lambda item: item['score'], reverse=True)
        return scored[:limit]

    def _dedupe_candidates(self, candidates):
        deduped = []
        for item in sorted(candidates, key=lambda item: item['score'], reverse=True):
            pose = item['pose']
            duplicate = False
            for kept in deduped:
                if (
                    abs(pose[0] - kept['pose'][0]) < 0.015
                    and abs(pose[1] - kept['pose'][1]) < 0.015
                    and abs(self._wrap_angle(pose[2] - kept['pose'][2])) < math.radians(2.0)
                ):
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(item)
        return deduped[:max(1, int(self.get_parameter('candidate_count').value))]

    def _score_pose(self, tx, ty, rot_pts, map_info):
        origin_x, origin_y, resolution, width, height, data = map_info
        endpoint_tolerance = max(0.02, float(self.get_parameter('endpoint_tolerance').value))
        free_space_penalty = max(0.1, float(self.get_parameter('free_space_penalty').value))
        sample_points = max(1, int(self.get_parameter('sample_points_per_beam').value))
        hit_threshold = int(self.get_parameter('hit_threshold').value)
        unknown_penalty = max(0.0, float(self.get_parameter('unknown_space_penalty').value))

        score = 0.0
        for rx, ry in rot_pts:
            mx = tx + rx
            my = ty + ry
            endpoint_cell = self._map_cell(mx, my, map_info)
            if endpoint_cell < 0:
                score -= unknown_penalty
                continue
            endpoint_dist = self._distance_to_obstacle(mx, my, map_info)
            score += max(0.0, 1.0 - endpoint_dist / endpoint_tolerance)

            for sample_idx in range(1, sample_points):
                ratio = sample_idx / float(sample_points)
                sx = tx + rx * ratio
                sy = ty + ry * ratio
                px = int((sx - origin_x) / resolution)
                py = int((sy - origin_y) / resolution)
                if not (0 <= px < width and 0 <= py < height):
                    score -= free_space_penalty
                    break
                cell = data[py * width + px]
                if cell < 0:
                    score -= unknown_penalty
                    break
                if cell >= hit_threshold:
                    score -= free_space_penalty
                    break

        return score

    def _distance_to_obstacle(self, wx, wy, map_info):
        origin_x, origin_y, resolution, width, height, data = map_info
        px = int((wx - origin_x) / resolution)
        py = int((wy - origin_y) / resolution)
        if not (0 <= px < width and 0 <= py < height):
            return 999.0
        if self.dist_field is not None:
            return float(self.dist_field[py, px])
        cell = data[py * width + px]
        if cell < 0:
            return 999.0
        return 0.0 if cell >= int(self.get_parameter('hit_threshold').value) else resolution * 2.0

    def _center_bias_penalty(self, center_pose, pose):
        dx = pose[0] - center_pose[0]
        dy = pose[1] - center_pose[1]
        dyaw = self._wrap_angle(pose[2] - center_pose[2])
        xy_penalty = math.hypot(dx, dy) * float(self.get_parameter('center_bias_xy_weight').value)
        yaw_penalty = abs(math.degrees(dyaw)) * float(self.get_parameter('center_bias_yaw_weight').value) / 10.0
        return xy_penalty + yaw_penalty

    def _evaluate_candidate(self, pose):
        prev_seq = self.amcl_pose_seq
        self.amcl_pose_msg = None
        self.amcl_pose_wall_time = 0.0
        self.publish_initial_pose(pose)
        wait_time = max(0.1, float(self.get_parameter('candidate_amcl_wait').value))
        updates = max(0, int(self.get_parameter('candidate_nomotion_updates').value))
        required_msgs = max(1, int(self.get_parameter('candidate_min_amcl_msgs').value))
        min_age = max(0.0, float(self.get_parameter('candidate_min_amcl_age').value))

        for _ in range(updates):
            future = self._request_nomotion_update()
            self._wait_future(future, wait_time / max(1, updates + 1))
            time.sleep(wait_time / max(1, updates + 1))

        deadline = time.time() + wait_time
        while time.time() < deadline:
            if (
                self.amcl_pose_msg is not None
                and self.amcl_pose_seq - prev_seq >= required_msgs
                and time.time() - self.amcl_pose_wall_time >= min_age
            ):
                break
            time.sleep(0.02)

        if self.amcl_pose_msg is None or self.amcl_pose_seq <= prev_seq:
            return 999.0, 999.0

        cov = self.amcl_pose_msg.pose.covariance
        xy_std = math.sqrt(max(cov[0], cov[7], 0.0))
        yaw_std = math.sqrt(max(cov[35], 0.0))
        return xy_std, yaw_std

    def _candidate_pose_delta(self, pose):
        if self.amcl_pose_msg is None:
            return 999.0
        amcl_pose = self.amcl_pose_msg.pose.pose
        dx = float(amcl_pose.position.x) - pose[0]
        dy = float(amcl_pose.position.y) - pose[1]
        return math.hypot(dx, dy)

    def _candidate_yaw_delta(self, pose):
        if self.amcl_pose_msg is None:
            return math.pi
        q = self.amcl_pose_msg.pose.pose.orientation
        amcl_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return abs(self._wrap_angle(amcl_yaw - pose[2]))

    def _request_nomotion_update(self):
        if not self.nomotion_cli.service_is_ready():
            self.nomotion_cli.wait_for_service(timeout_sec=0.1)
        if not self.nomotion_cli.service_is_ready():
            return None
        future = self.nomotion_cli.call_async(Empty.Request())
        return future

    def _wait_future(self, future: Future, timeout_sec: float):
        if future is None:
            return False
        deadline = time.time() + timeout_sec
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        return future.done()

    def _configured_pose(self):
        return (
            float(self.get_parameter('initial_pose_x').value),
            float(self.get_parameter('initial_pose_y').value),
            float(self.get_parameter('initial_pose_yaw').value),
        )

    def _initial_search_pose(self):
        amcl_pose = self._latest_amcl_pose()
        if amcl_pose is not None:
            self.get_logger().info(
                'Using current AMCL pose as search center: '
                f'x={amcl_pose[0]:.3f} m, y={amcl_pose[1]:.3f} m, '
                f'yaw={math.degrees(amcl_pose[2]):.1f} deg')
            return amcl_pose

        pose = self._configured_pose()
        if self.startup_motion_origin is None:
            return pose
        latest_pose = self._latest_odom_pose()
        if latest_pose is None:
            return pose

        init_x, init_y, init_yaw = pose
        odom_dx = latest_pose[0] - self.startup_motion_origin[0]
        odom_dy = latest_pose[1] - self.startup_motion_origin[1]
        odom_dyaw = self._wrap_angle(latest_pose[2] - self.startup_motion_origin[2])

        map_dx = math.cos(init_yaw) * odom_dx - math.sin(init_yaw) * odom_dy
        map_dy = math.sin(init_yaw) * odom_dx + math.cos(init_yaw) * odom_dy
        shifted_pose = (
            init_x + map_dx,
            init_y + map_dy,
            self._wrap_angle(init_yaw + odom_dyaw),
        )
        self.get_logger().info(
            'Startup odom delta applied to search center: '
            f'dx={map_dx:.3f} m, dy={map_dy:.3f} m, '
            f'dyaw={math.degrees(odom_dyaw):.1f} deg')
        return shifted_pose

    def _latest_amcl_pose(self):
        if self.amcl_pose_msg is None:
            return None
        pose = self.amcl_pose_msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (float(pose.position.x), float(pose.position.y), yaw)

    def _map_info(self):
        return (
            self.map_data.info.origin.position.x,
            self.map_data.info.origin.position.y,
            self.map_data.info.resolution,
            self.map_data.info.width,
            self.map_data.info.height,
            self.map_data.data,
        )

    def _build_distance_field(self, msg):
        if cv2 is None:
            self.get_logger().warn('cv2 not available, falling back to coarse obstacle endpoint scoring.')
            return
        width = msg.info.width
        height = msg.info.height
        hit_threshold = int(self.get_parameter('hit_threshold').value)
        free_mask = np.array(
            [1 if 0 <= val < hit_threshold else 0 for val in msg.data],
            dtype=np.uint8,
        )
        free_mask = free_mask.reshape((height, width))
        self.dist_field = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5) * float(msg.info.resolution)

    def log_pose_delta(self, pose):
        init_x, init_y, init_yaw = self._configured_pose()
        dx = pose[0] - init_x
        dy = pose[1] - init_y
        dyaw = self._wrap_angle(pose[2] - init_yaw)
        self.get_logger().info(
            f'Estimated start offset from configured pose: '
            f'dx={dx:.3f} m, dy={dy:.3f} m, dyaw={math.degrees(dyaw):.1f} deg')

    def publish_initial_pose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        yaw = pose[2]
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0] = float(self.get_parameter('covariance_xy').value)
        cov[7] = float(self.get_parameter('covariance_xy').value)
        cov[35] = float(self.get_parameter('covariance_yaw').value)
        msg.pose.covariance = cov

        repeats = max(1, int(self.get_parameter('publish_repeats').value))
        interval = max(0.05, float(self.get_parameter('publish_interval').value))
        for idx in range(repeats):
            if self.scan_data is not None:
                msg.header.stamp = self.scan_data.header.stamp
            else:
                msg.header.stamp = Time(seconds=0).to_msg()
            self.pose_pub.publish(msg)
            self.get_logger().info(f'Published initial pose to AMCL ({idx + 1}/{repeats}).')
            if idx + 1 < repeats:
                time.sleep(interval)

    def _record_scan_sample(self, msg):
        odom_pose = self._lookup_odom_pose(msg.header.stamp)
        if odom_pose is None:
            return
        self.scan_samples.append((msg, odom_pose))
        max_samples = max(1, int(self.get_parameter('scan_sample_count').value))
        if len(self.scan_samples) > max_samples:
            self.scan_samples = self.scan_samples[-max_samples:]

    def _build_aggregated_points(self):
        if not self.scan_samples:
            return []
        ref_pose = self.scan_samples[-1][1]
        points = []
        for scan_msg, odom_pose in self.scan_samples:
            points.extend(self._scan_to_reference_points(scan_msg, odom_pose, ref_pose))
        return points

    def _scan_to_reference_points(self, scan_msg, odom_pose, ref_pose):
        ox, oy, oyaw = odom_pose
        rx, ry, ryaw = ref_pose
        dyaw = self._wrap_angle(oyaw - ryaw)
        dx = ox - rx
        dy = oy - ry
        ref_dx = math.cos(-ryaw) * dx - math.sin(-ryaw) * dy
        ref_dy = math.sin(-ryaw) * dx + math.cos(-ryaw) * dy

        points = []
        angle = scan_msg.angle_min
        for rng in scan_msg.ranges:
            if scan_msg.range_min <= rng <= scan_msg.range_max and math.isfinite(rng):
                bx = rng * math.cos(angle)
                by = rng * math.sin(angle)
                px = math.cos(dyaw) * bx - math.sin(dyaw) * by + ref_dx
                py = math.sin(dyaw) * bx + math.cos(dyaw) * by + ref_dy
                points.append((px, py))
            angle += scan_msg.angle_increment
        return points

    def _lookup_odom_pose(self, stamp_msg):
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'base_footprint', Time.from_msg(stamp_msg))
        except TransformException:
            try:
                # Startup TF and scan timestamps can be slightly misaligned.
                # Fall back to the latest available odom pose so startup samples
                # still accumulate instead of stalling localization entirely.
                tf = self.tf_buffer.lookup_transform('odom', 'base_footprint', Time())
            except TransformException:
                return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.x, t.y, yaw)

    def _startup_motion_tick(self):
        if self.published or self.map_data is None or self.scan_data is None:
            return
        if not bool(self.get_parameter('enable_startup_scan_motion').value):
            return
        if self.startup_motion_started and self.startup_motion_origin is None:
            self.startup_motion_origin = self._latest_odom_pose()
            if self.startup_motion_origin is not None:
                self.get_logger().info('Captured startup motion origin from first valid odom sample.')

        enough_samples = len(self.scan_samples) >= max(
            1, int(self.get_parameter('scan_sample_count').value))
        enough_motion = self._startup_motion_sufficient()

        if enough_samples and enough_motion:
            if self._startup_motion_active():
                self._stop_robot()
                self.startup_motion_deadline = 0.0
            return
        if not self.startup_motion_started:
            origin = self._latest_odom_pose()
            if origin is None:
                return
            self.startup_motion_started = True
            self.startup_motion_deadline = time.time() + float(
                self.get_parameter('startup_scan_duration').value)
            self.startup_motion_origin = origin
            self.get_logger().info('Starting startup scan motion for multi-frame localization.')
        if self._startup_motion_active() and not enough_motion:
            msg = Twist()
            msg.linear.x = float(self.get_parameter('startup_scan_linear_speed').value)
            msg.angular.z = float(self.get_parameter('startup_scan_angular_speed').value)
            self.cmd_pub.publish(msg)
        else:
            self._stop_robot()
            self.startup_motion_deadline = 0.0

    def _startup_motion_active(self):
        return self.startup_motion_started and time.time() < self.startup_motion_deadline

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _startup_motion_sufficient(self):
        if not bool(self.get_parameter('enable_startup_scan_motion').value):
            return True
        if self.startup_motion_origin is None:
            return False
        latest_pose = self._latest_odom_pose()
        if latest_pose is None:
            return False
        dx = latest_pose[0] - self.startup_motion_origin[0]
        dy = latest_pose[1] - self.startup_motion_origin[1]
        travel = math.hypot(dx, dy)
        turn = abs(self._wrap_angle(latest_pose[2] - self.startup_motion_origin[2]))
        return (
            travel >= float(self.get_parameter('startup_min_translation').value)
            or turn >= math.radians(float(self.get_parameter('startup_min_rotation_deg').value))
        )

    def _latest_odom_pose(self):
        if not self.scan_samples:
            return None
        return self.scan_samples[-1][1]

    def _map_cell(self, wx, wy, map_info):
        origin_x, origin_y, resolution, width, height, data = map_info
        px = int((wx - origin_x) / resolution)
        py = int((wy - origin_y) / resolution)
        if not (0 <= px < width and 0 <= py < height):
            return 100
        return int(data[py * width + px])

    @staticmethod
    def _build_axis_steps(center, radius, step):
        if radius <= 0.0:
            return [center]
        count = int(math.floor(radius / max(step, 1e-6)))
        values = [center + offset * step for offset in range(-count, count + 1)]
        if all(abs(value - center) > 1e-9 for value in values):
            values.append(center)
        values.sort()
        return values

    @staticmethod
    def _wrap_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = AutoInitialPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
