#!/usr/bin/env python3
"""
Minimal 2D robot simulator for testing frontier exploration.

Supports two modes:
  - random indoor map generation for exploration testing
  - fixed map loading for known-map navigation testing

Simulates:
  - Robot pose integration from /cmd_vel
  - TF broadcast
  - LaserScan via raycasting on ground-truth map
  - Optional SLAM-like incremental map reveal

No Gazebo, no GPU required — runs on any machine.
"""

import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster
import yaml

# Nav2 static_layer expects TRANSIENT_LOCAL durability on /map
_MAP_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)


class SimEnvironment(Node):

    def __init__(self):
        super().__init__('sim_environment')

        # ── parameters ──
        self.declare_parameter('map_width', 10.0)     # meters
        self.declare_parameter('map_height', 10.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('num_rooms', 6)
        self.declare_parameter('num_obstacles', 8)
        self.declare_parameter('lidar_range', 8.0)
        self.declare_parameter('lidar_rays', 360)
        self.declare_parameter('lidar_hz', 10.0)
        self.declare_parameter('seed', -1)             # -1 = random
        self.declare_parameter('start_x', 0.0)         # robot start in world (meters)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('fixed_map_yaml', '')
        self.declare_parameter('publish_map_topic', True)
        self.declare_parameter('publish_map_odom_tf', True)

        self.map_w_m = float(self.get_parameter('map_width').value)
        self.map_h_m = float(self.get_parameter('map_height').value)
        self.res = float(self.get_parameter('resolution').value)
        self.num_rooms = int(self.get_parameter('num_rooms').value)
        self.num_obs = int(self.get_parameter('num_obstacles').value)
        self.lidar_range = float(self.get_parameter('lidar_range').value)
        self.lidar_rays = int(self.get_parameter('lidar_rays').value)
        lidar_hz = float(self.get_parameter('lidar_hz').value)
        seed = int(self.get_parameter('seed').value)
        self.start_x = float(self.get_parameter('start_x').value)
        self.start_y = float(self.get_parameter('start_y').value)
        self.start_yaw = float(self.get_parameter('start_yaw').value)
        self.fixed_map_yaml = str(self.get_parameter('fixed_map_yaml').value)
        self.publish_map_topic = bool(self.get_parameter('publish_map_topic').value)
        self.publish_map_odom_tf = bool(self.get_parameter('publish_map_odom_tf').value)
        self.robot_x = self.start_x
        self.robot_y = self.start_y
        self._map_count = 0

        if self.fixed_map_yaml:
            self.gt_map = self._load_map_from_yaml(self.fixed_map_yaml)
        else:
            self.gw = int(self.map_w_m / self.res)   # grid width in cells
            self.gh = int(self.map_h_m / self.res)
            self.ox = -self.map_w_m / 2.0            # map origin
            self.oy = -self.map_h_m / 2.0
            rng = np.random.default_rng(seed if seed >= 0 else None)
            self.gt_map = self._gen_map(rng, self.num_rooms, self.num_obs)
        self.revealed = np.zeros((self.gh, self.gw), dtype=bool)

        sx, sy = self._w2g(self.robot_x, self.robot_y)
        if sx is not None and not self.fixed_map_yaml:
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    nx, ny = sx+dx, sy+dy
                    if 0 <= nx < self.gw and 0 <= ny < self.gh:
                        self.gt_map[ny, nx] = 0
        elif sx is not None and self.gt_map[sy, sx] >= 50:
            self.get_logger().warn(
                f'Start pose ({self.robot_x:.2f}, {self.robot_y:.2f}) is in an occupied cell')

        # ── robot state ──
        self.robot_yaw = self.start_yaw
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_time = time.time()

        # ── comms ──
        self.tf_bc = TransformBroadcaster(self)

        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.cmd_sub2 = self.create_subscription(
            Twist, '/controller/cmd_vel', self._cmd_cb, 10)
        self.reset_sub = self.create_subscription(
            Empty, '/sim/reset', self._on_reset, 10)
        self.teleport_sub = self.create_subscription(
            PoseStamped, '/sim/teleport', self._on_teleport, 10)

        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.map_pub = None
        if self.publish_map_topic:
            self.map_pub = self.create_publisher(OccupancyGrid, '/map', _MAP_QOS)
        self.gt_map_pub = self.create_publisher(
            OccupancyGrid, '/ground_truth_map', _MAP_QOS)

        # timers
        self.create_timer(1.0 / 50.0, self._tick_pose)      # 50 Hz pose
        self.create_timer(1.0 / lidar_hz, self._tick_scan)   # lidar
        if self.publish_map_topic:
            self.create_timer(0.5, self._tick_map)           # 2 Hz map
        self.create_timer(5.0, self._tick_gt)                # ground truth (debug)

        self.get_logger().info(
            f'SimEnvironment: {self.map_w_m}×{self.map_h_m}m  '
            f'grid={self.gw}×{self.gh}  lidar={self.lidar_range}m×{self.lidar_rays}rays')
        self.get_logger().info(
            f'  Robot start: ({self.robot_x:.2f}, {self.robot_y:.2f}, '
            f'{math.degrees(self.robot_yaw):.1f} deg)')
        if self.fixed_map_yaml:
            self.get_logger().info(f'  Fixed map: {self.fixed_map_yaml}')

    # ================================================================ reset
    def _on_reset(self, _msg):
        """Generate a new random map and reset robot to start."""
        if self.fixed_map_yaml:
            self.robot_x = self.start_x
            self.robot_y = self.start_y
            self.robot_yaw = self.start_yaw
            self.cmd_v = 0.0
            self.cmd_w = 0.0
            self.revealed = np.zeros((self.gh, self.gw), dtype=bool)
            self.get_logger().info('Reset on fixed map')
            return

        self._map_count += 1
        rng = np.random.default_rng()
        self.gt_map = self._gen_map(rng, self.num_rooms, self.num_obs)
        self.revealed = np.zeros((self.gh, self.gw), dtype=bool)
        self.robot_x = self.start_x
        self.robot_y = self.start_y
        self.robot_yaw = self.start_yaw
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        # ensure start cell is free
        sx, sy = self._w2g(self.robot_x, self.robot_y)
        if sx is not None:
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < self.gw and 0 <= ny < self.gh:
                        self.gt_map[ny, nx] = 0
        self.get_logger().info(f'Reset → new map #{self._map_count}')

    def _on_teleport(self, msg: PoseStamped):
        """Instantly move robot to the given pose, scan, and publish map."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        q = msg.pose.orientation
        self.robot_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z))
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        # Scan and publish map immediately so the explorer
        # always has an up-to-date map at the teleported position
        self._tick_scan()
        self._tick_map()

    # ================================================================ map gen
    def _gen_map(self, rng, num_rooms, num_obs):
        """Generate a random indoor floor plan."""
        g = np.zeros((self.gh, self.gw), dtype=np.int8)

        # outer walls (2-cell thick)
        g[0:2, :] = 100;  g[-2:, :] = 100
        g[:, 0:2] = 100;  g[:, -2:] = 100

        margin = 20  # cells from edge

        for _ in range(num_rooms):
            horizontal = rng.random() < 0.5
            if horizontal:
                y = int(rng.integers(margin, self.gh - margin))
                x0 = int(rng.integers(2, self.gw // 2))
                x1 = int(rng.integers(self.gw // 2, self.gw - 2))
                # wall thickness 2 cells
                for t in range(2):
                    yy = min(y + t, self.gh - 1)
                    g[yy, x0:x1] = 100
                # door (gap 8-14 cells = 0.4 - 0.7 m)
                dw = int(rng.integers(10, 18))
                dx = int(rng.integers(x0 + 5, max(x0 + 6, x1 - dw)))
                for t in range(2):
                    yy = min(y + t, self.gh - 1)
                    g[yy, dx:dx+dw] = 0
            else:
                x = int(rng.integers(margin, self.gw - margin))
                y0 = int(rng.integers(2, self.gh // 2))
                y1 = int(rng.integers(self.gh // 2, self.gh - 2))
                for t in range(2):
                    xx = min(x + t, self.gw - 1)
                    g[y0:y1, xx] = 100
                dh = int(rng.integers(10, 18))
                dy = int(rng.integers(y0 + 5, max(y0 + 6, y1 - dh)))
                for t in range(2):
                    xx = min(x + t, self.gw - 1)
                    g[dy:dy+dh, xx] = 0

        # random box obstacles (furniture)
        for _ in range(num_obs):
            bx = int(rng.integers(margin, self.gw - margin - 10))
            by = int(rng.integers(margin, self.gh - margin - 10))
            bw = int(rng.integers(3, 12))
            bh = int(rng.integers(3, 12))
            g[by:by+bh, bx:bx+bw] = 100

        return g

    def _load_map_from_yaml(self, map_yaml_path):
        path = Path(os.path.expanduser(map_yaml_path)).resolve()
        with open(path, 'r', encoding='utf-8') as f:
            meta = yaml.safe_load(f)

        image_path = Path(meta['image'])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()

        pixels = self._read_pgm(image_path)
        self.res = float(meta['resolution'])
        origin = meta.get('origin', [0.0, 0.0, 0.0])
        self.ox = float(origin[0])
        self.oy = float(origin[1])
        self.gh, self.gw = pixels.shape
        self.map_w_m = self.gw * self.res
        self.map_h_m = self.gh * self.res

        negate = int(meta.get('negate', 0))
        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.25))
        occ_prob = pixels.astype(np.float32) / 255.0
        if not negate:
            occ_prob = 1.0 - occ_prob

        gt_map = np.full((self.gh, self.gw), -1, dtype=np.int8)
        gt_map[occ_prob >= occupied_thresh] = 100
        gt_map[occ_prob <= free_thresh] = 0
        return gt_map

    def _read_pgm(self, image_path: Path):
        with open(image_path, 'rb') as f:
            magic = f.readline().strip()
            if magic != b'P5':
                raise ValueError(f'Unsupported PGM format: {magic!r}')

            def _next_token():
                line = f.readline()
                while line.startswith(b'#'):
                    line = f.readline()
                return line

            dims = _next_token()
            while len(dims.split()) < 2:
                dims += _next_token()
            width, height = map(int, dims.split()[:2])
            maxval = int(_next_token())
            if maxval > 255:
                raise ValueError(f'Unsupported PGM maxval: {maxval}')
            data = np.frombuffer(f.read(width * height), dtype=np.uint8)
            if data.size != width * height:
                raise ValueError(f'PGM size mismatch for {image_path}')
            return data.reshape((height, width))

    # ================================================================ coord helpers
    def _w2g(self, wx, wy):
        gx = int((wx - self.ox) / self.res)
        gy = int((wy - self.oy) / self.res)
        if 0 <= gx < self.gw and 0 <= gy < self.gh:
            return gx, gy
        return None, None

    def _is_free(self, wx, wy):
        gx, gy = self._w2g(wx, wy)
        if gx is None:
            return False
        return self.gt_map[gy, gx] < 50

    # ================================================================ cmd_vel
    def _cmd_cb(self, msg: Twist):
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z

    # ================================================================ pose update
    def _tick_pose(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0.1:
            dt = 0.1  # cap for safety

        # integrate
        new_yaw = self.robot_yaw + self.cmd_w * dt
        new_x = self.robot_x + self.cmd_v * math.cos(new_yaw) * dt
        new_y = self.robot_y + self.cmd_v * math.sin(new_yaw) * dt

        # collision check — reject motion if new position is in obstacle
        if self._is_free(new_x, new_y):
            self.robot_x = new_x
            self.robot_y = new_y
        self.robot_yaw = new_yaw

        stamp = self.get_clock().now().to_msg()

        if self.publish_map_odom_tf:
            # TF: map -> odom (identity — no drift in sim)
            t1 = TransformStamped()
            t1.header.stamp = stamp
            t1.header.frame_id = 'map'
            t1.child_frame_id = 'odom'
            t1.transform.rotation.w = 1.0
            self.tf_bc.sendTransform(t1)

        # TF: odom -> base_footprint
        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'base_footprint'
        t2.transform.translation.x = self.robot_x
        t2.transform.translation.y = self.robot_y
        qz = math.sin(self.robot_yaw / 2.0)
        qw = math.cos(self.robot_yaw / 2.0)
        t2.transform.rotation.z = qz
        t2.transform.rotation.w = qw
        self.tf_bc.sendTransform(t2)

        # TF: base_footprint -> lidar_frame (identity)
        t3 = TransformStamped()
        t3.header.stamp = stamp
        t3.header.frame_id = 'base_footprint'
        t3.child_frame_id = 'lidar_frame'
        t3.transform.rotation.w = 1.0
        self.tf_bc.sendTransform(t3)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x = self.robot_x
        odom.pose.pose.position.y = self.robot_y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.cmd_v
        odom.twist.twist.angular.z = self.cmd_w
        self.odom_pub.publish(odom)

    # ================================================================ laser scan
    def _tick_scan(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'lidar_frame'
        scan.angle_min = -math.pi / 2.0
        scan.angle_max = math.pi / 2.0
        scan.angle_increment = math.pi / self.lidar_rays
        scan.range_min = 0.05
        scan.range_max = float(self.lidar_range)
        scan.time_increment = 0.0

        ranges = []
        step = self.res * 0.5  # half-cell steps for accuracy

        for i in range(self.lidar_rays):
            angle = scan.angle_min + i * scan.angle_increment
            world_angle = self.robot_yaw + angle
            dx = math.cos(world_angle) * step
            dy = math.sin(world_angle) * step

            r = 0.0
            hit = False
            cx, cy = self.robot_x, self.robot_y
            while r < self.lidar_range:
                r += step
                cx += dx
                cy += dy
                gx, gy = self._w2g(cx, cy)
                if gx is None:
                    break
                # reveal this cell
                self.revealed[gy, gx] = True
                if self.gt_map[gy, gx] > 50:
                    hit = True
                    break

            ranges.append(r if hit else float('inf'))

        # also reveal cells near the robot (even without direct hits)
        rx, ry = self._w2g(self.robot_x, self.robot_y)
        if rx is not None:
            rad = 6  # ~0.3m
            for dy in range(-rad, rad + 1):
                for dx_val in range(-rad, rad + 1):
                    nx, ny = rx + dx_val, ry + dy
                    if 0 <= nx < self.gw and 0 <= ny < self.gh:
                        self.revealed[ny, nx] = True

        scan.ranges = ranges
        self.scan_pub.publish(scan)

    # ================================================================ SLAM-like map
    def _tick_map(self):
        """Publish the map as the robot has discovered it."""
        if self.map_pub is None:
            return
        slam_map = np.full((self.gh, self.gw), -1, dtype=np.int8)
        # revealed cells get their ground-truth value
        slam_map[self.revealed] = self.gt_map[self.revealed]

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = float(self.res)
        msg.info.width = self.gw
        msg.info.height = self.gh
        msg.info.origin.position.x = self.ox
        msg.info.origin.position.y = self.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = slam_map.flatten().tolist()
        self.map_pub.publish(msg)

    def _tick_gt(self):
        """Publish ground truth map (for debug visualization)."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = float(self.res)
        msg.info.width = self.gw
        msg.info.height = self.gh
        msg.info.origin.position.x = self.ox
        msg.info.origin.position.y = self.oy
        msg.info.origin.orientation.w = 1.0
        msg.data = self.gt_map.flatten().tolist()
        self.gt_map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimEnvironment()
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
