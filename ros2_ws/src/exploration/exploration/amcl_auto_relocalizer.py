#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty


class AmclAutoRelocalizer(Node):
    def __init__(self) -> None:
        super().__init__('amcl_auto_relocalizer')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('status_topic', '/localization/status')
        self.declare_parameter('startup_delay', 2.0)
        self.declare_parameter('nomotion_update_hz', 2.0)
        self.declare_parameter('motionless_timeout', 5.0)
        self.declare_parameter('enable_motion_assist', True)
        self.declare_parameter('assist_linear_speed', 0.04)
        self.declare_parameter('assist_angular_speed', 0.35)
        self.declare_parameter('assist_duration', 8.0)
        self.declare_parameter('front_obstacle_angle_deg', 20.0)
        self.declare_parameter('front_obstacle_distance', 0.35)
        self.declare_parameter('localized_xy_std_threshold', 0.25)
        self.declare_parameter('localized_yaw_std_threshold', 0.35)
        self.declare_parameter('localized_consecutive_msgs', 4)
        self.declare_parameter('lost_xy_std_threshold', 0.60)
        self.declare_parameter('lost_yaw_std_threshold', 0.80)
        self.declare_parameter('lost_timeout', 3.0)
        self.declare_parameter('retry_cooldown', 12.0)

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)

        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._on_amcl_pose, 20)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._on_scan, 20)

        self.global_loc_cli = self.create_client(
            Empty, '/reinitialize_global_localization')
        self.nomotion_cli = self.create_client(
            Empty, '/request_nomotion_update')

        self.active = False
        self.phase = 'idle'
        self.phase_deadline = 0.0
        self.started_at = 0.0
        self.last_attempt_end = 0.0
        self.last_nomotion_call = 0.0
        self.last_good_time = 0.0
        self.last_bad_time = 0.0
        self.stable_count = 0
        self.front_min_range = math.inf
        self.motion_sign = 1.0

        self.create_timer(0.1, self._tick)
        self.create_timer(
            max(0.2, 1.0 / max(0.2, float(self.get_parameter('nomotion_update_hz').value))),
            self._nomotion_tick,
        )
        self.create_timer(float(self.get_parameter('startup_delay').value), self._startup_once)
        self._startup_triggered = False

        self.get_logger().info('AMCL auto relocalizer ready')

    def _startup_once(self) -> None:
        if self._startup_triggered:
            return
        self._startup_triggered = self._begin_relocalization('startup')

    def _on_scan(self, msg: LaserScan) -> None:
        angle_limit = math.radians(float(self.get_parameter('front_obstacle_angle_deg').value))
        front_min = math.inf
        angle = msg.angle_min
        for rng in msg.ranges:
            if abs(angle) <= angle_limit and math.isfinite(rng) and rng >= msg.range_min:
                front_min = min(front_min, rng)
            angle += msg.angle_increment
        self.front_min_range = front_min

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        cov = msg.pose.covariance
        xy_std = math.sqrt(max(cov[0], cov[7], 0.0))
        yaw_std = math.sqrt(max(cov[35], 0.0))
        now = time.time()

        localized = (
            xy_std <= float(self.get_parameter('localized_xy_std_threshold').value) and
            yaw_std <= float(self.get_parameter('localized_yaw_std_threshold').value)
        )
        lost = (
            xy_std >= float(self.get_parameter('lost_xy_std_threshold').value) or
            yaw_std >= float(self.get_parameter('lost_yaw_std_threshold').value)
        )

        if localized:
            self.last_good_time = now
            self.last_bad_time = 0.0
            self.stable_count += 1
            if self.active and self.stable_count >= int(
                    self.get_parameter('localized_consecutive_msgs').value):
                self._finish_relocalization(success=True)
            return

        self.stable_count = 0
        if lost:
            if self.last_bad_time == 0.0:
                self.last_bad_time = now
        else:
            self.last_bad_time = 0.0

    def _tick(self) -> None:
        now = time.time()

        if self.active:
            if now >= self.phase_deadline:
                if self.phase == 'motionless' and bool(self.get_parameter('enable_motion_assist').value):
                    self.phase = 'assist_motion'
                    self.phase_deadline = now + float(self.get_parameter('assist_duration').value)
                    self.motion_sign *= -1.0
                    self._publish_status('assist_motion')
                else:
                    self._finish_relocalization(success=False)
                    return

            if self.phase == 'assist_motion':
                self._publish_motion_command()
            elif self.phase == 'motionless':
                self._stop_robot()
            return

        if (
            self.last_bad_time > 0.0 and
            now - self.last_bad_time >= float(self.get_parameter('lost_timeout').value) and
            now - self.last_attempt_end >= float(self.get_parameter('retry_cooldown').value)
        ):
            self._begin_relocalization('runtime')

    def _nomotion_tick(self) -> None:
        if not self.active:
            return
        if not self.nomotion_cli.service_is_ready():
            return
        now = time.time()
        if now - self.last_nomotion_call < 0.3:
            return
        self.last_nomotion_call = now
        self.nomotion_cli.call_async(Empty.Request())

    def _begin_relocalization(self, reason: str) -> bool:
        if self.active:
            return False
        if not self.global_loc_cli.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn('reinitialize_global_localization service unavailable')
            return False

        self.active = True
        self.phase = 'motionless'
        self.started_at = time.time()
        self.phase_deadline = self.started_at + float(self.get_parameter('motionless_timeout').value)
        self.stable_count = 0
        self.last_bad_time = 0.0
        self.last_nomotion_call = 0.0

        self.global_loc_cli.call_async(Empty.Request())
        self._publish_status(f'relocalizing:{reason}')
        self.get_logger().warn(f'start AMCL relocalization ({reason})')
        return True

    def _finish_relocalization(self, success: bool) -> None:
        self.active = False
        self.phase = 'idle'
        self.phase_deadline = 0.0
        self.last_attempt_end = time.time()
        self._stop_robot()
        if success:
            self._publish_status('localized')
            self.get_logger().info('AMCL relocalization converged')
        else:
            self._publish_status('relocalization_timeout')
            self.get_logger().warn('AMCL relocalization timed out')

    def _publish_motion_command(self) -> None:
        msg = Twist()
        obstacle_distance = float(self.get_parameter('front_obstacle_distance').value)
        if self.front_min_range <= obstacle_distance:
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        else:
            msg.linear.x = float(self.get_parameter('assist_linear_speed').value)
            msg.angular.z = self.motion_sign * float(self.get_parameter('assist_angular_speed').value)
        self.cmd_pub.publish(msg)

    def _stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AmclAutoRelocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
