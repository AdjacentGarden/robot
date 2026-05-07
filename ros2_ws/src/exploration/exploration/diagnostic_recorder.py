#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException


class DiagnosticRecorder(Node):
    def __init__(self):
        super().__init__('diagnostic_recorder')

        default_log_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'log'))
        self.declare_parameter('log_root', default_log_root)
        self.declare_parameter('pose_log_hz', 2.0)

        log_root = str(self.get_parameter('log_root').value)
        pose_log_hz = float(self.get_parameter('pose_log_hz').value)

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_root, stamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._files = {
            'cmd_vel': open(os.path.join(self.log_dir, 'cmd_vel.jsonl'), 'a', buffering=1, encoding='utf-8'),
            'controller_cmd_vel': open(os.path.join(self.log_dir, 'controller_cmd_vel.jsonl'), 'a', buffering=1, encoding='utf-8'),
            'odom': open(os.path.join(self.log_dir, 'odom.jsonl'), 'a', buffering=1, encoding='utf-8'),
            'exploration_status': open(os.path.join(self.log_dir, 'exploration_status.jsonl'), 'a', buffering=1, encoding='utf-8'),
            'navigation_status': open(os.path.join(self.log_dir, 'navigation_status.jsonl'), 'a', buffering=1, encoding='utf-8'),
            'tf_pose': open(os.path.join(self.log_dir, 'tf_pose.jsonl'), 'a', buffering=1, encoding='utf-8'),
        }

        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 20)
        self.create_subscription(Twist, '/controller/cmd_vel', self._controller_cmd_vel_cb, 20)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 20)
        self.create_subscription(String, '/exploration/status', self._exploration_status_cb, 20)
        self.create_subscription(String, '/navigation/status', self._navigation_status_cb, 20)

        self.create_timer(max(0.1, 1.0 / max(pose_log_hz, 0.1)), self._log_tf_pose)
        self.get_logger().info(f'diagnostic logs -> {self.log_dir}')

    def destroy_node(self):
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:
                pass
        super().destroy_node()

    def _write(self, key, payload):
        payload['t'] = time.time()
        self._files[key].write(json.dumps(payload, ensure_ascii=True) + '\n')

    def _cmd_vel_cb(self, msg):
        self._write('cmd_vel', {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z,
        })

    def _controller_cmd_vel_cb(self, msg):
        self._write('controller_cmd_vel', {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z,
        })

    def _odom_cb(self, msg):
        self._write('odom', {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'linear_x': msg.twist.twist.linear.x,
            'angular_z': msg.twist.twist.angular.z,
            'frame_id': msg.header.frame_id,
            'child_frame_id': msg.child_frame_id,
        })

    def _exploration_status_cb(self, msg):
        self._write('exploration_status', {'data': msg.data})

    def _navigation_status_cb(self, msg):
        self._write('navigation_status', {'data': msg.data})

    def _log_tf_pose(self):
        for parent, child, key in (
            ('map', 'base_footprint', 'map_base_footprint'),
            ('odom', 'base_footprint', 'odom_base_footprint'),
        ):
            try:
                tf = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)
                self._write('tf_pose', {
                    'pair': key,
                    'x': t.x,
                    'y': t.y,
                    'z': t.z,
                    'yaw': yaw,
                })
            except TransformException:
                continue

    @staticmethod
    def _quat_to_yaw(x, y, z, w):
        return float(
            __import__('math').atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticRecorder()
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
