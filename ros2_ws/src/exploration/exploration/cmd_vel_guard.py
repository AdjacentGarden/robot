#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def _clip(value, low, high):
    return max(low, min(high, value))


def _sign(value, deadband=1e-6):
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


class CmdVelGuard(Node):
    """Guard navigation cmd_vel and allow quick angular reversals."""

    def __init__(self):
        super().__init__("cmd_vel_guard")

        self.declare_parameter("input_topic", "/raw_nav_cmd_vel")
        self.declare_parameter("output_topic", "/nav_cmd_vel")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("watchdog_timeout", 0.35)
        self.declare_parameter("max_linear", 0.30)
        self.declare_parameter("max_angular", 0.65)
        self.declare_parameter("max_linear_accel", 0.35)
        self.declare_parameter("max_linear_decel", 0.55)
        self.declare_parameter("max_angular_accel", 1.20)
        self.declare_parameter("max_angular_reversal_accel", 4.50)
        self.declare_parameter("angular_deadband", 0.015)
        self.declare_parameter("min_angular_speed", 0.10)
        self.declare_parameter("log_reversals", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.watchdog_timeout = float(self.get_parameter("watchdog_timeout").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.max_linear_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_linear_decel = float(self.get_parameter("max_linear_decel").value)
        self.max_angular_accel = float(self.get_parameter("max_angular_accel").value)
        self.max_angular_reversal_accel = float(
            self.get_parameter("max_angular_reversal_accel").value
        )
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.log_reversals = bool(self.get_parameter("log_reversals").value)

        self.target_v = 0.0
        self.target_w = 0.0
        self.out_v = 0.0
        self.out_w = 0.0
        self.last_input_time = 0.0
        self.last_tick_time = time.monotonic()
        self._last_reversal_log = 0.0

        self.create_subscription(Twist, self.input_topic, self._cmd_cb, 10)
        self.pub = self.create_publisher(Twist, self.output_topic, 10)
        self.create_timer(1.0 / max(1.0, self.publish_hz), self._tick)

        self.get_logger().info(
            "cmd_vel_guard started: %s -> %s, reversal_accel=%.2f"
            % (self.input_topic, self.output_topic, self.max_angular_reversal_accel)
        )

    def _cmd_cb(self, msg):
        self.target_v = _clip(float(msg.linear.x), -self.max_linear, self.max_linear)
        self.target_w = _clip(float(msg.angular.z), -self.max_angular, self.max_angular)
        if abs(self.target_w) < self.angular_deadband:
            self.target_w = 0.0
        self.last_input_time = time.monotonic()

    @staticmethod
    def _step(current, target, limit):
        delta = target - current
        if abs(delta) <= limit:
            return target
        return current + math.copysign(limit, delta)

    def _limit_linear(self, dt):
        accel = self.max_linear_accel if self.target_v >= self.out_v else self.max_linear_decel
        self.out_v = self._step(self.out_v, self.target_v, max(0.0, accel) * dt)

    def _limit_angular(self, dt):
        target_sign = _sign(self.target_w, self.angular_deadband)
        output_sign = _sign(self.out_w, self.angular_deadband)
        reversing = target_sign != 0 and output_sign != 0 and target_sign != output_sign

        accel = self.max_angular_reversal_accel if reversing else self.max_angular_accel
        prev_w = self.out_w
        self.out_w = self._step(self.out_w, self.target_w, max(0.0, accel) * dt)

        if self.target_w == 0.0 and abs(self.out_w) < self.angular_deadband:
            self.out_w = 0.0

        if self.target_w != 0.0 and _sign(self.out_w, self.angular_deadband) == target_sign:
            if abs(self.out_w) < self.min_angular_speed:
                self.out_w = math.copysign(self.min_angular_speed, self.target_w)

        if reversing and self.log_reversals:
            now = time.monotonic()
            if now - self._last_reversal_log > 0.25:
                self.get_logger().warn(
                    "angular reversal: raw %.3f -> %.3f, guarded %.3f -> %.3f"
                    % (prev_w, self.target_w, prev_w, self.out_w)
                )
                self._last_reversal_log = now

    def _tick(self):
        now = time.monotonic()
        dt = max(1e-3, min(0.2, now - self.last_tick_time))
        self.last_tick_time = now

        if self.last_input_time <= 0.0 or now - self.last_input_time > self.watchdog_timeout:
            self.target_v = 0.0
            self.target_w = 0.0

        self._limit_linear(dt)
        self._limit_angular(dt)

        msg = Twist()
        msg.linear.x = float(_clip(self.out_v, -self.max_linear, self.max_linear))
        msg.angular.z = float(_clip(self.out_w, -self.max_angular, self.max_angular))
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

