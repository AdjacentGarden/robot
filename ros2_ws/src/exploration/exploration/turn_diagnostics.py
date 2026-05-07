#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    from ros_robot_controller_msgs.msg import MotorsState
except Exception:
    MotorsState = None


def _sign(value, deadband):
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


class TurnDiagnostics(Node):
    def __init__(self):
        super().__init__("turn_diagnostics")
        self.declare_parameter("raw_topic", "/raw_nav_cmd_vel")
        self.declare_parameter("guarded_topic", "/nav_cmd_vel")
        self.declare_parameter("final_topic", "/cmd_vel")
        self.declare_parameter("motor_topic", "/ros_robot_controller/set_motor")
        self.declare_parameter("angular_deadband", 0.04)
        self.declare_parameter("stale_threshold", 0.6)

        self.raw_topic = str(self.get_parameter("raw_topic").value)
        self.guarded_topic = str(self.get_parameter("guarded_topic").value)
        self.final_topic = str(self.get_parameter("final_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)
        self.angular_deadband = float(self.get_parameter("angular_deadband").value)
        self.stale_threshold = float(self.get_parameter("stale_threshold").value)

        self.state = {
            "raw": {"sign": 0, "w": 0.0, "t": 0.0},
            "guarded": {"sign": 0, "w": 0.0, "t": 0.0},
            "final": {"sign": 0, "w": 0.0, "t": 0.0},
            "motor": {"sign": 0, "w": 0.0, "t": 0.0},
        }

        self.create_subscription(Twist, self.raw_topic, lambda msg: self._twist_cb("raw", msg), 10)
        self.create_subscription(Twist, self.guarded_topic, lambda msg: self._twist_cb("guarded", msg), 10)
        self.create_subscription(Twist, self.final_topic, lambda msg: self._twist_cb("final", msg), 10)
        if MotorsState is not None:
            self.create_subscription(MotorsState, self.motor_topic, self._motor_cb, 10)
        self.create_timer(0.2, self._check_chain)

    def _update(self, name, w):
        now = time.monotonic()
        sign = _sign(w, self.angular_deadband)
        prev = self.state[name]
        if sign != 0 and prev["sign"] != 0 and sign != prev["sign"]:
            self.get_logger().warn("%s angular sign reversal: %.3f -> %.3f" % (name, prev["w"], w))
        prev.update({"sign": sign, "w": w, "t": now})

    def _twist_cb(self, name, msg):
        self._update(name, float(msg.angular.z))

    def _motor_cb(self, msg):
        values = {int(item.id): float(item.rps) for item in msg.data}
        if 1 not in values or 2 not in values:
            return
        right_ground = values[1]
        left_ground = -values[2]
        self._update("motor", left_ground - right_ground)

    def _check_chain(self):
        now = time.monotonic()
        raw = self.state["raw"]
        guarded = self.state["guarded"]
        final = self.state["final"]
        motor = self.state["motor"]

        if raw["sign"] != 0 and guarded["sign"] != 0 and raw["sign"] != guarded["sign"]:
            if now - raw["t"] > self.stale_threshold:
                self.get_logger().error(
                    "guarded topic has not followed raw turn sign: raw=%.3f guarded=%.3f"
                    % (raw["w"], guarded["w"])
                )

        if guarded["sign"] != 0 and final["sign"] != 0 and guarded["sign"] != final["sign"]:
            if now - guarded["t"] > self.stale_threshold:
                self.get_logger().error(
                    "final /cmd_vel has not followed guarded turn sign: guarded=%.3f final=%.3f"
                    % (guarded["w"], final["w"])
                )

        if motor["t"] > 0 and final["sign"] != 0 and motor["sign"] != 0 and final["sign"] != motor["sign"]:
            if now - final["t"] > self.stale_threshold:
                self.get_logger().error(
                    "motor output sign differs from /cmd_vel: cmd=%.3f motor_est=%.3f"
                    % (final["w"], motor["w"])
                )


def main(args=None):
    rclpy.init(args=args)
    node = TurnDiagnostics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

