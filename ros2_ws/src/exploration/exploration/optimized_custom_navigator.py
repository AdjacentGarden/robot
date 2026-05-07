#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

from .custom_navigator import CustomNavigator


class OptimizedCustomNavigator(CustomNavigator):
    """CustomNavigator variant with fast angular sign reversal.

    The path follower can legitimately need to switch from a left turn to a
    right turn in consecutive control cycles.  The original angular acceleration
    limiter plus EMA smoothing can keep angular.z in the old direction for too
    long, so this class allows a faster path through zero when the turn sign
    changes.
    """

    def __init__(self):
        super().__init__()
        self.declare_parameter("max_angular_reversal_accel", 4.5)
        self.declare_parameter("turn_sign_lock_margin", 0.04)
        self.max_angular_reversal_accel = float(
            self.get_parameter("max_angular_reversal_accel").value
        )
        self.turn_sign_lock_margin = float(
            self.get_parameter("turn_sign_lock_margin").value
        )
        self.get_logger().info(
            "OptimizedCustomNavigator active: max_angular_reversal_accel=%.2f"
            % self.max_angular_reversal_accel
        )

    def _stabilize_turn_error(self, err):
        if abs(err) < self.steering_deadband_angle:
            self._last_turn_sign = 0.0
            return 0.0

        # Keep the previous side only at the exact +/-pi ambiguity.  Normal
        # path heading changes must be able to flip turn direction immediately.
        if (
            self._last_turn_sign != 0.0
            and self.turn_sign_lock_margin > 0.0
            and abs(abs(err) - math.pi) < self.turn_sign_lock_margin
        ):
            err = math.copysign(abs(err), self._last_turn_sign)

        sign = math.copysign(1.0, err) if abs(err) > 1e-3 else self._last_turn_sign
        self._last_turn_sign = sign if abs(err) > 0.05 else 0.0
        return err

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

        reversing = self.cur_w * w < -1e-4 and abs(w) > max(
            self.steering_deadband_angle,
            0.01,
        )
        angular_limit = (
            self.max_angular_reversal_accel if reversing else self.max_accel_theta
        )
        dw = w - self.cur_w
        max_dw = angular_limit * dt
        if abs(dw) > max_dw:
            w = self.cur_w + math.copysign(max_dw, dw)

        if abs(v) > 0.04:
            w += self.forward_angular_trim

        alpha = max(0.0, min(1.0, self.cmd_smoothing_alpha))
        if alpha > 0.0 and not reversing:
            v = (1.0 - alpha) * self.cur_v + alpha * v
            w = (1.0 - alpha) * self.cur_w + alpha * w

        if abs(w) < 1e-3:
            w = 0.0
        return v, w


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedCustomNavigator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

