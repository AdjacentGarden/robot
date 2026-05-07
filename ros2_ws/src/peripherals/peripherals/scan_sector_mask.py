#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def _normalize_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def _angle_diff(a: float, b: float) -> float:
    return _normalize_angle(a - b)


class ScanSectorMask(Node):
    def __init__(self) -> None:
        super().__init__('scan_sector_mask')

        self.declare_parameter('enabled', True)
        self.declare_parameter('mode', 'mask')
        self.declare_parameter('sector_centers_deg', [60.0, -60.0, 180.0])
        self.declare_parameter('sector_width_deg', 10.0)
        self.declare_parameter('sector_margin_deg', 2.0)
        self.declare_parameter('keep_sector_ranges_deg', [-180.0, 180.0])
        self.declare_parameter('replacement_value', float('inf'))

        self.enabled = bool(self.get_parameter('enabled').value)
        mode = str(self.get_parameter('mode').value).strip().lower()
        self.keep_mode = (mode == 'keep')

        centers_deg = list(self.get_parameter('sector_centers_deg').value)
        self.sector_centers_rad = [_normalize_angle(math.radians(float(x))) for x in centers_deg]
        self.sector_half_width_rad = math.radians(
            max(0.0, float(self.get_parameter('sector_width_deg').value) * 0.5)
            + max(0.0, float(self.get_parameter('sector_margin_deg').value))
        )

        keep_ranges_deg = [float(v) for v in list(self.get_parameter('keep_sector_ranges_deg').value)]
        if len(keep_ranges_deg) % 2 != 0:
            self.get_logger().warn(
                'keep_sector_ranges_deg has odd length, ignoring last value')
            keep_ranges_deg = keep_ranges_deg[:-1]
        self.keep_ranges_rad = []
        for i in range(0, len(keep_ranges_deg), 2):
            start_deg = keep_ranges_deg[i]
            end_deg = keep_ranges_deg[i + 1]
            self.keep_ranges_rad.append((
                _normalize_angle(math.radians(start_deg)),
                _normalize_angle(math.radians(end_deg)),
            ))

        self.replacement_value = float(self.get_parameter('replacement_value').value)

        self.sub = self.create_subscription(
            LaserScan,
            'scan_in',
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(LaserScan, 'scan_out', qos_profile_sensor_data)

        if self.keep_mode:
            self.get_logger().info(
                'scan_sector_mask started: enabled=%s, mode=keep, ranges(deg)=%s'
                % (
                    self.enabled,
                    [
                        (
                            round(math.degrees(s), 1),
                            round(math.degrees(e), 1),
                        )
                        for s, e in self.keep_ranges_rad
                    ],
                )
            )
        else:
            self.get_logger().info(
                'scan_sector_mask started: enabled=%s, mode=mask, centers(deg)=%s, width(deg)=%.2f, margin(deg)=%.2f'
                % (
                    self.enabled,
                    [round(math.degrees(v), 1) for v in self.sector_centers_rad],
                    float(self.get_parameter('sector_width_deg').value),
                    float(self.get_parameter('sector_margin_deg').value),
                )
            )

    def _in_mask_sector(self, angle_rad: float) -> bool:
        for c in self.sector_centers_rad:
            if abs(_angle_diff(angle_rad, c)) <= self.sector_half_width_rad:
                return True
        return False

    @staticmethod
    def _in_angular_range(angle_rad: float, start_rad: float, end_rad: float) -> bool:
        if start_rad <= end_rad:
            return start_rad <= angle_rad <= end_rad
        # wrapped range, e.g. [130deg, -130deg]
        return angle_rad >= start_rad or angle_rad <= end_rad

    def _in_keep_sector(self, angle_rad: float) -> bool:
        for start, end in self.keep_ranges_rad:
            if self._in_angular_range(angle_rad, start, end):
                return True
        return False

    def _should_mask(self, angle_rad: float) -> bool:
        if self.keep_mode:
            if not self.keep_ranges_rad:
                return False
            return not self._in_keep_sector(angle_rad)
        return self._in_mask_sector(angle_rad)

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self.enabled or not msg.ranges:
            self.pub.publish(msg)
            return

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max

        ranges = list(msg.ranges)
        angle = msg.angle_min
        masked_count = 0

        for i in range(len(ranges)):
            cur = _normalize_angle(angle + i * msg.angle_increment)
            if self._should_mask(cur):
                ranges[i] = self.replacement_value
                masked_count += 1

        out.ranges = ranges
        out.intensities = list(msg.intensities)

        if masked_count > 0:
            self.get_logger().info(
                f'masked {masked_count}/{len(ranges)} scan points',
                throttle_duration_sec=5.0,
            )

        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanSectorMask()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
