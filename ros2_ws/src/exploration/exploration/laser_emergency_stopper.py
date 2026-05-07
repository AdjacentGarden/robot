#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def _normalize_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def _angle_diff(a: float, b: float) -> float:
    return _normalize_angle(a - b)

class LaserEmergencyStopper(Node):
    def __init__(self):
        super().__init__('laser_emergency_stopper')
        
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('check_radius', 0.20)
        self.declare_parameter('min_points_for_block', 3)
        self.declare_parameter('max_continuous_move_time', 2.0)
        self.declare_parameter('forced_rest_time', 0.6)
        self.declare_parameter('max_vel_x', 0.22)
        self.declare_parameter('emergency_timeout', 0.0)
        self.declare_parameter('monitor_sector_center_deg', 180.0)
        self.declare_parameter('monitor_sector_width_deg', 100.0)
        self.declare_parameter('ignore_sector_centers_deg', [0.0])
        self.declare_parameter('ignore_sector_width_deg', 0.0)
        self.declare_parameter('ignore_sector_margin_deg', 0.0)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.check_radius = float(self.get_parameter('check_radius').value)
        self.min_points_for_block = int(self.get_parameter('min_points_for_block').value)
        self.max_continuous_move_time = float(self.get_parameter('max_continuous_move_time').value)
        self.forced_rest_time = float(self.get_parameter('forced_rest_time').value)
        self.max_vel_x = float(self.get_parameter('max_vel_x').value)
        self.emergency_timeout = float(self.get_parameter('emergency_timeout').value)

        monitor_center_deg = float(self.get_parameter('monitor_sector_center_deg').value)
        monitor_width_deg = max(1.0, float(self.get_parameter('monitor_sector_width_deg').value))
        self.monitor_center_rad = _normalize_angle(math.radians(monitor_center_deg))
        self.monitor_half_width_rad = min(math.pi, math.radians(monitor_width_deg) * 0.5)

        ignore_centers_deg = list(self.get_parameter('ignore_sector_centers_deg').value)
        self.ignore_centers_rad = [_normalize_angle(math.radians(float(v))) for v in ignore_centers_deg]
        ignore_width_deg = max(0.0, float(self.get_parameter('ignore_sector_width_deg').value))
        ignore_margin_deg = max(0.0, float(self.get_parameter('ignore_sector_margin_deg').value))
        self.ignore_half_width_rad = math.radians(ignore_width_deg * 0.5 + ignore_margin_deg)
        
        self.obstacle_detected = False
        self.obstacle_start_time = 0.0
        self.obstacle_suppressed = False
        self.min_obstacle_distance = float('inf')
        self.last_danger_points_count = 0
        self.debug_count = 0
        self._last_block_log_time = 0.0

        # Stop-and-Go state
        self.is_moving_state = False
        self.motion_start_time = 0.0
        self.forced_stop_end_time = 0.0
        
        # Subscribe to selected scan topic (default /scan)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_cb, qos_profile_sensor_data)
            
        # Subscribe to incoming cmd_vel from navigator
        self.cmd_sub = self.create_subscription(
            Twist, '/nav_cmd_vel', self._cmd_cb, 10)
            
        # Publish to real robot cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info(
            "LaserEmergencyStopper started: scan_topic=%s, check_radius=%.2f, min_points=%d, monitor_center=%.1fdeg, monitor_width=%.1fdeg, ignore_centers=%s"
            % (
                self.scan_topic,
                self.check_radius,
                self.min_points_for_block,
                math.degrees(self.monitor_center_rad),
                math.degrees(self.monitor_half_width_rad) * 2.0,
                [round(math.degrees(v), 1) for v in self.ignore_centers_rad],
            )
        )

    def _in_monitor_sector(self, angle_rad):
        return abs(_angle_diff(angle_rad, self.monitor_center_rad)) <= self.monitor_half_width_rad

    def _in_ignore_sector(self, angle_rad):
        for center in self.ignore_centers_rad:
            if abs(_angle_diff(angle_rad, center)) <= self.ignore_half_width_rad:
                return True
        return False

    def _scan_cb(self, msg):
        self.debug_count += 1
        danger_points_count = 0
        valid_points_count = 0
        used_points_count = 0
        min_distance = float('inf')
        
        # 保持与现有系统一致：输入激光已由前级 laser_filters 做过基础预处理
        # 这里只做角度筛选 + 有效性判断 + 近距离点计数
        for i, r in enumerate(msg.ranges):
            cur_angle = _normalize_angle(msg.angle_min + i * msg.angle_increment)
            if not self._in_monitor_sector(cur_angle):
                continue
            if self._in_ignore_sector(cur_angle):
                continue

            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
            used_points_count += 1
            valid_points_count += 1
            min_distance = min(min_distance, float(r))
            if r <= self.check_radius:
                danger_points_count += 1
                    
        has_obstacle = (danger_points_count >= self.min_points_for_block)
        self.min_obstacle_distance = min_distance
        self.last_danger_points_count = danger_points_count

        if self.debug_count % 100 == 0:
            self.get_logger().info(
                f"Received {self.debug_count} scan messages... used={used_points_count}, valid={valid_points_count}, "
                f"near({self.check_radius:.2f}m)={danger_points_count}, "
                f"range_min={msg.range_min:.3f}, range_max={msg.range_max:.2f}")
                
        # 突变沿检测：刚刚从安全变成危险时打印并紧急硬刹车
        if has_obstacle and not self.obstacle_detected:
            self.get_logger().warn(
                f"[EMERGENCY_STOP] obstacle_detected=true near_points={danger_points_count} "
                f"threshold={self.min_points_for_block} min_distance="
                f"{min_distance:.3f}m check_radius={self.check_radius:.2f}m")
            self.cmd_pub.publish(Twist()) # sending 0 to hard-stop immediately
            self.obstacle_start_time = self.get_clock().now().nanoseconds / 1e9
            self.obstacle_suppressed = False
            
        elif not has_obstacle and self.obstacle_detected:
            self.get_logger().info(
                f"[EMERGENCY_STOP] cleared near_points={danger_points_count} "
                f"min_distance={min_distance if math.isfinite(min_distance) else -1.0:.3f}m")
            self.obstacle_suppressed = False
            
        self.obstacle_detected = has_obstacle

    def _cmd_cb(self, msg):
        cmd_out = Twist()
        
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check if the incoming command is a moving command at high speed (>= 60% of max speed)
        is_high_speed_cmd = msg.linear.x >= (0.6 * self.max_vel_x)
        is_resting = False

        if current_time < self.forced_stop_end_time:
            is_resting = True
        else:
            # Step B & C: Stop-and-Go logic
            if is_high_speed_cmd:
                if not self.is_moving_state:
                    # Just started moving fast
                    self.is_moving_state = True
                    self.motion_start_time = current_time
                else:
                    # Have been moving fast. Check if we exceeded max continuous time
                    move_duration = current_time - self.motion_start_time
                    if move_duration >= self.max_continuous_move_time:
                        self.get_logger().info(f"Stop-and-Go: High speed moving for {move_duration:.1f}s, forcing rest for {self.forced_rest_time}s.")
                        self.forced_stop_end_time = current_time + self.forced_rest_time
                        self.is_moving_state = False
                        is_resting = True
            else:
                # Speed dropped below threshold, reset continuous fast moving state
                self.is_moving_state = False

        is_obstacle_blocking = self.obstacle_detected
        if is_obstacle_blocking and self.emergency_timeout > 0.0:
            if current_time - self.obstacle_start_time > self.emergency_timeout:
                if not self.obstacle_suppressed:
                    self.get_logger().warn(
                        f"[EMERGENCY_STOP] timeout_released after {self.emergency_timeout:.1f}s "
                        f"min_distance={self.min_obstacle_distance if math.isfinite(self.min_obstacle_distance) else -1.0:.3f}m "
                        f"near_points={self.last_danger_points_count}")
                    self.obstacle_suppressed = True
                is_obstacle_blocking = False

        # Apply stop logic, same as emergency_stopper
        if is_obstacle_blocking or is_resting:
            # If attempting to go forward or rotate in place, stop completely
            if msg.linear.x >= 0.0:
                cmd_out.linear.x = 0.0
                cmd_out.angular.z = 0.0
            else:
                # If backing up (linear.x < 0), let it pass for recovery
                cmd_out = msg
            if current_time - self._last_block_log_time > 0.2:
                reason = 'obstacle' if is_obstacle_blocking else 'forced_rest'
                self.get_logger().warn(
                    f"[EMERGENCY_STOP] gating_cmd reason={reason} "
                    f"obstacle_detected={str(self.obstacle_detected).lower()} "
                    f"min_distance={self.min_obstacle_distance if math.isfinite(self.min_obstacle_distance) else -1.0:.3f}m "
                    f"near_points={self.last_danger_points_count} "
                    f"in_linear_x={msg.linear.x:.3f} in_angular_z={msg.angular.z:.3f} "
                    f"out_linear_x={cmd_out.linear.x:.3f} out_angular_z={cmd_out.angular.z:.3f}")
                self._last_block_log_time = current_time
        else:
            cmd_out = msg
            
        self.cmd_pub.publish(cmd_out)

def main(args=None):
    rclpy.init(args=args)
    node = LaserEmergencyStopper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
