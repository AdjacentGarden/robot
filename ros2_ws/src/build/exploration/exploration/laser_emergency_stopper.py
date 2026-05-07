#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class LaserEmergencyStopper(Node):
    def __init__(self):
        super().__init__('laser_emergency_stopper')
        
        self.declare_parameter('check_radius', 0.20)
        self.declare_parameter('min_points_for_block', 3)

        self.check_radius = float(self.get_parameter('check_radius').value)
        self.min_points_for_block = int(self.get_parameter('min_points_for_block').value)
        
        self.obstacle_detected = False
        self.debug_count = 0
        
        # Subscribe to raw scan
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)
            
        # Subscribe to incoming cmd_vel from navigator
        self.cmd_sub = self.create_subscription(
            Twist, '/nav_cmd_vel', self._cmd_cb, 10)
            
        # Publish to real robot cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("LaserEmergencyStopper node started. Intercepting /nav_cmd_vel -> /cmd_vel")

    def _scan_cb(self, msg):
        self.debug_count += 1
        danger_points_count = 0
        valid_points_count = 0
        
        # 保持与现有系统一致：/scan 已由 laser_filters 预处理
        # 这里只做基础有效性判断 + 近距离点计数
        for r in msg.ranges:
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
            valid_points_count += 1
            if r <= self.check_radius:
                danger_points_count += 1
                    
        has_obstacle = (danger_points_count >= self.min_points_for_block)

        if self.debug_count % 100 == 0:
            self.get_logger().info(
                f"Received {self.debug_count} scan messages... valid={valid_points_count}, "
                f"near({self.check_radius:.2f}m)={danger_points_count}, "
                f"range_min={msg.range_min:.3f}, range_max={msg.range_max:.2f}")
                
        # 突变沿检测：刚刚从安全变成危险时打印并紧急硬刹车
        if has_obstacle and not self.obstacle_detected:
            self.get_logger().warn(
                f"Obstacle detected! near={danger_points_count} (threshold={self.min_points_for_block}) "
                f"inside {self.check_radius:.2f}m. Emergency brake.")
            self.cmd_pub.publish(Twist()) # sending 0 to hard-stop immediately
            
        elif not has_obstacle and self.obstacle_detected:
            self.get_logger().info("Obstacle cleared.")
            
        self.obstacle_detected = has_obstacle

    def _cmd_cb(self, msg):
        cmd_out = Twist()
        
        if self.obstacle_detected:
            # If attempting to go forward or rotate in place, stop completely
            if msg.linear.x >= 0.0:
                cmd_out.linear.x = 0.0
                cmd_out.angular.z = 0.0
            else:
                # If backing up (linear.x < 0), let it pass for recovery
                cmd_out = msg
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
