#!/usr/bin/env python3
"""
轻量级路径跟踪控制器 - 用于自主探索
订阅Explore Lite的目标点，使用纯追踪算法计算速度命令
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import TransformListener, TransformException, Buffer
import math

class SimplePathTracker(Node):
    def __init__(self):
        super().__init__('simple_path_tracker')
        
        # 声明参数
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('max_linear_vel', 0.3)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('goal_tolerance', 0.2)
        
        # 获取参数
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        
        # TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 订阅目标点（来自explore_lite或RViz）
        # 订阅多个可能的goal话题
        self.goal_sub1 = self.create_subscription(
            PoseStamped,
            'move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        self.goal_sub2 = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # 发布速度命令
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 当前目标
        self.current_goal = None
        self.reached_goal = False
        
        # 控制循环
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f'Simple Path Tracker started')
        self.get_logger().info(f'  map_frame: {self.map_frame}')
        self.get_logger().info(f'  base_frame: {self.base_frame}')
        self.get_logger().info(f'  Subscribing to: move_base_simple/goal and /goal_pose')
        self.get_logger().info(f'  Publishing cmd_vel')
    
    def goal_callback(self, msg):
        """接收新的目标点"""
        self.current_goal = msg
        self.reached_goal = False
        self.get_logger().info(f'🎯 New goal received!')
        self.get_logger().info(f'   Position: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self.get_logger().info(f'   Frame: {msg.header.frame_id}')
    
    def control_loop(self):
        """主控制循环"""
        if self.current_goal is None:
            return
        
        try:
            # 获取当前机器人位置（base_link相对于map）
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            # 获取机器人方向（从四元数）
            q = transform.transform.rotation
            robot_yaw = self.quaternion_to_yaw(q)
            
            # 计算与目标的相对位置
            goal_x = self.current_goal.pose.position.x
            goal_y = self.current_goal.pose.position.y
            
            dx = goal_x - robot_x
            dy = goal_y - robot_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # 检查是否到达目标
            if distance < self.goal_tolerance:
                if not self.reached_goal:
                    self.get_logger().info('✅ Goal reached!')
                    self.reached_goal = True
                vel_cmd = Twist()  # 停止
                self.vel_pub.publish(vel_cmd)
                return
            
            # 纯追踪算法
            goal_yaw = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(goal_yaw - robot_yaw)
            
            # 线速度：距离越近越慢
            linear_vel = self.max_linear_vel * min(1.0, distance / 1.0)
            if linear_vel < 0.05:
                linear_vel = 0.05  # 最小速度
            
            # 角速度：方向误差纠正
            if abs(yaw_error) > 0.1:  # 偏离超过5.7度才转向
                angular_vel = self.max_angular_vel * min(1.0, abs(yaw_error) / 1.57)
                if yaw_error < 0:
                    angular_vel = -angular_vel
                linear_vel *= 0.5  # 转向时减速
            else:
                angular_vel = 0.0
            
            # 发布速度命令
            vel_cmd = Twist()
            vel_cmd.linear.x = linear_vel
            vel_cmd.angular.z = angular_vel
            self.vel_pub.publish(vel_cmd)
            
            # 定期打印状态（每秒一次）
            if not hasattr(self, '_last_log_time'):
                self._last_log_time = self.get_clock().now()
                self._log_counter = 0
            
            current_time = self.get_clock().now()
            if (current_time - self._last_log_time).nanoseconds > 1e9:  # 1秒
                self.get_logger().info(
                    f'🚗 Tracking: dist={distance:.2f}m, linear_vel={linear_vel:.2f}m/s, angular_vel={angular_vel:.2f}rad/s'
                )
                self._last_log_time = current_time
            
        except TransformException as e:
            # 只在第一次失败时打印错误
            if not hasattr(self, '_tf_error_logged'):
                self.get_logger().error(f'❌ TF lookup failed: {e}')
                self.get_logger().error(f'   Looking for transform: {self.map_frame} -> {self.base_frame}')
                self._tf_error_logged = True
    
    @staticmethod
    def quaternion_to_yaw(q):
        """四元数转偏航角"""
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    
    @staticmethod
    def normalize_angle(angle):
        """角度归一化到[-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    tracker = SimplePathTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
