#!/usr/bin/env python3
"""
智能路径跟踪控制器 - 带动态避障功能
订阅Explore Lite的目标点 + 雷达数据，实现障碍物躲避和脱困
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, TransformException, Buffer
import math
import time
import random

class SmartPathTracker(Node):
    # 状态常量
    STATE_IDLE = 0
    STATE_TRACKING = 1
    STATE_AVOIDING = 2
    STATE_STUCK = 3
    
    def __init__(self):
        super().__init__('smart_path_tracker')
        
        # 声明参数
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('max_linear_vel', 0.3)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('goal_tolerance', 0.2)
        # 避障参数
        self.declare_parameter('safe_dist', 0.6)  # 触发避障的距离
        self.declare_parameter('clear_dist', 0.8) # 解除避障的距离
        self.declare_parameter('emergency_stop_dist', 0.2)  # 紧急停止距离
        self.declare_parameter('scan_topic', '/jetson/scan')
        
        # 获取参数
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.safe_dist = self.get_parameter('safe_dist').value
        self.clear_dist = self.get_parameter('clear_dist').value
        self.emergency_stop_dist = self.get_parameter('emergency_stop_dist').value
        self.scan_topic = self.get_parameter('scan_topic').value
        
        # TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 订阅雷达数据
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )
        
        # 订阅目标点
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
        
        # 状态变量
        self.current_goal = None
        self.state = self.STATE_IDLE
        self.reached_goal = False
        
        # 雷达扇区数据 (12个扇区，每个30°，提升精度)
        self.sectors = {
            'front_center': float('inf'),      # -15° to 15° (核心前方)
            'front_left_close': float('inf'),  # 15° to 45°
            'front_left': float('inf'),        # 45° to 75°
            'left_front': float('inf'),        # 75° to 105°
            'left': float('inf'),              # 105° to 135°
            'left_rear': float('inf'),         # 135° to 165°
            'rear': float('inf'),              # 165° to 180° & -180° to -165°
            'right_rear': float('inf'),        # -165° to -135°
            'right': float('inf'),             # -135° to -105°
            'right_front': float('inf'),       # -105° to -75°
            'front_right': float('inf'),       # -75° to -45°
            'front_right_close': float('inf')  # -45° to -15°
        }
        
        # 脱困计数器
        self.stuck_attempts = 0
        self.max_stuck_attempts = 3
        self.stuck_start_time = None
        self.stuck_phase = 'backup'  # 'backup' or 'rotate'
        self.stuck_rotate_direction = 1  # 1=左转, -1=右转（随机选择）
        
        # 避障转向方向 ('left' or 'right')
        self.avoid_direction = None
        
        # 控制循环 (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('🤖 Smart Path Tracker started (Level 1 Enhanced)')
        self.get_logger().info(f'  map_frame: {self.map_frame}')
        self.get_logger().info(f'  base_frame: {self.base_frame}')
        self.get_logger().info(f'  scan_topic: {self.scan_topic}')
        self.get_logger().info(f'  Sectors: 12 (30° resolution, improved from 6)')
        self.get_logger().info(f'  safe_dist: {self.safe_dist}m')
        self.get_logger().info(f'  clear_dist: {self.clear_dist}m')
        self.get_logger().info(f'  emergency_stop_dist: {self.emergency_stop_dist}m')
        self.get_logger().info(f'  Features: Dynamic speed, smart unstuck, progressive deceleration')
    
    def goal_callback(self, msg):
        """接收新的目标点"""
        self.current_goal = msg
        self.reached_goal = False
        self.state = self.STATE_TRACKING
        self.stuck_attempts = 0  # 重置脱困计数
        self.get_logger().info(f'🎯 New goal received!')
        self.get_logger().info(f'   Position: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self.get_logger().info(f'   State: TRACKING')
    
    def scan_callback(self, msg):
        """处理雷达数据，计算各扇区最小距离"""
        # 雷达数据分析：360度分成12个扇区（每个30°）
        num_readings = len(msg.ranges)
        
        # 初始化扇区数据
        sector_data = {
            'front_center': [],      # -15° to 15°
            'front_left_close': [],  # 15° to 45°
            'front_left': [],        # 45° to 75°
            'left_front': [],        # 75° to 105°
            'left': [],              # 105° to 135°
            'left_rear': [],         # 135° to 165°
            'rear': [],              # 165° to 180° & -180° to -165°
            'right_rear': [],        # -165° to -135°
            'right': [],             # -135° to -105°
            'right_front': [],       # -105° to -75°
            'front_right': [],       # -75° to -45°
            'front_right_close': []  # -45° to -15°
        }
        
        for i, r in enumerate(msg.ranges):
            # 跳过无效数据
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue
            
            # 计算当前角度
            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)
            
            # 分配到12个扇区（每个30°）
            if -15 <= angle_deg <= 15:
                sector_data['front_center'].append(r)
            elif 15 < angle_deg <= 45:
                sector_data['front_left_close'].append(r)
            elif 45 < angle_deg <= 75:
                sector_data['front_left'].append(r)
            elif 75 < angle_deg <= 105:
                sector_data['left_front'].append(r)
            elif 105 < angle_deg <= 135:
                sector_data['left'].append(r)
            elif 135 < angle_deg <= 165:
                sector_data['left_rear'].append(r)
            elif angle_deg > 165 or angle_deg < -165:
                sector_data['rear'].append(r)
            elif -165 <= angle_deg < -135:
                sector_data['right_rear'].append(r)
            elif -135 <= angle_deg < -105:
                sector_data['right'].append(r)
            elif -105 <= angle_deg < -75:
                sector_data['right_front'].append(r)
            elif -75 <= angle_deg < -45:
                sector_data['front_right'].append(r)
            elif -45 <= angle_deg < -15:
                sector_data['front_right_close'].append(r)
        
        # 计算每个扇区的最小距离
        for sector_name, readings in sector_data.items():
            if readings:
                self.sectors[sector_name] = min(readings)
            else:
                self.sectors[sector_name] = float('inf')
    
    def control_loop(self):
        """主控制循环"""
        # 紧急停止检查：检测所有扇区是否有极近障碍物
        min_distance = min(self.sectors.values())
        if min_distance < self.emergency_stop_dist:
            # 立即停止！
            vel_cmd = Twist()
            self.vel_pub.publish(vel_cmd)
            if not hasattr(self, '_emergency_logged') or not self._emergency_logged:
                self.get_logger().error(f'🚨 EMERGENCY STOP! Obstacle at {min_distance:.3f}m (threshold: {self.emergency_stop_dist}m)')
                self._emergency_logged = True
            return
        else:
            # 重置紧急停止标志
            if hasattr(self, '_emergency_logged'):
                if self._emergency_logged:
                    self.get_logger().info('✅ Emergency cleared, resuming operation')
                self._emergency_logged = False
        
        if self.current_goal is None:
            # 空闲状态
            if self.state != self.STATE_IDLE:
                self.state = self.STATE_IDLE
            return
        
        # 根据状态执行不同逻辑
        if self.state == self.STATE_IDLE:
            self.handle_idle()
        elif self.state == self.STATE_TRACKING:
            self.handle_tracking()
        elif self.state == self.STATE_AVOIDING:
            self.handle_avoiding()
        elif self.state == self.STATE_STUCK:
            self.handle_stuck()
    
    def handle_idle(self):
        """空闲状态：停止运动"""
        vel_cmd = Twist()
        self.vel_pub.publish(vel_cmd)
    
    def handle_tracking(self):
        """跟踪状态：朝目标前进，检测障碍物"""
        try:
            # 获取当前机器人位置
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            q = transform.transform.rotation
            robot_yaw = self.quaternion_to_yaw(q)
            
            # 计算与目标的距离
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
                    self.state = self.STATE_IDLE
                vel_cmd = Twist()
                self.vel_pub.publish(vel_cmd)
                return
            
            # 【改进1】更精细的前方障碍检测（考虑3个前方扇区）
            front_zones = [
                self.sectors['front_center'],
                self.sectors['front_left_close'],
                self.sectors['front_right_close']
            ]
            min_front_dist = min(front_zones)
            
            # 触发避障判断（前方3个扇区中任一小于安全距离）
            if min_front_dist < self.safe_dist:
                self.get_logger().warn(f'⚠️ Obstacle detected! Min front: {min_front_dist:.2f}m')
                self.state = self.STATE_AVOIDING
                # 【改进2】更智能的避障方向选择（综合4个扇区）
                left_space = min(
                    self.sectors['front_left_close'],
                    self.sectors['front_left'],
                    self.sectors['left_front']
                )
                right_space = min(
                    self.sectors['front_right_close'],
                    self.sectors['front_right'],
                    self.sectors['right_front']
                )
                self.avoid_direction = 'left' if left_space > right_space else 'right'
                self.get_logger().info(f'🔄 Avoiding to {self.avoid_direction} (L:{left_space:.2f}m R:{right_space:.2f}m)')
                return
            
            # 纯追踪算法
            goal_yaw = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(goal_yaw - robot_yaw)
            
            # 【改进3】动态速度调节（根据前方障碍距离）
            base_linear_vel = self.max_linear_vel * min(1.0, distance / 1.0)
            
            # 根据前方最小距离动态减速
            if min_front_dist < 0.3:
                # 极近障碍：完全停止
                linear_vel = 0.0
            elif min_front_dist < self.safe_dist:
                # 渐进减速区间 [0.3, 0.6] -> [0.05, base_vel]
                speed_factor = (min_front_dist - 0.3) / (self.safe_dist - 0.3)
                linear_vel = 0.05 + (base_linear_vel - 0.05) * speed_factor
            else:
                # 安全距离：正常速度
                linear_vel = base_linear_vel
            
            # 确保最小速度
            if linear_vel > 0 and linear_vel < 0.05:
                linear_vel = 0.05
            
            # 角速度计算
            if abs(yaw_error) > 0.1:
                angular_vel = self.max_angular_vel * min(1.0, abs(yaw_error) / 1.57)
                if yaw_error < 0:
                    angular_vel = -angular_vel
                # 转向时减速
                linear_vel *= 0.5
            else:
                angular_vel = 0.0
            
            # 发布速度
            vel_cmd = Twist()
            vel_cmd.linear.x = linear_vel
            vel_cmd.angular.z = angular_vel
            self.vel_pub.publish(vel_cmd)
            
            # 定期打印（包含前方障碍距离）
            self._log_state('TRACKING', f'dist={distance:.2f}m, front_obs={min_front_dist:.2f}m, v={linear_vel:.2f}m/s, ω={angular_vel:.2f}rad/s')
            
        except TransformException as e:
            if not hasattr(self, '_tf_error_logged'):
                self.get_logger().error(f'❌ TF lookup failed: {e}')
                self._tf_error_logged = True
    
    def handle_avoiding(self):
        """避障状态：原地转向，寻找通路"""
        # 检查前方是否已清空（3个前方扇区）
        front_zones = [
            self.sectors['front_center'],
            self.sectors['front_left_close'],
            self.sectors['front_right_close']
        ]
        min_front_dist = min(front_zones)
        
        if min_front_dist > self.clear_dist:
            self.get_logger().info(f'✅ Path clear! Front: {min_front_dist:.2f}m, resuming tracking')
            self.state = self.STATE_TRACKING
            return
        
        # 【改进4】更精确的被困检测（检查更多扇区）
        critical_sectors = [
            self.sectors['front_center'],
            self.sectors['front_left_close'],
            self.sectors['front_right_close'],
            self.sectors['front_left'],
            self.sectors['front_right'],
            self.sectors['left_front'],
            self.sectors['right_front']
        ]
        
        all_blocked = all(d < self.safe_dist for d in critical_sectors)
        
        if all_blocked:
            self.get_logger().warn('🚫 Stuck detected! All forward directions blocked')
            self.state = self.STATE_STUCK
            self.stuck_start_time = time.time()
            self.stuck_phase = 'backup'
            # 【改进5】随机选择脱困旋转方向
            self.stuck_rotate_direction = random.choice([-1, 1])
            return
        
        # 原地转向避障（转速根据空间大小调整）
        if self.avoid_direction == 'left':
            # 左转时检查左侧空间
            left_clearance = min(self.sectors['front_left'], self.sectors['left_front'])
            turn_speed = 0.5 if left_clearance > 1.0 else 0.3  # 空间大时快速转向
            angular_vel = self.max_angular_vel * turn_speed
        else:
            # 右转
            right_clearance = min(self.sectors['front_right'], self.sectors['right_front'])
            turn_speed = 0.5 if right_clearance > 1.0 else 0.3
            angular_vel = -self.max_angular_vel * turn_speed
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = angular_vel
        self.vel_pub.publish(vel_cmd)
        
        self._log_state('AVOIDING', f'direction={self.avoid_direction}, front={min_front_dist:.2f}m, ω={angular_vel:.2f}rad/s')
    
    def handle_stuck(self):
        """脱困状态：后退 + 旋转"""
        if self.stuck_attempts >= self.max_stuck_attempts:
            # 放弃当前目标
            self.get_logger().error(f'❌ Failed to unstuck after {self.max_stuck_attempts} attempts. Giving up goal.')
            self.current_goal = None
            self.state = self.STATE_IDLE
            self.stuck_attempts = 0
            return
        
        elapsed_time = time.time() - self.stuck_start_time
        
        vel_cmd = Twist()
        
        if self.stuck_phase == 'backup':
            # 【改进6】后退前检查后方安全
            rear_zones = [self.sectors['rear'], self.sectors['left_rear'], self.sectors['right_rear']]
            min_rear_dist = min(rear_zones)
            
            if min_rear_dist < 0.3:
                # 后方也被堵，直接跳到旋转
                self.get_logger().warn(f'⚠️ Rear blocked ({min_rear_dist:.2f}m), skip backup')
                self.stuck_phase = 'rotate'
                self.stuck_start_time = time.time()
                return
            
            # 后退阶段 (1.5秒或达到安全距离)
            if elapsed_time < 1.5:
                # 根据后方距离调整后退速度
                backup_speed = min(0.15, min_rear_dist * 0.3)
                vel_cmd.linear.x = -backup_speed
                self.vel_pub.publish(vel_cmd)
                self._log_state('STUCK', f'backing up... ({elapsed_time:.1f}s, rear:{min_rear_dist:.2f}m)')
            else:
                # 进入旋转阶段
                self.stuck_phase = 'rotate'
                self.stuck_start_time = time.time()
        
        elif self.stuck_phase == 'rotate':
            # 【改进7】智能旋转（根据左右空间选择角度）
            left_space = min(self.sectors['left_front'], self.sectors['left'])
            right_space = min(self.sectors['right_front'], self.sectors['right'])
            
            # 旋转时长根据空间情况调整（1.5-2.5秒）
            rotate_duration = 1.5 if max(left_space, right_space) > 1.5 else 2.5
            
            if elapsed_time < rotate_duration:
                # 使用初始化时随机选择的方向
                angular_vel = self.stuck_rotate_direction * self.max_angular_vel * 0.8
                vel_cmd.angular.z = angular_vel
                self.vel_pub.publish(vel_cmd)
                direction_name = 'left' if self.stuck_rotate_direction > 0 else 'right'
                self._log_state('STUCK', f'rotating {direction_name}... ({elapsed_time:.1f}s)')
            else:
                # 脱困尝试结束，重新尝试跟踪
                self.stuck_attempts += 1
                self.get_logger().info(f'🔄 Unstuck attempt {self.stuck_attempts}/{self.max_stuck_attempts} completed')
                self.state = self.STATE_TRACKING
                self.stuck_phase = 'backup'
                # 下次尝试换个方向
                self.stuck_rotate_direction *= -1
    
    def _log_state(self, state_name, info):
        """定期打印状态（每秒一次）"""
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = self.get_clock().now()
        
        current_time = self.get_clock().now()
        if (current_time - self._last_log_time).nanoseconds > 1e9:
            self.get_logger().info(f'🚗 [{state_name}] {info}')
            self._last_log_time = current_time
    
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
    tracker = SmartPathTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
