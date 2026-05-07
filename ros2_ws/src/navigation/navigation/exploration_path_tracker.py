#!/usr/bin/env python3
"""
Advanced path tracker (Level 2):
- Local costmap built from LaserScan
- Inflation-based obstacle cost model
- Multi-candidate trajectory scoring and selection
"""

import math
import time
import heapq
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class ExplorationPathTracker(Node):
    STATE_IDLE = 0
    STATE_TRACKING = 1

    def __init__(self):
        super().__init__('exploration_path_tracker')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('scan_topic', '/jetson/scan')
        self.declare_parameter('scan_topic_candidates', ['/scan', 'scan'])

        self.declare_parameter('max_linear_vel', 0.30)
        self.declare_parameter('max_angular_vel', 0.80)
        self.declare_parameter('goal_tolerance', 0.20)
        self.declare_parameter('emergency_stop_dist', 0.15)  # 降低急停距离
        self.declare_parameter('collision_dist', 0.08)  # 碰撞检测距离（极近）

        self.declare_parameter('costmap_size', 2.0)  # 扩大到2m
        self.declare_parameter('costmap_resolution', 0.05)  # 提高分辨率到5cm
        self.declare_parameter('inflation_radius', 0.30)  # 增大膨胀半径

        self.declare_parameter('horizon_sec', 1.2)  # 增加预测时长
        self.declare_parameter('sim_dt', 0.1)
        self.declare_parameter('num_candidates', 11)  # 候选轨迹数量

        self.declare_parameter('w_obstacle', 15.0)  # 增加障碍物权重
        self.declare_parameter('w_goal', 5.0)  # 增加目标吸引力
        self.declare_parameter('w_heading', 2.0)  # 增加朝向权重
        self.declare_parameter('w_smooth', 1.5)  # 增加平滑性

        # 卡住检测参数
        self.declare_parameter('stuck_timeout', 3.0)  # 卡住检测时间阈值（秒）
        self.declare_parameter('stuck_distance', 0.15)  # 卡住检测距离阈值（米）
        self.declare_parameter('stuck_rotation_speed', 0.6)  # 脱困旋转速度
        self.declare_parameter('enable_reverse', True)  # 是否允许后退

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.scan_topic_candidates = list(self.get_parameter('scan_topic_candidates').value)

        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.emergency_stop_dist = float(self.get_parameter('emergency_stop_dist').value)
        self.collision_dist = float(self.get_parameter('collision_dist').value)

        self.costmap_size = float(self.get_parameter('costmap_size').value)
        self.costmap_resolution = float(self.get_parameter('costmap_resolution').value)
        self.inflation_radius = float(self.get_parameter('inflation_radius').value)

        self.horizon_sec = float(self.get_parameter('horizon_sec').value)
        self.sim_dt = float(self.get_parameter('sim_dt').value)
        self.num_candidates = int(self.get_parameter('num_candidates').value)

        self.w_obstacle = float(self.get_parameter('w_obstacle').value)
        self.w_goal = float(self.get_parameter('w_goal').value)
        self.w_heading = float(self.get_parameter('w_heading').value)
        self.w_smooth = float(self.get_parameter('w_smooth').value)

        self.stuck_timeout = float(self.get_parameter('stuck_timeout').value)
        self.stuck_distance = float(self.get_parameter('stuck_distance').value)
        self.stuck_rotation_speed = float(self.get_parameter('stuck_rotation_speed').value)
        self.enable_reverse = bool(self.get_parameter('enable_reverse').value)

        self.grid_size = max(7, int(round(self.costmap_size / self.costmap_resolution)))
        if self.grid_size % 2 == 0:
            self.grid_size += 1
        self.grid_center = self.grid_size // 2

        self.costmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.scan_ready = False
        self.front_min_distance = float('inf')
        self.collision_detected = False  # 碰撞检测标志
        self.last_scan_time = 0.0
        self.last_scan_topic = 'N/A'
        self.log_throttle_dict = {}

        self.current_goal = None
        self.goal_start_time = None
        self.state = self.STATE_IDLE
        self.reached_goal = False
        self.last_cmd = (0.0, 0.0)

        # A* fields
        self.global_map = None
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 0.30
        self.astar_replan_interval = 3.0
        self.last_replan_time = 0.0

        # 卡住检测
        self.position_history = []  # [(x, y, timestamp), ...]
        self.stuck_start_time = None
        self.reverse_start_time = None  # 后退开始时间
        self.recovery_rotation_dir = 1  # 1=左转，-1=右转
        self.last_goal_distance = float('inf')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = qos_profile_sensor_data
        self.scan_subs = []
        all_scan_topics = [self.scan_topic] + self.scan_topic_candidates
        # Keep order while removing duplicates and filter out invalid topic names.
        unique_scan_topics = []
        seen = set()
        for topic in all_scan_topics:
            # Filter out empty, None, or topics with repeated slashes
            if topic and topic not in seen and '//' not in topic:
                unique_scan_topics.append(topic)
                seen.add(topic)
        for topic in unique_scan_topics:
            self.scan_subs.append(
                self.create_subscription(
                    LaserScan,
                    topic,
                    lambda msg, topic_name=topic: self.scan_callback(msg, topic_name),
                    sensor_qos,
                )
            )
        self.get_logger().info(f'Subscribed to scan topics: {unique_scan_topics}')

        self.goal_sub1 = self.create_subscription(PoseStamped, 'move_base_simple/goal', self.goal_callback, 10)
        self.goal_sub2 = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.global_map_callback, 10)
        self.path_pub = self.create_publisher(Path, '/exploration/global_path', 10)

        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # 反馈发布
        self.goal_reached_pub = self.create_publisher(Bool, 'goal_reached', 10)
        self.goal_failed_pub = self.create_publisher(Bool, 'goal_failed', 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Exploration Path Tracker started (Level 2 Enhanced)')
        self.get_logger().info(f'  scan_topics: {unique_scan_topics}')
        self.get_logger().info(f'  local_costmap: {self.grid_size}x{self.grid_size}, res={self.costmap_resolution:.3f}m')
        self.get_logger().info(f'  inflation_radius: {self.inflation_radius:.2f}m')
        self.get_logger().info(f'  emergency_stop: {self.emergency_stop_dist:.2f}m, collision: {self.collision_dist:.2f}m')
        self.get_logger().info(f'  num_candidates: {self.num_candidates}, enable_reverse: {self.enable_reverse}')
        self.get_logger().info(f'  stuck_detection: timeout={self.stuck_timeout:.1f}s, dist={self.stuck_distance:.2f}m')

    def global_map_callback(self, msg: OccupancyGrid) -> None:
        self.global_map = msg

    def plan_astar(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> List[Tuple[float, float]]:
        if not self.global_map:
            return []
        
        info = self.global_map.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        w = info.width
        h = info.height
        grid = np.array(self.global_map.data).reshape((h, w))

        def world_to_grid(wx, wy):
            return int((wx - ox) / res), int((wy - oy) / res)
            
        def grid_to_world(gx, gy):
            return ox + gx * res, oy + gy * res

        sx, sy = world_to_grid(start_x, start_y)
        gx, gy = world_to_grid(goal_x, goal_y)
        
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return []

        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        came_from = {}
        g_score = {(sx, sy): 0.0}

        def is_free(nx, ny):
            if not(0 <= nx < w and 0 <= ny < h): return False
            if grid[ny, nx] == -1: return True
            if grid[ny, nx] >= 50: return False
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        cx, cy = nx + dx, ny + dy
                        if 0 <= cx < w and 0 <= cy < h:
                            if grid[cy, cx] >= 50:
                                return False
            return True

        closed_set = set()
        
        while open_set:
            _, curr_x, curr_y = heapq.heappop(open_set)
            
            if curr_x == gx and curr_y == gy:
                path = []
                while (curr_x, curr_y) in came_from:
                    path.append(grid_to_world(curr_x, curr_y))
                    curr_x, curr_y = came_from[(curr_x, curr_y)]
                path.append(grid_to_world(sx, sy))
                path.reverse()
                return self.simplify_path(path)
                
            closed_set.add((curr_x, curr_y))
            
            for dx, dy, cost in [(0,1,1), (1,0,1), (0,-1,1), (-1,0,1), (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]:
                nx, ny = curr_x + dx, curr_y + dy
                if (nx, ny) in closed_set: continue
                
                if not is_free(nx, ny):
                    if not (nx == gx and ny == gy):
                        continue
                
                tentative_g_score = g_score[(curr_x, curr_y)] + cost
                
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (curr_x, curr_y)
                    g_score[(nx, ny)] = tentative_g_score
                    f = tentative_g_score + math.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_set, (f, nx, ny))
                    
            if len(closed_set) > 20000:
                self.get_logger().warn('A* search exceeded 20000 nodes, aborting')
                break
                
        return []

    def simplify_path(self, raw_path: List[Tuple[float, float]], interval=0.3) -> List[Tuple[float, float]]:
        if not raw_path: return []
        simplified = [raw_path[0]]
        for pt in raw_path[1:-1]:
            last_pt = simplified[-1]
            if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) >= interval:
                simplified.append(pt)
        simplified.append(raw_path[-1])
        return simplified

    def publish_global_path(self, waypoints: List[Tuple[float, float]]):
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in waypoints:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = 1.0
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)

    def goal_callback(self, msg: PoseStamped) -> None:
        self.current_goal = msg
        self.goal_start_time = time.time()
        self.state = self.STATE_TRACKING
        self.reached_goal = False
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(f'New goal: ({goal_x:.2f}, {goal_y:.2f}) frame={msg.header.frame_id}')

        if self.global_map is not None:
            try:
                tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
                rx = tf.transform.translation.x
                ry = tf.transform.translation.y
                waypoints = self.plan_astar(rx, ry, goal_x, goal_y)
                if waypoints and len(waypoints) >= 2:
                    self.waypoints = waypoints
                    self.current_waypoint_idx = 0
                    self.publish_global_path(waypoints)
                    self.get_logger().info(f'A* path: {len(waypoints)} waypoints')
                else:
                    self.waypoints = [(goal_x, goal_y)]
                    self.current_waypoint_idx = 0
            except Exception as e:
                self.get_logger().warn(f'Could not get TF for A*: {e}')
                self.waypoints = [(goal_x, goal_y)]
                self.current_waypoint_idx = 0
        else:
            self.waypoints = [(goal_x, goal_y)]
            self.current_waypoint_idx = 0

    def scan_callback(self, msg: LaserScan, topic_name: str) -> None:
        self.costmap.fill(0.0)
        self.front_min_distance = float('inf')
        self.collision_detected = False  # 重置碰撞标志
        self.last_scan_time = time.time()
        self.last_scan_topic = topic_name

        half_size = self.costmap_size * 0.5
        min_distance_all = float('inf')  # 全方向最近距离

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                continue

            # 更新全局最近距离
            if r < min_distance_all:
                min_distance_all = r

            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)

            # 前方30度范围
            if -15.0 <= angle_deg <= 15.0 and r < self.front_min_distance:
                self.front_min_distance = r

            x = r * math.cos(angle)
            y = r * math.sin(angle)
            if abs(x) > half_size or abs(y) > half_size:
                continue

            gx, gy = self.world_to_grid(x, y)
            if gx is None:
                continue

            self.costmap[gy, gx] = 255.0

        # 碰撞检测：任何方向有极近距离点
        if min_distance_all < self.collision_dist:
            self.collision_detected = True

        self.inflate_costmap()
        self.scan_ready = True

    def world_to_grid(self, x: float, y: float):
        gx = int(round(x / self.costmap_resolution)) + self.grid_center
        gy = int(round(y / self.costmap_resolution)) + self.grid_center
        if gx < 0 or gy < 0 or gx >= self.grid_size or gy >= self.grid_size:
            return None, None
        return gx, gy

    def inflate_costmap(self) -> None:
        obstacle_cells = np.argwhere(self.costmap >= 255.0)
        if obstacle_cells.size == 0:
            return

        inflate_cells = int(math.ceil(self.inflation_radius / self.costmap_resolution))

        for oy, ox in obstacle_cells:
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    nx = ox + dx
                    ny = oy + dy
                    if nx < 0 or ny < 0 or nx >= self.grid_size or ny >= self.grid_size:
                        continue

                    dist = math.hypot(dx * self.costmap_resolution, dy * self.costmap_resolution)
                    if dist > self.inflation_radius:
                        continue

                    if dist < 1e-6:
                        cost = 255.0
                    else:
                        # 增强型分层：距离越近代价越高（指数衰减）
                        ratio = dist / self.inflation_radius
                        # 使用平方衰减使近距离代价更高
                        cost = 255.0 * (1.0 - ratio * ratio)

                    if cost > self.costmap[ny, nx]:
                        self.costmap[ny, nx] = cost

    def generate_candidates(self, v_limit: float) -> List[Tuple[float, float]]:
        vmax = max(0.05, min(v_limit, self.max_linear_vel))
        candidates = [
            # 直行和大角度转弯
            (vmax, 0.0),
            (max(0.20, vmax * 0.8), 0.40),
            (max(0.20, vmax * 0.8), -0.40),
            (max(0.15, vmax * 0.65), 0.60),
            (max(0.15, vmax * 0.65), -0.60),
            (0.12, 0.80),
            (0.12, -0.80),
            # 原地转
            (0.0, 0.70),
            (0.0, -0.70),
            # 中速转弯
            (max(0.18, vmax * 0.70), 0.50),
            (max(0.18, vmax * 0.70), -0.50),
        ]
        
        # 如果允许后退，添加后退候选
        if self.enable_reverse:
            candidates.extend([
                (-0.10, 0.0),    # 直线后退
                (-0.08, 0.50),   # 后退左转
                (-0.08, -0.50),  # 后退右转
            ])
        
        # 根据num_candidates参数返回前N个
        return candidates[:self.num_candidates]

    def predict_trajectory(self, v: float, w: float) -> Tuple[List[Tuple[float, float]], float]:
        points: List[Tuple[float, float]] = []
        x = 0.0
        y = 0.0
        theta = 0.0
        steps = max(1, int(self.horizon_sec / self.sim_dt))

        for _ in range(steps):
            x += v * math.cos(theta) * self.sim_dt
            y += v * math.sin(theta) * self.sim_dt
            theta += w * self.sim_dt
            points.append((x, y))

        return points, theta

    def evaluate_trajectory(
        self,
        v: float,
        w: float,
        trajectory: List[Tuple[float, float]],
        end_theta: float,
        goal_local_x: float,
        goal_local_y: float,
    ) -> float:
        obstacle_sum = 0.0
        collision_penalty = 0.0

        for x, y in trajectory:
            gx, gy = self.world_to_grid(x, y)
            if gx is None:
                collision_penalty += 1000.0
                continue

            c = float(self.costmap[gy, gx])
            obstacle_sum += c
            if c >= 220.0:
                collision_penalty += 2000.0

        obstacle_cost = obstacle_sum / max(1, len(trajectory))

        end_x, end_y = trajectory[-1]
        goal_dist = math.hypot(goal_local_x - end_x, goal_local_y - end_y)

        target_heading = math.atan2(goal_local_y - end_y, goal_local_x - end_x)
        heading_err = abs(self.normalize_angle(target_heading - end_theta))

        prev_v, prev_w = self.last_cmd
        smooth_cost = abs(v - prev_v) * 2.0 + abs(w - prev_w)

        total = (
            self.w_obstacle * obstacle_cost
            + self.w_goal * goal_dist
            + self.w_heading * heading_err
            + self.w_smooth * smooth_cost
            + collision_penalty
        )
        return total

    def control_loop(self) -> None:
        if self.current_goal is None:
            self.state = self.STATE_IDLE
            self.publish_stop()
            return
            
        # 目标超时检查 (30秒)
        if self.goal_start_time and time.time() - self.goal_start_time > 30.0:
            self.publish_stop()
            self._log_once_per_sec('TIMEOUT', 'goal timeout, giving up')
            self.goal_failed_pub.publish(Bool(data=True))
            self.current_goal = None
            self.waypoints = []
            self.state = self.STATE_IDLE
            return

        if not self.scan_ready:
            self.publish_stop()
            self._log_once_per_sec('WAIT_SCAN', 'waiting for LaserScan data...')
            return

        if time.time() - self.last_scan_time > 1.5:
            self.publish_stop()
            self._log_once_per_sec(
                'SCAN_TIMEOUT',
                f'last scan too old, source={self.last_scan_topic}, age={time.time() - self.last_scan_time:.1f}s',
            )
            return

        # 碰撞检测：立即停止
        if self.collision_detected:
            self.publish_stop()
            self._log_once_per_sec('COLLISION', f'collision detected! min_dist < {self.collision_dist:.3f}m')
            return

        if self.front_min_distance < self.emergency_stop_dist:
            self.publish_stop()
            self._log_once_per_sec('EMERGENCY', f'front obstacle {self.front_min_distance:.3f}m')
            return

        try:
            # Use current time instead of rclpy.time.Time() (0) to get the latest exact transform
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except TransformException as exc:
            self.publish_stop()
            self._log_once_per_sec('TF', f'lookup failed: {exc}')
            return

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        q = transform.transform.rotation
        robot_yaw = self.quaternion_to_yaw(q)

        # Handle TF Origin Jumps (Jump > 0.5m)
        if hasattr(self, 'last_known_pos'):
            jump_dist = math.hypot(robot_x - self.last_known_pos[0], robot_y - self.last_known_pos[1])
            if jump_dist > 0.5:
                self.get_logger().warn(f'TF Jump detected: {jump_dist:.2f}m. Reinitializing state.')
                self.stuck_start_time = None
                self.last_goal_distance = float('inf')
                self.position_history.clear()
                # Optionally, you could also clear waypoints to force an immediate A* replan:
                self.waypoints = []
                self.last_known_pos = (robot_x, robot_y)
                return
        self.last_known_pos = (robot_x, robot_y)

        # 检查路点状态
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            self.publish_stop()
            return
            
        wp_x, wp_y = self.waypoints[self.current_waypoint_idx]
        wp_dist = math.hypot(wp_x - robot_x, wp_y - robot_y)
        
        is_last_wp = (self.current_waypoint_idx == len(self.waypoints) - 1)
        tol = self.goal_tolerance if is_last_wp else self.waypoint_tolerance
        
        if wp_dist < tol:
            if is_last_wp:
                if not self.reached_goal:
                    self.get_logger().info('Final goal reached!')
                    self.reached_goal = True
                    self.goal_reached_pub.publish(Bool(data=True))
                self.state = self.STATE_IDLE
                self.publish_stop()
                self.current_goal = None
                self.waypoints = []
                return
            else:
                self.current_waypoint_idx += 1
                wp_x, wp_y = self.waypoints[self.current_waypoint_idx]

        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        distance = math.hypot(goal_x - robot_x, goal_y - robot_y)
        
        dx = wp_x - robot_x
        dy = wp_y - robot_y

        # 卡住检测
        is_stuck = self.check_if_stuck(robot_x, robot_y)
        
        # 检测目标接近进度
        if distance < self.last_goal_distance - 0.05:
            # 正在接近目标，重置卡住计时器
            self.stuck_start_time = None
            self.last_goal_distance = distance
        else:
            # 距离没有明显缩小
            if self.stuck_start_time is None:
                self.stuck_start_time = time.time()
        
        if is_stuck and self.stuck_start_time:
            stuck_dur = time.time() - self.stuck_start_time
            
            # 策略3：彻底放弃
            if stuck_dur > 7.0:
                self.publish_stop()
                self.get_logger().warn(f'STUCK_FATAL: Stuck for >7s, abandoning goal!')
                self.goal_failed_pub.publish(Bool(data=True))
                self.current_goal = None
                self.goal_start_time = None
                self.state = self.STATE_IDLE
                self.stuck_start_time = None
                self.last_goal_distance = float('inf')
                self.position_history.clear()
                self.waypoints = []
                return
                
            # 策略2：尝试A*重规划（快速触发）
            if stuck_dur > 2.5 and (time.time() - self.last_replan_time > self.astar_replan_interval):
                self.last_replan_time = time.time()
                self.publish_stop()
                if self.global_map:
                    self.get_logger().info('Stuck >2.5s: Replanning A*...')
                    new_wps = self.plan_astar(robot_x, robot_y, goal_x, goal_y)
                    if new_wps and len(new_wps) >= 2:
                        self.waypoints = new_wps
                        self.current_waypoint_idx = 0
                        self.publish_global_path(new_wps)
                        self.stuck_start_time = time.time() # Reset to give it an attempt
                        self.get_logger().info('Replan successful, resuming.')
                        return
                    else:
                        self.get_logger().warn('Replan failed, waiting for timeout.')
                        # A*重规划失败，通知explorer选择新目标
                        self.publish_goal_failed()
            
            # 策略1：开环快速后退脱困（简化逻辑）
            if stuck_dur > 1.2:
                # 直接后退0.15米，不旋转
                # 计算需要的时间：0.15m / 0.15m/s = 1秒
                reverse_duration = 1.0
                if not hasattr(self, 'reverse_start_time') or self.reverse_start_time is None:
                    self.reverse_start_time = time.time()
                
                reverse_elapsed = time.time() - self.reverse_start_time
                if reverse_elapsed < reverse_duration:
                    cmd = (-0.15, 0.0)
                    self._log_once_per_sec('STUCK_REVERSE', f'open-loop reversing 0.15m ({reverse_elapsed:.1f}s)')
                else:
                    # 后退完成，重置stuck状态让系统重新规划
                    self.stuck_start_time = time.time()
                    self.reverse_start_time = None
                    self.publish_stop()
                    self.publish_goal_failed()  # 通知explorer放弃当前区域
                    self.get_logger().info('Reverse complete, requesting new frontier')
                    return
                
                self.publish_cmd(*cmd)
                self.last_cmd = cmd
                return

        # Transform goal vector to robot local frame.
        goal_local_x = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dy
        goal_local_y = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dy

        # Soft speed cap when near frontal obstacles.
        if self.front_min_distance < 0.35:
            v_limit = 0.10
        elif self.front_min_distance < 0.60:
            ratio = (self.front_min_distance - 0.35) / 0.25
            v_limit = 0.10 + max(0.0, min(1.0, ratio)) * (self.max_linear_vel - 0.10)
        else:
            v_limit = self.max_linear_vel

        candidates = self.generate_candidates(v_limit)

        best_cmd = None
        best_cost = float('inf')

        for v, w in candidates:
            trajectory, end_theta = self.predict_trajectory(v, w)
            cost = self.evaluate_trajectory(v, w, trajectory, end_theta, goal_local_x, goal_local_y)
            if cost < best_cost:
                best_cost = cost
                best_cmd = (v, w)

        if best_cmd is None or best_cost > 9000.0:
            # Recovery fallback: rotate toward freer side.
            left_cost = self.sample_side_cost(left=True)
            right_cost = self.sample_side_cost(left=False)
            if left_cost <= right_cost:
                cmd = (0.0, min(0.6, self.max_angular_vel))
            else:
                cmd = (0.0, -min(0.6, self.max_angular_vel))
            self.publish_cmd(*cmd)
            self.last_cmd = cmd
            self._log_once_per_sec('RECOVERY', f'fallback rotate, left_cost={left_cost:.1f}, right_cost={right_cost:.1f}')
            return

        self.publish_cmd(*best_cmd)
        self.last_cmd = best_cmd
        self.state = self.STATE_TRACKING
        
        # 增强日志信息
        stuck_info = " [STUCK]" if is_stuck else ""
        collision_info = " [COLL_RISK]" if self.collision_detected else ""
        self._log_once_per_sec(
            'TRACK',
            f'd={distance:.2f}m front={self.front_min_distance:.2f}m cmd=({best_cmd[0]:.2f},{best_cmd[1]:.2f}) cost={best_cost:.1f}{stuck_info}{collision_info}',
        )

    def sample_side_cost(self, left: bool) -> float:
        mid = self.grid_center
        if left:
            region = self.costmap[mid - 3:mid + 4, mid + 1:mid + 5]
        else:
            region = self.costmap[mid - 3:mid + 4, mid - 4:mid]
        if region.size == 0:
            return 999.0
        return float(np.mean(region))

    def publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.vel_pub.publish(msg)

    def publish_stop(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def publish_goal_failed(self) -> None:
        """Notify explorer that current goal is unreachable"""
        self.goal_failed_pub.publish(Bool(data=True))

    def _log_once_per_sec(self, tag: str, text: str) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = 0.0
        if now - self._last_log_time >= 1.0:
            self.get_logger().info(f'[{tag}] {text}')
            self._last_log_time = now

    def check_if_stuck(self, x: float, y: float) -> bool:
        """检测机器人是否卡住（在小范围内停留太久）"""
        now = time.time()
        
        # 更新位置历史
        self.position_history.append((x, y, now))
        
        # 保留最近5秒的历史
        self.position_history = [(px, py, pt) for px, py, pt in self.position_history if now - pt < 5.0]
        
        # 至少需要3秒历史数据
        if len(self.position_history) < 2:
            return False
        
        oldest_time = self.position_history[0][2]
        if now - oldest_time < self.stuck_timeout:
            return False
        
        # 计算在stuck_timeout时间内的最大移动距离
        max_distance = 0.0
        for px, py, _ in self.position_history:
            dist = math.hypot(x - px, y - py)
            if dist > max_distance:
                max_distance = dist
        
        # 如果移动距离小于阈值，认为卡住
        return max_distance < self.stuck_distance

    @staticmethod
    def quaternion_to_yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationPathTracker()
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
