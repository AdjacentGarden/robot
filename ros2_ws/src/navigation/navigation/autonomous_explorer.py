#!/usr/bin/env python3
import math
import time
import numpy as np
from collections import deque
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException

class AutonomousExplorer(Node):
    STATE_INIT = 0
    STATE_SPINNING = 1
    STATE_EXPLORING = 2
    STATE_COMPLETE = 3
    
    def __init__(self):
        super().__init__('autonomous_explorer')
        
        # ����
        self.declare_parameter('exploration_radius', 5.0)
        self.declare_parameter('init_delay', 5.0)
        self.declare_parameter('frontier_check_interval', 1.0)
        
        self.exploration_radius = self.get_parameter('exploration_radius').value
        self.init_delay = self.get_parameter('init_delay').value
        self.frontier_interval = self.get_parameter('frontier_check_interval').value
        
        # ״̬
        self.state = self.STATE_INIT
        self.start_pos = None
        self.init_start_time = None
        self.current_map = None
        self.current_goal = None
        self.robot_out_of_map = False  # 标记机器人是否在地图外
        
        # ��������תȦ״̬��¼
        self.blacklist = []
        self.spin_start_time = None
        self.spin_duration = 21.0
        self.recent_stuck = False  # 记录最近是否卡住，用于调整frontier选择策略 
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ����/����
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_reached_sub = self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)
        self.goal_failed_sub = self.create_subscription(Bool, '/goal_failed', self.goal_failed_callback, 10)
        
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)  # ����ԭ����ת
        self.marker_pub = self.create_publisher(MarkerArray, '/exploration/visualization', 10)
        
        self.timer = self.create_timer(self.frontier_interval, self.exploration_loop)
        
        self.get_logger().info('Autonomous Explorer started (Global BFS Planner)')
        self.get_logger().info(f'  exploration_radius: {self.exploration_radius}m')
    
    def map_callback(self, msg):
        self.current_map = msg
    
    def goal_reached_callback(self, msg):
        if msg.data:
            self.get_logger().info('Goal reached! Scanning environment...')
            self.current_goal = None
            self.state = self.STATE_SPINNING
            self.spin_start_time = None
            self.spin_duration = 4.0  # 半圈扫描环境
    
    def goal_failed_callback(self, msg):
        if msg.data and self.current_goal:
            self.get_logger().warn(f'Goal Failed! Blacklisting ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})')
            self.blacklist.append(self.current_goal)
            self.current_goal = None
            self.recent_stuck = True  # 标记最近卡住
            
            # תȦ����Ѱ�ҷ���
            self.state = self.STATE_SPINNING
            self.spin_start_time = None
            self.spin_duration = 8.0  # ȫȦɨ��
    
    def exploration_loop(self):
        if self.state == self.STATE_INIT:
            self.handle_init()
        elif self.state == self.STATE_SPINNING:
            self.handle_spinning()
        elif self.state == self.STATE_EXPLORING:
            self.handle_exploring()
    
    def handle_init(self):
        if self.init_start_time is None:
            self.init_start_time = time.time()
        
        if self.start_pos is None:
            try:
                transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                self.start_pos = (transform.transform.translation.x, transform.transform.translation.y)
                self.get_logger().info(f'Start position recorded: ({self.start_pos[0]:.2f}, {self.start_pos[1]:.2f})')
            except TransformException:
                return
        
        elapsed = time.time() - self.init_start_time
        if elapsed < self.init_delay:
            return
        
        self.state = self.STATE_SPINNING
        self.spin_start_time = None
        self.spin_duration = 8.0

    def handle_spinning(self):
        if self.spin_start_time is None:
            self.spin_start_time = time.time()
            self.get_logger().info('Spinning to scan environment...')
        
        elapsed = time.time() - self.spin_start_time
        if elapsed < self.spin_duration:
            msg = Twist()
            msg.angular.z = 0.3
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(Twist())
            self.state = self.STATE_EXPLORING

    def check_blacklist(self, wx, wy):
        for bx, by, _ in self.blacklist:  # blacklist存储的是(x, y, dist)三元组
            if math.hypot(wx - bx, wy - by) < 0.6:  # �������뾶0.6m
                return True
        return False

    def handle_exploring(self):
        if self.current_map is None or self.start_pos is None or self.current_goal is not None:
            return
            
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            robot_pos = (transform.transform.translation.x, transform.transform.translation.y)
        except TransformException:
            return
            
        frontiers = self.detect_frontiers_bfs(robot_pos)
        self.publish_visualization(frontiers, robot_pos)
        
        # 添加地图成熟度检测：如果地图还很小，不应该判断"探索完成"
        if not self.current_map:
            if not frontiers:
                self.get_logger().info('No frontiers and no map, waiting...')
            return
        
        # 计算地图统计（只计算一次）
        grid = np.array(self.current_map.data, dtype=np.int8).reshape(
            self.current_map.info.height, self.current_map.info.width)
        free_count = np.sum((grid >= 0) & (grid <= 20))
        
        if not frontiers:
            # 如果机器人在地图外，说明地图还没覆盖到机器人位置，继续旋转
            if self.robot_out_of_map:
                self.get_logger().warn(f'No frontiers and robot out of map, re-spinning to expand map...')
                self.state = self.STATE_SPINNING
                self.spin_start_time = None
                self.spin_duration = 8.0
                return
            elif free_count < 200:
                # 地图太小，说明还没开始建图，重新旋转扫描
                self.get_logger().warn(f'No frontiers but map is too small (free={free_count}), re-spinning...')
                self.state = self.STATE_SPINNING
                self.spin_start_time = None
                self.spin_duration = 8.0
                return
            else:
                self.get_logger().info(f'No more accessible frontiers found, exploration complete! (free={free_count})')
                self.state = self.STATE_COMPLETE
                return
            
        # 智能选择策略：优先广度覆盖，避免深入角落
        # 1. 动态调整距离过滤：初期地图小时允许更近的frontier，后期提高要求
        min_frontier_dist = 0.2 if free_count < 500 else 0.5
        self.get_logger().info(f'Using min_frontier_dist={min_frontier_dist:.2f}m (free_count={free_count})')
        far_frontiers = [f for f in frontiers if f[2] >= min_frontier_dist]
        
        if far_frontiers:
            if self.recent_stuck:
                # 最近卡住过，优先选择较远的frontier（2-4米），放弃当前区域
                far_range_frontiers = [f for f in far_frontiers if 2.0 <= f[2] <= 4.0]
                if far_range_frontiers:
                    best_frontier = min(far_range_frontiers, key=lambda f: f[2])
                    self.get_logger().info(f'[STUCK_RECOVERY] Selected far frontier at {best_frontier[2]:.2f}m')
                else:
                    # 没有2-4米的，选最远的
                    best_frontier = max(far_frontiers, key=lambda f: f[2])
                    self.get_logger().info(f'[STUCK_RECOVERY] Selected farthest frontier at {best_frontier[2]:.2f}m')
                self.recent_stuck = False  # 重置标志
            else:
                # 正常情况：优先1.5-3.5米的中等距离（广度优先）
                medium_frontiers = [f for f in far_frontiers if 1.5 <= f[2] <= 3.5]
                if medium_frontiers:
                    best_frontier = min(medium_frontiers, key=lambda f: f[2])
                    self.get_logger().info(f'Selected medium-range frontier at {best_frontier[2]:.2f}m')
                else:
                    # 没有中等距离，选最远的（优先覆盖更广的区域）
                    best_frontier = max(far_frontiers, key=lambda f: f[2])
                    self.get_logger().info(f'Selected farthest frontier at {best_frontier[2]:.2f}m')
        else:
            # 全是近距离，说明快探索完了
            best_frontier = min(frontiers, key=lambda f: f[2])
            self.get_logger().info(f'Only near frontiers available at {best_frontier[2]:.2f}m')
        
        self.current_goal = best_frontier
        
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = best_frontier[0]
        goal_msg.pose.position.y = best_frontier[1]
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)

    def detect_frontiers_bfs(self, robot_pos):
        if self.current_map is None:
            return []
            
        grid = np.array(self.current_map.data, dtype=np.int8).reshape(self.current_map.info.height, self.current_map.info.width)
        res = self.current_map.info.resolution
        ox = self.current_map.info.origin.position.x
        oy = self.current_map.info.origin.position.y
        h, w = grid.shape
        
        # 调试：输出地图统计
        free_count = np.sum((grid >= 0) & (grid <= 20))
        unknown_count_total = np.sum(grid == -1)
        occupied_count = np.sum(grid > 50)
        self.get_logger().info(f'Map stats: free={free_count}, unknown={unknown_count_total}, occupied={occupied_count}, total={h*w}')
        
        rx = int((robot_pos[0] - ox) / res)
        ry = int((robot_pos[1] - oy) / res)
        
        # 输出详细调试信息
        self.get_logger().info(f'Map info: origin=({ox:.2f}, {oy:.2f}), size=({w}x{h}), res={res:.3f}')
        self.get_logger().info(f'Robot world position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})')
        self.get_logger().info(f'Robot grid position: ({rx}, {ry})')
        
        # 如果机器人在地图边界外（初期SLAM地图可能还没覆盖到机器人位置）
        self.robot_out_of_map = False
        if not (0 <= rx < w and 0 <= ry < h): 
            self.get_logger().warn(f'Robot position out of map bounds! Searching for valid BFS start point...')
            self.robot_out_of_map = True
            # 在地图中搜索一个有效的起点（优先自由空间，其次未知区域）
            found_valid_start = False
            # 先搜索自由空间
            for search_y in range(h):
                for search_x in range(w):
                    if 0 <= grid[search_y, search_x] <= 20:
                        rx, ry = search_x, search_y
                        found_valid_start = True
                        break
                if found_valid_start:
                    break
            # 如果没有自由空间，搜索未知区域
            if not found_valid_start:
                for search_y in range(h):
                    for search_x in range(w):
                        if grid[search_y, search_x] == -1:
                            rx, ry = search_x, search_y
                            found_valid_start = True
                            break
                    if found_valid_start:
                        break
            
            if found_valid_start:
                self.get_logger().info(f'Found valid start in map: ({rx}, {ry}), value={grid[ry, rx]}')
            else:
                self.get_logger().error('Cannot find any valid point in map! Map may be all obstacles.')
                return []
        
        robot_grid_value = grid[ry, rx]
        self.get_logger().info(f'BFS search center grid value: {robot_grid_value}')
        
        # 关键修复：确定BFS的中心点和半径限制
        # 如果机器人在地图外，使用找到的valid start作为半径中心
        # 否则使用start_pos作为中心（保持原有的探索半径逻辑）
        if self.robot_out_of_map:
            # 使用找到的valid start point作为中心
            sx, sy = rx, ry
            self.get_logger().info(f'Using valid start point as radius center: ({sx}, {sy})')
        else:
            # 使用启动位置作为中心（正常情况）
            sx = int((self.start_pos[0] - ox) / res)
            sy = int((self.start_pos[1] - oy) / res)
        
        max_dist_cells = self.exploration_radius / res

        # 关键修复：找到机器人附近的一个确定的自由空间作为BFS起点
        # 扩大搜索范围，确保初期也能找到起点
        start_found = False
        search_radius = 30  # 扩大到30个格子（约1.5米），确保初期也能找到
        start_x, start_y = rx, ry
        
        # 如果机器人当前位置是自由空间或未知区域，都可以使用
        # 初期允许从未知区域开始，这样才能找到frontier
        robot_val = grid[ry, rx]
        if robot_val == -1 or (0 <= robot_val <= 20):
            start_found = True
            start_x, start_y = rx, ry
            self.get_logger().info(f'Using robot position as BFS start (value={robot_val})')
        else:
            # 否则在周围螺旋搜索一个自由空间或未知空间
            for r in range(1, search_radius + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        test_x, test_y = rx + dx, ry + dy
                        if 0 <= test_x < w and 0 <= test_y < h:
                            test_val = grid[test_y, test_x]
                            if test_val == -1 or (0 <= test_val <= 20):
                                start_x, start_y = test_x, test_y
                                start_found = True
                                break
                    if start_found: break
                if start_found: break
            if start_found:
                self.get_logger().info(f'Found BFS start at offset ({start_x-rx}, {start_y-ry}), value={grid[start_y, start_x]}')
        
        if not start_found:
            self.get_logger().warn('Cannot find valid space near robot to start BFS!')
            return []

        visited = np.zeros((h, w), dtype=bool)
        visited[start_y, start_x] = True
        queue = deque([(start_x, start_y, 0.0)])
        raw_frontiers = []
        
        while queue:
            x, y, dist = queue.popleft()
            
            if dist > max_dist_cells * 1.5 * res: continue
            
            val = grid[y, x]
            # 1. 当前节点可以是自由空间或未知区域（初期需要这样才能找到frontier）
            # 只排除障碍物（>50）
            if val > 50: continue
            
            # 2. Check nearby unknown cells (-1)
            unknown_count = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if grid[ny, nx] == -1:
                            unknown_count += 1
            
            # 3. Validation: needs at least 1 unknown cells nearby (降低要求避免初期过滤太多)
            # 注意：找到frontier后不要continue，要继续扩散BFS以覆盖所有可达区域
            if unknown_count >= 1:
                wx = ox + x * res
                wy = oy + y * res
                if not self.check_blacklist(wx, wy):
                    # 使用从机器人当前位置到frontier的直线距离（而非BFS路径距离）
                    real_dist = math.hypot(wx - robot_pos[0], wy - robot_pos[1])
                    raw_frontiers.append((x, y, wx, wy, real_dist))
            
            # 4. 继续向周围的自由空间或未知区域扩散（这样才能覆盖到frontier边界）
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    if math.hypot(nx - sx, ny - sy) * res <= self.exploration_radius:
                        nval = grid[ny, nx]
                        # 放宽扩散条件：允许向自由空间和未知区域扩散（不进入障碍物）
                        if nval == -1 or (0 <= nval <= 20):
                            visited[ny, nx] = True
                            queue.append((nx, ny, dist + res))

        # BFS完成，输出统计
        visited_count = np.sum(visited)
        self.get_logger().info(f'BFS completed: visited {visited_count} cells, found {len(raw_frontiers)} raw frontiers')
        
        frontiers = []
        if len(raw_frontiers) > 0:
            groups = []
            for f in raw_frontiers:
                x, y, wx, wy, dist = f
                placed = False
                for g in groups:
                    cx, cy = g['center_x'], g['center_y']
                    # 5. Clustering within 0.4 meters
                    if math.hypot(wx - cx, wy - cy) < 0.4:
                        g['points'].append(f)
                        g['center_x'] = sum(p[2] for p in g['points']) / len(g['points'])
                        g['center_y'] = sum(p[3] for p in g['points']) / len(g['points'])
                        g['min_dist'] = min(g['min_dist'], dist)
                        placed = True
                        break
                if not placed:
                    groups.append({
                        'center_x': wx, 'center_y': wy, 
                        'points': [f], 'min_dist': dist
                    })
            
            # 6. Noise filtering: 根据地图大小动态调整聚类要求
            # 初期地图小，降低要求；后期地图大，提高要求过滤噪点
            min_cluster_size = 2 if free_count < 500 else 3
            valid_groups = [g for g in groups if len(g['points']) >= min_cluster_size]
            
            # 调试日志
            self.get_logger().info(f'Frontier detection: {len(raw_frontiers)} raw -> {len(groups)} groups -> {len(valid_groups)} valid (min_cluster={min_cluster_size})')
            
            for g in valid_groups:
                frontiers.append((g['center_x'], g['center_y'], g['min_dist']))
        else:
            self.get_logger().info('Frontier detection: 0 raw frontiers found in BFS')

        if len(frontiers) > 10:
            frontiers = sorted(frontiers, key=lambda f: f[2])
            frontiers = frontiers[:10]
            
        return frontiers

    def publish_visualization(self, frontiers, robot_pos):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        
        # ɾ�����оɵ� marker
        m_del = Marker()
        m_del.action = 3
        markers.markers.append(m_del)
        
        if self.start_pos is None: return
        
        circle = Marker()
        circle.header.frame_id = 'map'
        circle.header.stamp = stamp
        circle.id = 0
        circle.type = Marker.CYLINDER
        circle.action = Marker.ADD
        circle.pose.position.x = self.start_pos[0]
        circle.pose.position.y = self.start_pos[1]
        circle.pose.position.z = 0.0
        circle.pose.orientation.w = 1.0
        circle.scale.x = self.exploration_radius * 2
        circle.scale.y = self.exploration_radius * 2
        circle.scale.z = 0.01
        circle.color.r = 0.0; circle.color.g = 1.0; circle.color.b = 0.0; circle.color.a = 0.15
        markers.markers.append(circle)
        
        for i, bp in enumerate(self.blacklist):
            bm = Marker()
            bm.header.frame_id = 'map'
            bm.header.stamp = stamp
            bm.id = 500 + i
            bm.type = Marker.CUBE
            bm.action = Marker.ADD
            bm.pose.position.x = bp[0]
            bm.pose.position.y = bp[1]
            bm.pose.position.z = 0.15
            bm.pose.orientation.w = 1.0
            bm.scale.x = 0.2; bm.scale.y = 0.2; bm.scale.z = 0.2
            bm.color.r = 1.0; bm.color.g = 0.0; bm.color.b = 0.0; bm.color.a = 0.8
            markers.markers.append(bm)
        
        for i, f in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.id = 1000 + i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = f[0]
            m.pose.position.y = f[1]
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1; m.scale.y = 0.1; m.scale.z = 0.1
            m.color.r = 0.0; m.color.g = 0.5; m.color.b = 1.0; m.color.a = 0.8
            markers.markers.append(m)
            
        if self.current_goal:
            gm = Marker()
            gm.header.frame_id = 'map'
            gm.header.stamp = stamp
            gm.id = 9999
            gm.type = Marker.SPHERE
            gm.action = Marker.ADD
            gm.pose.position.x = self.current_goal[0]
            gm.pose.position.y = self.current_goal[1]
            gm.pose.position.z = 0.3
            gm.pose.orientation.w = 1.0
            gm.scale.x = 0.25; gm.scale.y = 0.25; gm.scale.z = 0.25
            gm.color.r = 1.0; gm.color.g = 1.0; gm.color.b = 0.0; gm.color.a = 1.0
            markers.markers.append(gm)
            
        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
