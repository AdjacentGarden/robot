#!/usr/bin/env python3
"""
简化版探索节点 - 使用Nav2进行路径规划和避障
只负责frontier检测和目标发布，将导航任务交给Nav2
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from collections import deque
import math

class SimpleNav2Explorer(Node):
    def __init__(self):
        super().__init__('simple_nav2_explorer')
        
        # 参数
        self.declare_parameter('exploration_radius', 5.0)
        self.declare_parameter('min_frontier_size', 0.3)
        
        self.exploration_radius = self.get_parameter('exploration_radius').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        
        # 订阅地图
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 可视化
        self.marker_pub = self.create_publisher(
            MarkerArray, '/exploration/frontiers', 10)
        
        self.current_map = None
        self.exploring = False
        
        # 定时检查frontier
        self.create_timer(2.0, self.exploration_timer)
        
        self.get_logger().info('Simple Nav2 Explorer initialized')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 action server ready!')
    
    def map_callback(self, msg):
        self.current_map = msg
    
    def exploration_timer(self):
        if self.exploring or self.current_map is None:
            return
        
        frontiers = self.detect_frontiers()
        self.publish_markers(frontiers)
        
        if not frontiers:
            self.get_logger().info('No frontiers found, exploration complete!')
            return
        
        # 选择最近的frontier
        best = min(frontiers, key=lambda f: f['distance'])
        
        self.get_logger().info(
            f'Sending goal to Nav2: ({best["position"][0]:.2f}, {best["position"][1]:.2f})')
        
        self.send_nav2_goal(best['position'])
    
    def detect_frontiers(self):
        """简化的frontier检测"""
        grid = np.array(self.current_map.data).reshape(
            self.current_map.info.height, self.current_map.info.width)
        
        res = self.current_map.info.resolution
        ox = self.current_map.info.origin.position.x
        oy = self.current_map.info.origin.position.y
        h, w = grid.shape
        
        frontiers = []
        visited = np.zeros((h, w), dtype=bool)
        
        # 扫描地图找frontier
        for y in range(1, h-1):
            for x in range(1, w-1):
                if visited[y, x] or grid[y, x] != 0:  # 跳过非自由空间
                    continue
                
                # 检查是否有未知区域邻居
                has_unknown = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if grid[y+dy, x+dx] == -1:
                            has_unknown = True
                            break
                    if has_unknown:
                        break
                
                if has_unknown:
                    wx = ox + x * res
                    wy = oy + y * res
                    dist = math.hypot(wx, wy)  # 到原点距离
                    
                    if dist < self.exploration_radius:
                        frontiers.append({
                            'position': (wx, wy),
                            'distance': dist
                        })
                        visited[y, x] = True
        
        self.get_logger().info(f'Found {len(frontiers)} frontier points')
        return frontiers
    
    def send_nav2_goal(self, position):
        """发送目标到Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = position[0]
        goal_msg.pose.pose.position.y = position[1]
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.exploring = True
        
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback)
        send_goal_future.add_done_callback(self.nav_goal_response_callback)
    
    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected goal')
            self.exploring = False
            return
        
        self.get_logger().info('Nav2 accepted goal')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)
    
    def nav_feedback_callback(self, feedback_msg):
        # 可以在这里处理导航反馈
        pass
    
    def nav_result_callback(self, future):
        result = future.result().result
        self.exploring = False
        self.get_logger().info(f'Nav2 navigation completed')
    
    def publish_markers(self, frontiers):
        markers = MarkerArray()
        
        for i, f in enumerate(frontiers[:20]):  # 最多显示20个
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = f['position'][0]
            m.pose.position.y = f['position'][1]
            m.pose.position.z = 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            markers.markers.append(m)
        
        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNav2Explorer()
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
