#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped

class AutoInitialPoseEstimator(Node):
    def __init__(self):
        super().__init__('auto_initial_pose_estimator')
        self.map_data = None
        self.scan_data = None
        self.published = False

        # Parameters for base initial pose (center of our search)
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)

        # QoS for /map (typically latched / transient local)
        qos_profile = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.get_logger().info('Auto Initial Pose Estimator started, waiting for map and scan...')

    def map_callback(self, msg):
        if self.map_data is None:
            self.map_data = msg
            self.get_logger().info('Map received.')
            self.try_estimate_and_publish()

    def scan_callback(self, msg):
        if self.scan_data is None:
            self.scan_data = msg
            self.get_logger().info('First LaserScan received.')
            self.try_estimate_and_publish()

    def try_estimate_and_publish(self):
        if self.map_data is None or self.scan_data is None or self.published:
            return
        
        self.get_logger().info('Starting local grid search for precise initial pose...')
        best_pose = self.find_best_pose()
        self.publish_initial_pose(best_pose)
        self.published = True
        self.get_logger().info('Initial pose published successfully! Exiting logic loop.')
        
        # Cleanup subscriptions to stop processing
        self.destroy_subscription(self.map_sub)
        self.destroy_subscription(self.scan_sub)

    def find_best_pose(self):
        # Center of our search region
        init_x = float(self.get_parameter('initial_pose_x').value)
        init_y = float(self.get_parameter('initial_pose_y').value)
        init_yaw = float(self.get_parameter('initial_pose_yaw').value)

        # Build search space (±10cm positional, ±45 degrees angular)
        # Position step = 0.02m (2cm)
        x_steps = [init_x + x / 100.0 for x in range(-10, 11, 2)]
        y_steps = [init_y + y / 100.0 for y in range(-10, 11, 2)]
        
        # Angular step = 2 degrees
        yaw_steps = [init_yaw + math.radians(th) for th in range(-44, 45, 2)]

        # Map info
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        height = self.map_data.info.height
        data = self.map_data.data

        # Pre-process valid laser scan points into (x, y) relative to robot base
        ranges = self.scan_data.ranges
        angle_min = self.scan_data.angle_min
        angle_inc = self.scan_data.angle_increment
        range_min = self.scan_data.range_min
        range_max = self.scan_data.range_max

        base_pts = []
        for i, r in enumerate(ranges):
            if range_min <= r <= range_max and math.isfinite(r):
                angle = angle_min + i * angle_inc
                base_pts.append((r * math.cos(angle), r * math.sin(angle)))

        # Downsample points for speed if there are too many (e.g., > 180 points)
        if len(base_pts) > 180:
            step = len(base_pts) // 180
            base_pts = base_pts[::step]
            
        if not base_pts:
            self.get_logger().warn('No valid scan points. Falling back to default start pose.')
            return (init_x, init_y, init_yaw)

        best_score = -1
        best_pose = (init_x, init_y, init_yaw)

        # Brute force search
        for yaw in yaw_steps:
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            
            # Rotate all base points
            rot_pts = [(bx*cos_y - by*sin_y, bx*sin_y + by*cos_y) for bx, by in base_pts]
            
            for tx in x_steps:
                for ty in y_steps:
                    score = 0
                    for rx, ry in rot_pts:
                        mx = tx + rx
                        my = ty + ry
                        
                        px = int((mx - origin_x) / resolution)
                        py = int((my - origin_y) / resolution)
                        
                        if 0 <= px < width and 0 <= py < height:
                            idx = py * width + px
                            val = data[idx]
                            # Count matching obstacles
                            if val > 65:  
                                score += 1
                                
                    if score > best_score:
                        best_score = score
                        best_pose = (tx, ty, yaw)

        match_ratio = best_score / len(base_pts) * 100.0
        self.get_logger().info(
            f'Local Match Result: X={best_pose[0]:.3f}, Y={best_pose[1]:.3f}, '
            f'Yaw={math.degrees(best_pose[2]):.1f}° | Score: {best_score}/{len(base_pts)} '
            f'({match_ratio:.1f}%)')
            
        return best_pose

    def publish_initial_pose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = 0.0
        
        yaw = pose[2]
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Extremely small covariance array to force immediate tight particle convergence
        cov = [0.0] * 36
        cov[0] = 1e-9    # x variance
        cov[7] = 1e-9    # y variance
        cov[35] = 1e-9   # yaw variance
        msg.pose.covariance = cov
        
        # Publish multiple times with small delay to ensure AMCL receives it
        self.pose_pub.publish(msg)
        self.get_logger().info('High-confidence Initial Pose Sent directly to AMCL.')

def main(args=None):
    rclpy.init(args=args)
    node = AutoInitialPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
