#!/usr/bin/env python3
"""
Standalone map-navigation goal bridge.

Accepts RViz or external PoseStamped goals and forwards them to the
custom navigator's NavigateToPose action.
"""

import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class MapGoalBridge(Node):

    def __init__(self):
        super().__init__('map_goal_bridge')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('legacy_goal_topic', '/move_base_simple/goal')
        self.declare_parameter('nav_action_name', 'navigate_to_pose')

        goal_topic = str(self.get_parameter('goal_topic').value)
        legacy_goal_topic = str(self.get_parameter('legacy_goal_topic').value)
        nav_action_name = str(self.get_parameter('nav_action_name').value)

        self.goal_handle = None
        self.goal_send_time = 0.0

        self.nav_client = ActionClient(self, NavigateToPose, nav_action_name)
        self.status_pub = self.create_publisher(String, '/navigation/simple_nav_status', 10)
        self.goal_sub = self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)
        self.legacy_goal_sub = self.create_subscription(
            PoseStamped, legacy_goal_topic, self._on_goal, 10)
        self.timer = self.create_timer(0.5, self._tick)

        self.get_logger().info(
            f'MapGoalBridge ready | goal_topic={goal_topic} '
            f'legacy_goal_topic={legacy_goal_topic}')

    def _tick(self):
        msg = String()
        if self.goal_handle is None:
            msg.data = 'idle'
        else:
            msg.data = f'navigating for {time.time() - self.goal_send_time:.1f}s'
        self.status_pub.publish(msg)

    def _on_goal(self, msg):
        if msg.header.frame_id and msg.header.frame_id != 'map':
            self.get_logger().warn(
                f'Ignoring goal in frame {msg.header.frame_id}; expected map')
            return

        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('navigate_to_pose action unavailable')
            return

        if self.goal_handle is not None:
            self.get_logger().info('Preempting previous map-navigation goal')
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

        goal = NavigateToPose.Goal()
        goal.pose = msg
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        if goal.pose.pose.orientation.w == 0.0 and goal.pose.pose.orientation.z == 0.0:
            goal.pose.pose.orientation.w = 1.0

        self.goal_send_time = time.time()
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)
        self.get_logger().info(
            f'Forwarding goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def _on_goal_response(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().warn('Map-navigation goal rejected')
            self.goal_handle = None
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, fut):
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Map-navigation goal reached')
        else:
            self.get_logger().warn(f'Map-navigation goal finished with status={status}')
        self.goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = MapGoalBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
