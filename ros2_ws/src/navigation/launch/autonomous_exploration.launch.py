import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'exploration_radius', default_value='5.0',
            description='Exploration radius in meters'
        ),
        
        DeclareLaunchArgument(
            'init_delay',
            default_value='20.0',
            description='Delay before starting exploration (for RViz startup)'
        ),
        
        # 探索路径规划�?
        Node(
            package='navigation',
            executable='exploration_path_tracker',
            name='exploration_path_tracker',
            output='screen',
            parameters=[{
                'map_frame': 'map',
                'base_frame': 'base_footprint',
                'scan_topic': '/scan',
                'scan_topic_candidates': ['/scan', 'scan', '/jetson/scan'],
                'max_linear_vel': 0.3,
                'max_angular_vel': 0.8,
                'goal_tolerance': 0.3,  # 放宽到达容忍�?
                'emergency_stop_dist': 0.15,
                'collision_dist': 0.08,
                'costmap_size': 2.0,
                'costmap_resolution': 0.05,
                'inflation_radius': 0.30,
                'horizon_sec': 1.2,
                'sim_dt': 0.1,
                'num_candidates': 11,
                'w_obstacle': 15.0,
                'w_goal': 5.0,
                'w_heading': 2.0,
                'w_smooth': 1.5,
                'stuck_timeout': 3.0,
                'stuck_distance': 0.15,
                'stuck_rotation_speed': 0.6,
                'enable_reverse': True,
            }],
            remappings=[
                ('cmd_vel', '/controller/cmd_vel'),
            ]
        ),
        
        # 自主探索�?
        Node(
            package='navigation',
            executable='autonomous_explorer',
            name='autonomous_explorer',
            output='screen',
            parameters=[{
                'exploration_radius': LaunchConfiguration('exploration_radius'),
                'init_delay': LaunchConfiguration('init_delay'),
                'frontier_check_interval': 1.0,
            }]
        ),
    ])

