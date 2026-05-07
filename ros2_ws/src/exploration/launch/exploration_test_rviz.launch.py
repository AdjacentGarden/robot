"""
Desktop test: sim environment + custom navigator + frontier explorer + RViz.

No real robot, no SLAM Toolbox. Uses a 2D simulator with random map generation,
raycasting lidar, and SLAM-like incremental map reveal.

Usage:
  ros2 launch exploration exploration_test_rviz.launch.py
  ros2 launch exploration exploration_test_rviz.launch.py seed:=42
  ros2 launch exploration exploration_test_rviz.launch.py seed:=42 map_width:=15.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_log_root():
    return os.environ.get(
        'EXPLORATION_LOG_ROOT',
        os.path.join(os.path.expanduser('~'), '.ros', 'exploration_logs'),
    )


def generate_launch_description():
    pkg = get_package_share_directory('exploration')
    nav_params = os.path.join(pkg, 'config', 'custom_nav_params.yaml')
    explore_params = os.path.join(pkg, 'config', 'explore_params.yaml')
    rviz_cfg = os.path.join(pkg, 'rviz', 'exploration.rviz')

    args = [
        DeclareLaunchArgument('seed', default_value='-1',
                              description='Map RNG seed (-1 = random)'),
        DeclareLaunchArgument('map_width', default_value='10.0'),
        DeclareLaunchArgument('map_height', default_value='10.0'),
        DeclareLaunchArgument('num_rooms', default_value='6'),
        DeclareLaunchArgument('num_obstacles', default_value='8'),
        DeclareLaunchArgument('lidar_range', default_value='8.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('record_diagnostics', default_value='true'),
        DeclareLaunchArgument('log_root', default_value=_default_log_root()),
    ]

    # 1. Sim Environment (map + lidar + TF + /cmd_vel)
    sim = Node(
        package='exploration',
        executable='sim_environment',
        name='sim_environment',
        output='screen',
        parameters=[{
            'map_width':      LaunchConfiguration('map_width'),
            'map_height':     LaunchConfiguration('map_height'),
            'resolution':     0.05,
            'num_rooms':      LaunchConfiguration('num_rooms'),
            'num_obstacles':  LaunchConfiguration('num_obstacles'),
            'lidar_range':    LaunchConfiguration('lidar_range'),
            'lidar_rays':     360,
            'lidar_hz':       10.0,
            'seed':           LaunchConfiguration('seed'),
            'start_x':        0.0,
            'start_y':        0.0,
        }],
    )

    # 2. Custom Navigator (A* + Pure-Pursuit)
    navigator = Node(
        package='exploration',
        executable='custom_navigator',
        name='custom_navigator',
        output='screen',
        parameters=[nav_params],
    )

    # 3. Frontier Explorer (3 cycles for demo recording)
    explorer = Node(
        package='exploration',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[explore_params, {
            'num_cycles': 3,
        }],
    )

    recorder = Node(
        package='exploration',
        executable='diagnostic_recorder',
        name='diagnostic_recorder',
        output='screen',
        parameters=[{
            'log_root': LaunchConfiguration('log_root'),
            'pose_log_hz': 2.0,
        }],
        condition=IfCondition(LaunchConfiguration('record_diagnostics')),
    )

    # 4. RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(
        args + [
            sim,
            TimerAction(period=1.0, actions=[recorder]),
            rviz,
            TimerAction(period=3.0, actions=[navigator]),
            TimerAction(period=8.0, actions=[explorer]),
        ]
    )
