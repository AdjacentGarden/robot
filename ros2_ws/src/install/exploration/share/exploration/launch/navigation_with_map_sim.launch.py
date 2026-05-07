"""
Known-map navigation test in the lightweight simulator.

Components:
  1. SimEnvironment on a fixed occupancy map
  2. nav2_map_server + AMCL localization
  3. CustomNavigator (A* + pure-pursuit)
  4. MapGoalBridge (RViz/simple goal -> NavigateToPose)
  5. RViz (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _default_log_root():
    return os.environ.get(
        'EXPLORATION_LOG_ROOT',
        os.path.join(os.path.expanduser('~'), '.ros', 'exploration_logs'),
    )


def generate_launch_description():
    pkg = get_package_share_directory('exploration')
    nav_params = os.path.join(pkg, 'config', 'custom_nav_params.yaml')
    amcl_params = os.path.join(pkg, 'config', 'amcl_params.yaml')
    rviz_config = os.path.join(pkg, 'rviz', 'exploration.rviz')
    default_map = os.path.join(pkg, 'map', 'exploration_map.yaml')

    args = [
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('record_diagnostics', default_value='true'),
        DeclareLaunchArgument('log_root', default_value=_default_log_root()),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument('lidar_range', default_value='8.0'),
    ]

    map_file = LaunchConfiguration('map')
    rviz_flag = LaunchConfiguration('rviz')
    record_flag = LaunchConfiguration('record_diagnostics')
    log_root = LaunchConfiguration('log_root')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    lidar_range = LaunchConfiguration('lidar_range')

    localization_params = RewrittenYaml(
        source_file=amcl_params,
        param_rewrites={
            'yaml_filename': map_file,
            'set_initial_pose': 'true',
            'initial_pose.x': initial_pose_x,
            'initial_pose.y': initial_pose_y,
            'initial_pose.yaw': initial_pose_yaw,
        },
        convert_types=True,
    )

    sim = Node(
        package='exploration',
        executable='sim_environment',
        name='sim_environment',
        output='screen',
        parameters=[{
            'fixed_map_yaml': map_file,
            'start_x': initial_pose_x,
            'start_y': initial_pose_y,
            'lidar_range': lidar_range,
            'lidar_rays': 360,
            'lidar_hz': 10.0,
            'publish_map_topic': False,
            'publish_map_odom_tf': False,
        }],
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[localization_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[localization_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    custom_nav = Node(
        package='exploration',
        executable='custom_navigator',
        name='custom_navigator',
        output='screen',
        parameters=[nav_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    goal_bridge = Node(
        package='exploration',
        executable='map_goal_bridge',
        name='map_goal_bridge',
        output='screen',
    )

    recorder = Node(
        package='exploration',
        executable='diagnostic_recorder',
        name='diagnostic_recorder',
        output='screen',
        parameters=[{
            'log_root': log_root,
            'pose_log_hz': 2.0,
        }],
        condition=IfCondition(record_flag),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(rviz_flag),
    )

    return LaunchDescription(
        args + [
            sim,
            TimerAction(period=1.0, actions=[map_server, amcl, lifecycle]),
            TimerAction(period=2.0, actions=[recorder]),
            TimerAction(period=3.0, actions=[custom_nav, goal_bridge]),
            rviz,
        ]
    )
