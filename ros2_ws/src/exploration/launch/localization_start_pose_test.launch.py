"""
Simulation-only known-map localization test.

This launch intentionally avoids all real-robot bringup. It starts:
  fixed-map simulator + map_server + AMCL + precise startup pose estimation +
  optional runtime relocalization + RViz.
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
    rviz_cfg = os.path.join(pkg, 'rviz', 'exploration.rviz')
    default_map = os.path.join(pkg, 'map', 'exploration_map.yaml')

    args = [
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('record_diagnostics', default_value='true'),
        DeclareLaunchArgument('enable_runtime_relocalizer', default_value='false'),
        DeclareLaunchArgument('log_root', default_value=_default_log_root()),
        DeclareLaunchArgument('initial_pose_x', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_y', default_value='0.0'),
        DeclareLaunchArgument('initial_pose_yaw', default_value='0.0'),
        DeclareLaunchArgument('actual_start_x', default_value='0.0'),
        DeclareLaunchArgument('actual_start_y', default_value='0.0'),
        DeclareLaunchArgument('actual_start_yaw', default_value='0.0'),
        DeclareLaunchArgument('lidar_range', default_value='8.0'),
        DeclareLaunchArgument('search_x_range', default_value='0.20'),
        DeclareLaunchArgument('search_y_range', default_value='0.20'),
        DeclareLaunchArgument('search_yaw_range_deg', default_value='24.0'),
        DeclareLaunchArgument('min_match_ratio', default_value='0.07'),
        DeclareLaunchArgument('scan_sample_count', default_value='20'),
        DeclareLaunchArgument('startup_scan_duration', default_value='5.0'),
        DeclareLaunchArgument('startup_scan_linear_speed', default_value='0.05'),
        DeclareLaunchArgument('startup_scan_angular_speed', default_value='0.06'),
        DeclareLaunchArgument('startup_min_translation', default_value='0.20'),
        DeclareLaunchArgument('startup_min_rotation_deg', default_value='999.0'),
    ]

    map_file = LaunchConfiguration('map')
    rviz_flag = LaunchConfiguration('rviz')
    record_flag = LaunchConfiguration('record_diagnostics')
    runtime_relocalizer_flag = LaunchConfiguration('enable_runtime_relocalizer')
    log_root = LaunchConfiguration('log_root')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    actual_start_x = LaunchConfiguration('actual_start_x')
    actual_start_y = LaunchConfiguration('actual_start_y')
    actual_start_yaw = LaunchConfiguration('actual_start_yaw')
    lidar_range = LaunchConfiguration('lidar_range')
    search_x_range = LaunchConfiguration('search_x_range')
    search_y_range = LaunchConfiguration('search_y_range')
    search_yaw_range_deg = LaunchConfiguration('search_yaw_range_deg')
    min_match_ratio = LaunchConfiguration('min_match_ratio')
    scan_sample_count = LaunchConfiguration('scan_sample_count')
    startup_scan_duration = LaunchConfiguration('startup_scan_duration')
    startup_scan_linear_speed = LaunchConfiguration('startup_scan_linear_speed')
    startup_scan_angular_speed = LaunchConfiguration('startup_scan_angular_speed')
    startup_min_translation = LaunchConfiguration('startup_min_translation')
    startup_min_rotation_deg = LaunchConfiguration('startup_min_rotation_deg')

    localization_params = RewrittenYaml(
        source_file=amcl_params,
        param_rewrites={
            'yaml_filename': map_file,
            'set_initial_pose': 'true',
            'initial_pose.x': initial_pose_x,
            'initial_pose.y': initial_pose_y,
            'initial_pose.yaw': initial_pose_yaw,
            'tf_broadcast': 'true',
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
            'start_x': actual_start_x,
            'start_y': actual_start_y,
            'start_yaw': actual_start_yaw,
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

    amcl_auto_relocalizer = Node(
        package='exploration',
        executable='amcl_auto_relocalizer',
        name='amcl_auto_relocalizer',
        output='screen',
        parameters=[{
            # Startup pose correction is handled by auto_initial_pose_estimator
            # in this launch. Keep AMCL relocalization for runtime recovery only.
            'startup_delay': 999999.0,
            'enable_motion_assist': False,
            'lost_timeout': 3.0,
            'retry_cooldown': 12.0,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        condition=IfCondition(runtime_relocalizer_flag),
    )

    initial_pose_estimator = Node(
        package='exploration',
        executable='auto_initial_pose_estimator',
        name='auto_initial_pose_estimator',
        output='screen',
        parameters=[{
            'initial_pose_x': initial_pose_x,
            'initial_pose_y': initial_pose_y,
            'initial_pose_yaw': initial_pose_yaw,
            'search_x_range': search_x_range,
            'search_y_range': search_y_range,
            'search_yaw_range_deg': search_yaw_range_deg,
            'search_xy_step': 0.01,
            'search_yaw_step_deg': 1.0,
            'coarse_xy_step': 0.03,
            'coarse_yaw_step_deg': 3.0,
            'scan_sample_count': scan_sample_count,
            'startup_scan_duration': startup_scan_duration,
            'startup_scan_linear_speed': startup_scan_linear_speed,
            'startup_scan_angular_speed': startup_scan_angular_speed,
            'startup_min_translation': startup_min_translation,
            'startup_min_rotation_deg': startup_min_rotation_deg,
            'center_bias_xy_weight': 2.5,
            'center_bias_yaw_weight': 0.8,
            'publish_repeats': 5,
            'publish_interval': 0.25,
            'candidate_count': 6,
            'candidate_refine_radius': 0.05,
            'candidate_refine_yaw_deg': 5.0,
            'candidate_amcl_wait': 0.8,
            'candidate_nomotion_updates': 4,
            'candidate_cov_weight': 6.0,
            'candidate_pose_delta_weight': 8.0,
            'candidate_yaw_delta_weight': 2.5,
            'candidate_min_amcl_age': 0.20,
            'candidate_min_amcl_msgs': 2,
            'unknown_space_penalty': 0.8,
            'min_match_ratio': min_match_ratio,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    navigator = Node(
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
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=IfCondition(rviz_flag),
    )

    return LaunchDescription(
        args + [
            sim,
            TimerAction(period=1.0, actions=[map_server, amcl, lifecycle]),
            TimerAction(period=2.0, actions=[recorder]),
            TimerAction(period=2.5, actions=[initial_pose_estimator]),
            TimerAction(period=7.0, actions=[amcl_auto_relocalizer]),
            TimerAction(period=12.0, actions=[navigator, goal_bridge]),
            rviz,
        ]
    )
