"""
Real robot navigation with Cartographer pure localization.

Components:
  1. Robot base drivers (controller + lidar)
  2. Cartographer pure localization (load pbstream)
  3. Cartographer occupancy grid publisher
  4. CustomNavigator (A* + pure-pursuit)
  5. LaserEmergencyStopper (/nav_cmd_vel -> /cmd_vel)
  6. MapGoalBridge (RViz goal -> NavigateToPose)
  7. RViz (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def _default_log_root():
    return os.environ.get(
        'EXPLORATION_LOG_ROOT',
        os.path.join(os.path.expanduser('~'), '.ros', 'exploration_logs'),
    )


def _default_pbstream_path():
    return os.environ.get(
        'EXPLORATION_PBSTREAM_PATH',
        '/home/test/ros2_ws/src/exploration/map/c1_map.pbstream',
    )


def generate_launch_description():
    exploration_pkg = get_package_share_directory('exploration')
    slam_pkg = get_package_share_directory('slam')
    cartographer_pkg = get_package_share_directory('cartographer_ros')

    nav_params = os.path.join(exploration_pkg, 'config', 'custom_nav_real_robot_navigation_params.yaml')
    rviz_config = os.path.join(exploration_pkg, 'rviz', 'exploration.rviz')
    cartographer_config_dir = os.path.join(cartographer_pkg, 'configuration_files')

    sim_arg = DeclareLaunchArgument('sim', default_value='false')
    master_arg = DeclareLaunchArgument(
        'master_name', default_value=os.environ.get('MASTER', 'master'))
    robot_arg = DeclareLaunchArgument(
        'robot_name', default_value=os.environ.get('HOST', '/'))
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    record_arg = DeclareLaunchArgument('record_diagnostics', default_value='true')
    log_root_arg = DeclareLaunchArgument('log_root', default_value=_default_log_root())

    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='scan')
    carto_lua_arg = DeclareLaunchArgument(
        'cartographer_lua', default_value='c1_real_robot_2d_localization.lua')
    carto_resolution_arg = DeclareLaunchArgument('cartographer_resolution', default_value='0.05')
    load_state_arg = DeclareLaunchArgument(
        'load_state_filename',
        default_value=_default_pbstream_path(),
    )

    sim = LaunchConfiguration('sim')
    master_name = LaunchConfiguration('master_name')
    robot_name = LaunchConfiguration('robot_name')
    rviz_flag = LaunchConfiguration('rviz')
    record_flag = LaunchConfiguration('record_diagnostics')
    log_root = LaunchConfiguration('log_root')
    scan_topic = LaunchConfiguration('scan_topic')
    cartographer_lua = LaunchConfiguration('cartographer_lua')
    cartographer_resolution = LaunchConfiguration('cartographer_resolution')
    load_state_filename = LaunchConfiguration('load_state_filename')

    disable_shm = SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'include', 'robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name,
        }.items(),
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', cartographer_lua,
            '-load_state_filename', load_state_filename,
        ],
        remappings=[
            ('scan', scan_topic),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[
            {'use_sim_time': False},
            {'resolution': cartographer_resolution},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('/map', 'map'),
        ],
    )

    bringup = GroupAction(actions=[
        PushRosNamespace(robot_name),
        base_launch,
        TimerAction(period=5.0, actions=[cartographer_node, occupancy_grid_node]),
    ])

    custom_nav = Node(
        package='exploration',
        executable='optimized_custom_navigator',
        name='optimized_custom_navigator',
        output='screen',
        parameters=[nav_params, {
            'use_open_loop_pose': False,
            'max_vel_x': 0.750,
            'max_vel_theta': 1.680,
            'cmd_vel_topic': '/raw_nav_cmd_vel',
            'cmd_smoothing_alpha': 0.12,
            'max_angular_reversal_accel': 6.0,
            'turn_sign_lock_margin': 0.04,
            'recovery_head_turn_angle_deg': 75.0,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
    )

    cmd_guard = Node(
        package='exploration',
        executable='cmd_vel_guard',
        name='cmd_vel_guard',
        output='screen',
        parameters=[{
            'input_topic': '/raw_nav_cmd_vel',
            'output_topic': '/nav_cmd_vel',
            'max_linear': 0.825,
            'max_angular': 1.68,
            'max_angular_accel': 3.0,
            'max_angular_reversal_accel': 6.0,
            'min_angular_speed': 0.20,
            'watchdog_timeout': 0.35,
        }],
    )

    laser_stopper = Node(
        package='exploration',
        executable='laser_emergency_stopper',
        name='laser_emergency_stopper',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'check_radius': 0.24,
            'min_points_for_block': 2,
            'max_continuous_move_time': 99999.0,
            'max_vel_x': 0.750,
            'emergency_timeout': 3.0,
            'monitor_sector_center_deg': 180.0,
            'monitor_sector_width_deg': 100.0,
            'ignore_sector_centers_deg': [0.0],
            'ignore_sector_width_deg': 0.0,
            'ignore_sector_margin_deg': 0.0,
        }],
    )

    goal_bridge = Node(
        package='exploration',
        executable='map_goal_bridge',
        name='map_goal_bridge',
        output='screen',
        parameters=[{
            'goal_topic': '/exploration/goal',
            'legacy_goal_topic': '/goal_pose',
        }],
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
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
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

    recorder_timer = TimerAction(period=2.0, actions=[recorder])
    navigator_timer = TimerAction(period=12.0, actions=[custom_nav, cmd_guard, laser_stopper, goal_bridge])

    return LaunchDescription([
        sim_arg,
        master_arg,
        robot_arg,
        rviz_arg,
        record_arg,
        log_root_arg,
        scan_topic_arg,
        carto_lua_arg,
        carto_resolution_arg,
        load_state_arg,
        disable_shm,
        bringup,
        recorder_timer,
        navigator_timer,
        rviz,
    ])
