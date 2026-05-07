"""
Real robot: base drivers + Cartographer + custom navigator + frontier explorer.

Replaces SLAM Toolbox with Cartographer while keeping the same exploration stack:
  1. Robot base drivers (controller, lidar, IMU)
  2. Cartographer (mapping + occupancy grid)
  3. CustomNavigator (A* + pure-pursuit)
  4. FrontierExplorer (sends NavigateToPose goals)
  5. RViz (optional)

Usage:
  ros2 launch exploration exploration_real_robot_cartographer.launch.py
  ros2 launch exploration exploration_real_robot_cartographer.launch.py rviz:=false
  ros2 launch exploration exploration_real_robot_cartographer.launch.py disable_map_save:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
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
    # Package paths
    exploration_pkg = get_package_share_directory('exploration')
    slam_pkg = get_package_share_directory('slam')
    cartographer_pkg = get_package_share_directory('cartographer_ros')

    # Config files
    nav_params = os.path.join(exploration_pkg, 'config', 'custom_nav_real_robot_mapping_params.yaml')
    explore_params = os.path.join(exploration_pkg, 'config', 'explore_params.yaml')
    rviz_config = os.path.join(exploration_pkg, 'rviz', 'exploration.rviz')
    cartographer_config_dir = os.path.join(cartographer_pkg, 'configuration_files')

    # Launch arguments
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
        'cartographer_lua', default_value='c1_real_robot_2d.lua')
    carto_resolution_arg = DeclareLaunchArgument('cartographer_resolution', default_value='0.05')
    save_state_arg = DeclareLaunchArgument(
        'save_state_filename', default_value=_default_pbstream_path())
    disable_map_save_arg = DeclareLaunchArgument(
        'disable_map_save',
        default_value='false',
        description='If true, do not save pbstream map on shutdown.',
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
    save_state_filename = LaunchConfiguration('save_state_filename')
    disable_map_save = LaunchConfiguration('disable_map_save')

    # Disable FastDDS SHM
    disable_shm = SetEnvironmentVariable(
        'RMW_FASTRTPS_USE_QOS_FROM_XML', '0')

    # 1. Robot base + Cartographer
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'include', 'robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name,
        }.items(),
    )

    cartographer_node_with_save = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', cartographer_lua,
            '-save_state_filename', save_state_filename,
        ],
        remappings=[
            ('scan', scan_topic),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        condition=UnlessCondition(disable_map_save),
    )

    cartographer_node_without_save = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=[
            '-configuration_directory', cartographer_config_dir,
            '-configuration_basename', cartographer_lua,
        ],
        remappings=[
            ('scan', scan_topic),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        condition=IfCondition(disable_map_save),
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
        TimerAction(
            period=5.0,
            actions=[
                cartographer_node_with_save,
                cartographer_node_without_save,
                occupancy_grid_node,
            ],
        ),
    ])

    # 2. Optimized Custom Navigator
    custom_nav = Node(
        package='exploration',
        executable='optimized_custom_navigator',
        name='optimized_custom_navigator',
        output='screen',
        parameters=[nav_params, {
            'max_vel_x': 0.336,
            'max_vel_theta': 1.365,
            'cmd_vel_topic': '/raw_nav_cmd_vel',
            'cmd_smoothing_alpha': 0.12,
            'max_angular_reversal_accel': 6.0,
            'turn_sign_lock_margin': 0.04,
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
            'max_linear': 0.384,
            'max_angular': 1.30,
            'max_angular_accel': 2.8,
            'max_angular_reversal_accel': 6.0,
            'min_angular_speed': 0.18,
            'watchdog_timeout': 0.35,
        }],
    )
    nav_timer = TimerAction(period=15.0, actions=[custom_nav, cmd_guard])

    # 2.5 Laser Emergency Stopper
    laser_stopper = Node(
        package='exploration',
        executable='laser_emergency_stopper',
        name='laser_emergency_stopper',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'check_radius': 0.30,
            'min_points_for_block': 2,
            'max_continuous_move_time': 99999.0,
            'max_vel_x': 0.312,
            'forced_rest_time': 0.3,
            'monitor_sector_center_deg': 180.0,
            'monitor_sector_width_deg': 100.0,
            'ignore_sector_centers_deg': [0.0],
            'ignore_sector_width_deg': 0.0,
            'ignore_sector_margin_deg': 0.0,
        }]
    )
    stopper_timer = TimerAction(period=15.0, actions=[laser_stopper])

    # 3. Frontier Explorer
    explorer = Node(
        package='exploration',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[
            explore_params,
            {
                'enable_return_home': True,
                # Disable random waypoints after map completion, so robot stays after returning.
                'num_random_goals': 0,
            },
        ],
    )
    explorer_timer = TimerAction(period=20.0, actions=[explorer])

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
    recorder_timer = TimerAction(period=2.0, actions=[recorder])

    # 4. RViz (optional)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(rviz_flag),
    )

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
        save_state_arg,
        disable_map_save_arg,
        disable_shm,
        bringup,
        recorder_timer,
        nav_timer,
        stopper_timer,
        explorer_timer,
        rviz,
    ])
