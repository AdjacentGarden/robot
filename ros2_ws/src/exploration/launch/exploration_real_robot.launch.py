"""
Real robot: base drivers + SLAM Toolbox + custom navigator + frontier explorer.

Replaces the entire Nav2 stack with a single CustomNavigator node
(A* global planner + pure-pursuit local planner).

Components:
  1. Robot base drivers (controller, lidar, IMU)
  2. SLAM Toolbox (mapping + localization)
  3. CustomNavigator (A* + pure-pursuit — replaces Nav2)
  4. FrontierExplorer (sends NavigateToPose goals)
  5. RViz (optional)

Usage:
  ros2 launch exploration exploration_real_robot.launch.py
  ros2 launch exploration exploration_real_robot.launch.py rviz:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from nav2_common.launch import RewrittenYaml


def _default_log_root():
    return os.environ.get(
        'EXPLORATION_LOG_ROOT',
        os.path.join(os.path.expanduser('~'), '.ros', 'exploration_logs'),
    )


def generate_launch_description():
    # ── Package paths ──
    exploration_pkg = get_package_share_directory('exploration')

    compiled = 'True'
    if compiled == 'True':
        slam_pkg = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_pkg = os.path.join(home, 'ros2_ws/src/slam')

    # ── Config files ──
    nav_params = os.path.join(exploration_pkg, 'config', 'custom_nav_real_robot_params.yaml')
    explore_params = os.path.join(exploration_pkg, 'config', 'explore_params.yaml')
    slam_params = os.path.join(exploration_pkg, 'config', 'slam_real_robot_params.yaml')
    rviz_config = os.path.join(exploration_pkg, 'rviz', 'exploration.rviz')

    # ── Launch arguments ──
    sim_arg = DeclareLaunchArgument('sim', default_value='false')
    master_arg = DeclareLaunchArgument(
        'master_name', default_value=os.environ.get('MASTER', 'master'))
    robot_arg = DeclareLaunchArgument(
        'robot_name', default_value=os.environ.get('HOST', '/'))
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    record_arg = DeclareLaunchArgument('record_diagnostics', default_value='true')
    log_root_arg = DeclareLaunchArgument(
        'log_root',
        default_value=_default_log_root())

    sim = LaunchConfiguration('sim')
    master_name = LaunchConfiguration('master_name')
    robot_name = LaunchConfiguration('robot_name')
    rviz_flag = LaunchConfiguration('rviz')
    record_flag = LaunchConfiguration('record_diagnostics')
    log_root = LaunchConfiguration('log_root')

    # ── Disable FastDDS SHM (ARM64 crash fix) ──
    disable_shm = SetEnvironmentVariable(
        'RMW_FASTRTPS_USE_QOS_FROM_XML', '0')

    # 1. Robot base + SLAM
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'include', 'robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name,
        }.items(),
    )

    slam_rewritten = RewrittenYaml(
        source_file=slam_params,
        param_rewrites={
            'use_sim_time': 'false',
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
            'scan_topic': 'scan',
        },
        convert_types=True,
    )

    slam_launch = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_rewritten],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('/map', 'map'),
            ('/map_metadata', 'map_metadata'),
        ],
    )

    bringup = GroupAction(actions=[
        PushRosNamespace(robot_name),
        base_launch,
        TimerAction(period=5.0, actions=[slam_launch]),
    ])

    # 2. Custom Navigator (replaces ALL Nav2 nodes)
    custom_nav = Node(
        package='exploration',
        executable='custom_navigator',
        name='custom_navigator',
        output='screen',
        parameters=[nav_params],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('/cmd_vel', '/nav_cmd_vel')
        ],
    )

    nav_timer = TimerAction(period=15.0, actions=[custom_nav])
    
    # 2.5 Laser Emergency Stopper (intercepts /nav_cmd_vel and passes to /cmd_vel)
    laser_stopper = Node(
        package='exploration',
        executable='laser_emergency_stopper',
        name='laser_emergency_stopper',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'check_radius': 0.22,
            'min_points_for_block': 2,
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
        parameters=[explore_params],
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
        sim_arg, master_arg, robot_arg, rviz_arg, record_arg, log_root_arg,
        disable_shm,
        bringup,
        recorder_timer,
        nav_timer,
        stopper_timer,
        explorer_timer,
        rviz,
    ])
