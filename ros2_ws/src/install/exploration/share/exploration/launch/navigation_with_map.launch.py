"""
Navigation on an existing map with smooth path following.

Hybrid approach:
  Phase 1: AMCL + map_server for initial localization on known map
  Phase 2: After localization, switch to SLAM Toolbox for smooth TF updates
           (like exploration_real_robot) for accurate path following

Components:
  1. Robot base drivers
  2. nav2_map_server (provides known map for A* planning)
  3. AMCL (initial localization only, TF broadcast disabled after stabilization)
  4. SLAM Toolbox (provides smooth map->odom TF for path following)
  5. CustomNavigator (A* + pure-pursuit)
  6. MapGoalBridge (RViz goal -> NavigateToPose)
  7. RViz (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription,
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


def _find_repo_root() -> str:
    here = os.path.abspath(__file__)
    launch_dir = os.path.dirname(here)
    candidates = [
        os.path.abspath(os.path.join(launch_dir, '..', '..')),
        os.path.abspath(os.path.join(launch_dir, '..', '..', '..', '..', '..')),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'src', 'slam', 'launch', 'include', 'robot.launch.py')):
            return candidate
        if os.path.exists(os.path.join(candidate, 'slam', 'launch', 'include', 'robot.launch.py')):
            return candidate
    return candidates[0]


def _resolve_slam_pkg(compiled: str) -> str:
    if compiled == 'True':
        return get_package_share_directory('slam')

    home = os.path.expanduser('~')
    candidates = [
        os.path.join(_find_repo_root(), 'src', 'slam'),
        os.path.join(_find_repo_root(), 'slam'),
        os.path.join(home, 'ros2_ws', 'src', 'slam'),
    ]
    for candidate in candidates:
        robot_launch = os.path.join(candidate, 'launch', 'include', 'robot.launch.py')
        if os.path.exists(robot_launch):
            return candidate

    return candidates[0]


def generate_launch_description():
    exploration_pkg = get_package_share_directory('exploration')

    compiled = 'True'
    slam_pkg = _resolve_slam_pkg(compiled)

    nav_params = os.path.join(exploration_pkg, 'config', 'custom_nav_real_robot_params.yaml')
    amcl_params = os.path.join(exploration_pkg, 'config', 'amcl_params.yaml')
    slam_params = os.path.join(exploration_pkg, 'config', 'slam_real_robot_params.yaml')
    rviz_config = os.path.join(exploration_pkg, 'rviz', 'exploration.rviz')
    default_map = os.path.join(exploration_pkg, 'map', 'exploration_map.yaml')

    # Launch arguments
    sim_arg = DeclareLaunchArgument('sim', default_value='false')
    master_arg = DeclareLaunchArgument(
        'master_name', default_value=os.environ.get('MASTER', 'master'))
    robot_arg = DeclareLaunchArgument(
        'robot_name', default_value=os.environ.get('HOST', '/'))
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    map_arg = DeclareLaunchArgument('map', default_value=default_map)
    initial_pose_x_arg = DeclareLaunchArgument('initial_pose_x', default_value='0.0')
    initial_pose_y_arg = DeclareLaunchArgument('initial_pose_y', default_value='0.0')
    initial_pose_yaw_arg = DeclareLaunchArgument('initial_pose_yaw', default_value='0.0')
    record_arg = DeclareLaunchArgument('record_diagnostics', default_value='true')
    log_root_arg = DeclareLaunchArgument(
        'log_root',
        default_value=_default_log_root())

    sim = LaunchConfiguration('sim')
    master_name = LaunchConfiguration('master_name')
    robot_name = LaunchConfiguration('robot_name')
    rviz_flag = LaunchConfiguration('rviz')
    map_file = LaunchConfiguration('map')
    initial_pose_x = LaunchConfiguration('initial_pose_x')
    initial_pose_y = LaunchConfiguration('initial_pose_y')
    initial_pose_yaw = LaunchConfiguration('initial_pose_yaw')
    record_flag = LaunchConfiguration('record_diagnostics')
    log_root = LaunchConfiguration('log_root')

    disable_shm = SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '0')

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Robot base bringup
    # ══════════════════════════════════════════════════════════════════════════
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_pkg, 'launch', 'include', 'robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name,
        }.items(),
    )

    bringup = GroupAction(actions=[
        PushRosNamespace(robot_name),
        base_launch,
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: AMCL localization on known map (initial pose correction)
    # ══════════════════════════════════════════════════════════════════════════
    localization_params = RewrittenYaml(
        source_file=amcl_params,
        param_rewrites={
            'yaml_filename': map_file,
            'set_initial_pose': 'true',
            'initial_pose.x': initial_pose_x,
            'initial_pose.y': initial_pose_y,
            'initial_pose.yaw': initial_pose_yaw,
            # Enable TF broadcast for initial localization
            'tf_broadcast': 'true',
        },
        convert_types=True,
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
            # Keep this node startup-only to avoid runtime cmd_vel interference
            'enable_motion_assist': False,
            'lost_timeout': 999999.0,
            'retry_cooldown': 999999.0,
        }],
    )

    auto_initial_pose_estimator = Node(
        package='exploration',
        executable='auto_initial_pose_estimator',
        name='auto_initial_pose_estimator',
        output='screen',
        parameters=[{
            'initial_pose_x': initial_pose_x,
            'initial_pose_y': initial_pose_y,
            'initial_pose_yaw': initial_pose_yaw,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Switch to SLAM Toolbox for smooth TF (after AMCL stabilizes)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Shut down AMCL after startup localization.
    # This avoids dual map->odom TF publishers (AMCL + SLAM toolbox).
    shutdown_amcl = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'for i in $(seq 1 12); do '
            'ros2 lifecycle set /amcl shutdown && exit 0; '
            'sleep 1; '
            'done; '
            'exit 1'
        ],
        output='screen',
    )

    # SLAM Toolbox for smooth continuous TF updates
    # Publishes to /slam_map to avoid conflict with map_server's /map
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
            # Remap SLAM map to avoid conflict with known map from map_server
            ('/map', 'slam_map'),
            ('/map_metadata', 'slam_map_metadata'),
        ],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4: Navigation (uses known map for planning, SLAM TF for tracking)
    # ══════════════════════════════════════════════════════════════════════════
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
        parameters=[{'goal_topic': '/exploration/goal'}],
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

    # ══════════════════════════════════════════════════════════════════════════
    # Timeline:
    #   0s:  Robot base bringup
    #   2s:  Recorder
    #   3s:  map_server + AMCL + lifecycle (initial localization)
    #   4s:  amcl_auto_relocalizer
    #   5s:  auto_initial_pose_estimator
    #   8s:  AMCL relocalizer has finished startup convergence
    #   9s:  Start SLAM Toolbox (takes over TF publishing)
    #  11s:  Shutdown AMCL (ensure single map->odom publisher)
    #  12s:  custom_navigator + goal_bridge (ready for navigation)
    # ══════════════════════════════════════════════════════════════════════════
    localization_timer = TimerAction(period=3.0, actions=[map_server, amcl, lifecycle])
    relocalizer_timer = TimerAction(period=4.0, actions=[amcl_auto_relocalizer])
    precise_pose_timer = TimerAction(period=5.0, actions=[auto_initial_pose_estimator])
    shutdown_amcl_timer = TimerAction(period=11.0, actions=[shutdown_amcl])
    slam_timer = TimerAction(period=9.0, actions=[slam_launch])
    navigator_timer = TimerAction(period=12.0, actions=[custom_nav, goal_bridge])
    recorder_timer = TimerAction(period=2.0, actions=[recorder])

    return LaunchDescription([
        sim_arg, master_arg, robot_arg, rviz_arg, map_arg,
        initial_pose_x_arg, initial_pose_y_arg, initial_pose_yaw_arg,
        record_arg, log_root_arg,
        disable_shm,
        bringup,
        localization_timer,
        relocalizer_timer,
        precise_pose_timer,
        recorder_timer,
        shutdown_amcl_timer,
        slam_timer,
        navigator_timer,
        rviz,
    ])
