"""
Navigation on an existing map using a single localization chain.

Components:
  1. Robot base drivers
  2. nav2_map_server (known map)
  3. AMCL (continuous map->odom localization)
  4. AutoInitialPoseEstimator (local search around configured start pose)
    5. CustomNavigator (A* + pure-pursuit)
    6. LaserEmergencyStopper (/nav_cmd_vel -> /cmd_vel safety gate)
    7. MapGoalBridge (RViz goal -> NavigateToPose)
    8. RViz (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
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
    rviz_config = os.path.join(exploration_pkg, 'rviz', 'exploration.rviz')
    default_map = os.path.join(exploration_pkg, 'map', 'exploration_map.yaml')

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
    log_root_arg = DeclareLaunchArgument('log_root', default_value=_default_log_root())
    use_amcl_arg = DeclareLaunchArgument('use_amcl', default_value='false', description='Whether to use AMCL')

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
    use_amcl = LaunchConfiguration('use_amcl')

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

    bringup = GroupAction(actions=[
        PushRosNamespace(robot_name),
        base_launch,
    ])

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
        condition=IfCondition(use_amcl),
    )

    lifecycle_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
        condition=IfCondition(use_amcl),
    )

    lifecycle_no_amcl = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server'],
        }],
        condition=UnlessCondition(use_amcl),
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
        condition=UnlessCondition(use_amcl),
    )

    amcl_auto_relocalizer = Node(
        package='exploration',
        executable='amcl_auto_relocalizer',
        name='amcl_auto_relocalizer',
        output='screen',
        parameters=[{
            'enable_motion_assist': False,
            'lost_timeout': 999999.0,
            'retry_cooldown': 999999.0,
        }],
        condition=IfCondition(use_amcl),
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
            'search_x_range': 0.08,
            'search_y_range': 0.08,
            'search_yaw_range_deg': 12.0,
            'search_xy_step': 0.01,
            'search_yaw_step_deg': 1.0,
            'coarse_xy_step': 0.02,
            'coarse_yaw_step_deg': 2.0,
            'startup_scan_duration': 2.0,
            'startup_scan_linear_speed': 0.0,
            'startup_scan_angular_speed': 0.12,
            'startup_min_translation': 999.0,
            'startup_min_rotation_deg': 6.0,
            'center_bias_xy_weight': 8.0,
            'center_bias_yaw_weight': 2.0,
            'publish_repeats': 5,
            'publish_interval': 0.25,
            'min_match_ratio': 0.10,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
        condition=IfCondition(use_amcl),
    )

    custom_nav = Node(
        package='exploration',
        executable='custom_navigator',
        name='custom_navigator',
        output='screen',
        parameters=[nav_params, {
            'use_open_loop_pose': True,
            'open_loop_tf_stall_timeout': 0.8,
            'open_loop_min_cmd_linear': 0.04,
            'open_loop_min_cmd_angular': 0.08,
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static'), ('/cmd_vel', '/nav_cmd_vel')],
    )

    laser_stopper = Node(
        package='exploration',
        executable='laser_emergency_stopper',
        name='laser_emergency_stopper',
        output='screen',
        parameters=[{
            'scan_topic': '/scan',
            'check_radius': 0.26,
            'min_points_for_block': 2,
            'max_continuous_move_time': 99999.0,  # 设为极大值以取消走停逻辑
            'emergency_timeout': 3.0,             # 触发急停3秒后解除限制，允许脱困回退等动作
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

    localization_timer = TimerAction(period=3.0, actions=[map_server, amcl, lifecycle_amcl, lifecycle_no_amcl, static_tf])
    relocalizer_timer = TimerAction(period=4.0, actions=[amcl_auto_relocalizer])
    precise_pose_timer = TimerAction(period=5.0, actions=[auto_initial_pose_estimator])
    stopper_timer = TimerAction(period=8.5, actions=[laser_stopper])
    navigator_timer = TimerAction(period=9.0, actions=[custom_nav, goal_bridge])
    recorder_timer = TimerAction(period=2.0, actions=[recorder])

    return LaunchDescription([
        sim_arg,
        master_arg,
        robot_arg,
        rviz_arg,
        map_arg,
        initial_pose_x_arg,
        initial_pose_y_arg,
        initial_pose_yaw_arg,
        record_arg,
        log_root_arg,
        use_amcl_arg,
        disable_shm,
        bringup,
        localization_timer,
        relocalizer_timer,
        precise_pose_timer,
        recorder_timer,
        stopper_timer,
        navigator_timer,
        rviz,
    ])
