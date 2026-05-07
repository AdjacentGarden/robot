import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace, Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction

def generate_launch_description():
    """轻量级SLAM + Nav2 + m-explore探索系统（最小内存占用）"""
    
    compiled = 'True'
    
    # Launch arguments
    sim = LaunchConfiguration('sim', default='false')
    master_name = LaunchConfiguration('master_name', default=os.environ.get('MASTER', 'master'))
    robot_name = LaunchConfiguration('robot_name', default=os.environ.get('HOST', '/'))
    
    frame_prefix = '' if robot_name == '/' else '%s/' % robot_name
    
    # TF frames
    map_frame = 'map'
    odom_frame = 'odom'
    base_frame = 'base_footprint'
    
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
    
    # 配置文件
    nav2_params_file = os.path.join(slam_package_path, 'config', 'nav2_params.yaml')
    explore_config = os.path.join(slam_package_path, 'config', 'explore.yaml')
    
    # 1. 机器人和SLAM
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_package_path, 'launch/include/robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name
        }.items(),
    )
    
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_package_path, 'launch/include/slam_base.launch.py')),
        launch_arguments={
            'use_sim_time': 'false',
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'scan_topic': '{}/scan'.format(frame_prefix),
            'enable_save': 'true'
        }.items(),
    )
    
    # 2. 最小Nav2栈（只保留核心节点）
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('/tf', 'tf'),
                    ('/tf_static', 'tf_static')]
    )
    
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('/tf', 'tf'),
                    ('/tf_static', 'tf_static')]
    )
    
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('/tf', 'tf'),
                    ('/tf_static', 'tf_static')]
    )
    
    # 移除了behavior_server, waypoint_follower, velocity_smoother来节省内存
    
    lifecycle_manager_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': [
                'controller_server',
                'planner_server',
                'bt_navigator'
            ]
        }]
    )
    
    # 3. m-explore
    explore_node = Node(
        package='explore_lite',
        executable='explore',
        name='explore',
        output='screen',
        parameters=[
            explore_config,
            {'use_sim_time': False}
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ]
    )
    
    # 组装
    bringup_group = GroupAction(
        actions=[
            PushRosNamespace(robot_name),
            base_launch,
            TimerAction(period=5.0, actions=[slam_launch]),
        ]
    )
    
    nav2_timer = TimerAction(
        period=12.0,
        actions=[
            controller_server,
            planner_server,
            bt_navigator,
            lifecycle_manager_navigation
        ]
    )
    
    explore_timer = TimerAction(
        period=18.0,
        actions=[explore_node]
    )
    
    return LaunchDescription([
        bringup_group,
        nav2_timer,
        explore_timer,
    ])
