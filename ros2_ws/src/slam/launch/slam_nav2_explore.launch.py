import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace, Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction, SetEnvironmentVariable

def generate_launch_description():
    """完整的SLAM + Nav2 + m-explore探索系统"""
    
    compiled = 'True'
    
    # Launch arguments
    sim = LaunchConfiguration('sim', default='false')
    master_name = LaunchConfiguration('master_name', default=os.environ.get('MASTER', 'master'))
    robot_name = LaunchConfiguration('robot_name', default=os.environ.get('HOST', '/'))
    
    frame_prefix = '' if robot_name == '/' else '%s/' % robot_name
    
    # TF frames (无前缀)
    map_frame = 'map'
    odom_frame = 'odom'
    base_frame = 'base_footprint'
    
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
    
    # 配置文件路径
    nav2_params_file = os.path.join(slam_package_path, 'config', 'nav2_params_jetson.yaml')
    explore_config = os.path.join(slam_package_path, 'config', 'explore.yaml')
    
    # ================================
    # 0. 禁用FastDDS共享内存传输（ARM64/Jetson上已知问题）
    # 在启动前设置，仅需要一个环境变量
    # ================================
    disable_shm_transport = SetEnvironmentVariable(
        'RMW_FASTRTPS_USE_QOS_FROM_XML', '0'
    )
    
    # ================================
    # 1. 机器人驱动和SLAM
    # ================================
    
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
    
    # ================================
    # 2. Nav2导航栈
    # ================================
    
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
    
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
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
    
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('/tf', 'tf'),
                    ('/tf_static', 'tf_static')]
    )
    
    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[('/cmd_vel', '/controller/cmd_vel'),
                    ('/cmd_vel_smoothed', '/cmd_vel_nav'),
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static')]
    )
    
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
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother'
            ]
        }]
    )
    
    # ================================
    # 3. m-explore探索节点
    # ================================
    
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
    
    # ================================
    # 组装Launch
    # ================================
    
    bringup_group = GroupAction(
        actions=[
            PushRosNamespace(robot_name),
            base_launch,
            TimerAction(period=5.0, actions=[slam_launch]),
        ]
    )
    
    # Nav2在SLAM启动后启动（延迟20秒确保/map已发布）
    nav2_timer = TimerAction(
        period=20.0,
        actions=[
            controller_server,
            planner_server,
            behavior_server,
            bt_navigator,
            waypoint_follower,
            velocity_smoother,
            lifecycle_manager_navigation
        ]
    )
    
    # m-explore最后启动（等Nav2就绪，延迟30秒）
    explore_timer = TimerAction(
        period=30.0,
        actions=[explore_node]
    )
    
    return LaunchDescription([
        disable_shm_transport,
        bringup_group,
        nav2_timer,
        explore_timer,
    ])
