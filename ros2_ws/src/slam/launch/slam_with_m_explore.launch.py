import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace, Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, TimerAction

def generate_launch_description():
    compiled = 'True'
    
    # Launch arguments
    enable_save = LaunchConfiguration('enable_save', default='true')
    sim = LaunchConfiguration('sim', default='false')
    master_name = LaunchConfiguration('master_name', default=os.environ.get('MASTER', 'master'))
    robot_name = LaunchConfiguration('robot_name', default=os.environ.get('HOST', '/'))
    
    frame_prefix = '' if robot_name == '/' else '%s/' % robot_name
    
    # use_sim_time需要根据sim参数动态确定，但这里先设置默认值
    # 注意：LaunchConfiguration在运行时才解析，这里无法直接比较
    use_sim_time_str = 'false'  # 字符串形式，用于传递给其他launch文件
    use_sim_time_bool = False   # 布尔形式，用于m-explore节点
    
    # TF frames (无前缀)
    map_frame = 'map'
    odom_frame = 'odom'
    base_frame = 'base_footprint'
    
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
    
    # 基础launch（机器人驱动）
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_package_path, 'launch/include/robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name
        }.items(),
    )
    
    # SLAM launch
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_package_path, 'launch/include/slam_base.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time_str,
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'scan_topic': '{}/scan'.format(frame_prefix),
            'enable_save': enable_save
        }.items(),
    )
    
    # m-explore配置文件路径
    explore_config = os.path.join(slam_package_path, 'config', 'explore.yaml')
    
    # m-explore节点
    explore_node = Node(
        package='explore_lite',
        executable='explore',
        name='explore',
        output='screen',
        parameters=[
            explore_config,
            {'use_sim_time': use_sim_time_bool}  # 使用布尔值
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
    
    # m-explore稍后启动（等SLAM初始化）
    explore_timer = TimerAction(
        period=15.0,
        actions=[explore_node]
    )
    
    return LaunchDescription([
        bringup_group,
        explore_timer,
    ])
