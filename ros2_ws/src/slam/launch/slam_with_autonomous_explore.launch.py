import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import PushRosNamespace, Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction, TimerAction

def launch_setup(context):
    compiled = 'True'
    enable_save = LaunchConfiguration('enable_save', default='true').perform(context)
    sim = LaunchConfiguration('sim', default='false').perform(context)
    master_name = LaunchConfiguration('master_name', default=os.environ['MASTER']).perform(context)
    robot_name = LaunchConfiguration('robot_name', default=os.environ['HOST']).perform(context)
    
    # 新增：探索相关参�?
    exploration_radius = LaunchConfiguration('exploration_radius', default='4.0').perform(context)
    init_delay = LaunchConfiguration('init_delay', default='20.0').perform(context)

    enable_save_arg = DeclareLaunchArgument('enable_save', default_value=enable_save)
    sim_arg = DeclareLaunchArgument('sim', default_value=sim)
    master_name_arg = DeclareLaunchArgument('master_name', default_value=master_name)
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value=robot_name)
    exploration_radius_arg = DeclareLaunchArgument('exploration_radius', default_value=exploration_radius)
    init_delay_arg = DeclareLaunchArgument('init_delay', default_value=init_delay)

    frame_prefix = '' if robot_name == '/' else '%s/'%robot_name
    use_sim_time = 'true' if sim == 'true' else 'false'
    
    # 强制使用无前缀的TF frames（与原成功代码保持绝对一致）
    map_frame = 'map'
    odom_frame = 'odom'
    base_frame = 'base_footprint'
    
    # 话题需要考虑namespace前缀
    topic_prefix = '' if robot_name == '/' else '/%s'%robot_name

    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
        navigation_package_path = get_package_share_directory('navigation')
    else:
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
        navigation_package_path = os.path.join(home, 'ros2_ws/src/navigation')

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
            'use_sim_time': use_sim_time,
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'scan_topic': '{}/scan'.format(frame_prefix),
            'enable_save': enable_save
        }.items(),
    )

    # 计算cmd_vel话题（强制使用根namespace�?controller/cmd_vel�?
    cmd_vel_topic = '/controller/cmd_vel'
    
    scan_topic = '/scan' if robot_name == '/' else '{}/scan'.format(topic_prefix)
    scan_topic_candidates = ['/scan', 'scan']
    if topic_prefix:
        scan_topic_candidates.append('{}/scan'.format(topic_prefix))

    # 1. 探索专用局部路径规划节�?(Level 2 增强�?
    path_tracker_node = Node(
        package='navigation',
        executable='exploration_path_tracker',
        name='exploration_path_tracker',
        output='screen',
        parameters=[{
            'map_frame': map_frame,
            'base_frame': base_frame,
            'max_linear_vel': 0.1,
            'max_angular_vel': 0.3,
            'goal_tolerance': 0.05,
            'emergency_stop_dist': 0.12,
            'collision_dist': 0.06,
            'costmap_size': 2.0,
            'costmap_resolution': 0.05,
            'inflation_radius': 0.22,
            'horizon_sec': 1.2,
            'sim_dt': 0.1,
            'num_candidates': 11,
            'w_obstacle': 15.0,
            'w_goal': 5.0,
            'w_heading': 2.0,
            'w_smooth': 1.5,
            'stuck_timeout': 1.2,
            'stuck_distance': 0.08,
            'stuck_rotation_speed': 0.3,
            'enable_reverse': True,
            'scan_topic': scan_topic,
            'scan_topic_candidates': scan_topic_candidates,
        }],
        remappings=[
            ('cmd_vel', cmd_vel_topic),
        ]
    )
    
    # 2. 探索策略大脑节点
    explorer_node = Node(
        package='navigation',
        executable='autonomous_explorer',
        name='autonomous_explorer',
        output='screen',
        parameters=[{
            'exploration_radius': float(exploration_radius),
            'init_delay': float(init_delay),
            'frontier_check_interval': 1.0,
        }]
    )

    bringup_actions = [
        PushRosNamespace(robot_name),
        base_launch,
        TimerAction(
            period=5.0,
            actions=[slam_launch],
        ),
    ]
    
    bringup_launch = GroupAction(actions=bringup_actions)

    # 路径跟踪器在全局启动
    path_tracker_timer = TimerAction(
        period=10.0,
        actions=[path_tracker_node],
    )
    
    # Explorer可以在稍后启�?
    explorer_timer = TimerAction(
        period=12.0, 
        actions=[explorer_node],
    )

    return [
        sim_arg, master_name_arg, robot_name_arg, enable_save_arg, exploration_radius_arg, init_delay_arg,
        bringup_launch, path_tracker_timer, explorer_timer
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
