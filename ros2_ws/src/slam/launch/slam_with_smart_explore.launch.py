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
    enable_explore = LaunchConfiguration('enable_explore', default='true').perform(context)

    enable_save_arg = DeclareLaunchArgument('enable_save', default_value=enable_save)
    sim_arg = DeclareLaunchArgument('sim', default_value=sim)
    master_name_arg = DeclareLaunchArgument('master_name', default_value=master_name)
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value=robot_name)
    enable_explore_arg = DeclareLaunchArgument('enable_explore', default_value=enable_explore)

    frame_prefix = '' if robot_name == '/' else '%s/'%robot_name
    use_sim_time = 'true' if sim == 'true' else 'false'
    
    # 修复：强制使用无前缀的TF frames（因为实际系统不使用namespace）
    # 即使robot_name='jetson'，TF frames仍然是map, odom, base_footprint
    map_frame = 'map'
    odom_frame = 'odom'
    base_frame = 'base_footprint'
    
    # 但话题需要考虑namespace前缀
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

    # 计算cmd_vel话题（强制使用根namespace的/controller/cmd_vel）
    cmd_vel_topic = '/controller/cmd_vel'

    # 智能路径跟踪节点（带动态避障）
    # 节点内部已订阅 move_base_simple/goal 和 /goal_pose，无需remapping
    path_tracker_node = Node(
        package='navigation',
        executable='smart_path_tracker',  # 使用新的智能跟踪器
        name='smart_path_tracker',
        output='screen',
        parameters=[{
            'map_frame': map_frame,  # map (无前缀)
            'base_frame': base_frame,  # base_footprint (无前缀)
            'max_linear_vel': 0.3,
            'max_angular_vel': 0.5,
            'goal_tolerance': 0.2,
            'safe_dist': 0.6,   # 避障触发距离
            'clear_dist': 0.8,  # 避障解除距离
            'emergency_stop_dist': 0.2,  # 紧急停止距离
            'scan_topic': '/jetson/scan',  # 雷达话题
        }],
        remappings=[
            ('cmd_vel', cmd_vel_topic),  # 发布到 /controller/cmd_vel
        ]
    )

    # 构建启动列表（模仿成功的slam.launch.py结构）
    bringup_actions = [
        PushRosNamespace(robot_name),  # 关键！必须在最前面
        base_launch,
        TimerAction(
            period=5.0,
            actions=[slam_launch],
        ),
    ]
    
    # 路径跟踪器需要在namespace外启动（因为要接收全局/move_base_simple/goal）
    # 但frame参数已经包含了namespace前缀
    path_tracker_timer = TimerAction(
        period=10.0,
        actions=[path_tracker_node],
    )
    
    # 如果启用explore（只有在enable时才创建节点，避免找不到包的错误）
    if enable_explore == 'true':
        # Explore节点
        explore_node = Node(
            package='explore_lite',
            executable='explore',
            name='explore',
            parameters=[{
                'use_sim_time': use_sim_time == 'true',
                'max_frontier_size': 100,
                'min_frontier_size': 5,
                'potential_scale': 1e-3,
                'gain_scale': 1.0,
                'min_dist_potential_range': 0.5,
                'max_dist_potential_range': 20.0,
                'min_yaaw_velocity': 0.1,
                'max_yaaw_velocity': 1.0,
                'min_linear_velocity': 0.1,
                'max_linear_velocity': 0.5,
                'visualize': True,
            }],
            remappings=[
                ('map', 'map'),
                ('scan', 'scan'),
                ('odom', 'odom'),
            ]
        )
        
        explore_timer = TimerAction(
            period=20.0,  # 延后20秒启动explore，让SLAM先建立地图
            actions=[explore_node],
        )
    
    # 组织启动
    bringup_launch = GroupAction(actions=bringup_actions)
    
    # 构建返回列表
    return_list = [sim_arg, master_name_arg, robot_name_arg, enable_save_arg, enable_explore_arg, bringup_launch, path_tracker_timer]
    
    if enable_explore == 'true':
        return_list.append(explore_timer)
    
    return return_list

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
