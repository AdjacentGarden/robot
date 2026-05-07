import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import PushRosNamespace, Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction, TimerAction

def launch_setup(context):
    compiled = 'True'
    
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
        navigation_package_path = get_package_share_directory('navigation')
    else:
        # 使用用户主目录而不是硬编码/root
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
        navigation_package_path = os.path.join(home, 'ros2_ws/src/navigation')

    # 启动参数
    sim = LaunchConfiguration('sim', default='false').perform(context)
    robot_name = LaunchConfiguration('robot_name', default=os.environ['HOST']).perform(context)
    master_name = LaunchConfiguration('master_name', default=os.environ['MASTER']).perform(context)
    enable_save = LaunchConfiguration('enable_save', default='true').perform(context)
    enable_explore = LaunchConfiguration('enable_explore', default='true').perform(context)

    # 声明参数
    sim_arg = DeclareLaunchArgument('sim', default_value=sim)
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value=robot_name)
    master_name_arg = DeclareLaunchArgument('master_name', default_value=master_name)
    enable_save_arg = DeclareLaunchArgument('enable_save', default_value=enable_save)
    enable_explore_arg = DeclareLaunchArgument('enable_explore', default_value=enable_explore)

    # 计算参数
    frame_prefix = '' if robot_name == '/' else '%s/'%robot_name
    use_sim_time = 'true' if sim == 'true' else 'false'
    map_frame = '{}map'.format(frame_prefix)
    odom_frame = '{}odom'.format(frame_prefix)
    base_frame = '{}base_footprint'.format(frame_prefix)
    use_namespace = 'true' if robot_name != '/' else 'false'

    # 启动基础驱动
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_package_path, 'launch/include/robot.launch.py')),
        launch_arguments={
            'sim': sim,
            'master_name': master_name,
            'robot_name': robot_name
        }.items(),
    )

    # 启动SLAM建图
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

    # 启动Nav2导航栈（极简版，只有规划和控制）
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_package_path, 'launch', 'nav2_minimal.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(navigation_package_path, 'config', 'nav2_params_minimal.yaml'),
            'namespace': robot_name,
        }.items(),
    )

    # 启动自主探索节点（Explore Lite）
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

    # 构建启动列表
    bringup_actions = [
        PushRosNamespace(robot_name),
        base_launch,
        TimerAction(
            period=5.0,  # 等待基础驱动启动（参考slam.launch.py）
            actions=[slam_launch],
        ),
        TimerAction(
            period=15.0,  # 延迟15秒再启动Nav2，确保TF树完全建立
            actions=[navigation_launch],
        ),
    ]

    # 如果启用自主探索，添加explore节点（延后启动，等Nav2就绪）
    if enable_explore == 'true':
        explore_timer = TimerAction(
            period=35.0,  # 延后35秒启动explore，充分确保Nav2完全就绪
            actions=[explore_node],
        )
        bringup_actions.append(explore_timer)

    # 组织启动
    bringup_launch = GroupAction(actions=bringup_actions)

    return [
        sim_arg,
        robot_name_arg,
        master_name_arg,
        enable_save_arg,
        enable_explore_arg,
        bringup_launch
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
