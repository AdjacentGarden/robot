"""
Cartographer 2D mapping launch for a real robot using a single LaserScan topic.

This launch starts:
1) cartographer_node
2) cartographer_occupancy_grid_node
3) rviz2 (optional)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_topic = LaunchConfiguration('scan_topic')
    rviz = LaunchConfiguration('rviz')
    configuration_basename = LaunchConfiguration('configuration_basename')

    config_dir = PathJoinSubstitution([
        FindPackageShare('cartographer_ros'),
        'configuration_files',
    ])
    rviz_config = PathJoinSubstitution([
        FindPackageShare('cartographer_ros'),
        'configuration_files',
        'demo_2d.rviz',
    ])

    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value='scan')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true')
    configuration_basename_arg = DeclareLaunchArgument(
        'configuration_basename', default_value='c1_real_robot_2d.lua')

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', config_dir,
            '-configuration_basename', configuration_basename,
        ],
        remappings=[
            ('scan', scan_topic),
        ],
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'resolution': 0.05},
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        use_sim_time_arg,
        scan_topic_arg,
        rviz_arg,
        configuration_basename_arg,
        cartographer_node,
        occupancy_grid_node,
        rviz_node,
    ])
