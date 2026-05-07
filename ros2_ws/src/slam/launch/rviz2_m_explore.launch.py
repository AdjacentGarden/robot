import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """启动RViz2查看m-explore探索过程"""
    
    compiled = 'True'
    
    if compiled == 'True':
        slam_package_path = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_package_path = os.path.join(home, 'ros2_ws/src/slam')
    
    # RViz配置文件路径
    rviz_config_file = os.path.join(slam_package_path, 'rviz', 'm_explore.rviz')
    
    # 如果配置文件不存在，使用默认配置
    if not os.path.exists(rviz_config_file):
        rviz_config_file = ''
    
    # RViz2节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file] if rviz_config_file else [],
        parameters=[{
            'use_sim_time': False  # RViz通常使用实际时间
        }]
    )
    
    return LaunchDescription([
        rviz_node
    ])
