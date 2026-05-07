import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

def launch_setup(context):
    use_sim_time = LaunchConfiguration('use_sim_time', default='false').perform(context)
    namespace = LaunchConfiguration('namespace', default='').perform(context)
    params_file = LaunchConfiguration('params_file').perform(context)
    
    # 极简Nav2：只启动必需的节点
    nav2_nodes = GroupAction([
        # 路径规划服务器
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time == 'true'}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
        ),
        
        # 控制器服务器
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time == 'true'}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
        ),
        
        # 行为服务器（恢复动作等）
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time == 'true'}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
        ),
        
        # BT导航器
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time == 'true'}],
            remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
        ),
        
        # 生命周期管理器
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time == 'true'},
                {'autostart': True},
                {'node_names': ['planner_server', 'controller_server', 'behavior_server', 'bt_navigator']}
            ]
        ),
    ])
    
    if namespace and namespace != '/':
        nav2_nodes = GroupAction([
            PushRosNamespace(namespace),
            nav2_nodes
        ])
    
    return [nav2_nodes]

def generate_launch_description():
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('navigation'), 
            'config', 'nav2_params_minimal.yaml'
        ),
        description='Full path to param file to load'
    )
    
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for navigation'
    )
    
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_params_file_cmd,
        declare_namespace_cmd,
        OpaqueFunction(function=launch_setup)
    ])
