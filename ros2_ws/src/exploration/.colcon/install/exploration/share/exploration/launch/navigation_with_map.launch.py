"""
Standalone navigation on an existing map.

Components:
  1. Robot base drivers
  2. nav2_map_server + AMCL localization
  3. CustomNavigator (A* + pure-pursuit)
  4. MapGoalBridge (RViz/simple goal -> NavigateToPose)
  5. RViz (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    exploration_pkg = get_package_share_directory('exploration')

    compiled = 'True'
    if compiled == 'True':
        slam_pkg = get_package_share_directory('slam')
    else:
        home = os.path.expanduser('~')
        slam_pkg = os.path.join(home, 'ros2_ws/src/slam')

    nav_params = os.path.join(exploration_pkg, 'config', 'custom_nav_params.yaml')
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

    sim = LaunchConfiguration('sim')
    master_name = LaunchConfiguration('master_name')
    robot_name = LaunchConfiguration('robot_name')
    rviz_flag = LaunchConfiguration('rviz')
    map_file = LaunchConfiguration('map')

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
        param_rewrites={'yaml_filename': map_file},
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
    )

    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    custom_nav = Node(
        package='exploration',
        executable='custom_navigator',
        name='custom_navigator',
        output='screen',
        parameters=[nav_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    goal_bridge = Node(
        package='exploration',
        executable='map_goal_bridge',
        name='map_goal_bridge',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(rviz_flag),
    )

    localization_timer = TimerAction(period=3.0, actions=[map_server, amcl, lifecycle])
    navigator_timer = TimerAction(period=6.0, actions=[custom_nav, goal_bridge])

    return LaunchDescription([
        sim_arg, master_arg, robot_arg, rviz_arg, map_arg,
        disable_shm,
        bringup,
        localization_timer,
        navigator_timer,
        rviz,
    ])
