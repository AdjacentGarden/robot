import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    compiled = 'True'
    machine_type = os.environ['MACHINE_TYPE']
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        home = os.path.expanduser('~')
        peripherals_package_path = os.path.join(home, 'ros2_ws/src/peripherals')

    camera_launch = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(peripherals_package_path, 'launch/include/astra.launch.py')),
        ),
    ])

    return LaunchDescription([
        camera_launch
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
