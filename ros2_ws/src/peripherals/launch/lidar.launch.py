import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription, LaunchService
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    compiled = 'True'
    lidar_type = os.environ['LIDAR_TYPE']
    lidar_frame = LaunchConfiguration('lidar_frame', default='lidar_frame')
    scan_raw = LaunchConfiguration('scan_raw', default='scan_raw')
    scan_topic = LaunchConfiguration('scan_topic', default='scan')
    scan_pre_mask = LaunchConfiguration('scan_pre_mask', default='scan_pre_mask')

    lidar_frame_arg = DeclareLaunchArgument('lidar_frame', default_value=lidar_frame)
    scan_raw_arg = DeclareLaunchArgument('scan_raw', default_value=scan_raw)
    scan_topic_arg = DeclareLaunchArgument('scan_topic', default_value=scan_topic)
    scan_pre_mask_arg = DeclareLaunchArgument('scan_pre_mask', default_value=scan_pre_mask)

    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
    else:
        home = os.path.expanduser('~')
        peripherals_package_path = os.path.join(home, 'ros2_ws/src/peripherals')

    if lidar_type == 'LD19':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ldlidar_LD19.launch.py')
    elif lidar_type == 'A1':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/sllidar_a1.launch.py')
    elif lidar_type == 'C1':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/sllidar_c1.launch.py')
    elif lidar_type == 'C1':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_c1.yaml')
    elif lidar_type == 'G4':
        lidar_launch_path = os.path.join(peripherals_package_path, 'launch/include/ydlidar_g4.launch.py')

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch_path),
        launch_arguments={'lidar_frame': lidar_frame,
                          'scan_raw': scan_raw}.items())

    laser_filters_config = ''
    if lidar_type == 'A1':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_a1.yaml')
    elif lidar_type == 'C1':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_c1.yaml')
    elif lidar_type == 'G4':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_g4.yaml')
    elif lidar_type == 'LD14P':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_ld14p.yaml')
    elif lidar_type == 'LD19':
        laser_filters_config = os.path.join(peripherals_package_path, 'config/lidar_filters_config_ld19.yaml')
    laser_filter_node = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        output='screen',
        parameters=[laser_filters_config],
        remappings=[('scan', scan_raw),
                    ('scan_filtered', scan_pre_mask)]
    )

    scan_sector_mask_node = Node(
        package='peripherals',
        executable='scan_sector_mask',
        output='screen',
        parameters=[{
            'enabled': True,
            # Keep mode: preserve only configured usable sectors.
            # Current C1 mounting: lidar angle 0deg corresponds to base rear direction.
            # Kept sectors:
            # 1) base front 100deg -> lidar [-180,-130] U [130,180]
            # 2) rear usable wedges -> lidar [10,55] and [-55,-10]
            'mode': 'keep',
            'keep_sector_ranges_deg': [-180.0, -130.0, 130.0, 180.0, 10.0, 55.0, -55.0, -10.0],
            # Legacy mask params are kept for backward compatibility when mode='mask'.
            'sector_centers_deg': [0.0],
            'sector_width_deg': 260.0,
            'sector_margin_deg': 0.0,
            'replacement_value': float('inf'),
        }],
        remappings=[('scan_in', scan_pre_mask),
                    ('scan_out', scan_topic)]
    )

    return LaunchDescription([
        lidar_frame_arg,
        scan_raw_arg,
        scan_topic_arg,
        scan_pre_mask_arg,
        lidar_launch,
        laser_filter_node,
        scan_sector_mask_node,
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
