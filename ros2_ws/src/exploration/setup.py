import os
from glob import glob
from setuptools import setup

package_name = 'exploration'
package_files = lambda pattern: [path for path in glob(pattern) if os.path.isfile(path)]

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), package_files('launch/*')),
        (os.path.join('share', package_name, 'config'), package_files('config/*')),
        (os.path.join('share', package_name, 'map'), package_files('map/*')),
        (os.path.join('share', package_name, 'rviz'), package_files('rviz/*')),
        (os.path.join('share', package_name, 'localization_test'), package_files('localization_test/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yang',
    maintainer_email='1270161395@qq.com',
    description='High-performance 2D frontier-based autonomous exploration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_explorer=exploration.frontier_explorer:main',
            'sim_environment=exploration.sim_environment:main',
            'custom_navigator=exploration.custom_navigator:main',
            'optimized_custom_navigator=exploration.optimized_custom_navigator:main',
            'cmd_vel_guard=exploration.cmd_vel_guard:main',
            'turn_diagnostics=exploration.turn_diagnostics:main',
            'map_goal_bridge=exploration.map_goal_bridge:main',
            'diagnostic_recorder=exploration.diagnostic_recorder:main',
            'amcl_auto_relocalizer=exploration.amcl_auto_relocalizer:main',
            'auto_initial_pose_estimator=exploration.auto_initial_pose_estimator:main',
            'laser_emergency_stopper=exploration.laser_emergency_stopper:main',
        ],
    },
)
