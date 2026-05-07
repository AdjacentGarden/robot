import os
from glob import glob
from setuptools import setup

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='1270161395@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_path_tracker=navigation.simple_path_tracker:main',
            'smart_path_tracker=navigation.smart_path_tracker:main',
            'advanced_path_tracker=navigation.advanced_path_tracker:main',
            'exploration_path_tracker=navigation.exploration_path_tracker:main',
            'autonomous_explorer=navigation.autonomous_explorer:main',
            'simple_nav2_explorer=navigation.simple_nav2_explorer:main',
        ],
    },
)
