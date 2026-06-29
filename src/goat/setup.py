import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'goat'


def recursive_data_files(subdir):
    """Collect install (share path, [files]) tuples for a directory tree.

    Preserves the subdirectory layout under share/<package>/ so nested
    config/<robot>/*.xml and *.yaml are installed at the same relative path.
    """
    entries = []
    for path in glob(os.path.join(subdir, '**', '*'), recursive=True):
        if os.path.isfile(path):
            install_dir = os.path.join('share', package_name, os.path.dirname(path))
            entries.append((install_dir, [path]))
    return entries


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        *recursive_data_files('config'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='grape4314',
    maintainer_email='grape4314@gmail.com',
    description='MuJoCo simulator ROS2 node for the GOAT project.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'simulator_node = goat.nodes.simulator_node:main',
        ],
    },
)
