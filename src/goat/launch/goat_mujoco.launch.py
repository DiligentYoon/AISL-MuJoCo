from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot = LaunchConfiguration('robot')
    model_path = LaunchConfiguration('model_path')
    config_file = LaunchConfiguration('config_file')

    pkg = FindPackageShare('goat')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot',
            default_value='goat',
            description='Robot name; selects config/<robot>/.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([pkg, 'config', robot, 'fixed_goat.xml']),
            description='MJCF model path (defaults to config/<robot>/fixed_goat.xml).',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([pkg, 'config', robot, 'goat_mujoco.yaml']),
            description='YAML parameter file for the simulator node.',
        ),
        Node(
            package='goat',
            executable='goat_mujoco_node',
            name='goat_mujoco_node',
            output='screen',
            parameters=[config_file, {'model_path': model_path}],
        ),
    ])
