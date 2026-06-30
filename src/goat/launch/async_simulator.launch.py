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
            default_value='double_pendulum',
            description='Robot name; selects config/<robot>/.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution([pkg, 'config', robot, [robot, '.xml']]),
            description='MJCF model path (defaults to config/<robot>/<robot>.xml).',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([pkg, 'config', robot, 'async_simulator.yaml']),
            description='YAML parameter file for the timer-driven simulator node.',
        ),
        Node(
            package='goat',
            executable='async_simulator_node',
            name='async_simulator_node',
            output='screen',
            parameters=[config_file, {'model_path': model_path}],
        ),
    ])
