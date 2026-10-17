"""Launch the base state estimator (pat_gnc nav_node) only.

Expects the simulation (or the IMU, star tracker and relative-nav drivers) to
already be running: it reads /chaser/imu and /chaser/odom and publishes the
fused estimate on /chaser/nav/odom. Commands nothing, so unlike gnc.launch.py
it is safe alongside pat_arm_control.

Swap the estimator configuration with `nav_params:=/path/to/nav.yaml`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    nav_cfg = PathJoinSubstitution([FindPackageShare("pat_gnc"), "config", "nav.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("nav_params", default_value=nav_cfg),
        Node(package="pat_gnc", executable="nav_node", name="pat_nav",
             parameters=[LaunchConfiguration("nav_params"),
                         {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
    ])
