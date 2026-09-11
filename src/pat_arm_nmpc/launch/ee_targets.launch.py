"""Fixed-waypoint task-space setpoint source for the NMPC arms.

A stand-in mission node: cycles each arm through a loop of known-reachable
end-effector poses on /chaser/arm/<side>/ee_setpoint. In the default
`relative` mode it reads /chaser/arm/<side>/ee_pose (published by
arm_nmpc_node) and offsets from there, so the targets are reachable by
construction. Run it alongside full_stack_nmpc.launch.py.

Swap the waypoint set with `params_file:=/path/to/other.yaml`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = PathJoinSubstitution(
        [FindPackageShare("pat_arm_nmpc"), "config", "ee_targets.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=params_file),
        Node(
            package="pat_arm_nmpc", executable="ee_target_publisher_node",
            name="ee_target_publisher", output="screen", emulate_tty=True,
            parameters=[
                LaunchConfiguration("params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
