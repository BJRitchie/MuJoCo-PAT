"""Fixed-waypoint task-space setpoint source for the NMPC arms.

A stand-in mission node: cycles each arm through a loop of known-reachable
end-effector poses on /chaser/arm/<side>/ee_setpoint. In the default
`relative` mode it reads /chaser/arm/<side>/ee_pose (published by
arm_nmpc_node) and offsets from there, so the targets are reachable by
construction. Run it alongside full_stack_nmpc.launch.py.

Swap the waypoint set with `ee_targets_params_file:=/path/to/other.yaml`.
NOT `params_file` -- full_stack_nmpc.launch.py includes both this file and
arm_nmpc.launch.py in the same launch tree, and launch's LaunchConfiguration
values are global across an entire include tree, not scoped per file. Both
launch files used to declare their own argument named `params_file`
(arm_nmpc.launch.py's defaulting to nmpc.yaml, this one's to
ee_targets.yaml) -- since arm_nmpc.launch.py is included first,
DeclareLaunchArgument's "keep the current value if one's already set"
behavior meant THIS file's own `params_file` silently resolved to
nmpc.yaml instead, and every ee_target_publisher_node parameter fell back to
its C++ code default. It went unnoticed because those defaults look almost
identical to ee_targets.yaml's own -- until someone actually tried to
override something (a single held waypoint, `loop: false`), which is
exactly what exposed it. A distinct argument name here is what actually
fixes it, not a workaround.
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
        DeclareLaunchArgument("ee_targets_params_file", default_value=params_file),
        Node(
            package="pat_arm_nmpc", executable="ee_target_publisher_node",
            name="ee_target_publisher", output="screen", emulate_tty=True,
            parameters=[
                LaunchConfiguration("ee_targets_params_file"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        ),
    ])
