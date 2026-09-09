"""Launch the constrained task-space NMPC — one node per arm (left + right).

Expects the simulation (or real arm hardware) to already be running. Each node
needs /chaser/arm/joint_states and /chaser/odom, and publishes
/chaser/arm/torque_command. Task-space targets go on
/chaser/arm/<side>/ee_setpoint (geometry_msgs/PoseStamped, planar); with no
setpoint each arm holds its start pose.

NOTE: the controller models a FREE-FLOATING base (generalized-Jacobian
reaction dynamics). Run it against an undriven chaser — do NOT also run GNC
station-keeping, or the arm reaction is double-counted.

Swap tuning sets with `params_file:=/path/to/other.yaml`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = PathJoinSubstitution(
        [FindPackageShare("pat_arm_nmpc"), "config", "nmpc.yaml"])
    mjcf_path = PathJoinSubstitution(
        [FindPackageShare("pat_arm_nmpc"), "models", "pat_platform_planar.xml"])

    common = dict(package="pat_arm_nmpc", executable="arm_nmpc_node",
                  output="screen", emulate_tty=True)

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=params_file),
        DeclareLaunchArgument("mjcf_path", default_value=mjcf_path),
        Node(name="pat_arm_nmpc_left", parameters=[
            LaunchConfiguration("params_file"),
            {"mjcf_path": LaunchConfiguration("mjcf_path"),
             "use_sim_time": LaunchConfiguration("use_sim_time")}], **common),
        Node(name="pat_arm_nmpc_right", parameters=[
            LaunchConfiguration("params_file"),
            {"mjcf_path": LaunchConfiguration("mjcf_path"),
             "use_sim_time": LaunchConfiguration("use_sim_time")}], **common),
    ])
