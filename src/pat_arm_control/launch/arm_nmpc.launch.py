"""Launch the constrained task-space NMPC — one node per arm (left + right).

Expects the simulation (or real arm hardware) to already be running. Each node
needs /chaser/arm/joint_states and /chaser/odom, and publishes
/chaser/arm/torque_command. Task-space targets go on
/chaser/arm/<side>/ee_setpoint (geometry_msgs/PoseStamped, planar); with no
setpoint each arm holds its start pose.

NOTE: the controller models a FREE-FLOATING base (generalized-Jacobian
reaction dynamics). Run it against an undriven chaser — do NOT also run GNC
station-keeping, or the arm reaction is double-counted.

Each node loads two parameter files, in order: `common_params_file`
(arm_common.yaml: the settings every control law shares) and `params_file`
(nmpc.yaml: the NMPC's tuning and the per-arm blocks). Swap the NMPC tuning
set with `params_file:=/path/to/other.yaml`.

`odom_topic:=/chaser/nav/odom` reads the base state from pat_gnc's nav_node
estimate instead of the /chaser/odom measurement.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    common_params_file = PathJoinSubstitution(
        [FindPackageShare("pat_arm_control"), "config", "arm_common.yaml"])
    params_file = PathJoinSubstitution(
        [FindPackageShare("pat_arm_control"), "config", "nmpc.yaml"])
    mjcf_path = PathJoinSubstitution(
        [FindPackageShare("pat_arm_control"), "models", "pat_platform_planar.xml"])

    common = dict(package="pat_arm_control", executable="nmpc_node",
                  output="screen", emulate_tty=True)

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("common_params_file", default_value=common_params_file),
        DeclareLaunchArgument("params_file", default_value=params_file),
        DeclareLaunchArgument("mjcf_path", default_value=mjcf_path),
        # Per-tick timing trace, off when empty. Each node splices its own side
        # into the filename, so both arms can share this one value.
        DeclareLaunchArgument("trace_file", default_value=""),
        DeclareLaunchArgument("odom_topic", default_value="/chaser/odom"),
        Node(name="pat_arm_nmpc_left", parameters=[
            LaunchConfiguration("common_params_file"),
            LaunchConfiguration("params_file"),
            {"mjcf_path": LaunchConfiguration("mjcf_path"),
             "trace_file": LaunchConfiguration("trace_file"),
             "odom_topic": LaunchConfiguration("odom_topic"),
             "use_sim_time": LaunchConfiguration("use_sim_time")}], **common),
        Node(name="pat_arm_nmpc_right", parameters=[
            LaunchConfiguration("common_params_file"),
            LaunchConfiguration("params_file"),
            {"mjcf_path": LaunchConfiguration("mjcf_path"),
             "trace_file": LaunchConfiguration("trace_file"),
             "odom_topic": LaunchConfiguration("odom_topic"),
             "use_sim_time": LaunchConfiguration("use_sim_time")}], **common),
    ])
