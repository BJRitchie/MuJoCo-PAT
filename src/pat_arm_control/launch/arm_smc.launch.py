"""Launch the sliding-mode task-space controller — one node per arm (left + right).

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
(smc.yaml: the SMC's tuning and the per-arm blocks).

Another law can drive the other arm: the nodes only touch their own joints, so
this launch file can run beside arm_nmpc.launch.py if each is given one side
(drop the other Node action). Two laws on the SAME joints fight over
/chaser/arm/torque_command.

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
        [FindPackageShare("pat_arm_control"), "config", "smc.yaml"])
    mjcf_path = PathJoinSubstitution(
        [FindPackageShare("pat_arm_control"), "models", "pat_platform_planar.xml"])

    common = dict(package="pat_arm_control", executable="smc_node",
                  output="screen", emulate_tty=True)

    def arm(side: str) -> Node:
        return Node(name=f"pat_arm_smc_{side}", parameters=[
            LaunchConfiguration("common_params_file"),
            LaunchConfiguration("params_file"),
            {"mjcf_path": LaunchConfiguration("mjcf_path"),
             "trace_file": LaunchConfiguration("trace_file"),
             "odom_topic": LaunchConfiguration("odom_topic"),
             "use_sim_time": LaunchConfiguration("use_sim_time")}], **common)

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("common_params_file", default_value=common_params_file),
        DeclareLaunchArgument("params_file", default_value=params_file),
        DeclareLaunchArgument("mjcf_path", default_value=mjcf_path),
        # Per-tick timing trace, off when empty. Each node splices its own side
        # into the filename, so both arms can share this one value.
        DeclareLaunchArgument("trace_file", default_value=""),
        DeclareLaunchArgument("odom_topic", default_value="/chaser/odom"),
        arm("left"),
        arm("right"),
    ])
