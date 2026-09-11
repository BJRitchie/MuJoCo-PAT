"""Simulation + dual-arm task-space NMPC, with a FREE-FLOATING chaser.

pat_arm_nmpc's ArmNMPC models a free base (generalized-Jacobian reaction
dynamics), so the chaser is left undriven — its base drifts under arm
reaction, which is exactly what the controller expects. GNC station-keeping
is intentionally NOT included: thrusting to hold the chaser still would make
the NMPC double-count the arm→base reaction. Set `with_gnc:=true` only if you
know you want that coupling.

`with_targets:=true` also brings up ee_target_publisher, which walks each arm
through a fixed loop of known-reachable EE setpoints (otherwise the arms just
hold their start pose).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, ThisLaunchFileDir
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}.items()
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("with_gnc", default_value="false"),
        DeclareLaunchArgument("with_targets", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "simulation.launch.py"])),
            launch_arguments=use_sim_time),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "gnc.launch.py"])),
            launch_arguments=use_sim_time,
            condition=IfCondition(LaunchConfiguration("with_gnc"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("pat_arm_nmpc"), "launch", "arm_nmpc.launch.py"])),
            launch_arguments=use_sim_time),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("pat_arm_nmpc"), "launch", "ee_targets.launch.py"])),
            launch_arguments=use_sim_time,
            condition=IfCondition(LaunchConfiguration("with_targets"))),
    ])
