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

`telemetry:=true` brings up pat_telemetry's telemetry_node (merges
joint_states/torque_command by name, publishes each side's EE distance from
its setpoint - see /chaser/telemetry/*).

`plotjuggler:=true` additionally launches PlotJuggler itself with the saved
layout (`src/pat_telemetry/config/telemetry.xml`) preloaded - a separate flag
from `telemetry` since the GUI is a viewing convenience, not something every
telemetry-publishing launch needs (e.g. a headless run, or a laptop viewing a
remote ROS 2 domain instead).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}.items()
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("with_gnc", default_value="false"),
        DeclareLaunchArgument("with_targets", default_value="false"),
        DeclareLaunchArgument("telemetry", default_value="false"),
        DeclareLaunchArgument("plotjuggler", default_value="false"),
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
        Node(package="pat_telemetry", executable="telemetry_node", output="screen",
             parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
             condition=IfCondition(LaunchConfiguration("telemetry"))),
        TimerAction(
            # Delayed, not started alongside everything else: PlotJuggler's
            # ROS2 Topic Subscriber only sees topics that already exist at
            # the moment it starts (it doesn't retroactively pick up ones
            # that appear later), so starting it in the same instant as
            # pat_simulation/pat_arm_nmpc/etc. is a race - some publishers
            # may not have come up yet.
            period=0.5,
            actions=[ExecuteProcess(
                # --start_streamer is documented (`plotjuggler --help`) to
                # auto-start a streaming plugin, but telemetry.xml's own
                # <previouslyLoaded_Streamer> tag takes priority when present
                # (removing that tag broke reconnection, so it's back) -
                # expect the "start previously used streaming plugin?" and
                # "select ROS topics" dialogs to still need one click each.
                cmd=["ros2", "run", "plotjuggler", "plotjuggler", "--layout",
                     PathJoinSubstitution([FindPackageShare("pat_telemetry"), "config", "telemetry.xml"]),
                     "--start_streamer", "ROS2 Topic Subscriber"],
                output="screen")],
            condition=IfCondition(LaunchConfiguration("plotjuggler"))),
    ])
