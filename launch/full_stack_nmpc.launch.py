"""Simulation + dual-arm task-space NMPC, with a FREE-FLOATING chaser.

pat_arm_control's ArmNMPC models a free base (generalized-Jacobian reaction
dynamics), so the chaser is left undriven — its base drifts under arm
reaction, which is exactly what the controller expects. GNC station-keeping
is intentionally NOT included: thrusting to hold the chaser still would make
the NMPC double-count the arm→base reaction. Set `with_gnc:=true` only if you
know you want that coupling.

`with_targets:=true` also brings up ee_target_publisher, which walks each arm
through a fixed loop of known-reachable EE setpoints (otherwise the arms just
hold their start pose). `target_pattern:=circle` or `rectangle` has both arms
follow that shape as a continuous trajectory instead, which the NMPC tracks
with a per-node reference (see ee_targets.yaml).

`telemetry:=true` brings up pat_telemetry's telemetry_node (merges
joint_states/torque_command by name, publishes each side's EE distance from
its setpoint - see /chaser/telemetry/*).

`params_file:=/path/to/nmpc.yaml` swaps the NMPC tuning set, forwarded to
arm_nmpc.launch.py. Defaults to pat_arm_control's installed config/nmpc.yaml.

`with_nav:=true` brings up pat_gnc's nav_node (the IMU-driven base state
estimator, config `nav_params:=`, default pat_gnc/config/nav.yaml) and points
both arm controllers at its /chaser/nav/odom instead of /chaser/odom.

`sensor_profile:=` overlays a sensor configuration on the simulation. Flight
sensing end to end is
  sensor_profile:=<pat_simulation share>/config/sensors_flight.yaml
  with_nav:=true nav_params:=<pat_gnc share>/config/nav_flight.yaml

`visualise:=false` runs the simulation headless.

`trace_file:=/path/to/trace.csv` records every control tick's timing and writes
it at exit, one file per arm (trace_left.csv / trace_right.csv). Off by default.
Plot it with tools/nmpc_timing/plot_trace.py.

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
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression, ThisLaunchFileDir)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}.items()
    # The arms read the estimate when nav_node is running, the measurement
    # otherwise -- never a topic nobody publishes.
    odom_topic = PythonExpression([
        "'/chaser/nav/odom' if '", LaunchConfiguration("with_nav"),
        "' == 'true' else '/chaser/odom'"])
    nmpc_args = {"use_sim_time": LaunchConfiguration("use_sim_time"),
                 "params_file": LaunchConfiguration("params_file"),
                 "trace_file": LaunchConfiguration("trace_file"),
                 "odom_topic": odom_topic}.items()
    nav_args = {"use_sim_time": LaunchConfiguration("use_sim_time"),
                "nav_params": LaunchConfiguration("nav_params")}.items()
    nav_params = PathJoinSubstitution([FindPackageShare("pat_gnc"), "config", "nav.yaml"])
    # Repeats arm_nmpc.launch.py's own default rather than forwarding an empty
    # string: launch_arguments is fixed when the description is built, so there is
    # no way to omit the argument conditionally on its value.
    nmpc_params = PathJoinSubstitution(
        [FindPackageShare("pat_arm_control"), "config", "nmpc.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=nmpc_params),
        DeclareLaunchArgument("trace_file", default_value=""),
        DeclareLaunchArgument("with_gnc", default_value="false"),
        DeclareLaunchArgument("with_nav", default_value="false"),
        DeclareLaunchArgument("nav_params", default_value=nav_params),
        DeclareLaunchArgument("with_targets", default_value="false"),
        DeclareLaunchArgument("target_pattern", default_value="waypoints"),
        DeclareLaunchArgument("telemetry", default_value="false"),
        DeclareLaunchArgument("plotjuggler", default_value="false"),
        DeclareLaunchArgument("visualise", default_value="true"),
        DeclareLaunchArgument("sensor_profile", default_value=""),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "simulation.launch.py"])),
            launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time"),
                              "visualise": LaunchConfiguration("visualise"),
                              "sensor_profile": LaunchConfiguration("sensor_profile")}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "gnc.launch.py"])),
            launch_arguments=use_sim_time,
            condition=IfCondition(LaunchConfiguration("with_gnc"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "nav.launch.py"])),
            launch_arguments=nav_args,
            condition=IfCondition(LaunchConfiguration("with_nav"))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("pat_arm_control"), "launch", "arm_nmpc.launch.py"])),
            launch_arguments=nmpc_args),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("pat_arm_control"), "launch", "ee_targets.launch.py"])),
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
            # pat_simulation/pat_arm_control/etc. is a race - some publishers
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
