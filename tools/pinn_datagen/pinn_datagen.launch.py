"""One PINN trajectory episode: sim + both arm_nmpc nodes + ee_target_publisher
(a single fixed absolute target per arm) + the recorder, wired so the whole
tree shuts down cleanly the moment the recorder finishes.

Not included via full_stack_nmpc.launch.py -- that file doesn't expose
mjcf_path/params_file as its own launch arguments, and this needs several
more (trial_params_file, run_time_s, out_path) besides. Composed directly
from the same building blocks instead (simulation.launch.py's Node,
arm_nmpc.launch.py's two-instance pattern, ee_targets.launch.py's publisher),
same idiom throughout: a base params file, then a small inline override dict
(later entries win on shared keys).

`trial_params_file` (required, no default) is written fresh per attempt by
generate_pinn_trajectory.py -- one small YAML carrying BOTH
pat_simulation's `initial_arm_qpos` (this trial's randomized, prefiltered
init config) and ee_target_publisher's `left.waypoints`/`right.waypoints`
(this trial's FK-computed target pose per arm). ROS 2 params files support
multiple node namespaces in one file, so one YAML covers both nodes.

pinn_recorder_node.py is a loose script, not an installed ament executable
-- it launches via ExecuteProcess with the same --ros-args -p name:=value
CLI convention any C++ node accepts, not launch_ros.actions.Node (which
resolves package/executable through the ament index, not arbitrary file
paths). OnProcessExit's target_action works identically on either.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDER_SCRIPT = os.path.join(THIS_DIR, "pinn_recorder_node.py")


def generate_launch_description() -> LaunchDescription:
    sim_cfg = PathJoinSubstitution([FindPackageShare("pat_simulation"), "config", "simulation.yaml"])
    model_path_default = PathJoinSubstitution(
        [FindPackageShare("pat_simulation"), "models", "environment", "air_bearing_table.xml"])
    nmpc_params_default = PathJoinSubstitution([FindPackageShare("pat_arm_nmpc"), "config", "nmpc.yaml"])
    mjcf_path_default = PathJoinSubstitution(
        [FindPackageShare("pat_arm_nmpc"), "models", "pat_platform_planar.xml"])

    use_sim_time = LaunchConfiguration("use_sim_time")
    trial_params_file = LaunchConfiguration("trial_params_file")

    sim_node = Node(
        package="pat_simulation", executable="simulation_node", name="pat_simulation",
        parameters=[sim_cfg, trial_params_file,
                    {"model_path": LaunchConfiguration("model_path"), "use_sim_time": use_sim_time,
                     "visualise": LaunchConfiguration("visualise")}],
        output="screen", emulate_tty=True,
    )

    nmpc_common = dict(package="pat_arm_nmpc", executable="arm_nmpc_node",
                        output="screen", emulate_tty=True)
    nmpc_left = Node(name="pat_arm_nmpc_left", parameters=[
        LaunchConfiguration("nmpc_params_file"),
        {"mjcf_path": LaunchConfiguration("mjcf_path"), "use_sim_time": use_sim_time}], **nmpc_common)
    nmpc_right = Node(name="pat_arm_nmpc_right", parameters=[
        LaunchConfiguration("nmpc_params_file"),
        {"mjcf_path": LaunchConfiguration("mjcf_path"), "use_sim_time": use_sim_time}], **nmpc_common)

    # mode/loop/etc. supplied entirely inline, NOT config/ee_targets.yaml --
    # that file's default `mode: relative` would conflict with the single
    # fixed absolute target this episode wants. trial_params_file supplies
    # only left.waypoints/right.waypoints on top of this.
    ee_target = Node(
        package="pat_arm_nmpc", executable="ee_target_publisher_node", name="ee_target_publisher",
        parameters=[trial_params_file,
                    {"sides": ["left", "right"], "mode": "absolute", "loop": False,
                     "republish_hz": 5.0, "frame_id": "map", "use_sim_time": use_sim_time}],
        output="screen", emulate_tty=True,
    )

    recorder = ExecuteProcess(
        cmd=["python3", RECORDER_SCRIPT, "--ros-args",
             "-p", ["run_time_s:=", LaunchConfiguration("run_time_s")],
             "-p", ["out_path:=", LaunchConfiguration("out_path")]],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        # Headless by default -- this is a batch tool (usually run unattended,
        # often without a GLX/DRM device at all); pass visualise:=true to
        # watch one episode live.
        DeclareLaunchArgument("visualise", default_value="false"),
        DeclareLaunchArgument("model_path", default_value=model_path_default),
        DeclareLaunchArgument("nmpc_params_file", default_value=nmpc_params_default),
        DeclareLaunchArgument("mjcf_path", default_value=mjcf_path_default),
        DeclareLaunchArgument("trial_params_file"),  # required -- no default
        DeclareLaunchArgument("run_time_s", default_value="15.0"),
        DeclareLaunchArgument("out_path"),  # required -- no default
        sim_node, nmpc_left, nmpc_right, ee_target, recorder,
        RegisterEventHandler(OnProcessExit(
            target_action=recorder,
            on_exit=[LogInfo(msg="pinn_recorder finished — shutting down the episode"),
                     EmitEvent(event=Shutdown())],
        )),
    ])
