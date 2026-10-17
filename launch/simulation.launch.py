"""Launch the MuJoCo simulation node only.

`sensor_profile:=/path/to/overlay.yaml` loads a sensor configuration on top of
simulation.yaml, overriding only what it names -- e.g. pat_simulation's
config/sensors_flight.yaml for flight-representative sensing. Empty (the
default) leaves simulation.yaml's perfect sensors.

`visualise:=false` runs headless.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _sim_node(context):
    sim_cfg = PathJoinSubstitution([FindPackageShare("pat_simulation"), "config", "simulation.yaml"])
    model_xml = PathJoinSubstitution([FindPackageShare("pat_simulation"), "models",
                                      "environment", "air_bearing_table.xml"])
    # Later entries override earlier ones, so the profile goes after the base
    # file and before the launch-level overrides.
    params = [sim_cfg]
    profile = LaunchConfiguration("sensor_profile").perform(context)
    if profile:
        params.append(profile)
    params.append({"model_path": model_xml,
                   "visualise": LaunchConfiguration("visualise"),
                   "use_sim_time": LaunchConfiguration("use_sim_time")})
    return [Node(package="pat_simulation", executable="simulation_node",
                 name="pat_simulation", parameters=params,
                 output="screen", emulate_tty=True)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        # false runs headless: no viewer window, so nothing typed into it can
        # pause the simulation or toggle its rendering mid-run.
        DeclareLaunchArgument("visualise", default_value="true"),
        DeclareLaunchArgument("sensor_profile", default_value=""),
        OpaqueFunction(function=_sim_node),
    ])
