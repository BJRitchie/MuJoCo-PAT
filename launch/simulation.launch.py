"""Launch the MuJoCo simulation node only."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    sim_cfg   = PathJoinSubstitution([FindPackageShare("pat_simulation"), "config", "simulation.yaml"])
    model_xml = PathJoinSubstitution([FindPackageShare("pat_simulation"), "models",
                                      "environment", "air_bearing_table.xml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(package="pat_simulation", executable="simulation_node",
             name="pat_simulation",
             parameters=[sim_cfg, {"model_path": model_xml,
                                   "use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
    ])
