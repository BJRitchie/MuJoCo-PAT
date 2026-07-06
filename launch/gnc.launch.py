"""Launch the GNC node. Expects either simulation or real hardware already running."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    gnc_cfg = PathJoinSubstitution([FindPackageShare("pat_gnc"), "config", "pid.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(package="pat_gnc", executable="gnc_node", name="pat_gnc",
             parameters=[gnc_cfg, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
    ])
