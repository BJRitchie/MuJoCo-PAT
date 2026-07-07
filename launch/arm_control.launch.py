"""Launch the arm control node. Expects simulation or real arm hardware already running."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description() -> LaunchDescription:
    arm_cfg = PathJoinSubstitution([FindPackageShare("pat_robotics"), "config", "pid.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(package="pat_robotics", executable="arm_control_node", name="pat_robotics",
             parameters=[arm_cfg, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
    ])
