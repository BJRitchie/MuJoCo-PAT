"""Launch simulation + GNC together."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, ThisLaunchFileDir

def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "simulation.launch.py"])),
            launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([ThisLaunchFileDir(), "gnc.launch.py"])),
            launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items()),
    ])
