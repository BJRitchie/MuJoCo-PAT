# launch/display_robot.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():
    package_share_dir = get_package_share_directory('digital_twin_viz')

    urdf_path = os.path.join(package_share_dir, 'urdf', 'Platform_description.urdf')
    rviz_config_path = os.path.join(package_share_dir, 'rviz', 'digital_twin_config.rviz')

    # Read the URDF file contents
    with open(urdf_path, 'r') as urdf_file:
        robot_description = urdf_file.read()

    # rigid_body_tf_broadcaster needs a live mocap4r2 feed (mocap4r2_msgs)
    # to do anything -- without one connected it has nothing to subscribe
    # to and the package isn't even installed in this dev workspace, so it
    # crashes on import. Off by default so the URDF/RViz preview works with
    # no mocap hardware attached; pass use_mocap:=true once real mocap data
    # is actually available.
    use_mocap = LaunchConfiguration('use_mocap')

    return LaunchDescription([
        DeclareLaunchArgument('use_mocap', default_value='false'),
        Node(
            package='digital_twin_viz',
            executable='display_urdf_node',
            name='urdf_publisher',
            parameters=[{'urdf_path': urdf_path}]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='rsp',
            parameters=[{'robot_description': robot_description}],
	    remappings=[('/joint_states', '/joint_states_single')]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),
        Node(
            package='digital_twin_viz',
            executable='rigid_body_tf_broadcaster',
            name='rigid_tf_node',
            parameters=[{'target_body_name': '1'}],
            condition=IfCondition(use_mocap)
        ),
    ] + [
        # robot_state_publisher explicitly skips floating joints ("Floating
        # joint. Not adding segment from world to X") -- world_to_PAT/_CSO/
        # _calibrator are exactly that, so without rigid_body_tf_broadcaster
        # (or real mocap) NOTHING connects PAT_link/CSO_link/calibrator_link
        # (and everything hanging off PAT_link -- the whole arm) to the TF
        # tree at all. These publish the identity transform the URDF's own
        # world_to_* joints declare (xyz="0 0 0" rpy="0 0 0") as a stand-in,
        # active only while there's no live mocap feed to conflict with.
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'static_tf_{child}',
            arguments=['--frame-id', 'world', '--child-frame-id', child],
            condition=UnlessCondition(use_mocap)
        )
        for child in ('PAT_link', 'CSO_link', 'calibrator_link')
    ] + [
        # joint2/joint3/joint5 are revolute -- robot_state_publisher only
        # publishes a transform for a non-fixed joint once it's seen a
        # /joint_states_single value for it, and nothing else in this
        # display-only launch publishes one. Without this, link2 (and
        # everything past it: link3-6, gripper_base) never gets a transform
        # at all and ends up in its own disconnected TF tree, invisible in
        # RViz even though PAT_link/base_link/link1 render fine. No
        # joint_state_publisher[_gui] package is installed in this
        # workspace, so this republishes a fixed all-zero pose instead --
        # good enough to see the model; swap for a real publisher if you
        # need to pose the arm.
        Node(
            package='digital_twin_viz',
            executable='default_joint_state_publisher',
            name='default_joint_state_publisher'
        )
    ])

