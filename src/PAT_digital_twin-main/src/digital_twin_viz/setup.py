from setuptools import setup
import os

package_name = 'digital_twin_viz'

def get_data_files(folder_src, folder_dst):
    """
    Recursively collect all files under `folder_src` and place them in `folder_dst` for install.
    """
    data_files = []
    for dirpath, _, filenames in os.walk(folder_src):
        files = [os.path.join(dirpath, f) for f in filenames]
        if files:
            dst = os.path.join('share', package_name, folder_dst, os.path.relpath(dirpath, folder_src))
            data_files.append((dst, files))
    return data_files

# Get mesh files from urdf/meshes/
mesh_data_files = get_data_files('meshes', 'meshes')

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name + '/launch', ['launch/display_robot.launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/Platform_description.urdf']),
        ('share/' + package_name + '/rviz', ['rviz/digital_twin_config.rviz']),
        ('share/digital_twin_viz', ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/digital_twin_viz']),
    ] + mesh_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='your@email.com',
    description='Digital twin URDF display node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'display_urdf_node = digital_twin_viz.display_urdf_node:main',
            'rigid_body_tf_broadcaster = digital_twin_viz.rigid_body_tf_broadcaster:main',
            'default_joint_state_publisher = digital_twin_viz.default_joint_state_publisher:main'
        ],
    },
)
