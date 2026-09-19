# digital_twin_viz/default_joint_state_publisher.py
#
# Stand-in for joint_state_publisher[_gui] (neither is installed in this
# workspace): republishes a fixed, all-zero pose for the arm's revolute
# joints so robot_state_publisher has something to compute link2/3/5's
# (and everything past them: link4-6, gripper_base) transforms from. Without
# this those links never get a transform at all and sit in their own
# disconnected TF tree, invisible in RViz even though PAT_link/base_link/
# link1 render fine.
#
# Must stamp with real time -- a message stamped at t=0 (e.g. a plain
# `ros2 topic pub` with no header set) makes every /tf message
# robot_state_publisher emits from it carry the same t=0 stamp too, which
# tf2 treats as stale/non-current, so nothing ever resolves as "now".

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ['joint2', 'joint3', 'joint5']


class DefaultJointStatePublisher(Node):
    def __init__(self):
        super().__init__('default_joint_state_publisher')
        self.pub = self.create_publisher(JointState, 'joint_states_single', 10)
        self.timer = self.create_timer(0.5, self.publish_once)

    def publish_once(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [0.0] * len(JOINT_NAMES)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DefaultJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
