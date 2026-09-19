# digital_twin_viz/display_urdf_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy

class DisplayURDFNode(Node):
    def __init__(self):
        super().__init__('display_urdf_node')
        self.declare_parameter('urdf_path', '')
        urdf_path = self.get_parameter('urdf_path').value

        self.get_logger().info(f'Loading URDF from: {urdf_path}')
        try:
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
                
                qos = QoSProfile(depth=10)
                qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

                self.pub = self.create_publisher(String, 'robot_description', qos)
                msg = String()
                msg.data = urdf_content
                self.pub.publish(msg)
                self.get_logger().info('URDF published to /robot_description')
        except Exception as e:
            self.get_logger().error(f'Failed to read URDF: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DisplayURDFNode()
    rclpy.spin_once(node, timeout_sec=1.0)  # Only need one-time publish
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
