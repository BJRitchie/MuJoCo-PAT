import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from mocap4r2_msgs.msg import RigidBodies

class MocapTFBroadcaster(Node):
    def __init__(self):
        super().__init__('mocap_tf_broadcaster')
        self.broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            RigidBodies,
            '/rigid_bodies',
            self.rigid_body_callback,
            10  # QoS depth
        )

    def rigid_body_callback(self, msg):
        if not msg.rigidbodies:
            self.get_logger().warn('No rigid bodies in message.')
            return

        for rb in msg.rigidbodies:

            if rb.rigid_body_name == '1':
                tf_msg = TransformStamped()

                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = 'world'
                tf_msg.child_frame_id = 'calibrator_link'

                tf_msg.transform.translation.x = rb.pose.position.x
                tf_msg.transform.translation.y = rb.pose.position.y
                tf_msg.transform.translation.z = rb.pose.position.z

                tf_msg.transform.rotation.x = rb.pose.orientation.x
                tf_msg.transform.rotation.y = rb.pose.orientation.y
                tf_msg.transform.rotation.z = rb.pose.orientation.z
                tf_msg.transform.rotation.w = rb.pose.orientation.w

                self.broadcaster.sendTransform(tf_msg)

            if rb.rigid_body_name == '2':
                tf_msg = TransformStamped()

                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = 'world'
                tf_msg.child_frame_id = 'PAT_link'

                tf_msg.transform.translation.x = rb.pose.position.x
                tf_msg.transform.translation.y = rb.pose.position.y
                tf_msg.transform.translation.z = rb.pose.position.z

                tf_msg.transform.rotation.x = rb.pose.orientation.x
                tf_msg.transform.rotation.y = rb.pose.orientation.y
                tf_msg.transform.rotation.z = rb.pose.orientation.z
                tf_msg.transform.rotation.w = rb.pose.orientation.w

                self.broadcaster.sendTransform(tf_msg)

            elif rb.rigid_body_name == '3':
                tf_msg = TransformStamped()

                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = 'world'
                tf_msg.child_frame_id = 'CSO_link'

                tf_msg.transform.translation.x = rb.pose.position.x
                tf_msg.transform.translation.y = rb.pose.position.y
                tf_msg.transform.translation.z = rb.pose.position.z

                tf_msg.transform.rotation.x = rb.pose.orientation.x
                tf_msg.transform.rotation.y = rb.pose.orientation.y
                tf_msg.transform.rotation.z = rb.pose.orientation.z
                tf_msg.transform.rotation.w = rb.pose.orientation.w

                self.broadcaster.sendTransform(tf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MocapTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()