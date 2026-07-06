#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pat_msgs/msg/thruster_command.hpp>


#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/simulation_node.hpp" 

namespace simulation_node 
{

// === Public Interface ===========================================================
SimulationNode::SimulationNode() : Node("pat_simulation") {
    // Parameters 
    const auto model_path = declare_parameter<std::string>("model_path", "");
    const double sim_hz   = declare_parameter<double>("sim_rate_hz",  500.0);
    const double pub_hz   = declare_parameter<double>("pub_rate_hz",  100.0);
    imu_noise_            = declare_parameter<double>("imu_noise_std", 0.005);

    // Check the model path 
    if (model_path.empty())
        throw std::runtime_error("'model_path' parameter must be set via launch file");

    // Occupy sim class pointer 
    sim_ = std::make_unique<pat_simulation::MuJoCoSim>(model_path);
    RCLCPP_INFO(get_logger(), "MuJoCo loaded. dt=%.4f s", sim_->dt());

    // Publishers 
    ch_odom_ = create_publisher<nav_msgs::msg::Odometry>("/chaser/odom", 10);
    tg_odom_ = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
    ch_imu_  = create_publisher<sensor_msgs::msg::Imu>("/chaser/imu", 10);
    tf_br_   = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Subscriber
    thr_sub_ = create_subscription<pat_msgs::msg::ThrusterCommand>(
        "/chaser/thruster_command", 10,
        std::bind(&SimulationNode::thrusterSubscriberCallback, this, std::placeholders::_1));

    sim_tmr_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / sim_hz),
        std::bind(&SimulationNode::simTimerCallback, this)); 
    pub_tmr_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / pub_hz),
        std::bind(&SimulationNode::publish, this));

}

// === Private Methods ============================================================
void SimulationNode::publish() { 
    // Current time 
    const auto now = get_clock()->now();

    // Relevant satellites 
    const auto ch  = sim_->getChaserState();
    const auto tg  = sim_->getTargetState();

    // Publish data for each satellite 
    publishOdom(*ch_odom_, "map", "chaser", ch, now);
    publishOdom(*tg_odom_, "map", "target", tg, now);
    publishImu(ch, now);

    // Broadcast the transforms 
    broadcastTF("map", "chaser", ch, now);
    broadcastTF("map", "target", tg, now);
}

void SimulationNode::publishOdom(rclcpp::Publisher<nav_msgs::msg::Odometry>& pub,
                     const std::string& frame, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t) {

    // Init odom msg 
    nav_msgs::msg::Odometry m; 

    // Populate info
    m.header.stamp = t; 
    m.header.frame_id = frame; 
    m.child_frame_id = child;

    // Populate pose  
    m.pose.pose.position.x = s.x; 
    m.pose.pose.position.y = s.y;

    // Orientation 
    tf2::Quaternion q; 
    q.setRPY(0, 0, s.theta);
    m.pose.pose.orientation.w = q.w(); 
    m.pose.pose.orientation.x = q.x();
    m.pose.pose.orientation.y = q.y(); 
    m.pose.pose.orientation.z = q.z();

    // Velocities 
    m.twist.twist.linear.x = s.xdot; 
    m.twist.twist.linear.y = s.ydot;
    m.twist.twist.angular.z = s.thetadot;
    pub.publish(m);
}

void SimulationNode::publishImu(const pat_simulation::PlanarState& s, const rclcpp::Time& t) {
    // Init imu msg 
    sensor_msgs::msg::Imu m;

    // Populate header info 
    m.header.stamp = t; 
    m.header.frame_id = "chaser_imu";

    // Populate data and apply noise 
    std::normal_distribution<double> nd(0.0, imu_noise_);
    m.angular_velocity.z = s.thetadot + nd(rng_);
    m.linear_acceleration.x = nd(rng_); 
    m.linear_acceleration.y = nd(rng_);

    // Publish 
    ch_imu_->publish(m);
}

void SimulationNode::broadcastTF(const std::string& parent, const std::string& child,
                     const pat_simulation::PlanarState& s, const rclcpp::Time& t) {

    // Init transform msg 
    geometry_msgs::msg::TransformStamped tf;

    // Header info 
    tf.header.stamp = t; 
    tf.header.frame_id = parent; 
    tf.child_frame_id = child;

    // Position 
    tf.transform.translation.x = s.x; 
    tf.transform.translation.y = s.y;

    // Pose 
    tf2::Quaternion q; 
    q.setRPY(0, 0, s.theta);
    tf.transform.rotation.w = q.w(); 
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y(); 
    tf.transform.rotation.z = q.z();

    // Broadcast 
    tf_br_->sendTransform(tf);
}

// === Callbacks ==================================================================
void SimulationNode::thrusterSubscriberCallback(
    const pat_msgs::msg::ThrusterCommand::SharedPtr msg) {
    // Block 
    std::lock_guard<std::mutex> lk(mu_);

    // Update control forces
    for (size_t i = 0; i < ctrl_.size(); ++i) 
        ctrl_[i] = msg->force[i];
}

void SimulationNode::simTimerCallback() {
    std::lock_guard<std::mutex> lk(mu_); 
    sim_->step(ctrl_);
}

}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<simulation_node::SimulationNode>());
    rclcpp::shutdown();
}
