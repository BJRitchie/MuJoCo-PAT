#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <random>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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
    // Off for headless use (CI, batch tools like tools/pinn_datagen/, a
    // server with no GLX/DRM device) -- GLFW window creation is otherwise
    // unconditional and fails loudly (though non-fatally) without one.
    const bool visualise  = declare_parameter<bool>("visualise", true);

    // Check the model path
    if (model_path.empty())
        throw std::runtime_error("'model_path' parameter must be set via launch file");

    // Occupy sim class pointer
    sim_ = std::make_unique<pat_simulation::MuJoCoSim>(model_path, visualise);
    RCLCPP_INFO(get_logger(), "MuJoCo loaded. dt=%.4f s", sim_->dt());
    ctrl_.assign(sim_->model()->nu, 0.0);

    // Arm joints/actuators are entirely opt-in via config — empty by default.
    // Adding a joint (or a whole second arm) means extending these two lists
    // and the MJCF; no source changes here are needed.
    arm_joint_names_ = declare_parameter<std::vector<std::string>>(
        "arm_joint_names", std::vector<std::string>{});
    const auto arm_actuator_names = declare_parameter<std::vector<std::string>>(
        "arm_actuator_names", std::vector<std::string>{});
    for (const auto& n : arm_actuator_names)
        arm_actuator_ids_.push_back(sim_->actuatorId(n));

    // Per-launch initial arm configuration — offline dataset generation
    // (tools/pinn_datagen/) needs a different qpos0 per trajectory, which
    // MuJoCoSim's construction path (mj_resetData -> mj_forward, always
    // qpos0=0) cannot otherwise produce, and MuJoCoSim::reset() only restores
    // qpos0, not an arbitrary pose. Empty (default) => untouched all-zero
    // pose, fully backward compatible.
    const auto initial_arm_qpos = declare_parameter<std::vector<double>>(
        "initial_arm_qpos", std::vector<double>{});
    if (!initial_arm_qpos.empty()) {
        if (initial_arm_qpos.size() != arm_joint_names_.size())
            throw std::runtime_error(
                "'initial_arm_qpos' size must match 'arm_joint_names' "
                "(or be left empty for the default zero pose)");
        sim_->setJointPositions(arm_joint_names_, initial_arm_qpos);
        RCLCPP_INFO(get_logger(), "Applied initial_arm_qpos override (%zu joints)",
                    initial_arm_qpos.size());
    }

    // Same mechanism as initial_arm_qpos, for the target's 3 planar joints
    // (target_x, target_y, target_yaw) -- e.g. to park the target somewhere
    // a trial's sampled arm configs can't reach. Empty (default) => untouched
    // at its xacro pos.
    const auto initial_target_qpos = declare_parameter<std::vector<double>>(
        "initial_target_qpos", std::vector<double>{});
    if (!initial_target_qpos.empty()) {
        if (initial_target_qpos.size() != 3)
            throw std::runtime_error(
                "'initial_target_qpos' must have exactly 3 entries "
                "(target_x, target_y, target_yaw), or be left empty");
        sim_->setJointPositions({"target_x", "target_y", "target_yaw"}, initial_target_qpos);
        RCLCPP_INFO(get_logger(), "Applied initial_target_qpos override");
    }

    // Publishers
    ch_odom_ = create_publisher<nav_msgs::msg::Odometry>("/chaser/odom", 10);
    tg_odom_ = create_publisher<nav_msgs::msg::Odometry>("/target/odom", 10);
    ch_imu_  = create_publisher<sensor_msgs::msg::Imu>("/chaser/imu", 10);
    arm_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/chaser/arm/joint_states", 10);
    tf_br_   = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Subscriber
    thr_sub_ = create_subscription<pat_msgs::msg::ThrusterCommand>(
        "/chaser/thruster_command", 10,
        std::bind(&SimulationNode::thrusterSubscriberCallback, this, std::placeholders::_1));
    arm_torque_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/chaser/arm/torque_command", 10,
        std::bind(&SimulationNode::armTorqueSubscriberCallback, this, std::placeholders::_1));

    // Cosmetic ee_setpoint markers -- gracefully absent if the loaded MJCF
    // doesn't have the marker bodies (mocapId() returns -1, resolved once
    // here rather than by name on every message).
    const int marker_mocap_id_L = sim_->mocapId("ee_setpoint_marker_L");
    const int marker_mocap_id_R = sim_->mocapId("ee_setpoint_marker_R");
    // Draw the markers at the chaser's true world-frame height rather than a
    // guessed constant -- the arms rotate about world Z only, so every EE
    // site sits at this same height regardless of joint configuration.
    // Falls back to 0.0 if the model has no "chaser" body (already covered
    // by mocapId()'s -1 no-op path in that case anyway).
    {
        const int chaser_body_id = mj_name2id(sim_->model(), mjOBJ_BODY, "chaser");
        if (chaser_body_id >= 0) {
            marker_height_ = sim_->data()->xpos[3 * chaser_body_id + 2];
        }
    }
    // Lambdas, not std::bind -- std::bind's result has no fixed call
    // signature rclcpp's callback-type detection can cleanly introspect
    // once a non-placeholder argument (mocap_id) precedes the placeholder;
    // a lambda's operator() is unambiguous.
    ee_setpoint_marker_sub_L_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/left/ee_setpoint", 10,
        [this, marker_mocap_id_L](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            eeSetpointMarkerCallback(marker_mocap_id_L, msg);
        });
    ee_setpoint_marker_sub_R_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/right/ee_setpoint", 10,
        [this, marker_mocap_id_R](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
            eeSetpointMarkerCallback(marker_mocap_id_R, msg);
        });

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
    publishArmState(now);

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

void SimulationNode::publishArmState(const rclcpp::Time& t) {
    if (arm_joint_names_.empty()) return;
    const auto s = sim_->getJointStates(arm_joint_names_);
    sensor_msgs::msg::JointState m;
    m.header.stamp = t;
    m.name     = s.name;
    m.position = s.pos;
    m.velocity = s.vel;
    arm_state_pub_->publish(m);
}

// === Callbacks ==================================================================
void SimulationNode::thrusterSubscriberCallback(
    const pat_msgs::msg::ThrusterCommand::SharedPtr msg) {
    // Block
    std::lock_guard<std::mutex> lk(mu_);

    // Update control forces — bound is a literal 4, matching
    // ThrusterCommand.force[4], NOT ctrl_.size() (which now covers all
    // actuators including the arm; msg->force only ever has 4 elements).
    for (size_t i = 0; i < 4 && i < ctrl_.size(); ++i)
        ctrl_[i] = msg->force[i];
}

void SimulationNode::armTorqueSubscriberCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mu_);
    // Match by name against our configured arm joints — a message may
    // legitimately cover only a subset (e.g. one of several independent arm
    // controllers), so unmatched names/entries are simply ignored.
    for (size_t k = 0; k < msg->name.size() && k < msg->effort.size(); ++k) {
        for (size_t i = 0; i < arm_joint_names_.size(); ++i) {
            if (msg->name[k] == arm_joint_names_[i]) {
                ctrl_[arm_actuator_ids_[i]] = msg->effort[k];
                break;
            }
        }
    }
}

void SimulationNode::eeSetpointMarkerCallback(
    int mocap_id, const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (mocap_id < 0) return;  // model has no marker body for this side -- skip
    // marker_height_ (the chaser's true world Z, read once at startup), NOT
    // msg->pose.position.z -- ee_setpoint's z is conventionally unused/zero
    // (the planar task-space law never reads it, see arm_nmpc.cpp's
    // controlLaw), which would otherwise bury the marker at table level.
    std::lock_guard<std::mutex> lk(mu_);
    sim_->setMocapPose(mocap_id, msg->pose.position.x, msg->pose.position.y, marker_height_,
                        msg->pose.orientation.w, msg->pose.orientation.x,
                        msg->pose.orientation.y, msg->pose.orientation.z);
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
