#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "pat_arm_nmpc/arm_nmpc.h"

/// One ArmNMPC instance per arm (run two node processes, left + right).
///
/// Each tick the node assembles the full floating-base MuJoCo state
///   q = [base_pos(3), base_quat(4 wxyz), joint_angles(model N)]
///   v = [base_linvel(3), base_angvel(3), joint_vels(model N)]
/// from /chaser/odom (planar → 6-DOF slots, z/roll/pitch = 0) and the full
/// joint_states message, runs the controller, and publishes only its own
/// joints' effort on /chaser/arm/torque_command (name-matched, so the two
/// instances coexist on one topic).
class ArmNMPCNode : public rclcpp::Node {
public:
    ArmNMPCNode();

private:
    // --- Parameter loading (nmpc.yaml → NMPCParams) ----------------------- //
    pat_arm_nmpc::NMPCParams loadParams();
    std::vector<double> getJointArray(const std::string& name);   // length == joint_names_
    std::array<double, 3> getVec3(const std::string& name, std::array<double, 3> def);

    // --- Runtime -------------------------------------------------------------- //
    /// Copy the msg entries whose name is in `names` into dst[i] (dst sized to
    /// names.size()); entries not present in the msg are left untouched.
    static void applyNamedValues(const sensor_msgs::msg::JointState::SharedPtr& msg,
                                 const std::vector<double>& values,
                                 const std::vector<std::string>& names,
                                 std::vector<double>& dst);
    void onJointState(sensor_msgs::msg::JointState::SharedPtr msg);
    void onOdom(nav_msgs::msg::Odometry::SharedPtr msg);
    void onEeSetpoint(geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void controlLoop();

    std::string arm_side_;                        // "left" | "right"
    pat_arm_nmpc::NMPCParams params_;
    std::unique_ptr<pat_arm_nmpc::ArmNMPC> controller_;

    std::vector<std::string> joint_names_;        // this arm's joints (output, 3)
    std::vector<std::string> model_joint_names_;  // ALL model joints, model order (6)
    std::vector<std::size_t> owned_model_idx_;    // joint_names_[k] → model index

    std::vector<double> q_joints_, v_joints_;     // sized to model_joint_names_
    std::vector<double> ee_quat_{1.0, 0.0, 0.0, 0.0};   // desired EE quat [w,x,y,z]
    double ee_x_{0.0}, ee_y_{0.0};                      // desired EE position [m]
    double base_x_{0.0}, base_y_{0.0}, base_yaw_{0.0};
    double base_vx_{0.0}, base_vy_{0.0}, base_omega_{0.0};
    bool has_state_{false}, has_odom_{false}, has_setpoint_{false};

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ee_setpoint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr torque_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;  // measured EE pose (world)
    rclcpp::TimerBase::SharedPtr ctrl_tmr_;
    std::mutex mu_;
};
