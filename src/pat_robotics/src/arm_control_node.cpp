#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "pat_robotics/control/arm_controller.hpp"

using namespace std::chrono_literals;

/// Thin ROS 2 wrapper around ArmController. Manages exactly the joints
/// listed in the 'joint_names' parameter — this is what makes the node
/// joint-count-agnostic and lets multiple independent instances coexist
/// (each with its own subset of joint_names) against the same shared
/// /chaser/arm/* topics for a multi-arm setup, with no coordination needed.
class ArmControlNode : public rclcpp::Node {
public:
    ArmControlNode() : Node("pat_robotics") {
        joint_names_ = declare_parameter<std::vector<std::string>>(
            "joint_names", std::vector<std::string>{"arm_joint1", "arm_joint2", "arm_joint3"});
        const auto kp = declare_parameter<std::vector<double>>(
            "pid.kp", std::vector<double>(joint_names_.size(), 1.0));
        const auto ki = declare_parameter<std::vector<double>>(
            "pid.ki", std::vector<double>(joint_names_.size(), 0.0));
        const auto kd = declare_parameter<std::vector<double>>(
            "pid.kd", std::vector<double>(joint_names_.size(), 0.0));
        const auto u_max = declare_parameter<std::vector<double>>(
            "pid.u_max", std::vector<double>(joint_names_.size(),
                                              std::numeric_limits<double>::infinity()));
        const double ctrl_hz = declare_parameter<double>("control_hz", 50.0);

        if (kp.size() != joint_names_.size() || ki.size() != joint_names_.size() ||
            kd.size() != joint_names_.size() || u_max.size() != joint_names_.size()) {
            RCLCPP_ERROR(get_logger(),
                         "pid.kp/ki/kd/u_max must each have the same length as joint_names (%zu)",
                         joint_names_.size());
            throw std::runtime_error("pat_robotics: PID gains size mismatch with joint_names");
        }

        std::vector<pat_robotics::JointPID::Gains> gains(joint_names_.size());
        for (size_t i = 0; i < joint_names_.size(); ++i)
            gains[i] = {kp[i], ki[i], kd[i], u_max[i]};
        controller_ = std::make_unique<pat_robotics::ArmController>(gains, 1.0 / ctrl_hz);

        q_.assign(joint_names_.size(), 0.0);
        q_des_.assign(joint_names_.size(), 0.0);

        state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/chaser/arm/joint_states", 10,
            std::bind(&ArmControlNode::onJointState, this, std::placeholders::_1));
        setpoint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/chaser/arm/joint_setpoint", 10,
            std::bind(&ArmControlNode::onSetpoint, this, std::placeholders::_1));
        torque_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            "/chaser/arm/torque_command", 10);

        ctrl_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / ctrl_hz),
                                       std::bind(&ArmControlNode::controlLoop, this));

        RCLCPP_INFO(get_logger(), "Arm control: %zu joints, ctrl=%.0f Hz",
                    joint_names_.size(), ctrl_hz);
    }

private:
    // Copies whichever of our own joint_names_ entries appear in msg into
    // dst, by name — msg may cover other arms/joints we don't own, or only
    // a subset of our own; unmatched names on either side are left alone.
    void applyNamedValues(const sensor_msgs::msg::JointState::SharedPtr& msg,
                          const std::vector<double>& values, std::vector<double>& dst) {
        for (size_t k = 0; k < msg->name.size() && k < values.size(); ++k) {
            for (size_t i = 0; i < joint_names_.size(); ++i) {
                if (msg->name[k] == joint_names_[i]) { dst[i] = values[k]; break; }
            }
        }
    }

    void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mu_);
        applyNamedValues(msg, msg->position, q_);
        has_state_ = true;
    }

    void onSetpoint(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mu_);
        applyNamedValues(msg, msg->position, q_des_);
        has_setpoint_ = true;
    }

    void controlLoop() {
        std::vector<double> q, q_des;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!has_state_ || !has_setpoint_) return;
            q = q_; q_des = q_des_;
        }
        const auto u = controller_->compute(q, q_des);

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = get_clock()->now();
        cmd.name = joint_names_;
        cmd.effort = u;
        torque_pub_->publish(cmd);
    }

    std::unique_ptr<pat_robotics::ArmController> controller_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr setpoint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr torque_pub_;
    rclcpp::TimerBase::SharedPtr ctrl_tmr_;
    std::mutex mu_;
    std::vector<std::string> joint_names_;
    std::vector<double> q_, q_des_;
    bool has_state_{false}, has_setpoint_{false};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmControlNode>());
    rclcpp::shutdown();
}
