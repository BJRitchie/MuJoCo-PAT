#include <array>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64.hpp>

#include "pat_telemetry/telemetry_aggregator.hpp"

using pat_telemetry::TelemetryAggregator;

namespace
{
constexpr std::array<const char*, 2> kSides{"left", "right"};
}

// Merges /chaser/arm/joint_states + /chaser/arm/torque_command (by name, not
// array position - see TelemetryAggregator::mergeByName) into one
// /chaser/telemetry/joint_states, and publishes each side's EE distance from
// its setpoint. Deliberately touches only the existing sim/hardware-boundary
// topics (CLAUDE.md's topic contract table) - identical behaviour whether
// pat_simulation or real drivers are the publisher on the other end.
class TelemetryNode : public rclcpp::Node
{
public:
    TelemetryNode() : Node("pat_telemetry")
    {
        joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/chaser/arm/joint_states", 10,
            std::bind(&TelemetryNode::onJointStates, this, std::placeholders::_1));
        torque_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/chaser/arm/torque_command", 10,
            std::bind(&TelemetryNode::onTorque, this, std::placeholders::_1));
        merged_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            "/chaser/telemetry/joint_states", 10);

        for (size_t i = 0; i < kSides.size(); ++i) {
            const std::string side = kSides[i];
            ee_pose_sub_[i] = create_subscription<geometry_msgs::msg::PoseStamped>(
                "/chaser/arm/" + side + "/ee_pose", 10,
                [this, i](geometry_msgs::msg::PoseStamped::SharedPtr msg) { onEePose(i, msg); });
            ee_setpoint_sub_[i] = create_subscription<geometry_msgs::msg::PoseStamped>(
                "/chaser/arm/" + side + "/ee_setpoint", 10,
                [this, i](geometry_msgs::msg::PoseStamped::SharedPtr msg) { onEeSetpoint(i, msg); });
            distance_pub_[i] = create_publisher<std_msgs::msg::Float64>(
                "/chaser/telemetry/ee_distance/" + side, 10);
        }
    }

private:
    void onJointStates(sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mu_);

        sensor_msgs::msg::JointState out;
        out.header = msg->header;
        out.name = msg->name;
        out.position = msg->position;
        out.velocity = msg->velocity;
        out.effort = TelemetryAggregator::mergeByName(out.name, torque_names_, torque_effort_);
        merged_pub_->publish(out);
    }

    void onTorque(sensor_msgs::msg::JointState::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mu_);
        // /chaser/arm/torque_command carries only ONE side's 3 joints per
        // message (arm_nmpc_left/right each publish independently onto the
        // shared topic) - merge in place, don't replace the whole cache.
        TelemetryAggregator::updateByName(torque_names_, torque_effort_, msg->name, msg->effort);
    }

    void onEePose(size_t side, geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_setpoint_[side]) return;  // nothing to compare against yet

        std_msgs::msg::Float64 d;
        d.data = TelemetryAggregator::planarDistance(
            msg->pose.position.x, msg->pose.position.y,
            setpoint_x_[side], setpoint_y_[side]);
        distance_pub_[side]->publish(d);
    }

    void onEeSetpoint(size_t side, geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(mu_);
        setpoint_x_[side] = msg->pose.position.x;
        setpoint_y_[side] = msg->pose.position.y;
        has_setpoint_[side] = true;
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_, torque_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr merged_pub_;
    std::array<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr, 2> ee_pose_sub_, ee_setpoint_sub_;
    std::array<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr, 2> distance_pub_;

    std::mutex mu_;
    std::vector<std::string> torque_names_;
    std::vector<double> torque_effort_;
    std::array<double, 2> setpoint_x_{0.0, 0.0}, setpoint_y_{0.0, 0.0};
    std::array<bool, 2> has_setpoint_{false, false};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TelemetryNode>());
    rclcpp::shutdown();
}
