#include <cmath>
#include <memory>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <pat_msgs/msg/thruster_command.hpp>
#include <pat_msgs/msg/control_error.hpp>

#include "pat_gnc/interfaces/i_controller.hpp"
#include "pat_gnc/interfaces/i_navigator.hpp"
#include "pat_gnc/control/pid_controller.hpp"
#include "pat_gnc/navigation/direct_navigator.hpp"
#include "pat_gnc/navigation/ekf_navigator.hpp"

using namespace std::chrono_literals;

class GncNode : public rclcpp::Node {
public:
    GncNode() : Node("pat_gnc") {
        const auto kp      = declare_parameter<std::vector<double>>("pid.kp",    {1.2, 1.2, 0.8});
        const auto ki      = declare_parameter<std::vector<double>>("pid.ki",    {0.05, 0.05, 0.02});
        const auto kd      = declare_parameter<std::vector<double>>("pid.kd",    {0.4, 0.4, 0.2});
        const auto u_max_v = declare_parameter<std::vector<double>>("pid.u_max", {5.0, 5.0, 2.0});
        const double ctrl_hz = declare_parameter<double>("control_hz",    10.0);
        const double nav_hz  = declare_parameter<double>("navigation_hz", 50.0);
        hold_x_ = declare_parameter<double>("hold_offset_x", 0.30);
        hold_y_ = declare_parameter<double>("hold_offset_y", 0.0);

        // —— Navigator factory ——————————————————————————————————————————
        // Add new navigators as else-if blocks; no other changes needed.
        const auto nav_type = declare_parameter<std::string>("navigator", "direct");
        if (nav_type == "direct") {
            navigator_ = std::make_unique<pat_gnc::DirectNavigator>();
        } else if (nav_type == "ekf") {
            pat_gnc::PlanarEKF::Params ep;
            ep.dt = 1.0 / nav_hz;
            navigator_ = std::make_unique<pat_gnc::EkfNavigator>(ep);
        } else {
            RCLCPP_WARN(get_logger(), "Unknown navigator '%s', using 'direct'", nav_type.c_str());
            navigator_ = std::make_unique<pat_gnc::DirectNavigator>();
        }
        RCLCPP_INFO(get_logger(), "Navigator: %s", nav_type.c_str());

        // —— Controller factory ———————————————————————————————————————————
        // Add LQR, MPC, etc. as else-if blocks; no other changes needed.
        const auto ctrl_type = declare_parameter<std::string>("controller", "pid");
        if (ctrl_type == "pid") {
            pat_gnc::PIDController::Gains g;
            g.kp    = {kp[0],      kp[1],      kp[2]};
            g.ki    = {ki[0],      ki[1],      ki[2]};
            g.kd    = {kd[0],      kd[1],      kd[2]};
            g.u_max = {u_max_v[0], u_max_v[1], u_max_v[2]};
            controller_ = std::make_unique<pat_gnc::PIDController>(g, 1.0 / ctrl_hz);
        } else {
            RCLCPP_WARN(get_logger(), "Unknown controller '%s', using 'pid'", ctrl_type.c_str());
            controller_ = std::make_unique<pat_gnc::PIDController>(
                pat_gnc::PIDController::Gains{}, 1.0 / ctrl_hz);
        }
        RCLCPP_INFO(get_logger(), "Controller: %s  ctrl=%.0f Hz  nav=%.0f Hz",
                    ctrl_type.c_str(), ctrl_hz, nav_hz);

        // —— ROS 2 interface ———————————————————————————————————————————————
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/chaser/odom", 10,
            std::bind(&GncNode::onChaserOdom, this, std::placeholders::_1));
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/chaser/imu", 10,
            std::bind(&GncNode::onImu, this, std::placeholders::_1));
        target_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/target/odom", 10,
            std::bind(&GncNode::onTargetOdom, this, std::placeholders::_1));
        cmd_pub_ = create_publisher<pat_msgs::msg::ThrusterCommand>("/chaser/thruster_command", 10);
        err_pub_ = create_publisher<pat_msgs::msg::ControlError>("/chaser/gnc/control_error", 10);
        ctrl_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / ctrl_hz),
                                      std::bind(&GncNode::controlLoop, this));
    }

private:
    void onChaserOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mu_);
        const Eigen::Vector3d pose{msg->pose.pose.position.x,
                                   msg->pose.pose.position.y,
                                   quatToYaw(msg->pose.pose.orientation)};
        const Eigen::Vector3d twist{msg->twist.twist.linear.x,
                                    msg->twist.twist.linear.y,
                                    msg->twist.twist.angular.z};
        navigator_->onOdometry(pose, twist);
        has_chaser_ = true;
    }

    void onImu(const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mu_);
        navigator_->onImu(msg->angular_velocity.z);
    }

    void onTargetOdom(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(mu_);
        tx_ = msg->pose.pose.position.x; ty_ = msg->pose.pose.position.y;
        tyaw_ = quatToYaw(msg->pose.pose.orientation);
        has_target_ = true;
    }

    void controlLoop() {
        if (!has_chaser_ || !has_target_) return;
        pat_gnc::INavigator::StateVec est;
        double tx, ty, tyaw;
        { std::lock_guard<std::mutex> lk(mu_); est = navigator_->state();
          tx = tx_; ty = ty_; tyaw = tyaw_; }

        const double c = std::cos(tyaw), s = std::sin(tyaw);
        Eigen::Vector3d err{
            (tx + c * hold_x_ - s * hold_y_) - est(0),
            (ty + s * hold_x_ + c * hold_y_) - est(1),
            std::remainder(tyaw - est(2), 2.0 * M_PI)};

        const Eigen::Vector3d u = controller_->compute(err);

        pat_msgs::msg::ThrusterCommand cmd;
        cmd.header.stamp = get_clock()->now();
        cmd.force[0] = std::max(0.0,  u(0));
        cmd.force[1] = std::max(0.0, -u(0));
        cmd.force[2] = std::max(0.0,  u(1));
        cmd.force[3] = std::max(0.0, -u(1));
        cmd_pub_->publish(cmd);

        pat_msgs::msg::ControlError ce;
        ce.header.stamp = cmd.header.stamp;
        ce.pos_x = err(0); ce.pos_y = err(1); ce.yaw = err(2);
        ce.vel_x = -est(3); ce.vel_y = -est(4); ce.yaw_rate = -est(5);
        err_pub_->publish(ce);
    }

    static double quatToYaw(const geometry_msgs::msg::Quaternion& q) noexcept {
        return std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
    }

    std::unique_ptr<pat_gnc::IController> controller_;
    std::unique_ptr<pat_gnc::INavigator>  navigator_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_, target_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr   imu_sub_;
    rclcpp::Publisher<pat_msgs::msg::ThrusterCommand>::SharedPtr cmd_pub_;
    rclcpp::Publisher<pat_msgs::msg::ControlError>::SharedPtr    err_pub_;
    rclcpp::TimerBase::SharedPtr ctrl_tmr_;
    std::mutex mu_;
    bool   has_chaser_{false}, has_target_{false};
    double tx_{0}, ty_{0}, tyaw_{0};
    double hold_x_{0.30}, hold_y_{0.0};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GncNode>());
    rclcpp::shutdown();
}
