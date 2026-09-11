#include "pat_arm_nmpc/arm_nmpc_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include <Eigen/Dense>

using namespace std::chrono_literals;

namespace {
double quatToYaw(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
}  // namespace

ArmNMPCNode::ArmNMPCNode() : Node("pat_arm_nmpc") {
    arm_side_ = declare_parameter<std::string>("arm_side", "");
    if (arm_side_.empty())
        throw std::runtime_error("pat_arm_nmpc: 'arm_side' is required (\"left\"/\"right\")");

    const std::string mjcf_path = declare_parameter<std::string>("mjcf_path", "");
    if (mjcf_path.empty())
        throw std::runtime_error("pat_arm_nmpc: 'mjcf_path' is required");

    joint_names_ = declare_parameter<std::vector<std::string>>(
        "joint_names", std::vector<std::string>{});
    model_joint_names_ = declare_parameter<std::vector<std::string>>(
        "model_joint_names", std::vector<std::string>{});
    if (joint_names_.empty())
        throw std::runtime_error("pat_arm_nmpc: 'joint_names' must be non-empty");
    if (model_joint_names_.empty())
        throw std::runtime_error("pat_arm_nmpc: 'model_joint_names' must be non-empty");

    const double ctrl_hz = declare_parameter<double>("control_hz", 50.0);

    params_ = loadParams();
    controller_ = std::make_unique<pat_arm_nmpc::ArmNMPC>(mjcf_path, params_);

    if (controller_->numArmJoints() != static_cast<int>(model_joint_names_.size()))
        throw std::runtime_error(
            "pat_arm_nmpc: model_joint_names has " +
            std::to_string(model_joint_names_.size()) + " entries but the model has " +
            std::to_string(controller_->numArmJoints()) + " arm joints");

    // joint_names_[k] → index in model_joint_names_
    owned_model_idx_.reserve(joint_names_.size());
    for (const auto& jn : joint_names_) {
        auto it = std::find(model_joint_names_.begin(), model_joint_names_.end(), jn);
        if (it == model_joint_names_.end())
            throw std::runtime_error("pat_arm_nmpc: joint '" + jn +
                                     "' not found in model_joint_names");
        owned_model_idx_.push_back(
            static_cast<std::size_t>(std::distance(model_joint_names_.begin(), it)));
    }

    q_joints_.assign(model_joint_names_.size(), 0.0);
    v_joints_.assign(model_joint_names_.size(), 0.0);

    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/chaser/arm/joint_states", 10,
        std::bind(&ArmNMPCNode::onJointState, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/chaser/odom", 10,
        std::bind(&ArmNMPCNode::onOdom, this, std::placeholders::_1));
    ee_setpoint_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/" + arm_side_ + "/ee_setpoint", 10,
        std::bind(&ArmNMPCNode::onEeSetpoint, this, std::placeholders::_1));
    torque_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/chaser/arm/torque_command", 10);
    ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/" + arm_side_ + "/ee_pose", 10);
    ctrl_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / ctrl_hz),
                                  std::bind(&ArmNMPCNode::controlLoop, this));

    RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s]: model N=%d, owned=%zu, ee_site=%s, N=%d Ts=%.3f ctrl=%.0f Hz",
                arm_side_.c_str(), controller_->numArmJoints(), joint_names_.size(),
                params_.ee_site_name.c_str(), params_.N, params_.Ts, ctrl_hz);
}

// ─── Parameter loading ─────────────────────────────────────────────────────

std::vector<double> ArmNMPCNode::getJointArray(const std::string& name) {
    const auto v = declare_parameter<std::vector<double>>(name, std::vector<double>{});
    if (v.size() != joint_names_.size())
        throw std::runtime_error(
            name + ": expected " + std::to_string(joint_names_.size()) +
            " entries (one per joint_names), got " + std::to_string(v.size()));
    return v;
}

std::array<double, 3> ArmNMPCNode::getVec3(const std::string& name,
                                           std::array<double, 3> def) {
    const auto v = declare_parameter<std::vector<double>>(
        name, std::vector<double>{def[0], def[1], def[2]});
    if (v.size() != 3)
        throw std::runtime_error(name + ": expected 3 entries, got " +
                                 std::to_string(v.size()));
    return {v[0], v[1], v[2]};
}

pat_arm_nmpc::NMPCParams ArmNMPCNode::loadParams() {
    pat_arm_nmpc::NMPCParams p;

    p.ee_site_name  = declare_parameter<std::string>("ee_site_name", "arm_ee_site");
    p.owned_joints  = declare_parameter<std::vector<std::string>>(
        "owned_joints", std::vector<std::string>{});

    p.Ts            = declare_parameter<double>("nmpc.Ts", p.Ts);
    p.N             = declare_parameter<int>("nmpc.N", p.N);
    p.fullNonlinear = declare_parameter<bool>("nmpc.full_nonlinear", p.fullNonlinear);
    p.sqp_iters     = declare_parameter<int>("nmpc.sqp_iters", p.sqp_iters);
    p.Hg_damping    = declare_parameter<double>("nmpc.Hg_damping", p.Hg_damping);
    p.tau_max       = declare_parameter<double>("nmpc.tau_max", p.tau_max);
    p.rollout_v_clamp_mult =
        declare_parameter<double>("nmpc.rollout_v_clamp_mult", p.rollout_v_clamp_mult);
    p.terminal_velocity_constraint =
        declare_parameter<bool>("nmpc.terminal_velocity_constraint",
                                p.terminal_velocity_constraint);
    p.terminal_cost_multiplier =
        declare_parameter<double>("nmpc.terminal_cost_multiplier",
                                  p.terminal_cost_multiplier);

    const auto q_pos     = getVec3("weights.Q_pos",        {p.Qx, p.Qy, p.Qz});
    const auto q_vel     = getVec3("weights.Q_vel",        {p.Qdotx, p.Qdoty, p.Qdotz});
    const auto r_trans   = getVec3("weights.R",            {p.Rx, p.Ry, p.Rz});
    const auto q_ori     = getVec3("weights.Q_ori",        {p.Qwx, p.Qwy, p.Qwz});
    const auto q_angvel  = getVec3("weights.Q_angvel",     {p.Qdotwx, p.Qdotwy, p.Qdotwz});
    const auto r_ori     = getVec3("weights.R_ori",        {p.Rwx, p.Rwy, p.Rwz});
    const auto du_trans  = getVec3("weights.du_max_trans", {p.du_max_x, p.du_max_y, p.du_max_z});
    const auto du_rot    = getVec3("weights.du_max_rot",   {p.du_max_wx, p.du_max_wy, p.du_max_wz});
    p.Qx = q_pos[0];    p.Qy = q_pos[1];    p.Qz = q_pos[2];
    p.Qdotx = q_vel[0]; p.Qdoty = q_vel[1]; p.Qdotz = q_vel[2];
    p.Rx = r_trans[0];  p.Ry = r_trans[1];  p.Rz = r_trans[2];
    p.Qwx = q_ori[0];   p.Qwy = q_ori[1];   p.Qwz = q_ori[2];
    p.Qdotwx = q_angvel[0]; p.Qdotwy = q_angvel[1]; p.Qdotwz = q_angvel[2];
    p.Rwx = r_ori[0];   p.Rwy = r_ori[1];   p.Rwz = r_ori[2];
    p.du_max_x = du_trans[0];  p.du_max_y = du_trans[1];  p.du_max_z = du_trans[2];
    p.du_max_wx = du_rot[0];   p.du_max_wy = du_rot[1];   p.du_max_wz = du_rot[2];

    p.qp_max_iter    = declare_parameter<int>("qp.max_iter", p.qp_max_iter);
    p.qp_tol         = declare_parameter<double>("qp.tol", p.qp_tol);
    p.qp_reg_prim    = declare_parameter<double>("qp.reg_prim", p.qp_reg_prim);
    p.qp_warm_start  = declare_parameter<int>("qp.warm_start", p.qp_warm_start);
    p.joint_limit_slack_linear =
        declare_parameter<double>("qp.joint_slack_linear", p.joint_limit_slack_linear);
    p.joint_limit_slack_quadratic =
        declare_parameter<double>("qp.joint_slack_quadratic", p.joint_limit_slack_quadratic);
    p.torque_slack_linear =
        declare_parameter<double>("qp.torque_slack_linear", p.torque_slack_linear);
    p.torque_slack_quadratic =
        declare_parameter<double>("qp.torque_slack_quadratic", p.torque_slack_quadratic);

    // Per-joint limits: parallel arrays, entry i belongs to joint_names_[i].
    const auto q_min   = getJointArray("limits.q_min");
    const auto q_max   = getJointArray("limits.q_max");
    const auto qd_max  = getJointArray("limits.qd_max");
    const auto tau_max = getJointArray("limits.tau_max");
    p.joint_lims.clear();
    p.joint_lims.reserve(joint_names_.size());
    for (size_t i = 0; i < joint_names_.size(); ++i) {
        if (q_min[i] > q_max[i])
            throw std::runtime_error("limits: q_min > q_max for joint '" +
                                     joint_names_[i] + "'");
        p.joint_lims.push_back({joint_names_[i], q_min[i], q_max[i], qd_max[i], tau_max[i]});
    }

    return p;
}

// ─── Runtime ───────────────────────────────────────────────────────────────

void ArmNMPCNode::applyNamedValues(const sensor_msgs::msg::JointState::SharedPtr& msg,
                                   const std::vector<double>& values,
                                   const std::vector<std::string>& names,
                                   std::vector<double>& dst) {
    for (size_t k = 0; k < msg->name.size() && k < values.size(); ++k) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (msg->name[k] == names[i]) { dst[i] = values[k]; break; }
        }
    }
}

void ArmNMPCNode::onJointState(sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mu_);
    applyNamedValues(msg, msg->position, model_joint_names_, q_joints_);
    if (!msg->velocity.empty())
        applyNamedValues(msg, msg->velocity, model_joint_names_, v_joints_);
    has_state_ = true;
}

void ArmNMPCNode::onOdom(nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mu_);
    base_x_     = msg->pose.pose.position.x;
    base_y_     = msg->pose.pose.position.y;
    base_yaw_   = quatToYaw(msg->pose.pose.orientation);
    // simulation_node publishes world-frame linear velocity in the twist (it
    // copies the chaser slide-joint qvel directly), matching MuJoCo's
    // free-joint linvel convention.
    base_vx_    = msg->twist.twist.linear.x;
    base_vy_    = msg->twist.twist.linear.y;
    base_omega_ = msg->twist.twist.angular.z;
    has_odom_ = true;
}

void ArmNMPCNode::onEeSetpoint(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mu_);
    ee_x_ = msg->pose.position.x;
    ee_y_ = msg->pose.position.y;
    ee_quat_ = {msg->pose.orientation.w, msg->pose.orientation.x,
                msg->pose.orientation.y, msg->pose.orientation.z};
    has_setpoint_ = true;
}

void ArmNMPCNode::controlLoop() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!has_state_ || !has_odom_) {
        RCLCPP_WARN(get_logger(),
            "pat_arm_nmpc[%s]: controlLoop called but controller hasn't receieved state and/or odom information ", 
            arm_side_.c_str());
        return;
    }

    const int n = controller_->numArmJoints();
    Eigen::VectorXd q(7 + n), v(6 + n);
    q(0) = base_x_; q(1) = base_y_; q(2) = 0.0;
    q(3) = std::cos(base_yaw_ / 2.0);
    q(4) = 0.0; q(5) = 0.0;
    q(6) = std::sin(base_yaw_ / 2.0);
    v(0) = base_vx_; v(1) = base_vy_; v(2) = 0.0;
    v(3) = 0.0; v(4) = 0.0; v(5) = base_omega_;
    for (int i = 0; i < n; ++i) {
        q(7 + i) = q_joints_[static_cast<size_t>(i)];
        v(6 + i) = v_joints_[static_cast<size_t>(i)];
    }

    // Measured EE pose (world frame): published for downstream consumers
    // (target/mission nodes seed their waypoints off this) and, on the first
    // tick, used to seed a hold-pose setpoint.
    const auto meas = controller_->currentEePose(q, v);
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = get_clock()->now();
        ps.header.frame_id = "map";
        ps.pose.position.x = meas.pos.x();
        ps.pose.position.y = meas.pos.y();
        ps.pose.position.z = meas.pos.z();
        ps.pose.orientation.w = meas.quat[0];
        ps.pose.orientation.x = meas.quat[1];
        ps.pose.orientation.y = meas.quat[2];
        ps.pose.orientation.z = meas.quat[3];
        ee_pose_pub_->publish(ps);
    }

    if (!has_setpoint_) {
        ee_x_ = meas.pos.x();
        ee_y_ = meas.pos.y();
        ee_quat_ = {meas.quat[0], meas.quat[1], meas.quat[2], meas.quat[3]};
        has_setpoint_ = true;
        RCLCPP_INFO(get_logger(),
            "pat_arm_nmpc[%s]: no ee_setpoint yet — holding start pose "
            "(%.3f, %.3f)", arm_side_.c_str(), ee_x_, ee_y_);
    }

    controller_->setDesiredPos(ee_x_, ee_y_, 0.0);
    controller_->setDesiredOrient(ee_quat_[0], ee_quat_[1], ee_quat_[2], ee_quat_[3]);

    const Eigen::VectorXd tau = controller_->computeControl(q, v);

    sensor_msgs::msg::JointState cmd;
    cmd.header.stamp = get_clock()->now();
    cmd.name = joint_names_;
    cmd.effort.resize(joint_names_.size());
    for (size_t k = 0; k < joint_names_.size(); ++k)
        cmd.effort[k] = tau(static_cast<Eigen::Index>(owned_model_idx_[k]));
    torque_pub_->publish(cmd);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmNMPCNode>());
    rclcpp::shutdown();
}
