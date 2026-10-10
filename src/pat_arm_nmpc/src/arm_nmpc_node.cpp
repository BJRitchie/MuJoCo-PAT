#include "pat_arm_nmpc/arm_nmpc_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>

#include <Eigen/Dense>

using namespace std::chrono_literals;

// Unnamed namespace: helpers private to this file. Unnamed rather than named
// deliberately — it gives them internal linkage, so they never reach the linker
// and cannot collide with a same-named helper in another translation unit.
namespace {
double quatToYaw(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

/*! True when every element is finite — used to reject a poisoned measurement at
 *  the subscription instead of latching it into the cached state. */
bool allFinite(const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(),
                       [](double x) { return std::isfinite(x); });
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

    // See the header's own comment on base_anchor_x_/base_anchor_y_: corrects
    // /chaser/odom's chaser-body-anchor-relative x/y into true world-frame
    // coordinates, matching this MJCF's <body name="chaser" pos="...">.
    // Default 0,0 (no correction) for a model whose chaser has no such
    // anchor offset.
    base_anchor_x_ = declare_parameter<double>("base_anchor_x", 0.0);
    base_anchor_y_ = declare_parameter<double>("base_anchor_y", 0.0);

    params_ = loadParams();
    // Tier 1 replays a plan whose stages are Ts apart on ticks that are this far
    // apart, so the controller has to know the period. It comes from the timer rate
    // rather than the YAML so the two can never disagree.
    params_.control_period = 1.0 / ctrl_hz;
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

    // JointSnapshot carries fixed-size arrays so the control thread can copy a
    // whole measurement without allocating. Check the bound here rather than
    // trusting it at every write: a model with more joints than this would
    // otherwise overrun the snapshot silently.
    if (model_joint_names_.size() > kMaxJoints) {
        throw std::runtime_error(
            "pat_arm_nmpc: model_joint_names has " +
            std::to_string(model_joint_names_.size()) + " entries, more than the " +
            std::to_string(kMaxJoints) + " a JointSnapshot holds — raise "
            "ArmNMPCNode::kMaxJoints in arm_nmpc_node.h");
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
                                  
    // Publishing asynchronously costs actuation latency: a command computed at t
    // is not published until the next drain, so the drain period IS added delay
    // in the control loop. Ten times the control rate keeps the worst case at a
    // tenth of a period (1 ms at 100 Hz) and the mean at half that, which is
    // small against the solve itself. A timer that usually finds both queues
    // empty is nearly free, so the rate is set by the latency, not by the cost.
    //
    // Polling is the simple version. Signalling a dedicated publisher thread
    // from the tick would cut the delay to a thread wakeup, and belongs with the
    // realtime-thread change rather than here.
    drain_tmr_ = create_wall_timer(std::chrono::duration<double>(0.1 / ctrl_hz),
                                   std::bind(&ArmNMPCNode::drainOutputs, this));
    period_us_ = 1e6 / ctrl_hz;
    controller_->setTickBudgetUs(period_us_);

    // Sized here so the first tick does not allocate either. numArmJoints() is
    // fixed by the model the controller just loaded.
    q_.resize(7 + controller_->numArmJoints());
    v_.resize(6 + controller_->numArmJoints());

    // Per-tick trace. The default capacity is 10 minutes at the control rate; the
    // buffer is allocated now and recording simply stops when it is full, so a
    // long run cannot grow memory without bound or allocate on a tick.
    trace_file_ = declare_parameter<std::string>("trace_file", "");
    // Only has to hold what accumulates between drains (1 s), with generous slack.
    const int trace_cap = declare_parameter<int>(
        "trace_capacity", static_cast<int>(60.0 * ctrl_hz));
    if (!trace_file_.empty() && trace_cap > 0) {
        // One file per arm: two nodes sharing a path would interleave their rows
        // into nonsense, so the side goes in before the extension.
        std::string path = trace_file_;
        const auto dot = path.find_last_of('.');
        const auto sep = path.find_last_of('/');
        if (dot != std::string::npos && (sep == std::string::npos || dot > sep))
            path.insert(dot, "_" + arm_side_);
        else
            path += "_" + arm_side_ + ".csv";

        trace_out_.open(path);
        if (!trace_out_) {
            RCLCPP_ERROR(get_logger(), "pat_arm_nmpc[%s]: cannot open %s; not tracing",
                         arm_side_.c_str(), path.c_str());
        } else {
            trace_out_ << "# arm=" << arm_side_ << " period_us=" << period_us_
                       << " N=" << params_.N << " Ts=" << params_.Ts
                       << " warm_start=" << params_.qp_warm_start
                       << " qp_max_iter=" << params_.qp_max_iter << "\n"
                       << "t_us,tick_us,dynamics,jdotqdot,lambda,qpsolve,"
                          "rollout_dynamics,rollout_integrate,pinv,build_model,"
                          "defect,trust_region,finalize,hpipm_us,qp_iters,"
                          "qp_status,qp_iters1,qp_status1,mode,retried\n";
            trace_rows_.reserve(static_cast<std::size_t>(trace_cap));
            controller_->enableTrace(static_cast<std::size_t>(trace_cap));
            RCLCPP_INFO(get_logger(),
                        "pat_arm_nmpc[%s]: tracing every tick to %s",
                        arm_side_.c_str(), path.c_str());
        }
    }
    diag_tmr_  = create_wall_timer(std::chrono::seconds(1),
                                   std::bind(&ArmNMPCNode::reportDiagnostics, this));

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
    p.safe_mode_damping_frac =
        declare_parameter<double>("safe_mode.damping_frac", p.safe_mode_damping_frac);
    p.safe_mode_recovery_ticks =
        declare_parameter<int>("safe_mode.recovery_ticks", p.safe_mode_recovery_ticks);
    p.tier1_max_replay_ticks =
        declare_parameter<int>("safe_mode.tier1_max_replay_ticks", p.tier1_max_replay_ticks);
    p.tier3_latch_after_ticks =
        declare_parameter<int>("safe_mode.tier3_latch_after_ticks", p.tier3_latch_after_ticks);
    // control_period is not read here: it comes from control_hz, which the node's
    // own timer owns. The constructor fills it in before building the controller.
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
    p.Qx = q_pos[0];    p.Qy = q_pos[1];    p.Qz = q_pos[2];
    p.Qdotx = q_vel[0]; p.Qdoty = q_vel[1]; p.Qdotz = q_vel[2];
    p.Rx = r_trans[0];  p.Ry = r_trans[1];  p.Rz = r_trans[2];
    p.Qwx = q_ori[0];   p.Qwy = q_ori[1];   p.Qwz = q_ori[2];
    p.Qdotwx = q_angvel[0]; p.Qdotwy = q_angvel[1]; p.Qdotwz = q_angvel[2];
    p.Rwx = r_ori[0];   p.Rwy = r_ori[1];   p.Rwz = r_ori[2];
    p.du_max_frac = declare_parameter<double>("nmpc.du_max_frac", p.du_max_frac);

    // The planar law's θz channel is built from Qwz / Qdotwz / Rwz (see
    // ArmNMPC::controlLaw), but the documented contract (nmpc.yaml header) is
    // that weights.Q_pos / Q_vel / R are 3-vectors [x, y, θz] and Q_ori /
    // Q_angvel / R_ori are unused. Map the θz entries onto the fields the law
    // reads, otherwise the yaw weights a user tunes are silently ignored and
    // yaw runs with Q_ori/Q_angvel/R_ori's defaults instead (undamped, near-
    // zero effort cost against a plant gain ~100x the x/y channels', which
    // makes the SQP iteration diverge at larger Ts).
    p.Qwz    = q_pos[2];
    p.Qdotwz = q_vel[2];
    p.Rwz    = r_trans[2];

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
    p.soft_joint_limits =
        declare_parameter<bool>("qp.soft_joint_limits", p.soft_joint_limits);
    p.soft_torque_limits =
        declare_parameter<bool>("qp.soft_torque_limits", p.soft_torque_limits);
    p.relax_box_to_contain_x0 =
        declare_parameter<bool>("qp.relax_box_to_contain_x0", p.relax_box_to_contain_x0);
    p.relax_box_margin =
        declare_parameter<double>("qp.relax_box_margin", p.relax_box_margin);

    // Per-joint position/velocity limits: parallel arrays, entry i belongs to
    // joint_names_[i] — this node's OWN joints, the ones it box-constrains.
    const auto q_min   = getJointArray("limits.q_min");
    const auto q_max   = getJointArray("limits.q_max");
    const auto qd_max  = getJointArray("limits.qd_max");
    p.joint_lims.clear();
    p.joint_lims.reserve(joint_names_.size());
    for (size_t i = 0; i < joint_names_.size(); ++i) {
        if (q_min[i] > q_max[i])
            throw std::runtime_error("limits: q_min > q_max for joint '" +
                                     joint_names_[i] + "'");
        p.joint_lims.push_back({joint_names_[i], q_min[i], q_max[i], qd_max[i]});
    }

    // Torque limits are a property of the model's actuators, not of which arm
    // this node drives, so they are set once for every model joint in the
    // shared parameter block — one array, one place, both nodes.
    const auto model_tau_max = declare_parameter<std::vector<double>>(
        "model_tau_max", std::vector<double>{});
    if (model_tau_max.size() != model_joint_names_.size()) {
        throw std::runtime_error(
            "model_tau_max: expected " + std::to_string(model_joint_names_.size()) +
            " entries (one per model_joint_names), got " +
            std::to_string(model_tau_max.size()));
    }
    p.model_torque_lims.clear();
    p.model_torque_lims.reserve(model_joint_names_.size());
    for (size_t i = 0; i < model_joint_names_.size(); ++i) {
        p.model_torque_lims.emplace_back(model_joint_names_[i], model_tau_max[i]);
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
    // Reject a non-finite measurement rather than latching it: once it is in
    // q_joints_/v_joints_ it poisons every later tick, since these arrays are
    // only ever overwritten, never invalidated.
    if (!allFinite(msg->position) || !allFinite(msg->velocity)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), log_clock_, 1000,
            "pat_arm_nmpc[%s]: dropping joint_states with non-finite values",
            arm_side_.c_str());
        return;
    }
    applyNamedValues(msg, msg->position, model_joint_names_, q_joints_);
    if (!msg->velocity.empty())
        applyNamedValues(msg, msg->velocity, model_joint_names_, v_joints_);

    // Publish the accumulated arrays, not the message: a JointState need not
    // name every joint, so the running copy is the complete state and the
    // message is only the update to it.
    JointSnapshot snap;
    snap.n = static_cast<std::uint32_t>(q_joints_.size());
    for (std::size_t i = 0; i < q_joints_.size(); ++i) {
        snap.q[i] = q_joints_[i];
        snap.v[i] = v_joints_[i];
    }
    joint_mail_.write(snap);
}

void ArmNMPCNode::onOdom(nav_msgs::msg::Odometry::SharedPtr msg) {
    const auto& pp = msg->pose.pose.position;
    const auto& tl = msg->twist.twist.linear;

    if (!std::isfinite(pp.x) || !std::isfinite(pp.y) ||
        !std::isfinite(tl.x) || !std::isfinite(tl.y) ||
        !std::isfinite(msg->twist.twist.angular.z)) {
            
        RCLCPP_ERROR_THROTTLE(get_logger(), log_clock_, 1000,
            "pat_arm_nmpc[%s]: dropping odom with non-finite values",
            arm_side_.c_str());
        return;
    }
    BaseSnapshot snap;
    // + base_anchor_{x,y}: see the header's comment on why odom's raw value
    // isn't the true world position on its own.
    snap.x     = base_anchor_x_ + msg->pose.pose.position.x;
    snap.y     = base_anchor_y_ + msg->pose.pose.position.y;
    snap.yaw   = quatToYaw(msg->pose.pose.orientation);
    // simulation_node publishes world-frame linear velocity in the twist (it
    // copies the chaser slide-joint qvel directly), matching MuJoCo's
    // free-joint linvel convention.
    snap.vx    = msg->twist.twist.linear.x;
    snap.vy    = msg->twist.twist.linear.y;
    snap.omega = msg->twist.twist.angular.z;
    base_mail_.write(snap);
}

void ArmNMPCNode::onEeSetpoint(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    SetpointSnapshot snap;
    snap.x = msg->pose.position.x;
    snap.y = msg->pose.position.y;
    snap.quat[0] = msg->pose.orientation.w;
    snap.quat[1] = msg->pose.orientation.x;
    snap.quat[2] = msg->pose.orientation.y;
    snap.quat[3] = msg->pose.orientation.z;
    setpoint_mail_.write(snap);
}

ArmNMPCNode::~ArmNMPCNode() {
    // A destructor that lets an exception out during shutdown terminates the
    // process with no diagnostic, so swallow everything. The ofstream flushes
    // itself; this only catches the rows recorded since the last drain. Logging is
    // already gone by now, hence no message here.
    try {
        flushTrace();
    } catch (...) {
    }
}

void ArmNMPCNode::flushTrace() {
    if (!trace_out_ || !controller_) return;
    controller_->drainTrace(trace_rows_);
    if (trace_rows_.empty()) return;

    trace_out_ << std::fixed << std::setprecision(2);
    for (const auto& r : trace_rows_) {
        trace_out_ << r.t_us << ',' << r.tick_us << ',' << r.dynamics << ','
                   << r.jdotqdot << ',' << r.lambda << ',' << r.qpsolve << ','
                   << r.rollout_dynamics << ',' << r.rollout_integrate << ','
                   << r.pinv << ',' << r.build_model << ',' << r.defect << ','
                   << r.trust_region << ',' << r.finalize << ',' << r.hpipm_us
                   << ',' << r.qp_iters << ',' << r.qp_status
                   << ',' << r.qp_iters1 << ',' << r.qp_status1 << ','
                   << static_cast<int>(r.mode) << ','
                   << static_cast<int>(r.retried) << '\n';
    }
    trace_out_.flush();
    trace_written_ += static_cast<long>(trace_rows_.size());
}

double ArmNMPCNode::threadCpuUs() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e6 + static_cast<double>(ts.tv_nsec) * 1e-3;
}

void ArmNMPCNode::controlLoop() {
    const auto wake    = std::chrono::steady_clock::now();
    const double cpu_0 = threadCpuUs();
    recordWake(wake);

    // Take one coherent copy of each input up front. A failed read means either
    // nothing has arrived yet or the producer was mid-write, and in both cases
    // the previous copy still stands -- have_* latches once the first real value
    // lands, so the loop never runs on a default-constructed state.
    if (joint_mail_.read(joints_)) have_joints_ = true;
    if (base_mail_.read(base_))    have_base_   = true;
    if (!have_joints_ || !have_base_) {
        // Counted, not logged: the report turns this into one throttled line.
        n_waiting_for_input_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const int n = controller_->numArmJoints();
    // Sized once; resize is a no-op at an unchanged size, so no allocation after
    // the first tick.
    q_.resize(7 + n);
    v_.resize(6 + n);
    Eigen::VectorXd& q = q_;
    Eigen::VectorXd& v = v_;
    q(0) = base_.x; q(1) = base_.y; q(2) = 0.0;
    q(3) = std::cos(base_.yaw / 2.0);
    q(4) = 0.0; q(5) = 0.0;
    q(6) = std::sin(base_.yaw / 2.0);
    v(0) = base_.vx; v(1) = base_.vy; v(2) = 0.0;
    v(3) = 0.0; v(4) = 0.0; v(5) = base_.omega;
    for (int i = 0; i < n; ++i) {
        q(7 + i) = joints_.q[static_cast<size_t>(i)];
        v(6 + i) = joints_.v[static_cast<size_t>(i)];
    }

    // Measured EE pose (world frame): published for downstream consumers
    // (target/mission nodes seed their waypoints off this) and, on the first
    // tick, used to seed a hold-pose setpoint.
    const auto meas = controller_->currentEePose(q, v);
    {
        EePoseOut out;
        out.pos[0] = meas.pos.x();
        out.pos[1] = meas.pos.y();
        out.pos[2] = meas.pos.z();
        for (int i = 0; i < 4; ++i) out.quat[i] = meas.quat[i];
        if (!ee_pose_out_.push(out))
            n_ee_pose_dropped_.fetch_add(1, std::memory_order_relaxed);
    }

    // A published setpoint always wins; otherwise hold wherever the arm started,
    // captured once. Both land in the loop's own desired pose, so a tick is never
    // tracking a target that changed halfway through it.
    SetpointSnapshot sp;
    if (setpoint_mail_.read(sp)) {
        des_x_    = sp.x;
        des_y_    = sp.y;
        des_quat_ = {sp.quat[0], sp.quat[1], sp.quat[2], sp.quat[3]};
        have_des_ = true;
    } else if (!have_des_) {
        des_x_    = meas.pos.x();
        des_y_    = meas.pos.y();
        des_quat_ = {meas.quat[0], meas.quat[1], meas.quat[2], meas.quat[3]};
        have_des_ = true;
        // Left for the report to log: this is the control thread.
        seeded_x_ = des_x_;
        seeded_y_ = des_y_;
        seeded_hold_pose_.store(true, std::memory_order_release);
    }

    controller_->setDesiredPos(des_x_, des_y_, 0.0);
    controller_->setDesiredOrient(des_quat_[0], des_quat_[1], des_quat_[2], des_quat_[3]);

    // computeControl handles its own numerical failures by entering safe mode,
    // so an exception reaching here means something outside the control law
    // itself. Damp rather than let it escape spin() and kill the process: the
    // simulator (and a real motor driver) holds the last torque it was sent, so
    // dying mid-motion leaves a torque applied indefinitely.
    Eigen::VectorXd tau;
    try {
        tau = controller_->computeControl(q, v);
    } catch (const std::exception& e) {
        tau = controller_->dampingTorque(v);
        // Copied into a fixed buffer rather than logged: no formatting, no
        // allocation, no write from the control thread.
        std::snprintf(threw_msg_, kMsgLen, "%s", e.what());
        n_tick_threw_.fetch_add(1, std::memory_order_relaxed);
    }

    // Queue tier transitions for the report to log, once each rather than per
    // tick. Detected here because a 1 Hz sample of mode() would miss any
    // transition that came and went inside a window.
    const auto mode = controller_->mode();
    if (mode != prev_mode_) {
        ModeEvent ev;
        ev.mode            = static_cast<int>(mode);
        ev.qp_status       = controller_->lastQpStatus();
        ev.tier1_ticks     = controller_->tier1Ticks();
        ev.safe_mode_ticks = controller_->safeModeTicks();
        ev.tier1_entries   = controller_->tier1Entries();
        if (!mode_events_.push(ev))
            n_mode_events_dropped_.fetch_add(1, std::memory_order_relaxed);
        prev_mode_ = mode;
    }

    // Last line of defence: never hand a non-finite effort to an actuator that
    // will latch it. dampingTorque is finite by construction, so this should be
    // unreachable — which is exactly why it is worth asserting.
    if (!tau.allFinite()) {
        tau.setZero();
        n_nonfinite_torque_.fetch_add(1, std::memory_order_relaxed);
    }

    {
        TorqueOut out;
        out.n = static_cast<std::uint32_t>(joint_names_.size());
        for (size_t k = 0; k < joint_names_.size(); ++k)
            out.effort[k] = tau(static_cast<Eigen::Index>(owned_model_idx_[k]));
        // A dropped torque leaves the actuator on the previous command for an
        // extra period, so this is counted rather than ignored.
        if (!torque_out_.push(out))
            n_torque_dropped_.fetch_add(1, std::memory_order_relaxed);
    }

    // Time the whole callback, not just computeControl: state assembly, both
    // publishes and the torque copy are all on the control thread and all count
    // against the period. Recorded here rather than in a ScopedTimer because the
    // deadline accounting needs the value.
    const double body_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - wake).count();
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        loop_.add(body_us);
        cpu_.add(threadCpuUs() - cpu_0);
        if (period_us_ > 0.0 && body_us > period_us_) {
            ++deadline_misses_;
            ++consecutive_misses_;
            worst_consecutive_misses_ =
                std::max(worst_consecutive_misses_, consecutive_misses_);
        } else {
            consecutive_misses_ = 0;
        }
    }
}

void ArmNMPCNode::drainOutputs() {
    // Everything the tick produced since the last drain, published from this
    // thread so the control path never calls into the middleware. Drains the
    // whole queue rather than one entry, so a late drain catches up instead of
    // falling further behind.
    TorqueOut t;
    while (torque_out_.pop(t)) {
        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = get_clock()->now();
        cmd.name = joint_names_;
        cmd.effort.resize(t.n);
        for (std::uint32_t k = 0; k < t.n; ++k) cmd.effort[k] = t.effort[k];
        torque_pub_->publish(cmd);
    }

    EePoseOut p;
    while (ee_pose_out_.pop(p)) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = get_clock()->now();
        ps.header.frame_id = "map";
        ps.pose.position.x = p.pos[0];
        ps.pose.position.y = p.pos[1];
        ps.pose.position.z = p.pos[2];
        ps.pose.orientation.w = p.quat[0];
        ps.pose.orientation.x = p.quat[1];
        ps.pose.orientation.y = p.quat[2];
        ps.pose.orientation.z = p.quat[3];
        ee_pose_pub_->publish(ps);
    }
}

void ArmNMPCNode::recordWake(std::chrono::steady_clock::time_point wake) {
    if (!have_last_wake_) {
        last_wake_      = wake;
        have_last_wake_ = true;
        return;
    }
    const double dt_us =
        std::chrono::duration<double, std::micro>(wake - last_wake_).count();
    last_wake_ = wake;
    if (period_us_ <= 0.0) return;

    // last_wake_/have_last_wake_ above belong to this thread alone; jitter_ and
    // skipped_periods_ are what the report reads and resets.
    std::lock_guard<std::mutex> lk(stats_mu_);
    jitter_.add(std::abs(dt_us - period_us_));
    // Round down, so an interval inside one period counts as nothing skipped.
    const long periods = static_cast<long>(dt_us / period_us_);
    if (periods > 1) skipped_periods_ += periods - 1;
}

void ArmNMPCNode::reportDiagnostics() {
    // Everything the control tick recorded but could not log. Drained first, so
    // a window that then returns early below still gets these out.
    if (seeded_hold_pose_.exchange(false, std::memory_order_acq_rel)) {
        RCLCPP_INFO(get_logger(),
            "pat_arm_nmpc[%s]: no ee_setpoint yet — holding start pose (%.3f, %.3f)",
            arm_side_.c_str(), seeded_x_, seeded_y_);
    }
    if (const long n = n_waiting_for_input_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_WARN(get_logger(),
            "pat_arm_nmpc[%s]: waiting for joint_states and/or odom (%ld ticks)",
            arm_side_.c_str(), n);
    }
    if (const long n = n_tick_threw_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_ERROR(get_logger(),
            "pat_arm_nmpc[%s]: control tick threw %ld time(s), last: %s "
            "— commanded joint damping",
            arm_side_.c_str(), n, threw_msg_);
    }
    if (const long n = n_nonfinite_torque_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_ERROR(get_logger(),
            "pat_arm_nmpc[%s]: non-finite torque suppressed %ld time(s), commanded zero",
            arm_side_.c_str(), n);
    }
    // A drop here means the drain did not keep up with the tick, which costs a
    // held-stale actuator command -- worth surfacing loudly, not as a statistic.
    const long dropped_tau = n_torque_dropped_.exchange(0, std::memory_order_relaxed);
    const long dropped_ee  = n_ee_pose_dropped_.exchange(0, std::memory_order_relaxed);
    const long dropped_ev  = n_mode_events_dropped_.exchange(0, std::memory_order_relaxed);
    if (dropped_tau || dropped_ee || dropped_ev) {
        RCLCPP_ERROR(get_logger(),
            "pat_arm_nmpc[%s]: outbound queue overflow — dropped %ld torque, "
            "%ld ee_pose, %ld mode events; the publisher is not keeping up",
            arm_side_.c_str(), dropped_tau, dropped_ee, dropped_ev);
    }

    // Tier transitions, in the order the control loop saw them. Severity follows
    // the tier: a Tier 1 replay is a handled transient and should not read like a
    // fault.
    using Mode = pat_arm_nmpc::ArmNMPC::ControlMode;
    ModeEvent ev;
    while (mode_events_.pop(ev)) {
        switch (static_cast<Mode>(ev.mode)) {
        case Mode::PlanReplay:
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: TIER 1 — solve failed (QP status %d), replaying the "
                "last accepted plan",
                arm_side_.c_str(), ev.qp_status);
            break;
        case Mode::Damping:
            RCLCPP_ERROR(get_logger(),
                "pat_arm_nmpc[%s]: TIER 2 — joint damping, task abandoned "
                "(QP status %d, %ld replay ticks used so far)",
                arm_side_.c_str(), ev.qp_status, ev.tier1_ticks);
            break;
        case Mode::Latched:
            RCLCPP_ERROR(get_logger(),
                "pat_arm_nmpc[%s]: TIER 3 — LATCHED in damping; the solver did not "
                "recover and this will NOT clear on its own",
                arm_side_.c_str());
            break;
        case Mode::Nominal:
            RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s]: resumed tracking (%ld damped ticks, %ld replay "
                "ticks over %ld Tier 1 entries, in total)",
                arm_side_.c_str(), ev.safe_mode_ticks, ev.tier1_ticks,
                ev.tier1_entries);
            break;
        }
    }

    std::lock_guard<std::mutex> lk(stats_mu_);
    // Before the early return below: a window with no completed tick still has a
    // previous window's rows to get out, and the trace should not stall on it.
    flushTrace();
    if (loop_.count == 0) return;   // no tick completed in this window

    RCLCPP_INFO(get_logger(),
        "pat_arm_nmpc[%s] %ld ticks @ %.0f us | period jitter mean %.0f max %.0f us | "
        "callback wall mean %.0f max %.0f / cpu mean %.0f max %.0f us | "
        "deadline misses %ld (longest run %ld) | skipped periods %ld\n%s",
        arm_side_.c_str(), loop_.count, period_us_,
        jitter_.mean(), jitter_.max_us,
        loop_.mean(), loop_.max_us, cpu_.mean(), cpu_.max_us,
        deadline_misses_, worst_consecutive_misses_, skipped_periods_,
        controller_->profileReport().c_str());

    // Every statistic is per report window. Cumulative means hide a late spike:
    // one 20 ms tick in an hour of 2 ms ticks moves the mean by nothing, and the
    // spike is the whole point.
    jitter_.reset();
    loop_.reset();
    cpu_.reset();
    deadline_misses_          = 0;
    consecutive_misses_       = 0;
    worst_consecutive_misses_ = 0;
    skipped_periods_          = 0;
    controller_->resetProfile();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    // Construction throws stay fatal — a misconfigured controller must refuse to
    // run rather than degrade — but report them through the logger and a clean
    // exit code instead of an uncaught terminate, which prints nothing useful.
    std::shared_ptr<ArmNMPCNode> node;
    try {
        node = std::make_shared<ArmNMPCNode>();
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("pat_arm_nmpc"),
                     "failed to construct node: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::spin(node);
    rclcpp::shutdown();
}
