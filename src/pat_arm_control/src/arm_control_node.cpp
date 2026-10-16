#include "pat_arm_control/arm_control_node.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
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
    controller_ = std::make_unique<pat_arm_control::ArmNMPC>(mjcf_path, params_);

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
        
    // KEEP_LAST stated explicitly, because the control thread publishes this one
    // itself and KEEP_LAST is what makes that safe. A reliable writer whose reader
    // stops acking blocks up to max_blocking_time (100 ms by default) once its
    // history fills -- which would be ten missed deadlines -- but only under
    // KEEP_ALL. KEEP_LAST overwrites the oldest unacked sample instead and never
    // waits. Measured with the subscriber SIGSTOPped for 10 s: publish rose from
    // 17 to ~60 us mean, worst 209 us, and the loop held 99 ticks a window with
    // zero missed deadlines. Do NOT change this to KEEP_ALL, and do not drop back
    // to the bare `10` overload, whose depth-10 KEEP_LAST is only a default.
    torque_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        "/chaser/arm/torque_command", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    ee_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/" + arm_side_ + "/ee_pose", 10);

    // The control loop runs on its own thread (start()), not an executor timer.
    period_ns_ = static_cast<long>(1e9 / ctrl_hz);

    // 0 keeps the thread at normal priority, which is the default so that turning
    // realtime scheduling on is a deliberate, measurable change rather than
    // something a rebuild quietly alters. 70-80 sits above ordinary work and below
    // the kernel's own threads, which run at 99.
    rt_priority_ = declare_parameter<int>("rt_priority", 0);
    lock_memory_ = declare_parameter<bool>("lock_memory", false);
    rt_cpus_     = declare_parameter<std::vector<long>>("rt_cpus", std::vector<long>{});
                                  
    // Only ee_pose and the mode events come through here now, both of which
    // tolerate a drain interval of delay, so this runs at the control rate rather
    // than the ten-times-faster poll that the torque command needed.
    drain_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / ctrl_hz),
                                   std::bind(&ArmNMPCNode::drainOutputs, this));
    period_us_ = 1e6 / ctrl_hz;
    controller_->setTickBudgetUs(period_us_);

    // Sized here so the first tick does not allocate either. numArmJoints() is
    // fixed by the model the controller just loaded.
    q_.resize(7 + controller_->numArmJoints());
    v_.resize(6 + controller_->numArmJoints());

    // The torque message's fixed parts, set once: the tick only writes efforts
    // and the stamp into storage that already exists.
    torque_msg_.name = joint_names_;
    torque_msg_.effort.assign(joint_names_.size(), 0.0);

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

pat_arm_control::NMPCParams ArmNMPCNode::loadParams() {
    pat_arm_control::NMPCParams p;

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
    
    // The disturbance feedforward. Named under observer.* because it is
    // meaningless without an observer to feed it, and 0 by default so enabling
    // the observer still changes nothing about the command.
    p.ff_gain     = declare_parameter<double>("observer.ff_gain", p.ff_gain);
    p.ff_max_frac = declare_parameter<double>("observer.ff_max_frac", p.ff_max_frac);
    p.ff_slew     = declare_parameter<double>("observer.ff_slew", p.ff_slew);
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
    // Stop and join both realtime threads FIRST: they touch the controller, the
    // mailboxes and the trace buffer, all of which are about to be destroyed.
    // The observer goes first because it reads the controller (fullDynamics)
    // while the control thread only writes its own mailbox -- stopping the
    // reader before the thing it reads is the order that cannot race.
    obs_running_.store(false, std::memory_order_release);
    if (obs_thread_.joinable()) obs_thread_.join();
    ctrl_running_.store(false, std::memory_order_release);
    if (ctrl_thread_.joinable()) ctrl_thread_.join();

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

    // The disturbance feedforward, handed over BEFORE the solve so the command
    // and the QP's torque bounds use the same value. A third reader of
    // dist_mail_ alongside the publisher and the report, which a seqlock
    // supports. A read that loses its race, or an observer not yet converged,
    // leaves the retained value in place: the controller holds the last
    // feedforward rather than dropping to zero, since dropping to zero is
    // itself a torque step. With ff_gain at 0 this records and applies nothing.
    if (observer_ && dist_mail_.read(ctrl_dist_) && ctrl_dist_.ready) {
        const int n = static_cast<int>(model_joint_names_.size());
        for (int i = 0; i < n; ++i) {
            ff_in_(i) = static_cast<std::size_t>(i) < kMaxJoints
                            ? ctrl_dist_.d_joints[static_cast<std::size_t>(i)]
                            : 0.0;
        }
        controller_->setDisturbanceFeedforward(
            ff_in_, static_cast<double>(period_ns_) * 1e-9);
    }

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
        // Into the member message's existing storage, then straight out: no
        // allocation, and no queue between the command and the actuator.
        for (size_t k = 0; k < joint_names_.size(); ++k)
            torque_msg_.effort[k] = tau(static_cast<Eigen::Index>(owned_model_idx_[k]));
        timespec ts{};
        clock_gettime(CLOCK_REALTIME, &ts);
        torque_msg_.header.stamp.sec     = static_cast<std::int32_t>(ts.tv_sec);
        torque_msg_.header.stamp.nanosec = static_cast<std::uint32_t>(ts.tv_nsec);
        pat_arm_control::ScopedTimer timer(pub_torque_);
        torque_pub_->publish(torque_msg_);
    }

    // Hand the observer what was actually applied, after the non-finite guard
    // and after the controller's own per-joint clamping -- the planned torque
    // would make the clamped difference read as a disturbance. Indexed by this
    // node's owned joints, which is how observerStep reads it back. A mailbox
    // write is a few stores; it does not reach the allocator or a lock.
    if (observer_) {
        TauSnapshot s;
        s.n = static_cast<std::uint32_t>(
            std::min(joint_names_.size(), static_cast<std::size_t>(kMaxJoints)));
        for (std::uint32_t k = 0; k < s.n; ++k) {
            s.tau[k] = tau(static_cast<Eigen::Index>(owned_model_idx_[k]));
        }
        tau_own_mail_.write(s);
    }

    // Time the whole callback, not just computeControl: state assembly, both
    // publishes and the torque copy are all on the control thread and all count
    // against the period. Recorded here rather than in a ScopedTimer because the
    // deadline accounting needs the value.
    const double body_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - wake).count();
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

    publishStats();
}

void ArmNMPCNode::publishStats() {
    // One copy of everything the report needs, taken while this thread is the
    // only one touching any of it.
    StatsSnapshot s;
    s.jitter                   = jitter_;
    s.loop                     = loop_;
    s.cpu                      = cpu_;
    s.deadline_misses          = deadline_misses_;
    s.worst_consecutive_misses = worst_consecutive_misses_;
    s.skipped_periods          = skipped_periods_;
    controller_->snapshotProfile(s.profile);
    stats_mail_.write(s);

    // Clear only once the copy above is published, so the window the report is
    // about to read is the completed one rather than a half-cleared successor.
    // Every statistic is per report window: cumulative means hide a late spike,
    // since one 20 ms tick in an hour of 2 ms ticks moves the mean by nothing and
    // the spike is the whole point.
    if (stats_reset_.exchange(false, std::memory_order_acq_rel)) {
        jitter_.reset();
        loop_.reset();
        cpu_.reset();
        deadline_misses_          = 0;
        consecutive_misses_       = 0;
        worst_consecutive_misses_ = 0;
        skipped_periods_          = 0;
        controller_->resetProfile();
    }
}

void ArmNMPCNode::start() {
    // Locked before the thread starts, so a major fault cannot stall a tick.
    // Process-wide, which is why it lives here and not on the thread.
    //
    // MCL_ONFAULT is what makes the limit affordable: MCL_CURRENT|MCL_FUTURE
    // alone locks the whole ADDRESS SPACE, which for this node is ~620 MB of
    // mostly-untouched reservation against ~33 MB actually resident, and it fails
    // with ENOMEM against any sane RLIMIT_MEMLOCK. ONFAULT locks each page as it
    // is first touched instead, so the locked set tracks the resident set. The
    // cost is that the first touch of a page still faults -- the startup ticks
    // pay it, and everything the loop reaches stays locked thereafter.
    if (lock_memory_) {
        const int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT;
        mlock_result_.store(mlockall(flags) == 0 ? 0 : errno,
                            std::memory_order_relaxed);
    }
    ctrl_running_.store(true, std::memory_order_release);
    ctrl_thread_ = std::thread(&ArmNMPCNode::controlThread, this);

    // --- Disturbance observer ------------------------------------------- //
    // Off by default. Its estimate is published for scoring and nothing
    // consumes it, so switching it on changes the torque command not at all --
    // which is what makes it safe to leave in a running configuration while it
    // is being tuned.
    const std::string obs_type = declare_parameter<std::string>("observer", "none");
    const double obs_hz = declare_parameter<double>("observer.hz", 100.0);
    const double obs_gain = declare_parameter<double>("observer.gain", 20.0);
    const double obs_gain_base = declare_parameter<double>("observer.gain_base", -1.0);
    const double obs_mdot_clamp = declare_parameter<double>("observer.mdot_clamp", 0.0);
    const double obs_est_clamp = declare_parameter<double>("observer.estimate_clamp", 0.0);
    obs_rt_priority_ = declare_parameter<int>("rt_priority_observer", 0);
    obs_rt_cpus_ = declare_parameter<std::vector<long>>("rt_cpus_observer",
                                                        std::vector<long>{});

    if (obs_type == "momentum") {
        if (!(obs_hz > 0.0)) {
            throw std::runtime_error("pat_arm_nmpc: 'observer.hz' must be positive");
        }
        const int n = static_cast<int>(model_joint_names_.size());
        pat_arm_control::MomentumObserverParams op;
        op.nv = 6 + n;
        op.n_joints = n;
        op.dt = 1.0 / obs_hz;
        // One gain for the joint rows and, optionally, a different one for the
        // base. They are different physical quantities -- a base force in N
        // against a joint torque in N*m -- and the base rows sit on top of the
        // noisiest channel in the system (odom-derived base velocity), so being
        // able to slow them down independently is worth the extra parameter.
        op.gain = Eigen::VectorXd::Constant(op.nv, obs_gain);
        if (obs_gain_base >= 0.0) op.gain.head(6).setConstant(obs_gain_base);
        op.mdot_clamp = obs_mdot_clamp;
        op.estimate_clamp = obs_est_clamp;
        observer_ = std::make_unique<pat_arm_control::MomentumObserver>(op);

        // Size every observer-thread buffer here, so the thread itself never
        // reaches the allocator. fullDynamics sizes its own outputs on first
        // call; these are the ones it does not own.
        obs_q_.setZero(7 + n);
        obs_v_.setZero(6 + n);
        obs_tau_.setZero(6 + n);
        obs_d_joints_.setZero(n);
        ff_in_.setZero(n);
        obs_M_.setZero(6 + n, 6 + n);
        obs_bias_.setZero(6 + n);
        obs_R_.setZero(6, n);
        tau_joints_.assign(model_joint_names_.size(), 0.0);
        obs_period_ns_ = static_cast<long>(1e9 / obs_hz);

        // The shared torque_command topic, for the joints this node does not
        // own -- see TauSnapshot. Subscribing to a topic this node also
        // publishes is deliberate: it is the only place the other arm's applied
        // torque exists, and a momentum observer on the full model needs it.
        torque_echo_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/chaser/arm/torque_command", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
            std::bind(&ArmNMPCNode::onTorqueCommand, this, std::placeholders::_1));

        dist_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            "/chaser/arm/" + arm_side_ + "/disturbance",
            rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
            
        // This arm's joint rows, then the three planar base rows. The base
        // wrench is published because it is what drag and contact reaction
        // actually produce on the platform, and the reduction to joint space
        // discards it -- scoring the observer against an injected wrench needs
        // it. Named, so the vector describes itself.
        dist_msg_.name = joint_names_;
        for (const auto& jn : joint_names_) dist_msg_.name.push_back("raw_" + jn);
        dist_msg_.name.push_back("base_fx");
        dist_msg_.name.push_back("base_fy");
        dist_msg_.name.push_back("base_tz");
        dist_msg_.effort.assign(dist_msg_.name.size(), 0.0);

        obs_running_.store(true, std::memory_order_release);
        obs_thread_ = std::thread(&ArmNMPCNode::observerThread, this);
        RCLCPP_INFO(get_logger(),
                    "pat_arm_nmpc[%s]: momentum observer at %.0f Hz, gain %.1f "
                    "(base %.1f), estimate NOT fed to the control law",
                    arm_side_.c_str(), obs_hz, obs_gain,
                    obs_gain_base >= 0.0 ? obs_gain_base : obs_gain);
    } else if (obs_type != "none") {
        throw std::runtime_error("pat_arm_nmpc: unknown 'observer' type '" +
                                 obs_type + "' (expected none or momentum)");
    }
}

int ArmNMPCNode::applyRealtimePolicy(int priority) {
    if (priority <= 0) return -1;             // not requested
    sched_param sp{};
    sp.sched_priority = priority;
    // pthread_setschedparam returns the error rather than setting errno.
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
}

int ArmNMPCNode::applyAffinity(const std::vector<long>& cpus) {
    if (cpus.empty()) return -1;              // not requested
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const long cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(static_cast<int>(cpu), &set);
    }
    if (CPU_COUNT(&set) == 0) return EINVAL;   // every entry was out of range
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

void ArmNMPCNode::controlThread() {
    // Both from inside the thread, so only this one is affected and nothing it
    // later creates inherits either. Affinity first: it decides which cores the
    // thread can use, and the realtime scheduler respects an affinity mask but
    // will not choose a fast core on its own. The results are latched for the
    // report, since this thread does no logging.
    affinity_result_.store(applyAffinity(rt_cpus_), std::memory_order_relaxed);
    rt_result_.store(applyRealtimePolicy(rt_priority_), std::memory_order_relaxed);

    // Advance `next` by exactly one period. Written out rather than using
    // std::chrono so the deadline stays the timespec clock_nanosleep wants.
    const auto addPeriod = [this](timespec& t) {
        t.tv_nsec += period_ns_;
        while (t.tv_nsec >= 1000000000L) {
            t.tv_nsec -= 1000000000L;
            ++t.tv_sec;
        }
    };
    const auto isBefore = [](const timespec& a, const timespec& b) {
        return a.tv_sec < b.tv_sec ||
               (a.tv_sec == b.tv_sec && a.tv_nsec < b.tv_nsec);
    };

    timespec next{};
    clock_gettime(CLOCK_MONOTONIC, &next);
    while (ctrl_running_.load(std::memory_order_acquire)) {
        addPeriod(next);

        // Only if that deadline is ALREADY past did the last tick overrun, and
        // then the schedule steps forward to the next future one rather than
        // firing back-to-back to catch up: those periods are lost either way, and
        // running them at full rate only compounds the overrun. recordWake counts
        // them as skipped. Checked here, after the advance and before the sleep --
        // checking after the tick instead would always be true, since the deadline
        // just woken for is by definition in the past, and would step the schedule
        // twice per tick and halve the control rate.
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        while (isBefore(next, now)) addPeriod(next);

        // An absolute deadline, so a late tick does not shift every tick after it:
        // the schedule is fixed and lateness shows up as lateness. A relative
        // sleep (or rclcpp's wall timer) re-anchors instead, which hides the
        // overrun and reports the permanent offset as jitter.
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr)
               == EINTR) {
            // A signal, not a deadline change: wait out the rest of the period.
        }
        if (!ctrl_running_.load(std::memory_order_acquire)) break;

        controlLoop();
    }
}

bool ArmNMPCNode::observerStep() {
    // Same hand-off the control thread uses, read independently: a seqlock
    // supports any number of readers, each retrying on its own, so the observer
    // reads the SAME joint and base mailboxes rather than being given copies.
    // A read that loses its race leaves the retained snapshot in place, and the
    // step still runs on it -- one period of staleness, not a skipped step,
    // because a skipped step is a permanent offset in an integral.
    if (joint_mail_.read(obs_joints_)) obs_have_joints_ = true;
    if (base_mail_.read(obs_base_))    obs_have_base_   = true;
    if (!obs_have_joints_ || !obs_have_base_) {
        ++obs_no_state_;
        return false;
    }
    tau_own_mail_.read(obs_tau_own_);
    tau_ext_mail_.read(obs_tau_ext_);

    const int n = static_cast<int>(model_joint_names_.size());
    const int nv = 6 + n;

    // The free-floating base state, assembled exactly as controlLoop does it:
    // planar x/y and a yaw-only quaternion, the rest identically zero.
    obs_q_(0) = obs_base_.x;
    obs_q_(1) = obs_base_.y;
    obs_q_(2) = 0.0;
    obs_q_(3) = std::cos(obs_base_.yaw / 2.0);
    obs_q_(4) = 0.0;
    obs_q_(5) = 0.0;
    obs_q_(6) = std::sin(obs_base_.yaw / 2.0);
    obs_v_(0) = obs_base_.vx;
    obs_v_(1) = obs_base_.vy;
    obs_v_(2) = 0.0;
    obs_v_(3) = 0.0;
    obs_v_(4) = 0.0;
    obs_v_(5) = obs_base_.omega;
    for (int i = 0; i < n; ++i) {
        obs_q_(7 + i) = obs_joints_.q[static_cast<std::size_t>(i)];
        obs_v_(6 + i) = obs_joints_.v[static_cast<std::size_t>(i)];
    }

    // The applied torque, base rows zero because the base is unactuated. Start
    // from the topic, which covers every joint including the other arm's, then
    // overwrite the joints this node owns with the exact value the control
    // thread applied -- no DDS round trip on the entries that need not have one.
    obs_tau_.setZero();
    for (int i = 0; i < n && static_cast<std::uint32_t>(i) < obs_tau_ext_.n; ++i) {
        obs_tau_(6 + i) = obs_tau_ext_.tau[static_cast<std::size_t>(i)];
    }
    for (std::size_t k = 0; k < owned_model_idx_.size(); ++k) {
        if (static_cast<std::uint32_t>(k) >= obs_tau_own_.n) break;
        obs_tau_(6 + static_cast<Eigen::Index>(owned_model_idx_[k])) =
            obs_tau_own_.tau[k];
    }

    // The full-system dynamics on this thread's own workspace, so nothing is
    // shared with the control thread's rollout but the read-only model.
    controller_->fullDynamics(obs_q_, obs_v_, obs_ws_, obs_M_, obs_bias_, obs_R_);
    observer_->update(obs_M_, obs_bias_, obs_v_, obs_tau_);
    observer_->reduceToJoints(obs_R_, obs_d_joints_);

    DisturbanceSnapshot out;
    out.n = static_cast<std::uint32_t>(n);
    out.ready = observer_->ready();
    for (int i = 0; i < n && i < static_cast<int>(kMaxJoints); ++i) {
        out.d_joints[static_cast<std::size_t>(i)] = obs_d_joints_(i);
    }
    const Eigen::VectorXd& r = observer_->estimate();
    for (int i = 0; i < 6; ++i) out.d_base[static_cast<std::size_t>(i)] = r(i);
    // The joint rows of the full-system estimate, unreduced -- what
    // qfrc_constraint is directly comparable to.
    for (int i = 0; i < n && i < static_cast<int>(kMaxJoints); ++i) {
        out.d_raw[static_cast<std::size_t>(i)] = r(6 + i);
    }

    ++obs_steps_;
    out.steps        = obs_steps_;
    out.no_state     = obs_no_state_;
    out.max_mdot     = observer_->maxMdot();
    out.mdot_clamped = observer_->mdotClamped();
    out.est_clamped  = observer_->estimateClamped();
    dist_mail_.write(out);

    (void)nv;
    return true;
}

void ArmNMPCNode::observerThread() {
    // Affinity then policy, from inside the thread, same order and for the same
    // reasons as controlThread: the scheduler respects a mask but will not pick
    // a fast core on its own.
    obs_affinity_result_.store(applyAffinity(obs_rt_cpus_), std::memory_order_relaxed);
    obs_rt_result_.store(applyRealtimePolicy(obs_rt_priority_),
                         std::memory_order_relaxed);

    const auto addPeriod = [this](timespec& t) {
        t.tv_nsec += obs_period_ns_;
        while (t.tv_nsec >= 1000000000L) {
            t.tv_nsec -= 1000000000L;
            ++t.tv_sec;
        }
    };
    const auto isBefore = [](const timespec& a, const timespec& b) {
        return a.tv_sec < b.tv_sec ||
               (a.tv_sec == b.tv_sec && a.tv_nsec < b.tv_nsec);
    };

    timespec next{};
    clock_gettime(CLOCK_MONOTONIC, &next);
    while (obs_running_.load(std::memory_order_acquire)) {
        addPeriod(next);
        // Resync before sleeping, never after the step: checking afterwards is
        // always true -- the deadline just woken for is by definition past --
        // and steps the schedule twice per iteration, halving the rate. That
        // exact bug cost the control thread half its rate once already.
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        while (isBefore(next, now)) addPeriod(next);

        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr)
               == EINTR) {
        }
        if (!obs_running_.load(std::memory_order_acquire)) break;

        // A throw here must not take the thread down: the observer is an
        // accessory, and losing it should degrade the estimate rather than the
        // node. The estimate simply stops updating and the report says so.
        try {
            observerStep();
        } catch (const std::exception&) {
            observer_->reset();
            obs_have_joints_ = false;
            obs_have_base_   = false;
        }
    }
}

void ArmNMPCNode::publishDisturbance() {
    if (!dist_pub_) return;
    DisturbanceSnapshot d;
    if (!dist_mail_.read(d) || !d.ready) return;
    for (std::size_t k = 0; k < joint_names_.size(); ++k) {
        const std::size_t idx = owned_model_idx_[k];
        dist_msg_.effort[k] = idx < kMaxJoints ? d.d_joints[idx] : 0.0;
    }
    const std::size_t nj = joint_names_.size();
    // The same joints unreduced, for scoring against qfrc_constraint.
    for (std::size_t k = 0; k < nj; ++k) {
        const std::size_t idx = owned_model_idx_[k];
        dist_msg_.effort[nj + k] = idx < kMaxJoints ? d.d_raw[idx] : 0.0;
    }
    // The planar base rows, in the order the names declare: x force, y force,
    // yaw torque. Rows 2/3/4 are z force and x/y torque, which the table's
    // slide and hinge joints absorb and which therefore measure nothing.
    dist_msg_.effort[2 * nj + 0] = d.d_base[0];
    dist_msg_.effort[2 * nj + 1] = d.d_base[1];
    dist_msg_.effort[2 * nj + 2] = d.d_base[5];
    dist_msg_.header.stamp = get_clock()->now();
    dist_pub_->publish(dist_msg_);
}

void ArmNMPCNode::onTorqueCommand(sensor_msgs::msg::JointState::SharedPtr msg) {
    // The shared torque_command topic carries every arm's effort, name-matched,
    // so this is how the observer learns what the OTHER node instance applied.
    // It sees this node's own commands come back too; observerStep overwrites
    // those with the exact values rather than using the round trip.
    if (msg->effort.empty()) return;
    applyNamedValues(msg, msg->effort, model_joint_names_, tau_joints_);

    TauSnapshot s;
    s.n = static_cast<std::uint32_t>(
        std::min(tau_joints_.size(), static_cast<std::size_t>(kMaxJoints)));
    for (std::uint32_t i = 0; i < s.n; ++i) s.tau[i] = tau_joints_[i];
    tau_ext_mail_.write(s);
}

void ArmNMPCNode::drainOutputs() {
    // The measured EE pose only, and only because it is telemetry: delaying it by
    // a drain interval costs nothing, where delaying the torque command costs
    // actuation latency, so that one is published from the tick itself. Drains the
    // whole queue rather than one entry, so a late drain catches up instead of
    // falling further behind.
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
        pat_arm_control::ScopedTimer timer(pub_eepose_);
        ee_pose_pub_->publish(ps);
    }
    // The observer's estimate rides the same drain: it is telemetry too, and
    // publishing it from the observer thread would put rclcpp's allocator and
    // locks on a realtime path for no benefit.
    publishDisturbance();
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

    jitter_.add(std::abs(dt_us - period_us_));
    // Round down, so an interval inside one period counts as nothing skipped.
    const long periods = static_cast<long>(dt_us / period_us_);
    if (periods > 1) skipped_periods_ += periods - 1;
}

void ArmNMPCNode::reportDiagnostics() {
    // What the control thread got, said once. Reported rather than enforced: a
    // controller that will not start without a realtime policy is useless in a
    // development container, so it runs either way and says which it is.
    if (!rt_reported_.exchange(true, std::memory_order_acq_rel)) {
        const int rt = rt_result_.load(std::memory_order_relaxed);
        if (rt == 0) {
            RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s]: control thread at SCHED_FIFO priority %d",
                arm_side_.c_str(), rt_priority_);
        } else if (rt > 0) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: SCHED_FIFO priority %d refused (%s) — running at "
                "normal priority. Needs RLIMIT_RTPRIO >= %d (docker-compose.yml's "
                "ulimits.rtprio); this is NOT a realtime loop",
                arm_side_.c_str(), rt_priority_, std::strerror(rt), rt_priority_);
        } else {
            RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s]: control thread at normal priority "
                "(rt_priority is 0)", arm_side_.c_str());
        }
        const int aff = affinity_result_.load(std::memory_order_relaxed);
        if (aff == 0) {
            std::string cpus;
            for (const long c : rt_cpus_) cpus += std::to_string(c) + " ";
            RCLCPP_INFO(get_logger(), "pat_arm_nmpc[%s]: control thread pinned to CPUs %s",
                        arm_side_.c_str(), cpus.c_str());
        } else if (aff > 0) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: CPU affinity failed (%s)",
                arm_side_.c_str(), std::strerror(aff));
        } else if (rt_result_.load(std::memory_order_relaxed) == 0) {
            // FIFO without a mask is the case that measured as a regression.
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: SCHED_FIFO with no rt_cpus — on a host whose "
                "cores differ in speed the realtime scheduler may park this thread "
                "on the slowest ones. Set rt_cpus to the fast cores",
                arm_side_.c_str());
        }
        const int ml = mlock_result_.load(std::memory_order_relaxed);
        if (ml == 0) {
            RCLCPP_INFO(get_logger(), "pat_arm_nmpc[%s]: memory locked (mlockall)",
                        arm_side_.c_str());
        } else if (ml > 0) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: mlockall failed (%s) — a page fault can still "
                "stall a tick. Needs ulimits.memlock",
                arm_side_.c_str(), std::strerror(ml));
        }
    }

    // The observer thread's own placement, reported once and separately: the
    // two threads are only useful on DIFFERENT cores, so saying which each got
    // is what makes a misconfiguration visible rather than merely slow.
    if (observer_ && !obs_reported_.exchange(true, std::memory_order_acq_rel)) {
        const int rt = obs_rt_result_.load(std::memory_order_relaxed);
        if (rt == 0) {
            RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s]: observer thread at SCHED_FIFO priority %d",
                arm_side_.c_str(), obs_rt_priority_);
        } else if (rt > 0) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: observer SCHED_FIFO priority %d refused (%s) — "
                "running at normal priority",
                arm_side_.c_str(), obs_rt_priority_, std::strerror(rt));
        }
        const int aff = obs_affinity_result_.load(std::memory_order_relaxed);
        if (aff == 0) {
            std::string cpus;
            for (const long c : obs_rt_cpus_) cpus += std::to_string(c) + " ";
            RCLCPP_INFO(get_logger(),
                        "pat_arm_nmpc[%s]: observer thread pinned to CPUs %s",
                        arm_side_.c_str(), cpus.c_str());
        } else if (aff > 0) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: observer CPU affinity failed (%s)",
                arm_side_.c_str(), std::strerror(aff));
        } else if (!obs_rt_cpus_.empty() || !rt_cpus_.empty()) {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s]: observer has no rt_cpus_observer — it may share "
                "a core with the control thread, which is most of what having its "
                "own thread was for",
                arm_side_.c_str());
        }
    }

    // The observer's own window. Separate from the control report because they
    // run on separate clocks: a step the observer skipped is a permanent offset
    // in its integral, so `no_state` rising is a correctness problem and not
    // merely a dropped sample.
    if (observer_) {
        DisturbanceSnapshot d;
        if (dist_mail_.read(d)) {
            RCLCPP_INFO(get_logger(),
                "pat_arm_nmpc[%s] observer: %ld steps (%ld without state), "
                "d_base [%.3f %.3f %.3f] d_joints[0] %.3f | max|Mdot| %.1f, "
                "clamped Mdot %ld est %ld%s",
                arm_side_.c_str(), d.steps, d.no_state,
                d.d_base[0], d.d_base[1], d.d_base[5],
                d.n > 0 ? d.d_joints[0] : 0.0,
                d.max_mdot, d.mdot_clamped, d.est_clamped,
                d.ready ? "" : " [not yet converged]");
        } else {
            RCLCPP_WARN(get_logger(),
                "pat_arm_nmpc[%s] observer: no estimate published yet",
                arm_side_.c_str());
        }
    }

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
    // Only telemetry goes through a queue now, so a drop here costs a missing
    // diagnostic rather than a stale actuator command -- still worth saying,
    // because it means the drain is not keeping up with the tick.
    const long dropped_ee = n_ee_pose_dropped_.exchange(0, std::memory_order_relaxed);
    const long dropped_ev = n_mode_events_dropped_.exchange(0, std::memory_order_relaxed);
    if (dropped_ee || dropped_ev) {
        RCLCPP_WARN(get_logger(),
            "pat_arm_nmpc[%s]: outbound queue overflow — dropped %ld ee_pose, "
            "%ld mode events; the drain is not keeping up",
            arm_side_.c_str(), dropped_ee, dropped_ev);
    }

    // Tier transitions, in the order the control loop saw them. Severity follows
    // the tier: a Tier 1 replay is a handled transient and should not read like a
    // fault.
    using Mode = pat_arm_control::ArmNMPC::ControlMode;
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

    // Before the early return below: a window with no completed tick still has a
    // previous window's rows to get out, and the trace should not stall on it.
    flushTrace();

    StatsSnapshot s;
    if (!stats_mail_.read(s)) return;   // no tick has published yet
    if (s.loop.count == 0) return;      // no tick completed in this window

    RCLCPP_INFO(get_logger(),
        "pat_arm_nmpc[%s] %ld ticks @ %.0f us | period jitter mean %.0f max %.0f us | "
        "callback wall mean %.0f max %.0f / cpu mean %.0f max %.0f us | "
        "deadline misses %ld (longest run %ld) | skipped periods %ld\n%s",
        arm_side_.c_str(), s.loop.count, period_us_,
        s.jitter.mean(), s.jitter.max_us,
        s.loop.mean(), s.loop.max_us, s.cpu.mean(), s.cpu.max_us,
        s.deadline_misses, s.worst_consecutive_misses, s.skipped_periods,
        pat_arm_control::ArmNMPC::formatProfile(s.profile).c_str());

    // What publish() costs, with the histogram: the mean decides nothing here, the
    // worst case decides whether this call can go back on the control thread.
    if (pub_torque_.count > 0 || pub_eepose_.count > 0) {
        std::ostringstream pub;
        pub.setf(std::ios::fixed);
        pub.precision(1);
        const std::pair<const char*, const pat_arm_control::PhaseStat*> both[] = {
            {"torque", &pub_torque_}, {"ee_pose", &pub_eepose_},
        };
        for (const auto& [name, st] : both) {
            if (st->count == 0) continue;
            pub << "  " << name << "(n=" << st->count << " mean=" << st->mean()
                << " max=" << st->max_us << ") ";
        }
        pub << "\n  hist(publish_us) ";
        for (int i = 0; i < pat_arm_control::PhaseStat::kBins; ++i) {
            if (i + 1 == pat_arm_control::PhaseStat::kBins) pub << ">5000:";
            else pub << "<=" << pat_arm_control::PhaseStat::kEdgeUs[i] << ":";
            pub << (pub_torque_.bins[i] + pub_eepose_.bins[i]) << " ";
        }
        RCLCPP_INFO(get_logger(), "pat_arm_nmpc[%s] publish us%s",
                    arm_side_.c_str(), pub.str().c_str());
        pub_torque_.reset();
        pub_eepose_.reset();
    }

    // Ask the control thread to start a fresh window. It clears its own counters
    // right after publishing the next snapshot, so the reset never races a read.
    stats_reset_.store(true, std::memory_order_release);
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
    // After construction, so DDS has already created its threads at normal
    // priority and cannot inherit the control thread's policy.
    node->start();
    rclcpp::spin(node);
    rclcpp::shutdown();
}
