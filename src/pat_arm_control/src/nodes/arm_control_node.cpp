// ArmControlNode: construction, the subscriptions, and start-up.
// The control thread is in arm_control_loop.cpp, the observer's in
// arm_control_observer.cpp, the report in arm_control_report.cpp.
#include "arm_control_node.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include "pat_arm_control/control/controller_config.h"
#include "pat_rt/rt_thread.h"
#include "ros_param_source.h"

// Unnamed namespace: helpers private to this file. Unnamed rather than named
// deliberately — it gives them internal linkage, so they never reach the linker
// and cannot collide with a same-named helper in another translation unit.
namespace {
/*! A stamp or duration as seconds, by plain arithmetic. Not rclcpp::Time or
 *  rclcpp::Duration: those throw on a negative value, and nothing a message
 *  carries may be able to take the controller down. */
double toSeconds(std::int32_t sec, std::uint32_t nanosec) {
    return static_cast<double>(sec) + 1e-9 * static_cast<double>(nanosec);
}

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

ArmControlNode::ArmControlNode(const ControlLaw& law)
    : Node("pat_arm_" + law.tag), log_name_("pat_arm_" + law.tag) {
    arm_side_ = declare_parameter<std::string>("arm_side", "");
    if (arm_side_.empty())
        throw std::runtime_error(log_name_ + ": 'arm_side' is required (\"left\"/\"right\")");

    const std::string mjcf_path = declare_parameter<std::string>("mjcf_path", "");
    if (mjcf_path.empty())
        throw std::runtime_error(log_name_ + ": 'mjcf_path' is required");

    // Everything the controller reads goes through the shared loaders
    // (controller_config.h and the law's own), the same ones the benches use,
    // so a key means the same thing to both.
    RosParamSource src(*this);
    const pat_arm_control::ArmJoints joints = pat_arm_control::loadArmJoints(src);
    joint_names_       = joints.owned;
    model_joint_names_ = joints.model;

    const double ctrl_hz = declare_parameter<double>("control_hz", 50.0);

    // See the header's own comment on base_anchor_x_/base_anchor_y_: corrects
    // /chaser/odom's chaser-body-anchor-relative x/y into true world-frame
    // coordinates, matching this MJCF's <body name="chaser" pos="...">.
    const pat_arm_control::BaseAnchor anchor = pat_arm_control::loadBaseAnchor(src);
    base_anchor_x_ = anchor.x;
    base_anchor_y_ = anchor.y;

    // The law. Its control period comes from the timer rate rather than the
    // YAML, so the two can never disagree.
    LawInstance inst = law.make(mjcf_path, src, joints, 1.0 / ctrl_hz);
    controller_ = std::move(inst.controller);

    if (controller_->numArmJoints() != static_cast<int>(model_joint_names_.size()))
        throw std::runtime_error(
            log_name_ + ": model_joint_names has " +
            std::to_string(model_joint_names_.size()) + " entries but the model has " +
            std::to_string(controller_->numArmJoints()) + " arm joints");

    // joint_names_[k] → index in model_joint_names_ (loadArmJoints has already
    // checked every owned joint is a model joint).
    owned_model_idx_.reserve(joint_names_.size());
    for (const auto& jn : joint_names_) {
        auto it = std::find(model_joint_names_.begin(), model_joint_names_.end(), jn);
        owned_model_idx_.push_back(
            static_cast<std::size_t>(std::distance(model_joint_names_.begin(), it)));
    }

    // JointSnapshot carries fixed-size arrays so the control thread can copy a
    // whole measurement without allocating. Check the bound here rather than
    // trusting it at every write: a model with more joints than this would
    // otherwise overrun the snapshot silently.
    if (model_joint_names_.size() > kMaxJoints) {
        throw std::runtime_error(
            log_name_ + ": model_joint_names has " +
            std::to_string(model_joint_names_.size()) + " entries, more than the " +
            std::to_string(kMaxJoints) + " a JointSnapshot holds — raise "
            "ArmControlNode::kMaxJoints in arm_control_node.h");
    }

    q_joints_.assign(model_joint_names_.size(), 0.0);
    v_joints_.assign(model_joint_names_.size(), 0.0);

    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/chaser/arm/joint_states", 10,
        std::bind(&ArmControlNode::onJointState, this, std::placeholders::_1));

    // Where the base state comes from. /chaser/odom is the measurement itself
    // (mocap in the lab, exact in simulation); /chaser/nav/odom is pat_gnc's
    // nav_node estimate fused from the IMU, star tracker and relative-nav fix,
    // which is what flight provides. Same message, same meaning, so nothing
    // downstream changes with the choice.
    const std::string odom_topic =
        declare_parameter<std::string>("odom_topic", "/chaser/odom");
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 10,
        std::bind(&ArmControlNode::onOdom, this, std::placeholders::_1));
    ee_setpoint_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/chaser/arm/" + arm_side_ + "/ee_setpoint", 10,
        std::bind(&ArmControlNode::onEeSetpoint, this, std::placeholders::_1));
    // Depth 1: only the newest window matters, and each one supersedes the
    // last entirely.
    traj_sub_ = create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
        "/chaser/arm/" + arm_side_ + "/ee_trajectory", rclcpp::QoS(1).reliable(),
        std::bind(&ArmControlNode::onEeTrajectory, this, std::placeholders::_1));

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

    // Only ee_pose and the mode events come through here, both of which tolerate
    // a drain interval of delay, so this runs at the control rate.
    drain_tmr_ = create_wall_timer(std::chrono::duration<double>(1.0 / ctrl_hz),
                                   std::bind(&ArmControlNode::drainOutputs, this));
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

    // Per-tick trace. The buffer is allocated now and recording simply stops
    // when it is full, so a long run cannot grow memory without bound or
    // allocate on a tick.
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
            RCLCPP_ERROR(get_logger(), "%s[%s]: cannot open %s; not tracing",
                         log_name_.c_str(), arm_side_.c_str(), path.c_str());
        } else {
            trace_out_ << "# arm=" << arm_side_ << " period_us=" << period_us_
                       << inst.trace_meta << "\n"
                       << "t_us,tick_us,dynamics,jdotqdot,lambda,qpsolve,"
                          "rollout_dynamics,rollout_integrate,pinv,build_model,"
                          "defect,trust_region,finalize,hpipm_us,qp_iters,"
                          "qp_status,qp_iters1,qp_status1,mode,retried\n";
            trace_rows_.reserve(static_cast<std::size_t>(trace_cap));
            controller_->enableTrace(static_cast<std::size_t>(trace_cap));
            RCLCPP_INFO(get_logger(),
                        "%s[%s]: tracing every tick to %s",
                        log_name_.c_str(), arm_side_.c_str(), path.c_str());
        }
    }
    diag_tmr_  = create_wall_timer(std::chrono::seconds(1),
                                   std::bind(&ArmControlNode::reportDiagnostics, this));

    RCLCPP_INFO(get_logger(),
                "%s[%s]: model N=%d, owned=%zu, ee_site=%s, %s ctrl=%.0f Hz",
                log_name_.c_str(), arm_side_.c_str(), controller_->numArmJoints(),
                joint_names_.size(), inst.ee_site.c_str(), inst.summary.c_str(), ctrl_hz);
}

ArmControlNode::~ArmControlNode() {
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

void ArmControlNode::start() {
    // Locked before the thread starts, so a major fault cannot stall a tick.
    // Process-wide, which is why it lives here and not on the thread. See
    // pat_rt::lockMemory for why it is MCL_ONFAULT.
    if (lock_memory_) {
        mlock_result_.store(pat_rt::lockMemory(), std::memory_order_relaxed);
    }
    ctrl_running_.store(true, std::memory_order_release);
    ctrl_thread_ = std::thread(&ArmControlNode::controlThread, this);

    startObserver();
}

// ─── Subscriptions ─────────────────────────────────────────────────────────

void ArmControlNode::applyNamedValues(const sensor_msgs::msg::JointState::SharedPtr& msg,
                                      const std::vector<double>& values,
                                      const std::vector<std::string>& names,
                                      std::vector<double>& dst) {
    for (size_t k = 0; k < msg->name.size() && k < values.size(); ++k) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (msg->name[k] == names[i]) { dst[i] = values[k]; break; }
        }
    }
}

void ArmControlNode::onJointState(sensor_msgs::msg::JointState::SharedPtr msg) {
    // Reject a non-finite measurement rather than latching it: once it is in
    // q_joints_/v_joints_ it poisons every later tick, since these arrays are
    // only ever overwritten, never invalidated.
    if (!allFinite(msg->position) || !allFinite(msg->velocity)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), log_clock_, 1000,
            "%s[%s]: dropping joint_states with non-finite values",
            log_name_.c_str(), arm_side_.c_str());
        return;
    }
    applyNamedValues(msg, msg->position, model_joint_names_, q_joints_);
    if (!msg->velocity.empty())
        applyNamedValues(msg, msg->velocity, model_joint_names_, v_joints_);

    // Publish the accumulated arrays, not the message: a JointState need not
    // name every joint, so the running copy is the complete state and the
    // message is only the update to it.
    JointSnapshot snap;
    snap.stamp = toSeconds(msg->header.stamp.sec, msg->header.stamp.nanosec);
    snap.n = static_cast<std::uint32_t>(q_joints_.size());
    for (std::size_t i = 0; i < q_joints_.size(); ++i) {
        snap.q[i] = q_joints_[i];
        snap.v[i] = v_joints_[i];
    }
    joint_mail_.write(snap);
}

void ArmControlNode::onOdom(nav_msgs::msg::Odometry::SharedPtr msg) {
    const auto& pp = msg->pose.pose.position;
    const auto& tl = msg->twist.twist.linear;

    if (!std::isfinite(pp.x) || !std::isfinite(pp.y) ||
        !std::isfinite(tl.x) || !std::isfinite(tl.y) ||
        !std::isfinite(msg->twist.twist.angular.z)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), log_clock_, 1000,
            "%s[%s]: dropping odom with non-finite values",
            log_name_.c_str(), arm_side_.c_str());
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

void ArmControlNode::onEeSetpoint(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    SetpointSnapshot snap;
    snap.x = msg->pose.position.x;
    snap.y = msg->pose.position.y;
    snap.quat[0] = msg->pose.orientation.w;
    snap.quat[1] = msg->pose.orientation.x;
    snap.quat[2] = msg->pose.orientation.y;
    snap.quat[3] = msg->pose.orientation.z;
    snap.stamp = toSeconds(msg->header.stamp.sec, msg->header.stamp.nanosec);
    setpoint_mail_.write(snap);
}

void ArmControlNode::onEeTrajectory(
    trajectory_msgs::msg::MultiDOFJointTrajectory::SharedPtr msg) {
    using pat_arm_control::ReferenceWindow;
    // thread_local rather than a local: the window is 20 KB, too much to put on
    // the stack of every callback.
    static thread_local ReferenceWindow w;
    // toSeconds, not rclcpp::Time: a malformed window is rejected and counted
    // below, never thrown on.
    const double t0 = toSeconds(msg->header.stamp.sec, msg->header.stamp.nanosec);
    const std::size_t n = std::min(msg->points.size(), ReferenceWindow::kMaxPoints);
    if (msg->points.size() > ReferenceWindow::kMaxPoints) {
        traj_truncated_.fetch_add(1, std::memory_order_relaxed);
    }
    const auto reject = [this] {
        traj_rejected_.fetch_add(1, std::memory_order_relaxed);
    };
    if (n == 0 || msg->points[0].transforms.empty()) return reject();

    // The first point's orientation becomes the window's reference; every
    // point's yaw is its rotation about world Z relative to it. The EE frame
    // is rolled relative to the world, so carrying the orientation as a yaw
    // on a reference quaternion is what keeps a target tilt-free.
    const auto& r0 = msg->points[0].transforms[0].rotation;
    w.quat0[0] = r0.w; w.quat0[1] = r0.x; w.quat0[2] = r0.y; w.quat0[3] = r0.z;
    w.yaw0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& pt = msg->points[i];
        if (pt.transforms.empty()) return reject();
        auto& p = w.p[i];
        p.t = t0 + toSeconds(pt.time_from_start.sec, pt.time_from_start.nanosec);
        if (i > 0 && !(p.t > w.p[i - 1].t)) return reject();   // must increase
        const auto& tr = pt.transforms[0];
        p.x = tr.translation.x;
        p.y = tr.translation.y;
        // Yaw of q_i * conj(q0): the Z rotation that takes the reference
        // orientation to this point's.
        const auto& q = tr.rotation;
        const double rw =  q.w * r0.w + q.x * r0.x + q.y * r0.y + q.z * r0.z;
        const double rz = -q.w * r0.z - q.x * r0.y + q.y * r0.x + q.z * r0.w;
        p.yaw = 2.0 * std::atan2(rz, rw);
        p.vx = p.vy = p.wz = p.ax = p.ay = p.az = 0.0;
        if (!pt.velocities.empty()) {
            p.vx = pt.velocities[0].linear.x;
            p.vy = pt.velocities[0].linear.y;
            p.wz = pt.velocities[0].angular.z;
        }
        if (!pt.accelerations.empty()) {
            p.ax = pt.accelerations[0].linear.x;
            p.ay = pt.accelerations[0].linear.y;
            p.az = pt.accelerations[0].angular.z;
        }
        const double vals[] = {p.t, p.x, p.y, p.yaw, p.vx, p.vy, p.wz, p.ax, p.ay, p.az};
        for (const double v : vals) {
            if (!std::isfinite(v)) return reject();
        }
    }
    w.n = static_cast<std::uint32_t>(n);
    traj_mail_.write(w);
    traj_seq_.fetch_add(1, std::memory_order_release);
}
