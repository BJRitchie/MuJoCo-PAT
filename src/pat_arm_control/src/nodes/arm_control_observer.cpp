// ArmControlNode: the disturbance observer, on its own realtime thread.
#include "arm_control_node.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pat_arm_control/control/controller_config.h"
#include "pat_rt/rt_thread.h"
#include "ros_param_source.h"

void ArmControlNode::startObserver() {
    // Off by default. Its estimate is published for scoring and, with
    // observer.ff_gain at 0, consumed by nothing -- so switching it on changes the
    // torque command not at all, which is what makes it safe to leave in a running
    // configuration while it is being tuned.
    RosParamSource src(*this);
    const pat_arm_control::ObserverConfig oc = pat_arm_control::loadObserverConfig(src);
    const std::string& obs_type = oc.type;
    const double obs_hz         = oc.hz;
    const double obs_gain       = oc.gain;
    const double obs_gain_base  = oc.gain_base;
    const double obs_mdot_clamp = oc.mdot_clamp;
    const double obs_est_clamp  = oc.estimate_clamp;
    obs_rt_priority_ = declare_parameter<int>("rt_priority_observer", 0);
    obs_rt_cpus_ = declare_parameter<std::vector<long>>("rt_cpus_observer",
                                                        std::vector<long>{});

    if (obs_type == "momentum") {
        if (!(obs_hz > 0.0)) {
            throw std::runtime_error(log_name_ + ": 'observer.hz' must be positive");
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
            std::bind(&ArmControlNode::onTorqueCommand, this, std::placeholders::_1));

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
        obs_thread_ = std::thread(&ArmControlNode::observerThread, this);
        RCLCPP_INFO(get_logger(),
                    "%s[%s]: momentum observer at %.0f Hz, gain %.1f "
                    "(base %.1f), estimate NOT fed to the control law",
                    log_name_.c_str(), arm_side_.c_str(), obs_hz, obs_gain,
                    obs_gain_base >= 0.0 ? obs_gain_base : obs_gain);
    } else if (obs_type != "none") {
        throw std::runtime_error(log_name_ + ": unknown 'observer' type '" +
                                 obs_type + "' (expected none or momentum)");
    }
}

bool ArmControlNode::observerStep() {
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
    // shared with the control thread but the read-only model.
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
    return true;
}

void ArmControlNode::observerThread() {
    // Affinity then policy, from inside the thread, same order and for the same
    // reasons as controlThread: the scheduler respects a mask but will not pick
    // a fast core on its own.
    obs_affinity_result_.store(pat_rt::applyAffinity(obs_rt_cpus_),
                               std::memory_order_relaxed);
    obs_rt_result_.store(pat_rt::applyRealtimePolicy(obs_rt_priority_),
                         std::memory_order_relaxed);

    pat_rt::PeriodicWaker waker(obs_period_ns_);
    waker.start();
    while (obs_running_.load(std::memory_order_acquire)) {
        waker.waitNext();
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

void ArmControlNode::publishDisturbance() {
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

void ArmControlNode::onTorqueCommand(sensor_msgs::msg::JointState::SharedPtr msg) {
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
