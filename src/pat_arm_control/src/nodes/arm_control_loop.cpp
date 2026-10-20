// ArmControlNode: the realtime control thread, and the drain that publishes
// its telemetry off it. Nothing on the control thread may allocate, log, or
// call into rclcpp (see controlThread's comment in the header).
#include "arm_control_node.h"

#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "pat_rt/rt_thread.h"

void ArmControlNode::controlThread() {
    // Both from inside the thread, so only this one is affected and nothing it
    // later creates inherits either. Affinity first: it decides which cores the
    // thread can use, and the realtime scheduler respects an affinity mask but
    // will not choose a fast core on its own. The results are latched for the
    // report, since this thread does no logging.
    affinity_result_.store(pat_rt::applyAffinity(rt_cpus_), std::memory_order_relaxed);
    rt_result_.store(pat_rt::applyRealtimePolicy(rt_priority_),
                     std::memory_order_relaxed);

    // An absolute-deadline schedule; an overrun steps to the next future
    // deadline rather than catching up, and recordWake counts the lost periods
    // as skipped. See pat_rt::PeriodicWaker.
    pat_rt::PeriodicWaker waker(period_ns_);
    waker.start();
    while (ctrl_running_.load(std::memory_order_acquire)) {
        waker.waitNext();
        if (!ctrl_running_.load(std::memory_order_acquire)) break;

        controlLoop();
    }
}

double ArmControlNode::threadCpuUs() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) * 1e6 + static_cast<double>(ts.tv_nsec) * 1e-3;
}

void ArmControlNode::controlLoop() {
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
    if (setpoint_mail_.read(sp) && !(hold_from_traj_ && sp.stamp <= traj_hold_end_)) {
        // (A setpoint from before a run-out trajectory ended is older than the
        // point the arm is holding, so it does not pull the arm back.)
        des_x_    = sp.x;
        des_y_    = sp.y;
        des_quat_ = {sp.quat[0], sp.quat[1], sp.quat[2], sp.quat[3]};
        have_des_ = true;
        hold_from_traj_ = false;
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

    // A reference trajectory, when one is in force. Copied only when a new
    // window has arrived (traj_seq_ moved), not every tick: it is 20 KB.
    const std::uint64_t seq = traj_seq_.load(std::memory_order_acquire);
    if (seq != traj_seq_seen_ && traj_mail_.read(ctrl_traj_)) {
        traj_seq_seen_ = seq;
        controller_->setReferenceWindow(ctrl_traj_);
        have_traj_ = true;
    }
    ref_source_ = hold_from_traj_ ? kRefTrajHeld : kRefSetpoint;
    if (have_traj_) {
        const double t_now = joints_.stamp;
        if (t_now <= ctrl_traj_.endTime()) {
            controller_->setReferenceTime(t_now);
            ref_source_ = kRefTrajectory;
        } else {
            // Run out: stop following it, and hold where it ended rather than
            // falling back to whatever setpoint preceded it.
            controller_->clearReferenceWindow();
            have_traj_ = false;
            const pat_arm_control::RefPoint& last = ctrl_traj_.p[ctrl_traj_.n - 1];
            double q[4];
            ctrl_traj_.orientation(last.yaw, q);
            des_x_ = last.x;
            des_y_ = last.y;
            des_quat_ = {q[0], q[1], q[2], q[3]};
            hold_from_traj_ = true;
            traj_hold_end_ = ctrl_traj_.endTime();
            ref_source_ = kRefTrajHeld;
        }
    }

    controller_->setDesiredPos(des_x_, des_y_, 0.0);
    controller_->setDesiredOrient(des_quat_[0], des_quat_[1], des_quat_[2], des_quat_[3]);

    // The disturbance feedforward, handed over BEFORE the tick so the command
    // and any torque bounds the law uses take the same value. A third reader of
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

void ArmControlNode::publishStats() {
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
    s.ref_source  = ref_source_;
    s.ref_ticks   = controller_->referenceTicks();
    s.ref_outside = controller_->referenceSamplesOutside();
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

void ArmControlNode::recordWake(std::chrono::steady_clock::time_point wake) {
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

void ArmControlNode::drainOutputs() {
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
