#include "pat_arm_control/control/task_space_controller.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace pat_arm_control {

TaskSpaceController::TaskSpaceController(const std::string& mjcf_path,
                                         const ControllerParams& params)
    : FloatingBaseDynamics(mjcf_path)
    , ctrl_params_(params) {
    const ControllerParams& p = ctrl_params_;
    if (p.ee_site_name.empty()) {
        throw std::runtime_error(
            "TaskSpaceController: ee_site_name must be set");
    }
    if (p.safe_mode_damping_frac <= 0.0) {
        throw std::runtime_error(
            "TaskSpaceController: safe_mode_damping_frac must be > 0 — safe mode "
            "with zero damping would leave the arm coasting.");
    }
    if (p.safe_mode_recovery_ticks < 1) {
        throw std::runtime_error(
            "TaskSpaceController: safe_mode_recovery_ticks must be >= 1");
    }
    if (p.ff_gain < 0.0 || p.ff_gain > 1.0) {
        throw std::runtime_error(
            "TaskSpaceController: ff_gain must be in [0, 1] — it is the fraction "
            "of the disturbance estimate to cancel, not a controller gain.");
    }
    if (p.ff_max_frac <= 0.0 || p.ff_max_frac > 1.0) {
        throw std::runtime_error(
            "TaskSpaceController: ff_max_frac must be in (0, 1] — the "
            "feedforward has to be bounded by some fraction of the torque "
            "limit, and a feedforward allowed the whole limit leaves the "
            "tracking law none.");
    }
    if (p.ff_slew < 0.0) {
        throw std::runtime_error(
            "TaskSpaceController: ff_slew must not be negative (0 disables it)");
    }
    ee_site_id_ = siteIdFromName(p.ee_site_name);

    if (p.joint_lims.empty()) {
        throw std::runtime_error("TaskSpaceController: joint_lims must be set");
    }
    for (const auto& lims : p.joint_lims) {
        qlims[lims.name] = std::make_pair(lims.q_min, lims.q_max);
        vlims[lims.name] = lims.qd_max;
    }
    for (const auto& [jname, tau] : p.model_torque_lims) {
        torque_lims[jname] = tau;
    }

    // Re-index by arm-local joint index (0..n_joints_-1) for cheap per-tick
    // lookups instead of re-hashing joint names every control tick.
    //
    // Joints NOT listed in joint_lims (the other arm's, when one controller
    // owns a subset of a multi-arm model) are never box-constrained. They DO
    // appear in any all-model-joint torque constraint and rollout clamp, so
    // every joint needs a real torque limit -- model_torque_lims supplies one
    // per model joint and is required to be complete, so there is no fallback
    // magnitude to guess at (and none to silently disagree with the config).
    //
    // Position ranges come from the MJCF, which is what MuJoCo actually
    // enforces, so every joint gets a real range rather than a placeholder --
    // including the ones this controller does not own.
    modelQlimsByIndex = modelJointRanges();
    qlimsByIndex      = modelQlimsByIndex;   // tightened by config just below
    vlimsByIndex.assign(n_joints_, 10.0);
    torqueLimsByIndex.assign(n_joints_, -1.0);   // sentinel: must all be filled below
    for (const auto& [jname, tau] : torque_lims) {
        if (tau <= 0.0) {
            throw std::runtime_error(
                "TaskSpaceController: model_torque_lims entry for joint \"" + jname +
                "\" must be > 0, got " + std::to_string(tau));
        }
        torqueLimsByIndex[jointIndexFromName(jname)] = tau;
    }
    for (int i = 0; i < n_joints_; ++i) {
        if (torqueLimsByIndex[i] < 0.0) {
            throw std::runtime_error(
                "TaskSpaceController: model_torque_lims is missing arm-local joint "
                "index " + std::to_string(i) + " — it must name every joint in the "
                "model, not only the ones this controller owns (torque constraints "
                "and rollout clamps cover all of them).");
        }
    }
    // Eigen mirror of the same limits, built here because torque bounds need
    // them as a vector at every node and they do not change after construction.
    tau_lim_vec_.resize(n_joints_);
    for (int i = 0; i < n_joints_; ++i) tau_lim_vec_(i) = torqueLimsByIndex[i];

    // Disturbance feedforward. Sized and zeroed here so the control path never
    // allocates, and so the default (ff_gain 0) is a literal no-op: d_ff_ stays
    // zero and the expressions that use it reduce to their no-feedforward form.
    d_ff_.setZero(n_joints_);
    ff_max_vec_.resize(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
        ff_max_vec_(i) = p.ff_max_frac * torqueLimsByIndex[i];
    }

    for (const auto& [jname, bounds] : qlims) {
        int idx = jointIndexFromName(jname);
        // Intersect rather than replace: a configured range may tighten the
        // model's (a software limit inside the mechanical one) but must not
        // widen it past what MuJoCo will enforce.
        qlimsByIndex[idx].first  = std::max(qlimsByIndex[idx].first,  bounds.first);
        qlimsByIndex[idx].second = std::min(qlimsByIndex[idx].second, bounds.second);
        vlimsByIndex[idx]        = vlims.at(jname);
    }

    // Resolve which arm-local joint indices this controller owns. Empty
    // owned_joints means "all joints".
    ownedJointInds.clear();
    if (p.owned_joints.empty()) {
        ownedJointInds.resize(n_joints_);
        std::iota(ownedJointInds.begin(), ownedJointInds.end(), 0);
    } else {
        for (const auto& jname : p.owned_joints)
            ownedJointInds.push_back(jointIndexFromName(jname));
    }

    n_owned = static_cast<int>(ownedJointInds.size());
    if (n_owned == 0) {
        throw std::runtime_error(
            "TaskSpaceController: owned_joints resolved to zero joints — "
            "this controller would command nothing.");
    }

    // Safe-mode damping gains, owned joints only (the node publishes no effort
    // for the rest). Scaling by tau_max/qd_max keeps the damping torque inside
    // damping_frac*tau_max for any speed within qd_max, so safe mode never
    // relies on the output clamp to stay physical.
    kd_.assign(n_joints_, 0.0);
    for (int idx : ownedJointInds) {
        if (vlimsByIndex[idx] <= 0.0) {
            throw std::runtime_error(
                "TaskSpaceController: qd_max must be > 0 for owned joint at "
                "arm-local index " + std::to_string(idx) +
                " — safe-mode damping is scaled by it.");
        }
        kd_[idx] = p.safe_mode_damping_frac * torqueLimsByIndex[idx] / vlimsByIndex[idx];
    }

    // A one-step law by default; a law with a horizon calls this again.
    setReferenceNodes(1, 0.0);
}

// === One tick ================================================================

Eigen::VectorXd TaskSpaceController::computeControl(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    loadLiveState(q, v);        // writes mj_->data->qpos/qvel + mj_forward
    return controlLaw(q, v);    // reads the dynamics helpers off mj_->data
}

TaskSpaceController::EePose TaskSpaceController::currentEePose(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    loadLiveState(q, v);
    return { eePosition(ee_site_id_), eeOrientation(ee_site_id_) };
}

Eigen::VectorXd TaskSpaceController::controlLaw(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    try {
        return controlLawImpl(q, v);
    } catch (const std::exception&) {
        // A non-finite intermediate (checkNaN), a solver size mismatch, an Eigen
        // failure, or a MuJoCo error via the installed hook. No plan replay here
        // on purpose: the kinematics this tick produced are exactly what is
        // suspect. Damping depends on none of that machinery and is finite by
        // construction even if v is not.
        return failToDamping(v);
    }
}

// === Reference ===============================================================

void TaskSpaceController::setReferenceNodes(int n_nodes, double spacing) {
    node_ref_.resize(static_cast<std::size_t>(n_nodes));
    ref_spacing_ = spacing;
}

void TaskSpaceController::setReferenceWindow(const ReferenceWindow& window) {
    window_ = window;   // fixed-size copy into existing storage
    have_window_ = window_.n > 0;
}

void TaskSpaceController::planarRows(const Eigen::MatrixXd& J6, Eigen::MatrixXd& J3)
{
    J3.resize(3, J6.cols());
    J3.row(0) = J6.row(0);   // x
    J3.row(1) = J6.row(1);   // y
    J3.row(2) = J6.row(5);   // angular-Z
}

double TaskSpaceController::orientationErrorZ(const double ref_quat[4],
                                              const Eigen::Vector4d& ee_quat)
{
    mjtNum quat_d[4] = {ref_quat[0], ref_quat[1], ref_quat[2], ref_quat[3]};
    mjtNum quat_ee_arr[4] = {ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]};
    mjtNum e_orient_local[3];
    mju_subQuat(e_orient_local, quat_d, quat_ee_arr);
    mjtNum e_orient_world[3];
    mju_rotVecQuat(e_orient_world, e_orient_local, quat_ee_arr);
    return e_orient_world[2];
}

TaskSpaceController::NodeRef TaskSpaceController::staticReference() const {
    // desiredPos[2] (Z) and the X/Y angular components are unused: the task
    // space is planar (x, y, theta_z).
    NodeRef r;
    r.p[0] = desiredPos[0];  r.p[1] = desiredPos[1];
    r.v[0] = desiredVel[0];  r.v[1] = desiredVel[1];
    r.a[0] = desiredAcc[0];  r.a[1] = desiredAcc[1];
    for (int i = 0; i < 4; ++i) r.quat[i] = desiredQuat[i];
    r.w  = desiredAngVel[2];
    r.aw = desiredAngAcc[2];
    return r;
}

void TaskSpaceController::sampleNodeReferences() {
    const std::size_t n_nodes = node_ref_.size();
    if (!have_window_) {
        // The static setpoint at every node: exactly what each node is scored
        // against without a trajectory.
        const NodeRef r = staticReference();
        for (std::size_t k = 0; k < n_nodes; ++k) node_ref_[k] = r;
        return;
    }
    ++ref_ticks_;
    RefPoint rp;
    for (std::size_t k = 0; k < n_nodes; ++k) {
        // With preview each node sees the reference at its own time; without,
        // every node sees the reference at t_now.
        const double t = ctrl_params_.reference_preview
                             ? t_ref_ + static_cast<double>(k) * ref_spacing_
                             : t_ref_;
        if (!window_.sample(t, rp)) ++ref_outside_;
        NodeRef& r = node_ref_[k];
        r.p[0] = rp.x;   r.p[1] = rp.y;
        r.v[0] = rp.vx;  r.v[1] = rp.vy;
        r.a[0] = rp.ax;  r.a[1] = rp.ay;
        window_.orientation(rp.yaw, r.quat);
        r.w  = rp.wz;
        r.aw = rp.az;
    }
}

// === Torque mapping and feedforward ==========================================

void TaskSpaceController::setDisturbanceFeedforward(const Eigen::VectorXd& d_joints,
                                                    double dt)
{
    if (d_joints.size() != n_joints_) {
        throw std::runtime_error(
            "TaskSpaceController::setDisturbanceFeedforward: d_joints must have "
            "numArmJoints() entries");
    }
    // Gain 0 is the default and must be exactly a no-op, so that an observer
    // can run and be scored on a configuration that is otherwise unchanged.
    if (ctrl_params_.ff_gain == 0.0) {
        d_ff_.setZero();
        return;
    }

    const bool slew_on = ctrl_params_.ff_slew > 0.0 && dt > 0.0;
    const double max_step = slew_on ? ctrl_params_.ff_slew * dt : 0.0;
    bool saturated = false, slewed = false;

    for (int i = 0; i < n_joints_; ++i) {
        // A non-finite estimate must not reach the actuators, and it must not
        // poison the retained value either: hold the previous entry instead.
        if (!std::isfinite(d_joints[i])) continue;

        double target = ctrl_params_.ff_gain * d_joints[i];

        // Saturate before the slew limit, so the slew is spent approaching a
        // reachable value rather than chasing one the cap will reject anyway.
        const double cap = ff_max_vec_[i];
        if (target > cap)       { target = cap;  saturated = true; }
        else if (target < -cap) { target = -cap; saturated = true; }

        if (slew_on) {
            const double delta = target - d_ff_[i];
            if (delta > max_step)       { target = d_ff_[i] + max_step; slewed = true; }
            else if (delta < -max_step) { target = d_ff_[i] - max_step; slewed = true; }
        }
        d_ff_[i] = target;
    }
    if (saturated) ++ff_saturated_;
    if (slewed)    ++ff_slew_limited_;
}

Eigen::VectorXd TaskSpaceController::finalizeJointTorques(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& tau_task,
    const Eigen::VectorXd& Cv_joints)
{
    Eigen::VectorXd tau_joints;
    finalizeJointTorques(J, tau_task, Cv_joints, tau_joints);
    return tau_joints;
}

void TaskSpaceController::finalizeJointTorques(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& tau_task,
    const Eigen::VectorXd& Cv_joints, Eigen::VectorXd& tau_joints)
{
    // Bias first, then the product accumulated onto it: assigning the whole
    // expression at once evaluates J^T*tau_task into a temporary.
    tau_joints = Cv_joints;
    tau_joints.noalias() += J.transpose() * tau_task;

    // Cancel the estimated disturbance. The reduced dynamics are
    // H_g qddot + Cv_joints = tau + d, so commanding tau = J^T u + Cv - d
    // leaves the plant following what the law planned. d_ff_ is zero unless a
    // feedforward is configured, and a law with torque constraints shifts them
    // by the same amount, so the clamp below should not be what removes it.
    tau_joints -= d_ff_;
    for (int i = 0; i < n_joints_; ++i) {
        tau_joints[i] = std::max(-torqueLimsByIndex[i],
                                 std::min(torqueLimsByIndex[i], tau_joints[i]));
    }
}

// === Safe mode ===============================================================

Eigen::VectorXd TaskSpaceController::dampingTorque(const Eigen::VectorXd& v) const
{
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_joints_);
    if (v.size() < 6 + n_joints_) return tau;   // nothing trustworthy to damp against

    for (int idx : ownedJointInds) {
        const double vd = v[6 + idx];
        if (!std::isfinite(vd)) continue;   // a poisoned joint gets zero, never NaN

        const double lim = torqueLimsByIndex[idx];
        tau[idx] = std::max(-lim, std::min(lim, -kd_[idx] * vd));
    }
    return tau;
}

void TaskSpaceController::enterDamping()
{
    mode_ = ControlMode::Damping;
    consecutive_ok_ = 0;
    onEnterDamping();

    // Drop the feedforward on entry. Safe mode exists because something has
    // already gone wrong, and a bad state is exactly what produces a bad
    // disturbance estimate -- compensating it could compound the failure the
    // tier system is handling. dampingTorque never consults d_ff_, so what this
    // really protects is RECOVERY: without the reset, the first tracking tick
    // after safe mode would apply the stale pre-failure value as a step. Cleared
    // here so recovery ramps back up from nothing through the slew limit.
    d_ff_.setZero();
}

Eigen::VectorXd TaskSpaceController::failToDamping(const Eigen::VectorXd& v)
{
    // Tier 3 first: a latched controller stays down whatever this tick looked like.
    if (mode_ == ControlMode::Latched) {
        ++safe_mode_ticks_;
        return dampingTorque(v);
    }

    // Tier 2, and Tier 3 once it has gone on long enough to count as not recovering.
    enterDamping();
    ++safe_mode_ticks_;
    if (ctrl_params_.tier3_latch_after_ticks > 0 &&
        ++damping_run_ >= ctrl_params_.tier3_latch_after_ticks) {
        mode_ = ControlMode::Latched;
    }
    return dampingTorque(v);
}

bool TaskSpaceController::holdSafeMode()
{
    // Tier 2 is sticky: hold the damping command until the law has been healthy
    // for safe_mode_recovery_ticks in a row, so a marginal one cannot chatter the
    // output between damping and full tracking. Hysteresis only -- the handover on
    // recovery is a step, not a ramp. Tier 3 does not recover here at all.
    if (mode_ == ControlMode::Damping || mode_ == ControlMode::Latched) {
        ++safe_mode_ticks_;
        if (mode_ == ControlMode::Latched ||
            ++consecutive_ok_ < ctrl_params_.safe_mode_recovery_ticks) {
            return true;
        }
        mode_ = ControlMode::Nominal;   // recovered; the caller applies its command
        damping_run_ = 0;
    }
    return false;
}

void TaskSpaceController::clearLatch()
{
    if (mode_ != ControlMode::Latched) return;
    mode_ = ControlMode::Damping;   // back to auto-recovering, not straight to tracking
    damping_run_ = 0;
    consecutive_ok_ = 0;
}

// === Diagnostics =============================================================

std::string TaskSpaceController::formatProfile(const ProfileSnapshot& s)
{
    // Phases in execution order, so the report reads as a breakdown rather than a
    // bag of numbers. Every phase is listed; one a law does not have shows count
    // 0 and is skipped.
    const std::pair<const char*, const PhaseStat*> phases[] = {
        {"dynamics",   &s.dynamics},   {"jdotqdot",     &s.jdotqdot},
        {"roll_dyn",   &s.rollout_dynamics}, {"roll_intg", &s.rollout_integrate},
        {"pinv",       &s.pinv},       {"lambda",       &s.lambda},
        {"law",        &s.law},
        {"buildmodel", &s.build_model},{"defect",       &s.defect},
        {"qpsolve",    &s.qpsolve},    {"trust",        &s.trust_region},
        {"finalize",   &s.finalize},
    };
    const PhaseStat& tick = s.tick;

    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(1);
    os << "profile over " << tick.count << " ticks (us)";
    if (tick.count > 0) {
        os << "  TICK mean=" << tick.mean() << " max=" << tick.max_us;
        if (tick.budget_us > 0.0)
            os << " over_period=" << tick.over << "/" << tick.count;
        os << "\n  hist(tick) ";
        for (int i = 0; i < PhaseStat::kBins; ++i) {
            if (i + 1 == PhaseStat::kBins) os << ">5000:";
            else                           os << "<=" << PhaseStat::kEdgeUs[i] << ":";
            os << tick.bins[i] << " ";
        }
    }
    os << "\n  phases ";
    double attributed = 0.0;
    for (const auto& [name, p] : phases) {
        if (p->count == 0) continue;
        attributed += p->sum_us;
        os << name << "(n=" << p->count << " mean=" << p->mean()
           << " max=" << p->max_us << ") ";
    }
    // The gap is the honest measure of how much of the tick is still unexplained.
    if (tick.count > 0) {
        os << "\n  unattributed=" << (tick.sum_us - attributed) / tick.count
           << " us/tick";
    }
    if (s.qp_iters.count > 0) {
        // Summed over the tick's solves, so a retry can push this past the cap --
        // which is the signature worth seeing: the warm solve burned the whole cap
        // and the cold retry converged in the remainder.
        os << "\n  qp_iter/tick mean=" << s.qp_iters.mean()
           << " max=" << s.qp_iters.max
           << " (cap " << s.qp_max_iter << "/solve)"
           << "  hpipm_us mean=" << s.qp_time_ms.mean() * 1e3
           << " max=" << s.qp_time_ms.max * 1e3
           << "  retries=" << s.qp_retries;
    }
    // Only printed when non-zero: it should never happen, so a permanent "=0"
    // column would just be noise on every report.
    if (s.spd_fallbacks > 0)
        os << "  H_g NOT PD on " << s.spd_fallbacks << " solves";
    if (s.trace_cap > 0) {
        os << "  trace " << s.trace_size << "/" << s.trace_cap
           << (s.trace_full ? " FULL" : "");
    }
    // MuJoCo's own logging is redirected to counters, so this is the only place it
    // surfaces. Counts are since process start, not per window: an mj warning is a
    // standing condition worth seeing every report until it is dealt with.
    if (mjWarningCount() > 0) {
        os << "\n  MuJoCo warnings: " << mjWarningCount()
           << " (last: " << lastMjWarning() << ")";
    }
    if (mjErrorCount() > 0) {
        os << "\n  MuJoCo errors: " << mjErrorCount()
           << " (last: " << lastMjError() << ")";
    }
    // Soft-constraint slack belongs to a law with a QP; qp_max_iter is how a
    // snapshot says it has one.
    if (s.qp_max_iter > 0) {
        os << "  | peak slack: box=" << s.max_box_slack
           << " torque=" << s.max_tau_slack;
    }
    return os.str();
}

}  // namespace pat_arm_control
