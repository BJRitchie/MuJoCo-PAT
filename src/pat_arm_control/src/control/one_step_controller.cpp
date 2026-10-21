#include "pat_arm_control/control/one_step_controller.h"

namespace pat_arm_control {

OneStepController::OneStepController(const std::string& mjcf_path,
                                     const ControllerParams& params)
    : TaskSpaceController(mjcf_path, params) {
    // One reference node: the target at t_now, with its velocity and
    // acceleration fed forward. Already the base's default; stated because the
    // law depends on it.
    setReferenceNodes(1, 0.0);

    const int n = n_joints_;
    J6_.resize(6, n);
    J3_.resize(3, n);
    Jt_.resize(n, 3);
    HinvJt_.resize(n, 3);
    H_g_.resize(n, n);
    I_ = Eigen::MatrixXd::Identity(n, n);
    Cv_.resize(n);
    tau_task_.resize(3);
    out_.resize(n);
    llt_.compute(I_);   // sizes the factor's storage
}

void OneStepController::hgInverse(Eigen::MatrixXd& out) const
{
    if (hg_pd_) out = llt_.solve(I_);
    else        out = pinvDLS(H_g_, ctrl_params_.Hg_damping);
}

Eigen::VectorXd OneStepController::controlLawImpl(
    const Eigen::VectorXd& /*q*/, const Eigen::VectorXd& v)
{
    const auto start = std::chrono::steady_clock::now();

    sampleNodeReferences();
    const NodeRef& r = node_ref_[0];

    // --- The reduced model at the live state --------------------------------
    {
        ScopedTimer t(dynamics);
        computeBaseElim(nullptr, be_);
        computeSiteJacobian(ee_site_id_, nullptr, be_);
        gjm6(be_, J6_);
        planarRows(J6_, J3_);
        getDynamics(H_g_, Cv_, nullptr, be_);
    }
    Eigen::Matrix<double, 6, 1> jdot6;
    {
        ScopedTimer t(jdotqdot);
        jdot6 = jdotQdot6(ee_site_id_, nullptr, be_);
    }

    // --- Task-space error, e = p_d - p ---------------------------------------
    TaskState s;
    const Eigen::Vector3d p_ee = eePosition(ee_site_id_);
    // noalias: J3_ is dynamic, so without it the product is evaluated into a
    // heap temporary before it is copied into the fixed-size vector.
    s.v_ee.noalias() = J3_ * v.tail(n_joints_);
    s.e    << r.p[0] - p_ee[0], r.p[1] - p_ee[1],
              orientationErrorZ(r.quat, eeOrientation(ee_site_id_));
    s.edot << r.v[0] - s.v_ee[0], r.v[1] - s.v_ee[1], r.w - s.v_ee[2];
    s.hp   << jdot6[0], jdot6[1], jdot6[5];
    s.a_d  << r.a[0], r.a[1], r.aw;

    // --- L^-1 = J H_g^-1 J^T, from one Cholesky of H_g and a solve against J^T
    {
        ScopedTimer t(pinv);
        Jt_ = J3_.transpose();
        llt_.compute(H_g_);
        hg_pd_ = llt_.info() == Eigen::Success;
        if (hg_pd_) {
            HinvJt_ = llt_.solve(Jt_);
        } else {
            // H_g is not positive definite (a degenerate model): the damped
            // pseudo-inverse is bounded where the Cholesky has nothing to give.
            HinvJt_ = pinvDLS(H_g_, ctrl_params_.Hg_damping) * Jt_;
        }
    }
    {
        ScopedTimer t(lambda);
        s.Lam_inv.noalias() = J3_ * HinvJt_;
        checkNaN(s.Lam_inv.hasNaN(), "Lambda_inv");
    }

    Eigen::Vector3d tau_task;
    {
        ScopedTimer t(law);
        tau_task = taskForce(s);
        checkNaN(tau_task.hasNaN(), "tau_task");
    }

    Eigen::VectorXd result;
    if (holdSafeMode()) {
        result = dampingTorque(v);
    } else {
        ScopedTimer t(finalize);
        tau_task_ = tau_task;
        finalizeJointTorques(J3_, tau_task_, Cv_, out_);
        result = out_;
    }

    const auto stop = std::chrono::steady_clock::now();
    recordTick(start, std::chrono::duration<double, std::micro>(stop - start).count());
    return result;
}

void OneStepController::recordTick(std::chrono::steady_clock::time_point start,
                                   double tick_us)
{
    tick.add(tick_us);
    if (trace_cap_ == 0 || trace_.size() >= trace_cap_) return;
    if (!trace_t0_set_) {
        trace_t0_     = start;
        trace_t0_set_ = true;
    }
    TickTrace row{};
    row.t_us       = std::chrono::duration<double, std::micro>(start - trace_t0_).count();
    row.tick_us    = static_cast<float>(tick_us);
    row.qp_iters   = -1;     // no solver
    row.qp_iters1  = -1;
    row.mode       = static_cast<int8_t>(mode_);
    trace_.push_back(row);
}

void OneStepController::enableTrace(std::size_t capacity)
{
    trace_cap_ = capacity;
    trace_.clear();
    trace_.reserve(capacity);   // the one allocation; recording stops at capacity
    trace_t0_set_ = false;
}

void OneStepController::drainTrace(std::vector<TickTrace>& sink)
{
    sink.clear();
    sink.swap(trace_);
    trace_.reserve(trace_cap_);   // off the control path, so push_back never allocates
}

void OneStepController::snapshotProfile(ProfileSnapshot& out) const
{
    out.tick       = tick;
    out.dynamics   = dynamics;
    out.jdotqdot   = jdotqdot;
    out.pinv       = pinv;
    out.lambda     = lambda;
    out.law        = law;
    out.finalize   = finalize;
    out.trace_size = static_cast<long>(trace_.size());
    out.trace_cap  = static_cast<long>(trace_cap_);
    out.trace_full = trace_cap_ > 0 && trace_.size() >= trace_cap_;
}

void OneStepController::resetProfile()
{
    tick.reset();
    dynamics.reset();
    jdotqdot.reset();
    pinv.reset();
    lambda.reset();
    law.reset();
    finalize.reset();
}

}  // namespace pat_arm_control
