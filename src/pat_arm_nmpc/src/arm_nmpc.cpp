#include "pat_arm_nmpc/arm_nmpc.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pat_arm_nmpc {

ArmNMPC::ArmNMPC(
    const std::string& mjcf_path, const NMPCParams& params) 
    : ArmTaskSpaceController(mjcf_path)
    , params_(params) 
{
    // --- Check Controller Parameters --------------------------------- // 
    if (params_.ee_site_name.empty()) {
        throw std::runtime_error(
            "ArmNMPC: NMPCParams.ee_site_name must be set");
    }
    if (params_.N < 1) {
        throw std::runtime_error("ArmNMPC: NMPCParams.N must be >= 1");
    }
    if (params_.Ts <= 0.0) {
        throw std::runtime_error("ArmNMPC: NMPCParams.Ts must be > 0");
    }
    if (params_.sqp_iters < 1) {
        // 0 would skip the rollout loop entirely, leaving J_task0 at 0x0 and
        // producing a dimension mismatch in finalizeJointTorques.
        throw std::runtime_error("ArmNMPC: NMPCParams.sqp_iters must be >= 1");
    }
    if (params_.safe_mode_damping_frac <= 0.0) {
        throw std::runtime_error(
            "ArmNMPC: NMPCParams.safe_mode_damping_frac must be > 0 — safe mode "
            "with zero damping would leave the arm coasting.");
    }
    if (params_.safe_mode_recovery_ticks < 1) {
        throw std::runtime_error(
            "ArmNMPC: NMPCParams.safe_mode_recovery_ticks must be >= 1");
    }
    ee_site_id_ = siteIdFromName(params_.ee_site_name);

    if (params_.joint_lims.empty()) {
        throw std::runtime_error(
            "ArmNMPC: NMPCParams.joint_lims must be set");
    } else {
        // Parse joint lims etc 
        for (const auto& lims : params_.joint_lims) {
            qlims[lims.name] = std::make_pair(lims.q_min, lims.q_max);
            vlims[lims.name] = lims.qd_max;
        }
        for (const auto& [jname, tau] : params_.model_torque_lims) {
            torque_lims[jname] = tau;
        }

        // Re-index by arm-local joint index (0..n_joints_-1) for cheap per-tick
        // lookups instead of re-hashing joint names every control tick.
        //
        // Joints NOT listed in joint_lims (the other arm's, when one controller
        // owns a subset of a multi-arm model) are never box-constrained:
        // buildJointBoxBounds only touches ownedJointInds. They DO appear in the
        // all-model-joint torque constraint and in the rollout clamp, so every
        // joint needs a real torque limit -- model_torque_lims supplies one per
        // model joint and is required to be complete, so there is no fallback
        // magnitude to guess at (and none to silently disagree with the config).
        //
        // Position ranges come from the MJCF, which is what MuJoCo actually
        // enforces during the rollout, so every joint gets a real range rather
        // than a placeholder -- including the ones this controller does not own.
        modelQlimsByIndex = modelJointRanges();
        qlimsByIndex      = modelQlimsByIndex;   // tightened by config just below
        vlimsByIndex.assign(n_joints_, 10.0);
        torqueLimsByIndex.assign(n_joints_, -1.0);   // sentinel: must all be filled below
        for (const auto& [jname, tau] : torque_lims) {
            if (tau <= 0.0) {
                throw std::runtime_error(
                    "ArmNMPC: model_torque_lims entry for joint \"" + jname +
                    "\" must be > 0, got " + std::to_string(tau));
            }
            torqueLimsByIndex[jointIndexFromName(jname)] = tau;
        }
        for (int i = 0; i < n_joints_; ++i) {
            if (torqueLimsByIndex[i] < 0.0) {
                throw std::runtime_error(
                    "ArmNMPC: model_torque_lims is missing arm-local joint index " +
                    std::to_string(i) + " — it must name every joint in the model, "
                    "not only the ones this controller owns (the torque constraint "
                    "and the rollout clamp cover all of them).");
            }
        }
        // Eigen mirror of the same limits, built here because the torque
        // constraint needs them as a vector at every rollout node and they do
        // not change after construction.
        tau_lim_vec_.resize(n_joints_);
        for (int i = 0; i < n_joints_; ++i) tau_lim_vec_(i) = torqueLimsByIndex[i];
        for (const auto& [jname, bounds] : qlims) {
            int idx = jointIndexFromName(jname);
            // Intersect rather than replace: a configured range may tighten the
            // model's (a software limit inside the mechanical one) but must not
            // widen it past what MuJoCo will enforce.
            qlimsByIndex[idx].first  = std::max(qlimsByIndex[idx].first,  bounds.first);
            qlimsByIndex[idx].second = std::min(qlimsByIndex[idx].second, bounds.second);
            vlimsByIndex[idx]        = vlims.at(jname);
        }

        // Resolve which arm-local joint indices this controller owns (i.e.
        // whose Δq/qdot rows become box-constrained state). Empty owned_joints
        // means "all joints".
        ownedJointInds.clear();
        if (params_.owned_joints.empty()) {
            ownedJointInds.resize(n_joints_);
            std::iota(ownedJointInds.begin(), ownedJointInds.end(), 0);
        } else {
            for (const auto& jname : params_.owned_joints)
                ownedJointInds.push_back(jointIndexFromName(jname));
        }

        n_owned = static_cast<int>(ownedJointInds.size());
        if (n_owned == 0) {
            throw std::runtime_error(
                "ArmNMPC: owned_joints resolved to zero joints — "
                "this controller would have no box-constrained state at all.");
        }

        // Safe-mode damping gains, owned joints only (the node publishes no
        // effort for the rest). Scaling by tau_max/qd_max keeps the damping
        // torque inside damping_frac*tau_max for any speed within qd_max, so
        // safe mode never relies on the output clamp to stay physical.
        kd_.assign(n_joints_, 0.0);
        for (int idx : ownedJointInds) {
            if (vlimsByIndex[idx] <= 0.0) {
                throw std::runtime_error(
                    "ArmNMPC: qd_max must be > 0 for owned joint at arm-local index " +
                    std::to_string(idx) + " — safe-mode damping is scaled by it.");
            }
            kd_[idx] = params_.safe_mode_damping_frac *
                        torqueLimsByIndex[idx] / vlimsByIndex[idx];
        }
    }

    // --- Build the QuadProbSolver --------------------------------- //
    // This port is permanently the planar (x, y, theta_z) control law:
    //   task dim d = 3  (nu),  task-space block 2d+1 = 7  (nx_task),
    //   augmented state nx = nx_task + 2*n_owned (the Delta-q / qdot rows of the
    //   joints this controller actually commands).
    // Matches VORTEX's non-pose branch (use_pose_dims == false).
    const int d        = 3;
    const int nx_task  = 2 * d + 1;   // 7
    const int n_j      = n_joints_;

    quad_prob_solver::QuadProbSolverParams qp_params;
    qp_params.nx_task      = nx_task;
    qp_params.nu           = d;

    // Only the OWNED joints get Δq/qdot rows. The others would be states that
    // appear in neither the cost (Q is zero on every joint row) nor the
    // constraints (the box covers ownedJointInds only) and that no other row's
    // dynamics read -- free variables, carried at full Riccati cost. Their motion
    // is still predicted, in the nonlinear rollout, which uses the whole model.
    qp_params.nx           = nx_task + 2 * n_owned;
    qp_params.N            = params_.N;
    qp_params.qp_max_iter  = params_.qp_max_iter;
    qp_params.qp_tol       = params_.qp_tol;
    qp_params.qp_reg_prim  = params_.qp_reg_prim;
    qp_params.qp_warm_start = params_.qp_warm_start;
    qp_params.slack_penalty_linear    = params_.joint_limit_slack_linear;
    qp_params.slack_penalty_quadratic = params_.joint_limit_slack_quadratic;

    qp_params.ng = n_j;
    qp_params.general_slack_penalty_linear    = params_.torque_slack_linear;
    qp_params.general_slack_penalty_quadratic = params_.torque_slack_quadratic;

    qp_params.soft_joint_limits       = params_.soft_joint_limits;
    qp_params.soft_torque_limits      = params_.soft_torque_limits;
    qp_params.relax_box_to_contain_x0 = params_.relax_box_to_contain_x0;
    // Floor the guard's margin at one step's unavoidable drift, Ts*qd_max: a
    // joint at rated speed moves that far in a single step whatever the control
    // does, so a margin below it re-creates the infeasibility the guard exists to
    // remove. The solver cannot work this out for itself — it knows neither Ts
    // nor which rows are positions.
    {
        double qd_max_owned = 0.0;
        for (int idx : ownedJointInds)
            qd_max_owned = std::max(qd_max_owned, vlimsByIndex[idx]);
        qp_params.relax_box_margin =
            std::max(params_.relax_box_margin, params_.Ts * qd_max_owned);
    }

    // Box rows, in QP-state coordinates. The joint rows are now indexed by OWNED
    // position i, not by model joint index: the state carries n_owned of them, in
    // ownedJointInds order, so this is the identity map and not a lookup.
    qp_params.idxbx_k.resize(2 * n_owned);
    for (int i = 0; i < n_owned; ++i) {
        qp_params.idxbx_k[i]           = nx_task + i;              // Δq row
        qp_params.idxbx_k[n_owned + i] = nx_task + n_owned + i;    // qdot row
    }

    // Hard terminal constraint edot_N = 0 (task-space velocity error), for
    // recursive feasibility -- see NMPCParams::terminal_velocity_constraint's
    // doc comment. edot occupies rows [d, 2d) of the augmented state's
    // task-space block (buildAugmentedModel's layout: [e(d); edot(d); bias
    // channel(1); Δq(n_j); qdot(n_j)]).
    if (params_.terminal_velocity_constraint) {
        qp_params.idxbx_terminal.resize(d);
        for (int i = 0; i < d; ++i) {
            qp_params.idxbx_terminal[i] = d + i;
        }
    }

    qp_ = std::make_unique<quad_prob_solver::QuadProbSolver>(qp_params);
}

ArmNMPC::~ArmNMPC() = default;

// === Public entrypoints (called by the ROS node each control tick) ============ //

Eigen::VectorXd ArmNMPC::computeControl(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    loadLiveState(q, v);        // writes mj_->data->qpos/qvel + mj_forward
    return controlLaw(q, v);    // reads the dynamics helpers off mj_->data
}

ArmNMPC::EePose ArmNMPC::currentEePose(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    loadLiveState(q, v);
    return { eePosition(ee_site_id_), eeOrientation(ee_site_id_) };
}

// === CONTROL LAW ================================================================ //
Eigen::VectorXd ArmNMPC::controlLaw(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    try {
        return controlLawImpl(q, v);
    } catch (const std::exception&) {
        // A non-finite intermediate (checkNaN), a solver size mismatch, an Eigen
        // failure, or a MuJoCo error via the installed hook. No Tier 1 here on
        // purpose: the kinematics this tick produced are exactly what is suspect,
        // so replaying a wrench through them is not safe. Damping depends on none
        // of that machinery and is finite by construction even if v is not.
        return onSolveFailure(v, nullptr, nullptr);
    }
}

Eigen::VectorXd ArmNMPC::controlLawImpl(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    // Record start time of control loop
    const auto start = std::chrono::steady_clock::now();

    const int n_j = n_joints_;
    const int nx  = 7 + 2 * n_owned;   // 2*d+1 + 2*n_owned, d=3

    // --- Stage/terminal cost (task-space channels only) -------------------------
    // Members: the weights are fixed at construction, so rebuilding these every
    // tick allocated three matrices for a result that never changes.
    Eigen::MatrixXd& Q   = Q_;
    Eigen::MatrixXd& R   = R_;
    Eigen::MatrixXd& Q_N = Q_N_;
    Q.setZero(nx, nx);
    Q(0, 0) = params_.Qx;    Q(1, 1) = params_.Qy;    Q(2, 2) = params_.Qwz;
    Q(3, 3) = params_.Qdotx; Q(4, 4) = params_.Qdoty; Q(5, 5) = params_.Qdotwz;
    R = Eigen::Vector3d(params_.Rx, params_.Ry, params_.Rwz).asDiagonal();
    Q_N = Q * params_.terminal_cost_multiplier;   // == Q when multiplier is the default 1.0
    Eigen::VectorXd out;

    if (params_.fullNonlinear) {
        TaskSpaceEval eval = [this](
            const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
            const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
            const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
            NodeTaskSpace& ts_out) {
            evalTaskSpace(q_node, v_node, ee_pos, ee_quat, J_g6, jdot_qdot6, ts_out);
        };
        out = solveNonlinearMPC(q, v, /*d=*/3, eval, Q, R, Q_N);
    } else {
        // --- Current EE state -------------------------------------------------------
        Eigen::Vector3d  p_ee    = eePosition(ee_site_id_);
        Eigen::Vector4d  quat_ee = eeOrientation(ee_site_id_);   // [w,x,y,z]
        Eigen::MatrixXd  J_g6;
        {
            ScopedTimer t(lambda);
            J_g6 = gjm6(ee_site_id_);   // 6 x n_joints_: rows [0:3) lin, [3:6) ang
        }
        Eigen::MatrixXd J_g(3, n_j);
        J_g.row(0) = J_g6.row(0);   // x
        J_g.row(1) = J_g6.row(1);   // y
        J_g.row(2) = J_g6.row(5);   // angular-Z
        Eigen::Vector3d v_ee = J_g * v.tail(n_joints_);

        Eigen::Vector2d p_d(desiredPos[0], desiredPos[1]);   // desiredPos[2] (Z) unused
        Eigen::Vector2d v_d(desiredVel[0], desiredVel[1]);
        Eigen::Vector2d a_d(desiredAcc[0], desiredAcc[1]);
        mjtNum quat_d[4] = {desiredQuat[0], desiredQuat[1], desiredQuat[2], desiredQuat[3]};
        double w_d  = desiredAngVel[2];
        double aw_d = desiredAngAcc[2];

        // World-frame angle-axis (log-map) orientation error, Z-component
        // only -- see evalTaskSpacePositionOnly2D's identical derivation.
        mjtNum quat_ee_arr[4] = {quat_ee[0], quat_ee[1], quat_ee[2], quat_ee[3]};
        mjtNum e_orient_local[3];
        mju_subQuat(e_orient_local, quat_d, quat_ee_arr);
        mjtNum e_orient_world[3];
        mju_rotVecQuat(e_orient_world, e_orient_local, quat_ee_arr);
        double e_theta = e_orient_world[2];

        Eigen::Vector3d e, edot;
        e    << p_d - p_ee.head<2>(), e_theta;
        edot << v_d - v_ee.head<2>(), w_d - v_ee(2);

        Eigen::VectorXd Cv_joints;
        Eigen::MatrixXd H_g;
        computeGeneralizedInertia(H_g, Cv_joints);
        Eigen::MatrixXd H_g_inv_Jt, Lambda_inv;
        computeInertiaProducts(H_g, J_g, H_g_inv_Jt, Lambda_inv);

        Eigen::Matrix<double, 6, 1> jdot_qdot6;
        {
            ScopedTimer t(jdotqdot);
            jdot_qdot6 = jdotQdot6(ee_site_id_);
        }
        Eigen::Vector3d bias;
        bias << a_d - jdot_qdot6.head<2>(), aw_d - jdot_qdot6(5);

        Eigen::MatrixXd A, B;
        buildAugmentedModel(Lambda_inv, bias, H_g_inv_Jt, A, B);

        Eigen::VectorXd q0 = q.tail(n_j);   // current measured joint angles

        Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(nx);
        x_aug.head<3>()     = e;
        x_aug.segment<3>(3) = edot;
        x_aug(6) = 1.0;
        // Δq/qdot rows of x_aug stay 0 — see controlLawPositionOnly's identical
        // comment on why.

        Eigen::VectorXd lbx_j, ubx_j;
        buildJointBoxBounds(q0, lbx_j, ubx_j);

        Eigen::MatrixXd D;
        Eigen::VectorXd lg, ug;
        buildTorqueGeneralConstraint(J_g, Cv_joints, D, lg, ug);

        Eigen::VectorXd tau_task;
        if (!solveStageQP(A, B, Q, R, lbx_j, ubx_j, D, lg, ug,
                          x_aug, Q_N, tau_task)) {
            // This path keeps no multi-stage plan (it solves one stage), so Tier 1
            // has nothing to replay -- pass no kinematics and let it fall to Tier 2.
            const Eigen::VectorXd out_fail = onSolveFailure(v, nullptr, nullptr);
            recordTick(start);
            return out_fail;
        }
        if (mode_ == ControlMode::Damping || mode_ == ControlMode::Latched) {
            ++safe_mode_ticks_;
            if (mode_ == ControlMode::Latched ||
                ++consecutive_ok_ < params_.safe_mode_recovery_ticks) {
                recordTick(start);
                return dampingTorque(v);
            }
            mode_ = ControlMode::Nominal;   // recovered; fall through to tracking
            damping_run_ = 0;
        }

        finalizeJointTorques(J_g, tau_task, Cv_joints, out);
    }

    recordTick(start);
    return out;
}

// === Shared helpers ============================================================
// Factored out of the three controlLaw* functions below -- they differ only
// in task-space dimension d (3 for position-only, 2 for planar, 6 for pose),
// and Eigen's dynamic-sized matrices make one d-agnostic implementation of
// each step trivial to share. See armConstrainedNMPController.h for the doc
// comment on each.

void ArmNMPC::computeGeneralizedInertia(
    Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints)
{
    H_g.resize(n_joints_, n_joints_);
    Cv_joints.resize(n_joints_);
    ScopedTimer t(dynamics);
    getDynamics(H_g, Cv_joints);
}

void ArmNMPC::computeInertiaProducts(
    const Eigen::MatrixXd& H_g, const Eigen::MatrixXd& J,
    Eigen::MatrixXd& H_g_inv_Jt, Eigen::MatrixXd& Lambda_inv)
{
    {
        ScopedTimer t(pinv);
        // The damped-least-squares inverse of an SPD H_g, exactly, without an SVD.
        // For H_g = Q S Q^T the DLS inverse is Q diag(s/(s^2+d^2)) Q^T, which is
        // identically (H_g^2 + d^2 I)^-1 H_g, so the product actually wanted is
        //   X = (H_g^2 + d^2 I)^-1 (H_g J^T)
        // -- one Cholesky and a d-column solve in place of a 6x6 SVD.
        //
        // The damping is kept rather than dropped even though H_g is positive
        // definite and a plain solve would be well posed: it also regularises the
        // model the QP sees, and removing it costs 83% on the worst tick (worst
        // solve 14 -> 24 iterations on the warm-start scenario) for no gain in
        // tracking.
        // G, HgJt and the Cholesky are members: this runs once per rollout node,
        // so a local for any of them is N allocations a tick.
        const double lambda_sq = params_.Hg_damping * params_.Hg_damping;
        G_.noalias() = H_g * H_g;
        G_.diagonal().array() += lambda_sq;
        HgJt_.noalias() = H_g * J.transpose();
        if (!solveSpd(G_, HgJt_, H_g_inv_Jt, hg_llt_)) {
            ++spd_fallbacks_;
            H_g_inv_Jt = pinvDLS(H_g, params_.Hg_damping) * J.transpose();
        }
    }
    {
        ScopedTimer t(lambda);
        Lambda_inv.noalias() = J * H_g_inv_Jt;
    }
    checkNaN(Lambda_inv.hasNaN(), "Lambda_inv");
}

Eigen::VectorXd ArmNMPC::reachableWrenchBound(const Eigen::MatrixXd& J_task) const
{
    Eigen::VectorXd bound;
    reachableWrenchBound(J_task, bound);
    return bound;
}

void ArmNMPC::reachableWrenchBound(
    const Eigen::MatrixXd& J_task, Eigen::VectorXd& bound) const
{
    // Finite cap for channels the Jacobian barely couples to: tau/|J| runs away
    // as |J| -> 0, and HPIPM's factorization does not tolerate bounds spanning
    // many orders of magnitude.
    constexpr double kMaxBound = 1.0e3;

    const int d = static_cast<int>(J_task.rows());
    bound.resize(d);
    for (int j = 0; j < d; ++j) {
        double lim = kMaxBound;

        for (int i = 0; i < n_joints_; ++i) {
            const double a = std::abs(J_task(j, i));
            
            if (a > 1e-9) 
                lim = std::min(lim, torqueLimsByIndex[i] / a);
        }
        bound(j) = std::min(lim, kMaxBound);
    }
}

bool ArmNMPC::solveStageQP(
    Eigen::MatrixXd& A, Eigen::MatrixXd& B,
    Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
    Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
    Eigen::MatrixXd& D, Eigen::VectorXd& lg, Eigen::VectorXd& ug,
    Eigen::VectorXd& x_aug, Eigen::MatrixXd& Q_N,
    Eigen::VectorXd& tau_task_out)
{
    const int nx = static_cast<int>(A.rows());
    const int nu = static_cast<int>(B.cols());

    tau_task_out.resize(nu);
    bool ok = false;
    {
        ScopedTimer t(qpsolve);

        // Native HPIPM affine offset b stays zero every stage -- the frozen
        // acceleration bias is carried through A's constant channel instead
        // (x_aug's trailing "1" row).
        Eigen::VectorXd zero_b = Eigen::VectorXd::Zero(nx);

        ok = qp_->solve(
            A.data(), B.data(), zero_b.data(),
            Q.data(), R.data(),
            lbx_j.data(), ubx_j.data(),
            D.data(), lg.data(), ug.data(),
            x_aug.data(),
            Q_N.data());

        if (ok) {
            qp_->getU(0, tau_task_out.data());
            recordSlack();
        }
    }
    recordQpIters();
    // A non-finite solution is a failed solve, not an exception: it means the
    // same thing to the caller as a nonzero solver status.
    return ok && tau_task_out.allFinite();
}

// === Augmented prediction model ===============================================
// Successive-linearization LTV model: task-space error state (pos/vel + a
// constant channel carrying the frozen acceleration bias) augmented with
// Δq/qdot rows so joint position/velocity limits are plain box bounds.
void ArmNMPC::buildAugmentedModel(
    const Eigen::MatrixXd& Lambda_inv,
    const Eigen::VectorXd& bias, const Eigen::MatrixXd& H_g_inv_Jt,
    Eigen::MatrixXd& A, Eigen::MatrixXd& B)
{
    const int d   = static_cast<int>(Lambda_inv.rows());   // task dim == nu
    const int n_q = n_owned;            // joint rows carried in the state
    const double Ts = params_.Ts;
    const int nt  = 2 * d + 1;          // task-space block: pos + vel + bias channel
    const int nx  = nt + 2 * n_q;

    // Written straight into A/B's blocks. The task block used to be assembled in
    // its own A_task/B_task first and copied in, which cost two allocations per
    // rollout node for no benefit; setZero/setIdentity resize and fill in place,
    // where MatrixXd::Zero/Identity build a temporary to assign from.
    A.setZero(nx, nx);
    B.setZero(nx, d);

    // Task rows: [0, nt), the error double integrator plus the frozen-bias channel.
    A.block(0, 0, d, d).setIdentity();
    A.block(0, d, d, d).setIdentity();
    A.block(0, d, d, d) *= Ts;
    A.block(0, 2 * d, d, 1) = (Ts * Ts / 2.0) * bias;
    A.block(d, d, d, d).setIdentity();
    A.block(d, 2 * d, d, 1) = Ts * bias;
    A(2 * d, 2 * d)         = 1.0;

    B.block(0, 0, d, d) = -(Ts * Ts / 2.0) * Lambda_inv;
    B.block(d, 0, d, d) = -Ts * Lambda_inv;

    // Joint-space augmentation. Δq and qdot are states in their own right, so
    // they integrate their own dynamics rather than being reconstructed from
    // the task-space rows: the Cv_joints feedforward cancels the generalised
    // bias exactly (H_g qddot = J^T u), leaving qddot = H_g^-1 J^T u. Held
    // constant across the step, the same assumption and the same integration
    // order as the task rows above:
    //   qdot(k+1) = qdot(k) + Ts * H_g^-1 J^T u
    //   Δq(k+1)   = Δq(k) + Ts*qdot(k) + Ts²/2 * H_g^-1 J^T u
    // Sign is positive where the task rows' is negative: those track an error
    // (x_d − x), these track the joint quantities themselves.

    // Δq rows: [nt, nt+n_q)
    A.block(nt, nt, n_q, n_q).setIdentity();
    A.block(nt, nt + n_q, n_q, n_q).setIdentity();
    A.block(nt, nt + n_q, n_q, n_q) *= Ts;
    // qdot rows: [nt+n_q, nt+2*n_q)
    A.block(nt + n_q, nt + n_q, n_q, n_q).setIdentity();

    // H_g_inv_Jt is indexed by MODEL joint, the state by owned position, so the
    // control blocks gather. Row at a time, rather than through a gathered
    // temporary, so the rollout's ~N calls a tick stay allocation-free.
    for (int k = 0; k < n_q; ++k) {
        const int j = ownedJointInds[static_cast<std::size_t>(k)];
        B.block(nt + k, 0, 1, d)         = (Ts * Ts / 2.0) * H_g_inv_Jt.row(j);
        B.block(nt + n_q + k, 0, 1, d)   = Ts * H_g_inv_Jt.row(j);
    }
}

// Δq/qdot box bounds on the OWNED joints, offset by the current measurement q0.
void ArmNMPC::buildJointBoxBounds(
    const Eigen::VectorXd& q0, Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j)
{
    const int nOwned = static_cast<int>(ownedJointInds.size());
    lbx_j.resize(2 * nOwned);
    ubx_j.resize(2 * nOwned);
    for (int k = 0; k < nOwned; ++k) {
        int i = ownedJointInds[k];
        lbx_j(k)          = qlimsByIndex[i].first  - q0(i);
        ubx_j(k)          = qlimsByIndex[i].second - q0(i);
        lbx_j(nOwned + k) = -vlimsByIndex[i];
        ubx_j(nOwned + k) =  vlimsByIndex[i];
    }
}

// Per-joint torque limit as a general (polytopic) constraint lg <= D*u <= ug,
// with D = J^T mapping task-space torque to joint torque and Cv_joints the
// bias feedforward that finalizeJointTorques() adds back.
void ArmNMPC::buildTorqueGeneralConstraint(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& Cv_joints,
    Eigen::MatrixXd& D, Eigen::VectorXd& lg, Eigen::VectorXd& ug)
{
    D = J.transpose();   // n_j x d
    lg = -tau_lim_vec_ - Cv_joints;
    ug =  tau_lim_vec_ - Cv_joints;
}

// tau_joints = J^T*tau_task + Cv_joints, clamped per-joint to torqueLimsByIndex.
Eigen::VectorXd ArmNMPC::finalizeJointTorques(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& tau_task,
    const Eigen::VectorXd& Cv_joints)
{
    Eigen::VectorXd tau_joints;
    finalizeJointTorques(J, tau_task, Cv_joints, tau_joints);
    return tau_joints;
}

void ArmNMPC::finalizeJointTorques(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& tau_task,
    const Eigen::VectorXd& Cv_joints, Eigen::VectorXd& tau_joints)
{
    // Bias first, then the product accumulated onto it: assigning the whole
    // expression at once evaluates J^T*tau_task into a temporary.
    tau_joints = Cv_joints;
    tau_joints.noalias() += J.transpose() * tau_task;
    for (int i = 0; i < n_joints_; ++i) {
        tau_joints[i] = std::max(-torqueLimsByIndex[i],
                                 std::min(torqueLimsByIndex[i], tau_joints[i]));
    }
}

Eigen::VectorXd ArmNMPC::dampingTorque(const Eigen::VectorXd& v) const
{
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_joints_);
    if (v.size() < 6 + n_joints_) return tau;   // nothing trustworthy to damp against

    for (int idx : ownedJointInds) {
        const double vd = v[6 + idx];
        if (!std::isfinite(vd)) continue; // a poisoned joint gets zero, never NaN
        
        const double lim = torqueLimsByIndex[idx];
        tau[idx] = std::max(-lim, std::min(lim, -kd_[idx] * vd));
    }
    return tau;
}

void ArmNMPC::enterDamping()
{
    mode_ = ControlMode::Damping;
    consecutive_ok_ = 0;
    u_bar_.assign(params_.N, Eigen::VectorXd::Zero(3));
}

void ArmNMPC::clearLatch()
{
    if (mode_ != ControlMode::Latched) return;
    mode_ = ControlMode::Damping;   // back to auto-recovering, not straight to tracking
    damping_run_ = 0;
    consecutive_ok_ = 0;
}

void ArmNMPC::shiftNominalControl()
{
    const int N = static_cast<int>(u_bar_.size());
    if (N < 2) return;

    // The plan advances by the REAL time one tick covers, which is control_period,
    // not by one stage. Shifting a whole stage per tick -- as this did -- advances
    // the warm start Ts/control_period times too fast (2.5x at Ts=25 ms, 100 Hz),
    // so every tick linearises around a trajectory 15 ms of plan ahead of where the
    // arm will actually be.
    //
    // Fractional shift by linear interpolation: u_new[k] = u_old[k + f], with
    // repeat-last-hold past the end, which is what the whole-stage version did at
    // the tail. f == 1 reproduces it exactly.
    const double dt = params_.control_period > 0.0 ? params_.control_period : params_.Ts;
    const double f  = dt / params_.Ts;
    if (f >= 1.0) {                    // ticks at or slower than the grid: plain shift
        for (int k = 0; k < N - 1; ++k) u_bar_[k] = u_bar_[k + 1];
        return;
    }
    for (int k = 0; k < N; ++k) {
        const double src = static_cast<double>(k) + f;
        const int    i   = static_cast<int>(src);
        
        if (i >= N - 1) { 
            u_bar_[k] = u_bar_[N - 1]; 
            continue; 
        }
        const double w = src - static_cast<double>(i);

        // In place is safe here: f < 1 means i == k, so this reads entries k and
        // k+1 and writes k. Entry k+1 still holds its original value, because it
        // is not written until the next iteration.
        u_bar_[k] = (1.0 - w) * u_bar_[i] + w * u_bar_[i + 1];
    }
}

void ArmNMPC::acceptPlan()
{
    // Sizes match after the first tick, so this copy-assigns without allocating.
    u_accepted_         = u_bar_;
    have_accepted_plan_ = true;
    replay_ticks_       = 0;
}

int ArmNMPC::planStageForAge(int age_ticks) const
{
    if (u_accepted_.empty() || age_ticks < 1) return -1;
    const double dt = params_.control_period > 0.0 ? params_.control_period : params_.Ts;
    // age 1 is the first tick after the plan was accepted. The shift already
    // advanced u_bar_ by one stage before the snapshot, so that tick wants stage 0.
    const int stage = static_cast<int>(
        std::floor(static_cast<double>(age_ticks - 1) * dt / params_.Ts));
    return stage < static_cast<int>(u_accepted_.size()) ? stage : -1;
}

Eigen::VectorXd ArmNMPC::onSolveFailure(
    const Eigen::VectorXd& v, const Eigen::MatrixXd* J_task, const Eigen::VectorXd* Cv)
{
    // Tier 3 first: a latched controller stays down whatever this tick looked like.
    if (mode_ == ControlMode::Latched) {
        ++safe_mode_ticks_;
        return dampingTorque(v);
    }

    // Tier 1: replay the accepted plan, if there is one, the budget is not spent,
    // and this tick produced the kinematics needed to turn a wrench into torque.
    const int stage = planStageForAge(replay_ticks_ + 1);
    if (params_.tier1_max_replay_ticks > 0 && have_accepted_plan_ &&
        replay_ticks_ < params_.tier1_max_replay_ticks && stage >= 0 &&
        J_task != nullptr && Cv != nullptr) {
        if (mode_ != ControlMode::PlanReplay) {
            mode_ = ControlMode::PlanReplay;
            ++tier1_entries_;
        }
        ++replay_ticks_;
        ++tier1_ticks_;
        consecutive_ok_ = 0;
        damping_run_    = 0;
        return finalizeJointTorques(*J_task, u_accepted_[static_cast<std::size_t>(stage)], *Cv);
    }

    // Tier 2, and Tier 3 once it has gone on long enough to count as not recovering.
    enterDamping();
    ++safe_mode_ticks_;
    if (params_.tier3_latch_after_ticks > 0 &&
        ++damping_run_ >= params_.tier3_latch_after_ticks) {
        mode_ = ControlMode::Latched;
    }
    return dampingTorque(v);
}

int ArmNMPC::lastQpStatus() const
{
    return qp_ ? qp_->lastStatus() : 0;
}

void ArmNMPC::recordQpIters()
{
    if (!qp_) return;
    const int it = qp_->lastQpIters();
    if (it >= 0) {
        qp_iters.add(static_cast<double>(it));
        qp_time_ms.add(qp_->lastQpTimeMs());
    }
    if (qp_->lastRetried()) ++qp_retries;
}

void ArmNMPC::recordSlack()
{
    const int ns = qp_->nsAt(1);
    if (ns <= 0) return;
    if (slack_lo_.size() != ns) {          // first tick only
        slack_lo_.resize(ns);
        slack_hi_.resize(ns);
    }
    qp_->getSlack(1, slack_lo_.data(), slack_hi_.data());

    // Ask the solver how the buffer is laid out rather than assuming 2*nOwned box
    // rows come first: which groups are soft is configurable, and in this model
    // the box and general counts happen to be equal, so a wrong split would read
    // torque slack and label it box slack without any size mismatch to catch it.
    const int n_box = qp_->nsbxAt(1);
    const int n_gen = qp_->nsgAt(1);
    if (n_box > 0) {
        max_box_slack_ = std::max({max_box_slack_,
                                    slack_lo_.head(n_box).maxCoeff(),
                                    slack_hi_.head(n_box).maxCoeff()});
    }
    if (n_gen > 0) {
        max_tau_slack_ = std::max({max_tau_slack_,
                                    slack_lo_.segment(n_box, n_gen).maxCoeff(),
                                    slack_hi_.segment(n_box, n_gen).maxCoeff()});
    }
}

void ArmNMPC::snapshotPhaseSums()
{
    prev_sums_.dynamics          = dynamics.sum_us;
    prev_sums_.jdotqdot          = jdotqdot.sum_us;
    prev_sums_.lambda            = lambda.sum_us;
    prev_sums_.qpsolve           = qpsolve.sum_us;
    prev_sums_.rollout_dynamics  = rollout_dynamics.sum_us;
    prev_sums_.rollout_integrate = rollout_integrate.sum_us;
    prev_sums_.pinv              = pinv.sum_us;
    prev_sums_.build_model       = build_model.sum_us;
    prev_sums_.defect            = defect.sum_us;
    prev_sums_.trust_region      = trust_region.sum_us;
    prev_sums_.finalize          = finalize.sum_us;
    prev_sums_.hpipm_ms          = qp_time_ms.sum;
}

void ArmNMPC::enableTrace(std::size_t capacity)
{
    trace_cap_ = capacity;
    trace_.clear();
    // The one allocation, here and not on a tick. push_back below can then never
    // reallocate, because recording stops at capacity.
    trace_.reserve(capacity);
    trace_t0_set_ = false;
    // "Record from now", so the baseline is the sums as they stand. Leaving it at
    // zero would charge every tick that ran before this call to the first traced
    // one.
    snapshotPhaseSums();
}

void ArmNMPC::drainTrace(std::vector<TickTrace>& sink)
{
    sink.clear();
    sink.swap(trace_);
    // trace_ now holds whatever buffer `sink` brought, which on the first drain is
    // an empty one. Restoring the reservation here, off the control path, is what
    // keeps the push_back in recordTick from ever allocating.
    trace_.reserve(trace_cap_);
}

void ArmNMPC::recordTick(std::chrono::steady_clock::time_point start)
{
    const auto stop = std::chrono::steady_clock::now();
    const double tick_us =
        std::chrono::duration<double, std::micro>(stop - start).count();
    tick.add(tick_us);

    if (trace_cap_ == 0 || trace_.size() >= trace_cap_) return;

    if (!trace_t0_set_) {
        trace_t0_     = start;
        trace_t0_set_ = true;
    }

    // Each phase's share of THIS tick is how far its running sum moved.
    const auto take = [](double now, double& prev) {
        const double d = now - prev;
        prev = now;
        return static_cast<float>(d);
    };
    TickTrace r{};
    r.t_us    = std::chrono::duration<double, std::micro>(start - trace_t0_).count();
    r.tick_us = static_cast<float>(tick_us);
    r.dynamics          = take(dynamics.sum_us,          prev_sums_.dynamics);
    r.jdotqdot          = take(jdotqdot.sum_us,          prev_sums_.jdotqdot);
    r.lambda            = take(lambda.sum_us,            prev_sums_.lambda);
    r.qpsolve           = take(qpsolve.sum_us,           prev_sums_.qpsolve);
    r.rollout_dynamics  = take(rollout_dynamics.sum_us,  prev_sums_.rollout_dynamics);
    r.rollout_integrate = take(rollout_integrate.sum_us, prev_sums_.rollout_integrate);
    r.pinv              = take(pinv.sum_us,              prev_sums_.pinv);
    r.build_model       = take(build_model.sum_us,       prev_sums_.build_model);
    r.defect            = take(defect.sum_us,            prev_sums_.defect);
    r.trust_region      = take(trust_region.sum_us,      prev_sums_.trust_region);
    r.finalize          = take(finalize.sum_us,          prev_sums_.finalize);
    r.hpipm_us          = take(qp_time_ms.sum, prev_sums_.hpipm_ms) * 1.0e3f;
    r.qp_iters   = qp_ ? qp_->lastQpIters() : -1;
    r.qp_status  = static_cast<int16_t>(lastQpStatus());
    r.qp_iters1  = qp_ ? qp_->firstQpIters() : -1;
    r.qp_status1 = static_cast<int16_t>(qp_ ? qp_->firstQpStatus() : 0);
    r.mode      = static_cast<int8_t>(mode_);   // 0 nominal, 1 replay, 2 damping, 3 latched
    r.retried   = (qp_ && qp_->lastRetried()) ? 1 : 0;
    trace_.push_back(r);
}

void ArmNMPC::snapshotProfile(ProfileSnapshot& out) const
{
    out.tick              = tick;
    out.dynamics          = dynamics;
    out.jdotqdot          = jdotqdot;
    out.lambda            = lambda;
    out.qpsolve           = qpsolve;
    out.rollout_dynamics  = rollout_dynamics;
    out.rollout_integrate = rollout_integrate;
    out.pinv              = pinv;
    out.build_model       = build_model;
    out.defect            = defect;
    out.trust_region      = trust_region;
    out.finalize          = finalize;
    out.qp_iters          = qp_iters;
    out.qp_time_ms        = qp_time_ms;
    out.qp_retries        = qp_retries;
    out.spd_fallbacks     = spd_fallbacks_;
    out.trace_size        = static_cast<long>(trace_.size());
    out.trace_cap         = static_cast<long>(trace_cap_);
    out.trace_full        = traceFull();
    out.max_box_slack     = max_box_slack_;
    out.max_tau_slack     = max_tau_slack_;
    out.qp_max_iter       = params_.qp_max_iter;
}

std::string ArmNMPC::profileReport() const
{
    ProfileSnapshot s;
    snapshotProfile(s);
    return formatProfile(s);
}

std::string ArmNMPC::formatProfile(const ProfileSnapshot& s)
{
    // Phases in execution order, so the report reads as a breakdown rather than a
    // bag of numbers. Both paths are listed; the inactive one shows count 0.
    const std::pair<const char*, const PhaseStat*> phases[] = {
        {"dynamics",   &s.dynamics},   {"jdotqdot",     &s.jdotqdot},
        {"roll_dyn",   &s.rollout_dynamics}, {"roll_intg", &s.rollout_integrate},
        {"pinv",       &s.pinv},       {"lambda",       &s.lambda},
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
    os << "  | peak slack: box=" << s.max_box_slack
       << " torque=" << s.max_tau_slack;
    return os.str();
}

void ArmNMPC::resetProfile()
{
    tick.reset();       dynamics.reset();          jdotqdot.reset();
    lambda.reset();     qpsolve.reset();           rollout_dynamics.reset();
    rollout_integrate.reset();                     pinv.reset();
    build_model.reset(); defect.reset();           trust_region.reset();
    finalize.reset();   qp_iters.reset();          qp_time_ms.reset();
    qp_retries     = 0;
    spd_fallbacks_ = 0;
    // The PhaseStat sums the trace differences against have just gone to zero, so
    // the baseline has to follow or the next tick absorbs the whole window.
    snapshotPhaseSums();
    max_box_slack_ = 0.0;
    max_tau_slack_ = 0.0;
}

// === Full-nonlinear (multiple-shooting SQP) path ================================
// Reached only when NMPCParams::fullNonlinear is true. Ported from VORTEX's
// ArmConstrainedNMPController::evalTaskSpacePositionOnly2D/solveNonlinearMPC —
// same planar (x, y, theta_z) task-space law as controlLaw's linearized branch
// above, just re-evaluated at every rollout node instead of once at k=0.

// Per-node task-space error/bias -- identical math to controlLaw's own
// node-0 computation (see the comment there), just packaged into a
// NodeTaskSpace so solveNonlinearMPC can call it once per rollout node.
ArmNMPC::NodeTaskSpace ArmNMPC::evalTaskSpace(
    const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
    const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
    const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6) const
{
    NodeTaskSpace ts;
    evalTaskSpace(q_node, v_node, ee_pos, ee_quat, J_g6, jdot_qdot6, ts);
    return ts;
}

void ArmNMPC::evalTaskSpace(
    const Eigen::VectorXd& /*q_node*/, const Eigen::VectorXd& v_node,
    const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
    const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
    NodeTaskSpace& ts) const
{
    Eigen::Vector2d p_d(desiredPos[0], desiredPos[1]);   // desiredPos[2] (Z) unused
    Eigen::Vector2d v_d(desiredVel[0], desiredVel[1]);
    Eigen::Vector2d a_d(desiredAcc[0], desiredAcc[1]);
    mjtNum quat_d[4] = {desiredQuat[0], desiredQuat[1], desiredQuat[2], desiredQuat[3]};
    double w_d  = desiredAngVel[2];
    double aw_d = desiredAngAcc[2];

    ts.J_task.resize(3, n_joints_);
    ts.J_task.row(0) = J_g6.row(0);   // x
    ts.J_task.row(1) = J_g6.row(1);   // y
    ts.J_task.row(2) = J_g6.row(5);   // angular-Z
    Eigen::Vector3d v_ee = ts.J_task * v_node.tail(n_joints_);

    // World-frame angle-axis (log-map) orientation error, Z-component only —
    // see controlLaw's identical derivation.
    mjtNum quat_ee_arr[4] = {ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]};
    mjtNum e_orient_local[3];
    mju_subQuat(e_orient_local, quat_d, quat_ee_arr);
    mjtNum e_orient_world[3];
    mju_rotVecQuat(e_orient_world, e_orient_local, quat_ee_arr);
    double e_theta = e_orient_world[2];

    ts.e.resize(3);
    ts.e << p_d - ee_pos.head<2>(), e_theta;
    ts.edot.resize(3);
    ts.edot << v_d - v_ee.head<2>(), w_d - v_ee(2);
    ts.bias.resize(3);
    ts.bias << a_d - jdot_qdot6.head<2>(), aw_d - jdot_qdot6(5);
}

// Multiple-shooting SQP: rolls the nonlinear plant forward N steps under the
// current nominal control u_bar_, re-linearizes (A_k, B_k) at every node from
// that rollout, solves the resulting multi-stage QP for a correction, damps
// it through a du_max trust region, and shifts u_bar_ for next tick's warm
// start. Falls back to the node-0 single-stage QP (same as the linearized
// path) if the multi-stage solve fails. Run params_.sqp_iters times per tick
// (real-time-iteration style at the default of 1).
Eigen::VectorXd ArmNMPC::solveNonlinearMPC(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v,
    int d, const TaskSpaceEval& evalTaskSpace,
    Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
    Eigen::MatrixXd& Q_N)
{
    const int N   = params_.N;
    const int n_j = n_joints_;
    const int n_q = n_owned;        // joint rows in the QP state, owned joints only
    const double Ts = params_.Ts;
    const int nt  = 2 * d + 1;
    const int nx  = nt + 2 * n_q;

    // Warm start / re-init on first tick or a dimension change.
    if (static_cast<int>(u_bar_.size()) != N ||
        (N > 0 && static_cast<int>(u_bar_[0].size()) != d)) {
        u_bar_.assign(N, Eigen::VectorXd::Zero(d));
    }

    // Once-per-tick destinations, held for the same reason as the per-node ones.
    q0_ = q.tail(n_j);                  // fixed measurement this tick, for box bounds
    buildJointBoxBounds(q0_, lbx_j_, ubx_j_);
    Eigen::VectorXd& q0    = q0_;
    Eigen::VectorXd& lbx_j = lbx_j_;
    Eigen::VectorXd& ubx_j = ubx_j_;

    Eigen::MatrixXd& J_task0    = J_task0_;
    Eigen::VectorXd& Cv_joints0 = Cv_joints0_;
    tau_task_result_.setZero(d);
    Eigen::VectorXd& tau_task_result = tau_task_result_;
    bool lastIterOk = false;

    for (int iter = 0; iter < params_.sqp_iters; ++iter) {
        // Members, not locals: six vectors of N matrices plus N+1 state vectors
        // rebuilt every tick is the single largest block of per-tick allocation.
        // Sized once here and reused; each entry's own buffer then survives too,
        // because assigning a same-sized Eigen object does not reallocate.
        if (static_cast<int>(A_k_.size()) != N) {
            A_k_.resize(N); B_k_.resize(N); D_k_.resize(N);
            b_k_.resize(N); lg_k_.resize(N); ug_k_.resize(N);
            x_aug_nodes_.resize(N + 1);
        }
        auto& A_k = A_k_;  auto& B_k = B_k_;  auto& D_k = D_k_;
        auto& b_k = b_k_;  auto& lg_k = lg_k_; auto& ug_k = ug_k_;
        auto& x_aug_nodes = x_aug_nodes_;

        // Members, so the rollout's running state does not allocate a fresh pair
        // of state vectors every tick. Assigning same-sized Eigen objects reuses
        // the existing buffers.
        q_k_ = q;
        v_k_ = v;
        Eigen::VectorXd& q_k = q_k_;
        Eigen::VectorXd& v_k = v_k_;
        for (int k = 0; k <= N; ++k) {
            // Node 0 uses the same (q, v) as the live measured state -- a
            // scratch-buffer mj_forward at those exact inputs reproduces the
            // live computation exactly, so no special-casing is needed here.
            // nd_scratch_ is a member reused across nodes AND across ticks: a
            // local NodeDynamics would allocate its three dynamic Eigen members
            // at every one of the N+1 nodes. Only the current node's is ever
            // needed -- node 0's J_task/Cv are captured separately below.
            NodeDynamics& nd = nd_scratch_;
            {
                ScopedTimer t(rollout_dynamics);
                dynamicsAt(q_k, v_k, ee_site_id_, nd);
            }
            // Same reasoning as nd_scratch_: one instance for all N+1 nodes.
            NodeTaskSpace& ts = ts_scratch_;
            evalTaskSpace(q_k, v_k, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6, ts);

            // Straight into its slot: a local here, copied in below, cost an
            // allocation and a copy at every node.
            Eigen::VectorXd& x_aug_k = x_aug_nodes[k];
            x_aug_k.setZero(nx);
            x_aug_k.segment(0, d) = ts.e;
            x_aug_k.segment(d, d) = ts.edot;
            x_aug_k(2 * d) = 1.0;
            // Owned joints only, in ownedJointInds order, matching the state
            // layout and idxbx_k. q0/q_k/v_k stay full-model: they feed the box
            // bounds and the rollout, which both work in model indices.
            for (int i = 0; i < n_q; ++i) {
                const int j = ownedJointInds[static_cast<std::size_t>(i)];
                x_aug_k(nt + i)       = q_k(7 + j) - q0(j);   // Δq_k
                x_aug_k(nt + n_q + i) = v_k(6 + j);           // qdot_k
            }
            if (k == 0) {
                J_task0    = ts.J_task;
                Cv_joints0 = nd.Cv_joints;
            }

            if (k < N) {
                // Held across nodes, and the model is built straight into its
                // stage slots -- the locals these replace cost four allocations
                // and two full matrix copies at every node.
                computeInertiaProducts(nd.H_g, ts.J_task, hg_inv_Jt_, lambda_inv_);
                {
                    ScopedTimer t_bm(build_model);
                    buildAugmentedModel(lambda_inv_, ts.bias, hg_inv_Jt_,
                                        A_k[k], B_k[k]);
                }

                buildTorqueGeneralConstraint(ts.J_task, nd.Cv_joints, D_k[k], lg_k[k], ug_k[k]);

                // Propagate to node k+1 under the CURRENT nominal control.
                // Clamped to torqueLimsByIndex before integrating -- the QP's
                // torque bound is soft (slack-permitted), and a stale u_bar_
                // entry from a prior tick's slack-violating solve can
                // otherwise inject an unphysically large open-loop torque
                // here, which the real system would never actually apply
                // (finalizeJointTorques clamps every torque it hands out).
                // An unclamped rollout can run away into a kinematic
                // singularity within the horizon (confirmed via VORTEX
                // testing: J_task collapsing to near-zero singular values a
                // few nodes in, eventually producing a NaN Lambda_inv) --
                // keeping the rollout's plant model consistent with the real
                // actuator saturation avoids that.
                // The torque handed to the plant model is the same one
                // finalizeJointTorques() applies for real: J^T u + Cv_joints
                // (feedforward included), clamped as a whole -- so the
                // rollout predicts the plant that actually runs, and matches
                // the QP's own model (H_g qddot = J^T u).
                tau_joints_k_.noalias() = ts.J_task.transpose() * u_bar_[k];
                tau_joints_k_ += nd.Cv_joints;
                for (int i = 0; i < n_j; ++i) {
                    tau_joints_k_[i] = std::max(-torqueLimsByIndex[i],
                                                std::min(torqueLimsByIndex[i], tau_joints_k_[i]));
                }
                {
                    ScopedTimer t(rollout_integrate);
                    integrateStep(q_k, v_k, tau_joints_k_, Ts, q_next_, v_next_);
                }

                // Rollout velocity trust region: hard-clamp the PREDICTED
                // joint velocity to rollout_v_clamp_mult * vlimsByIndex
                // before it feeds the next node. This is what actually
                // prevents the open-loop rollout from running away (the
                // torque clamp above bounds the INPUT, but zero/small torque
                // plus already-nonzero velocity can still diverge via
                // Coriolis coupling alone). Applying this to the clamped
                // v_next_ before it's used to build the NEXT node's x_aug is
                // what makes the b_k defect (below) transparently reflect
                // the true (now-bounded) trajectory -- no separate change to
                // the defect formula is needed.
                for (int i = 0; i < n_j; ++i) {
                    double lim = params_.rollout_v_clamp_mult * vlimsByIndex[i];
                    v_next_[6 + i] = std::max(-lim, std::min(lim, v_next_[6 + i]));
                }

                // Rollout position clamp, to the MECHANICAL range only. The
                // velocity clamp alone still lets q_k walk to a physically
                // extreme configuration over the horizon (nothing else bounds
                // q_k during the OPEN-LOOP rollout), producing a kinematic
                // (Jacobian rank-loss) singularity independent of velocity.
                //
                // modelQlimsByIndex, NOT qlimsByIndex: clamping to the software
                // margin would make the joint box constraint vacuous. The box is
                // written on Δq = q_k − q0, so clamping the rollout into the
                // margin puts the nominal trajectory inside the box by
                // construction, and b_k then hands the QP that teleport as free
                // dynamics -- the constraint could never bind. Clamping to the
                // mechanical range keeps the singularity protection (that range
                // IS the reachable configuration space) while leaving the margin
                // for the QP to enforce, which is the whole point of having one.
                for (int i = 0; i < n_j; ++i) {
                    q_next_[7 + i] = std::max(modelQlimsByIndex[i].first,
                                              std::min(modelQlimsByIndex[i].second,
                                                        q_next_[7 + i]));
                }

                q_k = q_next_;
                v_k = v_next_;
            }
        }

        // Direct-multiple-shooting linearization defect at every node:
        // b_k = x_{k+1}_bar - A_k*x_k_bar - B_k*u_bar_[k].
        {
            ScopedTimer t_def(defect);
            for (int k = 0; k < N; ++k) {
                // Accumulated in place rather than as one expression: Eigen
                // evaluates a matrix-vector product nested inside a larger
                // expression into a temporary, which is a fresh allocation at
                // every node. noalias() is safe because b_k[k] is neither
                // operand.
                b_k[k] = x_aug_nodes[k + 1];
                b_k[k].noalias() -= A_k[k] * x_aug_nodes[k];
                b_k[k].noalias() -= B_k[k] * u_bar_[k];
            }
        }

        bool ok;
        {
            ScopedTimer t(qpsolve);
            ok = qp_->solveMultiStage(
                A_k, B_k, b_k, Q.data(), R.data(),
                lbx_j.data(), ubx_j.data(),
                D_k, lg_k, ug_k, x_aug_nodes[0].data(),
                Q_N.data());
        }
        recordQpIters();
        // A solver that reports success but returns non-finite controls is a
        // failed solve: letting it into u_bar_ poisons every later rollout.
        if (ok) {
            uk_.resize(d);
            for (int k = 0; k < N && ok; ++k) {
                qp_->getU(k, uk_.data());
                ok = uk_.allFinite();
            }
        }
        lastIterOk = ok;
        if (ok) {
            // Control trust region: damp each freshly solved uk relative to
            // u_bar_[k]'s value AT THE START of this iteration -- the point
            // the rollout was just linearized around. At sqp_iters=1 (the
            // default), u_bar_[k] entering iteration 0 already IS last
            // tick's trusted/shifted value, so this one clamp site covers
            // both tick-to-tick and (when sqp_iters>1) iteration-to-iteration
            // damping -- no separate site needed.
            //
            // Sized from node 0's Jacobian, the point the QP was just built
            // around, so the region tracks the arm's actual authority in this
            // configuration instead of a fixed wrench that stops binding the
            // moment either the configuration or the torque limits change.
            recordSlack();

            {
                ScopedTimer t_tr(trust_region);
                reachableWrenchBound(J_task0, wrench_bound_);
                du_max_ = params_.du_max_frac * wrench_bound_;
                const Eigen::VectorXd& du_max = du_max_;
                // uk_ is a member, so this loop and the finite check above reuse
                // one buffer instead of allocating a 3-vector per stage per tick.
                uk_.resize(d);
                for (int k = 0; k < N; ++k) {
                    qp_->getU(k, uk_.data());
                    u_bar_[k] = u_bar_[k] + (uk_ - u_bar_[k]).cwiseMax(-du_max).cwiseMin(du_max);
                }
            }
            tau_task_result = u_bar_[0];
        } else {
            // No approximate substitute is attempted: safe mode takes over
            // below, which is predictable in a way an unconstrained fallback
            // solve is not. The status reaches the ROS node via lastQpStatus()
            // rather than being logged here, since this runs every tick while
            // the solver is marginal.
            //
            // Remaining SQP iterations are skipped: they would relinearize
            // around an untrusted trajectory.
            break;
        }
    }

    // Shift for next tick's warm start (once per tick, after all SQP
    // iterations -- NOT once per inner iteration, which should keep
    // re-linearizing around the just-solved trajectory). Only valid when the
    // LAST iteration's solveMultiStage() actually succeeded -- when it didn't,
    // u_bar_[0..N-1] are whatever they were before this tick (stale, possibly
    // from an earlier failed tick too), and shifting would promote those into
    // next tick's rollout input. enterDamping() zeroes the whole trajectory
    // instead, so the next rollout coasts rather than compounding whatever
    // caused this tick's solve to fail.
    if (!lastIterOk) {
        // Tier 1/2/3. J_task0 and Cv_joints0 come from rollout node 0, i.e. the
        // CURRENT measurement, and are valid even though the solve failed -- the
        // rollout runs before the QP.
        return onSolveFailure(v, &J_task0, &Cv_joints0);
    }

    shiftNominalControl();
    // u_bar_[N-1] keeps its last value (repeat-last-hold). Snapshot after the
    // shift, so the plan's element 0 is the control for the NEXT tick.
    acceptPlan();

    // Tier 1 is a transient, so one good solve ends it -- no hysteresis, since
    // replay was still tracking the setpoint and the handover is between two
    // task-space commands rather than from damping.
    if (mode_ == ControlMode::PlanReplay) {
        mode_ = ControlMode::Nominal;
    }

    // Tier 2 is sticky: hold the damping command until the solver has been healthy
    // for safe_mode_recovery_ticks in a row, so a marginal solve cannot chatter the
    // output between damping and full tracking. Hysteresis only -- the handover on
    // recovery is a step, not a ramp. Tier 3 does not recover here at all.
    if (mode_ == ControlMode::Damping || mode_ == ControlMode::Latched) {
        ++safe_mode_ticks_;
        if (mode_ == ControlMode::Latched ||
            ++consecutive_ok_ < params_.safe_mode_recovery_ticks) {
            return dampingTorque(v);
        }
        mode_ = ControlMode::Nominal;
        damping_run_ = 0;
    }

    ScopedTimer t_fin(finalize);
    return finalizeJointTorques(J_task0, tau_task_result, Cv_joints0);
}

} // namespace pat_arm_nmpc