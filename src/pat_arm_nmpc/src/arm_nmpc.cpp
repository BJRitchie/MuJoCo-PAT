#include "pat_arm_nmpc/arm_nmpc.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pat_arm_nmpc {

namespace {
double avgOrNan(double t, long n) { return n > 0 ? t / static_cast<double>(n) : 0.0; }
}  // namespace

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
        qlimsByIndex = modelJointRanges();
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
    //   augmented state nx = nx_task + 2*n_joints (the Delta-q / qdot rows).
    // Matches VORTEX's non-pose branch (use_pose_dims == false).
    const int d        = 3;
    const int nx_task  = 2 * d + 1;   // 7
    const int n_j      = n_joints_;

    quad_prob_solver::QuadProbSolverParams qp_params;
    qp_params.nx_task      = nx_task;
    qp_params.nu           = d;
    qp_params.nx           = nx_task + 2 * n_j;
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

    qp_params.idxbx_k.resize(2 * n_owned);
    for (int i = 0; i < n_owned; ++i) {
        qp_params.idxbx_k[i]           = nx_task + ownedJointInds[i];         // Δq row for owned joint i
        qp_params.idxbx_k[n_owned + i] = nx_task + n_j + ownedJointInds[i];   // qdot row for owned joint i
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
        // A non-finite intermediate (checkNaN), a solver size mismatch, or an
        // Eigen failure. Damping depends on none of that machinery, so it is
        // still a valid command here — and it is finite by construction even if
        // the measured velocity is not.
        enterDamping();
        ++safe_mode_ticks_;
        return dampingTorque(v);
    }
}

Eigen::VectorXd ArmNMPC::controlLawImpl(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    // Record start time of control loop
    auto start = std::chrono::high_resolution_clock::now();

    const int n_j = n_joints_;
    const int nx  = 7 + 2 * n_j;   // 2*d+1 + 2*n_j, d=3

    // --- Stage/terminal cost (task-space channels only) -------------------------
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
    Q(0, 0) = params_.Qx;    Q(1, 1) = params_.Qy;    Q(2, 2) = params_.Qwz;
    Q(3, 3) = params_.Qdotx; Q(4, 4) = params_.Qdoty; Q(5, 5) = params_.Qdotwz;
    Eigen::MatrixXd R = Eigen::Vector3d(params_.Rx, params_.Ry, params_.Rwz).asDiagonal();
    Eigen::MatrixXd Q_N = Q * params_.terminal_cost_multiplier;   // == Q when multiplier is the default 1.0
    Eigen::VectorXd out;

    if (params_.fullNonlinear) {
        TaskSpaceEval eval = [this](
            const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
            const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
            const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6) {
            return evalTaskSpace(q_node, v_node, ee_pos, ee_quat, J_g6, jdot_qdot6);
        };
        out = solveNonlinearMPC(q, v, /*d=*/3, eval, Q, R, Q_N);
    } else {
        // --- Current EE state -------------------------------------------------------
        Eigen::Vector3d  p_ee    = eePosition(ee_site_id_);
        Eigen::Vector4d  quat_ee = eeOrientation(ee_site_id_);   // [w,x,y,z]
        Eigen::MatrixXd  J_g6;
        {
            ScopedTimer t(t_lambda_us_, n_lambda_);
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
        Eigen::MatrixXd H_g_inv = computeGeneralizedInertiaInv(Cv_joints);
        Eigen::MatrixXd Lambda_inv = computeLambdaInv(J_g, H_g_inv);

        Eigen::Matrix<double, 6, 1> jdot_qdot6;
        {
            ScopedTimer t(t_jdotqdot_us_, n_jdotqdot_);
            jdot_qdot6 = jdotQdot6(ee_site_id_);
        }
        Eigen::Vector3d bias;
        bias << a_d - jdot_qdot6.head<2>(), aw_d - jdot_qdot6(5);

        Eigen::MatrixXd A, B;
        buildAugmentedModel(Lambda_inv, bias, H_g_inv * J_g.transpose(), A, B);

        Eigen::VectorXd q0 = q.tail(n_j);   // current measured joint angles

        Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(nx);
        x_aug.head<3>()     = e;
        x_aug.segment<3>(3) = edot;
        x_aug(6) = 1.0;
        // Δq/qdot rows of x_aug stay 0 — see controlLawPositionOnly's identical
        // comment on why.

        Eigen::VectorXd lbx_j, ubx_j;
        buildJointBoxBounds(q0, lbx_j, ubx_j);
        const int nOwned = static_cast<int>(ownedJointInds.size());

        Eigen::MatrixXd D;
        Eigen::VectorXd lg, ug;
        buildTorqueGeneralConstraint(J_g, Cv_joints, D, lg, ug);

        Eigen::VectorXd tau_task;
        if (!solveStageQP(A, B, Q, R, lbx_j, ubx_j, D, lg, ug,
                          x_aug, nOwned, Q_N, tau_task)) {
            enterDamping();
            ++safe_mode_ticks_;
            printTimingInfo(start);
            return dampingTorque(v);
        }
        if (mode_ == ControlMode::Damping) {
            ++safe_mode_ticks_;
            if (++consecutive_ok_ < params_.safe_mode_recovery_ticks) {
                printTimingInfo(start);
                return dampingTorque(v);
            }
            mode_ = ControlMode::Nominal;   // recovered; fall through to tracking
        }

        out = finalizeJointTorques(J_g, tau_task, Cv_joints);
    }

    printTimingInfo(start);
    return out;
}

// === Shared helpers ============================================================
// Factored out of the three controlLaw* functions below -- they differ only
// in task-space dimension d (3 for position-only, 2 for planar, 6 for pose),
// and Eigen's dynamic-sized matrices make one d-agnostic implementation of
// each step trivial to share. See armConstrainedNMPController.h for the doc
// comment on each.

Eigen::MatrixXd ArmNMPC::computeGeneralizedInertiaInv(
    Eigen::VectorXd& Cv_joints)
{
    Eigen::MatrixXd H_g(n_joints_, n_joints_);
    Cv_joints.resize(n_joints_);
    {
        ScopedTimer t(t_dynamics_us_, n_dynamics_);
        getDynamics(H_g, Cv_joints);
    }
    Eigen::MatrixXd H_g_inv;
    {
        ScopedTimer t(t_dynamics_us_, n_dynamics_);
        H_g_inv = pinvDLS(H_g, params_.Hg_damping, params_.ee_site_name + " H_g");
    }
    return H_g_inv;
}

Eigen::VectorXd ArmNMPC::reachableWrenchBound(const Eigen::MatrixXd& J_task) const
{
    // Finite cap for channels the Jacobian barely couples to: tau/|J| runs away
    // as |J| -> 0, and HPIPM's factorization does not tolerate bounds spanning
    // many orders of magnitude.
    constexpr double kMaxBound = 1.0e3;

    const int d = static_cast<int>(J_task.rows());
    Eigen::VectorXd bound(d);
    for (int j = 0; j < d; ++j) {
        double lim = kMaxBound;

        for (int i = 0; i < n_joints_; ++i) {
            const double a = std::abs(J_task(j, i));
            
            if (a > 1e-9) 
                lim = std::min(lim, torqueLimsByIndex[i] / a);
        }
        bound(j) = std::min(lim, kMaxBound);
    }
    return bound;
}

Eigen::MatrixXd ArmNMPC::computeLambdaInv(
    const Eigen::MatrixXd& J, const Eigen::MatrixXd& H_g_inv)
{
    Eigen::MatrixXd Lambda_inv;
    {
        ScopedTimer t(t_lambda_us_, n_lambda_);
        Lambda_inv = (J * H_g_inv * J.transpose()).eval();
    }
    checkNaN(Lambda_inv.hasNaN(), "Lambda_inv");
    return Lambda_inv;
}

bool ArmNMPC::solveStageQP(
    Eigen::MatrixXd& A, Eigen::MatrixXd& B,
    Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
    Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
    Eigen::MatrixXd& D, Eigen::VectorXd& lg, Eigen::VectorXd& ug,
    Eigen::VectorXd& x_aug, int nOwned, Eigen::MatrixXd& Q_N,
    Eigen::VectorXd& tau_task_out)
{
    const int nx = static_cast<int>(A.rows());
    const int nu = static_cast<int>(B.cols());

    tau_task_out.resize(nu);
    bool ok = false;
    {
        ScopedTimer t(t_qpsolve_us_, n_qpsolve_);

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
            recordSlack(nOwned);
        }
    }
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
    const int n_j = n_joints_;
    const double Ts = params_.Ts;
    const int nt  = 2 * d + 1;          // task-space block: pos + vel + bias channel
    const int nx  = nt + 2 * n_j;

    Eigen::MatrixXd A_task = Eigen::MatrixXd::Zero(nt, nt);
    A_task.block(0, 0, d, d)     = Eigen::MatrixXd::Identity(d, d);
    A_task.block(0, d, d, d)     = Ts * Eigen::MatrixXd::Identity(d, d);
    A_task.block(0, 2 * d, d, 1) = (Ts * Ts / 2.0) * bias;
    A_task.block(d, d, d, d)     = Eigen::MatrixXd::Identity(d, d);
    A_task.block(d, 2 * d, d, 1) = Ts * bias;
    A_task(2 * d, 2 * d)         = 1.0;

    Eigen::MatrixXd B_task = Eigen::MatrixXd::Zero(nt, d);
    B_task.block(0, 0, d, d) = -(Ts * Ts / 2.0) * Lambda_inv;
    B_task.block(d, 0, d, d) = -Ts * Lambda_inv;

    A = Eigen::MatrixXd::Zero(nx, nx);
    B = Eigen::MatrixXd::Zero(nx, d);
    A.block(0, 0, nt, nt) = A_task;
    B.block(0, 0, nt, d)  = B_task;

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

    // Δq rows: [nt, nt+n_j)
    A.block(nt, nt, n_j, n_j)       = Eigen::MatrixXd::Identity(n_j, n_j);
    A.block(nt, nt + n_j, n_j, n_j) = Ts * Eigen::MatrixXd::Identity(n_j, n_j);
    B.block(nt, 0, n_j, d)          = (Ts * Ts / 2.0) * H_g_inv_Jt;
    // qdot rows: [nt+n_j, nt+2*n_j)
    A.block(nt + n_j, nt + n_j, n_j, n_j) = Eigen::MatrixXd::Identity(n_j, n_j);
    B.block(nt + n_j, 0, n_j, d)          = Ts * H_g_inv_Jt;
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
    Eigen::VectorXd tauLimVec(n_joints_);
    for (int i = 0; i < n_joints_; ++i) tauLimVec(i) = torqueLimsByIndex[i];
    lg = -tauLimVec - Cv_joints;
    ug =  tauLimVec - Cv_joints;
}

// tau_joints = J^T*tau_task + Cv_joints, clamped per-joint to torqueLimsByIndex.
Eigen::VectorXd ArmNMPC::finalizeJointTorques(
    const Eigen::MatrixXd& J, const Eigen::VectorXd& tau_task,
    const Eigen::VectorXd& Cv_joints)
{
    Eigen::VectorXd tau_joints = J.transpose() * tau_task + Cv_joints;
    for (int i = 0; i < n_joints_; ++i) {
        tau_joints[i] = std::max(-torqueLimsByIndex[i],
                                 std::min(torqueLimsByIndex[i], tau_joints[i]));
    }
    return tau_joints;
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

int ArmNMPC::lastQpStatus() const
{
    return qp_ ? qp_->lastStatus() : 0;
}

void ArmNMPC::recordSlack(int nOwned)
{
    const int ns = qp_->nsAt(1);
    if (ns <= 0) return;
    if (slack_lo_.size() != ns) {          // first tick only
        slack_lo_.resize(ns);
        slack_hi_.resize(ns);
    }
    qp_->getSlack(1, slack_lo_.data(), slack_hi_.data());

    const int n_box = 2 * nOwned;
    if (n_box > 0) {
        max_box_slack_ = std::max({max_box_slack_,
                                    slack_lo_.head(n_box).maxCoeff(),
                                    slack_hi_.head(n_box).maxCoeff()});
    }
    if (ns > n_box) {
        max_tau_slack_ = std::max({max_tau_slack_,
                                    slack_lo_.tail(ns - n_box).maxCoeff(),
                                    slack_hi_.tail(ns - n_box).maxCoeff()});
    }
}

void ArmNMPC::printTimingInfo(
    const std::chrono::time_point<std::chrono::high_resolution_clock> start,
    int modulo)
{
    auto stop = std::chrono::high_resolution_clock::now();
    t_total_us_ += std::chrono::duration<double, std::micro>(stop - start).count();
    ++n_total_;
    if (modulo > 0 && n_total_ % modulo == 0) {
        std::cout << "[ArmNMPC profile] avg over " << n_total_ << " calls (us):"
            << "  dynamics=" << (n_dynamics_ > 0 ? std::to_string(avgOrNan(t_dynamics_us_, n_dynamics_)) : "n/a")
            << "  jdotqdot=" << (n_jdotqdot_ > 0 ? std::to_string(avgOrNan(t_jdotqdot_us_, n_jdotqdot_)) : "n/a")
            << "  lambda="   << avgOrNan(t_lambda_us_,  n_lambda_)
            << "  qpsolve="  << avgOrNan(t_qpsolve_us_, n_qpsolve_)
            << "  TOTAL="    << avgOrNan(t_total_us_,   n_total_)
            << "  | peak slack: box=" << max_box_slack_
            << " torque="             << max_tau_slack_
            << "\n";
        // Peaks are per-report, not cumulative: a strain that has passed should
        // stop being reported.
        max_box_slack_ = 0.0;
        max_tau_slack_ = 0.0;
    }
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
    const Eigen::VectorXd& /*q_node*/, const Eigen::VectorXd& v_node,
    const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
    const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6) const
{
    NodeTaskSpace ts;
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
    return ts;
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
    const double Ts = params_.Ts;
    const int nOwned = static_cast<int>(ownedJointInds.size());
    const int nt  = 2 * d + 1;
    const int nx  = nt + 2 * n_j;

    // Warm start / re-init on first tick or a dimension change.
    if (static_cast<int>(u_bar_.size()) != N ||
        (N > 0 && static_cast<int>(u_bar_[0].size()) != d)) {
        u_bar_.assign(N, Eigen::VectorXd::Zero(d));
    }

    Eigen::VectorXd q0 = q.tail(n_j);   // fixed measurement this tick, for box bounds
    Eigen::VectorXd lbx_j, ubx_j;
    buildJointBoxBounds(q0, lbx_j, ubx_j);

    Eigen::MatrixXd J_task0;
    Eigen::VectorXd Cv_joints0;
    Eigen::VectorXd tau_task_result = Eigen::VectorXd::Zero(d);
    bool lastIterOk = false;

    for (int iter = 0; iter < params_.sqp_iters; ++iter) {
        std::vector<Eigen::MatrixXd> A_k(N), B_k(N), D_k(N);
        std::vector<Eigen::VectorXd> b_k(N), lg_k(N), ug_k(N);
        std::vector<Eigen::VectorXd> x_aug_nodes(N + 1);

        Eigen::VectorXd q_k = q, v_k = v;
        for (int k = 0; k <= N; ++k) {
            // Node 0 uses the same (q, v) as the live measured state -- a
            // scratch-buffer mj_forward at those exact inputs reproduces the
            // live computation exactly, so no special-casing is needed here.
            NodeDynamics nd;
            {
                ScopedTimer t(t_rollout_dynamics_us_, n_rollout_dynamics_);
                nd = dynamicsAt(q_k, v_k, ee_site_id_);
            }
            NodeTaskSpace ts = evalTaskSpace(q_k, v_k, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

            Eigen::VectorXd x_aug_k = Eigen::VectorXd::Zero(nx);
            x_aug_k.segment(0, d) = ts.e;
            x_aug_k.segment(d, d) = ts.edot;
            x_aug_k(2 * d) = 1.0;
            x_aug_k.segment(nt, n_j)       = q_k.tail(n_j) - q0;   // Δq_k
            x_aug_k.segment(nt + n_j, n_j) = v_k.tail(n_j);        // qdot_k
            x_aug_nodes[k] = x_aug_k;

            if (k == 0) {
                J_task0    = ts.J_task;
                Cv_joints0 = nd.Cv_joints;
            }

            if (k < N) {
                Eigen::MatrixXd H_g_inv = pinvDLS(
                    nd.H_g, params_.Hg_damping,
                    params_.ee_site_name + " H_g (nonlinear rollout node " + std::to_string(k) + ")");
                Eigen::MatrixXd Lambda_inv = computeLambdaInv(ts.J_task, H_g_inv);
                Eigen::MatrixXd H_g_inv_Jt = H_g_inv * ts.J_task.transpose();

                Eigen::MatrixXd A_node, B_node;
                buildAugmentedModel(Lambda_inv, ts.bias, H_g_inv_Jt, A_node, B_node);
                A_k[k] = A_node;
                B_k[k] = B_node;

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
                Eigen::VectorXd tau_joints_k = ts.J_task.transpose() * u_bar_[k] + nd.Cv_joints;
                for (int i = 0; i < n_j; ++i) {
                    tau_joints_k[i] = std::max(-torqueLimsByIndex[i],
                                                std::min(torqueLimsByIndex[i], tau_joints_k[i]));
                }
                Eigen::VectorXd q_next, v_next;
                {
                    ScopedTimer t(t_rollout_integrate_us_, n_rollout_integrate_);
                    integrateStep(q_k, v_k, tau_joints_k, Ts, q_next, v_next);
                }

                // Rollout velocity trust region: hard-clamp the PREDICTED
                // joint velocity to rollout_v_clamp_mult * vlimsByIndex
                // before it feeds the next node. This is what actually
                // prevents the open-loop rollout from running away (the
                // torque clamp above bounds the INPUT, but zero/small torque
                // plus already-nonzero velocity can still diverge via
                // Coriolis coupling alone). Applying this to the clamped
                // v_next before it's used to build the NEXT node's x_aug is
                // what makes the b_k defect (below) transparently reflect
                // the true (now-bounded) trajectory -- no separate change to
                // the defect formula is needed.
                for (int i = 0; i < n_j; ++i) {
                    double lim = params_.rollout_v_clamp_mult * vlimsByIndex[i];
                    v_next[6 + i] = std::max(-lim, std::min(lim, v_next[6 + i]));
                }

                // Rollout position clamp: hard-clamp predicted joint position
                // to qlimsByIndex too -- the velocity clamp alone still lets
                // q_k walk to a physically extreme/degenerate configuration
                // over the horizon (nothing else bounds q_k during the
                // OPEN-LOOP rollout; qlimsByIndex is only a SOFT QP box
                // constraint on the SOLUTION, stages 1..N, not a hard bound
                // on the rollout's own predicted trajectory), producing a
                // genuine kinematic (Jacobian rank-loss) singularity
                // independent of velocity. Same transparency to the b_k
                // defect math as the velocity clamp above.
                for (int i = 0; i < n_j; ++i) {
                    q_next[7 + i] = std::max(qlimsByIndex[i].first,
                                              std::min(qlimsByIndex[i].second, q_next[7 + i]));
                }

                q_k = q_next;
                v_k = v_next;
            }
        }

        // Direct-multiple-shooting linearization defect at every node:
        // b_k = x_{k+1}_bar - A_k*x_k_bar - B_k*u_bar_[k].
        for (int k = 0; k < N; ++k) {
            b_k[k] = x_aug_nodes[k + 1] - A_k[k] * x_aug_nodes[k] - B_k[k] * u_bar_[k];
        }

        bool ok;
        {
            ScopedTimer t(t_qpsolve_us_, n_qpsolve_);
            ok = qp_->solveMultiStage(
                A_k, B_k, b_k, Q.data(), R.data(),
                lbx_j.data(), ubx_j.data(),
                D_k, lg_k, ug_k, x_aug_nodes[0].data(),
                Q_N.data());
        }
        // A solver that reports success but returns non-finite controls is a
        // failed solve: letting it into u_bar_ poisons every later rollout.
        if (ok) {
            Eigen::VectorXd uk(d);
            for (int k = 0; k < N && ok; ++k) {
                qp_->getU(k, uk.data());
                ok = uk.allFinite();
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
            recordSlack(nOwned);

            const Eigen::VectorXd du_max =
                params_.du_max_frac * reachableWrenchBound(J_task0);
            for (int k = 0; k < N; ++k) {
                Eigen::VectorXd uk(d);
                qp_->getU(k, uk.data());
                u_bar_[k] = u_bar_[k] + (uk - u_bar_[k]).cwiseMax(-du_max).cwiseMin(du_max);
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
    if (lastIterOk) {
        for (int k = 0; k < N - 1; ++k) u_bar_[k] = u_bar_[k + 1];
        // u_bar_[N-1] keeps its last value (repeat-last-hold).
    } else {
        enterDamping();
    }

    // Safe mode is sticky: hold the damping command until the solver has been
    // healthy for safe_mode_recovery_ticks in a row, so a marginal solve cannot
    // chatter the output between damping and full tracking. Hysteresis only --
    // the handover on recovery is a step, not a ramp.
    if (mode_ == ControlMode::Damping) {
        ++safe_mode_ticks_;
        if (!lastIterOk || ++consecutive_ok_ < params_.safe_mode_recovery_ticks) {
            return dampingTorque(v);
        }
        mode_ = ControlMode::Nominal;
    }

    return finalizeJointTorques(J_task0, tau_task_result, Cv_joints0);
}

} // namespace pat_arm_nmpc