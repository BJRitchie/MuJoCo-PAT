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
    ee_site_id_ = siteIdFromName(params_.ee_site_name);

    if (params_.joint_lims.empty()) {
        throw std::runtime_error(
            "ArmNMPC: NMPCParams.joint_lims must be set");
    } else {
        // Parse joint lims etc 
        for (const auto& lims : params_.joint_lims) {
            qlims[lims.name] = std::make_pair(lims.q_min, lims.q_max); 
            vlims[lims.name] = lims.qd_max; 
            torque_lims[lims.name] = lims.tau_max; 
        }

        // Re-index by arm-local joint index (0..n_joints_-1) for cheap per-tick
        // lookups instead of re-hashing joint names every control tick.
        //
        // Joints NOT listed in joint_lims (the other arm's joints, when one
        // controller owns a subset of a multi-arm model) are never box-
        // constrained: buildJointBoxBounds only touches ownedJointInds, so the
        // q/v defaults set here for the non-owned joints are dead. They DO
        // appear in the all-model-joint torque general constraint, though, so
        // that default matters: give them the same nominal magnitude as the
        // global torque clamp. NOT 0 (a +-0 bound strains against their
        // Coriolis bias every tick) and NOT 1e9 (HPIPM's IPM factorization
        // fails with qp_status 2 once box magnitudes span ~1e0..1e9 — seen in
        // the VORTEX sim; keep every bound within an order of magnitude).
        const double tau_default = params_.tau_max > 0.0 ? params_.tau_max : 100.0;
        qlimsByIndex.assign(n_joints_, std::make_pair(-M_PI, M_PI));
        vlimsByIndex.assign(n_joints_, 10.0);
        torqueLimsByIndex.assign(n_joints_, tau_default);
        for (const auto& [jname, bounds] : qlims) {
            int idx = jointIndexFromName(jname);
            qlimsByIndex[idx]      = bounds;
            vlimsByIndex[idx]      = vlims.at(jname);
            torqueLimsByIndex[idx] = torque_lims.at(jname);
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
        Eigen::VectorXd du_max(3);
        du_max << params_.du_max_x, params_.du_max_y, params_.du_max_wz;

        out = solveNonlinearMPC(q, v, /*d=*/3, eval, Q, R, du_max, Q_N);
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

        Eigen::VectorXd v_d_task(3);
        v_d_task << v_d, w_d;

        Eigen::MatrixXd A, B;
        buildAugmentedModel(J_g, Lambda_inv, bias, v_d_task, A, B);

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

        Eigen::VectorXd tau_task = solveQPWithFallback(
            A, B, Q, R, lbx_j, ubx_j, D, lg, ug, x_aug, nOwned, Q_N);

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

Eigen::VectorXd ArmNMPC::solveQPWithFallback(
    Eigen::MatrixXd& A, Eigen::MatrixXd& B,
    Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
    Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
    Eigen::MatrixXd& D, Eigen::VectorXd& lg, Eigen::VectorXd& ug,
    Eigen::VectorXd& x_aug, int nOwned, Eigen::MatrixXd& Q_N)
{
    const int nx = static_cast<int>(A.rows());
    const int nu = static_cast<int>(B.cols());

    Eigen::VectorXd tau_task(nu);
    {
        ScopedTimer t(t_qpsolve_us_, n_qpsolve_);

        // Native HPIPM affine offset b stays zero every stage -- the frozen
        // acceleration bias is carried through A's constant channel instead
        // (x_aug's trailing "1" row), same trick as ArmNMPController's
        // closed-form Riccati recursion.
        Eigen::VectorXd zero_b = Eigen::VectorXd::Zero(nx);

        bool ok = qp_->solve(
            A.data(), B.data(), zero_b.data(),
            Q.data(), R.data(),
            lbx_j.data(), ubx_j.data(),
            D.data(), lg.data(), ug.data(),
            x_aug.data(),
            Q_N.data());
        
        if (!ok) {
            // The Δq/qdot joint-limit rows AND the torque general
            // constraint are both SOFT now (see NMPCParams::
            // joint_limit_slack_linear/torque_slack_linear's doc
            // comments), so a solve failure here is no longer the routine
            // "a limit made this infeasible" case — the slack absorbs
            // that instead. A nonzero status now means the IPM genuinely
            // didn't converge within qp_max_iter iterations (a purely
            // numerical failure), so this still fails SAFE (hold last good
            // torque) rather than zeroing, which would otherwise inject a
            // torque discontinuity into the next tick's linearization.
            std::cerr << "[WARNING] ArmNMPC["<< params_.ee_site_name.c_str() << "]: "
                      << "QP solve failed/did not converge — defaulting to backward Riccati recursion"
                      << std::endl; 

            // Default to backward Riccati recursion (unconstrained fallback,
            // same math as ArmNMPController's primary solve path -- but P/K
            // must be sized to nx/nu, NOT the bare task-space dims, since
            // Q/A/B/x_aug here are the Δq/qdot-augmented matrices, not the
            // un-augmented task-space-only ones. S is nu x nu (dynamic) --
            // B^T*P*B is genuinely nu x nu regardless of nx, since it's
            // sized by B's column count.
            Eigen::MatrixXd P = Q_N;                            // nx x nx, terminal cost
            Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nu, nx);  // nu x nx
            for (int k = params_.N - 1; k >= 0; --k) {
                Eigen::MatrixXd S = R + B.transpose() * P * B;
                K = S.ldlt().solve(B.transpose() * P * A);
                P = Q + A.transpose() * P * A - A.transpose() * P * B * K;
            }
            tau_task = -K * x_aug;

        }
        else {
            qp_->getU(0, tau_task.data());

            // Diagnostic: how hard are the soft constraints straining? A
            // large slack means the solve succeeded but is leaning heavily
            // on the penalty -- same "surface it, don't silently absorb
            // it" spirit as the Lambda_inv6 conditioning warning elsewhere.
            // Stage 1 combines both soft row types (box rows first, then
            // general/torque rows -- see the constructor's idxs_mid
            // layout), so size the buffer via nsAt() rather than assuming
            // 2*nOwned, and split accordingly.
            Eigen::VectorXd sl(qp_->nsAt(1)), su(qp_->nsAt(1));
            qp_->getSlack(1, sl.data(), su.data());
            double maxJointSlack = std::max(sl.head(2 * nOwned).maxCoeff(),
                                             su.head(2 * nOwned).maxCoeff());
            double maxTorqueSlack = std::max(sl.tail(n_joints_).maxCoeff(),
                                              su.tail(n_joints_).maxCoeff());
            if (maxJointSlack > 1e-3) {
                std::cerr << "[WARNING] ArmNMPC["<< params_.ee_site_name.c_str() << "]: "
                        << "joint-limit slack = " << maxJointSlack 
                        << " (soft constraint straining -- arm near/outside a limit)." 
                        << std::endl; 
            }
            if (maxTorqueSlack > 1e-3) {
                std::cerr << "[WARNING] ArmNMPC["<< params_.ee_site_name.c_str() << "]: "
                        << "torque slack = " << maxTorqueSlack 
                        << " (soft constraint straining -- commanded torque near/over limit)." 
                        << std::endl; 
            }
        }
    }
    checkNaN(tau_task.hasNaN(), "tau_task");
    return tau_task;
}

// === Augmented prediction model ===============================================
// Successive-linearization LTV model: task-space error state (pos/vel + a
// constant channel carrying the frozen acceleration bias) augmented with
// Δq/qdot rows so joint position/velocity limits are plain box bounds.
void ArmNMPC::buildAugmentedModel(
    const Eigen::MatrixXd& J_task, const Eigen::MatrixXd& Lambda_inv,
    const Eigen::VectorXd& bias, const Eigen::VectorXd& v_d_task,
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

    // Joint-space augmentation. qdot ≈ J_pinv * v_ee = J_pinv * (v_d_task - edot),
    // linear in edot with the v_d_task term folded into the constant channel.
    // Δq(k+1) = Δq(k) + Ts*qdot(k).
    Eigen::MatrixXd J_pinv = pinvDLS(
        J_task, params_.Hg_damping,
        params_.ee_site_name + " J_task (joint-limit augmentation)");
    Eigen::MatrixXd dq_from_edot   = Ts * J_pinv;        // n_j x d
    Eigen::MatrixXd qdot_from_edot = J_pinv;             // n_j x d
    Eigen::VectorXd qdot_bias      = J_pinv * v_d_task;  // n_j, frozen this tick

    A = Eigen::MatrixXd::Zero(nx, nx);
    B = Eigen::MatrixXd::Zero(nx, d);
    A.block(0, 0, nt, nt) = A_task;
    B.block(0, 0, nt, d)  = B_task;

    // Δq rows: [nt, nt+n_j)
    A.block(nt, nt, n_j, n_j)   = Eigen::MatrixXd::Identity(n_j, n_j);
    A.block(nt, d, n_j, d)      = -dq_from_edot;
    A.block(nt, 2 * d, n_j, 1)  = Ts * qdot_bias;
    // qdot rows: [nt+n_j, nt+2*n_j)
    A.block(nt + n_j, d, n_j, d)     = -qdot_from_edot;
    A.block(nt + n_j, 2 * d, n_j, 1) = qdot_bias;
    // B rows for Δq/qdot stay zero — control couples only through edot above.
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
            << "\n";
    }
}

// === Full-nonlinear (multiple-shooting SQP) path — NOT PORTED ==================
// Reached only when NMPCParams::fullNonlinear is true; the linearized path
// above is the supported one. Port solveNonlinearMPC/evalTaskSpace from VORTEX
// armConstrainedNMPController.cpp when the SQP path is needed.
ArmNMPC::NodeTaskSpace ArmNMPC::evalTaskSpace(
    const Eigen::VectorXd&, const Eigen::VectorXd&,
    const Eigen::Vector3d&, const Eigen::Vector4d&,
    const Eigen::MatrixXd&, const Eigen::Matrix<double, 6, 1>&) const
{
    throw std::logic_error("ArmNMPC::evalTaskSpace: fullNonlinear path not ported");
}

Eigen::VectorXd ArmNMPC::solveNonlinearMPC(
    const Eigen::VectorXd&, const Eigen::VectorXd&, int,
    const TaskSpaceEval&, Eigen::MatrixXd&, Eigen::MatrixXd&,
    const Eigen::VectorXd&, Eigen::MatrixXd&)
{
    throw std::logic_error("ArmNMPC::solveNonlinearMPC: fullNonlinear path not ported");
}

} // namespace pat_arm_nmpc