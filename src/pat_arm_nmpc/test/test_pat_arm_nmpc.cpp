/*! Model-fidelity tests for the NMPC formulation.
 *
 *  These drive the prediction-model construction directly, through a subclass
 *  that re-exports the protected seams, and compare it against MuJoCo's own
 *  forward dynamics evaluated on the same state. Nothing here touches ROS or
 *  the QP solver -- the question being asked is only "does the linearised
 *  model predict the plant it is supposed to predict".
 *
 *  PAT_NMPC_TEST_MJCF is the build-tree pat_platform_planar.xml (see
 *  CMakeLists.txt); the model has a free-floating base plus two planar 3R
 *  arms, so nq = 13, nv = 12, n_joints_ = 6.
 */
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "pat_arm_nmpc/arm_nmpc.h"

namespace pat_arm_nmpc {
namespace {

// Re-exports the protected model-construction seams so a test can build A/B at
// a known state rather than inferring them from closed-loop behaviour.
class ModelProbe : public ArmNMPC {
public:
    ModelProbe(const std::string& mjcf, const NMPCParams& p) : ArmNMPC(mjcf, p) {}

    using ArmNMPC::NodeTaskSpace;
    using ArmNMPC::buildAugmentedModel;
    using ArmNMPC::ee_site_id_;
    using ArmNMPC::enterDamping;
    using ArmNMPC::kd_;
    using ArmNMPC::torqueLimsByIndex;
    using ArmNMPC::vlimsByIndex;
    using ArmNMPC::evalTaskSpace;
    using ArmNMPC::ownedJointInds;
    using ArmNMPC::n_qpsolve_;
    using ArmNMPC::lastQpStatus;
    using ArmNMPC::mode;
    using ArmNMPC::params_;
    using ArmNMPC::qlimsByIndex;
    using ArmNMPC::qp_;
    using ArmNMPC::reachableWrenchBound;
    using ArmNMPC::t_qpsolve_us_;
    using ArmTaskSpaceController::dynamicsAt;
    using ArmTaskSpaceController::integrateStep;
    using ArmTaskSpaceController::modelJointRanges;
    using ArmTaskSpaceController::n_joints_;
    using ArmTaskSpaceController::pinvDLS;
};

constexpr const char* kMjcf = PAT_NMPC_TEST_MJCF;

// Left arm, matching nmpc.yaml's per-instance shape: this controller owns one
// arm's three joints out of the model's six.
NMPCParams leftArmParams(double Ts = 0.01)
{
    NMPCParams p;
    p.Ts             = Ts;
    p.N              = 5;
    p.ee_site_name   = "ee_site_L";
    p.fullNonlinear  = true;
    // Hg_damping deliberately left at the NMPCParams default so the
    // over-damping test below tracks that default rather than a local choice.
    p.owned_joints   = {"joint2_L", "joint3_L", "joint5_L"};
    for (const auto& name : p.owned_joints) {
        p.joint_lims.push_back({name, -M_PI, M_PI, /*qd_max=*/3.0});
    }
    // Every model joint, not just the owned ones — the controller rejects an
    // incomplete list, since the torque constraint covers all of them.
    for (const auto& name : {"joint2_L", "joint3_L", "joint5_L",
                              "joint2_R", "joint3_R", "joint5_R"}) {
        p.model_torque_lims.emplace_back(name, 0.25);
    }
    return p;
}

struct State {
    Eigen::VectorXd q, v;
};

/*! A deliberately generic state: base at the origin with identity orientation,
 *  both arms away from any singular (straight or folded) configuration, and
 *  every joint moving so the Coriolis terms are actually exercised.
 *
 *  Every angle sits strictly inside the MJCF joint ranges (joint2 [0, 3.14],
 *  joint3 [-2.967, 0], joint5 [-1.22, 1.22]). This matters: MuJoCo enforces
 *  those ranges with constraint forces, and a state outside them accelerates
 *  violently under forces the prediction model has no term for, which would
 *  make every comparison below meaningless. */
State referenceState(int n_joints)
{
    const double ja[6] = { 0.90, -1.20,  0.35,  1.10, -0.90, -0.40};
    const double jv[6] = { 0.12, -0.08,  0.05, -0.10,  0.06, -0.04};

    State s;
    s.q = Eigen::VectorXd::Zero(7 + n_joints);
    s.q(3) = 1.0;                     // base quaternion w
    s.v = Eigen::VectorXd::Zero(6 + n_joints);
    for (int i = 0; i < n_joints; ++i) {
        s.q(7 + i) = ja[i % 6];
        s.v(6 + i) = jv[i % 6];
    }
    return s;
}

// --- Dynamics reduction ---------------------------------------------------

TEST(NmpcDynamics, GeneralizedInertiaIsSymmetricPositiveDefinite)
{
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    ASSERT_EQ(nd.H_g.rows(), c.n_joints_);
    ASSERT_EQ(nd.H_g.cols(), c.n_joints_);
    EXPECT_LT((nd.H_g - nd.H_g.transpose()).cwiseAbs().maxCoeff(), 1e-9)
        << "H_g = H_mm - H_bm^T H_bb^-1 H_bm is symmetric by construction";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(nd.H_g);
    EXPECT_GT(es.eigenvalues().minCoeff(), 0.0)
        << "eigenvalues: " << es.eigenvalues().transpose();
}

TEST(NmpcDynamics, GeneralizedInertiaInverseIsNotOverDamped)
{
    // pinvDLS damps with s/(s^2 + lambda^2), so the damping only leaves the
    // inverse intact while H_g's singular values stay well above Hg_damping.
    // A planar arm's generalised inertia is small in absolute terms, so this
    // is a real risk rather than a theoretical one: over-damping here scales
    // Lambda_inv down by the same factor and the QP then models an arm far
    // more sluggish than the plant.
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(nd.H_g);
    const double min_sv = svd.singularValues()(svd.singularValues().size() - 1);

    const Eigen::MatrixXd damped = c.pinvDLS(nd.H_g, c.params_.Hg_damping, "");
    const Eigen::MatrixXd exact =
        nd.H_g.ldlt().solve(Eigen::MatrixXd::Identity(c.n_joints_, c.n_joints_));

    const double rel = (damped - exact).norm() / exact.norm();
    EXPECT_LT(rel, 0.01)
        << "damped H_g^-1 differs from the exact solve by " << (100.0 * rel)
        << "%.  Hg_damping=" << c.params_.Hg_damping
        << ", H_g singular values=" << svd.singularValues().transpose()
        << " (smallest " << min_sv << ")";
}

TEST(NmpcDynamics, GeneralizedBiasVanishesAtRest)
{
    // Gravity is disabled in the MJCF, so at zero velocity there is no
    // Coriolis/centrifugal term for the base elimination to carry through.
    ModelProbe c(kMjcf, leftArmParams());
    State s = referenceState(c.n_joints_);
    s.v.setZero();

    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    EXPECT_LT(nd.Cv_joints.cwiseAbs().maxCoeff(), 1e-9)
        << "Cv_joints = " << nd.Cv_joints.transpose();
}

TEST(NmpcDynamics, GeneralizedBiasIsNonZeroWhenMoving)
{
    // Guards the test above from passing for the wrong reason (a bias that is
    // always zero would satisfy it trivially).
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    EXPECT_GT(nd.Cv_joints.cwiseAbs().maxCoeff(), 1e-12);
}

// --- Joint limits ---------------------------------------------------------

TEST(NmpcLimits, EveryModelJointGetsItsMjcfRange)
{
    // The controller owns only the left arm, but the rollout advances all six
    // joints and MuJoCo enforces all six ranges. The right arm's bounds must
    // therefore be the model's, not a placeholder -- a rollout that walks a
    // non-owned joint past its range meets constraint forces the linearised
    // model cannot represent.
    ModelProbe c(kMjcf, leftArmParams());
    const auto ranges = c.modelJointRanges();

    ASSERT_EQ(static_cast<int>(ranges.size()), c.n_joints_);
    for (int i = 0; i < c.n_joints_; ++i) {
        EXPECT_LT(ranges[i].first, ranges[i].second) << "joint " << i;
        EXPECT_GT(ranges[i].first, -M_PI - 1e-9) << "joint " << i;
        EXPECT_LT(ranges[i].second, M_PI + 1e-9) << "joint " << i;
    }

    // joint2 [0, 3.14] / joint3 [-2.967, 0] / joint5 [-1.22, 1.22], per arm,
    // in MuJoCo body-tree order (left arm then right).
    const double expect_lo[6] = {0.0, -2.967, -1.22, 0.0, -2.967, -1.22};
    const double expect_hi[6] = {3.14,  0.0,   1.22, 3.14,  0.0,   1.22};
    for (int i = 0; i < c.n_joints_; ++i) {
        EXPECT_NEAR(ranges[i].first,  expect_lo[i], 1e-9) << "joint " << i << " lower";
        EXPECT_NEAR(ranges[i].second, expect_hi[i], 1e-9) << "joint " << i << " upper";
    }
}

TEST(NmpcLimits, TrustRegionTracksTheReachableWrench)
{
    // The trust region is a fraction of this bound, so it has to sit in the
    // same range as the wrench the arm can actually produce -- the defect the
    // old fixed 50 N / 5 N*m pair had was being orders of magnitude above it,
    // which made it unable to bind at all.
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

    const Eigen::VectorXd bound = c.reachableWrenchBound(ts.J_task);
    ASSERT_EQ(bound.size(), 3);

    // A wrench at the bound must be deliverable: applied on its own channel it
    // may not ask any joint for more than its configured limit.
    for (int j = 0; j < 3; ++j) {
        EXPECT_GT(bound(j), 0.0) << "channel " << j;
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        u(j) = bound(j);
        const Eigen::VectorXd tau = ts.J_task.transpose() * u;
        for (int i = 0; i < c.n_joints_; ++i) {
            EXPECT_LE(std::abs(tau[i]), 0.25 + 1e-9)
                << "channel " << j << " at its bound overloads joint " << i
                << " (tau " << tau[i] << ")";
        }
    }

    // And it must be tight rather than merely safe: scaling past the bound has
    // to overload something, or the region is loose enough never to bind.
    for (int j = 0; j < 3; ++j) {
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        u(j) = 1.01 * bound(j);
        const Eigen::VectorXd tau = ts.J_task.transpose() * u;
        EXPECT_GT(tau.cwiseAbs().maxCoeff(), 0.25)
            << "channel " << j << " bound " << bound(j) << " is not tight";
    }
}

// --- Rollout integrator ---------------------------------------------------

TEST(NmpcIntegrator, RestStaysAtRestWithNoAppliedTorque)
{
    // Gravity is off and the rollout applies torque only to the joint rows, so
    // a state at rest under zero torque must not move at all. Anything else
    // means a force the prediction model knows nothing about -- contact being
    // the obvious candidate, since the reference configuration is chosen by
    // hand and the internal model still carries the bus and arm geoms.
    const double Ts = 0.01;
    ModelProbe c(kMjcf, leftArmParams(Ts));
    State s = referenceState(c.n_joints_);
    s.v.setZero();

    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, Eigen::VectorXd::Zero(c.n_joints_), Ts, q1, v1);

    EXPECT_LT((v1 - s.v).cwiseAbs().maxCoeff(), 1e-9)
        << "velocity moved to " << v1.transpose();
    EXPECT_LT((q1 - s.q).cwiseAbs().maxCoeff(), 1e-9)
        << "position moved by " << (q1 - s.q).transpose();
}

TEST(NmpcIntegrator, JointPositionUpdateUsesMidIntervalVelocity)
{
    // Under a torque held constant across the step, the joint position must
    // advance by Ts*v + Ts^2/2*a, matching the Ts^2/2 coefficient the
    // augmented model's own task-space rows assume. Integrating with the
    // end-of-step velocity instead would double that term, so asserting the
    // residual is a small fraction of it discriminates between the two.
    const double Ts = 0.01;
    ModelProbe c(kMjcf, leftArmParams(Ts));
    const State s = referenceState(c.n_joints_);
    const int n_j = c.n_joints_;

    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts =
        c.evalTaskSpace(s.q, s.v, nd.ee_pos, nd.ee_quat, nd.J_g6, nd.jdot_qdot6);

    const Eigen::MatrixXd H_g_inv = c.pinvDLS(nd.H_g, c.params_.Hg_damping, "");
    const Eigen::Vector3d u(0.08, -0.05, 0.01);          // task wrench [N, N, N*m]
    const Eigen::VectorXd tau_j = ts.J_task.transpose() * u + nd.Cv_joints;

    // The Cv feedforward cancels the generalised bias exactly, leaving
    // qddot = H_g^-1 J^T u.
    const Eigen::VectorXd a = H_g_inv * ts.J_task.transpose() * u;

    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, tau_j, Ts, q1, v1);

    const Eigen::VectorXd dq_actual = q1.tail(n_j) - s.q.tail(n_j);
    const Eigen::VectorXd dq_expect = Ts * s.v.tail(n_j) + 0.5 * Ts * Ts * a;
    const double accel_term = (0.5 * Ts * Ts * a).norm();

    ASSERT_GT(accel_term, 0.0) << "test wrench produced no joint acceleration";
    EXPECT_LT((dq_actual - dq_expect).norm(), 0.1 * accel_term)
        << "residual " << (dq_actual - dq_expect).norm()
        << " vs Ts^2/2*a term " << accel_term;

    EXPECT_LT((v1.tail(n_j) - (s.v.tail(n_j) + Ts * a)).norm(), 0.05 * (Ts * a).norm())
        << "velocity update should be Ts*a to first order";
}

// --- Augmented prediction model ------------------------------------------

/*! Builds A/B at (q,v) and returns the model's one-step prediction alongside
 *  the plant's true one-step response, in the same augmented coordinates. */
struct OneStepComparison {
    Eigen::VectorXd x0, x_pred, x_true;
    int d = 3, n_j = 0;

    Eigen::VectorXd taskBlock(const Eigen::VectorXd& x) const { return x.head(2 * d); }
    Eigen::VectorXd jointBlock(const Eigen::VectorXd& x) const
    {
        return x.segment(2 * d + 1, 2 * n_j);
    }
};

OneStepComparison compareOneStep(ModelProbe& c, const Eigen::Vector3d& u)
{
    const int n_j   = c.n_joints_;
    const int d     = 3;
    const double Ts = c.params_.Ts;
    const State s   = referenceState(n_j);

    const NodeDynamics nd0 = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts0 =
        c.evalTaskSpace(s.q, s.v, nd0.ee_pos, nd0.ee_quat, nd0.J_g6, nd0.jdot_qdot6);

    const Eigen::MatrixXd H_g_inv = c.pinvDLS(nd0.H_g, c.params_.Hg_damping, "");
    const Eigen::MatrixXd H_g_inv_Jt = H_g_inv * ts0.J_task.transpose();
    const Eigen::MatrixXd Lambda_inv = ts0.J_task * H_g_inv_Jt;

    Eigen::MatrixXd A, B;
    c.buildAugmentedModel(Lambda_inv, ts0.bias, H_g_inv_Jt, A, B);

    OneStepComparison out;
    out.d   = d;
    out.n_j = n_j;

    const int nx = 2 * d + 1 + 2 * n_j;
    out.x0 = Eigen::VectorXd::Zero(nx);
    out.x0.segment(0, d)             = ts0.e;
    out.x0.segment(d, d)             = ts0.edot;
    out.x0(2 * d)                    = 1.0;
    out.x0.segment(2 * d + 1, n_j).setZero();       // Δq measured from q0
    out.x0.segment(2 * d + 1 + n_j, n_j) = s.v.tail(n_j);

    out.x_pred = A * out.x0 + B * u;

    // True plant response to the same wrench, deliberately unclamped so the
    // comparison stays inside the linear regime the model describes.
    const Eigen::VectorXd tau_j = ts0.J_task.transpose() * u + nd0.Cv_joints;
    Eigen::VectorXd q1, v1;
    c.integrateStep(s.q, s.v, tau_j, Ts, q1, v1);

    const NodeDynamics nd1 = c.dynamicsAt(q1, v1, c.ee_site_id_);
    const ModelProbe::NodeTaskSpace ts1 =
        c.evalTaskSpace(q1, v1, nd1.ee_pos, nd1.ee_quat, nd1.J_g6, nd1.jdot_qdot6);

    out.x_true = Eigen::VectorXd::Zero(nx);
    out.x_true.segment(0, d)             = ts1.e;
    out.x_true.segment(d, d)             = ts1.edot;
    out.x_true(2 * d)                    = 1.0;
    out.x_true.segment(2 * d + 1, n_j)   = q1.tail(n_j) - s.q.tail(n_j);
    out.x_true.segment(2 * d + 1 + n_j, n_j) = v1.tail(n_j);

    return out;
}

TEST(NmpcModel, TaskRowsPredictPlantOneStep)
{
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison r = compareOneStep(c, Eigen::Vector3d(0.08, -0.05, 0.01));

    const double err    = (r.taskBlock(r.x_pred) - r.taskBlock(r.x_true)).norm();
    const double change = (r.taskBlock(r.x_true) - r.taskBlock(r.x0)).norm();

    ASSERT_GT(change, 1e-9) << "the step produced no task-space motion to compare";
    EXPECT_LT(err, 0.05 * change)
        << "task rows captured " << (100.0 * (1.0 - err / change))
        << "% of the true one-step change (err " << err << ", change " << change << ")";
}

TEST(NmpcModel, JointRowsPredictPlantOneStep)
{
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison r = compareOneStep(c, Eigen::Vector3d(0.08, -0.05, 0.01));

    const double err    = (r.jointBlock(r.x_pred) - r.jointBlock(r.x_true)).norm();
    const double change = (r.jointBlock(r.x_true) - r.jointBlock(r.x0)).norm();

    ASSERT_GT(change, 1e-9) << "the step produced no joint motion to compare";
    EXPECT_LT(err, 0.05 * change)
        << "joint rows captured " << (100.0 * (1.0 - err / change))
        << "% of the true one-step change (err " << err << ", change " << change << ")";
}

TEST(NmpcModel, JointRowsRespondToControl)
{
    // The joint Δq/qdot rows carry the box constraints, so they have to move
    // when the control moves -- a model whose B block is zero on those rows
    // predicts the same joint trajectory whatever torque the QP chooses, and
    // the joint limits stop constraining anything the solver can act on.
    ModelProbe c(kMjcf, leftArmParams());
    const OneStepComparison a = compareOneStep(c, Eigen::Vector3d( 0.08, -0.05,  0.01));
    const OneStepComparison b = compareOneStep(c, Eigen::Vector3d(-0.08,  0.05, -0.01));

    const double pred_spread = (a.jointBlock(a.x_pred) - b.jointBlock(b.x_pred)).norm();
    const double true_spread = (a.jointBlock(a.x_true) - b.jointBlock(b.x_true)).norm();

    ASSERT_GT(true_spread, 1e-9) << "opposing wrenches produced no joint-motion difference";
    EXPECT_GT(pred_spread, 0.5 * true_spread)
        << "model predicts a " << pred_spread << " joint-row spread between opposing "
        << "wrenches where the plant gives " << true_spread;
}

// --- QP constraint contract ----------------------------------------------
//
// A double integrator, small enough that the answer is checkable by hand:
// drive position from 1 to 0 with the terminal velocity pinned. The point is
// not the trajectory but the two HARD constraints -- the stage-0 state pin and
// the terminal velocity row -- both of which reach HPIPM as box bounds whose
// lower and upper halves are equal. Anything that changes how those are
// declared to the solver has to keep both exactly satisfied.

struct DoubleIntegratorQP {
    quad_prob_solver::QuadProbSolverParams params;
    std::vector<Eigen::MatrixXd> A_k, B_k, D_k;
    std::vector<Eigen::VectorXd> b_k, lg_k, ug_k;
    Eigen::MatrixXd Q, R, Q_N;
    Eigen::VectorXd x0, lbx, ubx;
};

/*! State is [position, velocity, dummy]. The dummy is decoupled and exists only
 *  to push the index arrays out of alignment: with idxbx_k = {0, 2}, stage N's
 *  box list is [0, 2, 1], so the terminal velocity row sits at box POSITION 2
 *  while naming STATE 1. HPIPM's idxbxe is documented nowhere in the installed
 *  headers, so that misalignment is what tells the two candidate conventions
 *  apart: declare the equality by box position and the terminal velocity is
 *  pinned; declare it by state index and row 2 (the dummy's soft box) gets
 *  eliminated instead, leaving velocity free. */
DoubleIntegratorQP makeDoubleIntegratorQP(bool pin_terminal_velocity)
{
    const double Ts = 0.1;
    const int N = 6;

    DoubleIntegratorQP p;
    p.params.nx_task = 3;
    p.params.nx      = 3;
    p.params.nu      = 1;
    p.params.N       = N;
    // One general row bounding the control, standing in for the real
    // controller's torque constraint. Without it `u` is unbounded and ANY box is
    // reachable in a single step, which would make every hard-box test vacuous.
    p.params.ng      = 1;
    p.params.qp_max_iter = 200;
    p.params.qp_tol      = 1e-9;
    p.params.idxbx_k = {0, 2};
    if (pin_terminal_velocity) p.params.idxbx_terminal = {1};

    p.A_k.assign(N, Eigen::MatrixXd::Identity(3, 3));
    p.B_k.assign(N, Eigen::MatrixXd::Zero(3, 1));
    p.b_k.assign(N, Eigen::VectorXd::Zero(3));
    for (int k = 0; k < N; ++k) {
        p.A_k[k](0, 1) = Ts;
        p.B_k[k](0, 0) = 0.5 * Ts * Ts;
        p.B_k[k](1, 0) = Ts;
    }
    // |u| <= 10: enough authority to track, far too little to jump a box in one
    // step, which is what makes an out-of-box x0 genuinely infeasible.
    p.D_k.assign(N, Eigen::MatrixXd::Constant(1, 1, 1.0));
    p.lg_k.assign(N, Eigen::VectorXd::Constant(1, -10.0));
    p.ug_k.assign(N, Eigen::VectorXd::Constant(1,  10.0));

    p.Q = Eigen::MatrixXd::Zero(3, 3);
    p.Q(0, 0) = 1.0;
    p.R = Eigen::MatrixXd::Constant(1, 1, 1e-4);   // cheap control: drives hard, still moving at N
    p.Q_N = p.Q * 10.0;

    p.x0 = Eigen::VectorXd(3);
    p.x0 << 1.0, 0.0, 0.0;
    p.lbx = Eigen::VectorXd(2);   // one per idxbx_k row, wide enough not to bind
    p.ubx = Eigen::VectorXd(2);
    p.lbx << -10.0, -10.0;
    p.ubx <<  10.0,  10.0;
    return p;
}

bool solveQP(quad_prob_solver::QuadProbSolver& qp, DoubleIntegratorQP& p)
{
    return qp.solveMultiStage(
        p.A_k, p.B_k, p.b_k, p.Q.data(), p.R.data(),
        p.lbx.data(), p.ubx.data(),
        p.D_k, p.lg_k, p.ug_k, p.x0.data(), p.Q_N.data());
}

TEST(NmpcQP, TerminalVelocityIsFreeWithoutTheConstraint)
{
    // Establishes that the pin test below is meaningful: without the terminal
    // row, the optimum really does arrive at stage N still moving, so a passing
    // pin cannot be an accident of the cost.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/false);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p));

    Eigen::VectorXd x_at_N(3);
    qp.getX(p.params.N, x_at_N.data());
    EXPECT_GT(std::abs(x_at_N(1)), 1e-2)
        << "unpinned terminal velocity is " << x_at_N(1)
        << ", too close to zero for the pin test to prove anything";
}

TEST(NmpcQP, HardBoundsAreSatisfiedExactly)
{
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/true);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "double-integrator QP failed to solve";

    // Stage-0 pin: the solver's own stage-0 state must be the state handed in.
    Eigen::VectorXd x_at_0(3);
    qp.getX(0, x_at_0.data());
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(x_at_0(i), p.x0(i), 1e-8) << "stage-0 pin, row " << i;
    }

    // Terminal velocity row: hard, so it holds to solver tolerance. This is the
    // assertion that distinguishes the two idxbxe index conventions.
    Eigen::VectorXd x_at_N(3);
    qp.getX(p.params.N, x_at_N.data());
    EXPECT_NEAR(x_at_N(1), 0.0, 1e-6) << "terminal velocity should be pinned to zero";

    // And the solve is actually doing something: position is driven toward 0
    // from 1, so the first control must push back (negative).
    Eigen::VectorXd u0(1);
    qp.getU(0, u0.data());
    EXPECT_LT(u0(0), 0.0) << "u0 = " << u0(0);

    // The dummy row is decoupled (A = I, B = 0) and starts at zero, so it must
    // stay at zero at every stage. This is what actually catches a mis-declared
    // equality: declaring the wrong box row as an equality does NOT unpin the
    // terminal velocity — lb == ub still enforces that on its own — it instead
    // tells HPIPM that some OTHER row's genuine inequality is an equality, and
    // that row then gets collapsed onto a bound.
    for (int k = 0; k <= p.params.N; ++k) {
        Eigen::VectorXd xk(3);
        qp.getX(k, xk.data());
        EXPECT_NEAR(xk(2), 0.0, 1e-6)
            << "decoupled row moved at stage " << k << " (to " << xk(2)
            << ") — a box row that is not an equality has been eliminated as one";
    }
}

TEST(NmpcQP, SoftLayoutMatchesWhatTheSolverWasBuiltWith)
{
    // nsAt/nsbxAt/nsgAt must describe the wiring for every flag combination --
    // this is what lets a slack reader slice by asking instead of assuming, and
    // in the real model the box and general counts are equal, so a wrong split
    // would go unnoticed.
    for (bool soft_box : {true, false}) {
        for (bool soft_tau : {true, false}) {
            DoubleIntegratorQP p = makeDoubleIntegratorQP(true);
            p.params.soft_joint_limits  = soft_box;
            p.params.soft_torque_limits = soft_tau;
            quad_prob_solver::QuadProbSolver qp(p.params);

            const int nbx_k = static_cast<int>(p.params.idxbx_k.size());
            const int N     = p.params.N;
            for (int k = 0; k <= N; ++k) {
                const int want_box = (soft_box && k > 0) ? nbx_k : 0;
                const int want_gen = (soft_tau && k < N) ? p.params.ng : 0;
                EXPECT_EQ(qp.nsbxAt(k), want_box)
                    << "stage " << k << " box, soft_box=" << soft_box;
                EXPECT_EQ(qp.nsgAt(k), want_gen)
                    << "stage " << k << " general, soft_tau=" << soft_tau;
                EXPECT_EQ(qp.nsAt(k), want_box + want_gen) << "stage " << k;
            }
        }
    }
}

TEST(NmpcQP, HardBoxGoesInfeasibleWhenX0IsOutsideIt)
{
    // The mechanism behind "hard constraints failed at every loop": the box
    // constrains stages 1..N, the stage-0 pin fixes where the trajectory starts,
    // and bounded control cannot travel far in one step. Put x0 outside the box
    // and there is no feasible trajectory at all.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits  = false;
    p.params.soft_torque_limits = false;   // else the plan buys its way out, below
    p.x0 << 50.0, 0.0, 0.0;                // position box is [-10, 10]

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_FALSE(solveQP(qp, p))
        << "hard box with x0 outside it should not solve; status "
        << qp.lastStatus();
}

TEST(NmpcQP, SoftBoxToleratesX0OutsideIt)
{
    // The same problem stays solvable when the box is soft -- the contrast that
    // makes the hard-vs-soft comparison meaningful at all.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits = true;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();
}

TEST(NmpcQP, HardBoxWithSoftTorqueFailsWithANaNSolution)
{
    // The mixed configuration is the worst of the three, and worth pinning. A
    // hard box the state cannot reach, combined with a soft control bound the
    // solver can buy its way through, does not produce a clean infeasibility --
    // it produces status 3, a NaN solution. Consistent with the HPIPM behaviour
    // recorded on QuadProbSolverParams::ng, where soft general rows are the
    // fragile ones. The caller's allFinite() checks turn this into a failed
    // solve, so it reaches safe mode rather than the actuators, but anyone
    // choosing to harden only the box should know this is the failure they get.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits  = false;
    p.params.soft_torque_limits = true;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    EXPECT_FALSE(solveQP(qp, p));
    EXPECT_EQ(qp.lastStatus(), 3) << "expected NAN_SOL (3), got " << qp.lastStatus();
}

TEST(NmpcQP, GuardMakesAHardBoxSolvableAndReportsTheRelaxation)
{
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits         = false;
    p.params.relax_box_to_contain_x0   = true;
    p.params.relax_box_margin          = 1e-2;
    p.x0 << 50.0, 0.0, 0.0;

    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();

    // The pin is still honoured exactly — the guard widens the box, it does not
    // move the trajectory's start.
    Eigen::VectorXd x_at_0(3);
    qp.getX(0, x_at_0.data());
    EXPECT_NEAR(x_at_0(0), 50.0, 1e-8);

    // And it reports what it had to give, so a study can tell a guarded hard box
    // from no box at all.
    EXPECT_GT(qp.lastRelaxCount(), 0);
    EXPECT_GT(qp.lastRelaxMax(), 0.0);
}

TEST(NmpcQP, GuardIsInertWhenX0IsAlreadyInsideTheBox)
{
    // A ratchet, not a blanket relaxation: with the state inside its bounds the
    // guard must change nothing, or "hard + guard" would quietly mean "no box".
    DoubleIntegratorQP p = makeDoubleIntegratorQP(false);
    p.params.soft_joint_limits       = false;
    p.params.relax_box_to_contain_x0 = true;
    p.params.relax_box_margin        = 1e-3;
    p.x0 << 1.0, 0.0, 0.0;           // well inside [-10, 10]

    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p)) << "status " << qp.lastStatus();
    EXPECT_EQ(qp.lastRelaxCount(), 0);
    EXPECT_DOUBLE_EQ(qp.lastRelaxMax(), 0.0);
}

TEST(NmpcQP, DynamicsAreConsistentAcrossStages)
{
    // Guards the pin test above: a solver that satisfied the bounds but
    // ignored the dynamics would still pass it.
    DoubleIntegratorQP p = makeDoubleIntegratorQP(/*pin_terminal_velocity=*/true);
    quad_prob_solver::QuadProbSolver qp(p.params);
    ASSERT_TRUE(solveQP(qp, p));

    for (int k = 0; k < p.params.N; ++k) {
        Eigen::VectorXd xk(3), xk1(3), uk(1);
        qp.getX(k, xk.data());
        qp.getX(k + 1, xk1.data());
        qp.getU(k, uk.data());
        const Eigen::VectorXd pred = p.A_k[k] * xk + p.B_k[k] * uk + p.b_k[k];
        EXPECT_LT((xk1 - pred).cwiseAbs().maxCoeff(), 1e-7)
            << "stage " << k << " violates its own dynamics";
    }
}

// --- Soft-constraint slack -----------------------------------------------

NMPCParams liveParams()
{
    // The shape nmpc.yaml actually runs: SQP path, N=40, Ts=0.025.
    NMPCParams p = leftArmParams(0.025);
    p.N = 40;
    p.sqp_iters = 1;
    p.fullNonlinear = true;
    p.terminal_velocity_constraint = true;
    p.terminal_cost_multiplier = 100.0;
    p.Qx = 10.0;   p.Qy = 10.0;   p.Qwz = 1.0;
    p.Qdotx = 1.0; p.Qdoty = 1.0; p.Qdotwz = 3.0;
    p.Rx = 1e-1;   p.Ry = 1e-1;   p.Rwz = 1e-1;
    p.qp_max_iter = 100;
    p.qp_tol = 1e-5;
    for (auto& jl : p.joint_lims) { jl.qd_max = 2.0; }
    p.joint_lims[0].q_min =  0.0;   p.joint_lims[0].q_max = 3.14;
    p.joint_lims[1].q_min = -2.967; p.joint_lims[1].q_max = 0.0;
    p.joint_lims[2].q_min = -1.22;  p.joint_lims[2].q_max = 1.22;
    return p;
}

/*! Runs the real controller closed-loop against MuJoCo and reports the worst
 *  slack seen, so a test can distinguish "the soft constraints are idle" from
 *  "they are absorbing a violation". */
/*! Everything a soft-vs-hard comparison needs from one closed-loop run. The
 *  headline pair is availability and max_consecutive_damping: "failed at every
 *  loop" is a max-consecutive claim, and a mean alone cannot tell 2% scattered
 *  dropouts from one long outage. */
struct RunMetrics {
    double availability = 0.0;          // fraction of ticks that produced NMPC torque
    int    max_consecutive_damping = 0;
    // CAUTION: max_pos_violation is a running max over a run that BEGINS at the
    // violation, so it is pinned to the initial condition and reads ~the starting
    // offset for any controller, including a perfect one. Use initial/final/
    // recovered to ask whether the joint actually came back.
    double max_pos_violation = 0.0;     // [rad]   worst excursion past the config limits
    double pos_violation_integral = 0.0;// [rad s] distinguishes a transient from parking outside
    double initial_pos_violation = 0.0; // [rad]   at the first tick
    double final_pos_violation = 0.0;   // [rad]   at the last tick
    double recovered = 0.0;             // [rad]   initial - final; >0 means it came back
    double max_vel_violation = 0.0;     // [rad/s]
    double mean_ee_err = 0.0;           // [m]
    double max_box_slack = 0.0;
    double guard_relax_frac = 0.0;      // fraction of ticks the guard widened anything
    double guard_relax_max = 0.0;
    double mean_tick_us = 0.0;
    double max_tick_us = 0.0;
    double frac_over_budget = 0.0;      // ticks exceeding one control period
    std::map<int, int> status_hist;     // HPIPM status -> count
};

struct Scenario {
    double setpoint_offset = 0.06;      // [m] per axis, from the measured start pose
    double q_min_override  = 0.0;       // >0 raises joint_lims[0].q_min, parking it outside
};

RunMetrics runClosedLoop(NMPCParams params, const Scenario& sc, int ticks)
{
    const double dt = 1.0 / 200.0;      // control_hz
    if (sc.q_min_override > 0.0) params.joint_lims[0].q_min = sc.q_min_override;

    ModelProbe c(kMjcf, params);
    State s = referenceState(c.n_joints_);

    const ArmNMPC::EePose start = c.currentEePose(s.q, s.v);
    const double tx = start.pos.x() + sc.setpoint_offset;
    const double ty = start.pos.y() + sc.setpoint_offset;
    c.setDesiredPos(tx, ty, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    RunMetrics m;
    long damping_ticks = 0, relax_ticks = 0;
    int  run_damping = 0;
    double err_sum = 0.0, t_sum = 0.0;

    for (int t = 0; t < ticks; ++t) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        const double tick_us =
            std::chrono::duration<double, std::micro>(
                std::chrono::high_resolution_clock::now() - t0).count();

        t_sum += tick_us;
        m.max_tick_us = std::max(m.max_tick_us, tick_us);
        if (tick_us > dt * 1e6) m.frac_over_budget += 1.0;
        ++m.status_hist[c.lastQpStatus()];

        if (c.mode() == ArmNMPC::ControlMode::Damping) {
            ++damping_ticks;
            m.max_consecutive_damping = std::max(m.max_consecutive_damping, ++run_damping);
        } else {
            run_damping = 0;
        }

        if (c.qp_->lastRelaxCount() > 0) {
            ++relax_ticks;
            m.guard_relax_max = std::max(m.guard_relax_max, c.qp_->lastRelaxMax());
        }

        const int n_box = c.qp_->nsbxAt(1);
        if (n_box > 0) {
            Eigen::VectorXd sl(c.qp_->nsAt(1)), su(c.qp_->nsAt(1));
            c.qp_->getSlack(1, sl.data(), su.data());
            m.max_box_slack = std::max({m.max_box_slack,
                                         sl.head(n_box).maxCoeff(),
                                         su.head(n_box).maxCoeff()});
        }

        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;

        // Realised violation of the trajectory actually flown, against the
        // configured limits -- the safety payoff a hard constraint is meant to buy.
        double worst_pos = 0.0;
        for (int idx : c.ownedJointInds) {
            const double q = s.q[7 + idx];
            worst_pos = std::max({worst_pos,
                                   c.qlimsByIndex[idx].first - q,
                                   q - c.qlimsByIndex[idx].second});
            m.max_vel_violation = std::max(
                m.max_vel_violation, std::abs(s.v[6 + idx]) - c.vlimsByIndex[idx]);
        }
        worst_pos = std::max(0.0, worst_pos);
        m.max_pos_violation = std::max(m.max_pos_violation, worst_pos);
        m.pos_violation_integral += worst_pos * dt;
        if (t == 0) m.initial_pos_violation = worst_pos;
        m.final_pos_violation = worst_pos;

        const ArmNMPC::EePose now = c.currentEePose(s.q, s.v);
        err_sum += std::hypot(now.pos.x() - tx, now.pos.y() - ty);
    }

    m.recovered         = m.initial_pos_violation - m.final_pos_violation;
    m.availability      = 1.0 - static_cast<double>(damping_ticks) / ticks;
    m.guard_relax_frac  = static_cast<double>(relax_ticks) / ticks;
    m.mean_ee_err       = err_sum / ticks;
    m.mean_tick_us      = t_sum / ticks;
    m.frac_over_budget /= ticks;
    m.max_vel_violation = std::max(0.0, m.max_vel_violation);
    return m;
}

// Thin wrapper keeping the two soft-contract tests below expressed as they were.
double maxBoxSlackOverRun(const NMPCParams& params, int ticks)
{
    return runClosedLoop(params, Scenario{}, ticks).max_box_slack;
}

TEST(NmpcSlack, IdleWhenInsideTheLimits)
{
    // Nominal tracking must not lean on the soft constraints at all: any real
    // slack here would mean the penalty, not the cost, is shaping the solution.
    EXPECT_LT(maxBoxSlackOverRun(liveParams(), 200), 1e-6);
}

TEST(NmpcSlack, AbsorbsAViolationInsteadOfFailing)
{
    // Arm starts outside its own SOFTWARE limit (joint2_L is at 0.90; q_min is
    // raised to 1.60) while staying inside the MJCF range. That is the case the
    // soft box exists for: the QP must stay feasible and take slack, rather than
    // failing and dropping through to the fallback chain.
    NMPCParams p = liveParams();
    p.joint_lims[0].q_min = 1.60;
    EXPECT_GT(maxBoxSlackOverRun(p, 200), 1e-9);
}

// --- Safe mode ------------------------------------------------------------

/*! liveParams() with the QP crippled so it cannot converge. iter_max = 1 is the
 *  load-bearing part: the forced cold-start retry inside QuadProbSolver runs with
 *  the same cap, so it fails too and solveMultiStage genuinely returns false.
 *  This is an ordinary config path, so the fallback is reachable in a test
 *  without any production seam. */
NMPCParams failingParams()
{
    NMPCParams p = liveParams();
    p.qp_max_iter = 1;
    p.qp_tol      = 1e-14;
    return p;
}

TEST(SafeMode, DampingTorqueIsDissipative)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);

    // A spread of velocities, including sign flips, so this is not passing off
    // one lucky direction.
    for (double scale : {0.1, 1.0, -1.0, 3.7}) {
        Eigen::VectorXd v = s.v * scale;
        const Eigen::VectorXd tau = c.dampingTorque(v);
        EXPECT_LE(tau.dot(v.tail(c.n_joints_)), 0.0) << "scale " << scale;
    }

    // Owned joints only: the node publishes no effort for the rest, and writing
    // them would be a silent claim on the other arm.
    Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int i = 0; i < c.n_joints_; ++i) {
        const bool owned = std::find(c.ownedJointInds.begin(), c.ownedJointInds.end(), i)
                           != c.ownedJointInds.end();
        if (!owned) EXPECT_DOUBLE_EQ(tau[i], 0.0) << "non-owned joint " << i;
    }
    EXPECT_GT(tau.cwiseAbs().maxCoeff(), 0.0) << "owned joints produced no damping";
}

TEST(SafeMode, DampingTorqueIsFiniteWithNonFiniteVelocity)
{
    // Safe mode is most often reached BECAUSE the state went non-finite, so
    // -kd*v must not hand NaN to an actuator that will latch it forever.
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    const int owned0 = c.ownedJointInds.front();

    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Eigen::VectorXd v = s.v;
        v[6 + owned0] = bad;
        const Eigen::VectorXd tau = c.dampingTorque(v);
        EXPECT_TRUE(tau.allFinite()) << "tau = " << tau.transpose();
        EXPECT_DOUBLE_EQ(tau[owned0], 0.0) << "poisoned joint should get zero";
    }
}

TEST(SafeMode, DampingRespectsTorqueLimits)
{
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    s.v.tail(c.n_joints_).setConstant(50.0);   // far past any rated speed

    const Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int idx : c.ownedJointInds) {
        EXPECT_LE(std::abs(tau[idx]), c.torqueLimsByIndex[idx] + 1e-12)
            << "joint " << idx;
    }
}

TEST(SafeMode, DampingIsNonSaturatingWithinRatedSpeed)
{
    // kd = frac * tau_max / qd_max, so at exactly qd_max the command should sit
    // at frac*tau_max — i.e. safe mode never needs the output clamp to stay
    // physical. That is the property that makes the gain derivable at all.
    NMPCParams p = liveParams();
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    for (int idx : c.ownedJointInds) s.v[6 + idx] = c.vlimsByIndex[idx];

    const Eigen::VectorXd tau = c.dampingTorque(s.v);
    for (int idx : c.ownedJointInds) {
        EXPECT_NEAR(std::abs(tau[idx]),
                    p.safe_mode_damping_frac * c.torqueLimsByIndex[idx], 1e-9)
            << "joint " << idx;
    }
}

TEST(SafeMode, DampingIsStableInDiscreteTime)
{
    // Explicit velocity damping diverges if kd*dt exceeds the joint's inertia.
    // Nothing at runtime checks this, so pin it: it is the one way a raised
    // damping_frac could make safe mode actively unsafe.
    const double dt = 1.0 / 200.0;          // control_hz in nmpc.yaml
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);

    for (int idx : c.ownedJointInds) {
        EXPECT_LT(c.kd_[idx] * dt, nd.H_g(idx, idx))
            << "joint " << idx << ": kd=" << c.kd_[idx]
            << " dt=" << dt << " H_g=" << nd.H_g(idx, idx)
            << " — raise control_hz or lower safe_mode_damping_frac";
    }
}

TEST(SafeMode, QpFailureEntersDampingAndCommandsIt)
{
    ModelProbe c(kMjcf, failingParams());
    const State s = referenceState(c.n_joints_);

    const Eigen::VectorXd tau = c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_TRUE(tau.isApprox(c.dampingTorque(s.v)))
        << "tau = " << tau.transpose();
    EXPECT_GT(c.safeModeTicks(), 0);
}

TEST(SafeMode, StaysDampingUntilRecoveryTicks)
{
    // The mode must not clear on the first good solve, or a marginal solver
    // would chatter the output between damping and full tracking.
    NMPCParams p = liveParams();
    p.safe_mode_recovery_ticks = 5;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);

    c.enterDamping();
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);

    for (int t = 1; t < p.safe_mode_recovery_ticks; ++t) {
        c.computeControl(s.q, s.v);
        EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping)
            << "left safe mode after only " << t << " good solves";
    }
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "never recovered after " << p.safe_mode_recovery_ticks << " good solves";
}

TEST(SafeMode, NonFiniteStateEntersDampingWithoutThrowing)
{
    // checkNaN used to throw here and take the process with it.
    ModelProbe c(kMjcf, liveParams());
    State s = referenceState(c.n_joints_);
    s.v[6 + c.ownedJointInds.front()] = std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd tau;
    EXPECT_NO_THROW(tau = c.computeControl(s.q, s.v));
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_TRUE(tau.allFinite()) << "tau = " << tau.transpose();
}

TEST(SafeMode, DampingBringsTheArmToRest)
{
    // End to end: with the QP unable to solve, closed-loop damping must actually
    // dissipate the joint velocity rather than merely being signed correctly.
    const double dt = 1.0 / 200.0;
    ModelProbe c(kMjcf, failingParams());
    State s = referenceState(c.n_joints_);

    auto ownedSpeed = [&c](const Eigen::VectorXd& v) {
        double sq = 0.0;
        for (int idx : c.ownedJointInds) sq += v[6 + idx] * v[6 + idx];
        return std::sqrt(sq);
    };

    const double v0 = ownedSpeed(s.v);
    double v_mid = 0.0;
    for (int t = 0; t < 1200; ++t) {          // 6 s at 200 Hz
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;
        if (t == 599) v_mid = ownedSpeed(s.v);
    }
    const double v_end = ownedSpeed(s.v);

    // Still decaying at the end rather than settling on a floor: the slowest
    // joint's time constant is H_ii/kd_i, so an absolute target would encode
    // this arm's inertia rather than the property being tested.
    EXPECT_LT(v_mid, 0.25 * v0) << "v0=" << v0 << " v_mid=" << v_mid;
    EXPECT_LT(v_end, 0.5 * v_mid)
        << "decay stalled: v_mid=" << v_mid << " v_end=" << v_end;
}

TEST(SafeMode, RejectsInvalidParams)
{
    {   // sqp_iters = 0 skipped the rollout loop and left J_task0 at 0x0
        NMPCParams p = liveParams();
        p.sqp_iters = 0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
    {
        NMPCParams p = liveParams();
        p.safe_mode_damping_frac = 0.0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
    {
        NMPCParams p = liveParams();
        p.safe_mode_recovery_ticks = 0;
        EXPECT_THROW(ModelProbe(kMjcf, p), std::runtime_error);
    }
}

// --- Hard vs soft constraints ---------------------------------------------

TEST(HardConstraints, RolloutClampDoesNotSatisfyTheBoxForFree)
{
    // The rollout clamps its predicted joint positions to keep the open-loop
    // trajectory out of a kinematic singularity. It must clamp to the MECHANICAL
    // range, not to the configured software margin: the box is written on
    // Δq = q_k − q0, so clamping into the margin would put the nominal trajectory
    // inside the box by construction, and the multiple-shooting defect b_k would
    // hand the QP that teleport as free dynamics. The constraint could then never
    // bind, soft or hard.
    //
    // joint2_L sits at 0.90 with q_min raised to 1.60, so the box is [0.70, 2.24]
    // and Δq is pinned to 0 at stage 0. Bounded torque moves Δq by ~0.017 in one
    // 25 ms step, so the plan must stay near that, NOT jump to the box edge.
    NMPCParams p = liveParams();
    p.joint_lims[0].q_min = 1.60;
    p.soft_joint_limits   = true;      // soft, so this stays feasible and observable
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    const ArmNMPC::EePose st = c.currentEePose(s.q, s.v);
    c.setDesiredPos(st.pos.x() + 0.06, st.pos.y() + 0.06, st.pos.z());
    c.setDesiredOrient(st.quat(0), st.quat(1), st.quat(2), st.quat(3));

    c.computeControl(s.q, s.v);
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal);

    const int idx   = c.ownedJointInds[0];
    const double lo = c.qlimsByIndex[idx].first - s.q[7 + idx];   // 0.70
    Eigen::VectorXd x1(7 + 2 * c.n_joints_);
    c.qp_->getX(1, x1.data());

    EXPECT_LT(x1[7 + idx], 0.25 * lo)
        << "stage 1 planned Δq = " << x1[7 + idx] << ", suspiciously close to the "
        << "box edge " << lo << " that bounded torque cannot reach in one step";
}

TEST(HardConstraints, SoftBoxCarriesRealSlackWhenTheArmIsOutsideItsLimit)
{
    // The other half: with the box genuinely unreachable, a SOFT box must show
    // slack of the order of the violation. Slack near zero here would mean the
    // constraint is being satisfied by something other than the control.
    NMPCParams p = liveParams();
    p.soft_joint_limits = true;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_GT(m.max_box_slack, 1e-3)
        << "box slack " << m.max_box_slack << " is too small for an arm parked "
        << "0.7 rad outside its limit";
    // A soft box does NOT come free here: straining it this hard costs both time
    // and some availability (measured ~0.7-0.8, with a handful of NaN solves),
    // because soft general rows are the fragile ones in HPIPM. Asserted loosely,
    // as the point is that soft degrades gracefully where raw hard does not.
    EXPECT_GT(m.availability, 0.5)
        << "availability " << m.availability << ", longest damping run "
        << m.max_consecutive_damping;
}

TEST(HardConstraints, RawHardBoxFailsWhenTheArmStartsOutsideItsLimit)
{
    // The historical failure, now honest and reproducible: the box cannot contain
    // the measured state, one step cannot reach it, and with no guard the QP has
    // no answer at all. Encoded deliberately so it regresses loudly if the
    // rollout clamp ever starts papering over it again.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = false;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_LT(m.availability, 0.5)
        << "expected mostly safe mode, got availability " << m.availability;
    EXPECT_GT(m.max_consecutive_damping, 50)
        << "expected a sustained outage, longest run " << m.max_consecutive_damping;
}

TEST(HardConstraints, GuardKeepsAHardBoxFeasibleFromOutsideTheLimit)
{
    // Same scenario, guard on: the box is widened just enough to contain the
    // measurement, so the solve stays feasible throughout.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = true;
    const RunMetrics m = runClosedLoop(p, Scenario{0.06, 1.60}, 200);

    EXPECT_GT(m.availability, 0.99)
        << "availability " << m.availability
        << ", longest damping run " << m.max_consecutive_damping;
    EXPECT_GT(m.guard_relax_frac, 0.0) << "the guard should have been needed here";
}

TEST(HardConstraints, GuardMarginIsFlooredAtOneStepOfDrift)
{
    // A joint at rated speed moves Ts*qd_max in one step whatever the control
    // does, so a guard margin below that re-creates the infeasibility the guard
    // exists to remove. ArmNMPC raises the configured value to that floor — and
    // the raise was silently overwritten once, which nothing could observe.
    NMPCParams p = liveParams();
    p.relax_box_to_contain_x0 = true;
    p.relax_box_margin        = 1e-9;      // far below the floor
    ModelProbe c(kMjcf, p);

    double qd_max_owned = 0.0;
    for (int idx : c.ownedJointInds)
        qd_max_owned = std::max(qd_max_owned, c.vlimsByIndex[idx]);
    const double floor = p.Ts * qd_max_owned;

    ASSERT_GT(floor, p.relax_box_margin) << "test is vacuous if the floor is lower";
    EXPECT_NEAR(c.qp_->relaxMargin(), floor, 1e-12)
        << "configured " << p.relax_box_margin << " should have been raised to "
        << floor;
}

TEST(HardConstraints, GuardMarginHonoursALargerConfiguredValue)
{
    // The floor raises, never lowers: a deliberately generous margin must survive.
    NMPCParams p = liveParams();
    p.relax_box_to_contain_x0 = true;
    p.relax_box_margin        = 0.5;
    ModelProbe c(kMjcf, p);
    EXPECT_NEAR(c.qp_->relaxMargin(), 0.5, 1e-12);
}

TEST(HardConstraints, GuardIsNotNeededWhenTheArmIsInsideItsLimits)
{
    // The guard must not be silently load-bearing in normal operation: if it
    // fired every tick here, a "hard" box would really be no box at all and any
    // comparison against soft would be measuring nothing.
    NMPCParams p = liveParams();
    p.soft_joint_limits       = false;
    p.relax_box_to_contain_x0 = true;
    const RunMetrics m = runClosedLoop(p, Scenario{}, 200);

    EXPECT_LT(m.guard_relax_frac, 0.01)
        << "guard fired on " << (100.0 * m.guard_relax_frac) << "% of nominal ticks";
    EXPECT_GT(m.availability, 0.99);
}

TEST(HardConstraints, HardTorqueDoesNotChangeTheNominalCommand)
{
    // Hardening the torque rows changes the QP's PLAN, not the applied torque:
    // the controller already clamps per joint before publishing. Nominal
    // behaviour should therefore be essentially unchanged.
    NMPCParams soft = liveParams();
    NMPCParams hard = liveParams();
    hard.soft_torque_limits = false;

    const RunMetrics ms = runClosedLoop(soft, Scenario{}, 100);
    const RunMetrics mh = runClosedLoop(hard, Scenario{}, 100);

    EXPECT_GT(mh.availability, 0.99) << "hard torque rows cost availability";
    EXPECT_NEAR(mh.mean_ee_err, ms.mean_ee_err, 0.25 * ms.mean_ee_err + 1e-4)
        << "soft " << ms.mean_ee_err << " vs hard " << mh.mean_ee_err;
}

TEST(DISABLED_ConstraintStudy, BoxIsActuallyEnforced)
{
    // Diagnostic: with joint2_L at 0.90 and q_min raised to 1.60, the Δq box for
    // that joint is [0.70, 2.24] while Δq is pinned to 0 at stage 0. Print what
    // the solver actually does with that, for both soft and hard.
    for (bool soft : {true, false}) {
        NMPCParams p = liveParams();
        p.joint_lims[0].q_min = 1.60;
        p.soft_joint_limits   = soft;
        ModelProbe c(kMjcf, p);
        State s = referenceState(c.n_joints_);
        const ArmNMPC::EePose st = c.currentEePose(s.q, s.v);
        c.setDesiredPos(st.pos.x() + 0.06, st.pos.y() + 0.06, st.pos.z());
        c.setDesiredOrient(st.quat(0), st.quat(1), st.quat(2), st.quat(3));

        c.computeControl(s.q, s.v);

        const int nx = 7 + 2 * c.n_joints_;
        Eigen::VectorXd x0(nx), x1(nx);
        c.qp_->getX(0, x0.data());
        c.qp_->getX(1, x1.data());
        const int row = 7 + c.ownedJointInds[0];   // Δq of the offending joint
        printf("DIAG soft=%d  qlim=[%.3f,%.3f] q0=%.3f -> box=[%.3f,%.3f]  "
               "x0[dq]=%.4f x1[dq]=%.4f  status=%d mode=%s nsbx(1)=%d\n",
               soft, c.qlimsByIndex[c.ownedJointInds[0]].first,
               c.qlimsByIndex[c.ownedJointInds[0]].second, s.q[7 + c.ownedJointInds[0]],
               c.qlimsByIndex[c.ownedJointInds[0]].first - s.q[7 + c.ownedJointInds[0]],
               c.qlimsByIndex[c.ownedJointInds[0]].second - s.q[7 + c.ownedJointInds[0]],
               x0[row], x1[row], c.lastQpStatus(),
               c.mode() == ArmNMPC::ControlMode::Damping ? "DAMP" : "nom",
               c.qp_->nsbxAt(1));
    }
}

/*! The experiment. Disabled so it never runs in CI:
 *    ./build/pat_arm_nmpc/test_pat_arm_nmpc \
 *        --gtest_also_run_disabled_tests --gtest_filter='DISABLED_ConstraintStudy.*'
 */
TEST(DISABLED_ConstraintStudy, Ladder)
{
    struct Cfg { const char* name; bool soft_box; bool soft_tau; bool guard; };
    const Cfg cfgs[] = {
        {"soft/soft   ", true,  true,  false},
        {"soft/hardtau", true,  false, false},   // soft box needs no guard
        {"hard-raw    ", false, false, false},
        {"hard-guarded", false, false, true },
        {"hardbox-only", false, true,  true },
    };
    struct Sc { const char* name; Scenario sc; };
    const Sc scs[] = {
        {"nominal (constraints idle)", Scenario{0.06, 0.0}},
        {"aggressive setpoint",        Scenario{0.25, 0.0}},
        {"starts outside a limit",     Scenario{0.06, 1.60}},
    };

    printf("\nHard vs soft constraints. availability = fraction of ticks with an\n"
           "NMPC command (rest is safe-mode damping); maxdamp = longest unbroken\n"
           "damping run; pos_viol = worst realised excursion past the configured\n"
           "joint limits on the flown trajectory; relax = fraction of ticks the\n"
           "guard had to widen a box.\n"
           "NOTE a hard box provides no RESTORING force: soft slack penalises\n"
           "being outside and pulls back, hard+guard only forbids getting worse.\n");

    for (bool term_vel : {true, false}) {
        printf("\n=== terminal_velocity_constraint = %s ===\n", term_vel ? "true" : "false");
        for (const auto& s : scs) {
            printf("\n  scenario: %s\n", s.name);
            printf("    %-13s  avail  maxdamp  viol:init->final  recov   integral  "
                   "ee_err   box_slack  relax%%   tick_us(mean/max)  over%%  status\n", "config");
            for (const auto& cf : cfgs) {
                NMPCParams p = liveParams();
                p.terminal_velocity_constraint = term_vel;
                p.soft_joint_limits            = cf.soft_box;
                p.soft_torque_limits           = cf.soft_tau;
                p.relax_box_to_contain_x0      = cf.guard;
                const RunMetrics m = runClosedLoop(p, s.sc, 300);

                std::string hist;
                for (const auto& [st, n] : m.status_hist)
                    hist += std::to_string(st) + ":" + std::to_string(n) + " ";
                printf("    %-13s %5.1f%%  %7d  %6.4f->%-6.4f  %+6.4f  %7.4f  "
                       "%7.4f  %9.2e  %5.1f%%  %8.0f/%8.0f  %4.1f%%  %s\n",
                       cf.name, 100.0 * m.availability, m.max_consecutive_damping,
                       m.initial_pos_violation, m.final_pos_violation, m.recovered,
                       m.pos_violation_integral, m.mean_ee_err,
                       m.max_box_slack, 100.0 * m.guard_relax_frac,
                       m.mean_tick_us, m.max_tick_us, 100.0 * m.frac_over_budget,
                       hist.c_str());
            }
        }
    }
}

}  // namespace
}  // namespace pat_arm_nmpc
