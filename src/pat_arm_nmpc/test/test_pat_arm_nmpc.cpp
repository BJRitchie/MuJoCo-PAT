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
#include <cmath>
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
    using ArmNMPC::evalTaskSpace;
    using ArmNMPC::ownedJointInds;
    using ArmNMPC::n_qpsolve_;
    using ArmNMPC::params_;
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
    p.params.ng      = 0;
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
    p.D_k.clear();
    p.lg_k.clear();
    p.ug_k.clear();

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
double maxBoxSlackOverRun(const NMPCParams& params, int ticks)
{
    const double dt = 1.0 / 200.0;
    ModelProbe c(kMjcf, params);
    State s = referenceState(c.n_joints_);

    const ArmNMPC::EePose start = c.currentEePose(s.q, s.v);
    c.setDesiredPos(start.pos.x() + 0.06, start.pos.y() + 0.06, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    double worst = 0.0;
    for (int t = 0; t < ticks; ++t) {
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        const int ns = c.qp_->nsAt(1);
        if (ns > 0) {
            Eigen::VectorXd sl(ns), su(ns);
            c.qp_->getSlack(1, sl.data(), su.data());
            const int n_box = 2 * static_cast<int>(c.ownedJointInds.size());
            worst = std::max({worst, sl.head(n_box).maxCoeff(), su.head(n_box).maxCoeff()});
        }
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;
    }
    return worst;
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

}  // namespace
}  // namespace pat_arm_nmpc
