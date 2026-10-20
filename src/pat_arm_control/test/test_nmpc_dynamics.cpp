/*! The dynamics the controller is built on: the base-eliminated generalised
 *  inertia and bias, the joint limits taken from the model, the rollout
 *  integrator, and the thread-safe full-system query the observer runs on.
 *
 *  Each is compared against MuJoCo's own forward dynamics on the same state.
 *  The question is only "does this compute the plant it claims to", never the
 *  closed loop.
 */
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>

#include <Eigen/Dense>

#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

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

    const Eigen::MatrixXd damped = c.pinvDLS(nd.H_g, c.params_.Hg_damping);
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

/*! The model build gets H_g's damped inverse from a Cholesky of H_g^2 + d^2 I
 *  rather than an SVD. That is an identity for symmetric H_g, not an
 *  approximation, and this pins it: if the two ever diverge, the prediction model
 *  has silently changed. */
TEST(NmpcDynamics, CholeskyDampedInverseMatchesTheSvdOne)
{
    ModelProbe c(kMjcf, leftArmParams());
    const State s = referenceState(c.n_joints_);
    const NodeDynamics nd = c.dynamicsAt(s.q, s.v, c.ee_site_id_);
    const double lambda = c.params_.Hg_damping;

    const Eigen::MatrixXd via_svd = c.pinvDLS(nd.H_g, lambda);

    Eigen::MatrixXd G = nd.H_g * nd.H_g;
    G.diagonal().array() += lambda * lambda;
    Eigen::MatrixXd via_chol;
    ASSERT_TRUE(c.solveSpd(G, nd.H_g, via_chol));

    EXPECT_LT((via_chol - via_svd).norm() / via_svd.norm(), 1e-9)
        << "chol:\n" << via_chol << "\nsvd:\n" << via_svd;
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

    const Eigen::MatrixXd H_g_inv = c.pinvDLS(nd.H_g, c.params_.Hg_damping);
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

// === FloatingBaseDynamics::fullDynamics ==================================== //
// The query the observer thread runs on. Its whole reason for existing is that
// dynamicsAt() and integrateStep() evaluate hypothetical states on mutable
// members, so a second thread cannot use them; these tests check that the
// Workspace form really does avoid that, because the alternative design (a
// second FloatingBaseDynamics, i.e. a second MJCF parse and a second model) was
// rejected on the strength of it.

TEST(FullDynamics, AgreesWithTheLiveStatePath)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const int nv = c.numDof();
    ASSERT_EQ(nv, 6 + c.n_joints_);
    ASSERT_EQ(c.numConfig(), 7 + c.n_joints_);

    // The live path: load the state the way a control tick does, then read the
    // base-eliminated dynamics off it.
    c.loadLiveState(s.q, s.v);
    Eigen::MatrixXd H_g_live;
    Eigen::VectorXd Cv_live;
    c.getDynamics(H_g_live, Cv_live);

    // The Workspace path, at the same state.
    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    c.fullDynamics(s.q, s.v, ws, M, bias, R);

    ASSERT_EQ(M.rows(), nv);
    ASSERT_EQ(M.cols(), nv);
    ASSERT_EQ(bias.size(), nv);
    ASSERT_EQ(R.rows(), 6);
    ASSERT_EQ(R.cols(), c.n_joints_);

    // The mass matrix must be symmetric and positive definite, or nothing
    // downstream of it means anything.
    EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_GT(M.ldlt().vectorD().minCoeff(), 0.0);

    // Reconstruct the reduced quantities from the full ones and check they
    // match what the live path produced. This is the real equivalence claim:
    // H_g = H_mm - H_bm^T H_bb^-1 H_bm, and Cv = c_m - H_bm^T H_bb^-1 c_b.
    const Eigen::MatrixXd H_bm = M.topRightCorner(6, c.n_joints_);
    const Eigen::MatrixXd H_g_rebuilt =
        M.bottomRightCorner(c.n_joints_, c.n_joints_) - H_bm.transpose() * R;
    const Eigen::VectorXd Cv_rebuilt =
        bias.tail(c.n_joints_) - R.transpose() * bias.head(6);

    EXPECT_LT((H_g_rebuilt - H_g_live).cwiseAbs().maxCoeff(), 1e-9)
        << "the full mass matrix does not reduce to the live H_g";
    EXPECT_LT((Cv_rebuilt - Cv_live).cwiseAbs().maxCoeff(), 1e-9)
        << "qfrc_bias does not reduce to the live Cv_joints";
}

TEST(FullDynamics, DoesNotDisturbTheLiveStateOrTheRollout)
{
    // The observer thread must not change what the next control tick computes.
    //
    // Compared against a SECOND controller rather than against this one's own
    // previous tick: the NMPC warm-starts, so two identical computeControl
    // calls legitimately differ by a long way (0.19 N*m here), and asserting
    // they match would be testing the warm start, not the interference.
    auto drive = [](bool interleave) {
        ModelProbe c(kMjcf, liveParams());
        const State s = referenceState(c.n_joints_);
        c.setDesiredPos(0.30, 0.25, 0.0);
        c.computeControl(s.q, s.v);          // establishes the warm start

        if (interleave) {
            // Evaluate somewhere else entirely, repeatedly, exactly as the
            // observer thread would between two control ticks.
            FloatingBaseDynamics::Workspace ws;
            Eigen::MatrixXd M, R;
            Eigen::VectorXd bias;
            Eigen::VectorXd q2 = s.q, v2 = s.v;
            for (int i = 0; i < 20; ++i) {
                q2(7) += 0.01;
                v2(6) = 0.3;
                c.fullDynamics(q2, v2, ws, M, bias, R);
            }
        }
        return c.computeControl(s.q, s.v);   // the tick that must be unaffected
    };

    const Eigen::VectorXd clean = drive(false);
    const Eigen::VectorXd interleaved = drive(true);
    EXPECT_LT((interleaved - clean).cwiseAbs().maxCoeff(), 1e-12)
        << "fullDynamics perturbed the controller's own state\nclean       "
        << clean.transpose() << "\ninterleaved " << interleaved.transpose();
}

TEST(FullDynamics, TwoThreadsWithSeparateWorkspacesAgreeWithSerial)
{
    // The claim that justifies sharing one FloatingBaseDynamics between the
    // control thread and the observer thread. mjModel is read-only once loaded;
    // what must not be shared is the mjData, and a Workspace is exactly that.
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    const int nv = c.numDof();

    // Two different states, so a workspace mix-up produces a wrong answer
    // rather than coincidentally the right one.
    Eigen::VectorXd qA = s.q, vA = s.v;
    Eigen::VectorXd qB = s.q, vB = s.v;
    qB(7) += 0.25;
    vB(6) = -0.4;
    vB(7) = 0.2;

    // Serial references.
    Eigen::MatrixXd MA_ref, RA_ref, MB_ref, RB_ref;
    Eigen::VectorXd bA_ref, bB_ref;
    {
        FloatingBaseDynamics::Workspace ws;
        c.fullDynamics(qA, vA, ws, MA_ref, bA_ref, RA_ref);
        c.fullDynamics(qB, vB, ws, MB_ref, bB_ref, RB_ref);
    }
    ASSERT_GT((MA_ref - MB_ref).cwiseAbs().maxCoeff(), 1e-6)
        << "the two states must differ, or this test cannot detect a mix-up";

    // Now concurrently, each thread with its own Workspace, hammering the
    // shared model for long enough to interleave.
    constexpr int kIters = 400;
    std::atomic<long> mismatches{0};
    auto worker = [&](const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                      const Eigen::MatrixXd& M_ref, const Eigen::VectorXd& b_ref,
                      const Eigen::MatrixXd& R_ref) {
        FloatingBaseDynamics::Workspace ws;
        Eigen::MatrixXd M, R;
        Eigen::VectorXd bias;
        for (int i = 0; i < kIters; ++i) {
            c.fullDynamics(q, v, ws, M, bias, R);
            if ((M - M_ref).cwiseAbs().maxCoeff() > 1e-12 ||
                (bias - b_ref).cwiseAbs().maxCoeff() > 1e-12 ||
                (R - R_ref).cwiseAbs().maxCoeff() > 1e-12) {
                mismatches.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::thread ta([&] { worker(qA, vA, MA_ref, bA_ref, RA_ref); });
    std::thread tb([&] { worker(qB, vB, MB_ref, bB_ref, RB_ref); });
    ta.join();
    tb.join();

    std::printf("[fulldyn] %d concurrent evaluations per thread, nv=%d, "
                "%ld mismatches\n", kIters, nv, mismatches.load());
    EXPECT_EQ(mismatches.load(), 0)
        << "concurrent fullDynamics with separate workspaces diverged from "
           "serial -- something mutable is still shared";
}

TEST(FullDynamics, SteadyStateCallsDoNotAllocate)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);

    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    // First call creates the mjData and sizes the outputs; that one is allowed
    // to allocate, and the observer thread makes it during bring-up.
    c.fullDynamics(s.q, s.v, ws, M, bias, R);

    long allocs = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 20; ++i) c.fullDynamics(s.q, s.v, ws, M, bias, R);
        allocs = scope.n();
    }
    std::printf("[alloc] fullDynamics steady state %.2f per call\n",
                static_cast<double>(allocs) / 20.0);
    EXPECT_EQ(allocs, 0);
}

TEST(FullDynamics, RejectsAMisSizedState)
{
    ModelProbe c(kMjcf, liveParams());
    FloatingBaseDynamics::Workspace ws;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    const Eigen::VectorXd q_bad = Eigen::VectorXd::Zero(3);
    const Eigen::VectorXd v_ok = Eigen::VectorXd::Zero(c.numDof());
    EXPECT_THROW(c.fullDynamics(q_bad, v_ok, ws, M, bias, R), std::runtime_error);
}

}  // namespace
}  // namespace pat_arm_control
