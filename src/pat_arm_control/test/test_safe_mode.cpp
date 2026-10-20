/*! The graded response to a solve that did not produce a usable answer:
 *  Tier 1 replays the last accepted plan, Tier 2 damps the joints, Tier 3
 *  latches. And the damping itself, which has to be safe however wrong the
 *  model is: dissipative, finite, inside the torque limits, and stable in
 *  discrete time.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

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
    // A non-finite state must become safe mode, not an exception that escapes
    // the tick and takes the process with it.
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
    {   // sqp_iters = 0 would skip the rollout loop and leave J_task0 at 0x0
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

// --- Tiered failure response ---------------------------------------------- //

/*! Stage mapping: the plan's stages are Ts apart, the ticks control_period apart,
 *  so a stage is held for the ticks it spans. Getting this wrong runs through the
 *  plan 2.5x too fast at Ts=25 ms and a 10 ms period. */
TEST(Tiers, PlanStageHoldsForTheTicksAStageSpans)
{
    // Pins its own Ts and period rather than inheriting liveParams(): this is a
    // test of the stage-hold arithmetic, so it must not change meaning when the
    // deployed horizon is retuned.
    NMPCParams p = liveParams();
    p.Ts = 0.025;
    p.control_period = 0.010;             // 2.5 ticks per stage
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);           // produces an accepted plan

    // age 1,2 -> stage 0 (0 and 10 ms into a 25 ms stage); age 3,4,5 -> stage 1; ...
    EXPECT_EQ(c.planStageForAge(1), 0);
    EXPECT_EQ(c.planStageForAge(2), 0);
    EXPECT_EQ(c.planStageForAge(3), 0);
    EXPECT_EQ(c.planStageForAge(4), 1);
    EXPECT_EQ(c.planStageForAge(6), 2);
    EXPECT_EQ(c.planStageForAge(0), -1) << "age 0 is the accepting tick itself";
    // Past the horizon the plan is exhausted and Tier 1 must give up.
    EXPECT_EQ(c.planStageForAge(1 + static_cast<int>(2.5 * p.N) + 5), -1);
}

/*! A single failed solve should replay the plan, not abandon the task, which
 *  would cost safe_mode_recovery_ticks of damping. */
TEST(Tiers, OneFailedSolveReplaysThePlanInsteadOfDamping)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    const Eigen::VectorXd good = c.computeControl(s.q, s.v);
    ASSERT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal);

    // Force exactly one failed solve by starving the solver, then restore it.
    c.qp_->setIterMax(1);
    const Eigen::VectorXd replayed = c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::PlanReplay);
    EXPECT_EQ(c.tier1Entries(), 1);
    EXPECT_TRUE(replayed.allFinite());
    // A replay is a task-space command, so it should look nothing like damping,
    // which at near-zero velocity is near-zero torque.
    EXPECT_GT(replayed.norm(), 1e-6);

    c.qp_->setIterMax(p.qp_max_iter);
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "one good solve should end a Tier 1 replay -- no hysteresis";
}

/*! The replay budget has to be finite: a plan is only valid near the state it was
 *  computed at, so a solver that stays broken must reach damping. */
TEST(Tiers, ReplayBudgetExpiresIntoDamping)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    p.tier1_max_replay_ticks = 3;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    for (int i = 0; i < 3; ++i) {
        c.computeControl(s.q, s.v);
        EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::PlanReplay) << "replay tick " << i;
    }
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping) << "budget spent, must damp";
    EXPECT_EQ(c.tier1Ticks(), 3);
}

/*! Tier 1 off must fall straight to damping, so the tiers can be switched out if
 *  they ever misbehave in flight. */
TEST(Tiers, DisablingTier1FallsStraightToDamping)
{
    NMPCParams p = liveParams();
    p.tier1_max_replay_ticks = 0;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Damping);
    EXPECT_EQ(c.tier1Ticks(), 0);
}

/*! Tier 3 stops the controller flapping between damping and tracking forever, and
 *  must not clear itself. */
TEST(Tiers, LatchesAfterSustainedDampingAndNeedsAnExplicitClear)
{
    NMPCParams p = liveParams();
    p.tier1_max_replay_ticks  = 0;     // go straight to damping
    p.tier3_latch_after_ticks = 4;
    ModelProbe c(kMjcf, p);
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);
    ASSERT_TRUE(c.latched());

    // Healthy solves must NOT clear it.
    c.qp_->setIterMax(p.qp_max_iter);
    for (int i = 0; i < 3 * p.safe_mode_recovery_ticks; ++i) c.computeControl(s.q, s.v);
    EXPECT_TRUE(c.latched()) << "a latch that times out is not a latch";

    c.clearLatch();
    EXPECT_FALSE(c.latched());
    for (int i = 0; i < p.safe_mode_recovery_ticks + 1; ++i) c.computeControl(s.q, s.v);
    EXPECT_EQ(c.mode(), ArmNMPC::ControlMode::Nominal)
        << "after clearing, normal Tier 2 recovery should apply";
}

/*! Tier 1 reuses only the stale task WRENCH; the Jacobian and the Cv feedforward
 *  are re-derived at the current measurement. So the same plan stage at a different
 *  state must produce a different joint torque. */
TEST(Tiers, ReplayUsesCurrentKinematicsNotAStaleTorque)
{
    NMPCParams p = liveParams();
    p.control_period = 0.010;
    ModelProbe c(kMjcf, p);
    State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);

    c.qp_->setIterMax(1);
    const Eigen::VectorXd at_rest = c.computeControl(s.q, s.v);

    // Same plan, same stage age, but the arm is now moving: Cv changes, so the
    // commanded torque must too.
    for (int idx : c.ownedJointInds) s.v[6 + idx] = 0.4;
    ModelProbe c2(kMjcf, p);
    c2.setDesiredPos(0.30, 0.25, 0.0);
    c2.computeControl(s.q, Eigen::VectorXd::Zero(s.v.size()));
    c2.qp_->setIterMax(1);
    const Eigen::VectorXd moving = c2.computeControl(s.q, s.v);

    ASSERT_EQ(c2.mode(), ArmNMPC::ControlMode::PlanReplay);
    EXPECT_GT((moving - at_rest).norm(), 1e-9)
        << "replay looks like it is reusing a torque rather than a wrench";
}

}  // namespace
}  // namespace pat_arm_control
