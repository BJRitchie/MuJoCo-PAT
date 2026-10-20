/*! The QP and its constraints: the solver's contract on a hand-checkable
 *  double integrator, the soft constraints' slack in closed loop, and the
 *  measured behaviour of hard versus soft joint and torque rows.
 *
 *  The hard/soft defaults (soft box, hard torque rows) were chosen from these
 *  and from DISABLED_ConstraintStudy.Ladder; see the README's "Constraint
 *  configuration" before changing either.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

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
    // Straining the box this hard still costs time, but with HARD torque rows it
    // no longer costs much availability: ~96.7% on both x86-64 and aarch64, with
    // the joint fully recovered. This assertion is the one that caught soft torque
    // rows collapsing on aarch64 (21.7%, joint never recovered) while passing on
    // x86-64 -- keep it, and keep it on availability, because that is the column
    // that moved.
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

TEST(HardConstraints, TorqueRowHardnessDoesNotChangeTheNominalCommand)
{
    // Hardening the torque rows changes the QP's PLAN, not the applied torque:
    // the controller already clamps per joint before publishing. Nominal
    // behaviour is therefore essentially unchanged either way, which is what
    // makes the hard default free -- it buys robustness near infeasibility
    // without costing anything here.
    //
    // `hard` is the live configuration and `soft` the variant, the opposite way
    // round from when soft torque was the default. Set explicitly rather than
    // taken from liveParams(), so this keeps comparing two different things if
    // the default moves again.
    NMPCParams hard = liveParams();
    NMPCParams soft = liveParams();
    hard.soft_torque_limits = false;
    soft.soft_torque_limits = true;

    const RunMetrics mh = runClosedLoop(hard, Scenario{}, 100);
    const RunMetrics ms = runClosedLoop(soft, Scenario{}, 100);

    EXPECT_GT(mh.availability, 0.99) << "hard torque rows cost availability";
    EXPECT_NEAR(mh.mean_ee_err, ms.mean_ee_err, 0.25 * ms.mean_ee_err + 1e-4)
        << "hard " << mh.mean_ee_err << " vs soft " << ms.mean_ee_err;
}

}  // namespace
}  // namespace pat_arm_control
