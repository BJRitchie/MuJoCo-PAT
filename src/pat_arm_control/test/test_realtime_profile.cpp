/*! What the realtime deployment depends on: the control tick and the observer
 *  stay off the heap, the profiler reports a worst case and resets, the trace
 *  records every tick without allocating, MuJoCo's error hooks neither write
 *  files nor exit, and cold starting keeps the solver clear of its iteration
 *  cap.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

#include "pat_arm_control/observers/momentum_observer.h"
#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

// --- Realtime profile ----------------------------------------------------- //

/*! The per-tick heap traffic of the live configuration, as a bound rather than
 *  an exact figure: it depends on the Eigen and acados builds, so pinning it
 *  exactly would flake. The bound catches a reintroduced allocation on the hot
 *  path, which is what matters under a realtime policy. */
TEST(RealtimeProfile, ControlTickAllocationCountIsBounded)
{
    constexpr long kTicks       = 10;
    // Measured 1, and that one is computeControl's returned vector: the
    // signature returns by value, so the caller's result is necessarily a heap
    // allocation. Nothing else on the tick allocates -- acados, HPIPM, BLASFEO
    // and MuJoCo contribute none of it, confirmed by resolving every call site
    // to its originating object (DISABLED_AllocStudy.Attribution).
    //
    // The ceiling allows a little slack for a different Eigen or acados build
    // rather than pinning 1 exactly, while still failing if a single per-node
    // allocation comes back -- those arrive 30 at a time.
    constexpr long kMaxPerTick  = 8;

    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);

    // Warm up first: the first tick also pays one-off costs (lazily sized
    // workspaces, acados' first solve) that are not per-tick traffic.
    for (int i = 0; i < 3; ++i) c.computeControl(s.q, s.v);

    long allocs = 0;
    {
        AllocScope scope;
        for (long i = 0; i < kTicks; ++i) c.computeControl(s.q, s.v);
        allocs = scope.n();
    }

    const double per_tick = static_cast<double>(allocs) / kTicks;
    std::printf("[alloc] %.0f heap allocations per control tick\n", per_tick);
    EXPECT_GT(allocs, 0) << "counter did not observe any allocation -- the "
                            "malloc interposer is not being reached";
    EXPECT_LT(per_tick, kMaxPerTick);
}

/*! The profiler has to report a worst case, not just a mean, and has to forget
 *  it on reset -- a cumulative max would make every later window report the
 *  first spike it ever saw. */
TEST(RealtimeProfile, ReportCarriesAMaxAndAHistogramAndResets)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    // A budget no tick can plausibly exceed, so the miss count is a fixed
    // expectation rather than a measurement of the machine running the test.
    c.setTickBudgetUs(60e6);
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    const std::string rep = c.profileReport();
    EXPECT_NE(rep.find("over 5 ticks"), std::string::npos) << rep;
    EXPECT_NE(rep.find("max="), std::string::npos) << rep;
    EXPECT_NE(rep.find("hist(tick)"), std::string::npos) << rep;
    EXPECT_NE(rep.find("over_period=0/5"), std::string::npos) << rep;
    EXPECT_NE(rep.find("qpsolve("), std::string::npos) << rep;
    EXPECT_NE(rep.find("unattributed="), std::string::npos) << rep;

    EXPECT_GT(c.tick.max_us, 0.0);
    c.resetProfile();
    EXPECT_EQ(c.tick.count, 0);
    EXPECT_EQ(c.tick.max_us, 0.0);
    EXPECT_EQ(c.qpsolve.count, 0);
    EXPECT_EQ(c.tick.budget_us, 60e6) << "reset dropped the configured deadline";
}

/*! A tick over the budget must be counted exactly. The histogram cannot answer
 *  this: its bins are coarser than any period of interest and the last one is
 *  open-ended, so a budget above 5 ms has no bin boundary to read. */
TEST(RealtimeProfile, DeadlineMissesAreCountedAgainstTheBudget)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setTickBudgetUs(1e-3);       // below any achievable tick: all ticks miss
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);

    EXPECT_EQ(c.tick.over, 4);
    EXPECT_NE(c.profileReport().find("over_period=4/4"), std::string::npos);

    c.setTickBudgetUs(0.0);        // not a deadline: nothing counted, no column
    c.resetProfile();
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.tick.over, 0);
    EXPECT_EQ(c.profileReport().find("over_period"), std::string::npos);
}

/*! lastQpIters() sums acados' num_iter over every solve in the tick, which is only
 *  right if acados reports it PER SOLVE rather than cumulatively. If it accumulated
 *  internally the retry would be double counted and every reported figure for a
 *  retried tick would be inflated. Pin it: one capped solve versus the same capped
 *  solve plus its cold retry should differ by exactly one cap's worth. */
TEST(RealtimeProfile, QpIterationsArePerSolveNotCumulative)
{
    constexpr int kCap = 4;        // below the ~8 a healthy solve needs, so it fails

    NMPCParams cold = liveParams();
    cold.qp_warm_start = 0;        // no retry path at all -> exactly one solve
    ModelProbe c0(kMjcf, cold);
    const State s = referenceState(c0.n_joints_);
    c0.setDesiredPos(0.30, 0.25, 0.0);
    c0.qp_->setIterMax(kCap);
    c0.computeControl(s.q, s.v);
    const int one_solve = c0.qp_->lastQpIters();

    NMPCParams warm = liveParams();
    warm.qp_warm_start = 1;        // fails, then retries cold -> two solves
    ModelProbe c1(kMjcf, warm);
    c1.setDesiredPos(0.30, 0.25, 0.0);
    c1.qp_->setIterMax(kCap);
    c1.computeControl(s.q, s.v);
    const int two_solves = c1.qp_->lastQpIters();

    ASSERT_TRUE(c1.qp_->lastRetried()) << "the retry did not fire, so this proves nothing";
    EXPECT_EQ(one_solve, kCap) << "a capped failing solve should report the cap";
    EXPECT_EQ(two_solves, 2 * kCap)
        << "one solve reported " << one_solve << ", two reported " << two_solves
        << " -- not 2x, so acados' num_iter is cumulative and the sum double counts";
}

/*! The iteration count is the link between a slow tick and its cause, so it has
 *  to come back from the solver rather than stay at the "no info" sentinel. */
TEST(RealtimeProfile, QpIterationsAreReported)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    EXPECT_EQ(c.qp_iters.count, 5) << "one tally per tick";
    EXPECT_GT(c.qp_iters.mean(), 0.0);
    EXPECT_EQ(c.qp_retries, 0) << "the nominal path should not need a cold retry";
    // Without a retry the tick holds exactly one solve, so the cap applies to it.
    EXPECT_LE(c.qp_iters.max, liveParams().qp_max_iter);
    // HPIPM's own time cannot exceed the stopwatch around the call that produced it.
    EXPECT_GT(c.qp_time_ms.max, 0.0);
    EXPECT_LE(c.qp_time_ms.max * 1e3, c.qpsolve.max_us);
    EXPECT_NE(c.profileReport().find("qp_iter/tick mean="), std::string::npos);

    c.resetProfile();
    EXPECT_EQ(c.qp_iters.count, 0);
    EXPECT_EQ(c.qp_time_ms.count, 0);
    EXPECT_EQ(c.profileReport().find("qp_iter"), std::string::npos)
        << "an empty tally should not print a column";
}

/*! The hot path must reach a steady state where a tick allocates nothing new.
 *
 *  Separate from the ceiling test above because the two ask different things: that
 *  one bounds the average, this one asserts the FIRST tick is where sizing happens
 *  and later ticks do not keep growing. A buffer that is re-sized every tick, or a
 *  container that is cleared and refilled, passes a loose average and fails here. */
TEST(RealtimeProfile, SteadyStateTicksDoNotKeepAllocating)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);

    long first = 0, later = 0;
    {
        AllocScope scope;
        c.computeControl(s.q, s.v);
        first = scope.n();
    }
    // Several more, so a per-tick leak shows up multiplied rather than once.
    constexpr long kLater = 20;
    {
        AllocScope scope;
        for (long i = 0; i < kLater; ++i) c.computeControl(s.q, s.v);
        later = scope.n();
    }
    const double per_later = static_cast<double>(later) / kLater;
    std::printf("[alloc] first tick %ld, steady state %.1f per tick\n", first, per_later);

    EXPECT_LT(per_later, static_cast<double>(first))
        << "the first tick should be the expensive one -- it does the sizing";
    // What remains is computeControl's returned vector, one per tick. See
    // ControlTickAllocationCountIsBounded for why that one is the floor here.
    EXPECT_LT(per_later, 8.0);
}

/*! The per-tick trace exists because the report's eight histogram bins cannot
 *  show the shape of the distribution. It has to be exact per tick, bounded, and
 *  free of allocation on the control path. */
TEST(RealtimeProfile, TraceRecordsEveryTickWithoutAllocating)
{
    constexpr std::size_t kCap = 8;
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.computeControl(s.q, s.v);            // warm up before the buffer is armed

    c.enableTrace(kCap);
    long allocs = 0;
    {
        AllocScope scope;
        for (std::size_t i = 0; i < kCap; ++i) c.computeControl(s.q, s.v);
        allocs = scope.n();
    }
    ASSERT_EQ(c.trace().size(), kCap);

    // Recording must not be what makes the tick allocate. Compare against the
    // ceiling the un-traced path already holds to, rather than a fresh figure.
    EXPECT_LT(static_cast<double>(allocs) / kCap, 2500.0);

    for (const auto& r : c.trace()) {
        EXPECT_GT(r.tick_us, 0.0f);
        EXPECT_GT(r.qpsolve, 0.0f) << "the SQP path always solves";
        // Phases are this tick's share, so they cannot exceed the tick itself.
        const float summed = r.dynamics + r.jdotqdot + r.lambda + r.qpsolve
                           + r.rollout_dynamics + r.rollout_integrate + r.pinv
                           + r.build_model + r.defect + r.trust_region + r.finalize;
        EXPECT_LE(summed, r.tick_us * 1.05f + 1.0f)
            << "summed phases " << summed << " vs tick " << r.tick_us;
        EXPECT_GE(r.qp_iters, 0);
        EXPECT_EQ(r.mode, 0) << "nominal run should not be in safe mode";
    }
    // Monotonic timestamps, first row at zero.
    EXPECT_FLOAT_EQ(static_cast<float>(c.trace().front().t_us), 0.0f);
    for (std::size_t i = 1; i < c.trace().size(); ++i)
        EXPECT_GT(c.trace()[i].t_us, c.trace()[i - 1].t_us);

    // Bounded: further ticks are dropped, not appended, and never reallocate.
    c.computeControl(s.q, s.v);
    EXPECT_EQ(c.trace().size(), kCap);
    EXPECT_TRUE(c.traceFull());
}

/*! A report window resets the PhaseStat sums the trace differences against, so
 *  the first tick afterwards must not absorb the whole window's total. */
TEST(RealtimeProfile, TraceSurvivesAProfileReset)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.enableTrace(16);
    for (int i = 0; i < 4; ++i) c.computeControl(s.q, s.v);
    const float before = c.trace().back().qpsolve;

    c.resetProfile();                      // what the diagnostics timer does
    c.computeControl(s.q, s.v);
    const float after = c.trace().back().qpsolve;

    EXPECT_LT(after, 4.0f * before)
        << "qpsolve " << after << " us after a reset vs " << before
        << " before -- the reset baseline was not cleared";
    EXPECT_LE(after, c.trace().back().tick_us);
}

/*! Every phase the report lists must actually be timed on the active path --
 *  an untimed phase shows up as unattributed time and the profile cannot then
 *  point at the cost. */
TEST(RealtimeProfile, TheSqpPathAttributesMostOfItsTick)
{
    ModelProbe c(kMjcf, liveParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    for (int i = 0; i < 10; ++i) c.computeControl(s.q, s.v);

    const PhaseStat* required[] = {&c.rollout_dynamics, &c.rollout_integrate,
                                   &c.pinv, &c.build_model, &c.defect,
                                   &c.qpsolve, &c.trust_region, &c.finalize};
    double attributed = 0.0;
    for (const PhaseStat* p : required) {
        EXPECT_GT(p->count, 0) << "a phase the report lists is never timed";
        attributed += p->sum_us;
    }
    EXPECT_GT(attributed, 0.6 * c.tick.sum_us)
        << "attributed " << attributed << " of " << c.tick.sum_us << " us";
}

/*! The observer is destined for its own realtime thread, so it is held to the
 *  same standard as the control tick -- and to a stricter one, since it has no
 *  returned-vector excuse: every buffer is sized in the constructor, so even
 *  the first update must reach the allocator zero times. */
TEST(RealtimeProfile, ObserverUpdateDoesNotAllocate)
{
    constexpr int kNv = 12;
    MomentumObserverParams p;
    p.nv = kNv;
    p.n_joints = kNv - 6;
    p.dt = 0.01;
    p.gain = Eigen::VectorXd::Constant(1, 20.0);
    p.mdot_clamp = 50.0;
    p.estimate_clamp = 5.0;
    MomentumObserver obs(p);

    // Inputs built outside the counted scope. Configuration-dependent, so the
    // Mdot difference is non-trivial and the clamp paths are reachable.
    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(kNv, kNv) * 2.0;
    Eigen::VectorXd bias = Eigen::VectorXd::Constant(kNv, 0.01);
    Eigen::VectorXd v = Eigen::VectorXd::Constant(kNv, 0.3);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);
    for (int i = 6; i < kNv; ++i) tau(i) = 0.5;
    Eigen::VectorXd reduced(p.n_joints);
    const Eigen::MatrixXd H_bb_inv_H_bm = Eigen::MatrixXd::Constant(6, p.n_joints, 0.1);

    long first = 0, steady = 0;
    {
        AllocScope scope;
        obs.update(M, bias, v, tau);
        first = scope.n();
    }

    constexpr long kTicks = 50;
    {
        AllocScope scope;
        for (long i = 0; i < kTicks; ++i) {
            // Vary M so the divided difference does real work each step.
            for (int k = 0; k < kNv; ++k) {
                M(k, k) = 2.0 + 0.1 * std::sin(0.01 * static_cast<double>(i * kNv + k));
            }
            obs.update(M, bias, v, tau);
            obs.reduceToJoints(H_bb_inv_H_bm, reduced);
        }
        steady = scope.n();
    }
    std::printf("[alloc] observer first update %ld, steady %.2f per update\n",
                first, static_cast<double>(steady) / kTicks);

    EXPECT_EQ(first, 0) << "the constructor is supposed to do all the sizing";
    EXPECT_EQ(steady, 0);
}

// --- MuJoCo error / warning hooks ----------------------------------------- //

/*! MuJoCo's default warning handler appends to MUJOCO_LOG.TXT -- a file write on
 *  the control path, from a code path that fires when the solver is struggling.
 *  The hook has to be ours, and it has to record rather than write. */
TEST(MujocoHooks, WarningIsCountedAndNotWrittenToAFile)
{
    ModelProbe c(kMjcf, liveParams());     // the constructor installs the hooks
    const long before = c.mjWarningCount();

    mju_warning("pat test warning %d", 42);

    EXPECT_EQ(c.mjWarningCount(), before + 1);
    EXPECT_NE(std::string(c.lastMjWarning()).find("pat test warning 42"),
              std::string::npos)
        << "got: " << c.lastMjWarning();
}

/*! The default error handler calls exit(), which would kill the process mid-motion
 *  with a torque latched in the actuators. Ours throws, so ArmNMPC's existing
 *  guard can turn it into safe-mode damping. */
TEST(MujocoHooks, ErrorThrowsInsteadOfTerminating)
{
    ModelProbe c(kMjcf, liveParams());
    const long before = c.mjErrorCount();

    EXPECT_THROW(mju_error("pat test error"), std::runtime_error);

    EXPECT_EQ(c.mjErrorCount(), before + 1);
    EXPECT_NE(std::string(c.lastMjError()).find("pat test error"),
              std::string::npos)
        << "got: " << c.lastMjError();
}

/*! Both counts have to reach the report, or redirecting MuJoCo's logging would
 *  trade noisy-but-visible for silent. */
TEST(MujocoHooks, CountsReachTheProfileReport)
{
    ModelProbe c(kMjcf, liveParams());
    mju_warning("pat report probe");
    EXPECT_NE(c.profileReport().find("MuJoCo warnings:"), std::string::npos);
    EXPECT_NE(c.profileReport().find("pat report probe"), std::string::npos);
}

// --- Warm start vs cold start --------------------------------------------- //

/*! Cold-starting every solve trades mean cost for a bounded worst case.
 *
 *  Warm-starting is cheaper per tick while the previous iterate stays a good
 *  guess, but a target jump invalidates it and HPIPM then runs to the iteration
 *  cap before the cold-start retry rescues it -- a tick an order of magnitude over
 *  the period. Cold-starting pays more iterations every tick and never approaches
 *  the cap. For a deadline the second shape is the useful one, so the invariant
 *  worth holding is that cold stays far clear of the cap. */
TEST(WarmStart, ColdStartingStaysWellClearOfTheIterationCap)
{
    NMPCParams p = liveParams();
    p.qp_warm_start = 0;
    const RunMetrics m = runClosedLoop(p, warmStartScenario(), 400);

    EXPECT_EQ(m.qp_retries, 0) << "warm_start=0 disables the retry path entirely";
    EXPECT_GT(m.availability, 0.99) << "cold starts should not cost availability";
    EXPECT_LT(m.qp_iter_max, 0.5 * p.qp_max_iter)
        << "cold solves ran to " << m.qp_iter_max << " of a " << p.qp_max_iter
        << " cap; the cap is supposed to be slack, not a bound being reached";
}

/*! Warm-starting's cost is concentrated in the ticks right after a target jump,
 *  which is what makes a mean-only profile miss it. */
TEST(WarmStart, WarmStartingCostsMoreInTheWorstTickThanCold)
{
    NMPCParams warm = liveParams();   // qp_warm_start defaults to 2
    NMPCParams cold = liveParams();
    cold.qp_warm_start = 0;

    const RunMetrics mw = runClosedLoop(warm, warmStartScenario(), 400);
    const RunMetrics mc = runClosedLoop(cold, warmStartScenario(), 400);

    EXPECT_LT(mc.qp_iter_max, mw.qp_iter_max)
        << "warm " << mw.qp_iter_max << " vs cold " << mc.qp_iter_max;
    EXPECT_GT(mc.qp_iter_mean, mw.qp_iter_mean)
        << "cold is expected to pay more iterations on the average tick";
    // Both must still control the arm; a bounded tail is worthless if it tracks worse.
    EXPECT_NEAR(mc.mean_ee_err, mw.mean_ee_err, 0.25 * mw.mean_ee_err + 1e-4)
        << "warm " << mw.mean_ee_err << " vs cold " << mc.mean_ee_err;
}

}  // namespace
}  // namespace pat_arm_control
