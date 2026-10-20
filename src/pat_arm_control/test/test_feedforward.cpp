/*! The disturbance feedforward: what stands between an observer's estimate and
 *  the actuators.
 *
 *  Compensating an estimate closes a loop around it, so each guard gets its own
 *  test: the default must be a literal no-op, the compensation must be the
 *  stated fraction and nothing more, and the cap, the slew limit and the
 *  safe-mode reset must each bind.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

//! liveParams with the feedforward enabled, so each test states only what it
//! is actually varying.
NMPCParams ffParams(double gain, double max_frac = 0.3, double slew = 0.0)
{
    NMPCParams p = liveParams();
    p.ff_gain     = gain;
    p.ff_max_frac = max_frac;
    p.ff_slew     = slew;
    return p;
}

//! A disturbance on the owned joints only, which is what the reduction yields.
Eigen::VectorXd ffEstimate(int n_joints, double value)
{
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_joints);
    for (int i = 0; i < n_joints && i < 3; ++i) d(i) = value * (i == 1 ? -1.0 : 1.0);
    return d;
}

/*! An estimate sized as a fraction of each joint's feedforward CAP, so a test
 *  that wants to stay clear of the saturation says so instead of guessing a
 *  magnitude. The harness runs 0.25 N*m limits on every owned joint where the
 *  shipped config runs 10/10/5, so a hardcoded N*m value means something quite
 *  different in each and silently saturates in one of them. */
Eigen::VectorXd ffEstimateBelowCap(const ModelProbe& c, double frac_of_cap)
{
    Eigen::VectorXd d = Eigen::VectorXd::Zero(c.n_joints_);
    for (int i = 0; i < c.n_joints_ && i < 3; ++i) {
        const double cap = c.params_.ff_max_frac *
                           c.torqueLimsByIndex[static_cast<std::size_t>(i)];
        d(i) = frac_of_cap * cap * (i == 1 ? -1.0 : 1.0);
    }
    return d;
}

TEST(Feedforward, GainOfZeroIsABitExactNoOp)
{
    // The gate that lets the observer run on a configuration whose numbers are
    // already committed: offering an estimate at gain 0 must change nothing at
    // all, not merely change it by a little.
    const State s = referenceState(6);

    ModelProbe baseline(kMjcf, liveParams());
    baseline.setDesiredPos(0.30, 0.25, 0.0);
    const Eigen::VectorXd tau_ref = baseline.computeControl(s.q, s.v);

    ModelProbe offered(kMjcf, ffParams(0.0));
    offered.setDesiredPos(0.30, 0.25, 0.0);
    offered.setDisturbanceFeedforward(ffEstimate(offered.n_joints_, 2.0), 0.010);
    const Eigen::VectorXd tau = offered.computeControl(s.q, s.v);

    ASSERT_EQ(tau.size(), tau_ref.size());
    for (int i = 0; i < tau.size(); ++i) EXPECT_EQ(tau(i), tau_ref(i)) << "joint " << i;
    EXPECT_EQ(offered.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0);
}

TEST(Feedforward, CommandCancelsExactlyTheGainTimesTheEstimate)
{
    // Checked at finalizeJointTorques rather than through computeControl. The
    // bounds shift changes the QP's feasible set, so when the torque constraint
    // is ACTIVE the solver's u legitimately moves and the end-to-end command is
    // not the baseline minus d -- this harness runs 0.25 N*m limits, where it
    // is active. The claim being made here is about the command the controller
    // forms from a given solution, and that is exact.
    const State s = referenceState(6);

    auto commandWith = [&](double gain, const Eigen::VectorXd* d) {
        ModelProbe c(kMjcf, ffParams(gain));
        c.loadLiveState(s.q, s.v);
        Eigen::MatrixXd H_g;
        Eigen::VectorXd Cv;
        c.getDynamics(H_g, Cv);
        const Eigen::MatrixXd J = c.gjm6(c.ee_site_id_);
        // A fixed task wrench, so the only thing differing between the two
        // calls is the feedforward.
        Eigen::VectorXd tau_task = Eigen::VectorXd::Zero(J.rows());
        tau_task(0) = 0.4;
        tau_task(1) = -0.25;
        if (d) c.setDisturbanceFeedforward(*d, 0.010);
        Eigen::VectorXd out;
        c.finalizeJointTorques(J, tau_task, Cv, out);
        return out;
    };

    ModelProbe probe(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(probe, 0.5);

    const Eigen::VectorXd tau_ref = commandWith(1.0, nullptr);
    const Eigen::VectorXd tau_ff  = commandWith(1.0, &d);
    const Eigen::VectorXd tau_half = [&] {
        const Eigen::VectorXd dh = ffEstimateBelowCap(probe, 0.5);
        return commandWith(0.5, &dh);
    }();

    for (int i = 0; i < tau_ref.size(); ++i) {
        // Only where the per-joint clamp is not what decided the answer.
        const double lim = probe.torqueLimsByIndex[static_cast<std::size_t>(i)];
        if (std::fabs(tau_ref(i)) >= lim - 1e-9) continue;
        EXPECT_NEAR(tau_ff(i), tau_ref(i) - d(i), 1e-9)
            << "joint " << i << ": full gain did not cancel the estimate";
        EXPECT_NEAR(tau_half(i), tau_ref(i) - 0.5 * d(i), 1e-9)
            << "joint " << i << ": half gain did not cancel half of it";
    }
}

TEST(Feedforward, TorqueBoundsShiftWithTheCommand)
{
    // The other half of the change, and the reason a torque limit still means
    // what it says: the QP's general constraint bounds the APPLIED torque, so
    // when the command gains a -d_ff term both bounds must move by +d_ff. If
    // they did not, the solver would plan against a budget the command does not
    // have and the clamp in finalizeJointTorques would quietly take it back.
    const State s = referenceState(6);

    auto boundsWith = [&](const Eigen::VectorXd* d) {
        ModelProbe c(kMjcf, ffParams(1.0));
        c.loadLiveState(s.q, s.v);
        Eigen::MatrixXd H_g;
        Eigen::VectorXd Cv;
        c.getDynamics(H_g, Cv);
        const Eigen::MatrixXd J = c.gjm6(c.ee_site_id_);
        if (d) c.setDisturbanceFeedforward(*d, 0.010);
        Eigen::MatrixXd D;
        Eigen::VectorXd lg, ug;
        c.buildTorqueGeneralConstraint(J, Cv, D, lg, ug);
        return std::make_pair(lg, ug);
    };

    ModelProbe probe(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(probe, 0.5);

    const auto ref = boundsWith(nullptr);
    const auto shifted = boundsWith(&d);

    for (int i = 0; i < d.size(); ++i) {
        EXPECT_NEAR(shifted.first(i), ref.first(i) + d(i), 1e-12)
            << "lower bound, joint " << i;
        EXPECT_NEAR(shifted.second(i), ref.second(i) + d(i), 1e-12)
            << "upper bound, joint " << i;
    }
    // The width is untouched: the feedforward moves the window, it does not
    // widen it, so the arm keeps exactly the authority it had.
    for (int i = 0; i < d.size(); ++i) {
        EXPECT_NEAR(shifted.second(i) - shifted.first(i),
                    ref.second(i) - ref.first(i), 1e-12) << "joint " << i;
    }
}

TEST(Feedforward, SaturationBoundsAWildEstimate)
{
    // The estimate is unbounded by construction -- a momentum residual during
    // hard contact is whatever the contact was -- and it goes to the actuators.
    ModelProbe ff(kMjcf, ffParams(1.0, 0.25));
    const Eigen::VectorXd d = Eigen::VectorXd::Constant(ff.n_joints_, 1000.0);
    ff.setDisturbanceFeedforward(d, 0.010);

    const Eigen::VectorXd& applied = ff.disturbanceFeedforward();
    for (int i = 0; i < ff.n_joints_; ++i) {
        const double cap = 0.25 * ff.torqueLimsByIndex[static_cast<std::size_t>(i)];
        EXPECT_LE(std::fabs(applied(i)), cap + 1e-12)
            << "joint " << i << " exceeded its cap";
    }
    EXPECT_GT(ff.ffSaturated(), 0) << "the saturation did not report itself";
    // The cap is derived per joint rather than being one global number. It
    // cannot be checked by comparing two joints here -- this harness gives
    // every owned joint the same 0.25 N*m limit, where the shipped config uses
    // 10/10/5 -- so check it against the limit each joint actually has.
    for (int i = 0; i < ff.n_joints_; ++i) {
        const double lim = ff.torqueLimsByIndex[static_cast<std::size_t>(i)];
        if (lim <= 0.0) continue;        // a joint this controller does not own
        EXPECT_NEAR(std::fabs(applied(i)), 0.25 * lim, 1e-12) << "joint " << i;
    }
}

TEST(Feedforward, SlewLimitBoundsHowFastItMoves)
{
    // Contact makes the residual step within a tick or two; handing that step
    // to a motor is a torque discontinuity.
    constexpr double kDt = 0.010;
    constexpr double kSlew = 1.0;                  // N*m/s
    const double kStep = kSlew * kDt;              // 0.01 N*m per tick

    ModelProbe ff(kMjcf, ffParams(1.0, 1.0, kSlew));
    // A target inside the cap, so the slew is what bounds the approach and not
    // the saturation. ff_max_frac is 1.0 here, so the cap is the torque limit.
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.8);
    const double target = d(0);
    ASSERT_GT(target, 12 * kStep) << "the target must be several ticks away";

    double prev = 0.0;
    for (int tick = 0; tick < 10; ++tick) {
        ff.setDisturbanceFeedforward(d, kDt);
        const double now = ff.disturbanceFeedforward()(0);
        EXPECT_LE(now - prev, kStep + 1e-12)
            << "tick " << tick << " moved " << (now - prev) << " N*m";
        prev = now;
    }
    EXPECT_GT(ff.ffSlewLimited(), 0);
    // Ten ticks of 0.01 cannot have arrived -- that is the point.
    EXPECT_LT(prev, target);
    EXPECT_NEAR(prev, 10 * kStep, 1e-9);

    // Given enough ticks it does arrive, and stops there rather than
    // overshooting past the value it was asked for.
    for (int tick = 0; tick < 500; ++tick) ff.setDisturbanceFeedforward(d, kDt);
    EXPECT_NEAR(ff.disturbanceFeedforward()(0), target, 1e-9);
}

TEST(Feedforward, SafeModeEntryClearsIt)
{
    // Safe mode means something already went wrong, and a bad state is exactly
    // what produces a bad estimate. What the reset really protects is recovery:
    // without it the first tracking tick afterwards applies the stale value as
    // a step.
    ModelProbe ff(kMjcf, ffParams(1.0));
    ff.setDisturbanceFeedforward(ffEstimateBelowCap(ff, 0.8), 0.010);
    ASSERT_GT(ff.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0);

    ff.enterDamping();
    EXPECT_EQ(ff.disturbanceFeedforward().cwiseAbs().maxCoeff(), 0.0)
        << "a stale feedforward survived safe-mode entry";
}

TEST(Feedforward, HoldsItsValueWhenNoFreshEstimateArrives)
{
    // A tick with no new estimate keeps the last one rather than dropping to
    // zero, because dropping to zero is itself a torque step. The node relies
    // on this when a mailbox read loses its race.
    ModelProbe ff(kMjcf, ffParams(1.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(d, 0.010);
    const Eigen::VectorXd held = ff.disturbanceFeedforward();

    // No further call: computeControl must still see the retained value.
    const State s = referenceState(6);
    ff.setDesiredPos(0.30, 0.25, 0.0);
    ff.computeControl(s.q, s.v);
    for (int i = 0; i < held.size(); ++i) {
        EXPECT_DOUBLE_EQ(ff.disturbanceFeedforward()(i), held(i));
    }
}

TEST(Feedforward, NonFiniteEstimateIsIgnoredPerJoint)
{
    // A NaN must not reach an actuator, and must not poison the retained value
    // either: that joint holds what it had.
    ModelProbe ff(kMjcf, ffParams(1.0));
    Eigen::VectorXd good = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(good, 0.010);
    const double before = ff.disturbanceFeedforward()(0);
    ASSERT_NE(before, 0.0);

    Eigen::VectorXd poisoned = good;
    poisoned(0) = std::nan("");
    poisoned(1) = std::numeric_limits<double>::infinity();
    ff.setDisturbanceFeedforward(poisoned, 0.010);

    EXPECT_TRUE(ff.disturbanceFeedforward().allFinite());
    EXPECT_DOUBLE_EQ(ff.disturbanceFeedforward()(0), before)
        << "a NaN overwrote a good value instead of being skipped";
}

TEST(Feedforward, AppliedTorqueStillRespectsTheLimit)
{
    // The bounds shift exists so a torque limit keeps meaning what it says. A
    // large estimate plus an aggressive setpoint must still produce a command
    // inside the per-joint limits.
    const State s = referenceState(6);
    ModelProbe ff(kMjcf, ffParams(1.0, 0.5));
    ff.setDesiredPos(1.20, 0.80, 0.0);          // far away, so the law pushes hard
    ff.setDisturbanceFeedforward(
        Eigen::VectorXd::Constant(ff.n_joints_, 50.0), 0.010);

    for (int tick = 0; tick < 20; ++tick) {
        const Eigen::VectorXd tau = ff.computeControl(s.q, s.v);
        for (int i = 0; i < tau.size(); ++i) {
            EXPECT_LE(std::fabs(tau(i)),
                      ff.torqueLimsByIndex[static_cast<std::size_t>(i)] + 1e-9)
                << "tick " << tick << " joint " << i;
        }
    }
}

TEST(Feedforward, RejectsBadConfiguration)
{
    auto bad = [](const NMPCParams& p) {
        EXPECT_THROW((ModelProbe(kMjcf, p)), std::runtime_error);
    };
    bad(ffParams(-0.1));                  // negative gain
    bad(ffParams(1.5));                   // gain above full compensation
    bad(ffParams(0.5, 0.0));              // an unbounded feedforward
    bad(ffParams(0.5, 1.5));              // a cap above the torque limit
    bad(ffParams(0.5, 0.3, -1.0));        // negative slew
}

TEST(Feedforward, RejectsAMisSizedEstimate)
{
    ModelProbe ff(kMjcf, ffParams(1.0));
    EXPECT_THROW(ff.setDisturbanceFeedforward(
                     Eigen::VectorXd::Zero(ff.n_joints_ + 1), 0.010),
                 std::runtime_error);
}

TEST(RealtimeProfile, FeedforwardHandoffDoesNotAllocate)
{
    // It runs on the control thread, once per tick, before the solve.
    ModelProbe ff(kMjcf, ffParams(1.0, 0.3, 5.0));
    const Eigen::VectorXd d = ffEstimateBelowCap(ff, 0.6);
    ff.setDisturbanceFeedforward(d, 0.010);      // first call, sizing already done

    long allocs = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 50; ++i) ff.setDisturbanceFeedforward(d, 0.010);
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

}  // namespace
}  // namespace pat_arm_control
