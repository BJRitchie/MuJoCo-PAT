/*! The controller following a reference trajectory with a per-node reference.
 *
 *  Following a trajectory means each rollout node is scored against the
 *  reference at its own time. These pin the two properties that matter: a
 *  trajectory that does not move is EXACTLY the static setpoint (so nothing
 *  committed moves), and with preview the controller reacts to a change it can
 *  only know about from the future part of the window. The window and the
 *  shapes themselves are tested in test_reference.cpp.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

#include "pat_arm_control/reference/reference_window.h"
#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

/*! A window of `n` points 10 ms apart from `t0`, position given per sample by
 *  `pos(t)`, velocity and acceleration zero, orientation `quat` held. */
template <typename Pos>
ReferenceWindow makeWindow(double t0, int n, Pos pos,
                           const Eigen::Vector4d& quat = Eigen::Vector4d(1, 0, 0, 0))
{
    ReferenceWindow w;
    for (int i = 0; i < n; ++i) {
        RefPoint& p = w.p[i];
        p.t = t0 + 0.01 * i;
        pos(p.t, p.x, p.y);
    }
    w.n = static_cast<std::uint32_t>(n);
    for (int i = 0; i < 4; ++i) w.quat0[i] = quat(i);
    return w;
}

/*! Aim a controller at where the arm actually is: position AND orientation.
 *  This harness runs 0.25 N*m limits, so a target far from the EE -- and the
 *  default identity orientation is 90 degrees from this EE frame -- pins every
 *  joint at its limit, where any two references give the same clamped
 *  command and a comparison proves nothing. */
ArmNMPC::EePose aimAtCurrentPose(ModelProbe& c, const State& s)
{
    const ArmNMPC::EePose ee = c.currentEePose(s.q, s.v);
    c.setDesiredPos(ee.pos.x(), ee.pos.y(), ee.pos.z());
    c.setDesiredOrient(ee.quat(0), ee.quat(1), ee.quat(2), ee.quat(3));
    return ee;
}

//! True when no joint's command sits within 1% of its torque limit.
bool offTheTorqueLimit(const ModelProbe& c, const Eigen::VectorXd& tau)
{
    for (int i = 0; i < tau.size(); ++i) {
        if (std::fabs(tau(i)) >= 0.99 * c.torqueLimsByIndex[static_cast<std::size_t>(i)]) {
            return false;
        }
    }
    return true;
}

TEST(Reference, AConstantWindowIsExactlyTheStaticSetpoint)
{
    // The regression gate for the whole feature: a window that holds one point
    // must reproduce the setpoint path bit for bit, tick after tick (the warm
    // start carries state between ticks, so several are compared).
    const State s = referenceState(6);
    ModelProbe setpoint(kMjcf, liveParams());
    const ArmNMPC::EePose ee = aimAtCurrentPose(setpoint, s);
    // Offset a little from the current pose, so the controller is working.
    const double gx = ee.pos.x() + 0.01, gy = ee.pos.y() - 0.01;
    setpoint.setDesiredPos(gx, gy, ee.pos.z());

    ModelProbe traj(kMjcf, liveParams());
    traj.setReferenceWindow(makeWindow(0.0, 200, [gx, gy](double, double& x, double& y) {
        x = gx;
        y = gy;
    }, ee.quat));
    for (int tick = 0; tick < 8; ++tick) {
        traj.setReferenceTime(0.2 + 0.01 * tick);
        const Eigen::VectorXd a = setpoint.computeControl(s.q, s.v);
        const Eigen::VectorXd b = traj.computeControl(s.q, s.v);
        ASSERT_EQ(a.size(), b.size());
        for (int i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a(i), b(i)) << "tick " << tick << " joint " << i;
        }
    }
    EXPECT_EQ(traj.referenceSamplesOutside(), 0);
    EXPECT_EQ(traj.referenceTicks(), 8);
}

TEST(Reference, PreviewReactsToAChangeOnlyTheFutureOfTheWindowShows)
{
    // The reference holds still now and moves 0.6 s from now -- inside the
    // ~1 s horizon, beyond node 0. With preview the controller sees it coming
    // and the command changes. Without preview every node sees "now", where
    // nothing has changed, so the command is exactly the static setpoint's.
    const State s = referenceState(6);
    const double t_now = 0.5;

    ModelProbe setpoint(kMjcf, liveParams());
    const ArmNMPC::EePose ee = aimAtCurrentPose(setpoint, s);
    const double x0 = ee.pos.x(), y0 = ee.pos.y();
    const ReferenceWindow w = makeWindow(0.0, 200, [=](double t, double& x, double& y) {
        x = (t >= t_now + 0.6) ? x0 + 0.02 : x0;
        y = y0;
    }, ee.quat);
    const Eigen::VectorXd tau_static = setpoint.computeControl(s.q, s.v);
    ASSERT_TRUE(offTheTorqueLimit(setpoint, tau_static))
        << "a saturated command cannot show the difference this test looks for";

    NMPCParams no_preview = liveParams();
    no_preview.reference_preview = false;
    ModelProbe reactive(kMjcf, no_preview);
    reactive.setReferenceWindow(w);
    reactive.setReferenceTime(t_now);
    const Eigen::VectorXd tau_reactive = reactive.computeControl(s.q, s.v);

    ModelProbe preview(kMjcf, liveParams());
    preview.setReferenceWindow(w);
    preview.setReferenceTime(t_now);
    const Eigen::VectorXd tau_preview = preview.computeControl(s.q, s.v);

    for (int i = 0; i < tau_static.size(); ++i) {
        EXPECT_EQ(tau_reactive(i), tau_static(i)) << "joint " << i;
    }
    EXPECT_GT((tau_preview - tau_static).cwiseAbs().maxCoeff(), 1e-6)
        << "a change 0.6 s ahead must reach the current command";
}

TEST(Reference, NodesBeyondTheWindowAreHeldAndCounted)
{
    // The window ends 0.3 s after t_now. With N = 30 at Ts = 0.033, nodes
    // k = 10..30 fall past it -- 21 per tick -- and are held at the last point.
    const State s = referenceState(6);
    ModelProbe c(kMjcf, liveParams());
    c.setReferenceWindow(makeWindow(1.0, 31, [](double, double& x, double& y) {
        x = 0.30;
        y = 0.25;
    }));
    c.setReferenceTime(1.0);
    (void)c.computeControl(s.q, s.v);
    EXPECT_EQ(c.referenceSamplesOutside(), 21);
}

TEST(Reference, ClearingTheWindowReturnsToTheSetpoint)
{
    const State s = referenceState(6);
    ModelProbe c(kMjcf, liveParams());
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setReferenceWindow(makeWindow(0.0, 200, [](double, double& x, double& y) {
        x = 0.31;
        y = 0.25;
    }));
    EXPECT_TRUE(c.referenceActive());
    c.setReferenceTime(0.5);
    (void)c.computeControl(s.q, s.v);
    c.clearReferenceWindow();
    EXPECT_FALSE(c.referenceActive());
    (void)c.computeControl(s.q, s.v);
    EXPECT_EQ(c.referenceTicks(), 1) << "only the tick with a window counts";

    // An empty window is not a window.
    c.setReferenceWindow(ReferenceWindow{});
    EXPECT_FALSE(c.referenceActive());
}

TEST(RealtimeProfile, FollowingATrajectoryAddsNoAllocation)
{
    // Handing over a window and sampling it at every node must cost nothing:
    // a trajectory tick allocates exactly what a setpoint tick does (the
    // returned torque vector).
    const State s = referenceState(6);
    const ReferenceWindow w = makeWindow(0.0, 200, [](double t, double& x, double& y) {
        x = 0.30 + 0.02 * std::sin(t);
        y = 0.25 + 0.02 * std::cos(t);
    });

    ModelProbe setpoint(kMjcf, liveParams());
    setpoint.setDesiredPos(0.30, 0.25, 0.0);
    ModelProbe traj(kMjcf, liveParams());
    traj.setReferenceWindow(w);
    for (int i = 0; i < 5; ++i) {       // warm both past first-tick sizing
        (void)setpoint.computeControl(s.q, s.v);
        traj.setReferenceTime(0.2 + 0.01 * i);
        (void)traj.computeControl(s.q, s.v);
    }

    long a_set = 0, a_traj = 0, a_handoff = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 20; ++i) (void)setpoint.computeControl(s.q, s.v);
        a_set = scope.n();
    }
    {
        AllocScope scope;
        for (int i = 0; i < 20; ++i) {
            traj.setReferenceTime(0.3 + 0.01 * i);
            (void)traj.computeControl(s.q, s.v);
        }
        a_traj = scope.n();
    }
    {
        AllocScope scope;
        for (int i = 0; i < 20; ++i) traj.setReferenceWindow(w);
        a_handoff = scope.n();
    }
    EXPECT_EQ(a_traj, a_set);
    EXPECT_EQ(a_handoff, 0);
}

}  // namespace
}  // namespace pat_arm_control
