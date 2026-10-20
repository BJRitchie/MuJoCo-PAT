/*! Tests for the teleoperated reference.
 *
 *  The operator's keys become this class's command, and its windows become
 *  what the NMPC tracks, so the properties checked are the ones a person
 *  driving the arm relies on:
 *
 *    - a held key ramps to a capped speed and a released key brings the target
 *      to rest, both at the acceleration limit;
 *    - the target never leaves its box or yaw range, arrives at a wall at rest,
 *      and can still slide along it;
 *    - a window is exactly what stepping would produce if the latest command
 *      were held, and its past part is the recorded history: nothing about an
 *      earlier command survives except the state it left behind;
 *    - homing returns exactly to home and yields to a new command;
 *    - garbage in (NaN, time going backwards, a stalled caller) is harmless;
 *    - nothing allocates once constructed.
 */
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "pat_arm_control/reference/teleop_keys.h"
#include "pat_arm_control/reference/teleop_reference.h"
#include "pat_rt/testing/alloc_counter.h"

using pat_arm_control::Anchor;
using pat_arm_control::ArmKeys;
using pat_arm_control::PanelKeys;
using pat_arm_control::ReferenceWindow;
using pat_arm_control::TeleopPanel;
using pat_arm_control::RefPoint;
using pat_arm_control::TeleopCommand;
using pat_arm_control::TeleopReference;

namespace {

const Anchor kHome{-0.123, -0.418, 0.3};
constexpr double kDt = 0.01;

TeleopReference::Params params() { return TeleopReference::Params{}; }

//! Step `r` from its current time for `duration` seconds in kDt steps.
void drive(TeleopReference& r, const TeleopCommand& cmd, double duration) {
    const double t0 = r.current().t;
    const int n = static_cast<int>(std::lround(duration / kDt));
    for (int i = 1; i <= n; ++i) r.step(t0 + kDt * i, cmd);
}

}  // namespace

// ============================== rates ===================================== //

TEST(Teleop, AHeldKeyRampsAtTheAccelLimitThenCruises) {
    const auto p = params();
    TeleopReference r(kHome, p);
    const TeleopCommand fwd{0.05, 0.0, 0.0};

    drive(r, fwd, 0.1);
    EXPECT_NEAR(r.current().vx, p.max_lin_accel * 0.1, 1e-12);
    EXPECT_NEAR(r.current().ax, p.max_lin_accel, 1e-9);

    // Reaches 0.05 m/s in exactly speed / accel = 0.2 s, covering a*T^2/2.
    drive(r, fwd, 0.1);
    EXPECT_NEAR(r.current().vx, 0.05, 1e-12);
    EXPECT_NEAR(r.current().x - kHome.x, 0.5 * p.max_lin_accel * 0.2 * 0.2, 1e-12);

    // Then constant velocity, no acceleration, nothing off-axis.
    drive(r, fwd, 0.6);
    EXPECT_NEAR(r.current().vx, 0.05, 1e-12);
    EXPECT_NEAR(r.current().ax, 0.0, 1e-12);
    EXPECT_NEAR(r.current().x - kHome.x, 0.005 + 0.05 * 0.6, 1e-12);
    EXPECT_EQ(r.current().y, kHome.y);
    EXPECT_EQ(r.current().yaw, kHome.yaw);
}

TEST(Teleop, AReleasedKeyStopsAtTheAccelLimitAndStays) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {0.05, 0.0, 0.0}, 0.5);
    const double x_release = r.current().x;

    drive(r, {}, 0.2);   // 0.05 / 0.25 = 0.2 s to stop
    EXPECT_NEAR(r.current().vx, 0.0, 1e-12);
    EXPECT_NEAR(r.current().x - x_release, 0.05 * 0.2 / 2.0, 1e-12);

    const RefPoint stopped = r.current();
    drive(r, {}, 1.0);
    EXPECT_EQ(r.current().x, stopped.x);
    EXPECT_EQ(r.current().vx, 0.0);
}

TEST(Teleop, ADiagonalIsNoFasterThanAStraightLine) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {1.0, 1.0, 0.0}, 0.5);   // far above the cap, both axes; 0.4 s ramp
    EXPECT_NEAR(std::hypot(r.current().vx, r.current().vy), p.max_lin_speed, 1e-12);
    EXPECT_NEAR(r.current().vx, r.current().vy, 1e-12);
}

TEST(Teleop, YawRateIsCappedAndRamped) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {0.0, 0.0, 10.0}, 0.1);
    EXPECT_NEAR(r.current().wz, p.max_yaw_accel * 0.1, 1e-12);
    drive(r, {0.0, 0.0, 10.0}, 0.2);
    EXPECT_NEAR(r.current().wz, p.max_yaw_rate, 1e-12);
}

// ============================== bounds ==================================== //

TEST(Teleop, NeverLeavesTheBoxAndArrivesAtTheWallAtRest) {
    const auto p = params();
    TeleopReference r(kHome, p);
    double prev_vx = 0.0;
    for (int i = 1; i <= 600; ++i) {   // 6 s at full speed: far past the wall
        r.step(kDt * i, {1.0, 0.0, 0.0});
        const RefPoint& c = r.current();
        ASSERT_LE(c.x - kHome.x, p.dx_max + 1e-12) << "step " << i;
        // Braking for the wall is never harder than the limit.
        ASSERT_LE(std::abs(c.vx - prev_vx), p.max_lin_accel * kDt + 1e-12) << "step " << i;
        prev_vx = c.vx;
    }
    EXPECT_NEAR(r.current().x - kHome.x, p.dx_max, 1e-9);
    EXPECT_NEAR(r.current().vx, 0.0, 1e-9);
}

TEST(Teleop, PushingIntoAWallStillSlidesAlongIt) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {-0.05, 0.0, 0.0}, 1.0);   // home sits on dx_min = 0: pinned there
    EXPECT_EQ(r.current().x, kHome.x + p.dx_min);
    EXPECT_EQ(r.current().vx, 0.0);

    drive(r, {-0.05, 0.05, 0.0}, 1.0);
    EXPECT_EQ(r.current().x, kHome.x + p.dx_min);
    EXPECT_GT(r.current().y, kHome.y + 0.02);
    EXPECT_GT(r.current().vy, 0.0);
}

TEST(Teleop, EveryAxisHasItsWalls) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {0.0, -1.0, 0.0}, 5.0);
    EXPECT_NEAR(r.current().y - kHome.y, p.dy_min, 1e-9);
    drive(r, {0.0, 1.0, 0.0}, 8.0);
    EXPECT_NEAR(r.current().y - kHome.y, p.dy_max, 1e-9);
    drive(r, {0.0, 0.0, 1.0}, 5.0);
    EXPECT_NEAR(r.current().yaw - kHome.yaw, p.yaw_max, 1e-9);
    EXPECT_NEAR(r.current().wz, 0.0, 1e-9);
    drive(r, {0.0, 0.0, -1.0}, 5.0);
    EXPECT_NEAR(r.current().yaw - kHome.yaw, -p.yaw_max, 1e-9);
}

TEST(Teleop, TheYawRangeHoldsAcrossTheWrap) {
    // Home near +pi: the range straddles the +/-pi seam.
    TeleopReference r({0.0, 0.0, 3.0}, params());
    drive(r, {0.0, 0.0, 1.0}, 5.0);
    EXPECT_NEAR(std::remainder(r.current().yaw - 3.0, 2.0 * M_PI), params().yaw_max, 1e-9);
    EXPECT_LT(r.current().yaw, 0.0);   // wrapped past +pi
}

// ============================ prediction ================================== //

TEST(Teleop, AWindowIsExactlyWhatSteppingWouldDoWithTheCommandHeld) {
    TeleopReference r(kHome, params());
    drive(r, {0.04, 0.03, 0.2}, 0.5);
    const TeleopCommand next{-0.02, 0.06, -0.4};

    ReferenceWindow w;
    ASSERT_TRUE(r.predict(r.current().t, 1.5, kDt, next, w));
    ASSERT_EQ(w.n, 151u);

    TeleopReference stepped = r;
    for (std::uint32_t i = 0; i < w.n; ++i) {
        if (i > 0) stepped.step(w.p[i].t, next);
        const RefPoint& a = w.p[i];
        const RefPoint& b = stepped.current();
        ASSERT_NEAR(a.x, b.x, 1e-12) << i;
        ASSERT_NEAR(a.y, b.y, 1e-12) << i;
        ASSERT_NEAR(a.yaw, b.yaw, 1e-12) << i;
        ASSERT_NEAR(a.vx, b.vx, 1e-12) << i;
        ASSERT_NEAR(a.vy, b.vy, 1e-12) << i;
        ASSERT_NEAR(a.wz, b.wz, 1e-12) << i;
    }
}

TEST(Teleop, PredictingNeverMovesTheTarget) {
    TeleopReference r(kHome, params());
    drive(r, {0.05, 0.0, 0.0}, 0.3);
    const RefPoint before = r.current();
    ReferenceWindow w;
    r.predict(before.t - 0.1, 2.0, kDt, {0.0, 0.1, 0.5}, w);
    EXPECT_EQ(r.current().x, before.x);
    EXPECT_EQ(r.current().vx, before.vx);
    EXPECT_EQ(r.current().t, before.t);
}

TEST(Teleop, AReleaseShowsUpInTheNextWindowAsAStop) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {0.05, 0.0, 0.0}, 0.5);

    ReferenceWindow w;
    r.predict(r.current().t, 1.0, kDt, {}, w);
    // Decelerates at the limit and is at rest by 0.2 s; holds from there.
    EXPECT_NEAR(w.p[10].vx, 0.025, 1e-12);
    EXPECT_NEAR(w.p[20].vx, 0.0, 1e-12);
    EXPECT_EQ(w.p[100].x, w.p[20].x);
}

TEST(Teleop, OnlyTheLatestCommandShapesTheFuture) {
    // Same state, two commands: the windows share every point up to now
    // (history) and differ only after it. Nothing of an earlier command is
    // remembered beyond the state it produced.
    TeleopReference r(kHome, params());
    drive(r, {0.05, 0.0, 0.0}, 0.5);
    const double now = r.current().t;

    ReferenceWindow a, b;
    r.predict(now - 0.1, 1.0, kDt, {0.05, 0.0, 0.0}, a);
    r.predict(now - 0.1, 1.0, kDt, {0.0, -0.05, 0.0}, b);
    ASSERT_EQ(a.n, b.n);
    for (std::uint32_t i = 0; i < a.n; ++i) {
        if (a.p[i].t <= now) {
            EXPECT_EQ(a.p[i].x, b.p[i].x) << i;
            EXPECT_EQ(a.p[i].y, b.p[i].y) << i;
        }
    }
    EXPECT_GT(a.p[a.n - 1].x, b.p[b.n - 1].x);
    EXPECT_LT(b.p[b.n - 1].y, a.p[a.n - 1].y);
}

TEST(Teleop, TheLookbackIsTheRecordedHistory) {
    TeleopReference r(kHome, params());
    drive(r, {0.05, 0.0, 0.0}, 0.3);
    const RefPoint then = r.current();   // t = 0.3, mid-cruise
    drive(r, {0.0, 0.05, 0.0}, 0.1);     // turned since

    ReferenceWindow w;
    r.predict(then.t, 0.5, kDt, {}, w);
    EXPECT_NEAR(w.p[0].t, then.t, 1e-12);
    EXPECT_NEAR(w.p[0].x, then.x, 1e-12);
    EXPECT_NEAR(w.p[0].y, then.y, 1e-12);
    EXPECT_NEAR(w.p[0].vx, then.vx, 1e-12);

    // Older than the history keeps: the oldest point, held.
    r.predict(r.current().t - 5.0, 0.0, kDt, {}, w);
    EXPECT_EQ(w.n, 1u);
    EXPECT_TRUE(std::isfinite(w.p[0].x));
    EXPECT_GE(w.p[0].x, kHome.x);
}

TEST(Teleop, WindowVelocityAndAccelerationMatchThePosition) {
    TeleopReference r(kHome, params());
    drive(r, {0.03, 0.02, 0.1}, 0.15);   // mid-ramp, so acceleration is live
    ReferenceWindow w;
    // A turn in every axis, kept clear of the walls (a wall clamp is not smooth).
    r.predict(r.current().t, 1.0, kDt, {0.04, 0.05, -0.2}, w);
    for (std::uint32_t i = 0; i + 1 < w.n; ++i) {
        const RefPoint &a = w.p[i], &b = w.p[i + 1];
        // Constant acceleration over each sample: the mean velocity moves the
        // position, and the next point's acceleration moves the velocity.
        EXPECT_NEAR((b.x - a.x) / kDt, 0.5 * (a.vx + b.vx), 1e-9) << i;
        EXPECT_NEAR((b.y - a.y) / kDt, 0.5 * (a.vy + b.vy), 1e-9) << i;
        EXPECT_NEAR((b.vx - a.vx) / kDt, b.ax, 1e-9) << i;
        EXPECT_NEAR((b.vy - a.vy) / kDt, b.ay, 1e-9) << i;
        EXPECT_NEAR((b.wz - a.wz) / kDt, b.az, 1e-9) << i;
        EXPECT_GT(b.t, a.t);
    }
}

TEST(Teleop, TheWindowIsCappedAtCapacityAndSaysSo) {
    TeleopReference r(kHome, params());
    ReferenceWindow w;
    EXPECT_FALSE(r.predict(0.0, 10.0, kDt, {}, w));
    EXPECT_EQ(w.n, ReferenceWindow::kMaxPoints);
}

// ============================== homing ==================================== //

TEST(Teleop, HomingReturnsExactlyHomeWithinTheLimits) {
    const auto p = params();
    TeleopReference r(kHome, p);
    drive(r, {0.08, 0.06, 0.4}, 1.5);
    ASSERT_GT(std::hypot(r.current().x - kHome.x, r.current().y - kHome.y), 0.05);

    r.goHome();
    double prev_vx = r.current().vx, prev_vy = r.current().vy;
    for (int i = 0; i < 1000 && r.homing(); ++i) {
        r.step(r.current().t + kDt, {});
        const RefPoint& c = r.current();
        ASSERT_LE(std::hypot(c.vx - prev_vx, c.vy - prev_vy), p.max_lin_accel * kDt + 1e-9);
        ASSERT_LE(std::hypot(c.vx, c.vy), p.max_lin_speed + 1e-12);
        prev_vx = c.vx;
        prev_vy = c.vy;
    }
    EXPECT_FALSE(r.homing());
    EXPECT_EQ(r.current().x, kHome.x);
    EXPECT_EQ(r.current().y, kHome.y);
    EXPECT_EQ(r.current().yaw, kHome.yaw);
    EXPECT_EQ(r.current().vx, 0.0);
    EXPECT_EQ(r.current().wz, 0.0);
}

TEST(Teleop, ANewCommandTakesOverFromHoming) {
    TeleopReference r(kHome, params());
    drive(r, {0.05, 0.0, 0.0}, 2.0);
    r.goHome();
    drive(r, {}, 0.5);
    EXPECT_TRUE(r.homing());
    EXPECT_LT(r.current().vx, 0.0);   // heading home

    r.step(r.current().t + kDt, {0.0, 0.05, 0.0});
    EXPECT_FALSE(r.homing());
}

TEST(Teleop, AWindowPredictsTheWayHome) {
    TeleopReference r(kHome, params());
    drive(r, {0.08, 0.0, 0.0}, 2.0);
    r.goHome();
    ReferenceWindow w;
    r.predict(r.current().t, 2.5, kDt, {}, w);
    EXPECT_LT(w.p[w.n - 1].x, r.current().x - 0.05);
    EXPECT_TRUE(r.homing());   // still, after predicting
}

// =========================== garbage in =================================== //

TEST(Teleop, NonFiniteCommandsCountAsZero) {
    TeleopReference r(kHome, params());
    drive(r, {std::nan(""), INFINITY, -INFINITY}, 1.0);
    EXPECT_EQ(r.current().x, kHome.x);
    EXPECT_EQ(r.current().y, kHome.y);
    EXPECT_EQ(r.current().yaw, kHome.yaw);
}

TEST(Teleop, TimeGoingBackwardsIsIgnoredAndAStallIsCapped) {
    const auto p = params();
    TeleopReference r(kHome, p);
    r.reset(10.0);
    r.step(9.0, {0.05, 0.0, 0.0});
    EXPECT_EQ(r.current().t, 10.0);
    EXPECT_EQ(r.current().x, kHome.x);
    r.step(std::nan(""), {0.05, 0.0, 0.0});
    EXPECT_EQ(r.current().x, kHome.x);

    // A 60 s gap advances only 1 s: 0.2 s of ramp, then 0.8 s at 0.05 m/s.
    r.step(70.0, {0.05, 0.0, 0.0});
    EXPECT_EQ(r.current().t, 70.0);
    EXPECT_NEAR(r.current().x - kHome.x, 0.005 + 0.05 * 0.8, 1e-9);
}

TEST(Teleop, ResetPutsItHomeAtRest) {
    TeleopReference r(kHome, params());
    drive(r, {0.05, 0.05, 0.3}, 1.0);
    r.goHome();
    r.reset(42.0);
    EXPECT_EQ(r.current().t, 42.0);
    EXPECT_EQ(r.current().x, kHome.x);
    EXPECT_EQ(r.current().vx, 0.0);
    EXPECT_FALSE(r.homing());
}

TEST(Teleop, RejectsNonsenseParameters) {
    auto bad = [](auto mutate) {
        auto p = params();
        mutate(p);
        return [p] { TeleopReference r(kHome, p); };
    };
    EXPECT_THROW(bad([](auto& p) { p.max_lin_speed = 0.0; })(), std::invalid_argument);
    EXPECT_THROW(bad([](auto& p) { p.max_yaw_accel = -1.0; })(), std::invalid_argument);
    EXPECT_THROW(bad([](auto& p) { p.dx_min = 0.01; })(), std::invalid_argument);
    EXPECT_THROW(bad([](auto& p) { p.dy_max = -0.01; })(), std::invalid_argument);
    EXPECT_THROW(bad([](auto& p) { p.yaw_max = 4.0; })(), std::invalid_argument);
    EXPECT_THROW(bad([](auto& p) { p.history_s = -1.0; })(), std::invalid_argument);
    TeleopReference r(kHome, params());
    ReferenceWindow w;
    EXPECT_THROW(r.predict(0.0, 1.0, 0.0, {}, w), std::invalid_argument);
}

// ============================ realtime ==================================== //

TEST(Teleop, SteppingAndPredictingAllocateNothing) {
    TeleopReference r(kHome, params());
    ReferenceWindow w;
    long allocs = 0;
    {
        pat_rt::testing::AllocScope scope;
        for (int i = 1; i <= 200; ++i) {   // long enough to cycle the history
            r.step(kDt * i, {0.05, 0.02, 0.1});
            r.predict(kDt * i - 0.1, 2.0, kDt, {0.05, 0.02, 0.1}, w);
        }
        r.goHome();
        for (int i = 201; i <= 400; ++i) r.step(kDt * i, {});
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

// ============================== keys ====================================== //

TEST(TeleopKeys, EachKeyDrivesItsWorldAxis) {
    auto cmd = [](auto set) {
        ArmKeys k;
        set(k);
        return pat_arm_control::commandFromKeys(k, 0.05, 0.3);
    };
    EXPECT_EQ(cmd([](ArmKeys& k) { k.plus_x = true; }).vx, 0.05);
    EXPECT_EQ(cmd([](ArmKeys& k) { k.minus_x = true; }).vx, -0.05);
    EXPECT_EQ(cmd([](ArmKeys& k) { k.plus_y = true; }).vy, 0.05);
    EXPECT_EQ(cmd([](ArmKeys& k) { k.minus_y = true; }).vy, -0.05);
    EXPECT_EQ(cmd([](ArmKeys& k) { k.ccw = true; }).wz, 0.3);
    EXPECT_EQ(cmd([](ArmKeys& k) { k.cw = true; }).wz, -0.3);
    EXPECT_TRUE(cmd([](ArmKeys&) {}).isZero());
}

TEST(TeleopKeys, OppositesCancelAndDiagonalsKeepTheSpeed) {
    ArmKeys k;
    k.plus_x = k.minus_x = true;
    k.ccw = k.cw = true;
    EXPECT_TRUE(pat_arm_control::commandFromKeys(k, 0.05, 0.3).isZero());

    ArmKeys d;
    d.plus_x = d.minus_y = d.ccw = true;
    const TeleopCommand c = pat_arm_control::commandFromKeys(d, 0.05, 0.3);
    EXPECT_NEAR(std::hypot(c.vx, c.vy), 0.05, 1e-15);
    EXPECT_GT(c.vx, 0.0);
    EXPECT_LT(c.vy, 0.0);
    EXPECT_NEAR(c.vx, -c.vy, 1e-15);
    EXPECT_EQ(c.wz, 0.3);   // rotation is independent of translation
}

TEST(TeleopPanel, StartsPausedAndTogglesOnThePressOnly) {
    TeleopPanel panel;
    ArmKeys fwd;
    fwd.plus_x = true;
    EXPECT_FALSE(panel.active());
    EXPECT_TRUE(panel.command(fwd, 0.05, 0.3).isZero());

    PanelKeys tab;
    tab.toggle = true;
    panel.update(tab);
    EXPECT_TRUE(panel.active());
    for (int i = 0; i < 10; ++i) panel.update(tab);   // held: no flicker
    EXPECT_TRUE(panel.active());
    EXPECT_EQ(panel.command(fwd, 0.05, 0.3).vx, 0.05);

    panel.update({});
    panel.update(tab);   // second press
    EXPECT_FALSE(panel.active());
    EXPECT_TRUE(panel.command(fwd, 0.05, 0.3).isZero());
}

TEST(TeleopPanel, StopZeroesEverythingOnlyWhileHeld) {
    TeleopPanel panel;
    PanelKeys k;
    k.toggle = true;
    panel.update(k);
    ArmKeys fwd;
    fwd.plus_x = true;

    PanelKeys stop;
    stop.stop = true;
    panel.update(stop);
    EXPECT_TRUE(panel.command(fwd, 0.05, 0.3).isZero());
    EXPECT_TRUE(panel.active());
    panel.update({});
    EXPECT_EQ(panel.command(fwd, 0.05, 0.3).vx, 0.05);
}

TEST(TeleopPanel, SpeedScaleStepsPerPressWithinItsLimits) {
    TeleopPanel panel;
    PanelKeys on;
    on.toggle = true;
    panel.update(on);
    panel.update({});
    ArmKeys fwd;
    fwd.plus_x = true;

    PanelKeys faster;
    faster.faster = true;
    panel.update(faster);
    panel.update(faster);   // held: one step only
    EXPECT_DOUBLE_EQ(panel.scale(), 1.25);
    EXPECT_DOUBLE_EQ(panel.command(fwd, 0.04, 0.3).vx, 0.05);

    for (int i = 0; i < 20; ++i) {
        panel.update(faster);
        panel.update({});
    }
    EXPECT_DOUBLE_EQ(panel.scale(), TeleopPanel::kScaleMax);

    PanelKeys slower;
    slower.slower = true;
    for (int i = 0; i < 20; ++i) {
        panel.update(slower);
        panel.update({});
    }
    EXPECT_DOUBLE_EQ(panel.scale(), TeleopPanel::kScaleMin);
}

TEST(TeleopPanel, HomeFiresOnceAndOnlyWhileActiveQuitAlways) {
    TeleopPanel panel;
    PanelKeys home;
    home.home = true;
    EXPECT_FALSE(panel.update(home).home);   // paused: ignored
    panel.update({});

    PanelKeys on;
    on.toggle = true;
    panel.update(on);
    panel.update({});
    EXPECT_TRUE(panel.update(home).home);
    EXPECT_FALSE(panel.update(home).home);   // held: once

    TeleopPanel paused;
    PanelKeys quit;
    quit.quit = true;
    EXPECT_TRUE(paused.update(quit).quit);
}
