/*! Tests for the reference window and the reference trajectories.
 *
 *  The NMPC will follow whatever these produce, at every node of its horizon,
 *  so an error here is not a crash but a controller faithfully tracking the
 *  wrong thing. The checks that matter most:
 *
 *    - every published velocity and acceleration is the true derivative of the
 *      published position (checked by finite differences, through the
 *      approach, the circle's ramp and steady state) -- the NMPC feeds them
 *      forward, so a wrong one is a wrong command;
 *    - the circle is continuous everywhere, so it is FEASIBLE;
 *    - the rectangle's velocity jumps at the corners and nowhere else, so its
 *      infeasibility is exactly the intended kind;
 *    - the window interpolates correctly and holds sensibly at its ends.
 */
#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "pat_arm_control/reference/reference_trajectory.h"
#include "pat_arm_control/reference/reference_window.h"

using pat_arm_control::Anchor;
using pat_arm_control::CircleTrajectory;
using pat_arm_control::RectangleTrajectory;
using pat_arm_control::ReferenceTrajectory;
using pat_arm_control::ReferenceWindow;
using pat_arm_control::RefPoint;
using pat_arm_control::TrajectorySpec;

namespace {

// A plausible left-arm home EE pose.
const Anchor kHome{-0.123, -0.418, 0.3};

RefPoint pt(double t, double x, double vx = 0.0, double yaw = 0.0) {
    RefPoint p;
    p.t = t;
    p.x = x;
    p.vx = vx;
    p.yaw = yaw;
    return p;
}

/*! Central-difference check of velocity and acceleration against position at
 *  local time `tau`. Only valid where the trajectory is smooth over +/-h. */
void expectDerivativesConsistent(const ReferenceTrajectory& tr, double tau,
                                 double tol_v, double tol_a) {
    const double h = 1e-4;
    const RefPoint m = tr.at(tau - h), c = tr.at(tau), p = tr.at(tau + h);
    EXPECT_NEAR(c.vx, (p.x - m.x) / (2 * h), tol_v) << "vx at tau=" << tau;
    EXPECT_NEAR(c.vy, (p.y - m.y) / (2 * h), tol_v) << "vy at tau=" << tau;
    EXPECT_NEAR(c.ax, (p.vx - m.vx) / (2 * h), tol_a) << "ax at tau=" << tau;
    EXPECT_NEAR(c.ay, (p.vy - m.vy) / (2 * h), tol_a) << "ay at tau=" << tau;
}

}  // namespace

// ============================ ReferenceWindow ============================= //

TEST(ReferenceWindow, ExactAtSamplesAndLinearBetween) {
    ReferenceWindow w;
    w.p[0] = pt(1.0, 0.0, 2.0);
    w.p[1] = pt(1.1, 0.2, 4.0);
    w.p[2] = pt(1.2, 0.3, 6.0);
    w.n = 3;
    RefPoint r;
    ASSERT_TRUE(w.sample(1.1, r));
    EXPECT_DOUBLE_EQ(r.x, 0.2);
    EXPECT_DOUBLE_EQ(r.vx, 4.0);
    ASSERT_TRUE(w.sample(1.05, r));
    EXPECT_NEAR(r.x, 0.1, 1e-12);
    EXPECT_NEAR(r.vx, 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(r.t, 1.05);
    ASSERT_TRUE(w.sample(1.175, r));
    EXPECT_NEAR(r.x, 0.275, 1e-12);
}

TEST(ReferenceWindow, YawGoesTheShortWayRound) {
    // +3.1 and -3.1 rad are 0.083 rad apart across pi, not 6.2 rad apart
    // through zero. Halfway must be pi, not zero.
    ReferenceWindow w;
    w.p[0] = pt(0.0, 0.0, 0.0, 3.1);
    w.p[1] = pt(1.0, 0.0, 0.0, -3.1);
    w.n = 2;
    RefPoint r;
    ASSERT_TRUE(w.sample(0.5, r));
    EXPECT_NEAR(std::fabs(r.yaw), M_PI, 1e-9);
}

TEST(ReferenceWindow, HoldsTheLastPositionAtRestPastTheEnd) {
    // A trajectory that has run out means "stop there": the position holds and
    // the velocity and acceleration go to zero rather than carrying on.
    ReferenceWindow w;
    w.p[0] = pt(0.0, 0.0, 1.0);
    w.p[1] = pt(0.1, 0.1, 1.0);
    w.p[1].ax = 3.0;
    w.n = 2;
    RefPoint r;
    EXPECT_FALSE(w.sample(0.5, r)) << "outside the window must be reported";
    EXPECT_DOUBLE_EQ(r.x, 0.1);
    EXPECT_DOUBLE_EQ(r.vx, 0.0);
    EXPECT_DOUBLE_EQ(r.ax, 0.0);
    EXPECT_DOUBLE_EQ(r.t, 0.5);
    // Exactly at the last sample is inside, with its own velocity.
    EXPECT_TRUE(w.sample(0.1, r));
    EXPECT_DOUBLE_EQ(r.vx, 1.0);
}

TEST(ReferenceWindow, HoldsTheFirstSampleBeforeTheStart) {
    ReferenceWindow w;
    w.p[0] = pt(2.0, 0.5, 1.0);
    w.p[1] = pt(2.1, 0.6, 1.0);
    w.n = 2;
    RefPoint r;
    EXPECT_FALSE(w.sample(1.0, r));
    EXPECT_DOUBLE_EQ(r.x, 0.5);
    EXPECT_DOUBLE_EQ(r.t, 1.0);
}

TEST(ReferenceWindow, EmptyAndSinglePoint) {
    ReferenceWindow w;
    RefPoint r = pt(9.0, 9.0);
    EXPECT_TRUE(w.empty());
    EXPECT_FALSE(w.sample(0.0, r));
    EXPECT_DOUBLE_EQ(r.x, 9.0) << "an empty window leaves the output untouched";

    w.p[0] = pt(1.0, 0.25, 0.5);
    w.n = 1;
    EXPECT_TRUE(w.sample(1.0, r));
    EXPECT_DOUBLE_EQ(r.x, 0.25);
    EXPECT_FALSE(w.sample(2.0, r));
    EXPECT_DOUBLE_EQ(r.x, 0.25);
    EXPECT_DOUBLE_EQ(r.vx, 0.0);
}

TEST(ReferenceWindow, OrientationComposesTheYawOntoTheReferenceFrame) {
    // The EE frame is rolled relative to the world, so the target orientation
    // is the reference quaternion turned about world Z -- never a bare yaw.
    ReferenceWindow w;
    const double r = std::sqrt(0.5);
    w.quat0[0] = r; w.quat0[1] = r; w.quat0[2] = 0.0; w.quat0[3] = 0.0;   // Rx(90)
    w.yaw0 = 0.4;
    double q[4];
    w.orientation(0.4, q);   // at yaw0: exactly quat0
    for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(q[i], w.quat0[i]);

    // A quarter turn: Rz(90) * Rx(90), computed independently.
    w.orientation(0.4 + M_PI / 2, q);
    const double cz = std::cos(M_PI / 4), sz = std::sin(M_PI / 4);
    const double expect[4] = {cz * r, cz * r, sz * r, sz * r};
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(q[i], expect[i], 1e-12);
    EXPECT_NEAR(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0, 1e-12);
}

// ============================== approach ================================== //

TEST(Approach, StartsAtTheAnchorAndEndsAtTheShapeBothAtRest) {
    CircleTrajectory::Params p;
    CircleTrajectory c(kHome, p);
    double sx, sy;
    c.startPoint(sx, sy);

    const RefPoint a = c.at(0.0);
    EXPECT_DOUBLE_EQ(a.x, kHome.x);
    EXPECT_DOUBLE_EQ(a.y, kHome.y);
    EXPECT_DOUBLE_EQ(a.yaw, kHome.yaw);

    const RefPoint e = c.at(p.approach_s - 1e-9);
    EXPECT_NEAR(e.x, sx, 1e-9);
    EXPECT_NEAR(e.y, sy, 1e-9);
    EXPECT_NEAR(std::hypot(e.vx, e.vy), 0.0, 1e-6);

    // Mid-approach it is moving, and its derivatives are consistent.
    EXPECT_GT(std::hypot(c.at(0.5 * p.approach_s).vx, c.at(0.5 * p.approach_s).vy), 0.0);
    for (double tau : {0.3, 1.0, 1.7, 2.6}) expectDerivativesConsistent(c, tau, 1e-6, 1e-4);
}

// =============================== circle =================================== //

TEST(Circle, EnteredAtTheNearestPointWithAConstantRadius) {
    CircleTrajectory::Params p;
    CircleTrajectory c(kHome, p);
    const double cx = kHome.x + p.center_dx, cy = kHome.y + p.center_dy;
    double sx, sy;
    c.startPoint(sx, sy);
    // Nearest point: on the line from the centre towards the anchor.
    EXPECT_NEAR(std::hypot(sx - cx, sy - cy), p.radius, 1e-12);
    EXPECT_NEAR(std::hypot(kHome.x - sx, kHome.y - sy),
                std::hypot(kHome.x - cx, kHome.y - cy) - p.radius, 1e-12);
    for (double tau = p.approach_s; tau < p.approach_s + 3 * p.period; tau += 0.137) {
        const RefPoint r = c.at(tau);
        EXPECT_NEAR(std::hypot(r.x - cx, r.y - cy), p.radius, 1e-12) << "tau=" << tau;
        EXPECT_DOUBLE_EQ(r.yaw, kHome.yaw);
    }
}

TEST(Circle, DerivativesAreConsistentThroughTheRampAndAfter) {
    CircleTrajectory::Params p;
    CircleTrajectory c(kHome, p);
    const double s0 = p.approach_s;
    for (double s : {0.05, 0.5, 1.0, 1.5, 1.95, 2.5, 5.0, 11.3}) {
        expectDerivativesConsistent(c, s0 + s, 1e-6, 1e-4);
    }
}

TEST(Circle, IsContinuousEverywhereSoItIsFeasible) {
    // No jump in position, velocity or acceleration anywhere -- across the
    // approach/shape join and the ramp/steady join included.
    CircleTrajectory::Params p;
    CircleTrajectory c(kHome, p);
    const double h = 1e-6;
    for (double tau : {p.approach_s, p.approach_s + p.ramp_s}) {
        const RefPoint a = c.at(tau - h), b = c.at(tau + h);
        EXPECT_NEAR(a.x, b.x, 1e-6);
        EXPECT_NEAR(a.vx, b.vx, 1e-5);
        EXPECT_NEAR(a.vy, b.vy, 1e-5);
        EXPECT_NEAR(a.ax, b.ax, 1e-4);
        EXPECT_NEAR(a.ay, b.ay, 1e-4);
    }
}

TEST(Circle, SteadySpeedAndPeriodAndDirection) {
    CircleTrajectory::Params p;
    for (const bool ccw : {true, false}) {
        p.ccw = ccw;
        CircleTrajectory c(kHome, p);
        const double cx = kHome.x + p.center_dx, cy = kHome.y + p.center_dy;
        const double tau = p.approach_s + p.ramp_s + 1.234;
        const RefPoint r = c.at(tau);
        EXPECT_NEAR(std::hypot(r.vx, r.vy), p.radius * 2 * M_PI / p.period, 1e-12);
        const RefPoint later = c.at(tau + p.period);
        EXPECT_NEAR(later.x, r.x, 1e-9);
        EXPECT_NEAR(later.y, r.y, 1e-9);
        // Sign of (r - c) x v: positive is counter-clockwise.
        const double cross = (r.x - cx) * r.vy - (r.y - cy) * r.vx;
        EXPECT_EQ(cross > 0.0, ccw);
    }
}

// ============================== rectangle ================================= //

TEST(Rectangle, VisitsItsCornersExactlyAtConstantSpeed) {
    RectangleTrajectory::Params p;
    RectangleTrajectory r(kHome, p);
    const double s0 = p.approach_s;
    // Which edge comes first depends on the start corner and direction, so the
    // arrival time at each corner is accumulated from the corners themselves.
    double t = 0.0;
    for (int k = 0; k <= 4; ++k) {
        double cx, cy;
        r.corner(k, cx, cy);
        const RefPoint q = r.at(s0 + t);
        EXPECT_NEAR(q.x, cx, 1e-12) << "corner " << k;
        EXPECT_NEAR(q.y, cy, 1e-12) << "corner " << k;
        if (k < 4) {
            double nx, ny;
            r.corner(k + 1, nx, ny);
            t += std::hypot(nx - cx, ny - cy) / p.speed;
        }
    }
    EXPECT_NEAR(t, r.loopPeriod(), 1e-12) << "one lap is the perimeter at speed";
    for (double s = 0.0; s < 2.0 * r.loopPeriod(); s += 0.0731) {
        const RefPoint q = r.at(s0 + s);
        EXPECT_NEAR(std::hypot(q.vx, q.vy), p.speed, 1e-12);
        EXPECT_DOUBLE_EQ(q.ax, 0.0);
        EXPECT_DOUBLE_EQ(q.ay, 0.0);
    }
}

TEST(Rectangle, VelocityJumpsAtTheCornersAndNowhereElse) {
    // The intended infeasibility: an instantaneous change of direction at each
    // corner. Away from the corners the velocity is constant.
    RectangleTrajectory::Params p;
    RectangleTrajectory r(kHome, p);
    const double s0 = p.approach_s;
    double t_corner = 0.0;
    for (int k = 0; k < 4; ++k) {
        double ax, ay, bx, by;
        r.corner(k, ax, ay);
        r.corner(k + 1, bx, by);
        const double edge_t = std::hypot(bx - ax, by - ay) / p.speed;
        // Mid-edge: velocity identical either side.
        const RefPoint m1 = r.at(s0 + t_corner + 0.4 * edge_t);
        const RefPoint m2 = r.at(s0 + t_corner + 0.6 * edge_t);
        EXPECT_DOUBLE_EQ(m1.vx, m2.vx);
        EXPECT_DOUBLE_EQ(m1.vy, m2.vy);
        t_corner += edge_t;
        // At the next corner: a 90 degree turn, i.e. the velocity change has
        // magnitude speed * sqrt(2).
        const RefPoint before = r.at(s0 + t_corner - 1e-6);
        const RefPoint after  = r.at(s0 + t_corner + 1e-6);
        EXPECT_NEAR(std::hypot(after.vx - before.vx, after.vy - before.vy),
                    p.speed * std::sqrt(2.0), 1e-9) << "corner " << k + 1;
        // ...while the position is continuous through it.
        EXPECT_NEAR(after.x, before.x, 1e-6);
        EXPECT_NEAR(after.y, before.y, 1e-6);
    }
}

TEST(Rectangle, ASampleExactlyOnACornerAlreadyHasTheNewDirection) {
    // The jump belongs to the corner itself: a window sample that lands
    // exactly on one must carry the OUTGOING velocity, or the sampled
    // reference turns a whole sample late. Dimensions exact in binary, so the
    // corner time is exact and the test is not at the mercy of rounding.
    RectangleTrajectory::Params p;
    p.center_dx = 0.5;
    p.center_dy = 0.0;
    p.width = 0.25;
    p.height = 0.125;
    p.speed = 0.5;
    p.approach_s = 0.0;
    RectangleTrajectory r(Anchor{0.0, 0.0, 0.0}, p);
    double x0, y0, x1, y1, x2, y2;
    r.corner(0, x0, y0);
    r.corner(1, x1, y1);
    r.corner(2, x2, y2);
    const double t1 = std::hypot(x1 - x0, y1 - y0) / p.speed;   // exactly 0.25 or 0.5
    const RefPoint at_corner = r.at(t1);
    EXPECT_DOUBLE_EQ(at_corner.x, x1);
    EXPECT_DOUBLE_EQ(at_corner.y, y1);
    const double ux = (x2 - x1) / std::hypot(x2 - x1, y2 - y1);
    const double uy = (y2 - y1) / std::hypot(x2 - x1, y2 - y1);
    EXPECT_DOUBLE_EQ(at_corner.vx, p.speed * ux);
    EXPECT_DOUBLE_EQ(at_corner.vy, p.speed * uy);
}

TEST(Rectangle, StartsAtTheNearestCornerAndClosesOnItself) {
    for (const bool ccw : {true, false}) {
        RectangleTrajectory::Params p;
        p.ccw = ccw;
        RectangleTrajectory r(kHome, p);
        double sx, sy;
        r.startPoint(sx, sy);
        const double cx = kHome.x + p.center_dx, cy = kHome.y + p.center_dy;
        // The nearest corner to an anchor at (-0.29, 0) from the centre is a
        // left-hand one (x = cx - w/2).
        EXPECT_NEAR(sx, cx - 0.5 * p.width, 1e-12);
        EXPECT_NEAR(std::fabs(sy - cy), 0.5 * p.height, 1e-12);
        const RefPoint lap = r.at(p.approach_s + r.loopPeriod());
        EXPECT_NEAR(lap.x, sx, 1e-9);
        EXPECT_NEAR(lap.y, sy, 1e-9);
        // Direction: (corner1 - corner0) x (corner2 - corner1) > 0 for CCW.
        double x0, y0, x1, y1, x2, y2;
        r.corner(0, x0, y0);
        r.corner(1, x1, y1);
        r.corner(2, x2, y2);
        const double cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
        EXPECT_EQ(cross > 0.0, ccw);
    }
}

// =========================== sampling, factory ============================ //

TEST(SampleWindow, AbsoluteTimesAtTheRequestedSpacing) {
    CircleTrajectory c(kHome, CircleTrajectory::Params{});
    ReferenceWindow w;
    const double t_start = 100.0, t0 = 105.0;
    ASSERT_TRUE(pat_arm_control::sampleWindow(c, t_start, t0, 2.0, 0.01, w));
    EXPECT_EQ(w.n, 201u);
    EXPECT_DOUBLE_EQ(w.startTime(), 105.0);
    EXPECT_NEAR(w.endTime(), 107.0, 1e-9);
    for (std::uint32_t i = 0; i < w.n; i += 37) {
        const RefPoint direct = c.at(w.p[i].t - t_start);
        EXPECT_DOUBLE_EQ(w.p[i].x, direct.x);
        EXPECT_DOUBLE_EQ(w.p[i].vy, direct.vy);
    }
    // Interpolating the window reproduces the trajectory closely between
    // samples: 10 ms on a 6 cm circle at 8 s is far finer than it bends.
    RefPoint r;
    ASSERT_TRUE(w.sample(106.005, r));
    const RefPoint direct = c.at(106.005 - t_start);
    EXPECT_NEAR(r.x, direct.x, 1e-6);
    EXPECT_NEAR(r.y, direct.y, 1e-6);
}

TEST(SampleWindow, CapsAtCapacityAndSaysSo) {
    RectangleTrajectory r(kHome, RectangleTrajectory::Params{});
    ReferenceWindow w;
    EXPECT_FALSE(pat_arm_control::sampleWindow(r, 0.0, 0.0, 10.0, 0.01, w));
    EXPECT_EQ(w.n, ReferenceWindow::kMaxPoints);
}

TEST(Factory, BuildsBothPatternsAndRejectsNonsense) {
    TrajectorySpec spec;
    spec.pattern = "circle";
    EXPECT_NE(dynamic_cast<CircleTrajectory*>(
                  pat_arm_control::makeTrajectory(spec, kHome).get()), nullptr);
    spec.pattern = "rectangle";
    EXPECT_NE(dynamic_cast<RectangleTrajectory*>(
                  pat_arm_control::makeTrajectory(spec, kHome).get()), nullptr);
    spec.pattern = "triangle";
    EXPECT_THROW(pat_arm_control::makeTrajectory(spec, kHome), std::invalid_argument);

    spec.pattern = "circle";
    spec.circle.radius = 0.0;
    EXPECT_THROW(pat_arm_control::makeTrajectory(spec, kHome), std::invalid_argument);
    spec.circle.radius = 0.05;
    spec.rectangle.speed = -1.0;
    spec.pattern = "rectangle";
    EXPECT_THROW(pat_arm_control::makeTrajectory(spec, kHome), std::invalid_argument);
}
