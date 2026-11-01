#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>

#include <Eigen/Dense>

#include "pat_arm_control/estimation/joint_kalman_filter.h"
#include "pat_arm_control/estimation/joint_reading_filter.h"
#include "pat_rt/testing/alloc_counter.h"

namespace pat_arm_control {
namespace {

constexpr int kJoints = 6;
constexpr double kH = 0.01;      // the control period, and the joint_states period

//! Seeded noisy readings of six joints moving smoothly.
struct Readings {
    std::mt19937 rng{11};
    std::normal_distribution<double> g{0.0, 1.0};
    double q[kJoints]{}, qd[kJoints]{};
    void at(double t) {
        for (int i = 0; i < kJoints; ++i) {
            const double w = 0.5 + 0.3 * i;
            q[i] = 0.3 * std::sin(w * t) + 5e-4 * g(rng);
            qd[i] = 0.3 * w * std::cos(w * t) + 1e-2 * g(rng);
        }
    }
};

}  // namespace

// The live filter and contact_bench's --joint-filter kalman are the same filter:
// one reading per period through JointReadingFilter matches JointKalmanFilter
// stepped by the control period, which is what the bench does.
TEST(JointReadingFilter, OneReadingPerPeriodIsTheBenchFilter) {
    const JointKalmanParams p;
    JointReadingFilter live(p, kJoints);
    JointKalmanFilter bench(p, kJoints);
    Readings r;
    Eigen::VectorXd zq(kJoints), zv(kJoints);
    for (int k = 0; k < 2000; ++k) {
        // Stamps as a node sees them: seconds, accumulated in double.
        const double stamp = 100.0 + k * kH;
        r.at(k * kH);
        for (int i = 0; i < kJoints; ++i) { zq(i) = r.q[i]; zv(i) = r.qd[i]; }
        ASSERT_TRUE(live.update(stamp, r.q, r.qd));
        bench.update(kH, zq, zv);
        for (int i = 0; i < kJoints; ++i) {
            // The stamp difference differs from kH only in the last bits.
            ASSERT_NEAR(live.position(i), bench.position(i), 1e-12) << "tick " << k;
            ASSERT_NEAR(live.velocity(i), bench.velocity(i), 1e-10) << "tick " << k;
        }
    }
    EXPECT_EQ(live.fused(), 2000u);
    EXPECT_EQ(live.repeats(), 0u);
}

TEST(JointReadingFilter, TheFirstReadingStartsTheFilterAtIt) {
    JointReadingFilter f(JointKalmanParams{}, kJoints);
    EXPECT_FALSE(f.ready());
    Readings r;
    r.at(0.0);
    EXPECT_TRUE(f.update(5.0, r.q, r.qd));
    ASSERT_TRUE(f.ready());
    for (int i = 0; i < kJoints; ++i) {
        EXPECT_DOUBLE_EQ(f.position(i), r.q[i]);
        EXPECT_DOUBLE_EQ(f.velocity(i), r.qd[i]);
    }
}

// The control thread can read one joint_states message twice. Fusing it again
// would count its noise twice; the estimate must not move.
TEST(JointReadingFilter, ARepeatedMessageIsNotFusedAgain) {
    JointReadingFilter f(JointKalmanParams{}, kJoints);
    Readings r;
    r.at(0.0);
    f.update(1.00, r.q, r.qd);
    r.at(kH);
    f.update(1.01, r.q, r.qd);
    const double q0 = f.position(2), v0 = f.velocity(2);
    EXPECT_FALSE(f.update(1.01, r.q, r.qd));
    EXPECT_EQ(f.position(2), q0);
    EXPECT_EQ(f.velocity(2), v0);
    EXPECT_EQ(f.repeats(), 1u);
}

// A missed message: the next one is fused over the time that actually passed,
// exactly as the bench filter stepped twice would be over one 2-period step.
TEST(JointReadingFilter, AMissedMessageStepsByTheRealInterval) {
    const JointKalmanParams p;
    JointReadingFilter live(p, kJoints);
    JointKalmanFilter ref(p, kJoints);
    Readings r;
    Eigen::VectorXd zq(kJoints), zv(kJoints);
    auto feed = [&](double t, double h) {
        r.at(t);
        for (int i = 0; i < kJoints; ++i) { zq(i) = r.q[i]; zv(i) = r.qd[i]; }
        live.update(t, r.q, r.qd);
        ref.update(h, zq, zv);
    };
    feed(0.0, kH);
    feed(0.01, 0.01);
    feed(0.03, 0.02);              // 0.02 s since the last reading fused
    for (int i = 0; i < kJoints; ++i) {
        EXPECT_NEAR(live.position(i), ref.position(i), 1e-12);
        EXPECT_NEAR(live.velocity(i), ref.velocity(i), 1e-10);
    }
}

// A stamp that goes backwards (a restarted simulator) restarts the filter at
// that reading, instead of refusing every message from then on.
TEST(JointReadingFilter, AStampGoingBackwardsRestartsTheFilter) {
    JointReadingFilter f(JointKalmanParams{}, kJoints);
    Readings r;
    for (int k = 0; k < 50; ++k) { r.at(k * kH); f.update(10.0 + k * kH, r.q, r.qd); }
    r.at(0.0);
    EXPECT_TRUE(f.update(0.0, r.q, r.qd));
    EXPECT_EQ(f.restarts(), 1u);
    for (int i = 0; i < kJoints; ++i) EXPECT_DOUBLE_EQ(f.position(i), r.q[i]);
    r.at(kH);
    EXPECT_TRUE(f.update(kH, r.q, r.qd)) << "and carries on from there";
}

TEST(JointReadingFilter, ANonFiniteStampIsIgnored) {
    JointReadingFilter f(JointKalmanParams{}, kJoints);
    Readings r;
    r.at(0.0);
    EXPECT_FALSE(f.update(std::numeric_limits<double>::quiet_NaN(), r.q, r.qd));
    EXPECT_FALSE(f.ready());
}

// It runs on the realtime control thread, which must not allocate.
TEST(JointReadingFilter, UpdateDoesNotAllocate) {
    JointReadingFilter f(JointKalmanParams{}, kJoints);
    Readings r;
    r.at(0.0);
    f.update(0.0, r.q, r.qd);
    {
        // The interposer must be reached at all, or a zero below proves nothing.
        // The pointer escapes through the asm, so the pair cannot be elided.
        pat_rt::testing::AllocScope probe;
        void* m = std::malloc(64);
        asm volatile("" : : "g"(m) : "memory");
        std::free(m);
        ASSERT_GT(probe.n(), 0) << "the malloc interposer is not being reached";
    }
    long allocs = 0;
    {
        pat_rt::testing::AllocScope scope;
        for (int k = 1; k < 100; ++k) {
            f.update(k * kH, r.q, r.qd);        // fused
            f.update(k * kH, r.q, r.qd);        // repeat
        }
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

}  // namespace pat_arm_control
