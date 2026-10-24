#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <stdexcept>

#include <Eigen/Dense>

#include "pat_arm_control/estimation/joint_kalman_filter.h"
#include "pat_rt/testing/alloc_counter.h"

namespace pat_arm_control {
namespace {

constexpr double kH = 0.01;      // the control period
constexpr int kTicks = 3000;     // 30 s
constexpr int kSettle = 200;     // ticks excluded from the statistics

struct Errors {
    double pos_rms = 0.0, vel_rms = 0.0;      // filter
    double raw_pos_rms = 0.0, raw_vel_rms = 0.0;
    double vel_mean = 0.0;                    // signed, for bias
    double nees = 0.0;                        // mean, 2 if honest
};

/*! Run one joint along `truth(t) -> (q, qd)` with seeded encoder noise and
 *  score the filter and the raw readings against the truth. */
template <class Truth>
Errors run(const JointKalmanParams& p, Truth truth, unsigned seed = 7) {
    JointKalmanFilter f(p, 1);
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    const double sv = p.vel_meas_std > 0.0 ? p.vel_meas_std : 1e-2;
    Errors e;
    int n = 0;
    for (int k = 0; k < kTicks; ++k) {
        const auto [q, qd] = truth(k * kH);
        Eigen::VectorXd zq(1), zv(1);
        zq << q + p.pos_meas_std * g(rng);
        zv << qd + sv * g(rng);
        f.update(kH, zq, zv);
        if (k < kSettle) continue;
        const Eigen::Vector2d err(f.position(0) - q, f.velocity(0) - qd);
        e.pos_rms += err(0) * err(0);
        e.vel_rms += err(1) * err(1);
        e.raw_pos_rms += (zq(0) - q) * (zq(0) - q);
        e.raw_vel_rms += (zv(0) - qd) * (zv(0) - qd);
        e.vel_mean += err(1);
        e.nees += err.dot(f.covariance(0).inverse() * err);
        ++n;
    }
    e.pos_rms = std::sqrt(e.pos_rms / n);
    e.vel_rms = std::sqrt(e.vel_rms / n);
    e.raw_pos_rms = std::sqrt(e.raw_pos_rms / n);
    e.raw_vel_rms = std::sqrt(e.raw_vel_rms / n);
    e.vel_mean /= n;
    e.nees /= n;
    return e;
}

TEST(JointKalmanFilter, FirstReadingInitialisesTheState) {
    JointKalmanFilter f(JointKalmanParams{}, 2);
    EXPECT_FALSE(f.ready());
    Eigen::VectorXd q(2), qd(2);
    q << 0.3, -1.2;
    qd << 0.05, -0.02;
    f.update(kH, q, qd);
    ASSERT_TRUE(f.ready());
    EXPECT_DOUBLE_EQ(f.position(0), 0.3);
    EXPECT_DOUBLE_EQ(f.position(1), -1.2);
    EXPECT_DOUBLE_EQ(f.velocity(0), 0.05);
    EXPECT_DOUBLE_EQ(f.velocity(1), -0.02);
}

TEST(JointKalmanFilter, ConstantVelocityIsTrackedWithLessNoiseThanTheEncoder) {
    // A joint the model believes changes velocity slowly: the filter averages
    // over a long window, cuts the velocity noise several times over (about 6x
    // here) and never adds position noise.
    JointKalmanParams p;
    p.accel_psd = 1e-4;
    const Errors e = run(p, [](double t) { return std::make_pair(0.2 + 0.3 * t, 0.3); });
    EXPECT_LT(e.vel_rms, e.raw_vel_rms / 3.0);
    EXPECT_LT(e.pos_rms, e.raw_pos_rms);
}

TEST(JointKalmanFilter, AnAcceleratingJointIsFollowedWithoutABias) {
    // A sinusoid with 2 rad/s^2 peak acceleration: a wide enough accel_psd
    // follows it without a standing velocity lag, and still beats the reading.
    JointKalmanParams p;
    p.accel_psd = 1.0;
    const auto sine = [](double t) {
        return std::make_pair(0.5 * std::sin(2.0 * t), 1.0 * std::cos(2.0 * t));
    };
    const Errors e = run(p, sine);
    EXPECT_LT(std::abs(e.vel_mean), 1e-3);
    EXPECT_LT(e.vel_rms, e.raw_vel_rms);
}

TEST(JointKalmanFilter, VelocityIsRecoveredFromPositionsAlone) {
    // No velocity reading: the filter differentiates the positions, through
    // its model, and still lands within a few mrad/s on a steady joint.
    JointKalmanParams p;
    p.vel_meas_std = 0.0;
    p.accel_psd = 1e-4;
    const Errors e = run(p, [](double t) { return std::make_pair(0.3 * t, 0.3); });
    EXPECT_LT(e.vel_rms, 5e-3);
    EXPECT_LT(e.pos_rms, p.pos_meas_std);
}

TEST(JointKalmanFilter, TheCovarianceIsHonestWhenTheModelIsRight) {
    // Truth drawn exactly from the filter's own model (the step noise sampled
    // from Q itself, through its Cholesky factor): the mean NEES must be close
    // to the state dimension, 2.
    JointKalmanParams p;
    p.accel_psd = 0.5;
    Eigen::Matrix2d Q;
    Q << kH * kH * kH / 3.0, kH * kH / 2.0, kH * kH / 2.0, kH;
    Q *= p.accel_psd;
    const Eigen::Matrix2d L = Q.llt().matrixL();
    std::mt19937 rng(11);
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::Vector2d s = Eigen::Vector2d::Zero();
    const Errors e = run(p, [&](double) {
        const auto out = std::make_pair(s(0), s(1));
        s = Eigen::Vector2d(s(0) + kH * s(1), s(1)) + L * Eigen::Vector2d(g(rng), g(rng));
        return out;
    });
    EXPECT_NEAR(e.nees, 2.0, 0.2);
}

TEST(JointKalmanFilter, BadSettingsAreRefused) {
    JointKalmanParams p;
    p.pos_meas_std = 0.0;
    EXPECT_THROW(JointKalmanFilter(p, 1), std::invalid_argument);
    p = JointKalmanParams{};
    p.accel_psd = -1.0;
    EXPECT_THROW(JointKalmanFilter(p, 1), std::invalid_argument);
    p = JointKalmanParams{};
    p.accel_psd = std::nan("");
    EXPECT_THROW(JointKalmanFilter(p, 1), std::invalid_argument);
    EXPECT_THROW(JointKalmanFilter(JointKalmanParams{}, 0), std::invalid_argument);

    JointKalmanFilter f(JointKalmanParams{}, 2);
    Eigen::VectorXd one(1);
    one << 0.0;
    EXPECT_THROW(f.update(kH, one, one), std::invalid_argument);
    Eigen::VectorXd two = Eigen::VectorXd::Zero(2);
    f.update(kH, two, two);
    EXPECT_THROW(f.update(0.0, two, two), std::invalid_argument);
}

TEST(JointKalmanFilter, UpdateDoesNotAllocate) {
    JointKalmanFilter f(JointKalmanParams{}, 3);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(3), qd = Eigen::VectorXd::Zero(3);
    f.update(kH, q, qd);
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
        for (int k = 0; k < 100; ++k) f.update(kH, q, qd);
        allocs = scope.n();
    }
    EXPECT_EQ(allocs, 0);
}

}  // namespace
}  // namespace pat_arm_control
