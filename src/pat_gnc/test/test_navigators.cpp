#include <cmath>
#include <memory>

#include <gtest/gtest.h>
#include "pat_gnc/navigation/direct_navigator.hpp"
#include "pat_gnc/navigation/ekf_navigator.hpp"
#include "pat_gnc/navigation/imu_ekf_navigator.hpp"

TEST(DirectNavigator, StoresPoseAndTwist) {
    pat_gnc::DirectNavigator nav;
    EXPECT_FALSE(nav.ready());
    nav.onOdometry(0.0, {1.0, 2.0, 0.5}, {0.1, 0.2, 0.05});
    EXPECT_TRUE(nav.ready());
    const auto s = nav.state();
    EXPECT_NEAR(s(0), 1.0, 1e-10); EXPECT_NEAR(s(3), 0.1, 1e-10);
    EXPECT_TRUE(nav.covariance().isZero()) << "a pass-through has no uncertainty of its own";
}
TEST(DirectNavigator, ImuIsNoop) {
    pat_gnc::DirectNavigator nav;
    nav.onOdometry(0.0, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    nav.onImu(0.01, {5.0, 5.0}, 99.9);
    EXPECT_NEAR(nav.state()(5), 0.0, 1e-10);
}
TEST(EkfNavigator, MovesStateTowardMeasurement) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    pat_gnc::EkfNavigator nav(p);
    nav.onOdometry(0.0, {1.0, 2.0, 0.5}, {});
    EXPECT_NEAR(nav.state()(0), 1.0, 0.1);
}
TEST(NavigatorInterface, AllImplementINavigator) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    std::unique_ptr<pat_gnc::INavigator> navs[] = {
        std::make_unique<pat_gnc::DirectNavigator>(),
        std::make_unique<pat_gnc::EkfNavigator>(p),
        std::make_unique<pat_gnc::ImuEkfNavigator>(pat_gnc::ImuEkfNavigator::Config{}),
    };
    for (auto& nav : navs) {
        nav->onImu(0.0, {0.0, 0.0}, 0.0);
        nav->onOdometry(0.0, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
        EXPECT_TRUE(nav->ready());
        EXPECT_NEAR(nav->state()(0), 1.0, 0.1);
    }
}

// =========================== ImuEkfNavigator ============================== //

namespace {
using pat_gnc::ImuEkf;
using pat_gnc::ImuEkfNavigator;

//! Stationary body: zero specific force, zero rate, at 500 Hz from `t0`.
void stillImu(ImuEkfNavigator& nav, double t0, double t1) {
    for (double t = t0; t <= t1 + 1e-9; t += 0.002) nav.onImu(t, {0.0, 0.0}, 0.0);
}
}  // namespace

TEST(ImuEkfNavigator, FirstOdometryStartsTheFilterAtTheImuTime) {
    ImuEkfNavigator nav(ImuEkfNavigator::Config{});
    stillImu(nav, 0.0, 1.0);
    EXPECT_FALSE(nav.ready()) << "IMU alone gives no pose to start from";
    nav.onOdometry(0.9, {1.0, 2.0, 0.4}, {});
    EXPECT_TRUE(nav.ready());
    // Started at the latest IMU time (1.0), not at the older fix's stamp, so
    // the next IMU sample is one ordinary interval later rather than a gap.
    EXPECT_NEAR(nav.filter().time(), 1.0, 1e-9);
    nav.onImu(1.002, {0.0, 0.0}, 0.0);
    EXPECT_EQ(nav.filter().diagnostics().imu_gaps, 0u);
    EXPECT_NEAR(nav.state()(0), 1.0, 1e-9);
    EXPECT_NEAR(nav.state()(2), 0.4, 1e-9);
}

TEST(ImuEkfNavigator, LatencyPlacesEachInstrumentAtItsOwnAge) {
    // One message, two instruments of different age. With latency configured
    // the filter replays each into its own step; with none, both are applied
    // at the stamp.
    ImuEkfNavigator::Config cfg;
    cfg.latency_yaw = 0.1;
    cfg.latency_pos = 0.2;
    ImuEkfNavigator nav(cfg);
    nav.onImu(0.0, {0.0, 0.0}, 0.0);
    nav.onOdometry(0.0, {0.0, 0.0, 0.0}, {});
    stillImu(nav, 0.002, 1.0);
    nav.onOdometry(1.0, {0.001, 0.0, 0.001}, {});
    const auto& d = nav.filter().diagnostics();
    EXPECT_EQ(d.replays, 2u);
    // 0.2 s at 500 Hz is exactly 100 steps. The stamp 1.0 - 0.2 sits a
    // rounding error off a step boundary; placement at the NEAREST step lands
    // it on that boundary regardless of which side the rounding went.
    EXPECT_EQ(d.max_replay_steps, 100u);
    EXPECT_EQ(d.pos_updates, 1u);
    EXPECT_EQ(d.yaw_updates, 1u);
}

TEST(ImuEkfNavigator, TwistIsIgnoredUnlessAskedFor) {
    for (const bool use : {false, true}) {
        ImuEkfNavigator::Config cfg;
        cfg.use_twist = use;
        ImuEkfNavigator nav(cfg);
        nav.onImu(0.0, {0.0, 0.0}, 0.0);
        nav.onOdometry(0.0, {0.0, 0.0, 0.0}, {});
        stillImu(nav, 0.002, 0.1);
        nav.onOdometry(0.1, {0.0, 0.0, 0.0}, {0.03, 0.0, 0.0});
        EXPECT_EQ(nav.filter().diagnostics().vel_updates, use ? 1u : 0u);
    }
}

TEST(ImuEkfNavigator, AHeldReadingIsFusedOnceNotOncePerMessage) {
    // A 2 Hz position fix republished on 100 Hz odometry by a sample-and-hold
    // driver: fifty messages carry the same reading. It is one measurement, and
    // fusing it fifty times would shrink the covariance as if it were fifty.
    // Yaw changes every message here, so it is fused every time.
    for (const bool skip : {true, false}) {
        ImuEkfNavigator::Config cfg;
        cfg.skip_repeated = skip;
        ImuEkfNavigator nav(cfg);
        nav.onImu(0.0, {0.0, 0.0}, 0.0);
        nav.onOdometry(0.0, {0.0, 0.0, 0.0}, {});
        for (int k = 1; k <= 100; ++k) {
            const double t = 0.01 * k;
            stillImu(nav, t - 0.008, t);
            const double held_x = (k <= 50) ? 0.001 : 0.002;   // changes once
            nav.onOdometry(t, {held_x, 0.0, 1e-5 * k}, {});
        }
        const auto& d = nav.filter().diagnostics();
        EXPECT_EQ(d.pos_updates, skip ? 2u : 100u);
        EXPECT_EQ(d.yaw_updates, 100u);
    }
}

TEST(ImuEkfNavigator, YawRateIsTheBiasCorrectedGyroWithItsVariance) {
    ImuEkfNavigator nav(ImuEkfNavigator::Config{});
    nav.onImu(0.0, {0.0, 0.0}, 0.0);
    nav.onOdometry(0.0, {0.0, 0.0, 0.0}, {});
    nav.onImu(0.002, {0.0, 0.0}, 0.25);
    EXPECT_NEAR(nav.state()(5), 0.25 - nav.filter().state()(ImuEkf::BG), 1e-12);

    const auto C = nav.covariance();
    const auto& P = nav.filter().covariance();
    EXPECT_TRUE((C.topLeftCorner<5, 5>().isApprox(P.topLeftCorner<5, 5>())));
    EXPECT_DOUBLE_EQ(C(5, 5), nav.filter().yawRateVariance());
    // omega = gyro - b_g, so cov(omega, theta) = -cov(b_g, theta).
    EXPECT_DOUBLE_EQ(C(5, 2), -P(ImuEkf::BG, ImuEkf::TH));
    EXPECT_TRUE(C.isApprox(C.transpose()));
    EXPECT_GT(C.ldlt().vectorD().minCoeff(), 0.0) << "a covariance must be positive definite";
}

TEST(ImuEkfNavigator, TracksAConstantAccelerationFromImuAlone) {
    // 0.1 m/s^2 along body x with the body at yaw pi/2 is +y in the world.
    // Pose fixes every 0.1 s agree with that, so the estimate follows.
    ImuEkfNavigator nav(ImuEkfNavigator::Config{});
    const double a = 0.1, th = M_PI / 2;
    nav.onImu(0.0, {a, 0.0}, 0.0);
    nav.onOdometry(0.0, {0.0, 0.0, th}, {});
    for (int k = 1; k <= 1000; ++k) {
        const double t = 0.002 * k;
        nav.onImu(t, {a, 0.0}, 0.0);
        if (k % 50 == 0) nav.onOdometry(t, {0.0, 0.5 * a * t * t, th}, {});
    }
    const auto s = nav.state();
    EXPECT_NEAR(s(1), 0.5 * a * 4.0, 2e-3);
    EXPECT_NEAR(s(4), a * 2.0, 5e-3);
    EXPECT_NEAR(s(0), 0.0, 2e-3);
}
