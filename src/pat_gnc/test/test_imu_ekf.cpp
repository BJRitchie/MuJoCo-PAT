/*! Tests for ImuEkf against a synthetic planar plant.
 *
 *  The plant is a rigid body in the plane driven by a known, smooth body-frame
 *  acceleration and yaw rate, integrated with RK4 at a step far finer than the
 *  IMU's. Its IMU reports the instantaneous true values plus the bias and noise
 *  under test. No MuJoCo, no ROS: the filter's maths is checked against a
 *  trajectory whose truth is known exactly.
 *
 *  What is asserted, and why each matters:
 *    - propagation reproduces the trajectory from a perfect IMU, so the
 *      integration scheme is right before any estimation is involved;
 *    - a gyro bias is learned from yaw fixes and an accelerometer bias from
 *      position fixes, and the accelerometer bias is NOT learned without
 *      position -- observability tested, not assumed;
 *    - the filter is CONSISTENT: its covariance matches its actual error
 *      (NEES within chi-square bounds over a Monte-Carlo set), which is what
 *      makes the covariance it publishes usable by anything downstream;
 *    - a late measurement replays to exactly the on-time result, gross
 *      outliers are gated, and the worst-case replay fits inside one IMU
 *      period, NEES included under realistic sensor latency;
 *    - every call is allocation-free, so it can run on a realtime thread.
 */
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <new>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pat_gnc/navigation/imu_ekf.hpp"

// --- Heap-allocation counter ------------------------------------------------ //
// Replaces the global operator new for this test executable. Counting is off
// except inside an AllocScope, so gtest's own allocations are not counted.
namespace {
std::atomic<bool> g_count_allocs{false};
std::atomic<long> g_allocs{0};
}  // namespace

void* operator new(std::size_t n) {
    if (g_count_allocs.load(std::memory_order_relaxed)) {
        g_allocs.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

struct AllocScope {
    AllocScope() {
        g_allocs.store(0);
        g_count_allocs.store(true);
    }
    ~AllocScope() { g_count_allocs.store(false); }
    long count() const { return g_allocs.load(); }
};

using pat_gnc::ImuEkf;

double wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

/*! The truth: a planar rigid body under a smooth, known excitation, with the
 *  IMU errors it will report. */
struct Plant {
    Eigen::Vector2d p{0.3, -0.2};
    Eigen::Vector2d v{0.0, 0.0};
    double th = 0.4;
    double t = 0.0;

    double gyro_bias = 0.0;
    Eigen::Vector2d accel_bias{0.0, 0.0};
    double gyro_noise = 0.0, accel_noise = 0.0;
    double gyro_walk = 0.0, accel_walk = 0.0;
    std::mt19937_64 rng{1};
    std::normal_distribution<double> n01{0.0, 1.0};

    //! Scales the body acceleration. Attitude error only shows up in position
    //! through the acceleration it misdirects, so the test that needs that
    //! coupling to be strong turns this up.
    double accel_gain = 1.0;

    // Rich enough that attitude and translation both change throughout, so
    // every coupling in the error dynamics is exercised.
    Eigen::Vector2d accelBody(double s) const {
        return accel_gain *
               Eigen::Vector2d(0.2 * std::sin(0.7 * s), 0.15 * std::cos(0.5 * s));
    }
    static double yawRate(double s) { return 0.3 * std::sin(0.4 * s) + 0.05; }

    //! RK4 on (p, v, th) to time t + h, with 8 substeps.
    void advance(double h) {
        // A bias walk accumulates over the whole interval, at the density's
        // rate, independent of how the interval is split for integration.
        gyro_bias += gyro_walk * std::sqrt(h) * n01(rng);
        accel_bias.x() += accel_walk * std::sqrt(h) * n01(rng);
        accel_bias.y() += accel_walk * std::sqrt(h) * n01(rng);

        constexpr int kSub = 8;
        const double s = h / kSub;
        for (int i = 0; i < kSub; ++i) {
            struct D { Eigen::Vector2d dp, dv; double dth; };
            const auto f = [this](double tt, const Eigen::Vector2d& vv, double tht) {
                const double c = std::cos(tht), sn = std::sin(tht);
                const Eigen::Vector2d ab = accelBody(tt);
                return D{vv, {c * ab.x() - sn * ab.y(), sn * ab.x() + c * ab.y()},
                         yawRate(tt)};
            };
            const D k1 = f(t, v, th);
            const D k2 = f(t + s / 2, v + s / 2 * k1.dv, th + s / 2 * k1.dth);
            const D k3 = f(t + s / 2, v + s / 2 * k2.dv, th + s / 2 * k2.dth);
            const D k4 = f(t + s, v + s * k3.dv, th + s * k3.dth);
            p  += s / 6 * (k1.dp + 2 * k2.dp + 2 * k3.dp + k4.dp);
            v  += s / 6 * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv);
            th += s / 6 * (k1.dth + 2 * k2.dth + 2 * k3.dth + k4.dth);
            t  += s;
        }
        th = wrap(th);
    }

    Eigen::Vector2d accelMeas() {
        return accelBody(t) + accel_bias +
               accel_noise * Eigen::Vector2d(n01(rng), n01(rng));
    }
    double gyroMeas() { return yawRate(t) + gyro_bias + gyro_noise * n01(rng); }

    //! The true value of the filter's 8-state, in its layout.
    ImuEkf::Vec truth() const {
        ImuEkf::Vec x;
        x << p.x(), p.y(), th, v.x(), v.y(), gyro_bias, accel_bias.x(),
            accel_bias.y();
        return x;
    }
};

//! Error with the angle wrapped, so +pi/-pi is a small error.
ImuEkf::Vec errorOf(const ImuEkf& f, const Plant& pl) {
    ImuEkf::Vec e = pl.truth() - f.state();
    e(ImuEkf::TH) = wrap(e(ImuEkf::TH));
    return e;
}

/*! Chi-square quantile, Wilson-Hilferty approximation. Accurate to well under
 *  1% for the hundreds of degrees of freedom the consistency test uses. */
double chi2Quantile(double z, double k) {
    const double a = 2.0 / (9.0 * k);
    return k * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

/*! Drive the filter alongside the plant for `seconds`, IMU at `imu_hz`,
 *  yaw fixes at `yaw_hz` and position fixes at `pos_hz` (0 disables). The
 *  fixes are drawn with the filter's own configured noise at the moment they
 *  are TAKEN, and delivered `yaw_lag` / `pos_lag` IMU steps later stamped
 *  with that moment -- a sensor latency, as the star tracker and the
 *  relative-nav sensor have. `each` is called after every IMU step. */
template <typename Each>
void run(ImuEkf& f, Plant& pl, double seconds, double imu_hz, double yaw_hz,
         double pos_hz, Each each, int yaw_lag = 0, int pos_lag = 0) {
    const double dt = 1.0 / imu_hz;
    const int steps = static_cast<int>(std::lround(seconds * imu_hz));
    const int yaw_every = yaw_hz > 0 ? static_cast<int>(std::lround(imu_hz / yaw_hz)) : 0;
    const int pos_every = pos_hz > 0 ? static_cast<int>(std::lround(imu_hz / pos_hz)) : 0;
    const auto& prm = f.params();
    struct Pending {
        int due;
        bool is_yaw;
        double t, yaw;
        Eigen::Vector2d p;
    };
    std::vector<Pending> pending;
    for (int k = 1; k <= steps; ++k) {
        pl.advance(dt);
        f.propagate(pl.t, pl.accelMeas(), pl.gyroMeas());
        if (yaw_every && k % yaw_every == 0) {
            pending.push_back({k + yaw_lag, true, pl.t,
                               wrap(pl.th + prm.yaw_meas_std * pl.n01(pl.rng)), {}});
        }
        if (pos_every && k % pos_every == 0) {
            pending.push_back({k + pos_lag, false, pl.t, 0.0,
                               pl.p + prm.pos_meas_std *
                                          Eigen::Vector2d(pl.n01(pl.rng), pl.n01(pl.rng))});
        }
        // Deliver whatever is due, in the order it was taken.
        for (std::size_t i = 0; i < pending.size();) {
            if (pending[i].due <= k) {
                if (pending[i].is_yaw) f.updateYaw(pending[i].t, pending[i].yaw);
                else                   f.updatePosition(pending[i].t, pending[i].p);
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        each(k);
    }
}

ImuEkf::Params quietParams() {
    ImuEkf::Params p;
    p.gyro_noise_std = 1e-9;
    p.accel_noise_std = 1e-9;
    p.gyro_bias_walk = 1e-9;
    p.accel_bias_walk = 1e-9;
    return p;
}

}  // namespace

// =========================== propagation ================================== //

TEST(ImuEkf, NotReadyUntilInitialisedAndImuBeforeThenIsHarmless) {
    ImuEkf f(quietParams());
    EXPECT_FALSE(f.ready());
    f.propagate(0.0, {1.0, 0.0}, 0.5);
    f.propagate(0.01, {1.0, 0.0}, 0.5);
    EXPECT_FALSE(f.ready());
    EXPECT_TRUE(f.state().isZero()) << "nothing to integrate before a pose fix";
    f.initialise(0.01, 1.0, 2.0, 0.3);
    EXPECT_TRUE(f.ready());
    EXPECT_DOUBLE_EQ(f.state()(ImuEkf::PX), 1.0);
    EXPECT_DOUBLE_EQ(f.state()(ImuEkf::TH), 0.3);
}

TEST(ImuEkf, PerfectImuReproducesTheTrajectory) {
    // No noise, no bias, exact initial state, no updates at all: what is left
    // is the integration scheme. Trapezoidal input and midpoint rotation are
    // second order, so over 20 s of continuous turning at 500 Hz the drift is
    // sub-millimetre. Taking only the end sample (first order) fails this:
    // 3.3 mm of position and 3e-4 rad of yaw.
    Plant pl;
    ImuEkf f(quietParams());
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
    run(f, pl, 20.0, 500.0, 0.0, 0.0, [](int) {});

    const ImuEkf::Vec e = errorOf(f, pl);
    EXPECT_LT(e.segment<2>(ImuEkf::PX).norm(), 1e-3);
    EXPECT_LT(e.segment<2>(ImuEkf::VX).norm(), 1e-4);
    EXPECT_LT(std::fabs(e(ImuEkf::TH)), 1e-6);
}

TEST(ImuEkf, CovarianceGrowsWithoutFixesAndShrinksWithThem) {
    ImuEkf::Params prm;
    Plant pl;
    ImuEkf f(prm);
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
    const double p0 = f.covariance()(ImuEkf::PX, ImuEkf::PX);
    run(f, pl, 2.0, 500.0, 0.0, 0.0, [](int) {});
    const double p_grown = f.covariance()(ImuEkf::PX, ImuEkf::PX);
    EXPECT_GT(p_grown, p0);
    f.updatePosition(pl.t, pl.p);
    EXPECT_LT(f.covariance()(ImuEkf::PX, ImuEkf::PX), p_grown);
}

TEST(ImuEkf, RepeatedAndLongGapSamplesAreRefusedAndCounted) {
    ImuEkf::Params prm = quietParams();
    prm.max_imu_gap = 0.05;
    ImuEkf f(prm);
    f.initialise(0.0, 0.0, 0.0, 0.0);
    f.propagate(0.01, {1.0, 0.0}, 0.0);
    const ImuEkf::Vec before = f.state();
    f.propagate(0.01, {1.0, 0.0}, 0.0);       // repeated timestamp
    f.propagate(0.005, {1.0, 0.0}, 0.0);      // backwards
    EXPECT_EQ(f.diagnostics().imu_out_of_order, 2u);
    EXPECT_TRUE(f.state().isApprox(before));
    f.propagate(0.5, {1.0, 0.0}, 0.0);        // 0.49 s gap
    EXPECT_EQ(f.diagnostics().imu_gaps, 1u);
    EXPECT_TRUE(f.state().isApprox(before)) << "a gap must not be bridged";
}

// ============================= updates ==================================== //

TEST(ImuEkf, YawInnovationWrapsAcrossPi) {
    ImuEkf::Params prm = quietParams();
    prm.init_yaw_std = 0.1;
    ImuEkf f(prm);
    f.initialise(0.0, 0.0, 0.0, M_PI - 0.01);
    // The measurement is 0.02 rad away across the cut. Unwrapped it would be a
    // 2*pi - 0.02 innovation and drag the estimate the long way round.
    const double nis = f.updateYaw(0.0, -M_PI + 0.01);
    EXPECT_LT(nis, 1.0);
    const double est = f.state()(ImuEkf::TH);
    EXPECT_LT(std::fabs(wrap(est - (-M_PI + 0.01))), 0.02);
    EXPECT_GE(est, -M_PI);
    EXPECT_LE(est, M_PI);
}

TEST(ImuEkf, GyroBiasIsLearnedFromYawFixes) {
    ImuEkf::Params prm;
    Plant pl;
    pl.gyro_bias = 0.02;              // 1.1 deg/s, large for the default prior
    pl.gyro_noise = prm.gyro_noise_std;
    pl.accel_noise = prm.accel_noise_std;
    ImuEkf f(prm);
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
    run(f, pl, 60.0, 500.0, 5.0, 2.0, [](int) {});

    const double est = f.state()(ImuEkf::BG);
    const double sd = std::sqrt(f.covariance()(ImuEkf::BG, ImuEkf::BG));
    EXPECT_NEAR(est, 0.02, 3.0 * sd);
    EXPECT_LT(sd, 2e-3) << "the bias should be well determined after 60 s";
}

TEST(ImuEkf, AccelBiasIsLearnedOnlyWithAPositionFix) {
    // Without an absolute position, a constant accelerometer bias is
    // indistinguishable from a real acceleration: it is unobservable, and the
    // filter must say so by keeping its uncertainty rather than inventing an
    // estimate. With position fixes it converges.
    ImuEkf::Params prm;
    const Eigen::Vector2d bias(0.05, -0.03);
    const auto trial = [&](double pos_hz, double& est_err, double& sd_ratio) {
        Plant pl;
        pl.accel_bias = bias;
        pl.gyro_noise = prm.gyro_noise_std;
        pl.accel_noise = prm.accel_noise_std;
        ImuEkf f(prm);
        f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
        f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
        run(f, pl, 60.0, 500.0, 5.0, pos_hz, [](int) {});
        est_err = (f.state().segment<2>(ImuEkf::BAX) - bias).norm();
        sd_ratio = std::sqrt(f.covariance()(ImuEkf::BAX, ImuEkf::BAX)) /
                   prm.init_accel_bias_std;
    };
    double err_with = 0, sd_with = 0, err_without = 0, sd_without = 0;
    trial(2.0, err_with, sd_with);
    trial(0.0, err_without, sd_without);

    EXPECT_LT(err_with, 0.01);
    EXPECT_LT(sd_with, 0.2) << "position fixes make the bias observable";
    EXPECT_GT(sd_without, 0.7)
        << "with no position fix the bias uncertainty must stay near its prior";
}

TEST(ImuEkf, PositionFixesAloneCorrectYawWhileAccelerating) {
    // With no star tracker, yaw is still observable while the body
    // accelerates: a yaw error misdirects the integrated acceleration, which
    // shows up as a position error, and the filter can only attribute it to
    // yaw through the dv/dth block of the error Jacobian. If that block is
    // wrong -- even in sign -- the correction goes the wrong way. With a star
    // tracker running, yaw is pinned too tightly for any other test to notice.
    ImuEkf::Params prm;
    prm.init_yaw_std = 0.2;
    Plant pl;
    pl.accel_gain = 5.0;                         // ~1 m/s^2
    pl.gyro_noise = prm.gyro_noise_std;
    pl.accel_noise = prm.accel_noise_std;
    ImuEkf f(prm);
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th + 0.15);   // 8.6 deg out
    run(f, pl, 30.0, 500.0, 0.0, 5.0, [](int) {});

    const double err = wrap(f.state()(ImuEkf::TH) - pl.th);
    const double sd = std::sqrt(f.covariance()(ImuEkf::TH, ImuEkf::TH));
    EXPECT_LT(std::fabs(err), 0.01) << "yaw error after 30 s: " << err;
    EXPECT_LT(std::fabs(err), 3.0 * sd + 1e-3);
}

// ======================== delayed measurements ============================ //

TEST(ImuEkf, ALateMeasurementGivesExactlyTheOnTimeResult) {
    // Two filters see the identical IMU stream and the identical
    // measurements. One receives every fix the moment it is taken. The other
    // receives a yaw fix 50 steps (100 ms) late and a position fix 30 steps
    // late, each stamped with when it was taken, and in between both apply
    // two position fixes on time. Replay has to put each late fix where it
    // belongs, re-apply everything after it, AND leave the corrected history
    // behind -- the second late fix lands after the first one's step, so it
    // restarts from a state that must already include the first correction.
    // The two filters must agree, not approximately, but to rounding.
    Plant pl;
    pl.gyro_bias = 0.01;
    pl.gyro_noise = 0.005;
    pl.accel_noise = 0.01;
    ImuEkf::Params prm;
    ImuEkf on_time(prm), late(prm);
    const Eigen::Vector2d a0 = pl.accelMeas();
    const double g0 = pl.gyroMeas();
    on_time.propagate(0.0, a0, g0);
    late.propagate(0.0, a0, g0);
    on_time.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
    late.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);

    double t_yaw = 0.0, yaw = 0.0, t_pos = 0.0;
    Eigen::Vector2d pos_late = Eigen::Vector2d::Zero();
    for (int k = 1; k <= 300; ++k) {
        pl.advance(0.002);
        const Eigen::Vector2d a = pl.accelMeas();
        const double g = pl.gyroMeas();
        on_time.propagate(pl.t, a, g);
        late.propagate(pl.t, a, g);
        if (k == 100) {
            t_yaw = pl.t;
            yaw = wrap(pl.th + 0.004);
            on_time.updateYaw(t_yaw, yaw);
        }
        if (k == 120 || k == 140) {
            const Eigen::Vector2d p = pl.p + Eigen::Vector2d(0.003, -0.002);
            on_time.updatePosition(pl.t, p);
            late.updatePosition(pl.t, p);
        }
        if (k == 130) {
            t_pos = pl.t;
            pos_late = pl.p + Eigen::Vector2d(-0.002, 0.004);
            on_time.updatePosition(t_pos, pos_late);
        }
        if (k == 150) late.updateYaw(t_yaw, yaw);
        if (k == 160) {
            // Both late fixes are in. Before this the filters legitimately
            // differ: the on-time one has had the step-130 position fix since
            // step 130.
            late.updatePosition(t_pos, pos_late);
            EXPECT_TRUE(late.state().isApprox(on_time.state(), 1e-12));
            EXPECT_TRUE(late.covariance().isApprox(on_time.covariance(), 1e-12));
        }
    }
    EXPECT_TRUE(late.state().isApprox(on_time.state(), 1e-12));
    EXPECT_TRUE(late.covariance().isApprox(on_time.covariance(), 1e-12));
    EXPECT_EQ(late.diagnostics().replays, 2u);
    EXPECT_EQ(late.diagnostics().max_replay_steps, 50u);
    EXPECT_EQ(late.diagnostics().yaw_updates, 1u) << "a replay is not a new update";
    EXPECT_EQ(late.diagnostics().pos_updates, 3u);
}

TEST(ImuEkf, AMeasurementOlderThanTheHistoryIsDroppedAndCounted) {
    ImuEkf::Params prm;
    prm.history_len = 8;
    ImuEkf f(prm);
    f.initialise(0.0, 0.0, 0.0, 0.0);
    for (int k = 1; k <= 20; ++k) f.propagate(0.002 * k, {0.0, 0.0}, 0.0);
    const ImuEkf::Vec x = f.state();
    const ImuEkf::Mat P = f.covariance();
    EXPECT_LT(f.updateYaw(0.004, 0.5), 0.0);
    EXPECT_EQ(f.diagnostics().too_old, 1u);
    EXPECT_TRUE(f.state() == x);
    EXPECT_TRUE(f.covariance() == P);
    EXPECT_GE(f.updateYaw(0.002 * 15, 0.0), 0.0) << "inside the history: placed";
}

TEST(ImuEkf, AMeasurementBeforeInitialisationIsCountedNotApplied) {
    ImuEkf f(ImuEkf::Params{});
    EXPECT_LT(f.updatePosition(0.0, {1.0, 1.0}), 0.0);
    EXPECT_EQ(f.diagnostics().not_ready, 1u);
}

TEST(ImuEkf, WorstCaseReplayFitsInsideOneImuPeriod) {
    // The realtime bound. A measurement stamped at the oldest kept step makes
    // the filter recompute the whole history, measurements included -- the
    // most work one update can cause. That must finish well inside the 2 ms
    // between 500 Hz IMU samples, or the estimator thread falls behind its
    // input. Budgeted at half the period. Measured on the x86 dev box at
    // ~230 us for the 250-step replay (~0.9 us per step), which leaves room
    // for a target a few times slower.
    ImuEkf::Params prm;
    prm.history_len = 256;
    ImuEkf f(prm);
    Plant pl;
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, pl.p.x(), pl.p.y(), pl.th);
    std::vector<double> t_at(301, 0.0);
    for (int k = 1; k <= 300; ++k) {
        pl.advance(0.002);
        f.propagate(pl.t, pl.accelMeas(), pl.gyroMeas());
        t_at[static_cast<std::size_t>(k)] = pl.t;
        if (k % 10 == 0) f.updateYaw(pl.t, pl.th);
        if (k % 25 == 0) f.updatePosition(pl.t, pl.p);
    }
    // The ring holds steps 45..300; step 50 is near its oldest end.
    double worst_us = 0.0;
    for (int rep = 0; rep < 20; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        f.updateYaw(t_at[50], pl.th);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count();
        if (us > worst_us) worst_us = us;
    }
    EXPECT_EQ(f.diagnostics().max_replay_steps, 250u);
    EXPECT_LT(worst_us, 1000.0) << "worst replay of 250 steps took " << worst_us << " us";
    RecordProperty("worst_replay_us", std::to_string(worst_us));
}

// =============================== gating =================================== //

TEST(ImuEkf, AGrossOutlierIsRejectedAndLeavesTheStateUntouched) {
    ImuEkf::Params prm;
    ImuEkf f(prm);
    f.initialise(0.0, 1.0, 2.0, 0.3);
    for (int k = 1; k <= 50; ++k) f.propagate(0.002 * k, {0.0, 0.0}, 0.0);
    const ImuEkf::Vec x = f.state();
    const ImuEkf::Mat P = f.covariance();
    // ~20 sigma away in x.
    const double sd = std::sqrt(P(ImuEkf::PX, ImuEkf::PX) +
                                prm.pos_meas_std * prm.pos_meas_std);
    const double nis = f.updatePosition(0.1, {1.0 + 20.0 * sd, 2.0});
    EXPECT_GT(nis, prm.gate_pos);
    EXPECT_EQ(f.diagnostics().pos_rejected, 1u);
    EXPECT_EQ(f.diagnostics().pos_updates, 0u);
    EXPECT_TRUE(f.state() == x) << "a rejected measurement must change nothing";
    EXPECT_TRUE(f.covariance() == P);

    // An ordinary one is still accepted.
    EXPECT_LT(f.updatePosition(0.1, {1.0 + 0.5 * sd, 2.0}), prm.gate_pos);
    EXPECT_EQ(f.diagnostics().pos_updates, 1u);
}

TEST(ImuEkf, AStarTrackerGlitchIsGatedButAZeroGateDisablesIt) {
    for (const double gate : {10.83, 0.0}) {
        ImuEkf::Params prm;
        prm.gate_yaw = gate;
        ImuEkf f(prm);
        f.initialise(0.0, 0.0, 0.0, 0.0);
        f.updateYaw(0.0, 0.5);       // 0.5 rad off, with a 0.01 rad prior
        if (gate > 0.0) {
            EXPECT_EQ(f.diagnostics().yaw_rejected, 1u);
            EXPECT_DOUBLE_EQ(f.state()(ImuEkf::TH), 0.0);
        } else {
            EXPECT_EQ(f.diagnostics().yaw_updates, 1u);
            EXPECT_GT(f.state()(ImuEkf::TH), 0.0);
        }
    }
}

// ============================ consistency ================================= //

namespace {

/*! The defining test of a filter that publishes a covariance. For each
 *  instant, the error normalised by the filter's own covariance, averaged over
 *  M independent runs, is chi-square with 8M degrees of freedom divided by M,
 *  if and only if the filter's covariance is right (Bar-Shalom, Li &
 *  Kirubarajan 2001, ch. 5). Too high is an overconfident filter; too low is
 *  a pessimistic one. The plant draws every noise from the filter's own
 *  Params, so a consistent filter must pass. */
void expectConsistent(int yaw_lag, int pos_lag) {
    constexpr int kRuns = 25;
    constexpr double kSeconds = 30.0, kImuHz = 500.0;
    constexpr int kEvery = 50;                         // sample NEES at 10 Hz
    constexpr int kSkip = 30;                          // ignore the first 3 s
    const int samples = static_cast<int>(kSeconds * kImuHz) / kEvery;
    std::vector<double> nees_sum(static_cast<std::size_t>(samples), 0.0);

    ImuEkf::Params prm;
    for (int r = 0; r < kRuns; ++r) {
        Plant pl;
        pl.rng.seed(1000 + r);
        pl.gyro_noise = prm.gyro_noise_std;
        pl.accel_noise = prm.accel_noise_std;
        pl.gyro_walk = prm.gyro_bias_walk;
        pl.accel_walk = prm.accel_bias_walk;
        // Initial biases drawn from the filter's prior, and the initial pose
        // perturbed by its prior, so the start is consistent too.
        pl.gyro_bias = prm.init_gyro_bias_std * pl.n01(pl.rng);
        pl.accel_bias = prm.init_accel_bias_std *
                        Eigen::Vector2d(pl.n01(pl.rng), pl.n01(pl.rng));
        ImuEkf f(prm);
        f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
        f.initialise(0.0, pl.p.x() + prm.init_pos_std * pl.n01(pl.rng),
                     pl.p.y() + prm.init_pos_std * pl.n01(pl.rng),
                     pl.th + prm.init_yaw_std * pl.n01(pl.rng));
        // The plant starts at rest; so does the filter, but with a velocity
        // prior, so the true velocity is drawn from it to match.
        pl.v = prm.init_vel_std * Eigen::Vector2d(pl.n01(pl.rng), pl.n01(pl.rng));

        run(f, pl, kSeconds, kImuHz, 10.0, 2.0, [&](int k) {
            if (k % kEvery) return;
            const ImuEkf::Vec e = errorOf(f, pl);
            const double nees = e.dot(f.covariance().ldlt().solve(e));
            nees_sum[static_cast<std::size_t>(k / kEvery - 1)] += nees;
        }, yaw_lag, pos_lag);
    }

    const double dof = 8.0 * kRuns;
    const double lo = chi2Quantile(-1.96, dof) / kRuns;
    const double hi = chi2Quantile(+1.96, dof) / kRuns;
    int inside = 0, counted = 0;
    double mean = 0.0;
    for (int i = kSkip; i < samples; ++i) {
        const double avg = nees_sum[static_cast<std::size_t>(i)] / kRuns;
        mean += avg;
        ++counted;
        if (avg >= lo && avg <= hi) ++inside;
    }
    mean /= counted;
    const double frac = static_cast<double>(inside) / counted;
    // 95% bounds, so about 5% of instants fall outside by chance alone; and
    // the instants are correlated in time, so allow some margin over that.
    EXPECT_GT(frac, 0.85) << "average NEES " << mean << ", 95% band ["
                          << lo << ", " << hi << "]";
    EXPECT_NEAR(mean, 8.0, 1.5);
}

}  // namespace

TEST(ImuEkf, NeesIsWithinChiSquareBoundsOverMonteCarloRuns) {
    expectConsistent(0, 0);
}

TEST(ImuEkf, NeesStaysWithinBoundsWithRealisticSensorLatency) {
    // Star tracker 100 ms late, relative-nav fix 200 ms late. The NEES is
    // sampled at the CURRENT time, after each late fix has been replayed in,
    // so this tests that recomputation leaves the published covariance as
    // trustworthy as on-time measurements do.
    expectConsistent(50, 100);
}

// ============================== realtime ================================== //

TEST(ImuEkf, EveryCallIsAllocationFreeIncludingTheFirst) {
    // Constructed outside the scope: the history ring is the one allocation,
    // and it happens there. Everything after -- including delayed updates,
    // replays and a ring that has wrapped -- must not touch the heap.
    ImuEkf::Params prm;
    prm.history_len = 64;
    ImuEkf f(prm);
    Plant pl;
    AllocScope scope;
    f.propagate(0.0, pl.accelMeas(), pl.gyroMeas());
    f.initialise(0.0, 0.0, 0.0, 0.0);
    double t_late = 0.0;
    for (int k = 1; k <= 300; ++k) {
        pl.advance(0.002);
        f.propagate(pl.t, pl.accelMeas(), pl.gyroMeas());
        if (k % 20 == 0) f.updateYaw(pl.t, pl.th);
        if (k % 50 == 0) f.updatePosition(pl.t, pl.p);
        if (k % 50 == 0) f.updateVelocity(pl.t, pl.v);
        if (k % 40 == 0) t_late = pl.t;
        if (k % 40 == 30) f.updateYaw(t_late, pl.th);     // 30 steps late
        if (k == 250) f.updateYaw(0.001, 0.0);            // older than the ring
    }
    (void)f.yawRate();
    (void)f.yawRateVariance();
    EXPECT_EQ(scope.count(), 0);
    EXPECT_GT(f.diagnostics().replays, 0u);
    EXPECT_EQ(f.diagnostics().too_old, 1u);
}
