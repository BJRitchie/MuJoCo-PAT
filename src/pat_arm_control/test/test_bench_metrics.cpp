/*! The comparison metrics, each against a signal whose answer is known in closed
 *  form. A metric that is slightly wrong yields a plausible wrong number that no
 *  later stage would catch, so each definition is pinned here once.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "metrics.h"

namespace pat_arm_control {
namespace {
namespace m = bench::metrics;

//! t = 0, dt, 2dt, ... over `seconds`, inclusive of both ends.
std::vector<double> timeAxis(double seconds, double dt) {
    std::vector<double> t;
    const int n = static_cast<int>(std::lround(seconds / dt));
    for (int k = 0; k <= n; ++k) t.push_back(k * dt);
    return t;
}

template <class F>
std::vector<double> sample(const std::vector<double>& t, F f) {
    std::vector<double> x;
    x.reserve(t.size());
    for (double tk : t) x.push_back(f(tk));
    return x;
}

TEST(BenchMetrics, MomentsOfAConstantAndOfASine) {
    const std::vector<double> c(100, 3.0);
    EXPECT_DOUBLE_EQ(m::mean(c), 3.0);
    EXPECT_DOUBLE_EQ(m::variance(c), 0.0);
    EXPECT_DOUBLE_EQ(m::meanSquare(c), 9.0);
    EXPECT_DOUBLE_EQ(m::rms(c), 3.0);

    // A sine over whole periods: mean 0, variance = mean square = A^2/2.
    const double A = 2.0, w = 2.0 * M_PI;   // 1 Hz
    const auto t = timeAxis(4.0, 0.001);
    const auto x = sample(t, [&](double tk) { return A * std::sin(w * tk); });
    std::vector<double> whole(x.begin(), x.end() - 1);   // drop the duplicate endpoint
    EXPECT_NEAR(m::mean(whole), 0.0, 1e-9);
    EXPECT_NEAR(m::variance(whole), A * A / 2.0, 1e-6);
    EXPECT_NEAR(m::rms(whole), A / std::sqrt(2.0), 1e-6);
    EXPECT_NEAR(m::peakAbs(whole), A, 1e-6);
}

TEST(BenchMetrics, VarianceRemovesTheMeanButMeanSquareDoesNot) {
    // The error's variance and its MSE differ exactly by the squared bias.
    const std::vector<double> x = {4.0, 6.0, 4.0, 6.0};
    EXPECT_DOUBLE_EQ(m::mean(x), 5.0);
    EXPECT_DOUBLE_EQ(m::variance(x), 1.0);
    EXPECT_DOUBLE_EQ(m::meanSquare(x), 26.0);
    EXPECT_DOUBLE_EQ(m::meanSquare(x), m::variance(x) + m::mean(x) * m::mean(x));
}

TEST(BenchMetrics, TheWindowStartsAtFrom) {
    const std::vector<double> x = {100.0, 100.0, 1.0, 1.0, 1.0};
    EXPECT_DOUBLE_EQ(m::mean(x, 2), 1.0);
    EXPECT_DOUBLE_EQ(m::variance(x, 2), 0.0);
    EXPECT_DOUBLE_EQ(m::peakAbs(x, 2), 1.0);
}

TEST(BenchMetrics, AnEmptyWindowIsNaNNotZero) {
    const std::vector<double> x = {1.0, 2.0};
    EXPECT_TRUE(std::isnan(m::mean(x, 5)));
    EXPECT_TRUE(std::isnan(m::variance(x, 5)));
    EXPECT_TRUE(std::isnan(m::rms(x, 5)));
    EXPECT_TRUE(std::isnan(m::percentile(x, 50, 5)));
    EXPECT_TRUE(std::isnan(m::fractionAtLeast(x, 1.0, 5)));
}

TEST(BenchMetrics, IntegralOfSquaredSineIsAHalfAmplitudeSquaredTimesDuration) {
    // The integral of squared control input for a torque A sin(wt) over T seconds
    // (whole periods) is A^2 T / 2.
    const double A = 0.25, w = 2.0 * M_PI * 2.0, T = 5.0;   // 2 Hz, 10 periods
    const auto t = timeAxis(T, 0.0005);
    const auto tau = sample(t, [&](double tk) { return A * std::sin(w * tk); });
    EXPECT_NEAR(m::integralSquared(t, tau), A * A * T / 2.0, 1e-6);
    EXPECT_NEAR(m::integralAbs(t, tau), A * T * 2.0 / M_PI, 1e-4);   // mean |sin| = 2/pi
    EXPECT_NEAR(m::integral(t, tau), 0.0, 1e-6);
}

TEST(BenchMetrics, IntegralsUseTheActualSampleTimes) {
    // Non-uniform spacing: a constant 2 over [0, 0.1] then [0.1, 1.0] is 2.
    const std::vector<double> t = {0.0, 0.1, 1.0};
    const std::vector<double> x = {2.0, 2.0, 2.0};
    EXPECT_DOUBLE_EQ(m::integral(t, x), 2.0);
    EXPECT_DOUBLE_EQ(m::integralSquared(t, x), 4.0);
}

TEST(BenchMetrics, TotalVariationSeparatesChatterFromASteadyCommand) {
    // Both commands sit at +-0.25 about the same mean; one switches every tick.
    const std::vector<double> chatter = {0.25, -0.25, 0.25, -0.25, 0.25};
    const std::vector<double> steady(5, 0.25);
    EXPECT_DOUBLE_EQ(m::totalVariation(chatter), 4 * 0.5);
    EXPECT_DOUBLE_EQ(m::totalVariation(steady), 0.0);
    EXPECT_DOUBLE_EQ(m::rmsStep(chatter), 0.5);
    EXPECT_DOUBLE_EQ(m::rmsStep(steady), 0.0);
    // A monotone ramp's total variation is just its rise.
    EXPECT_DOUBLE_EQ(m::totalVariation({0.0, 1.0, 2.0, 3.0}), 3.0);
}

TEST(BenchMetrics, FractionAtLeastCountsMagnitudes) {
    const std::vector<double> x = {0.25, -0.25, 0.1, 0.0};
    EXPECT_DOUBLE_EQ(m::fractionAtLeast(x, 0.2475), 0.5);   // both signs count
}

TEST(BenchMetrics, PercentileMatchesNumpyLinearInterpolation) {
    std::vector<double> x;
    for (int i = 100; i >= 1; --i) x.push_back(i);   // unsorted on purpose
    EXPECT_DOUBLE_EQ(m::percentile(x, 0), 1.0);
    EXPECT_DOUBLE_EQ(m::percentile(x, 100), 100.0);
    EXPECT_NEAR(m::percentile(x, 50), 50.5, 1e-12);
    EXPECT_NEAR(m::percentile(x, 99), 99.01, 1e-12);   // numpy.percentile(1..100, 99)
}

TEST(BenchMetrics, UnwrapRemovesTheBranchCutButKeepsRealMotion) {
    // A base turning steadily through +-pi, as the simulator reports it.
    std::vector<double> wrapped, truth;
    for (int k = 0; k < 200; ++k) {
        const double a = 0.05 * k;   // 10 rad total, > 1.5 turns
        truth.push_back(a);
        wrapped.push_back(std::remainder(a, 2.0 * M_PI));
    }
    const auto u = m::unwrap(wrapped);
    for (std::size_t k = 0; k < u.size(); ++k) EXPECT_NEAR(u[k], truth[k], 1e-9);

    // A small wobble about zero is untouched.
    const auto w = m::unwrap({0.0, 0.1, -0.1, 0.05});
    EXPECT_DOUBLE_EQ(w[2], -0.1);
}

TEST(BenchMetrics, FirstIndexAtOrAfterFindsTheWindowStart) {
    const std::vector<double> t = {0.0, 0.5, 1.0, 1.5};
    EXPECT_EQ(m::firstIndexAtOrAfter(t, 0.5), 1u);
    EXPECT_EQ(m::firstIndexAtOrAfter(t, 0.6), 2u);
    EXPECT_EQ(m::firstIndexAtOrAfter(t, 9.0), t.size());
}

TEST(BenchMetrics, FirstOrderStepHasTheClosedFormRiseAndSettlingTimes) {
    // r = 1 - exp(-t/tau): 10-90% rise time = tau ln 9; 2% settling = tau ln 50.
    const double tau = 0.4;
    const auto t = timeAxis(5.0, 0.0005);
    const auto r = sample(t, [&](double tk) { return 1.0 - std::exp(-tk / tau); });
    const auto s = m::stepResponse(t, r, 0.0);
    EXPECT_NEAR(s.rise_time, tau * std::log(9.0), 1e-3);
    EXPECT_NEAR(s.settle_time, tau * std::log(50.0), 2e-3);
    EXPECT_DOUBLE_EQ(s.overshoot_pct, 0.0);
    EXPECT_NEAR(s.steady_state_error, std::exp(-5.0 / tau), 1e-3);
}

TEST(BenchMetrics, UnderdampedStepOvershootsByTheClosedFormAmount) {
    // Second order, zeta = 0.5: overshoot = exp(-pi zeta / sqrt(1-zeta^2)) = 16.3%.
    const double zeta = 0.5, wn = 6.0, wd = wn * std::sqrt(1.0 - zeta * zeta);
    const auto t = timeAxis(8.0, 0.0005);
    const auto r = sample(t, [&](double tk) {
        return 1.0 - std::exp(-zeta * wn * tk) *
               (std::cos(wd * tk) + zeta / std::sqrt(1.0 - zeta * zeta) * std::sin(wd * tk));
    });
    const auto s = m::stepResponse(t, r, 0.0);
    EXPECT_NEAR(s.overshoot_pct,
                100.0 * std::exp(-M_PI * zeta / std::sqrt(1.0 - zeta * zeta)), 0.05);
    EXPECT_NEAR(s.steady_state_error, 0.0, 1e-4);
    EXPECT_TRUE(std::isfinite(s.settle_time));
}

TEST(BenchMetrics, StepTimesAreMeasuredFromTheStep) {
    // The same first-order response, but the step happens at t = 2.
    const double tau = 0.4, t_step = 2.0;
    const auto t = timeAxis(8.0, 0.0005);
    const auto r = sample(t, [&](double tk) {
        return tk < t_step ? 0.0 : 1.0 - std::exp(-(tk - t_step) / tau);
    });
    const auto s = m::stepResponse(t, r, t_step);
    EXPECT_NEAR(s.rise_time, tau * std::log(9.0), 2e-3);
    EXPECT_NEAR(s.settle_time, tau * std::log(50.0), 3e-3);
}

TEST(BenchMetrics, AResponseThatNeverSettlesOrRisesIsNaN) {
    const auto t = timeAxis(2.0, 0.01);
    const std::vector<double> flat(t.size(), 0.05);   // never reaches 90%
    const auto s = m::stepResponse(t, flat, 0.0);
    EXPECT_TRUE(std::isnan(s.rise_time));
    EXPECT_TRUE(std::isnan(s.settle_time));
    EXPECT_DOUBLE_EQ(s.overshoot_pct, 0.0);
    EXPECT_NEAR(s.steady_state_error, 0.95, 1e-12);

    // A late excursion is not forgiven by an early entry into the band.
    std::vector<double> late(t.size(), 1.0);
    late[t.size() - 5] = 1.5;
    const auto s2 = m::stepResponse(t, late, 0.0);
    EXPECT_GT(s2.settle_time, 1.9);
}

}  // namespace
}  // namespace pat_arm_control
