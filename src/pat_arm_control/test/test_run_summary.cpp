/*! The per-run summary: a synthetic series whose every figure of merit is known
 *  in closed form, so the bookkeeping (units, windows, which joints, which
 *  columns) is pinned independently of any controller or plant.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "run_summary.h"

namespace pat_arm_control {
namespace {
using bench::NamedValues;
using bench::RunContext;
using bench::RunSeries;

constexpr double kDt = 0.01, kT = 10.0;

//! 10 s at 100 Hz. EE sits (3, 4) mm off its reference (a 5 mm error) while the
//! reference moves along +x. Joint 0 carries a 1 Hz sine of 0.2 N*m, joint 1 is
//! idle. The base yaws at 10 mrad/s.
RunSeries synthetic() {
    RunSeries s;
    s.setJoints(2);
    s.tau_limit = {0.25, 0.25};
    const int n = static_cast<int>(std::lround(kT / kDt)) + 1;
    for (int k = 0; k < n; ++k) {
        const double t = k * kDt;
        s.t.push_back(t);
        s.rx.push_back(0.1 * t);   s.ry.push_back(0.0);
        s.rvx.push_back(0.1);      s.rvy.push_back(0.0);
        s.ex.push_back(0.1 * t + 0.003);
        s.ey.push_back(0.004);
        s.eth.push_back(0.002);
        s.tau[0].push_back(0.2 * std::sin(2.0 * M_PI * t));
        s.tau[1].push_back(0.0);
        s.qd[0].push_back(1.0);
        s.qd[1].push_back(0.0);
        s.base_x.push_back(0.0);   s.base_y.push_back(0.0);
        s.base_yaw.push_back(0.01 * t);
        s.base_wz.push_back(0.01);
        s.tick_us.push_back(100.0);
        s.mode.push_back(0);
    }
    return s;
}

std::map<std::string, double> asMap(const NamedValues& v) {
    return {v.begin(), v.end()};
}

TEST(RunSummary, TrackingMetricsInMillimetres) {
    RunContext ctx;
    ctx.window_start_s = 5.0;
    const auto m = asMap(bench::summarise(synthetic(), ctx));
    EXPECT_NEAR(m.at("rmse_mm"), 5.0, 1e-9);
    EXPECT_NEAR(m.at("err_mean_mm"), 5.0, 1e-9);
    EXPECT_NEAR(m.at("mse_mm2"), 25.0, 1e-9);
    EXPECT_NEAR(m.at("err_var_mm2"), 0.0, 1e-9);
    EXPECT_NEAR(m.at("err_peak_mm"), 5.0, 1e-9);
    EXPECT_NEAR(m.at("theta_rms_mrad"), 2.0, 1e-9);
    // Reference moving along +x with the EE 3 mm ahead and 4 mm to the side.
    EXPECT_NEAR(m.at("along_mean_mm"), 3.0, 1e-9);
    EXPECT_NEAR(m.at("cross_rms_mm"), 4.0, 1e-9);
}

TEST(RunSummary, EffortIntegralsAndRailFraction) {
    RunContext ctx;
    ctx.window_start_s = 5.0;
    const auto m = asMap(bench::summarise(synthetic(), ctx));
    // Joint 0: A^2 T / 2 over whole periods; joint 1 contributes nothing.
    EXPECT_NEAR(m.at("tau2_int_win"), 0.2 * 0.2 * 5.0 / 2.0, 1e-4);
    EXPECT_NEAR(m.at("tau2_int_total"), 0.2 * 0.2 * 10.0 / 2.0, 1e-4);
    EXPECT_NEAR(m.at("tau_peak"), 0.2, 1e-3);
    EXPECT_NEAR(m.at("rail_frac"), 0.0, 1e-12);
    // Pooled over both joints, half the samples are zero: variance = A^2/2 / 2 +
    // the between-joint spread, here simply mean(x^2) with mean 0.
    EXPECT_NEAR(m.at("tau_var"), 0.2 * 0.2 / 2.0 / 2.0, 1e-4);
    // Mechanical work: integral of |tau * qd| with qd = 1 is integral |tau|.
    EXPECT_NEAR(m.at("work_int"), 0.2 * 5.0 * 2.0 / M_PI, 1e-3);
    // A 1 Hz sine of 0.2 moves by 4 * 0.2 per period.
    EXPECT_NEAR(m.at("chatter_tv"), 0.2 * 4.0 * 5.0, 1e-2);
}

TEST(RunSummary, RailFractionCountsOwnedJointTicksAtTheLimit) {
    RunSeries s = synthetic();
    for (double& x : s.tau[1]) x = 0.25;   // joint 1 pinned at its limit
    RunContext ctx;
    ctx.window_start_s = 5.0;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_NEAR(m.at("rail_frac"), 0.5, 1e-9);   // one of two joints, every tick
}

TEST(RunSummary, BaseYawIsMeasuredFromWhereItStartedAndIntegratedSquared) {
    RunContext ctx;
    ctx.window_start_s = 5.0;
    const auto m = asMap(bench::summarise(synthetic(), ctx));
    // psi = 0.01 t: the integral of psi^2 over [0, 10] is 1e-4 * 10^3 / 3.
    EXPECT_NEAR(m.at("yaw2_int_total"), 1e-4 * 1000.0 / 3.0, 1e-6);
    EXPECT_NEAR(m.at("yaw2_int_win"), 1e-4 * (1000.0 - 125.0) / 3.0, 1e-6);
    EXPECT_NEAR(m.at("yaw_peak_mrad"), 100.0, 1e-9);
    EXPECT_NEAR(m.at("yaw_final_mrad"), 100.0, 1e-9);
    EXPECT_NEAR(m.at("wz2_int_total"), 1e-4 * 10.0, 1e-9);
    EXPECT_NEAR(m.at("wz_abs_int_total"), 0.01 * 10.0, 1e-9);
}

TEST(RunSummary, YawOffsetIgnoresWhereTheBaseHappenedToPoint) {
    // The same turn, started from a heading near the branch cut and wrapped.
    RunSeries s = synthetic();
    const double start = M_PI - 0.02;
    for (std::size_t k = 0; k < s.size(); ++k) {
        s.base_yaw[k] = std::remainder(start + 0.01 * s.t[k], 2.0 * M_PI);
    }
    RunContext ctx;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_NEAR(m.at("yaw_peak_mrad"), 100.0, 1e-6);
    EXPECT_NEAR(m.at("yaw2_int_total"), 1e-4 * 1000.0 / 3.0, 1e-6);
}

TEST(RunSummary, TimingPercentilesAndHealthCounters) {
    RunSeries s = synthetic();
    s.tick_us[3] = 900.0;   // one slow tick
    RunContext ctx;
    ctx.safe_ticks = 7;
    ctx.tier1_entries = 2;
    ctx.qp_iter_mean = 6.5;
    ctx.qp_iter_max = 9;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_DOUBLE_EQ(m.at("tick_us_p50"), 100.0);
    EXPECT_DOUBLE_EQ(m.at("tick_us_max"), 900.0);
    EXPECT_GT(m.at("tick_us_mean"), 100.0);
    EXPECT_DOUBLE_EQ(m.at("safe_ticks"), 7.0);
    EXPECT_DOUBLE_EQ(m.at("tier1_entries"), 2.0);
    EXPECT_DOUBLE_EQ(m.at("qp_iter_mean"), 6.5);

    ctx.has_solver = false;   // a closed-form law has no QP to report
    const auto m2 = asMap(bench::summarise(s, ctx));
    EXPECT_TRUE(std::isnan(m2.at("qp_iter_mean")));
}

TEST(RunSummary, StepColumnsFollowTheEndEffectorsProgressAlongTheStep) {
    // A first-order response of the EE along a diagonal 6 x 8 cm step at t = 1:
    // the closed-form rise time and settling time must come out in seconds, and
    // the steady-state error in millimetres.
    const double tau = 0.3, t_step = 1.0, dx = 0.06, dy = 0.08;   // length 0.1 m
    RunSeries s = synthetic();
    for (std::size_t k = 0; k < s.size(); ++k) {
        const double t = s.t[k];
        const double r = t < t_step ? 0.0 : 1.0 - std::exp(-(t - t_step) / tau);
        s.ex[k] = 0.5 + r * dx;
        s.ey[k] = 0.2 + r * dy;
    }
    RunContext ctx;
    ctx.has_step = true;
    ctx.step_dx = dx; ctx.step_dy = dy; ctx.step_t = t_step;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_NEAR(m.at("step_rise_s"), tau * std::log(9.0), 2e-3);
    EXPECT_NEAR(m.at("step_settle_s"), tau * std::log(50.0), 2e-2);
    EXPECT_DOUBLE_EQ(m.at("step_overshoot_pct"), 0.0);
    EXPECT_NEAR(m.at("step_sse_mm"), 100.0 * std::exp(-(kT - t_step) / tau), 1e-3);
}

std::string slurp(const std::string& path);

TEST(RunSummary, JointLimitColumnsMeasureTheWorstJointOutsideItsBox) {
    // Joint 0 starts 0.5 rad below its limit and is brought back linearly over
    // the first 5 s; joint 1 stays inside. Two ticks touch a hard stop.
    RunSeries s = synthetic();
    s.q_min = {0.0, -1.0};
    s.q_max = {1.0, 1.0};
    for (std::size_t k = 0; k < s.size(); ++k) {
        const double t = s.t[k];
        s.q[0].push_back(t < 5.0 ? -0.5 + 0.1 * t : 0.2);
        s.q[1].push_back(0.0);
        s.ncon.push_back(k < 2 ? 1 : 0);
    }
    RunContext ctx;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_NEAR(m.at("qlim_peak_rad"), 0.5, 1e-12);
    EXPECT_NEAR(m.at("qlim_int_rad_s"), 0.5 * 5.0 / 2.0, 1e-6);   // triangle
    EXPECT_NEAR(m.at("qlim_outside_frac"), 500.0 / 1001.0, 1e-3);
    EXPECT_DOUBLE_EQ(m.at("qlim_final_rad"), 0.0);
    EXPECT_NEAR(m.at("stop_frac"), 2.0 / 1001.0, 1e-12);

    // Not recorded: NaN, not a reassuring zero.
    const auto m0 = asMap(bench::summarise(synthetic(), ctx));
    EXPECT_TRUE(std::isnan(m0.at("qlim_peak_rad")));
    EXPECT_TRUE(std::isnan(m0.at("stop_frac")));
}

TEST(RunSummary, EstimateErrorColumnsAreRmsOfWhatTheControllerWasGiven) {
    RunSeries s = synthetic();
    for (std::size_t k = 0; k < s.size(); ++k) {
        s.est_x.push_back(0.003);   s.est_y.push_back(0.004);   // 5 mm position error
        s.est_yaw.push_back(0.0005);
        s.est_vx.push_back(0.006);  s.est_vy.push_back(0.008);  // 10 mm/s
        s.est_wz.push_back(0.001);
    }
    RunContext ctx;
    ctx.window_start_s = 5.0;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_NEAR(m.at("est_pos_rms_mm"), 5.0, 1e-9);
    EXPECT_NEAR(m.at("est_yaw_rms_mrad"), 0.5, 1e-9);
    EXPECT_NEAR(m.at("est_vel_rms_mm_s"), 10.0, 1e-9);
    EXPECT_NEAR(m.at("est_wz_rms_mrad_s"), 1.0, 1e-9);

    // Without a measurement model the controller read the plant: no columns.
    const auto m0 = asMap(bench::summarise(synthetic(), ctx));
    EXPECT_TRUE(std::isnan(m0.at("est_pos_rms_mm")));
}

TEST(RunSummary, LabelsBecomeColumnsAfterTheSeed) {
    const std::string path = ::testing::TempDir() + "labels.csv";
    std::remove(path.c_str());
    RunContext ctx;
    ctx.scenario = "circle";
    ctx.law = "ompc";
    ctx.labels = {{"state_source", "ekf"}, {"noise_scale", "2"}};
    bench::appendSummaryCsv(path, ctx, bench::summarise(synthetic(), ctx));
    std::istringstream is(slurp(path));
    std::string header, row;
    std::getline(is, header);
    std::getline(is, row);
    EXPECT_EQ(header.rfind("scenario,law,tag,seed,state_source,noise_scale,n_ticks", 0), 0u);
    EXPECT_EQ(row.rfind("circle,ompc,,0,ekf,2,", 0), 0u);
}

TEST(RunSummary, StepColumnsAreNaNWithoutAStep) {
    RunContext ctx;
    const auto m = asMap(bench::summarise(synthetic(), ctx));
    EXPECT_TRUE(std::isnan(m.at("step_rise_s")));
    EXPECT_TRUE(std::isnan(m.at("step_overshoot_pct")));
    EXPECT_TRUE(std::isnan(m.at("step_settle_s")));
    EXPECT_TRUE(std::isnan(m.at("step_sse_mm")));
}

TEST(RunSummary, AStationaryReferenceHasNoAlongOrCrossTrack) {
    RunSeries s = synthetic();
    for (std::size_t k = 0; k < s.size(); ++k) { s.rvx[k] = 0.0; s.rvy[k] = 0.0; }
    RunContext ctx;
    const auto m = asMap(bench::summarise(s, ctx));
    EXPECT_TRUE(std::isnan(m.at("along_mean_mm")));
    EXPECT_TRUE(std::isnan(m.at("cross_rms_mm")));
}

std::string slurp(const std::string& path) {
    std::ifstream is(path);
    std::stringstream ss;
    ss << is.rdbuf();
    return ss.str();
}

TEST(RunSummary, SeriesCsvHasOneRowPerTickAndAConsistentHeader) {
    const std::string path = ::testing::TempDir() + "series.csv";
    const RunSeries s = synthetic();
    bench::writeSeriesCsv(path, s);
    std::istringstream is(slurp(path));
    std::string header, line;
    std::getline(is, header);
    std::size_t cols = std::count(header.begin(), header.end(), ',') + 1;
    EXPECT_EQ(header.rfind("t_s,", 0), 0u);
    EXPECT_NE(header.find("tau0,tau1"), std::string::npos);
    std::size_t rows = 0;
    while (std::getline(is, line)) {
        ++rows;
        EXPECT_EQ(static_cast<std::size_t>(std::count(line.begin(), line.end(), ',')) + 1, cols);
    }
    EXPECT_EQ(rows, s.size());
}

TEST(RunSummary, SummaryCsvAppendsRowsUnderOneHeader) {
    const std::string path = ::testing::TempDir() + "summary.csv";
    std::remove(path.c_str());
    RunContext ctx;
    ctx.scenario = "circle";
    ctx.law = "smc";
    ctx.tag = "shipped";
    const NamedValues v = bench::summarise(synthetic(), ctx);
    bench::appendSummaryCsv(path, ctx, v);
    ctx.law = "ompc";
    ctx.seed = 3;
    bench::appendSummaryCsv(path, ctx, v);

    std::istringstream is(slurp(path));
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(is, line)) lines.push_back(line);
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0].rfind("scenario,law,tag,seed,n_ticks", 0), 0u);
    EXPECT_EQ(lines[1].rfind("circle,smc,shipped,0,", 0), 0u);
    EXPECT_EQ(lines[2].rfind("circle,ompc,shipped,3,", 0), 0u);
}

}  // namespace
}  // namespace pat_arm_control
