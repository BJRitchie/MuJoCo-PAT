/*! A series written to disk and read back scores identically to the original, so
 *  a run recorded elsewhere (the live stack) and scored from its CSV is scored by
 *  the same arithmetic as one scored in memory.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "run_summary.h"
#include "series_io.h"

namespace pat_arm_control {
namespace {
using bench::RunContext;
using bench::RunSeries;

RunSeries synthetic(bool with_est, bool with_q) {
    RunSeries s;
    s.setJoints(2);
    s.tau_limit = {0.25, 0.25};
    s.q_min = {-1.0, -1.0};
    s.q_max = {1.0, 1.0};
    for (int k = 0; k < 501; ++k) {
        const double t = 0.01 * k;
        s.t.push_back(t);
        s.rx.push_back(0.1 * t);   s.ry.push_back(0.02 * std::sin(t));
        s.rvx.push_back(0.1);      s.rvy.push_back(0.02 * std::cos(t));
        s.ex.push_back(0.1 * t + 0.003);
        s.ey.push_back(0.02 * std::sin(t) + 0.004);
        s.eth.push_back(0.002 * std::sin(3.0 * t));
        s.tau[0].push_back(0.2 * std::sin(6.28 * t));
        s.tau[1].push_back(0.05 * std::cos(2.0 * t));
        s.qd[0].push_back(0.5);    s.qd[1].push_back(-0.25);
        if (with_q) {
            s.q[0].push_back(0.3 * std::sin(t));
            s.q[1].push_back(1.2 * std::sin(t));   // leaves its box
        }
        s.ncon.push_back(k % 50 == 0 ? 1 : 0);
        s.base_x.push_back(1e-3 * t);  s.base_y.push_back(2e-3 * t);
        s.base_yaw.push_back(0.01 * t); s.base_wz.push_back(0.01);
        s.tick_us.push_back(700.0 + 5.0 * (k % 7));
        s.mode.push_back(k > 400 ? 2 : 0);
        if (with_est) {
            s.est_x.push_back(1e-3);  s.est_y.push_back(-2e-3);
            s.est_yaw.push_back(1e-4);
            s.est_vx.push_back(1e-3); s.est_vy.push_back(0.0);
            s.est_wz.push_back(2e-4);
        }
    }
    if (!with_q) s.q.clear();
    return s;
}

std::string tempPath(const char* name) {
    return std::string(::testing::TempDir()) + name;
}

RunContext context() {
    RunContext ctx;
    ctx.scenario = "circle";
    ctx.law = "nmpc";
    ctx.tag = "sqp";
    ctx.window_start_s = 1.0;
    return ctx;
}

void expectSameSummary(const RunSeries& a, const RunSeries& b) {
    const auto va = bench::summarise(a, context());
    const auto vb = bench::summarise(b, context());
    ASSERT_EQ(va.size(), vb.size());
    for (std::size_t i = 0; i < va.size(); ++i) {
        ASSERT_EQ(va[i].first, vb[i].first);
        if (std::isnan(va[i].second)) {
            EXPECT_TRUE(std::isnan(vb[i].second)) << va[i].first;
        } else {
            // The CSV carries nine significant digits.
            EXPECT_NEAR(va[i].second, vb[i].second,
                        1e-6 * (1.0 + std::fabs(va[i].second))) << va[i].first;
        }
    }
}

TEST(SeriesIo, ARoundTripScoresTheSame) {
    const std::string path = tempPath("series_roundtrip.csv");
    const RunSeries orig = synthetic(true, true);
    bench::writeSeriesCsv(path, orig);
    RunSeries back = bench::readSeriesCsv(path);
    back.tau_limit = orig.tau_limit;
    back.q_min = orig.q_min;
    back.q_max = orig.q_max;
    ASSERT_EQ(back.size(), orig.size());
    expectSameSummary(orig, back);
    std::remove(path.c_str());
}

TEST(SeriesIo, OptionalColumnsMayBeAbsent) {
    const std::string path = tempPath("series_minimal.csv");
    const RunSeries orig = synthetic(false, false);
    bench::writeSeriesCsv(path, orig);
    RunSeries back = bench::readSeriesCsv(path);
    back.tau_limit = orig.tau_limit;
    EXPECT_TRUE(back.est_x.empty());
    EXPECT_TRUE(back.q.empty());
    expectSameSummary(orig, back);
    std::remove(path.c_str());
}

TEST(SeriesIo, UnknownColumnsAreIgnored) {
    const std::string path = tempPath("series_extra.csv");
    bench::writeSeriesCsv(path, synthetic(false, true));
    // Append a column the reader has never heard of.
    std::string text, line;
    {
        std::ifstream is(path);
        bool first = true;
        while (std::getline(is, line)) {
            text += line + (first ? ",future_column\n" : ",1.5\n");
            first = false;
        }
    }
    {
        std::ofstream os(path);
        os << text;
    }
    EXPECT_NO_THROW(bench::readSeriesCsv(path));
    std::remove(path.c_str());
}

TEST(SeriesIo, WindowsLineEndingsAreAccepted) {
    const std::string path = tempPath("series_crlf.csv");
    const RunSeries orig = synthetic(true, true);
    bench::writeSeriesCsv(path, orig);
    std::string text, line;
    {
        std::ifstream is(path);
        while (std::getline(is, line)) text += line + "\r\n";
    }
    {
        std::ofstream os(path, std::ios::binary);
        os << text;
    }
    const RunSeries back = bench::readSeriesCsv(path);
    EXPECT_EQ(back.size(), orig.size());
    EXPECT_EQ(back.est_wz.size(), orig.est_wz.size());
    std::remove(path.c_str());
}

TEST(SeriesIo, BrokenFilesAreRefused) {
    const std::string path = tempPath("series_broken.csv");
    EXPECT_THROW(bench::readSeriesCsv(tempPath("does_not_exist.csv")), std::runtime_error);
    {
        std::ofstream os(path);
        os << "t_s,ref_x\n0,0\n";
    }
    EXPECT_THROW(bench::readSeriesCsv(path), std::runtime_error);   // missing columns
    bench::writeSeriesCsv(path, synthetic(false, false));
    {
        std::ofstream os(path, std::ios::app);
        os << "1,2,3\n";
    }
    EXPECT_THROW(bench::readSeriesCsv(path), std::runtime_error);   // ragged row
    std::remove(path.c_str());
}

}  // namespace
}  // namespace pat_arm_control
