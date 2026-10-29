#include "run_summary.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "metrics.h"

namespace pat_arm_control {
namespace bench {

namespace {

namespace m = metrics;

//! Euclidean distance of the EE from its reference at each tick [m].
std::vector<double> errorNorm(const RunSeries& s) {
    std::vector<double> e(s.size());
    for (std::size_t k = 0; k < s.size(); ++k) {
        e[k] = std::hypot(s.ex[k] - s.rx[k], s.ey[k] - s.ry[k]);
    }
    return e;
}

//! Every owned joint's samples from `from` on, concatenated: the pooled series.
std::vector<double> pooled(const std::vector<std::vector<double>>& per_joint,
                           std::size_t from) {
    std::vector<double> out;
    for (const auto& j : per_joint) {
        for (std::size_t k = from; k < j.size(); ++k) out.push_back(j[k]);
    }
    return out;
}

constexpr double kMm = 1e3;

}  // namespace

NamedValues summarise(const RunSeries& s, const RunContext& ctx) {
    NamedValues v;
    const auto add = [&v](const char* name, double value) { v.emplace_back(name, value); };

    const std::size_t n = s.size();
    const std::size_t from = m::firstIndexAtOrAfter(s.t, ctx.window_start_s);
    const std::vector<double> e = errorNorm(s);

    add("n_ticks", static_cast<double>(n));
    add("window_start_s", ctx.window_start_s);

    // --- Tracking -----------------------------------------------------------
    add("err_mean_mm", m::mean(e, from) * kMm);
    add("mse_mm2", m::meanSquare(e, from) * kMm * kMm);
    add("rmse_mm", m::rms(e, from) * kMm);
    add("err_var_mm2", m::variance(e, from) * kMm * kMm);
    add("err_peak_mm", m::peakAbs(e, from) * kMm);
    add("theta_rms_mrad", m::rms(s.eth, from) * 1e3);

    // Along- and cross-track error against the reference's direction of travel,
    // for the ticks where it is moving. Along is signed: negative is trailing.
    double along = 0.0, cross_sq = 0.0;
    std::size_t moving = 0;
    for (std::size_t k = from; k < n; ++k) {
        const double sp = std::hypot(s.rvx[k], s.rvy[k]);
        if (sp <= 1e-9) continue;
        const double ux = s.rvx[k] / sp, uy = s.rvy[k] / sp;
        const double dx = s.ex[k] - s.rx[k], dy = s.ey[k] - s.ry[k];
        along += dx * ux + dy * uy;
        const double c = -dx * uy + dy * ux;
        cross_sq += c * c;
        ++moving;
    }
    add("along_mean_mm", moving ? along / static_cast<double>(moving) * kMm : m::kNaN);
    add("cross_rms_mm",
        moving ? std::sqrt(cross_sq / static_cast<double>(moving)) * kMm : m::kNaN);

    // --- Step response ------------------------------------------------------
    // The EE's progress along the step as a fraction of its length, signed (so
    // overshoot reads above 1), from its own start. NaN columns when the run
    // had no step, so every row of a study has the same columns.
    double rise = m::kNaN, overshoot = m::kNaN, settle = m::kNaN, sse = m::kNaN;
    const double step_len = std::hypot(ctx.step_dx, ctx.step_dy);
    if (ctx.has_step && step_len > 0.0 && n > 1) {
        const double ux = ctx.step_dx / step_len, uy = ctx.step_dy / step_len;
        std::vector<double> r(n);
        for (std::size_t k = 0; k < n; ++k) {
            r[k] = ((s.ex[k] - s.ex[0]) * ux + (s.ey[k] - s.ey[0]) * uy) / step_len;
        }
        const m::StepResponse sr = m::stepResponse(s.t, r, ctx.step_t, 0.02);
        rise = sr.rise_time;
        overshoot = sr.overshoot_pct;
        settle = sr.settle_time;
        sse = sr.steady_state_error * step_len * kMm;
    }
    add("step_rise_s", rise);
    add("step_overshoot_pct", overshoot);
    add("step_settle_s", settle);
    add("step_sse_mm", sse);

    // --- Control effort -----------------------------------------------------
    double tau2_win = 0.0, tau2_total = 0.0, tv = 0.0, work = 0.0;
    for (std::size_t j = 0; j < s.tau.size(); ++j) {
        tau2_win += m::integralSquared(s.t, s.tau[j], from);
        tau2_total += m::integralSquared(s.t, s.tau[j], 0);
        tv += m::totalVariation(s.tau[j], from);
        std::vector<double> power(s.tau[j].size());
        for (std::size_t k = 0; k < power.size(); ++k) power[k] = s.tau[j][k] * s.qd[j][k];
        work += m::integralAbs(s.t, power, from);
    }
    add("tau2_int_win", tau2_win);
    add("tau2_int_total", tau2_total);
    const std::vector<double> tau_pool = pooled(s.tau, from);
    add("tau_var", m::variance(tau_pool));
    add("tau_peak", m::peakAbs(tau_pool));
    double rail = 0.0, rail_n = 0.0;
    for (std::size_t j = 0; j < s.tau.size(); ++j) {
        const double lim = j < s.tau_limit.size() ? s.tau_limit[j] : 0.0;
        for (std::size_t k = from; k < s.tau[j].size(); ++k) {
            rail_n += 1.0;
            if (lim > 0.0 && std::fabs(s.tau[j][k]) >= 0.99 * lim) rail += 1.0;
        }
    }
    add("rail_frac", rail_n > 0.0 ? rail / rail_n : m::kNaN);
    add("chatter_tv", tv);
    double step_sq = 0.0, step_n = 0.0;
    for (const auto& j : s.tau) {
        for (std::size_t k = from; k + 1 < j.size(); ++k) {
            const double d = j[k + 1] - j[k];
            step_sq += d * d;
            step_n += 1.0;
        }
    }
    add("tau_step_rms", step_n > 0.0 ? std::sqrt(step_sq / step_n) : m::kNaN);
    add("work_int", work);

    // --- Joint limits -------------------------------------------------------
    // How far outside its CONFIGURED box the worst owned joint was, tick by tick
    // (0 inside), over the whole run: a joint that starts outside and is brought
    // back shows up in the integral and the final value. NaN when the positions
    // or the limits were not recorded.
    const bool have_q = !s.q.empty() && s.q.size() == s.q_min.size() &&
                        s.q.size() == s.q_max.size() && s.q[0].size() == n && n > 0;
    std::vector<double> viol;
    if (have_q) {
        viol.assign(n, 0.0);
        for (std::size_t j = 0; j < s.q.size(); ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                const double v = std::max({0.0, s.q_min[j] - s.q[j][k], s.q[j][k] - s.q_max[j]});
                viol[k] = std::max(viol[k], v);
            }
        }
    }
    add("qlim_peak_rad", have_q ? m::peakAbs(viol) : m::kNaN);
    add("qlim_int_rad_s", have_q ? m::integral(s.t, viol) : m::kNaN);
    add("qlim_outside_frac", have_q ? m::fractionAtLeast(viol, 1e-6) : m::kNaN);
    add("qlim_final_rad", have_q ? viol.back() : m::kNaN);
    double stop = 0.0;
    for (const int c : s.ncon) stop += c > 0 ? 1.0 : 0.0;
    add("stop_frac", s.ncon.empty() ? m::kNaN : stop / static_cast<double>(s.ncon.size()));

    // --- Base attitude ------------------------------------------------------
    // Yaw offset from where the base started, unwrapped so a turn through the
    // branch cut is not read as a full revolution.
    std::vector<double> psi = m::unwrap(s.base_yaw);
    const double psi0 = psi.empty() ? 0.0 : psi.front();
    for (double& p : psi) p -= psi0;
    add("yaw2_int_total", m::integralSquared(s.t, psi, 0));
    add("yaw2_int_win", m::integralSquared(s.t, psi, from));
    add("yaw_peak_mrad", m::peakAbs(psi) * 1e3);
    add("yaw_final_mrad", psi.empty() ? m::kNaN : psi.back() * 1e3);
    add("wz2_int_total", m::integralSquared(s.t, s.base_wz, 0));
    add("wz_abs_int_total", m::integralAbs(s.t, s.base_wz, 0));
    add("base_drift_m", n ? std::hypot(s.base_x.back() - s.base_x.front(),
                                       s.base_y.back() - s.base_y.front())
                          : m::kNaN);

    // --- What the controller was given, against the plant's truth -----------
    // Empty when the controller read the plant directly, so these are NaN then.
    const bool have_est = !s.est_x.empty();
    add("est_pos_rms_mm", have_est
            ? std::sqrt(m::meanSquare(s.est_x, from) + m::meanSquare(s.est_y, from)) * kMm
            : m::kNaN);
    add("est_yaw_rms_mrad", have_est ? m::rms(s.est_yaw, from) * 1e3 : m::kNaN);
    add("est_vel_rms_mm_s", have_est
            ? std::sqrt(m::meanSquare(s.est_vx, from) + m::meanSquare(s.est_vy, from)) * kMm
            : m::kNaN);
    add("est_wz_rms_mrad_s", have_est ? m::rms(s.est_wz, from) * 1e3 : m::kNaN);

    // --- Timing -------------------------------------------------------------
    add("tick_us_mean", m::mean(s.tick_us));
    add("tick_us_p50", m::percentile(s.tick_us, 50));
    add("tick_us_p95", m::percentile(s.tick_us, 95));
    add("tick_us_p99", m::percentile(s.tick_us, 99));
    add("tick_us_max", m::peakAbs(s.tick_us));

    // --- Health -------------------------------------------------------------
    add("safe_ticks", static_cast<double>(ctx.safe_ticks));
    add("tier1_entries", static_cast<double>(ctx.tier1_entries));
    add("ref_outside", static_cast<double>(ctx.ref_outside));
    add("ff_saturated", static_cast<double>(ctx.ff_saturated));
    add("ff_slew_limited", static_cast<double>(ctx.ff_slew_limited));
    add("qp_iter_mean", ctx.has_solver ? ctx.qp_iter_mean : m::kNaN);
    add("qp_iter_max", ctx.has_solver ? ctx.qp_iter_max : m::kNaN);
    return v;
}

void writeSeriesCsv(const std::string& path, const RunSeries& s) {
    std::ofstream os(path);
    if (!os) throw std::runtime_error("cannot write " + path);
    os << "t_s,ref_x,ref_y,ref_vx,ref_vy,ee_x,ee_y,err_m,eth_rad";
    for (std::size_t j = 0; j < s.tau.size(); ++j) os << ",tau" << j;
    for (std::size_t j = 0; j < s.qd.size(); ++j) os << ",qd" << j;
    const bool have_q = !s.q.empty() && s.q[0].size() == s.size();
    if (have_q) {
        for (std::size_t j = 0; j < s.q.size(); ++j) os << ",q" << j;
    }
    const bool have_ncon = s.ncon.size() == s.size();
    if (have_ncon) os << ",ncon";
    os << ",base_x,base_y,base_yaw,base_wz,tick_us,mode";
    const bool have_est = !s.est_x.empty();
    if (have_est) os << ",est_x,est_y,est_yaw,est_vx,est_vy,est_wz";
    const bool have_mom = s.mom_px.size() == s.size() && s.size() > 0;
    if (have_mom) os << ",mom_px,mom_py,mom_lz,bus_px,bus_py,bus_lz";
    const bool have_ke = s.ke.size() == s.size() && s.size() > 0;
    if (have_ke) os << ",ke_j";
    os << '\n';
    char buf[64];
    const auto put = [&](double x) {
        std::snprintf(buf, sizeof buf, "%.9g", x);
        os << buf;
    };
    for (std::size_t k = 0; k < s.size(); ++k) {
        put(s.t[k]);
        for (const double x : {s.rx[k], s.ry[k], s.rvx[k], s.rvy[k], s.ex[k], s.ey[k]}) {
            os << ',';
            put(x);
        }
        os << ',';
        put(std::hypot(s.ex[k] - s.rx[k], s.ey[k] - s.ry[k]));
        os << ',';
        put(s.eth[k]);
        for (const auto& j : s.tau) { os << ','; put(j[k]); }
        for (const auto& j : s.qd)  { os << ','; put(j[k]); }
        if (have_q) {
            for (const auto& j : s.q) { os << ','; put(j[k]); }
        }
        if (have_ncon) os << ',' << s.ncon[k];
        for (const double x : {s.base_x[k], s.base_y[k], s.base_yaw[k], s.base_wz[k],
                               s.tick_us[k]}) {
            os << ',';
            put(x);
        }
        os << ',' << s.mode[k];
        if (have_est) {
            for (const double x : {s.est_x[k], s.est_y[k], s.est_yaw[k], s.est_vx[k],
                                   s.est_vy[k], s.est_wz[k]}) {
                os << ',';
                put(x);
            }
        }
        if (have_mom) {
            for (const double x : {s.mom_px[k], s.mom_py[k], s.mom_lz[k], s.bus_px[k],
                                   s.bus_py[k], s.bus_lz[k]}) {
                os << ',';
                put(x);
            }
        }
        if (have_ke) {
            os << ',';
            put(s.ke[k]);
        }
        os << '\n';
    }
}

void appendSummaryCsv(const std::string& path, const RunContext& ctx,
                      const NamedValues& values) {
    bool need_header = true;
    {
        std::ifstream probe(path);
        need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream os(path, std::ios::app);
    if (!os) throw std::runtime_error("cannot write " + path);
    if (need_header) {
        os << "scenario,law,tag,seed";
        for (const auto& [label, text] : ctx.labels) {
            (void)text;
            os << ',' << label;
        }
        for (const auto& [name, value] : values) {
            (void)value;
            os << ',' << name;
        }
        os << '\n';
    }
    os << ctx.scenario << ',' << ctx.law << ',' << ctx.tag << ',' << ctx.seed;
    for (const auto& [label, text] : ctx.labels) {
        (void)label;
        os << ',' << text;
    }
    char buf[64];
    for (const auto& [name, value] : values) {
        (void)name;
        std::snprintf(buf, sizeof buf, "%.9g", value);
        os << ',' << buf;
    }
    os << '\n';
}

}  // namespace bench
}  // namespace pat_arm_control
