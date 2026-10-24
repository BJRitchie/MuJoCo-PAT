#include "metrics.h"

#include <algorithm>
#include <cmath>

namespace pat_arm_control {
namespace bench {
namespace metrics {

namespace {

//! Number of samples in the window, 0 if `from` is past the end.
std::size_t count(const std::vector<double>& x, std::size_t from) {
    return from < x.size() ? x.size() - from : 0;
}

//! Trapezoidal integral of f(x[k]) over the window.
template <class F>
double trapezoid(const std::vector<double>& t, const std::vector<double>& x,
                 std::size_t from, F f) {
    const std::size_t n = std::min(t.size(), x.size());
    if (from + 1 >= n) return 0.0;
    double sum = 0.0;
    for (std::size_t k = from; k + 1 < n; ++k) {
        sum += 0.5 * (f(x[k]) + f(x[k + 1])) * (t[k + 1] - t[k]);
    }
    return sum;
}

}  // namespace

double mean(const std::vector<double>& x, std::size_t from) {
    const std::size_t n = count(x, from);
    if (n == 0) return kNaN;
    double sum = 0.0;
    for (std::size_t k = from; k < x.size(); ++k) sum += x[k];
    return sum / static_cast<double>(n);
}

double variance(const std::vector<double>& x, std::size_t from) {
    const std::size_t n = count(x, from);
    if (n == 0) return kNaN;
    const double m = mean(x, from);
    double sum = 0.0;
    for (std::size_t k = from; k < x.size(); ++k) sum += (x[k] - m) * (x[k] - m);
    return sum / static_cast<double>(n);
}

double meanSquare(const std::vector<double>& x, std::size_t from) {
    const std::size_t n = count(x, from);
    if (n == 0) return kNaN;
    double sum = 0.0;
    for (std::size_t k = from; k < x.size(); ++k) sum += x[k] * x[k];
    return sum / static_cast<double>(n);
}

double rms(const std::vector<double>& x, std::size_t from) {
    return std::sqrt(meanSquare(x, from));
}

double peakAbs(const std::vector<double>& x, std::size_t from) {
    if (count(x, from) == 0) return kNaN;
    double peak = 0.0;
    for (std::size_t k = from; k < x.size(); ++k) peak = std::max(peak, std::fabs(x[k]));
    return peak;
}

double integral(const std::vector<double>& t, const std::vector<double>& x,
                std::size_t from) {
    return trapezoid(t, x, from, [](double v) { return v; });
}

double integralSquared(const std::vector<double>& t, const std::vector<double>& x,
                       std::size_t from) {
    return trapezoid(t, x, from, [](double v) { return v * v; });
}

double integralAbs(const std::vector<double>& t, const std::vector<double>& x,
                   std::size_t from) {
    return trapezoid(t, x, from, [](double v) { return std::fabs(v); });
}

double totalVariation(const std::vector<double>& x, std::size_t from) {
    if (count(x, from) < 2) return 0.0;
    double sum = 0.0;
    for (std::size_t k = from; k + 1 < x.size(); ++k) sum += std::fabs(x[k + 1] - x[k]);
    return sum;
}

double rmsStep(const std::vector<double>& x, std::size_t from) {
    if (count(x, from) < 2) return kNaN;
    double sum = 0.0;
    for (std::size_t k = from; k + 1 < x.size(); ++k) {
        const double d = x[k + 1] - x[k];
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(count(x, from) - 1));
}

double fractionAtLeast(const std::vector<double>& x, double level, std::size_t from) {
    const std::size_t n = count(x, from);
    if (n == 0) return kNaN;
    std::size_t hits = 0;
    for (std::size_t k = from; k < x.size(); ++k) {
        if (std::fabs(x[k]) >= level) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(n);
}

double percentile(const std::vector<double>& x, double p, std::size_t from) {
    const std::size_t n = count(x, from);
    if (n == 0) return kNaN;
    std::vector<double> s(x.begin() + static_cast<std::ptrdiff_t>(from), x.end());
    std::sort(s.begin(), s.end());
    const double pos = std::clamp(p, 0.0, 100.0) / 100.0 * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, n - 1);
    return s[lo] + (pos - static_cast<double>(lo)) * (s[hi] - s[lo]);
}

std::vector<double> unwrap(const std::vector<double>& angles) {
    std::vector<double> out(angles.size());
    double offset = 0.0;
    for (std::size_t k = 0; k < angles.size(); ++k) {
        if (k > 0) {
            const double d = angles[k] - angles[k - 1];
            // A jump of more than half a turn is the branch cut, not motion.
            offset -= 2.0 * M_PI * std::round(d / (2.0 * M_PI));
        }
        out[k] = angles[k] + offset;
    }
    return out;
}

std::size_t firstIndexAtOrAfter(const std::vector<double>& t, double t_start) {
    return static_cast<std::size_t>(
        std::lower_bound(t.begin(), t.end(), t_start) - t.begin());
}

StepResponse stepResponse(const std::vector<double>& t, const std::vector<double>& r,
                          double t_step, double tol) {
    StepResponse out;
    const std::size_t n = std::min(t.size(), r.size());
    const std::size_t i0 = firstIndexAtOrAfter(t, t_step);
    if (i0 + 1 >= n) return out;

    // First time r crosses `level` going up, linearly interpolated between the
    // two samples either side, measured from the step.
    const auto crossing = [&](double level) {
        for (std::size_t k = i0; k + 1 < n; ++k) {
            if (r[k] < level && r[k + 1] >= level) {
                const double f = (level - r[k]) / (r[k + 1] - r[k]);
                return t[k] + f * (t[k + 1] - t[k]) - t_step;
            }
        }
        return kNaN;
    };
    const double t10 = crossing(0.1), t90 = crossing(0.9);
    if (std::isfinite(t10) && std::isfinite(t90)) out.rise_time = t90 - t10;

    double peak = r[i0];
    for (std::size_t k = i0; k < n; ++k) peak = std::max(peak, r[k]);
    out.overshoot_pct = peak > 1.0 ? (peak - 1.0) * 100.0 : 0.0;

    // Settled once it never leaves the band again: the time of the LAST sample
    // outside it, so a late excursion is not forgiven by an early entry.
    std::size_t last_out = i0;
    bool ever_out = false;
    for (std::size_t k = i0; k < n; ++k) {
        if (std::fabs(r[k] - 1.0) > tol) { last_out = k; ever_out = true; }
    }
    if (!ever_out) {
        out.settle_time = 0.0;
    } else if (last_out + 1 < n) {
        out.settle_time = t[last_out + 1] - t_step;
    }   // else still outside the band at the end of the record: stays NaN

    const std::size_t tail = std::max<std::size_t>(1, (n - i0) / 10);
    double sum = 0.0;
    for (std::size_t k = n - tail; k < n; ++k) sum += r[k];
    out.steady_state_error = std::fabs(1.0 - sum / static_cast<double>(tail));
    return out;
}

}  // namespace metrics
}  // namespace bench
}  // namespace pat_arm_control
