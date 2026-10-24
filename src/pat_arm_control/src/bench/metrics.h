#pragma once
/*! The figures of merit for comparing control laws, as pure functions of
 *  recorded time series.
 *
 *  Kept apart from the bench so that each definition is unit-tested against a
 *  signal whose answer is known (test_bench_metrics.cpp): a metric that is
 *  slightly wrong produces a confident, plausible, wrong number, and nothing
 *  downstream would catch it.
 *
 *  Conventions, all of them stated here once rather than at each use:
 *    - Series are sampled at the control tick, with times in seconds.
 *    - `from` is the index of the first sample to include; everything before it
 *      is ignored. It is how a scoring window (after the approach, after the
 *      transient) is expressed.
 *    - variance() is the POPULATION variance (divide by n). The series here are
 *      thousands of samples long, where n and n-1 are indistinguishable, and a
 *      deterministic run has no sampling to correct for.
 *    - Integrals are trapezoidal in the actual sample times.
 *    - Anything undefined (an empty window, a threshold never reached) is NaN,
 *      never 0, so a missing value cannot read as a good one.
 */
#include <cstddef>
#include <limits>
#include <vector>

namespace pat_arm_control {
namespace bench {
namespace metrics {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

//! Arithmetic mean of x[from:]. NaN if the window is empty.
double mean(const std::vector<double>& x, std::size_t from = 0);
//! Population variance of x[from:].
double variance(const std::vector<double>& x, std::size_t from = 0);
//! Mean square, mean(x^2): the MSE when x is an error.
double meanSquare(const std::vector<double>& x, std::size_t from = 0);
//! sqrt(meanSquare).
double rms(const std::vector<double>& x, std::size_t from = 0);
//! max |x| over x[from:].
double peakAbs(const std::vector<double>& x, std::size_t from = 0);

//! Integral of x dt over the window, trapezoidal.
double integral(const std::vector<double>& t, const std::vector<double>& x,
                std::size_t from = 0);
//! Integral of x^2 dt: the "integral of squared control input" for x = torque,
//! and the attitude-disturbance energy for x = base yaw offset.
double integralSquared(const std::vector<double>& t, const std::vector<double>& x,
                       std::size_t from = 0);
//! Integral of |x| dt.
double integralAbs(const std::vector<double>& t, const std::vector<double>& x,
                   std::size_t from = 0);
//! Total variation, sum |x[k+1]-x[k]|: how much a command moves, regardless of
//! where it sits. Large for a chattering command, small for a smooth one that
//! reaches the same level.
double totalVariation(const std::vector<double>& x, std::size_t from = 0);
//! rms of the tick-to-tick change, sqrt(mean((x[k+1]-x[k])^2)).
double rmsStep(const std::vector<double>& x, std::size_t from = 0);
//! Fraction of samples in x[from:] whose magnitude is at least `level`.
double fractionAtLeast(const std::vector<double>& x, double level, std::size_t from = 0);

//! p-th percentile (0..100) of x[from:], linearly interpolated between order
//! statistics (numpy's default), so it matches what a plot script computes.
double percentile(const std::vector<double>& x, double p, std::size_t from = 0);

/*! Angles with the 2*pi jumps removed, so a base that turns through the branch
 *  cut does not register as a full-turn excursion. */
std::vector<double> unwrap(const std::vector<double>& angles);

/*! Index of the first sample with t >= t_start; t.size() if there is none. */
std::size_t firstIndexAtOrAfter(const std::vector<double>& t, double t_start);

/*! Shape of a step response, from the NORMALISED progress r(t) running from 0
 *  to 1 (the fraction of the step covered, signed, so overshoot shows as r > 1).
 *  All times are measured from t_step. */
struct StepResponse {
    double rise_time = kNaN;       //!< time from r = 10% to r = 90% [s]
    double overshoot_pct = kNaN;   //!< (max r - 1) * 100, 0 if r never exceeds 1
    double settle_time = kNaN;     //!< until r stays within 1 +/- tol to the end [s]
    double steady_state_error = kNaN;   //!< |1 - mean r| over the last tenth of the record
};
/*! `t` and `r` are the full record; samples before t_step are ignored. `tol` is
 *  the settling band (0.02 = 2%). The settle time is NaN if the response is
 *  still outside the band at the end of the record. */
StepResponse stepResponse(const std::vector<double>& t, const std::vector<double>& r,
                          double t_step, double tol = 0.02);

}  // namespace metrics
}  // namespace bench
}  // namespace pat_arm_control
