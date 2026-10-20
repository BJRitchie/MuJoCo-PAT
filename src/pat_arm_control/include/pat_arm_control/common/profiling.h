#pragma once
/*! Fixed-cost timing instruments for code on a realtime control path.
 *
 *  Shared by every control law and by the node: each law times its own phases,
 *  the node times the whole callback, and both report through the same types.
 *  Nothing here allocates, sorts or branches on data size.
 */
#include <chrono>

namespace pat_arm_control {

/*! Timing accumulator for one phase of the control tick.
 *
 *  Carries a max and a histogram as well as the mean, because for a deadline the
 *  mean is nearly useless: a cumulative average over thousands of ticks barely
 *  moves for the one spike that blows the period. Everything here is fixed-cost —
 *  no allocation, no sorting, no branching on data size — so it is affordable on
 *  the control path.
 *
 *  Bin edges straddle a 5 ms period so the deadline itself is a boundary: the last
 *  two bins are "most of the period" and "over it". */
struct PhaseStat {
    static constexpr int kBins = 8;
    //! Upper edges in microseconds; the final bin is everything above 5 ms.
    static constexpr double kEdgeUs[kBins] = {50, 100, 250, 500, 1000, 2500, 5000, 1e30};

    double sum_us = 0.0;
    double max_us = 0.0;
    long   count  = 0;
    long   bins[kBins] = {};

    //! Deadline for this phase, in microseconds; 0 means "not a deadline". Counted
    //! exactly rather than read off the histogram, whose bins are far coarser than
    //! any period of interest and whose last one is open-ended.
    double budget_us = 0.0;
    long   over      = 0;

    void add(double us) {
        sum_us += us;
        ++count;
        if (us > max_us) max_us = us;
        if (budget_us > 0.0 && us > budget_us) ++over;
        for (int i = 0; i < kBins; ++i) {
            if (us <= kEdgeUs[i]) { ++bins[i]; break; }
        }
    }
    //! Clears the measurements but keeps the deadline — that is configuration.
    void reset() {
        const double budget = budget_us;
        *this     = PhaseStat{};
        budget_us = budget;
    }
    double mean() const { return count > 0 ? sum_us / static_cast<double>(count) : 0.0; }
};

/*! Mean and max of a quantity that is not a duration. Separate from PhaseStat
 *  rather than a reuse of it: PhaseStat's histogram edges are microseconds, and a
 *  count binned against them would read as a time. */
struct Tally {
    double sum   = 0.0;
    double max   = 0.0;
    long   count = 0;

    void add(double v) { sum += v; ++count; if (v > max) max = v; }
    void reset() { *this = Tally{}; }
    double mean() const { return count > 0 ? sum / static_cast<double>(count) : 0.0; }
};

//! RAII stopwatch feeding a PhaseStat. Kept out of the tick's own accounting: the
//! measurement itself is two clock reads and a handful of adds.
struct ScopedTimer {
    explicit ScopedTimer(PhaseStat& stat)
        : stat_(stat), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        const auto stop = std::chrono::steady_clock::now();
        stat_.add(std::chrono::duration<double, std::micro>(stop - start_).count());
    }
    PhaseStat& stat_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace pat_arm_control
