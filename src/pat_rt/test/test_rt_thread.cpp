/*! Tests for rt_thread.h.
 *
 *  The mailbox and ring are tested in pat_arm_control's suite (RtMailbox.*,
 *  SpscRing.*), which predates this package and exercises them under the
 *  control loop's own access pattern. These cover the thread helpers.
 */
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "pat_rt/rt_thread.h"

using pat_rt::PeriodicWaker;

TEST(RtThread, NotRequestedIsReportedAsMinusOne) {
    // -1 is "nothing was asked for", distinct from 0 (success) and from an
    // errno, so a node's report can say "not requested" rather than "failed".
    EXPECT_EQ(pat_rt::applyRealtimePolicy(0), -1);
    EXPECT_EQ(pat_rt::applyRealtimePolicy(-5), -1);
    EXPECT_EQ(pat_rt::applyAffinity({}), -1);
}

TEST(RtThread, AffinityWithOnlyOutOfRangeCpusIsEinval) {
    EXPECT_EQ(pat_rt::applyAffinity({-1, CPU_SETSIZE + 3}), EINVAL);
}

TEST(RtThread, AffinityToTheCurrentCpuSucceedsAndIsApplied) {
    // Run on a scratch thread so the test runner's own mask is untouched.
    int rc = -2;
    int cpu_after = -1;
    std::thread t([&] {
        const int cpu = sched_getcpu();
        rc = pat_rt::applyAffinity({cpu});
        cpu_after = sched_getcpu();
        // Pinned to one CPU, the thread is there and nowhere else.
        if (rc == 0 && cpu_after != cpu) cpu_after = -100;
    });
    t.join();
    EXPECT_EQ(rc, 0);
    EXPECT_GE(cpu_after, 0);
}

TEST(PeriodicWaker, HoldsTheRateOverManyPeriods) {
    // 5 ms period, 40 wakes: the schedule is absolute, so the total elapsed time
    // is 40 periods regardless of how long each iteration's work took.
    constexpr long kPeriodNs = 5'000'000;
    constexpr int kWakes = 40;
    PeriodicWaker w(kPeriodNs);
    const auto t0 = std::chrono::steady_clock::now();
    w.start();
    for (int i = 0; i < kWakes; ++i) w.waitNext();
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double expected = kWakes * kPeriodNs * 1e-9;
    EXPECT_GE(elapsed, expected * 0.999);
    EXPECT_LT(elapsed, expected + 0.05) << "a relative sleep would accumulate drift";
}

TEST(PeriodicWaker, AnOverrunSkipsLostPeriodsRatherThanCatchingUp) {
    // One iteration overruns by ~3.5 periods. A catch-up schedule would then
    // fire the next few wakes back-to-back; this one steps to the next FUTURE
    // deadline, so the wake after the overrun is still a real sleep.
    constexpr long kPeriodNs = 10'000'000;
    PeriodicWaker w(kPeriodNs);
    w.start();
    w.waitNext();
    std::this_thread::sleep_for(std::chrono::milliseconds(35));   // overrun
    w.waitNext();                                                  // resyncs
    const auto t1 = std::chrono::steady_clock::now();
    w.waitNext();                                                  // one full period
    const double gap =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    EXPECT_GT(gap, 0.008) << "fired back-to-back: the schedule tried to catch up";
}

TEST(PeriodicWaker, CheckingAfterTheWorkWouldHalveTheRate) {
    // The resync runs before the sleep. Run for 20 periods with no work and
    // count wakes: exactly 20, not 10.
    constexpr long kPeriodNs = 4'000'000;
    PeriodicWaker w(kPeriodNs);
    std::atomic<int> wakes{0};
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::nanoseconds(20 * kPeriodNs + 2'000'000);
    w.start();
    while (std::chrono::steady_clock::now() < deadline) {
        w.waitNext();
        ++wakes;
    }
    EXPECT_GE(wakes.load(), 19);
    EXPECT_LE(wakes.load(), 22);
}
