#pragma once
/*! Realtime thread primitives shared by every node that runs a periodic loop on
 *  its own thread: the arm controller's control and observer threads, and the
 *  base navigator's estimator thread.
 *
 *  Header-only and ROS-free. Every function reports its outcome as a return
 *  code instead of logging, because the threads that call them must not log;
 *  each node latches the code and reports it once from its executor.
 *
 *  Linux-specific (pthread affinity, SCHED_FIFO, mlockall, CLOCK_MONOTONIC),
 *  which is every target this workspace deploys to.
 */
#include <cerrno>
#include <ctime>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

namespace pat_rt {

/*! Request SCHED_FIFO at `priority` for the CALLING thread.
 *
 *  Returns 0 on success, -1 if `priority` <= 0 (not requested), otherwise the
 *  errno-style code. Failing must not stop the caller: the limit is
 *  RLIMIT_RTPRIO, which a container need not grant, and a node that refuses to
 *  run in the development environment is worse than one that reports it is not
 *  realtime.
 *
 *  Call from inside the thread, so only that thread is affected and nothing it
 *  later creates inherits the policy. Call it AFTER applyAffinity: the realtime
 *  scheduler respects an affinity mask but will not choose a fast core on its
 *  own.
 */
inline int applyRealtimePolicy(int priority) {
    if (priority <= 0) return -1;             // not requested
    sched_param sp{};
    sp.sched_priority = priority;
    // pthread_setschedparam returns the error rather than setting errno.
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
}

/*! Restrict the CALLING thread to `cpus`.
 *
 *  Returns 0 on success, -1 if `cpus` is empty (not requested), EINVAL if every
 *  entry was out of range, otherwise the errno-style code.
 *
 *  Not optional on a hybrid-core x86 host, and measured: with SCHED_FIFO and no
 *  affinity, both arms' control threads ran exclusively on the dev box's
 *  E-cores and the mean tick doubled, 1080 -> 2118 us. Linux's realtime
 *  scheduler places by priority and which CPU is free, with no notion of the
 *  core-capability asymmetry that CFS handles through Thread Director, so an
 *  unpinned FIFO thread can land on the slowest cores in the machine and stay
 *  there. Takes the set as an argument so each thread of a node can be pinned
 *  to its own cores.
 */
inline int applyAffinity(const std::vector<long>& cpus) {
    if (cpus.empty()) return -1;              // not requested
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const long cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(static_cast<int>(cpu), &set);
    }
    if (CPU_COUNT(&set) == 0) return EINVAL;   // every entry was out of range
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

/*! Lock the process's pages, so a major fault cannot stall a periodic loop.
 *
 *  Returns 0 on success, otherwise errno. Process-wide, so call it once from
 *  the node before starting its threads, not from a thread.
 *
 *  MCL_ONFAULT is what makes the limit affordable: MCL_CURRENT|MCL_FUTURE alone
 *  locks the whole ADDRESS SPACE, which for the arm node is ~620 MB of
 *  mostly-untouched reservation against ~33 MB actually resident, and it fails
 *  with ENOMEM against any sane RLIMIT_MEMLOCK. ONFAULT locks each page as it is
 *  first touched instead, so the locked set tracks the resident set. The cost is
 *  that the first touch of a page still faults -- the startup iterations pay
 *  it, and everything the loop reaches stays locked thereafter.
 */
inline int lockMemory() {
    const int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT;
    return mlockall(flags) == 0 ? 0 : errno;
}

/*! Wakes a thread on a fixed schedule of absolute deadlines.
 *
 *  Absolute rather than relative (clock_nanosleep with TIMER_ABSTIME), so a
 *  late iteration does not shift every iteration after it: the schedule is
 *  fixed and lateness shows up as lateness. A relative sleep -- or rclcpp's
 *  wall timer -- re-anchors instead, which hides the overrun and reports the
 *  permanent offset as jitter.
 *
 *  Usage, from inside the thread:
 *
 *      PeriodicWaker waker(period_ns);
 *      waker.start();
 *      while (running) {
 *          waker.waitNext();
 *          if (!running) break;
 *          step();
 *      }
 *
 *  Allocation-free and syscall-only: safe on a realtime thread.
 */
class PeriodicWaker {
public:
    explicit PeriodicWaker(long period_ns) : period_ns_(period_ns) {}

    //! Anchor the schedule at the current time. The first waitNext() then
    //! returns one period from now.
    void start() { clock_gettime(CLOCK_MONOTONIC, &next_); }

    /*! Sleep until the next deadline.
     *
     *  If that deadline is ALREADY past -- the previous iteration overran --
     *  the schedule steps forward to the next future one rather than firing
     *  back-to-back to catch up: those periods are lost either way, and running
     *  them at full rate only compounds the overrun. The check runs here,
     *  after the advance and BEFORE the sleep. Checking after the work instead
     *  would always be true, since the deadline just woken for is by
     *  definition in the past, and would step the schedule twice per
     *  iteration and halve the rate. */
    void waitNext() {
        addPeriod(next_);
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        while (isBefore(next_, now)) addPeriod(next_);
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_, nullptr)
               == EINTR) {
            // A signal, not a deadline change: wait out the rest of the period.
        }
    }

    long periodNs() const { return period_ns_; }

private:
    // Advance `t` by exactly one period. Written out rather than using
    // std::chrono so the deadline stays the timespec clock_nanosleep wants.
    void addPeriod(timespec& t) const {
        t.tv_nsec += period_ns_;
        while (t.tv_nsec >= 1000000000L) {
            t.tv_nsec -= 1000000000L;
            ++t.tv_sec;
        }
    }
    static bool isBefore(const timespec& a, const timespec& b) {
        return a.tv_sec < b.tv_sec ||
               (a.tv_sec == b.tv_sec && a.tv_nsec < b.tv_nsec);
    }

    long     period_ns_;
    timespec next_{};
};

}  // namespace pat_rt
