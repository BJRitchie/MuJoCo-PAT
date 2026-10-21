// ArmControlNode: the 1 Hz diagnostics report and the per-tick trace writer.
// Both run on the executor, never on the control thread. The line formats are
// parsed by tools/nmpc_timing (summarise_timing.py, plot_trace.py,
// tail_source.py): change them together.
#include "arm_control_node.h"

#include <cstring>
#include <iomanip>
#include <sstream>

#include "pat_arm_control/control/task_space_controller.h"

void ArmControlNode::flushTrace() {
    if (!trace_out_ || !controller_) return;
    controller_->drainTrace(trace_rows_);
    if (trace_rows_.empty()) return;

    trace_out_ << std::fixed << std::setprecision(2);
    for (const auto& r : trace_rows_) {
        trace_out_ << r.t_us << ',' << r.tick_us << ',' << r.dynamics << ','
                   << r.jdotqdot << ',' << r.lambda << ',' << r.qpsolve << ','
                   << r.rollout_dynamics << ',' << r.rollout_integrate << ','
                   << r.pinv << ',' << r.build_model << ',' << r.defect << ','
                   << r.trust_region << ',' << r.finalize << ',' << r.hpipm_us
                   << ',' << r.qp_iters << ',' << r.qp_status
                   << ',' << r.qp_iters1 << ',' << r.qp_status1 << ','
                   << static_cast<int>(r.mode) << ','
                   << static_cast<int>(r.retried) << '\n';
    }
    trace_out_.flush();
    trace_written_ += static_cast<long>(trace_rows_.size());
}

void ArmControlNode::reportDiagnostics() {
    const char* tag  = log_name_.c_str();
    const char* side = arm_side_.c_str();

    // What the control thread got, said once. Reported rather than enforced: a
    // controller that will not start without a realtime policy is useless in a
    // development container, so it runs either way and says which it is.
    if (!rt_reported_.exchange(true, std::memory_order_acq_rel)) {
        const int rt = rt_result_.load(std::memory_order_relaxed);
        if (rt == 0) {
            RCLCPP_INFO(get_logger(),
                "%s[%s]: control thread at SCHED_FIFO priority %d",
                tag, side, rt_priority_);
        } else if (rt > 0) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: SCHED_FIFO priority %d refused (%s) — running at "
                "normal priority. Needs RLIMIT_RTPRIO >= %d (docker-compose.yml's "
                "ulimits.rtprio); this is NOT a realtime loop",
                tag, side, rt_priority_, std::strerror(rt), rt_priority_);
        } else {
            RCLCPP_INFO(get_logger(),
                "%s[%s]: control thread at normal priority "
                "(rt_priority is 0)", tag, side);
        }
        const int aff = affinity_result_.load(std::memory_order_relaxed);
        if (aff == 0) {
            std::string cpus;
            for (const long c : rt_cpus_) cpus += std::to_string(c) + " ";
            RCLCPP_INFO(get_logger(), "%s[%s]: control thread pinned to CPUs %s",
                        tag, side, cpus.c_str());
        } else if (aff > 0) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: CPU affinity failed (%s)",
                tag, side, std::strerror(aff));
        } else if (rt_result_.load(std::memory_order_relaxed) == 0) {
            // FIFO without a mask is the case that measured as a regression.
            RCLCPP_WARN(get_logger(),
                "%s[%s]: SCHED_FIFO with no rt_cpus — on a host whose "
                "cores differ in speed the realtime scheduler may park this thread "
                "on the slowest ones. Set rt_cpus to the fast cores",
                tag, side);
        }
        const int ml = mlock_result_.load(std::memory_order_relaxed);
        if (ml == 0) {
            RCLCPP_INFO(get_logger(), "%s[%s]: memory locked (mlockall)",
                        tag, side);
        } else if (ml > 0) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: mlockall failed (%s) — a page fault can still "
                "stall a tick. Needs ulimits.memlock",
                tag, side, std::strerror(ml));
        }
    }

    // The observer thread's own placement, reported once and separately: the
    // two threads are only useful on DIFFERENT cores, so saying which each got
    // is what makes a misconfiguration visible rather than merely slow.
    if (observer_ && !obs_reported_.exchange(true, std::memory_order_acq_rel)) {
        const int rt = obs_rt_result_.load(std::memory_order_relaxed);
        if (rt == 0) {
            RCLCPP_INFO(get_logger(),
                "%s[%s]: observer thread at SCHED_FIFO priority %d",
                tag, side, obs_rt_priority_);
        } else if (rt > 0) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: observer SCHED_FIFO priority %d refused (%s) — "
                "running at normal priority",
                tag, side, obs_rt_priority_, std::strerror(rt));
        }
        const int aff = obs_affinity_result_.load(std::memory_order_relaxed);
        if (aff == 0) {
            std::string cpus;
            for (const long c : obs_rt_cpus_) cpus += std::to_string(c) + " ";
            RCLCPP_INFO(get_logger(),
                        "%s[%s]: observer thread pinned to CPUs %s",
                        tag, side, cpus.c_str());
        } else if (aff > 0) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: observer CPU affinity failed (%s)",
                tag, side, std::strerror(aff));
        } else if (!obs_rt_cpus_.empty() || !rt_cpus_.empty()) {
            RCLCPP_WARN(get_logger(),
                "%s[%s]: observer has no rt_cpus_observer — it may share "
                "a core with the control thread, which is most of what having its "
                "own thread was for",
                tag, side);
        }
    }

    // The observer's own window. Separate from the control report because they
    // run on separate clocks: a step the observer skipped is a permanent offset
    // in its integral, so `no_state` rising is a correctness problem and not
    // merely a dropped sample.
    if (observer_) {
        DisturbanceSnapshot d;
        if (dist_mail_.read(d)) {
            RCLCPP_INFO(get_logger(),
                "%s[%s] observer: %ld steps (%ld without state), "
                "d_base [%.3f %.3f %.3f] d_joints[0] %.3f | max|Mdot| %.1f, "
                "clamped Mdot %ld est %ld%s",
                tag, side, d.steps, d.no_state,
                d.d_base[0], d.d_base[1], d.d_base[5],
                d.n > 0 ? d.d_joints[0] : 0.0,
                d.max_mdot, d.mdot_clamped, d.est_clamped,
                d.ready ? "" : " [not yet converged]");
        } else {
            RCLCPP_WARN(get_logger(),
                "%s[%s] observer: no estimate published yet",
                tag, side);
        }
    }

    // Everything the control tick recorded but could not log. Drained first, so
    // a window that then returns early below still gets these out.
    if (seeded_hold_pose_.exchange(false, std::memory_order_acq_rel)) {
        RCLCPP_INFO(get_logger(),
            "%s[%s]: no ee_setpoint yet — holding start pose (%.3f, %.3f)",
            tag, side, seeded_x_, seeded_y_);
    }
    if (const long n = n_waiting_for_input_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_WARN(get_logger(),
            "%s[%s]: waiting for joint_states and/or odom (%ld ticks)",
            tag, side, n);
    }
    if (const long n = n_tick_threw_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_ERROR(get_logger(),
            "%s[%s]: control tick threw %ld time(s), last: %s "
            "— commanded joint damping",
            tag, side, n, threw_msg_);
    }
    if (const long n = n_nonfinite_torque_.exchange(0, std::memory_order_relaxed)) {
        RCLCPP_ERROR(get_logger(),
            "%s[%s]: non-finite torque suppressed %ld time(s), commanded zero",
            tag, side, n);
    }
    // Only telemetry goes through a queue, so a drop here costs a missing
    // diagnostic rather than a stale actuator command -- still worth saying,
    // because it means the drain is not keeping up with the tick.
    const long dropped_ee = n_ee_pose_dropped_.exchange(0, std::memory_order_relaxed);
    const long dropped_ev = n_mode_events_dropped_.exchange(0, std::memory_order_relaxed);
    if (dropped_ee || dropped_ev) {
        RCLCPP_WARN(get_logger(),
            "%s[%s]: outbound queue overflow — dropped %ld ee_pose, "
            "%ld mode events; the drain is not keeping up",
            tag, side, dropped_ee, dropped_ev);
    }

    // Tier transitions, in the order the control loop saw them. Severity follows
    // the tier: a Tier 1 replay is a handled transient and should not read like a
    // fault.
    using Mode = pat_arm_control::IArmController::ControlMode;
    ModeEvent ev;
    while (mode_events_.pop(ev)) {
        switch (static_cast<Mode>(ev.mode)) {
        case Mode::PlanReplay:
            RCLCPP_WARN(get_logger(),
                "%s[%s]: TIER 1 — solve failed (QP status %d), replaying the "
                "last accepted plan",
                tag, side, ev.qp_status);
            break;
        case Mode::Damping:
            if (controller_->hasSolver()) {
                RCLCPP_ERROR(get_logger(),
                    "%s[%s]: TIER 2 — joint damping, task abandoned "
                    "(QP status %d, %ld replay ticks used so far)",
                    tag, side, ev.qp_status, ev.tier1_ticks);
            } else {
                RCLCPP_ERROR(get_logger(),
                    "%s[%s]: TIER 2 — joint damping, task abandoned "
                    "(the control tick failed)",
                    tag, side);
            }
            break;
        case Mode::Latched:
            if (controller_->hasSolver()) {
                RCLCPP_ERROR(get_logger(),
                    "%s[%s]: TIER 3 — LATCHED in damping; the solver did not "
                    "recover and this will NOT clear on its own",
                    tag, side);
            } else {
                RCLCPP_ERROR(get_logger(),
                    "%s[%s]: TIER 3 — LATCHED in damping; the control law did "
                    "not recover and this will NOT clear on its own",
                    tag, side);
            }
            break;
        case Mode::Nominal:
            RCLCPP_INFO(get_logger(),
                "%s[%s]: resumed tracking (%ld damped ticks, %ld replay "
                "ticks over %ld Tier 1 entries, in total)",
                tag, side, ev.safe_mode_ticks, ev.tier1_ticks,
                ev.tier1_entries);
            break;
        }
    }

    // Before the early return below: a window with no completed tick still has a
    // previous window's rows to get out, and the trace should not stall on it.
    flushTrace();

    StatsSnapshot s;
    if (!stats_mail_.read(s)) return;   // no tick has published yet
    if (s.loop.count == 0) return;      // no tick completed in this window

    RCLCPP_INFO(get_logger(),
        "%s[%s] %ld ticks @ %.0f us | period jitter mean %.0f max %.0f us | "
        "callback wall mean %.0f max %.0f / cpu mean %.0f max %.0f us | "
        "deadline misses %ld (longest run %ld) | skipped periods %ld\n%s",
        tag, side, s.loop.count, period_us_,
        s.jitter.mean(), s.jitter.max_us,
        s.loop.mean(), s.loop.max_us, s.cpu.mean(), s.cpu.max_us,
        s.deadline_misses, s.worst_consecutive_misses, s.skipped_periods,
        pat_arm_control::TaskSpaceController::formatProfile(s.profile).c_str());

    // Only once a trajectory has been seen, so a setpoint-only run's log is
    // unchanged. Counts are cumulative: a window that keeps running short of
    // the horizon shows as nodes held past it, a malformed one as rejected.
    const long rejected = traj_rejected_.load(std::memory_order_relaxed);
    const long truncated = traj_truncated_.load(std::memory_order_relaxed);
    if (s.ref_ticks > 0 || rejected > 0) {
        static const char* kSource[] = {"setpoint", "trajectory", "trajectory ended, holding"};
        RCLCPP_INFO(get_logger(),
            "%s[%s] reference: %s | %ld trajectory ticks, %ld node refs held "
            "past the window | windows rejected %ld, truncated %ld",
            tag, side, kSource[s.ref_source], s.ref_ticks, s.ref_outside,
            rejected, truncated);
    }

    // What publish() costs, with the histogram: the mean decides nothing here, the
    // worst case decides whether this call can go back on the control thread.
    if (pub_torque_.count > 0 || pub_eepose_.count > 0) {
        std::ostringstream pub;
        pub.setf(std::ios::fixed);
        pub.precision(1);
        const std::pair<const char*, const pat_arm_control::PhaseStat*> both[] = {
            {"torque", &pub_torque_}, {"ee_pose", &pub_eepose_},
        };
        for (const auto& [name, st] : both) {
            if (st->count == 0) continue;
            pub << "  " << name << "(n=" << st->count << " mean=" << st->mean()
                << " max=" << st->max_us << ") ";
        }
        pub << "\n  hist(publish_us) ";
        for (int i = 0; i < pat_arm_control::PhaseStat::kBins; ++i) {
            if (i + 1 == pat_arm_control::PhaseStat::kBins) pub << ">5000:";
            else pub << "<=" << pat_arm_control::PhaseStat::kEdgeUs[i] << ":";
            pub << (pub_torque_.bins[i] + pub_eepose_.bins[i]) << " ";
        }
        RCLCPP_INFO(get_logger(), "%s[%s] publish us%s",
                    tag, side, pub.str().c_str());
        pub_torque_.reset();
        pub_eepose_.reset();
    }

    // Ask the control thread to start a fresh window. It clears its own counters
    // right after publishing the next snapshot, so the reset never races a read.
    stats_reset_.store(true, std::memory_order_release);
}
