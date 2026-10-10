#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "pat_arm_nmpc/arm_nmpc.h"
#include "pat_arm_nmpc/rt_mailbox.h"

/// One ArmNMPC instance per arm (run two node processes, left + right).
///
/// Each tick the node assembles the full floating-base MuJoCo state
///   q = [base_pos(3), base_quat(4 wxyz), joint_angles(model N)]
///   v = [base_linvel(3), base_angvel(3), joint_vels(model N)]
/// from /chaser/odom (planar → 6-DOF slots, z/roll/pitch = 0) and the full
/// joint_states message, runs the controller, and publishes only its own
/// joints' effort on /chaser/arm/torque_command (name-matched, so the two
/// instances coexist on one topic).
class ArmNMPCNode : public rclcpp::Node {
public:
    /*! The measurements and the setpoint, handed from the subscription callbacks
     *  to the control loop one whole value at a time.
     *
     *  Fixed-size arrays rather than vectors because SeqlockMailbox copies these
     *  by plain assignment on both sides: a vector would allocate on the control
     *  thread, which is the one thing the hot path no longer does. The node
     *  refuses to construct if the model has more joints than kMaxJoints, so the
     *  bound is checked once rather than trusted.
     *
     *  Each is written by exactly one callback and read only by the control loop,
     *  which is what makes the single-producer/single-consumer requirement hold.
     *  They are separate mailboxes, not one struct, because three different
     *  callbacks produce them at three different rates. */
    static constexpr std::size_t kMaxJoints = 32;

    struct JointSnapshot {
        double        q[kMaxJoints]{};
        double        v[kMaxJoints]{};
        std::uint32_t n{0};          //!< how many of q/v are populated
    };
    struct BaseSnapshot {
        double x{0.0}, y{0.0}, yaw{0.0};
        double vx{0.0}, vy{0.0}, omega{0.0};
    };
    struct SetpointSnapshot {
        double x{0.0}, y{0.0};
        double quat[4]{1.0, 0.0, 0.0, 0.0};   //!< [w, x, y, z]
    };

    /*! What the tick produces, queued for a non-realtime thread to publish.
     *
     *  `rclcpp::Publisher::publish` is not safe to call from a realtime thread:
     *  depending on the middleware it can allocate, take a lock, or touch a
     *  socket. These carry the numbers out as POD and the drain builds the
     *  messages, so the tick's output costs one queue slot. */
    struct TorqueOut {
        double        effort[kMaxJoints]{};
        std::uint32_t n{0};
    };
    struct EePoseOut {
        double pos[3]{};
        double quat[4]{1.0, 0.0, 0.0, 0.0};
    };

    /*! A control-mode transition, queued so the report can log each one exactly
     *  once. The tick cannot log it itself -- RCLCPP_* formats and writes -- and
     *  the report cannot infer it by sampling `mode()` at 1 Hz, which would miss
     *  any transition that came and went inside a window. */
    struct ModeEvent {
        int  mode{0};              //!< pat_arm_nmpc::ArmNMPC::ControlMode
        int  qp_status{0};
        long tier1_ticks{0};
        long safe_mode_ticks{0};
        long tier1_entries{0};
    };

    ArmNMPCNode();
    //! Writes the per-tick trace out, if one was requested. In the destructor so a
    //! Ctrl-C'd run still saves: nothing is written while the loop is running.
    ~ArmNMPCNode() override;

private:
    // --- Parameter loading (nmpc.yaml → NMPCParams) ----------------------- //
    pat_arm_nmpc::NMPCParams loadParams();
    std::vector<double> getJointArray(const std::string& name);   // length == joint_names_
    std::array<double, 3> getVec3(const std::string& name, std::array<double, 3> def);

    // --- Runtime -------------------------------------------------------------- //
    /// Copy the msg entries whose name is in `names` into dst[i] (dst sized to
    /// names.size()); entries not present in the msg are left untouched.
    static void applyNamedValues(const sensor_msgs::msg::JointState::SharedPtr& msg,
                                 const std::vector<double>& values,
                                 const std::vector<std::string>& names,
                                 std::vector<double>& dst);
    void onJointState(sensor_msgs::msg::JointState::SharedPtr msg);
    void onOdom(nav_msgs::msg::Odometry::SharedPtr msg);
    void onEeSetpoint(geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void controlLoop();

    //! The clock every throttled log message is rate-limited against. Steady, NOT
    //! the node clock: launch files default use_sim_time:=true and nothing
    //! publishes /clock, so get_clock()->now() is frozen at zero and a throttle
    //! gated on it fires once and is then silent for the rest of the run -- which
    //! silently hid the "waiting for joint_states" and dropped-measurement
    //! warnings. Rate-limiting output is a wall-clock concern either way.
    rclcpp::Clock log_clock_{RCL_STEADY_TIME};

    std::string arm_side_;                        // "left" | "right"
    pat_arm_nmpc::NMPCParams params_;
    std::unique_ptr<pat_arm_nmpc::ArmNMPC> controller_;

    std::vector<std::string> joint_names_;        // this arm's joints (output, 3)
    std::vector<std::string> model_joint_names_;  // ALL model joints, model order (6)
    std::vector<std::size_t> owned_model_idx_;    // joint_names_[k] → model index

    //! The joint-state callback's own running copy, sized to model_joint_names_.
    //! Kept across messages because a JointState need not carry every joint --
    //! applyNamedValues leaves the absent ones alone -- so the accumulated array
    //! is the thing worth publishing, not the message. Touched only by that
    //! callback; the control loop sees it through joint_mail_.
    std::vector<double> q_joints_, v_joints_;
    // /chaser/odom reports chaser_x/y — the raw slide-joint values relative
    // to the chaser body's own MJCF anchor (air_bearing_table.xacro:
    // <body name="chaser" pos="-0.5 0 0.1525">), NOT the chaser's true
    // world-frame position. This controller's OWN internal dynamics model
    // (pat_platform_planar.xacro) has a <freejoint> body at pos="0 0 0", so
    // feeding odom's raw value straight into q(0)/q(1) makes the
    // controller's whole internal notion of "world frame" silently shifted
    // by this anchor — invisible as long as everything stays internal
    // (tracking only cares about relative error, target vs current EE, both
    // computed in the same shifted frame), but wrong for anything that
    // compares this stack's ee_pose/ee_setpoint against the REAL MuJoCo
    // world frame (confirmed via pat_simulation's ee_setpoint marker:
    // published ee_pose was off from the true site position by exactly
    // (0.5, 0, -0.1525), matching the anchor). base_anchor_xy corrects it at
    // the odom read (onOdom), before it ever reaches q(0)/q(1) — defaults to
    // 0,0 (backward compatible with a model whose chaser has no such
    // anchor).
    double base_anchor_x_{0.0}, base_anchor_y_{0.0};

    //! The hand-off itself. Written by onJointState / onOdom / onEeSetpoint,
    //! read by controlLoop, with no lock between them -- see rt_mailbox.h.
    pat_arm_nmpc::SeqlockMailbox<JointSnapshot>    joint_mail_;
    pat_arm_nmpc::SeqlockMailbox<BaseSnapshot>     base_mail_;
    pat_arm_nmpc::SeqlockMailbox<SetpointSnapshot> setpoint_mail_;

    //! The control loop's private working copies: filled from the mailboxes at
    //! the top of each tick and owned by that thread alone for the rest of it, so
    //! the state cannot change underneath a tick that is halfway through using it.
    //! Retained between ticks so a read that loses its race (or a producer that
    //! has gone quiet) leaves the loop running on the last good value rather than
    //! on a default-constructed one.
    JointSnapshot    joints_{};
    BaseSnapshot     base_{};
    bool             have_joints_{false}, have_base_{false};

    //! The desired pose the loop is tracking, also control-thread-private. Taken
    //! from setpoint_mail_ when a setpoint has ever been published, and otherwise
    //! seeded once from the measured pose so an unattended node holds station
    //! instead of driving to the origin. Private state rather than the shared
    //! setpoint, because the seeding is the control loop writing a value only it
    //! should own -- writing it back into the subscription's slot would make the
    //! loop a second producer on that mailbox.
    double des_x_{0.0}, des_y_{0.0};
    std::array<double, 4> des_quat_{1.0, 0.0, 0.0, 0.0};
    bool   have_des_{false};

    //! The state vectors the tick assembles. Members, not locals: a VectorXd per
    //! tick is an allocation per tick on the one thread that must not make them.
    Eigen::VectorXd q_, v_;

    //! Outbound queues and the timer that drains them. The drain runs at twice
    //! the control rate so a tick's output waits well under one period, and the
    //! queues are deep enough that a late drain backs up rather than drops.
    pat_arm_nmpc::SpscRing<TorqueOut, 16>  torque_out_;
    pat_arm_nmpc::SpscRing<EePoseOut, 16>  ee_pose_out_;
    pat_arm_nmpc::SpscRing<ModeEvent, 16>  mode_events_;
    rclcpp::TimerBase::SharedPtr           drain_tmr_;
    void drainOutputs();

    /*! Things the tick must report but cannot log, since RCLCPP_* formats and
     *  writes. Counted here and surfaced by the report instead. Atomic because
     *  the control thread increments them and the report reads and clears them;
     *  relaxed, because a diagnostic counter that is a tick out of date is still
     *  a correct diagnostic. */
    std::atomic<long> n_waiting_for_input_{0};   //!< ticks with no state yet
    std::atomic<long> n_tick_threw_{0};          //!< exceptions out of computeControl
    std::atomic<long> n_nonfinite_torque_{0};    //!< suppressed non-finite commands
    std::atomic<long> n_torque_dropped_{0};      //!< outbound queue was full
    std::atomic<long> n_ee_pose_dropped_{0};
    std::atomic<long> n_mode_events_dropped_{0};
    //! The last exception message, copied into a fixed buffer rather than a
    //! std::string so the tick neither allocates nor formats. Written by the
    //! control thread and read by the report, so a torn message would garble one
    //! log line rather than cost correctness.
    static constexpr std::size_t kMsgLen = 192;
    char threw_msg_[kMsgLen]{};
    //! Set once when the loop seeds its own hold pose, with the pose it chose, so
    //! the report can log what the tick could not.
    std::atomic<bool> seeded_hold_pose_{false};
    double seeded_x_{0.0}, seeded_y_{0.0};

    // Last mode seen, so safe-mode entry and recovery are logged once per
    // transition rather than once per control tick.
    pat_arm_nmpc::ArmNMPC::ControlMode prev_mode_{
        pat_arm_nmpc::ArmNMPC::ControlMode::Nominal};

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ee_setpoint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr torque_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;  // measured EE pose (world)
    rclcpp::TimerBase::SharedPtr ctrl_tmr_;

    // --- Loop-level timing -------------------------------------------------
    // Measured on steady_clock, never the ROS clock: launch files default
    // use_sim_time:=true and nothing publishes /clock, so get_clock()->now() is
    // frozen at zero and unusable for any elapsed-time measurement.
    //
    // The controller times its own internal phases; these are the numbers only
    // the node can see -- when the tick actually woke relative to when it should
    // have, how long the whole callback took, and whether it fit the period.
    double period_us_{0.0};
    std::chrono::steady_clock::time_point last_wake_{};
    bool   have_last_wake_{false};
    //! Jitter as the error in the interval between consecutive wakes, |dt -
    //! period|, not as the offset from an ideal schedule laid down at startup:
    //! rclcpp's wall timer re-anchors when it fires late, so a schedule offset
    //! acquired during the slow first ticks would persist for the whole run and
    //! be reported forever as jitter it is not. The interval is also what the
    //! control law actually experiences, and what cyclictest measures.
    pat_arm_nmpc::PhaseStat jitter_;
    pat_arm_nmpc::PhaseStat loop_;        //!< whole controlLoop callback, wall us
    //! The same callback in thread CPU time. Paired with loop_ this separates the
    //! two causes of a long tick that wall time alone conflates: cpu ~= wall means
    //! the work really took that long, cpu << wall means the thread was off the
    //! CPU. Which one dominates decides whether the fix is a faster solver or a
    //! realtime scheduling policy.
    pat_arm_nmpc::PhaseStat cpu_;
    static double threadCpuUs();
    long   deadline_misses_{0};
    long   consecutive_misses_{0};
    long   worst_consecutive_misses_{0};
    //! Periods lost to wakes that never happened: an interval of n periods means
    //! the controller ran open-loop for n-1 of them.
    long   skipped_periods_{0};

    //! Record the interval since the previous wake. Called first in controlLoop,
    //! so the startup ticks that return early still count.
    void recordWake(std::chrono::steady_clock::time_point wake);

    //! Low-rate timer that logs the timing report. Deliberately a separate
    //! callback: the control tick must not do formatted I/O.
    rclcpp::TimerBase::SharedPtr diag_tmr_;
    void reportDiagnostics();

    //! Per-tick trace. The control loop only stores into a preallocated buffer;
    //! the rows are drained and appended here from the diagnostics timer, once a
    //! second, so a killed process keeps everything up to the last drain. Opened
    //! once at construction, never reopened.
    std::string   trace_file_;
    std::ofstream trace_out_;
    //! Drain destination, reused so the swap inside drainTrace keeps its buffer.
    std::vector<pat_arm_nmpc::ArmNMPC::TickTrace> trace_rows_;
    long trace_written_{0};
    void flushTrace();

    //! Guards the diagnostics statistics only -- the PhaseStats, the deadline
    //! counters and the controller's profile, which the control tick accumulates
    //! and reportDiagnostics reads and resets. The measurement path no longer
    //! uses it; see the mailboxes above.
    //!
    //! This is the one cross-thread item the hand-off has not resolved. It is
    //! harmless today, because every callback still runs on the one executor
    //! thread, but a control loop moved to SCHED_FIFO must not take a lock that
    //! the diagnostics timer holds. Resolving it belongs with the thread change:
    //! the accumulation has to stay on the control thread, with the report
    //! reading a published snapshot and requesting the reset.
    std::mutex stats_mu_;
};
