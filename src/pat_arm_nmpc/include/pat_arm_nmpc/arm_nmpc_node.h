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

    std::vector<double> q_joints_, v_joints_;     // sized to model_joint_names_
    std::vector<double> ee_quat_{1.0, 0.0, 0.0, 0.0};   // desired EE quat [w,x,y,z]
    double ee_x_{0.0}, ee_y_{0.0};                      // desired EE position [m]
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
    double base_x_{0.0}, base_y_{0.0}, base_yaw_{0.0};
    double base_vx_{0.0}, base_vy_{0.0}, base_omega_{0.0};
    bool has_state_{false}, has_odom_{false}, has_setpoint_{false};

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

    std::mutex mu_;
};
