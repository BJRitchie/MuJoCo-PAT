#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "pat_arm_control/planning/reference_trajectory.h"

/// Cycles each arm's task-space setpoint through a list of known-reachable
/// end-effector poses — the simplest form of the "mission/teleop node" the
/// topic contract lists as not-yet-built. Useful for bringing the arm
/// controllers up and eyeballing tracking without a real mission planner.
///
/// Modes:
///   "relative" (default) — waypoints are planar offsets (dx, dy, dyaw) from
///       the arm's measured start pose, read from /chaser/arm/<side>/ee_pose
///       (published by arm_nmpc_node). Small offsets are reachable by
///       construction and survive xacro geometry changes.
///   "absolute" — waypoints are world-frame poses (x, y, yaw), published
///       verbatim; no /ee_pose subscription, caller owns reachability.
///
/// Per side, waypoint k is held for `dwell_sec` and republished at
/// `republish_hz` (a lone PoseStamped races DDS discovery and is usually
/// dropped), then the index advances. `loop:=false` stops on the last one.
///
/// Patterns (`pattern`):
///   "waypoints" (default) — the waypoint loop above, on ee_setpoint.
///   "circle" / "rectangle" — a continuous trajectory, the SAME shape for both
///       arms, each placed from its own start pose (relative mode only). Built
///       by the controller library's ReferenceTrajectory, sampled `window_s`
///       ahead every `sample_dt` and published on ee_trajectory at
///       `trajectory_hz`, so the NMPC can score every node of its horizon
///       against where the target will be. The current reference point also
///       goes out on ee_setpoint, for the viewer's markers, telemetry and
///       anything else that tracks a single target.
class EeTargetPublisher : public rclcpp::Node {
public:
    EeTargetPublisher();

private:
    struct Waypoint { double a, b, yaw; };  // relative: (dx, dy, dyaw); absolute: (x, y, yaw)
    struct Arm {
        std::string side;
        std::vector<Waypoint> waypoints;
        std::size_t idx = 0;
        bool have_home = false;
        bool done = false;
        double home_x = 0.0, home_y = 0.0, home_z = 0.0;
        double home_qw = 1.0, home_qx = 0.0, home_qy = 0.0, home_qz = 0.0;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr home_sub;

        // Trajectory patterns only. Built when the home pose arrives; time is
        // measured from then, on the ROS clock (the simulator's /clock).
        std::unique_ptr<pat_arm_control::ReferenceTrajectory> traj;
        double t_start = 0.0;
        rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_pub;
    };

    //! Read the trajectory pattern's parameters into spec_.
    void loadTrajectorySpec();
    //! Sample every arm's trajectory and publish the window and current point.
    void publishTrajectories();

    /// Read "<side>.waypoints" (flat [x,y,yaw] triples); fall back to a default
    /// relative diamond when absent (throws in absolute mode — no sane default).
    std::vector<Waypoint> loadWaypoints(const std::string& side);

    void onHome(std::size_t idx, const geometry_msgs::msg::PoseStamped& m);
    geometry_msgs::msg::PoseStamped resolve(const Arm& arm) const;
    void republish();   // re-send every arm's current target
    void step();        // advance every arm's waypoint index

    std::string mode_;
    std::string frame_id_;
    double dwell_ = 6.0;
    bool loop_ = true;
    std::vector<Arm> arms_;
    rclcpp::TimerBase::SharedPtr republish_tmr_;
    rclcpp::TimerBase::SharedPtr step_tmr_;

    std::string pattern_ = "waypoints";
    pat_arm_control::TrajectorySpec spec_;
    double window_s_ = 2.0;
    double sample_dt_ = 0.01;
    double lookback_s_ = 0.1;   //!< how far before now each window starts
    rclcpp::TimerBase::SharedPtr traj_tmr_;
    pat_arm_control::ReferenceWindow window_;   //!< reused, 20 KB
};
