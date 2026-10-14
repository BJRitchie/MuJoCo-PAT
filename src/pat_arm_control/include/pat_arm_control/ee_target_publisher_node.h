#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

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
    };

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
};
