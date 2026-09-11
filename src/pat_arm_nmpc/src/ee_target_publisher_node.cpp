#include "pat_arm_nmpc/ee_target_publisher_node.h"

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace {

// Hamilton quaternion [w, x, y, z].
struct Quat { double w, x, y, z; };

Quat qmul(const Quat& a, const Quat& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

// Rotation of `yaw` about world +Z.
Quat qz(double yaw) { return {std::cos(yaw / 2.0), 0.0, 0.0, std::sin(yaw / 2.0)}; }

}  // namespace

EeTargetPublisher::EeTargetPublisher() : Node("ee_target_publisher") {
    const auto sides = declare_parameter<std::vector<std::string>>(
        "sides", std::vector<std::string>{"left", "right"});
    mode_     = declare_parameter<std::string>("mode", "relative");
    dwell_    = declare_parameter<double>("dwell_sec", 6.0);
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    loop_     = declare_parameter<bool>("loop", true);
    const double republish_hz = declare_parameter<double>("republish_hz", 2.0);

    if (sides.empty())
        throw std::runtime_error("ee_target_publisher: 'sides' must be non-empty");
    if (mode_ != "relative" && mode_ != "absolute")
        throw std::runtime_error(
            "ee_target_publisher: 'mode' must be 'relative' or 'absolute'");
    if (dwell_ <= 0.0)
        throw std::runtime_error("ee_target_publisher: 'dwell_sec' must be > 0");
    if (republish_hz <= 0.0)
        throw std::runtime_error("ee_target_publisher: 'republish_hz' must be > 0");

    arms_.reserve(sides.size());
    for (const auto& side : sides) {
        Arm arm;
        arm.side      = side;
        arm.waypoints = loadWaypoints(side);
        arm.pub = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/chaser/arm/" + side + "/ee_setpoint", 10);
        if (mode_ == "relative") {
            const std::size_t idx = arms_.size();
            arm.home_sub = create_subscription<geometry_msgs::msg::PoseStamped>(
                "/chaser/arm/" + side + "/ee_pose", rclcpp::QoS(1),
                [this, idx](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                    onHome(idx, *m);
                });
        } else {
            arm.have_home = true;   // absolute: waypoints are already world poses
        }
        arms_.push_back(std::move(arm));
    }

    // Single-threaded executor (plain rclcpp::spin) serialises these two timers
    // and the /ee_pose callbacks — no locking needed.
    republish_tmr_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / republish_hz),
        std::bind(&EeTargetPublisher::republish, this));
    step_tmr_ = create_wall_timer(
        std::chrono::duration<double>(dwell_),
        std::bind(&EeTargetPublisher::step, this));

    RCLCPP_INFO(get_logger(),
        "ee_target_publisher: mode=%s dwell=%.1fs loop=%s — %zu arm(s)",
        mode_.c_str(), dwell_, loop_ ? "true" : "false", arms_.size());
}

std::vector<EeTargetPublisher::Waypoint>
EeTargetPublisher::loadWaypoints(const std::string& side) {
    const auto flat = declare_parameter<std::vector<double>>(
        side + ".waypoints", std::vector<double>{});
    std::vector<Waypoint> wps;
    if (flat.empty()) {
        if (mode_ == "absolute")
            throw std::runtime_error(
                "ee_target_publisher: mode=absolute needs '" + side +
                ".waypoints' (flat [x, y, yaw] triples)");
        // Default demo: a small diamond about the start pose, ending home.
        wps = {{0.08, 0.00, 0.00}, {0.00, 0.10, 0.15},
               {-0.08, 0.00, 0.00}, {0.00, -0.10, -0.15},
               {0.00, 0.00, 0.00}};
        RCLCPP_INFO(get_logger(),
            "ee_target_publisher[%s]: no '%s.waypoints' — using the default "
            "relative diamond (5 pts, <=0.10 m / 0.15 rad from start)",
            side.c_str(), side.c_str());
    } else {
        if (flat.size() % 3 != 0)
            throw std::runtime_error(
                "ee_target_publisher: '" + side +
                ".waypoints' length must be a multiple of 3 ([x, y, yaw] triples)");
        for (std::size_t i = 0; i + 2 < flat.size(); i += 3)
            wps.push_back({flat[i], flat[i + 1], flat[i + 2]});
    }
    return wps;
}

void EeTargetPublisher::onHome(std::size_t idx,
                               const geometry_msgs::msg::PoseStamped& m) {
    Arm& arm = arms_[idx];
    if (arm.have_home) return;   // freeze the reference on the first sample
    arm.home_x  = m.pose.position.x;
    arm.home_y  = m.pose.position.y;
    arm.home_z  = m.pose.position.z;
    arm.home_qw = m.pose.orientation.w;
    arm.home_qx = m.pose.orientation.x;
    arm.home_qy = m.pose.orientation.y;
    arm.home_qz = m.pose.orientation.z;
    arm.have_home = true;
    RCLCPP_INFO(get_logger(),
        "ee_target_publisher[%s]: home EE pose (%.3f, %.3f) — cycling %zu waypoint(s)",
        arm.side.c_str(), arm.home_x, arm.home_y, arm.waypoints.size());
}

geometry_msgs::msg::PoseStamped
EeTargetPublisher::resolve(const Arm& arm) const {
    const Waypoint& wp = arm.waypoints[arm.idx];
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = now();
    ps.header.frame_id = frame_id_;
    Quat q;
    if (mode_ == "relative") {
        ps.pose.position.x = arm.home_x + wp.a;
        ps.pose.position.y = arm.home_y + wp.b;
        ps.pose.position.z = arm.home_z;
        q = qmul(qz(wp.yaw),
                 {arm.home_qw, arm.home_qx, arm.home_qy, arm.home_qz});  // yaw delta about world Z
    } else {
        ps.pose.position.x = wp.a;
        ps.pose.position.y = wp.b;
        ps.pose.position.z = 0.0;
        q = qz(wp.yaw);
    }
    ps.pose.orientation.w = q.w;
    ps.pose.orientation.x = q.x;
    ps.pose.orientation.y = q.y;
    ps.pose.orientation.z = q.z;
    return ps;
}

void EeTargetPublisher::republish() {
    for (const Arm& arm : arms_) {
        if (!arm.have_home || arm.waypoints.empty()) continue;
        arm.pub->publish(resolve(arm));
    }
}

void EeTargetPublisher::step() {
    for (Arm& arm : arms_) {
        if (!arm.have_home || arm.done || arm.waypoints.empty()) continue;
        std::size_t next = arm.idx + 1;
        if (next >= arm.waypoints.size()) {
            if (!loop_) {
                arm.done = true;
                RCLCPP_INFO(get_logger(),
                    "ee_target_publisher[%s]: reached last waypoint — holding",
                    arm.side.c_str());
                continue;
            }
            next = 0;
        }
        arm.idx = next;
        const auto ps = resolve(arm);
        RCLCPP_INFO(get_logger(),
            "ee_target_publisher[%s]: waypoint %zu/%zu -> (%.3f, %.3f)",
            arm.side.c_str(), arm.idx + 1, arm.waypoints.size(),
            ps.pose.position.x, ps.pose.position.y);
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EeTargetPublisher>());
    rclcpp::shutdown();
}
