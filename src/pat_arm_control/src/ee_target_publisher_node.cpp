#include "pat_arm_control/ee_target_publisher_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
    pattern_  = declare_parameter<std::string>("pattern", "waypoints");
    const double trajectory_hz = declare_parameter<double>("trajectory_hz", 20.0);
    window_s_  = declare_parameter<double>("window_s", 2.0);
    sample_dt_ = declare_parameter<double>("sample_dt", 0.01);
    lookback_s_ = declare_parameter<double>("window_lookback_s", 0.1);
    const bool trajectory = pattern_ != "waypoints";
    if (trajectory) {
        if (pattern_ != "circle" && pattern_ != "rectangle")
            throw std::runtime_error(
                "ee_target_publisher: 'pattern' must be waypoints, circle or rectangle");
        if (mode_ != "relative")
            throw std::runtime_error(
                "ee_target_publisher: circle/rectangle are placed from each arm's "
                "start pose, so they need mode=relative");
        if (!(trajectory_hz > 0.0) || !(sample_dt_ > 0.0) || !(window_s_ > 0.0) ||
            lookback_s_ < 0.0)
            throw std::runtime_error(
                "ee_target_publisher: trajectory_hz, sample_dt and window_s must be > 0, "
                "window_lookback_s >= 0");
        const double points = (window_s_ + lookback_s_) / sample_dt_ + 1.0;
        if (points > static_cast<double>(pat_arm_control::ReferenceWindow::kMaxPoints))
            throw std::runtime_error(
                "ee_target_publisher: (window_s + window_lookback_s) / sample_dt needs " +
                std::to_string(static_cast<int>(points)) + " points, more than a window "
                "holds (" + std::to_string(pat_arm_control::ReferenceWindow::kMaxPoints) +
                ")");
        loadTrajectorySpec();
    }

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
        if (!trajectory) arm.waypoints = loadWaypoints(side);
        arm.pub = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/chaser/arm/" + side + "/ee_setpoint", 10);
        if (trajectory) {
            // Depth 1, as the controller subscribes: each window supersedes the
            // last entirely, so a backlog would only be stale windows.
            arm.traj_pub = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
                "/chaser/arm/" + side + "/ee_trajectory", rclcpp::QoS(1).reliable());
        }
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
    if (trajectory) {
        traj_tmr_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / trajectory_hz),
            std::bind(&EeTargetPublisher::publishTrajectories, this));
        RCLCPP_INFO(get_logger(),
            "ee_target_publisher: pattern=%s for %zu arm(s), %.1f s window every "
            "%.0f ms at %.0f Hz", pattern_.c_str(), arms_.size(), window_s_,
            sample_dt_ * 1e3, trajectory_hz);
        return;
    }
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
        // Default demo: reach-out pattern biased in +x (away from the bus,
        // in the arm's own extension direction) -- the start pose is folded
        // in close to the bus, so a diamond symmetric about it would just
        // fold back in just as far on its other half. Never within 0.10 of
        // home. Keep in sync with config/ee_targets.yaml's default.
        wps = {{0.20, 0.00, 0.00}, {0.15, 0.12, 0.15},
               {0.25, 0.00, 0.00}, {0.15, -0.12, -0.15},
               {0.10, 0.00, 0.00}};
        RCLCPP_INFO(get_logger(),
            "ee_target_publisher[%s]: no '%s.waypoints' — using the default "
            "reach-out pattern (5 pts, 0.10-0.25 m from start)",
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
    if (pattern_ != "waypoints") {
        // The shape is placed from here and its clock starts now. Yaw is held
        // at 0 relative to the home orientation, which every point carries.
        arm.traj = pat_arm_control::makeTrajectory(
            spec_, pat_arm_control::Anchor{arm.home_x, arm.home_y, 0.0});
        arm.t_start = now().seconds();
        RCLCPP_INFO(get_logger(),
            "ee_target_publisher[%s]: home EE pose (%.3f, %.3f) — %s, lap %.1f s "
            "after a %.1f s approach",
            arm.side.c_str(), arm.home_x, arm.home_y, pattern_.c_str(),
            arm.traj->loopPeriod(), arm.traj->approachDuration());
        return;
    }
    RCLCPP_INFO(get_logger(),
        "ee_target_publisher[%s]: home EE pose (%.3f, %.3f) — cycling %zu waypoint(s)",
        arm.side.c_str(), arm.home_x, arm.home_y, arm.waypoints.size());
}

void EeTargetPublisher::loadTrajectorySpec() {
    // Defaults are the library's, so an unset parameter means the same shape
    // the benches measured.
    spec_.pattern = pattern_;
    auto& c = spec_.circle;
    c.center_dx  = declare_parameter<double>("circle.center_dx", c.center_dx);
    c.center_dy  = declare_parameter<double>("circle.center_dy", c.center_dy);
    c.radius     = declare_parameter<double>("circle.radius", c.radius);
    c.period     = declare_parameter<double>("circle.period", c.period);
    c.ccw        = declare_parameter<bool>("circle.ccw", c.ccw);
    c.ramp_s     = declare_parameter<double>("circle.ramp_s", c.ramp_s);
    auto& r = spec_.rectangle;
    r.center_dx  = declare_parameter<double>("rectangle.center_dx", r.center_dx);
    r.center_dy  = declare_parameter<double>("rectangle.center_dy", r.center_dy);
    r.width      = declare_parameter<double>("rectangle.width", r.width);
    r.height     = declare_parameter<double>("rectangle.height", r.height);
    r.speed      = declare_parameter<double>("rectangle.speed", r.speed);
    r.ccw        = declare_parameter<bool>("rectangle.ccw", r.ccw);
    const double approach = declare_parameter<double>("approach_s", c.approach_s);
    c.approach_s = approach;
    r.approach_s = approach;
    // Validate now, not at the first home pose, so a bad config fails at launch.
    (void)pat_arm_control::makeTrajectory(spec_, pat_arm_control::Anchor{});
}

void EeTargetPublisher::publishTrajectories() {
    for (Arm& arm : arms_) {
        if (!arm.have_home || !arm.traj) continue;
        // The window starts a little in the PAST. The controller evaluates it at
        // the stamp of the state it is solving on, which is always older than
        // this node's clock, so a window starting now would leave the
        // controller's first node just before it.
        const double t_now = now().seconds();
        // Never before time zero: a ROS stamp cannot be negative, and this runs
        // from the very start of a simulation.
        const double t0 = std::max(0.0, t_now - lookback_s_);
        pat_arm_control::sampleWindow(*arm.traj, arm.t_start, t0,
                                      window_s_ + lookback_s_, sample_dt_, window_);
        const Quat home{arm.home_qw, arm.home_qx, arm.home_qy, arm.home_qz};

        trajectory_msgs::msg::MultiDOFJointTrajectory msg;
        msg.header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(t0 * 1e9)),
                                        RCL_ROS_TIME);
        msg.header.frame_id = frame_id_;
        msg.joint_names = {"ee_" + arm.side};
        msg.points.resize(window_.n);
        for (std::uint32_t i = 0; i < window_.n; ++i) {
            const auto& p = window_.p[i];
            auto& pt = msg.points[i];
            pt.transforms.resize(1);
            pt.velocities.resize(1);
            pt.accelerations.resize(1);
            pt.transforms[0].translation.x = p.x;
            pt.transforms[0].translation.y = p.y;
            pt.transforms[0].translation.z = arm.home_z;
            // Yaw about world Z on top of the home orientation: the EE frame
            // is rolled relative to the world, so a bare yaw would be a tilt.
            const Quat q = qmul(qz(p.yaw), home);
            pt.transforms[0].rotation.w = q.w;
            pt.transforms[0].rotation.x = q.x;
            pt.transforms[0].rotation.y = q.y;
            pt.transforms[0].rotation.z = q.z;
            pt.velocities[0].linear.x = p.vx;
            pt.velocities[0].linear.y = p.vy;
            pt.velocities[0].angular.z = p.wz;
            pt.accelerations[0].linear.x = p.ax;
            pt.accelerations[0].linear.y = p.ay;
            pt.accelerations[0].angular.z = p.az;
            pt.time_from_start = rclcpp::Duration::from_seconds(p.t - t0);
        }
        arm.traj_pub->publish(msg);

        // The current reference point as a plain setpoint as well.
        const pat_arm_control::RefPoint cur = arm.traj->at(t_now - arm.t_start);
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = now();
        ps.header.frame_id = frame_id_;
        ps.pose.position.x = cur.x;
        ps.pose.position.y = cur.y;
        ps.pose.position.z = arm.home_z;
        ps.pose.orientation = msg.points[0].transforms[0].rotation;
        arm.pub->publish(ps);
    }
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
