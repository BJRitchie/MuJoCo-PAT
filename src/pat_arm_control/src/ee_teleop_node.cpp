#include "pat_arm_control/ee_teleop_node.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <set>
#include <stdexcept>

#include <unistd.h>

// Xlib last: it defines macros (None, Status, Bool, True, False) that would
// collide with names in the ROS and standard headers above.
#include <X11/Xlib.h>

namespace {

// Configured order of an arm's keys, matching ArmKeys.
constexpr const char* kArmKeyRoles[6] = {"+x", "-x", "+y", "-y", "ccw", "cw"};
// Configured order of the control keys, matching PanelKeys.
constexpr const char* kPanelKeyRoles[6] = {"active/paused", "quit", "stop (held)",
                                           "home", "slower", "faster"};

std::vector<std::string> defaultArmKeys(const std::string& side) {
    if (side == "left") return {"w", "s", "a", "d", "z", "x"};
    if (side == "right") return {"i", "k", "j", "l", "comma", "period"};
    return {};
}

//! A keysym name as the key cap shows it, for the banner.
std::string cap(const std::string& name) {
    if (name == "comma") return ",";
    if (name == "period") return ".";
    if (name == "bracketleft") return "[";
    if (name == "bracketright") return "]";
    if (name == "space") return "space";
    if (name == "Escape") return "Esc";
    return name;
}

}  // namespace

EeTeleopNode::EeTeleopNode() : Node("ee_teleop") {
    const auto sides = declare_parameter<std::vector<std::string>>(
        "sides", std::vector<std::string>{"left", "right"});
    const double poll_hz = declare_parameter<double>("poll_hz", 50.0);
    linear_   = declare_parameter<double>("linear_speed", linear_);
    angular_  = declare_parameter<double>("angular_speed", angular_);
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    const bool focus_lock = declare_parameter<bool>("focus_lock", true);
    const auto panel = declare_parameter<std::vector<std::string>>(
        "panel_keys", std::vector<std::string>{"Tab", "Escape", "space", "h",
                                               "bracketleft", "bracketright"});
    if (sides.empty())
        throw std::runtime_error("ee_teleop: 'sides' must be non-empty");
    if (!(poll_hz > 0.0) || !(linear_ > 0.0) || !(angular_ > 0.0))
        throw std::runtime_error(
            "ee_teleop: poll_hz, linear_speed and angular_speed must be > 0");
    if (panel.size() != 6)
        throw std::runtime_error(
            "ee_teleop: 'panel_keys' needs 6 names: active/paused, quit, stop, home, "
            "slower, faster");

    dpy_ = XOpenDisplay(nullptr);
    if (!dpy_)
        throw std::runtime_error(
            "ee_teleop: cannot open the X display (DISPLAY='" +
            std::string(std::getenv("DISPLAY") ? std::getenv("DISPLAY") : "") +
            "'). In the container, run `xhost +local:docker` on the host first.");

    // Resolve every key once. A key bound twice is a configuration error:
    // it would drive two things at once.
    std::set<unsigned char> used;
    auto bind = [&](const std::string& name) {
        const unsigned char kc = keycode(name);
        if (!used.insert(kc).second)
            throw std::runtime_error("ee_teleop: key '" + name + "' is bound twice");
        return kc;
    };
    for (std::size_t i = 0; i < 6; ++i) {
        panel_names_[i] = panel[i];
        panel_keys_[i] = bind(panel[i]);
    }
    for (const auto& side : sides) {
        const auto names = declare_parameter<std::vector<std::string>>(
            side + ".keys", defaultArmKeys(side));
        if (names.size() != 6)
            throw std::runtime_error(
                "ee_teleop: '" + side + ".keys' needs 6 keysym names: +x, -x, +y, -y, "
                "ccw, cw");
        Arm arm;
        arm.side = side;
        for (std::size_t i = 0; i < 6; ++i) {
            arm.key_names[i] = names[i];
            arm.keys[i] = bind(names[i]);
        }
        // Depth 1: the receiver only ever wants the latest command.
        arm.cmd_pub = create_publisher<geometry_msgs::msg::TwistStamped>(
            "/chaser/arm/" + side + "/ee_teleop", rclcpp::QoS(1));
        arm.reset_pub = create_publisher<std_msgs::msg::Empty>(
            "/chaser/arm/" + side + "/ee_teleop_reset", rclcpp::QoS(1));
        arms_.push_back(std::move(arm));
    }

    // The window focused now is the one this was launched from. Lock to it
    // unless the X server reports no focus window to lock to.
    if (focus_lock) {
        Window w = 0;
        int revert = 0;
        XGetInputFocus(dpy_, &w, &revert);
        if (w == None || w == PointerRoot)
            RCLCPP_WARN(get_logger(),
                "ee_teleop: no focused window to lock to — keys typed into ANY "
                "window will drive the arms while active");
        else
            focus_window_ = w;
    }

    // On Ctrl-C, tell every arm to stop while the publishers still work,
    // rather than leave the receiver to time the last command out.
    shutdown_cb_ = get_node_base_interface()->get_context()->add_pre_shutdown_callback(
        [this] { publishZero(); });

    printBanner();
    rawTerminal();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / poll_hz),
                               std::bind(&EeTeleopNode::poll, this));
}

EeTeleopNode::~EeTeleopNode() {
    get_node_base_interface()->get_context()->remove_pre_shutdown_callback(shutdown_cb_);
    restoreTerminal();
    if (!last_status_.empty()) std::printf("\n");   // leave the status line intact
    if (dpy_) XCloseDisplay(dpy_);
}

unsigned char EeTeleopNode::keycode(const std::string& name) const {
    const KeySym sym = XStringToKeysym(name.c_str());
    if (sym == NoSymbol)
        throw std::runtime_error("ee_teleop: unknown keysym '" + name +
                                 "' (use X11 names: w, comma, period, Tab, space...)");
    const KeyCode kc = XKeysymToKeycode(dpy_, sym);
    if (kc == 0)
        throw std::runtime_error("ee_teleop: the keyboard has no key for '" + name + "'");
    return kc;
}

void EeTeleopNode::poll() {
    char map[32];
    XQueryKeymap(dpy_, map);

    // Out of focus, every key reads as released: the panel sees the release
    // and the arms stop, as if the operator had let go.
    bool focused = true;
    if (focus_window_) {
        Window w = 0;
        int revert = 0;
        XGetInputFocus(dpy_, &w, &revert);
        focused = (w == focus_window_);
    }
    const auto held = [&](unsigned char kc) {
        return focused && ((map[kc >> 3] >> (kc & 7)) & 1);
    };

    pat_arm_control::PanelKeys pk;
    pk.toggle = held(panel_keys_[0]);
    pk.quit   = held(panel_keys_[1]);
    pk.stop   = held(panel_keys_[2]);
    pk.home   = held(panel_keys_[3]);
    pk.slower = held(panel_keys_[4]);
    pk.faster = held(panel_keys_[5]);
    const auto ev = panel_.update(pk);
    if (ev.quit) {
        publishZero();
        restoreTerminal();
        rclcpp::shutdown();
        return;
    }

    const auto stamp = now();
    for (Arm& arm : arms_) {
        pat_arm_control::ArmKeys k;
        k.plus_x  = held(arm.keys[0]);
        k.minus_x = held(arm.keys[1]);
        k.plus_y  = held(arm.keys[2]);
        k.minus_y = held(arm.keys[3]);
        k.ccw     = held(arm.keys[4]);
        k.cw      = held(arm.keys[5]);
        arm.last = panel_.command(k, linear_, angular_);

        geometry_msgs::msg::TwistStamped m;
        m.header.stamp = stamp;
        m.header.frame_id = frame_id_;
        m.twist.linear.x = arm.last.vx;
        m.twist.linear.y = arm.last.vy;
        m.twist.angular.z = arm.last.wz;
        arm.cmd_pub->publish(m);
        if (ev.home) arm.reset_pub->publish(std_msgs::msg::Empty());
    }
    printStatus(focused);
}

void EeTeleopNode::publishZero() {
    const auto stamp = now();
    for (Arm& arm : arms_) {
        geometry_msgs::msg::TwistStamped m;
        m.header.stamp = stamp;
        m.header.frame_id = frame_id_;
        arm.cmd_pub->publish(m);
    }
}

void EeTeleopNode::printBanner() const {
    std::printf("\nPAT arm teleop — world frame, %.3f m/s, %.2f rad/s at 1.00x\n",
                linear_, angular_);
    for (const Arm& arm : arms_) {
        std::printf("  %-6s", arm.side.c_str());
        for (std::size_t i = 0; i < 6; ++i)
            std::printf("  %s %s", cap(arm.key_names[i]).c_str(), kArmKeyRoles[i]);
        std::printf("\n");
    }
    std::printf("  ");
    for (std::size_t i = 0; i < 6; ++i)
        std::printf("%s %s   ", cap(panel_names_[i]).c_str(), kPanelKeyRoles[i]);
    std::printf("\n");
    std::printf(focus_window_
        ? "  Keys count only while THIS terminal has focus. Starts paused.\n\n"
        : "  Keys typed into ANY window count while active. Starts paused.\n\n");
    std::fflush(stdout);
}

void EeTeleopNode::printStatus(bool focused) {
    char buf[256];
    int n = std::snprintf(buf, sizeof buf, "[%s] %.2fx",
                          !focused             ? "no focus"
                          : !panel_.active()   ? "PAUSED  "
                          : panel_.moving()    ? "ACTIVE  "
                                               : "STOPPED ",
                          panel_.scale());
    for (const Arm& arm : arms_) {
        if (n < 0 || n >= static_cast<int>(sizeof buf)) break;
        n += std::snprintf(buf + n, sizeof buf - n, " | %s %+.3f %+.3f %+.2f",
                           arm.side.c_str(), arm.last.vx, arm.last.vy, arm.last.wz);
    }
    // Only on change: the line is rewritten in place, and a terminal flooded
    // at the poll rate is slow to scroll and to read.
    if (last_status_ == buf) return;
    last_status_ = buf;
    std::printf("\r%s\033[K", buf);
    std::fflush(stdout);
}

void EeTeleopNode::rawTerminal() {
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved_term_) != 0) return;
    termios t = saved_term_;
    t.c_lflag &= ~static_cast<tcflag_t>(ECHO | ICANON);   // ISIG stays: Ctrl-C works
    if (tcsetattr(STDIN_FILENO, TCSANOW, &t) == 0) term_saved_ = true;
}

void EeTeleopNode::restoreTerminal() {
    if (!term_saved_) return;
    // Discard everything typed while driving, or the shell would run it.
    tcflush(STDIN_FILENO, TCIFLUSH);
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_term_);
    term_saved_ = false;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EeTeleopNode>());
    rclcpp::shutdown();
}
