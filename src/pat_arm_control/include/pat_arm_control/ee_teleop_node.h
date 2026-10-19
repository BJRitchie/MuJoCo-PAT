#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <termios.h>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>

#include "pat_arm_control/planning/teleop_keys.h"

struct _XDisplay;   // Xlib's Display; Xlib itself stays out of this header

/// Keyboard teleoperation of the arms' end-effector targets.
///
/// Polls the X server for which keys are HELD (XQueryKeymap) rather than
/// reading characters from the terminal, so presses and releases are exact,
/// several keys work at once (diagonals, both arms together) and an arm stops
/// the moment its key is let go. Run it in its own terminal; it needs an X
/// display, which is why no launch file starts it:
///
///   ros2 run pat_arm_control ee_teleop_node
///
/// Default keys (world frame, remappable in config/ee_teleop.yaml):
///   left arm   w/s +x/-x   a/d +y/-y   z/x ccw/cw
///   right arm  i/k +x/-x   j/l +y/-y   ,/. ccw/cw
///   Tab active/paused   space (held) stop   h home   [ / ] speed   Esc quit
///
/// Publishes, every poll, each arm's world-frame velocity command on
/// /chaser/arm/<side>/ee_teleop (geometry_msgs/TwistStamped; zero when idle or
/// paused, so the receiver can tell "stopped" from "gone"), and an
/// std_msgs/Empty on /chaser/arm/<side>/ee_teleop_reset when home is pressed.
/// It only states intent: ee_target_publisher (pattern: teleop) owns the
/// target, its limits and the trajectory the NMPC follows.
///
/// XQueryKeymap sees keys typed into ANY window, so the node starts paused
/// and, with `focus_lock`, ignores the keyboard unless the window that was
/// focused when it started (the terminal it was launched from) still is.
class EeTeleopNode : public rclcpp::Node {
public:
    EeTeleopNode();
    ~EeTeleopNode() override;

private:
    //! One arm's six motion keys as X keycodes, in ArmKeys order.
    struct Arm {
        std::string side;
        std::array<unsigned char, 6> keys{};
        std::array<std::string, 6> key_names;   //!< as configured, for the banner
        pat_arm_control::TeleopCommand last;    //!< for the status line
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub;
        rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_pub;
    };

    void poll();
    //! Resolve a keysym name ("w", "comma", "Tab") to a keycode; throws if the
    //! name is unknown or the keyboard has no such key.
    unsigned char keycode(const std::string& name) const;
    //! Publish a zero command to every arm (on pause, quit and shutdown).
    void publishZero();
    void printBanner() const;
    void printStatus(bool focused);

    //! Terminal: no echo, so the keys held do not scribble over the status
    //! line; restored (and typed-ahead input discarded) on exit.
    void rawTerminal();
    void restoreTerminal();

    _XDisplay* dpy_ = nullptr;
    unsigned long focus_window_ = 0;   //!< Xlib Window; 0 = no focus lock
    std::vector<Arm> arms_;
    std::array<unsigned char, 6> panel_keys_{};   //!< PanelKeys order
    std::array<std::string, 6> panel_names_;
    pat_arm_control::TeleopPanel panel_;
    double linear_ = 0.05, angular_ = 0.3;
    std::string frame_id_;
    std::string last_status_;
    termios saved_term_{};
    bool term_saved_ = false;
    rclcpp::PreShutdownCallbackHandle shutdown_cb_;
    rclcpp::TimerBase::SharedPtr timer_;
};
