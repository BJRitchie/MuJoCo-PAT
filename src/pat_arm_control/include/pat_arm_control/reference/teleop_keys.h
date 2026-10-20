#pragma once
/*! Keyboard teleop logic, kept apart from where the key states come from.
 *
 *  ee_teleop_node reads which keys are held from X11 and hands them here as
 *  plain booleans: one ArmKeys per arm, one PanelKeys for the controls. What
 *  the keys mean -- the command each arm gets, pause, speed scale, home --
 *  lives here, header-only and dependency-free, so it is unit-tested without
 *  a display.
 */
#include <algorithm>
#include <cmath>

#include "pat_arm_control/reference/teleop_reference.h"

namespace pat_arm_control {

//! The six motion keys of one arm, true while held. World frame.
struct ArmKeys {
    bool plus_x = false, minus_x = false;
    bool plus_y = false, minus_y = false;
    bool ccw = false, cw = false;   //!< +yaw, -yaw
};

/*! The command one arm's held keys ask for. Opposite keys cancel. A diagonal
 *  is normalised, so the arm moves at `linear` [m/s] in every direction;
 *  rotation is `angular` [rad/s], independent of translation. */
inline TeleopCommand commandFromKeys(const ArmKeys& k, double linear, double angular) {
    const double ux = (k.plus_x ? 1.0 : 0.0) - (k.minus_x ? 1.0 : 0.0);
    const double uy = (k.plus_y ? 1.0 : 0.0) - (k.minus_y ? 1.0 : 0.0);
    const double uw = (k.ccw ? 1.0 : 0.0) - (k.cw ? 1.0 : 0.0);
    const double n = std::hypot(ux, uy);
    TeleopCommand c;
    if (n > 0.0) {
        c.vx = linear * ux / n;
        c.vy = linear * uy / n;
    }
    c.wz = angular * uw;
    return c;
}

//! The control keys, true while held.
struct PanelKeys {
    bool toggle = false;   //!< active <-> paused
    bool quit = false;
    bool stop = false;     //!< held: every arm commanded to zero
    bool home = false;     //!< send every target home
    bool slower = false, faster = false;   //!< step the speed scale
};

/*! Pause, speed scale and the one-shot controls. Toggle, home, quit and the
 *  speed steps act on the PRESS, not while held, so holding a key does not
 *  flicker the pause or race the scale to its limit.
 *
 *  Starts paused: the keyboard is read globally, so motion should begin only
 *  when the operator says so. While paused every command is zero and home is
 *  ignored; quit always works. */
class TeleopPanel {
public:
    static constexpr double kScaleStep = 0.25;
    static constexpr double kScaleMin = 0.25;
    static constexpr double kScaleMax = 2.0;

    struct Events {
        bool home = false;   //!< send every target home, this poll
        bool quit = false;
    };

    Events update(const PanelKeys& k) noexcept {
        Events e;
        if (pressed(k.toggle, prev_.toggle)) active_ = !active_;
        if (pressed(k.faster, prev_.faster)) scale_ = std::min(kScaleMax, scale_ + kScaleStep);
        if (pressed(k.slower, prev_.slower)) scale_ = std::max(kScaleMin, scale_ - kScaleStep);
        e.home = active_ && pressed(k.home, prev_.home);
        e.quit = pressed(k.quit, prev_.quit);
        stop_ = k.stop;
        prev_ = k;
        return e;
    }

    bool active() const noexcept { return active_; }
    double scale() const noexcept { return scale_; }
    //! Whether motion keys count this poll: active, and stop not held.
    bool moving() const noexcept { return active_ && !stop_; }

    /*! An arm's command for this poll: from its keys at the scaled speeds
     *  while moving, zero otherwise. */
    TeleopCommand command(const ArmKeys& k, double linear, double angular) const noexcept {
        if (!moving()) return {};
        return commandFromKeys(k, linear * scale_, angular * scale_);
    }

private:
    static bool pressed(bool now, bool before) noexcept { return now && !before; }

    PanelKeys prev_{};
    bool active_ = false;
    bool stop_ = false;
    double scale_ = 1.0;
};

}  // namespace pat_arm_control
