#pragma once
/*! A teleoperated end-effector reference: a velocity command from a human,
 *  turned into a trajectory the NMPC can follow with preview.
 *
 *  The operator commands a world-frame velocity (vx, vy, wz). This class owns
 *  the target pose that command moves and keeps it well behaved:
 *
 *    - rates follow the command under an acceleration limit and are capped at
 *      a maximum speed, so a key press is a smooth ramp, not a step;
 *    - the position stays inside a box around home and the yaw within
 *      +/-yaw_max of it, braking ahead of each wall so the target arrives at
 *      rest rather than stopping dead;
 *    - goHome() drives the target back to home under the same limits.
 *
 *  Only the LATEST command matters. step() integrates it up to now, and
 *  predict() samples a window by rolling the same dynamics forward on the
 *  assumption that the latest command persists: "where the target goes if
 *  nothing changes". Every window replaces the last one, so no command is
 *  ever queued or replayed, and a released key shows up in the very next
 *  window as the deceleration it causes, which the NMPC sees across its
 *  whole horizon.
 *
 *  Zero ROS and allocation-free, like the rest of this library: the planner
 *  node drives it from its timer, and the tests drive it with plain numbers.
 */
#include "pat_arm_control/reference/reference_trajectory.h"
#include "pat_arm_control/reference/reference_window.h"

namespace pat_arm_control {

//! A requested end-effector velocity, world frame [m/s, m/s, rad/s].
struct TeleopCommand {
    double vx = 0.0, vy = 0.0, wz = 0.0;
    bool isZero() const noexcept { return vx == 0.0 && vy == 0.0 && wz == 0.0; }
};

class TeleopReference {
public:
    struct Params {
        double max_lin_speed = 0.10;   //!< [m/s] cap on the planar speed
        double max_yaw_rate  = 0.6;    //!< [rad/s]
        double max_lin_accel = 0.25;   //!< [m/s^2] on the planar velocity vector
        double max_yaw_accel = 2.0;    //!< [rad/s^2]
        //! The box the target stays in, as offsets from home [m]. It must
        //! contain home (0, 0), where the target starts.
        double dx_min = 0.0, dx_max = 0.25;
        double dy_min = -0.12, dy_max = 0.12;
        double yaw_max = 0.3;          //!< [rad] either side of home, < pi
        //! [s] of past states kept for the part of a window that lies before
        //! now (the planner starts each window slightly in the past).
        double history_s = 0.25;
    };

    /*! A target resting at `home`, at time 0. Throws std::invalid_argument on
     *  a non-positive limit, a box that excludes home, or yaw_max outside
     *  (0, pi). */
    TeleopReference(const Anchor& home, const Params& p);

    //! Put the target at home, at rest, at absolute time `t` [s]; forgets the
    //! history and cancels homing.
    void reset(double t);

    /*! Advance the target to absolute time `t` under `cmd`. A step backwards
     *  in time is ignored; a gap longer than 1 s is treated as 1 s, so a
     *  stalled caller cannot fling the target across its box. A non-zero
     *  command cancels homing; a non-finite component counts as zero. */
    void step(double t, const TeleopCommand& cmd);

    //! Drive back to home under the normal limits until there, or until a
    //! non-zero command takes over.
    void goHome() noexcept { homing_ = true; }
    bool homing() const noexcept { return homing_; }

    //! The target now: pose, rates and the acceleration that produced them.
    const RefPoint& current() const noexcept { return cur_; }
    const Anchor& home() const noexcept { return home_; }

    /*! Sample a window into `out`: points every `dt` from absolute time `t0`
     *  to `t0 + window_s` inclusive. Points at or before current().t come
     *  from the recorded history (holding its oldest point before that);
     *  later ones roll the dynamics forward from current() with `cmd` held.
     *  Sets `out.n` and the points only; the orientation reference (quat0,
     *  yaw0) is the caller's. Capped at ReferenceWindow::kMaxPoints; returns
     *  false if the cap shortened the window. */
    bool predict(double t0, double window_s, double dt, const TeleopCommand& cmd,
                 ReferenceWindow& out) const;

private:
    //! Advance `s` by `h` seconds under `cmd`, sub-stepping so the result
    //! does not depend on how the caller splits time.
    void integrate(RefPoint& s, bool& homing, double h, const TeleopCommand& cmd) const;
    //! One sub-step, h <= kMaxSubstep.
    void subStep(RefPoint& s, bool& homing, double h, const TeleopCommand& cmd) const;
    void pushHistory();

    static constexpr double kMaxSubstep = 0.01;   //!< [s]
    static constexpr double kMaxGap = 1.0;        //!< [s] longest step() honoured

    Anchor home_;
    Params p_;
    RefPoint cur_{};
    bool homing_ = false;
    ReferenceWindow history_;   //!< past states, oldest first; 20 KB, reused
};

}  // namespace pat_arm_control
