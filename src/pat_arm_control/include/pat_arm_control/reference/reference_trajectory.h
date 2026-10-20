#pragma once
/*! End-effector reference trajectories: the shapes the planner asks the arms
 *  to follow, evaluated analytically and sampled into a ReferenceWindow for
 *  the NMPC.
 *
 *  Library code (zero ROS) so that ee_target_publisher and the offline
 *  benches generate the SAME reference from the same parameters -- a bench
 *  result then describes what the live arm is asked to do.
 *
 *  Every trajectory is defined relative to an anchor: the arm's home EE pose,
 *  world frame. It begins with an APPROACH segment, a rest-to-rest move from
 *  the anchor to the shape's start point, so no shape begins with a jump in
 *  position. Yaw is held at the anchor's yaw throughout.
 *
 *  Time is local: at(0) is the start of the approach.
 *
 *  Two shapes, chosen for what each tests:
 *
 *    circle     smooth and feasible. The angular rate ramps in, so position,
 *               velocity and acceleration are all continuous: a circle tests
 *               how closely the controller follows a reference it CAN follow.
 *    rectangle  sharp corners at constant speed, deliberately. The velocity
 *               reverses direction instantaneously at each corner, which no
 *               torque-limited arm can do, so the controller is forced to
 *               trade accuracy for feasibility. A controller that sees the
 *               corner coming over its horizon can begin the turn early and
 *               cut it; one that reacts only to the current error overshoots
 *               after it. That difference is what the shape exists to show.
 */
#include <memory>
#include <string>

#include "pat_arm_control/reference/reference_window.h"

namespace pat_arm_control {

//! The arm's home EE pose, world frame. Every trajectory is placed from here.
struct Anchor {
    double x = 0.0, y = 0.0, yaw = 0.0;
};

class ReferenceTrajectory {
public:
    explicit ReferenceTrajectory(const Anchor& anchor, double approach_s);
    virtual ~ReferenceTrajectory() = default;

    /*! The reference at local time `tau` [s] since the trajectory began. The
     *  returned point's `t` is `tau`. Before 0 it holds the anchor. */
    RefPoint at(double tau) const;

    //! Where the shape itself starts, i.e. where the approach ends.
    virtual void startPoint(double& x, double& y) const = 0;
    //! Time for one lap of the shape [s], approach excluded.
    virtual double loopPeriod() const = 0;
    double approachDuration() const noexcept { return approach_s_; }
    const Anchor& anchor() const noexcept { return anchor_; }

protected:
    //! The shape at local shape time `s` >= 0 (approach already removed).
    //! Implementations fill position, velocity and acceleration; yaw is set
    //! by at().
    virtual void shapeAt(double s, RefPoint& out) const = 0;

    Anchor anchor_;
    double approach_s_;
};

/*! A circle about `anchor + (center_dx, center_dy)`, entered at the point
 *  nearest the anchor so the approach is as short as possible.
 *
 *  The angular rate ramps from rest to 2*pi/period over `ramp_s` along a
 *  smoothstep, so velocity and acceleration are continuous from the end of
 *  the approach onward. ramp_s = 0 starts at full speed, with a velocity jump
 *  that is then the only infeasible part of the circle. */
class CircleTrajectory final : public ReferenceTrajectory {
public:
    struct Params {
        double center_dx = 0.17;   //!< [m] centre, relative to the anchor
        double center_dy = 0.0;
        double radius    = 0.06;   //!< [m]
        double period    = 8.0;    //!< [s] per lap once up to speed
        bool   ccw       = true;   //!< direction, seen from above
        double ramp_s    = 2.0;    //!< [s] angular-rate ramp-in
        double approach_s = 3.0;   //!< [s] anchor -> start point
    };

    CircleTrajectory(const Anchor& anchor, const Params& p);

    void startPoint(double& x, double& y) const override;
    double loopPeriod() const override { return p_.period; }

protected:
    void shapeAt(double s, RefPoint& out) const override;

private:
    Params p_;
    double cx_, cy_;   //!< world-frame centre
    double phi0_;      //!< angle of the start point about the centre
};

/*! An axis-aligned rectangle about `anchor + (center_dx, center_dy)`, `width`
 *  along world x and `height` along world y, traversed at constant `speed`
 *  starting from the corner nearest the anchor.
 *
 *  Exact corners at constant speed: position is continuous, velocity jumps at
 *  every corner (and at the start, from rest to `speed`), acceleration is zero
 *  on every edge. The infinite acceleration a corner implies is not published
 *  -- an impulse has no sampled value -- so the reference says only where the
 *  arm should be and how fast it should be going, and leaves the controller
 *  to find what its torque limits allow. */
class RectangleTrajectory final : public ReferenceTrajectory {
public:
    struct Params {
        double center_dx = 0.17;   //!< [m] centre, relative to the anchor
        double center_dy = 0.0;
        double width     = 0.12;   //!< [m] along world x
        double height    = 0.10;   //!< [m] along world y
        double speed     = 0.055;  //!< [m/s] along the path, constant
        bool   ccw       = true;   //!< direction, seen from above
        double approach_s = 3.0;   //!< [s] anchor -> start corner
    };

    RectangleTrajectory(const Anchor& anchor, const Params& p);

    void startPoint(double& x, double& y) const override;
    double loopPeriod() const override;
    //! Corner `i` (0..3) in traversal order, world frame; 0 is the start.
    void corner(int i, double& x, double& y) const;

protected:
    void shapeAt(double s, RefPoint& out) const override;

private:
    Params p_;
    double cx_[4], cy_[4];   //!< corners in traversal order, world frame
    double edge_[4];         //!< length of the edge leaving each corner
};

/*! Which shape, and its parameters: the one description both the planner
 *  node and the benches build a trajectory from. */
struct TrajectorySpec {
    std::string pattern = "circle";   //!< "circle" or "rectangle"
    CircleTrajectory::Params circle;
    RectangleTrajectory::Params rectangle;
};

//! Build the trajectory `spec` names. Throws std::invalid_argument on an
//! unknown pattern or invalid parameters.
std::unique_ptr<ReferenceTrajectory> makeTrajectory(const TrajectorySpec& spec,
                                                    const Anchor& anchor);

/*! Sample `traj` into `out`: points every `dt` seconds of ABSOLUTE time from
 *  `t0` to `t0 + window_s` inclusive, for a trajectory that began at absolute
 *  time `t_start`. Each point's `t` is absolute. Capped at
 *  ReferenceWindow::kMaxPoints; returns false if the cap shortened the window.
 */
bool sampleWindow(const ReferenceTrajectory& traj, double t_start, double t0,
                  double window_s, double dt, ReferenceWindow& out);

}  // namespace pat_arm_control
