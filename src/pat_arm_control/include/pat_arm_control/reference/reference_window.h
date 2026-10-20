#pragma once
/*! A sampled end-effector reference trajectory: what the NMPC tracks when it
 *  follows a trajectory instead of a single setpoint.
 *
 *  The planner samples its trajectory a short way ahead -- further than the
 *  NMPC's horizon -- and the controller interpolates it at each rollout node's
 *  own time, so the optimiser knows where the target WILL be at every node
 *  rather than only where it is now. The controller never sees the shape that
 *  produced the samples, so any planner can drive it.
 *
 *  Fixed capacity and trivially copyable, deliberately: it crosses from a ROS
 *  callback to the realtime control thread through a pat_rt::SeqlockMailbox,
 *  which copies by plain assignment, and nothing on that thread may allocate.
 *  Zero ROS, like the rest of this library.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pat_arm_control {

/*! One planar reference sample, world frame: pose, velocity and acceleration
 *  of the end effector at absolute time `t` [s]. */
struct RefPoint {
    double t = 0.0;
    double x = 0.0, y = 0.0, yaw = 0.0;      //!< [m, m, rad]
    double vx = 0.0, vy = 0.0, wz = 0.0;     //!< [m/s, m/s, rad/s]
    double ax = 0.0, ay = 0.0, az = 0.0;     //!< [m/s^2, m/s^2, rad/s^2]
};

struct ReferenceWindow {
    //! A 2 s window at 10 ms is 201 points; the NMPC's horizon is ~1 s.
    static constexpr std::size_t kMaxPoints = 256;

    RefPoint p[kMaxPoints]{};
    std::uint32_t n = 0;   //!< points in use, strictly increasing in t

    /*! The orientation a point's yaw is measured against.
     *
     *  The EE site's frame is not aligned with the world -- the arm mount
     *  rolls it 90 degrees -- so a target quaternion cannot be built from a yaw
     *  alone: that would also demand a tilt the planar arm cannot make. A
     *  point's full target orientation is instead Rz(yaw - yaw0) * quat0,
     *  world-frame yaw composed onto this reference: exactly quat0 while the
     *  planner holds its yaw at yaw0. The planner sets quat0 to the arm's home
     *  EE orientation. [w, x, y, z]. */
    double quat0[4] = {1.0, 0.0, 0.0, 0.0};
    double yaw0 = 0.0;   //!< [rad] the yaw that quat0 corresponds to

    /*! The full target orientation for a sampled `yaw`, [w, x, y, z]. */
    void orientation(double yaw, double q[4]) const noexcept {
        const double h = 0.5 * std::remainder(yaw - yaw0, 2.0 * M_PI);
        const double cz = std::cos(h), sz = std::sin(h);
        // (cz, 0, 0, sz) * quat0
        q[0] = cz * quat0[0] - sz * quat0[3];
        q[1] = cz * quat0[1] - sz * quat0[2];
        q[2] = cz * quat0[2] + sz * quat0[1];
        q[3] = cz * quat0[3] + sz * quat0[0];
    }

    bool empty() const noexcept { return n == 0; }
    double startTime() const noexcept { return n ? p[0].t : 0.0; }
    double endTime() const noexcept { return n ? p[n - 1].t : 0.0; }

    /*! The reference at absolute time `t`, linearly interpolated between the
     *  bracketing samples, into `out` (whose `t` is set to `t`).
     *
     *  Before the first sample it holds the first. Past the last it holds the
     *  last POSITION with zero velocity and acceleration -- a trajectory that
     *  has run out means "stop there", not "carry on at the last speed".
     *
     *  Yaw is interpolated along the short way round, so samples at +3.1 and
     *  -3.1 rad meet at pi rather than sweeping back through zero.
     *
     *  Returns true while `t` lies within the window, false when it was held
     *  at either end or the window is empty (`out` untouched then), so the
     *  caller can count references it had to invent. */
    bool sample(double t, RefPoint& out) const noexcept {
        if (n == 0) return false;
        if (t <= p[0].t) {
            out = p[0];
            out.t = t;
            return t == p[0].t;
        }
        if (t >= p[n - 1].t) {
            out = p[n - 1];
            const bool inside = (t == p[n - 1].t);
            if (!inside) {
                out.vx = out.vy = out.wz = 0.0;
                out.ax = out.ay = out.az = 0.0;
            }
            out.t = t;
            return inside;
        }
        // First sample strictly after t; the one before it is <= t.
        const RefPoint* hi = std::upper_bound(
            p, p + n, t, [](double tt, const RefPoint& rp) { return tt < rp.t; });
        const RefPoint* lo = hi - 1;
        const double s = (t - lo->t) / (hi->t - lo->t);
        const auto lerp = [s](double a, double b) { return a + s * (b - a); };
        out.t  = t;
        out.x  = lerp(lo->x, hi->x);
        out.y  = lerp(lo->y, hi->y);
        out.yaw = std::remainder(
            lo->yaw + s * std::remainder(hi->yaw - lo->yaw, 2.0 * M_PI), 2.0 * M_PI);
        out.vx = lerp(lo->vx, hi->vx);
        out.vy = lerp(lo->vy, hi->vy);
        out.wz = lerp(lo->wz, hi->wz);
        out.ax = lerp(lo->ax, hi->ax);
        out.ay = lerp(lo->ay, hi->ay);
        out.az = lerp(lo->az, hi->az);
        return true;
    }
};

}  // namespace pat_arm_control
