#include "pat_arm_control/reference/teleop_reference.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pat_arm_control {

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("teleop reference: ") + what);
}

double finiteOrZero(double v) { return std::isfinite(v) ? v : 0.0; }

/*! The fastest rate toward a stop `d` ahead, from rate `v` toward it now,
 *  that still lets the target arrive there at rest decelerating at `a`.
 *
 *  Measured at the END of the sub-step `h`: the new rate v' must satisfy
 *  v'^2 <= 2 a d' with d' = d - (v + v') h / 2 the distance left after the
 *  sub-step moves at the mean rate. Taking the continuous curve sqrt(2 a d) at
 *  the start of the sub-step instead lets the target enter braking a step
 *  late and carry the excess v^2 - 2 a d, which the discrete integration
 *  conserves, all the way into the stop. */
double stoppingSpeed(double d, double v, double a, double h) {
    const double disc = a * a * h * h + 8.0 * a * (d - 0.5 * v * h);
    if (disc <= 0.0) return 0.0;
    return std::max(0.0, 0.5 * (std::sqrt(disc) - a * h));
}

//! Limit rate `u` along an axis at offset `e` inside [lo, hi], currently
//! moving at `v`, so the target can still stop at the wall it is heading for.
double brakeForWalls(double u, double v, double e, double lo, double hi, double a, double h) {
    if (u > 0.0) return std::min(u, stoppingSpeed(hi - e, v, a, h));
    if (u < 0.0) return std::max(u, -stoppingSpeed(e - lo, -v, a, h));
    return u;
}

//! Hard wall: pin `pos` inside [lo, hi] and drop any rate pushing outward.
void clampToWalls(double& pos, double& vel, double& acc, double lo, double hi) {
    if (pos > hi) {
        pos = hi;
        if (vel > 0.0) vel = acc = 0.0;
    } else if (pos < lo) {
        pos = lo;
        if (vel < 0.0) vel = acc = 0.0;
    }
}

}  // namespace

TeleopReference::TeleopReference(const Anchor& home, const Params& p)
    : home_(home), p_(p) {
    require(p.max_lin_speed > 0.0, "max_lin_speed must be positive");
    require(p.max_yaw_rate > 0.0, "max_yaw_rate must be positive");
    require(p.max_lin_accel > 0.0, "max_lin_accel must be positive");
    require(p.max_yaw_accel > 0.0, "max_yaw_accel must be positive");
    require(p.dx_min <= 0.0 && p.dx_max >= 0.0, "the x bounds must contain home");
    require(p.dy_min <= 0.0 && p.dy_max >= 0.0, "the y bounds must contain home");
    require(p.yaw_max > 0.0 && p.yaw_max < M_PI, "yaw_max must lie in (0, pi)");
    require(p.history_s >= 0.0, "history_s must be non-negative");
    reset(0.0);
}

void TeleopReference::reset(double t) {
    cur_ = RefPoint{};
    cur_.t = t;
    cur_.x = home_.x;
    cur_.y = home_.y;
    cur_.yaw = home_.yaw;
    homing_ = false;
    history_.n = 0;
    pushHistory();
}

void TeleopReference::step(double t, const TeleopCommand& cmd) {
    double h = t - cur_.t;
    if (!(h > 0.0)) return;   // backwards, or NaN
    h = std::min(h, kMaxGap);
    integrate(cur_, homing_, h, cmd);
    cur_.t = t;
    pushHistory();
}

bool TeleopReference::predict(double t0, double window_s, double dt,
                              const TeleopCommand& cmd, ReferenceWindow& out) const {
    require(dt > 0.0, "sample dt must be positive");
    require(window_s >= 0.0, "window must be non-negative");
    const std::size_t wanted = static_cast<std::size_t>(std::floor(window_s / dt + 1e-9)) + 1;
    const std::size_t n = std::min(wanted, ReferenceWindow::kMaxPoints);

    // The future is a rollout from now with the latest command held; it is a
    // copy, so predicting never moves the target itself.
    RefPoint s = cur_;
    bool homing = homing_;
    for (std::size_t i = 0; i < n; ++i) {
        const double t = t0 + dt * static_cast<double>(i);
        if (t <= cur_.t) {
            history_.sample(t, out.p[i]);
        } else {
            integrate(s, homing, t - s.t, cmd);
            s.t = t;
            out.p[i] = s;
        }
    }
    out.n = static_cast<std::uint32_t>(n);
    return n == wanted;
}

void TeleopReference::integrate(RefPoint& s, bool& homing, double h,
                                const TeleopCommand& cmd) const {
    const int steps = std::max(1, static_cast<int>(std::ceil(h / kMaxSubstep - 1e-9)));
    const double hs = h / steps;
    for (int i = 0; i < steps; ++i) subStep(s, homing, hs, cmd);
}

void TeleopReference::subStep(RefPoint& s, bool& homing, double h,
                              const TeleopCommand& cmd) const {
    const double cvx = finiteOrZero(cmd.vx), cvy = finiteOrZero(cmd.vy),
                 cwz = finiteOrZero(cmd.wz);
    if (cvx != 0.0 || cvy != 0.0 || cwz != 0.0) homing = false;

    const double ex = s.x - home_.x, ey = s.y - home_.y;
    const double eyaw = std::remainder(s.yaw - home_.yaw, 2.0 * M_PI);
    const double a_lin = p_.max_lin_accel, a_yaw = p_.max_yaw_accel;

    // The rate the target wants this sub-step.
    double ux, uy, uw;
    if (homing) {
        // Straight back to home, as fast as it can while still stopping there.
        const double d = std::hypot(ex, ey);
        const double v_toward = d > 0.0 ? -(ex * s.vx + ey * s.vy) / d : 0.0;
        const double sp = std::min(p_.max_lin_speed, stoppingSpeed(d, v_toward, a_lin, h));
        ux = d > 0.0 ? -ex / d * sp : 0.0;
        uy = d > 0.0 ? -ey / d * sp : 0.0;
        const double w_toward = -std::copysign(s.wz, eyaw);
        uw = -std::copysign(
            std::min(p_.max_yaw_rate, stoppingSpeed(std::abs(eyaw), w_toward, a_yaw, h)), eyaw);
    } else {
        // The command, capped by speed (as a vector, so a diagonal is no
        // faster than a straight line) and by the walls ahead.
        ux = cvx;
        uy = cvy;
        const double sp = std::hypot(ux, uy);
        if (sp > p_.max_lin_speed) {
            ux *= p_.max_lin_speed / sp;
            uy *= p_.max_lin_speed / sp;
        }
        uw = std::clamp(cwz, -p_.max_yaw_rate, p_.max_yaw_rate);
        ux = brakeForWalls(ux, s.vx, ex, p_.dx_min, p_.dx_max, a_lin, h);
        uy = brakeForWalls(uy, s.vy, ey, p_.dy_min, p_.dy_max, a_lin, h);
        uw = brakeForWalls(uw, s.wz, eyaw, -p_.yaw_max, p_.yaw_max, a_yaw, h);
    }

    // Move the rates toward it under the acceleration limit; the planar limit
    // is on the vector, so the target turns along a straight-line change in
    // velocity rather than one axis finishing first.
    double dvx = ux - s.vx, dvy = uy - s.vy;
    const double dv = std::hypot(dvx, dvy), dv_max = a_lin * h;
    if (dv > dv_max) {
        dvx *= dv_max / dv;
        dvy *= dv_max / dv;
    }
    const double dw = std::clamp(uw - s.wz, -a_yaw * h, a_yaw * h);

    // Exact under constant acceleration over the sub-step.
    s.ax = dvx / h;
    s.ay = dvy / h;
    s.az = dw / h;
    s.x += (s.vx + 0.5 * dvx) * h;
    s.y += (s.vy + 0.5 * dvy) * h;
    s.yaw += (s.wz + 0.5 * dw) * h;
    s.vx += dvx;
    s.vy += dvy;
    s.wz += dw;

    // Backstop for the discrete braking: never outside the box.
    clampToWalls(s.x, s.vx, s.ax, home_.x + p_.dx_min, home_.x + p_.dx_max);
    clampToWalls(s.y, s.vy, s.ay, home_.y + p_.dy_min, home_.y + p_.dy_max);
    double off = std::remainder(s.yaw - home_.yaw, 2.0 * M_PI);
    clampToWalls(off, s.wz, s.az, -p_.yaw_max, p_.yaw_max);
    s.yaw = std::remainder(home_.yaw + off, 2.0 * M_PI);

    // Within one sub-step's worth of acceleration of being home at rest:
    // settle exactly there. The jump this makes is no bigger than one
    // sub-step at the limit could have made.
    if (homing && std::hypot(s.x - home_.x, s.y - home_.y) <= a_lin * h * h &&
        std::hypot(s.vx, s.vy) <= a_lin * h &&
        std::abs(off) <= a_yaw * h * h && std::abs(s.wz) <= a_yaw * h) {
        s.x = home_.x;
        s.y = home_.y;
        s.yaw = home_.yaw;
        s.vx = s.vy = s.wz = 0.0;
        s.ax = s.ay = s.az = 0.0;
        homing = false;
    }
}

void TeleopReference::pushHistory() {
    // Drop what is older than history_s, keeping the last point before the
    // cutoff so the cutoff itself can still be interpolated.
    const double cutoff = cur_.t - p_.history_s;
    std::uint32_t first = 0;
    while (first + 1 < history_.n && history_.p[first + 1].t <= cutoff) ++first;
    if (first == 0 && history_.n == ReferenceWindow::kMaxPoints) first = 1;
    if (first > 0) {
        std::copy(history_.p + first, history_.p + history_.n, history_.p);
        history_.n -= first;
    }
    history_.p[history_.n++] = cur_;
}

}  // namespace pat_arm_control
