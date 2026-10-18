#include "pat_arm_control/planning/reference_trajectory.h"

#include <cmath>
#include <stdexcept>

namespace pat_arm_control {

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("reference trajectory: ") + what);
}

}  // namespace

// ============================== base ====================================== //

ReferenceTrajectory::ReferenceTrajectory(const Anchor& anchor, double approach_s)
    : anchor_(anchor), approach_s_(approach_s) {
    require(approach_s >= 0.0, "approach_s must be non-negative");
}

RefPoint ReferenceTrajectory::at(double tau) const {
    RefPoint r;
    r.t = tau;
    r.yaw = anchor_.yaw;
    if (tau <= 0.0) {
        r.x = anchor_.x;
        r.y = anchor_.y;
        return r;
    }
    if (tau < approach_s_) {
        // Rest-to-rest quintic, s(u) = 10u^3 - 15u^4 + 6u^5: zero velocity and
        // acceleration at both ends, so the approach joins the anchor and the
        // shape's start without a jump in either.
        double sx, sy;
        startPoint(sx, sy);
        const double T = approach_s_;
        const double u = tau / T;
        const double u2 = u * u, u3 = u2 * u;
        const double s   = u3 * (10.0 - 15.0 * u + 6.0 * u2);
        const double sd  = 30.0 * u2 * (1.0 - 2.0 * u + u2) / T;
        const double sdd = 60.0 * u * (1.0 - 3.0 * u + 2.0 * u2) / (T * T);
        const double dx = sx - anchor_.x, dy = sy - anchor_.y;
        r.x = anchor_.x + dx * s;
        r.y = anchor_.y + dy * s;
        r.vx = dx * sd;
        r.vy = dy * sd;
        r.ax = dx * sdd;
        r.ay = dy * sdd;
        return r;
    }
    shapeAt(tau - approach_s_, r);
    r.t = tau;
    r.yaw = anchor_.yaw;
    return r;
}

// ============================== circle ==================================== //

CircleTrajectory::CircleTrajectory(const Anchor& anchor, const Params& p)
    : ReferenceTrajectory(anchor, p.approach_s), p_(p) {
    require(p.radius > 0.0, "circle radius must be positive");
    require(p.period > 0.0, "circle period must be positive");
    require(p.ramp_s >= 0.0, "circle ramp_s must be non-negative");
    cx_ = anchor.x + p.center_dx;
    cy_ = anchor.y + p.center_dy;
    // Enter at the point nearest the anchor; an anchor at the centre has no
    // nearest point, so any angle will do.
    const double dx = anchor.x - cx_, dy = anchor.y - cy_;
    phi0_ = (dx == 0.0 && dy == 0.0) ? 0.0 : std::atan2(dy, dx);
}

void CircleTrajectory::startPoint(double& x, double& y) const {
    x = cx_ + p_.radius * std::cos(phi0_);
    y = cy_ + p_.radius * std::sin(phi0_);
}

void CircleTrajectory::shapeAt(double s, RefPoint& out) const {
    // Swept angle theta(s), its rate and its acceleration. During the ramp the
    // rate follows a smoothstep 3u^2 - 2u^3 from 0 to w, whose integral is
    // w*ramp*(u^3 - u^4/2); afterwards the rate is constant.
    const double w = 2.0 * M_PI / p_.period;
    double th, thd, thdd;
    if (p_.ramp_s > 0.0 && s < p_.ramp_s) {
        const double u = s / p_.ramp_s;
        th   = w * p_.ramp_s * (u * u * u - 0.5 * u * u * u * u);
        thd  = w * u * u * (3.0 - 2.0 * u);
        thdd = w * 6.0 * u * (1.0 - u) / p_.ramp_s;
    } else {
        th   = w * (0.5 * p_.ramp_s + (s - p_.ramp_s));
        thd  = w;
        thdd = 0.0;
    }
    const double dir = p_.ccw ? 1.0 : -1.0;
    const double phi = phi0_ + dir * th;
    const double phid = dir * thd, phidd = dir * thdd;
    const double c = std::cos(phi), sn = std::sin(phi);
    const double r = p_.radius;
    out.x  = cx_ + r * c;
    out.y  = cy_ + r * sn;
    out.vx = -r * phid * sn;
    out.vy =  r * phid * c;
    // Tangential (from the rate changing) plus centripetal.
    out.ax = -r * phidd * sn - r * phid * phid * c;
    out.ay =  r * phidd * c  - r * phid * phid * sn;
    out.wz = 0.0;
    out.az = 0.0;
}

// ============================= rectangle ================================== //

RectangleTrajectory::RectangleTrajectory(const Anchor& anchor, const Params& p)
    : ReferenceTrajectory(anchor, p.approach_s), p_(p) {
    require(p.width > 0.0 && p.height > 0.0, "rectangle width and height must be positive");
    require(p.speed > 0.0, "rectangle speed must be positive");
    const double cx = anchor.x + p.center_dx, cy = anchor.y + p.center_dy;
    const double hw = 0.5 * p.width, hh = 0.5 * p.height;
    // Counter-clockwise from the bottom-left corner.
    const double ccw_x[4] = {cx - hw, cx + hw, cx + hw, cx - hw};
    const double ccw_y[4] = {cy - hh, cy - hh, cy + hh, cy + hh};
    // Start at the corner nearest the anchor.
    int start = 0;
    double best = 1e300;
    for (int i = 0; i < 4; ++i) {
        const double d = std::hypot(ccw_x[i] - anchor.x, ccw_y[i] - anchor.y);
        if (d < best) { best = d; start = i; }
    }
    // Traversal order: forward through the CCW list, or backward for CW.
    for (int k = 0; k < 4; ++k) {
        const int i = p.ccw ? (start + k) % 4 : (start - k + 4) % 4;
        cx_[k] = ccw_x[i];
        cy_[k] = ccw_y[i];
    }
    for (int k = 0; k < 4; ++k) {
        edge_[k] = std::hypot(cx_[(k + 1) % 4] - cx_[k], cy_[(k + 1) % 4] - cy_[k]);
    }
}

void RectangleTrajectory::startPoint(double& x, double& y) const {
    x = cx_[0];
    y = cy_[0];
}

double RectangleTrajectory::loopPeriod() const {
    return 2.0 * (p_.width + p_.height) / p_.speed;
}

void RectangleTrajectory::corner(int i, double& x, double& y) const {
    x = cx_[((i % 4) + 4) % 4];
    y = cy_[((i % 4) + 4) % 4];
}

void RectangleTrajectory::shapeAt(double s, RefPoint& out) const {
    // Distance along the perimeter, wrapped to one lap. Exactly at a corner the
    // point belongs to the edge LEAVING it, so the velocity there is the new
    // direction's -- the jump happens at the corner, not one sample after it.
    const double perimeter = 2.0 * (p_.width + p_.height);
    double d = std::fmod(p_.speed * s, perimeter);
    if (d < 0.0) d += perimeter;
    int k = 0;
    while (k < 3 && d >= edge_[k]) {
        d -= edge_[k];
        ++k;
    }
    const int k1 = (k + 1) % 4;
    const double ux = (cx_[k1] - cx_[k]) / edge_[k];
    const double uy = (cy_[k1] - cy_[k]) / edge_[k];
    out.x  = cx_[k] + ux * d;
    out.y  = cy_[k] + uy * d;
    out.vx = p_.speed * ux;
    out.vy = p_.speed * uy;
    out.ax = out.ay = 0.0;
    out.wz = out.az = 0.0;
}

// ============================== helpers =================================== //

std::unique_ptr<ReferenceTrajectory> makeTrajectory(const TrajectorySpec& spec,
                                                    const Anchor& anchor) {
    if (spec.pattern == "circle") {
        return std::make_unique<CircleTrajectory>(anchor, spec.circle);
    }
    if (spec.pattern == "rectangle") {
        return std::make_unique<RectangleTrajectory>(anchor, spec.rectangle);
    }
    throw std::invalid_argument("reference trajectory: unknown pattern '" + spec.pattern +
                                "' (expected circle or rectangle)");
}

bool sampleWindow(const ReferenceTrajectory& traj, double t_start, double t0,
                  double window_s, double dt, ReferenceWindow& out) {
    require(dt > 0.0, "sample dt must be positive");
    require(window_s >= 0.0, "window must be non-negative");
    const std::size_t wanted = static_cast<std::size_t>(std::floor(window_s / dt + 1e-9)) + 1;
    const std::size_t n = std::min(wanted, ReferenceWindow::kMaxPoints);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = t0 + dt * static_cast<double>(i);
        out.p[i] = traj.at(t - t_start);
        out.p[i].t = t;
    }
    out.n = static_cast<std::uint32_t>(n);
    return n == wanted;
}

}  // namespace pat_arm_control
