#include "pat_gnc/planar_dynamics.hpp"
#include <cmath>

namespace pat_gnc {

PlanarDynamics::State PlanarDynamics::f(const State& x, const Input& u) const noexcept {
    const double c = std::cos(x(2)), s = std::sin(x(2));
    State xd;
    xd(0) = x(3); xd(1) = x(4); xd(2) = x(5);
    xd(3) = (c*u(0) - s*u(1)) / p_.mass;
    xd(4) = (s*u(0) + c*u(1)) / p_.mass;
    xd(5) = u(2) / p_.inertia_z;
    return xd;
}

PlanarDynamics::State PlanarDynamics::step(const State& x, const Input& u, double dt) const noexcept {
    const State k1 = f(x,              u);
    const State k2 = f(x + 0.5*dt*k1, u);
    const State k3 = f(x + 0.5*dt*k2, u);
    const State k4 = f(x + dt*k3,     u);
    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

PlanarDynamics::State PlanarDynamics::relativeState(
    const State& chaser, const State& target) noexcept {
    const double c = std::cos(target(2)), s = std::sin(target(2));
    auto rot = [c, s](const Eigen::Vector2d& v) -> Eigen::Vector2d {
        return { c*v(0) + s*v(1), -s*v(0) + c*v(1) };
    };
    const Eigen::Vector2d dp = rot(chaser.head<2>() - target.head<2>());
    const Eigen::Vector2d dv = rot(chaser.segment<2>(3) - target.segment<2>(3));
    State rel;
    rel << dp(0), dp(1), chaser(2)-target(2), dv(0), dv(1), chaser(5)-target(5);
    return rel;
}

} // namespace pat_gnc
