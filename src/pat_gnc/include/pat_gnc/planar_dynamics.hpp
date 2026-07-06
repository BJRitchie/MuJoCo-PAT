#pragma once
#include <Eigen/Dense>

namespace pat_gnc {

class PlanarDynamics {
public:
    using State = Eigen::Matrix<double, 6, 1>;
    using Input = Eigen::Matrix<double, 3, 1>;
    struct Params {
        double mass;
        double inertia_z;
        Params() : mass(15.0), inertia_z(0.05) {}
    };

    explicit PlanarDynamics(const Params& p = Params()) : p_(p) {}

    State f(const State& x, const Input& u) const noexcept;
    State step(const State& x, const Input& u, double dt) const noexcept;
    static State relativeState(const State& chaser, const State& target) noexcept;

private:
    Params p_;
};

} // namespace pat_gnc
