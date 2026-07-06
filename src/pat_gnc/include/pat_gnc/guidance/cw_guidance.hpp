#pragma once
#include <Eigen/Dense>
#include <vector>

namespace pat_gnc {

/// Clohessy-Wiltshire relative motion equations — Phase 2 implementation.
/// TODO Phase 2: implement phi() and approachWaypoints().
class CWGuidance {
public:
    struct Params { double n{0.0011}; };
    explicit CWGuidance(const Params& p = {}) : p_(p) {}
    Eigen::Matrix4d phi(double t) const noexcept;
    std::vector<Eigen::Vector2d> approachWaypoints(
        const Eigen::Vector2d& hold_point,
        const Eigen::Vector2d& approach_dir_unit,
        double standoff_m, int n_waypoints = 20) const;
private:
    Params p_;
};

} // namespace pat_gnc
