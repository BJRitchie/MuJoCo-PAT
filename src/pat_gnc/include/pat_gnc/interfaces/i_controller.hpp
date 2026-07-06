#pragma once
#include <Eigen/Dense>

namespace pat_gnc {

/// Abstract interface for 3-axis planar controllers.
/// Add LQR, MPC, or other control laws by implementing this interface.
/// Select at runtime via the 'controller' ROS 2 parameter.
class IController {
public:
    virtual ~IController() = default;
    /// Compute control demand for error [ex, ey, e_yaw].
    /// @return [ux, uy, u_yaw]  (N, N, N·m)
    virtual Eigen::Vector3d compute(const Eigen::Vector3d& error) noexcept = 0;
    virtual void reset() noexcept = 0;
};

} // namespace pat_gnc
