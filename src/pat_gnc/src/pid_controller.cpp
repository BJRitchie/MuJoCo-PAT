#include "pat_gnc/control/pid_controller.hpp"

namespace pat_gnc {

Eigen::Vector3d PIDController::compute(const Eigen::Vector3d& error) noexcept {
    integ_ += error * dt_;
    const Eigen::Vector3d deriv = (error - prev_) / dt_;
    prev_ = error;
    const Eigen::Vector3d u =
        g_.kp.cwiseProduct(error) +
        g_.ki.cwiseProduct(integ_) +
        g_.kd.cwiseProduct(deriv);
    return u.cwiseMax(-g_.u_max).cwiseMin(g_.u_max);
}

} // namespace pat_gnc
