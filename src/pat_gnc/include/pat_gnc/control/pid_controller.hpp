#pragma once
#include "pat_gnc/interfaces/i_controller.hpp"
#include <limits>

namespace pat_gnc {

/// MVP PID controller implementing IController.
/// Add LQR/MPC in control/ by implementing IController.
class PIDController final : public IController {
public:
    struct Gains {
        Eigen::Vector3d kp{Eigen::Vector3d::Zero()};
        Eigen::Vector3d ki{Eigen::Vector3d::Zero()};
        Eigen::Vector3d kd{Eigen::Vector3d::Zero()};
        Eigen::Vector3d u_max{
            Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity())};
    };

    PIDController(const Gains& g, double dt) : g_(g), dt_(dt) {}

    Eigen::Vector3d compute(const Eigen::Vector3d& error) noexcept override;
    void reset() noexcept override { integ_.setZero(); prev_.setZero(); }

private:
    Gains g_;
    double dt_;
    Eigen::Vector3d integ_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d prev_ {Eigen::Vector3d::Zero()};
};

} // namespace pat_gnc
