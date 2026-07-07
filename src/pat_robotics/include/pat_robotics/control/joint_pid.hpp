#pragma once
#include "pat_robotics/interfaces/i_joint_controller.hpp"
#include <limits>

namespace pat_robotics {

/// Scalar PID for a single revolute joint. Torque output, direct actuation
/// (no internal MuJoCo servo) — see MJCF <motor> actuators.
class JointPID final : public IJointController {
public:
    struct Gains {
        double kp{0.0};
        double ki{0.0};
        double kd{0.0};
        double u_max{std::numeric_limits<double>::infinity()};
    };

    JointPID(const Gains& g, double dt) : g_(g), dt_(dt) {}

    double compute(double error) noexcept override;
    void reset() noexcept override { integ_ = 0.0; prev_ = 0.0; }

private:
    Gains g_;
    double dt_;
    double integ_{0.0};
    double prev_{0.0};
};

} // namespace pat_robotics
