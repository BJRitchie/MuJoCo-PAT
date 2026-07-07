#include "pat_robotics/control/arm_controller.hpp"
#include <cmath>

namespace pat_robotics {

ArmController::ArmController(const std::vector<JointPID::Gains>& gains, double dt) {
    joints_.reserve(gains.size());
    for (const auto& g : gains) joints_.emplace_back(g, dt);
}

std::vector<double> ArmController::compute(const std::vector<double>& q,
                                            const std::vector<double>& q_des) noexcept {
    std::vector<double> u(joints_.size(), 0.0);
    for (size_t i = 0; i < joints_.size() && i < q.size() && i < q_des.size(); ++i) {
        // Revolute joints: wrap error to [-pi, pi] per project convention.
        const double err = std::remainder(q_des[i] - q[i], 2.0 * M_PI);
        u[i] = joints_[i].compute(err);
    }
    return u;
}

void ArmController::reset() noexcept {
    for (auto& j : joints_) j.reset();
}

} // namespace pat_robotics
