#include "pat_robotics/control/joint_pid.hpp"
#include <algorithm>

namespace pat_robotics {

double JointPID::compute(double error) noexcept {
    integ_ += error * dt_;
    const double deriv = (error - prev_) / dt_;
    prev_ = error;
    const double u = g_.kp * error + g_.ki * integ_ + g_.kd * deriv;
    return std::clamp(u, -g_.u_max, g_.u_max);
}

} // namespace pat_robotics
