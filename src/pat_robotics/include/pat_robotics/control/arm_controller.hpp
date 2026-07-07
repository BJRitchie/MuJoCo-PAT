#pragma once
#include "pat_robotics/control/joint_pid.hpp"
#include <cstddef>
#include <vector>

namespace pat_robotics {

/// Independent per-joint PID position control for an N-DOF revolute arm.
/// N is determined at construction from the number of Gains passed in — not
/// hardcoded — so adding/removing joints is a config change, not a code
/// change. q/q_des/the returned torques are all matched by position; callers
/// must build them in the same joint order every time (see arm_control_node).
class ArmController {
public:
    ArmController(const std::vector<JointPID::Gains>& gains, double dt);

    /// @param q      current joint positions (rad)
    /// @param q_des  desired joint positions (rad)
    /// @return joint torques (N·m)
    std::vector<double> compute(const std::vector<double>& q,
                                 const std::vector<double>& q_des) noexcept;
    void reset() noexcept;
    size_t numJoints() const noexcept { return joints_.size(); }

private:
    std::vector<JointPID> joints_;
};

} // namespace pat_robotics
