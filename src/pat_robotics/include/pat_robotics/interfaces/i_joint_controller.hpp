#pragma once

namespace pat_robotics {

/// Abstract interface for a single-joint (scalar) controller.
/// One instance runs per joint — an N-joint arm holds N independent instances.
class IJointController {
public:
    virtual ~IJointController() = default;
    /// Compute torque command for scalar position error (rad).
    virtual double compute(double error) noexcept = 0;
    virtual void reset() noexcept = 0;
};

} // namespace pat_robotics
