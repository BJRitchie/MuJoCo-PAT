#pragma once
#include <Eigen/Dense>
#include <vector>

namespace pat_robotics {

/// Abstract interface for manipulator arm control.
/// Implement for your specific arm geometry, DOF count, and control law.
/// Full implementation arrives in Phase 3.
class IManipulator {
public:
    virtual ~IManipulator() = default;

    /// Forward kinematics: joint angles → end-effector pose [x,y,z,rx,ry,rz]
    virtual Eigen::Matrix<double, 6, 1> forwardKinematics(
        const std::vector<double>& q) const = 0;

    /// Inverse kinematics: desired pose → joint angles.
    /// @return true if a solution was found
    virtual bool inverseKinematics(
        const Eigen::Matrix<double, 6, 1>& target_pose,
        std::vector<double>& q_out) const = 0;

    /// Geometric Jacobian J(q)
    virtual Eigen::MatrixXd jacobian(const std::vector<double>& q) const = 0;
};

} // namespace pat_robotics
