#pragma once
#include <Eigen/Dense>

namespace pat_gnc {

/// Abstract interface for state estimation and navigation.
/// Add EKF, UKF, or other estimators by implementing this interface.
/// Select at runtime via the 'navigator' ROS 2 parameter.
class INavigator {
public:
    /// State: [px, py, theta, vx, vy, omega]  (m, m, rad, m/s, m/s, rad/s)
    using StateVec = Eigen::Matrix<double, 6, 1>;

    virtual ~INavigator() = default;

    /// High-rate IMU-driven prediction step.
    virtual void onImu(double yaw_rate) noexcept = 0;

    /// Odometry update from motion capture (lab) or GPS/vision (real hardware).
    /// @param pose   [x, y, theta]
    /// @param twist  [vx, vy, omega] — zeros are acceptable if unavailable
    virtual void onOdometry(const Eigen::Vector3d& pose,
                            const Eigen::Vector3d& twist) noexcept = 0;

    virtual StateVec state()             const noexcept = 0;
    virtual void     reset(const StateVec& x0) noexcept = 0;
};

} // namespace pat_gnc
