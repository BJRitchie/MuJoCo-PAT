#pragma once
#include <Eigen/Dense>

namespace pat_gnc {

/// Abstract interface for state estimation and navigation.
/// Add EKF, UKF, or other estimators by implementing this interface.
/// Select at runtime via the 'navigator' ROS 2 parameter.
///
/// Every input carries the time it was TAKEN (the message header stamp, in
/// seconds), so an estimator that handles sensor latency can place it, and one
/// that does not can ignore it.
class INavigator {
public:
    /// State: [px, py, theta, vx, vy, omega]  (m, m, rad, m/s, m/s, rad/s)
    using StateVec = Eigen::Matrix<double, 6, 1>;
    /// Covariance of StateVec, same ordering.
    using StateCov = Eigen::Matrix<double, 6, 6>;

    virtual ~INavigator() = default;

    /// IMU sample: body-frame specific force [m/s^2] (x, y) and body yaw rate
    /// [rad/s], raw, biases included.
    virtual void onImu(double t, const Eigen::Vector2d& accel,
                       double yaw_rate) noexcept = 0;

    /// Odometry update from motion capture (lab) or relative navigation and a
    /// star tracker (flight).
    /// @param pose   [x, y, theta]
    /// @param twist  [vx, vy, omega] — zeros are acceptable if unavailable
    virtual void onOdometry(double t, const Eigen::Vector3d& pose,
                            const Eigen::Vector3d& twist) noexcept = 0;

    virtual StateVec state()      const noexcept = 0;
    /// The estimate's uncertainty. A navigator that passes measurements
    /// through unfiltered has none of its own to report and returns zero.
    virtual StateCov covariance() const noexcept = 0;
    /// False until the navigator has enough input to give a meaningful state.
    virtual bool     ready()      const noexcept = 0;
    virtual void     reset(const StateVec& x0) noexcept = 0;
};

} // namespace pat_gnc
