#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"
#include "pat_gnc/navigation/imu_ekf.hpp"

namespace pat_gnc {

/// IMU-driven navigator: wraps ImuEkf (see imu_ekf.hpp for the filter and
/// the literature it follows). The IMU propagates; odometry corrects.
/// Activate by setting the 'navigator' ROS 2 parameter to "imu_ekf".
///
/// One odometry message carries two instruments -- relative-navigation
/// position and star-tracker yaw -- under one stamp, so the stamp cannot say
/// how old each part is. Each instrument's latency is therefore configured,
/// and a measurement is placed at (stamp - latency). A wrong latency shows up
/// as a bias in the estimate, not as a failure.
class ImuEkfNavigator final : public INavigator {
public:
    struct Config {
        ImuEkf::Params ekf;
        double latency_yaw = 0.0;   //!< [s] star tracker
        double latency_pos = 0.0;   //!< [s] relative-nav position
        double latency_vel = 0.0;   //!< [s] odometry twist, if used
        /// Fuse the odometry twist as a velocity measurement. Off by default:
        /// flight has no velocity sensor, and in the lab the mocap twist is a
        /// differentiated pose, so fusing it as well as the pose counts the
        /// same information twice.
        bool use_twist = false;
    };

    explicit ImuEkfNavigator(const Config& cfg);

    void onImu(double t, const Eigen::Vector2d& accel,
               double yaw_rate) noexcept override;
    void onOdometry(double t, const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& twist) noexcept override;

    StateVec state()      const noexcept override;
    StateCov covariance() const noexcept override;
    bool     ready()      const noexcept override { return ekf_.ready(); }
    /// Restart from x0's pose. Velocity and biases restart at zero with the
    /// filter's prior uncertainty: the filter cannot be told a velocity it
    /// has not estimated.
    void     reset(const StateVec& x0) noexcept override;

    const ImuEkf& filter() const noexcept { return ekf_; }
    const Config& config() const noexcept { return cfg_; }

private:
    Config cfg_;
    ImuEkf ekf_;
    bool have_imu_{false};
};

} // namespace pat_gnc
