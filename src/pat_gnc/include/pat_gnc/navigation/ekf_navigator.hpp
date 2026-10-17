#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"
#include "pat_gnc/navigation/planar_ekf.hpp"

namespace pat_gnc {

/// Constant-velocity pose filter: wraps PlanarEKF.
/// IMU arrival only drives a fixed-dt prediction step -- the sample's values
/// and timestamps are not used -- and pose measurements correct it. Suited to
/// mocap-rate pose; for IMU-driven navigation with biases and sensor latency,
/// use ImuEkfNavigator ('navigator: imu_ekf').
/// Activate by setting the 'navigator' ROS 2 parameter to "ekf".
class EkfNavigator final : public INavigator {
public:
    explicit EkfNavigator(const PlanarEKF::Params& params) : ekf_(params) {}

    void onImu(double, const Eigen::Vector2d&, double) noexcept override {
        ekf_.predict();
    }

    void onOdometry(double, const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& /*twist*/) noexcept override {
        ekf_.update(pose);
        have_odom_ = true;
    }

    StateVec state()      const noexcept override { return ekf_.state(); }
    StateCov covariance() const noexcept override { return ekf_.covariance(); }
    bool     ready()      const noexcept override { return have_odom_; }

    void reset(const StateVec& x0) noexcept override {
        ekf_.reset(x0, PlanarEKF::StateMat::Identity() * 0.01);
    }

private:
    PlanarEKF ekf_;
    bool have_odom_{false};
};

} // namespace pat_gnc
