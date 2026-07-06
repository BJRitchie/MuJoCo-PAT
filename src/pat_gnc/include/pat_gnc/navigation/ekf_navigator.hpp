#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"
#include "pat_gnc/navigation/planar_ekf.hpp"

namespace pat_gnc {

/// Phase 2 navigator: wraps PlanarEKF.
/// Uses pose measurements for EKF update; IMU arrival rate drives prediction.
/// Activate by setting the 'navigator' ROS 2 parameter to "ekf".
class EkfNavigator final : public INavigator {
public:
    explicit EkfNavigator(const PlanarEKF::Params& params) : ekf_(params) {}

    void onImu(double /*yaw_rate*/) noexcept override {
        ekf_.predict();
        // TODO Phase 2: incorporate yaw_rate into the IMU-model predict step
    }

    void onOdometry(const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& /*twist*/) noexcept override {
        ekf_.update(pose);
    }

    StateVec state() const noexcept override { return ekf_.state(); }

    void reset(const StateVec& x0) noexcept override {
        ekf_.reset(x0, PlanarEKF::StateMat::Identity() * 0.01);
    }

private:
    PlanarEKF ekf_;
};

} // namespace pat_gnc
