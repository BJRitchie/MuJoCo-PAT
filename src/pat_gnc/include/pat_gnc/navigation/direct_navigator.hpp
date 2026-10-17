#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"

namespace pat_gnc {

/// MVP navigator: passes odometry pose and twist directly as the state estimate.
/// Works well in simulation where odometry is exact. Upgrade to EkfNavigator
/// for real hardware noise rejection.
class DirectNavigator final : public INavigator {
public:
    void onImu(double, const Eigen::Vector2d&, double) noexcept override {}

    void onOdometry(double, const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& twist) noexcept override {
        x_.head<3>() = pose;
        x_.tail<3>() = twist;
        have_odom_ = true;
    }

    StateVec state()      const noexcept override { return x_; }
    StateCov covariance() const noexcept override { return StateCov::Zero(); }
    bool     ready()      const noexcept override { return have_odom_; }
    void     reset(const StateVec& x0) noexcept override { x_ = x0; }

private:
    StateVec x_{StateVec::Zero()};
    bool have_odom_{false};
};

} // namespace pat_gnc
