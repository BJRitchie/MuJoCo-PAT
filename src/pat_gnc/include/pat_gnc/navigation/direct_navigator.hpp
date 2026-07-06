#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"

namespace pat_gnc {

/// MVP navigator: passes odometry pose and twist directly as the state estimate.
/// Works well in simulation where odometry is exact. Upgrade to EkfNavigator
/// for real hardware noise rejection.
class DirectNavigator final : public INavigator {
public:
    void onImu(double) noexcept override {}

    void onOdometry(const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& twist) noexcept override {
        x_.head<3>() = pose;
        x_.tail<3>() = twist;
    }

    StateVec state()             const noexcept override { return x_; }
    void     reset(const StateVec& x0) noexcept override { x_ = x0; }

private:
    StateVec x_{StateVec::Zero()};
};

} // namespace pat_gnc
