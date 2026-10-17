#include "pat_gnc/navigation/imu_ekf_navigator.hpp"

namespace pat_gnc {

ImuEkfNavigator::ImuEkfNavigator(const Config& cfg) : cfg_(cfg), ekf_(cfg.ekf) {}

void ImuEkfNavigator::onImu(double t, const Eigen::Vector2d& accel,
                            double yaw_rate) noexcept {
    ekf_.propagate(t, accel, yaw_rate);
    have_imu_ = true;
}

void ImuEkfNavigator::onOdometry(double t, const Eigen::Vector3d& pose,
                                 const Eigen::Vector3d& twist) noexcept {
    if (!ekf_.ready()) {
        // The first fix starts the filter. It starts at the latest IMU time,
        // not at the fix's own (older) time: the history begins there, and
        // the fix's age is absorbed by the initial uncertainty rather than by
        // an IMU interval longer than the gap limit.
        ekf_.initialise(have_imu_ ? ekf_.time() : t, pose.x(), pose.y(), pose.z());
        last_pos_ = pose.head<2>();
        last_yaw_ = pose.z();
        last_vel_ = twist.head<2>();
        have_last_ = true;
        return;
    }
    // Exact comparison on purpose: a repeat is the same double the driver held,
    // not a nearby one.
    const bool skip = cfg_.skip_repeated && have_last_;
    if (cfg_.use_position && !(skip && pose.head<2>() == last_pos_)) {
        ekf_.updatePosition(t - cfg_.latency_pos, pose.head<2>());
        last_pos_ = pose.head<2>();
    }
    if (!(skip && pose.z() == last_yaw_)) {
        ekf_.updateYaw(t - cfg_.latency_yaw, pose.z());
        last_yaw_ = pose.z();
    }
    if (cfg_.use_twist && !(skip && twist.head<2>() == last_vel_)) {
        ekf_.updateVelocity(t - cfg_.latency_vel, twist.head<2>());
        last_vel_ = twist.head<2>();
    }
    have_last_ = true;
}

INavigator::StateVec ImuEkfNavigator::state() const noexcept {
    const ImuEkf::Vec& x = ekf_.state();
    StateVec s;
    s << x(ImuEkf::PX), x(ImuEkf::PY), x(ImuEkf::TH), x(ImuEkf::VX), x(ImuEkf::VY),
        ekf_.yawRate();
    return s;
}

INavigator::StateCov ImuEkfNavigator::covariance() const noexcept {
    // [px, py, th, vx, vy] map one-to-one onto the filter's first five states.
    // The yaw rate is not a state: it is the gyro sample minus the estimated
    // bias, so its variance is the sample noise plus the bias variance, and
    // its covariance with every other state is minus that state's covariance
    // with the bias.
    const ImuEkf::Mat& P = ekf_.covariance();
    StateCov C = StateCov::Zero();
    C.topLeftCorner<5, 5>() = P.topLeftCorner<5, 5>();
    for (int i = 0; i < 5; ++i) {
        C(5, i) = C(i, 5) = -P(ImuEkf::BG, i);
    }
    C(5, 5) = ekf_.yawRateVariance();
    return C;
}

void ImuEkfNavigator::reset(const StateVec& x0) noexcept {
    ekf_.initialise(ekf_.time(), x0(0), x0(1), x0(2));
}

}  // namespace pat_gnc
