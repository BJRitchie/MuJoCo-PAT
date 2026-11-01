#pragma once
#include "pat_gnc/interfaces/i_navigator.hpp"

namespace pat_gnc {

/// The flight sensors with no estimator: the baseline the IMU EKF is judged
/// against. Activate by setting the 'navigator' ROS 2 parameter to "raw".
///
///   position  the last relative-navigation fix, as it arrives (held, late)
///   yaw       the last star-tracker reading, as it arrives
///   yaw rate  the last gyro sample, raw (bias and noise included)
///   velocity  the difference between the last two DISTINCT fixes, over the
///             time between their arrivals; zero until there are two
///
/// A sensor slower than the odometry message carrying it is republished by a
/// sample-and-hold driver, so a new fix is one whose position changed: a
/// genuinely new noisy reading is never bit-identical to the last (the same
/// rule as ImuEkfNavigator's skip_repeated).
///
/// This is exactly pat_arm_control's contact_bench `--state-source raw`
/// (src/bench/state_source.cpp), so the live stack and the bench compare the
/// same no-estimator case. Keep the two in step.
class RawFlightNavigator final : public INavigator {
public:
    void onImu(double, const Eigen::Vector2d&, double yaw_rate) noexcept override {
        x_(5) = yaw_rate;
    }

    void onOdometry(double t, const Eigen::Vector3d& pose,
                    const Eigen::Vector3d& /*twist: flight has no velocity sensor*/)
        noexcept override {
        x_(0) = pose(0);
        x_(1) = pose(1);
        x_(2) = pose(2);
        if (!have_fix_) {
            have_fix_ = true;
        } else if (pose(0) != fix_x_ || pose(1) != fix_y_) {
            const double dt = t - fix_t_;
            if (dt > 0.0) {
                x_(3) = (pose(0) - fix_x_) / dt;
                x_(4) = (pose(1) - fix_y_) / dt;
            }
        } else {
            return;                       // a repeat of the held fix
        }
        fix_t_ = t;
        fix_x_ = pose(0);
        fix_y_ = pose(1);
    }

    StateVec state()      const noexcept override { return x_; }
    StateCov covariance() const noexcept override { return StateCov::Zero(); }
    bool     ready()      const noexcept override { return have_fix_; }
    void     reset(const StateVec& x0) noexcept override {
        x_ = x0;
        have_fix_ = false;
    }

private:
    StateVec x_{StateVec::Zero()};
    bool have_fix_{false};
    double fix_t_{0.0}, fix_x_{0.0}, fix_y_{0.0};
};

} // namespace pat_gnc
