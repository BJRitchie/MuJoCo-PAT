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
/// and a measurement is placed at (stamp - latency).
///
/// The latency has to be RIGHT, to about a millisecond. While the body turns,
/// a timing error of dt is a yaw innovation of omega * dt, and against an
/// accurate star tracker that is many sigma: in pat_simulation's nav_bench at
/// ~0.3 rad/s, 1 ms of error raises the NEES from 8 to 11-15 and 10 ms has the
/// innovation gate rejecting a quarter of all star-tracker fixes. A latency
/// that cannot be known that well calls for drivers that stamp each reading
/// with its acquisition time, so the latency here can be zero.
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
        /// Apply each instrument's reading only when it differs from the last
        /// one applied. A sensor slower than the odometry message carrying it
        /// is republished by a sample-and-hold driver -- a 2 Hz position fix
        /// rides on 100 Hz odometry fifty times -- and fusing every repeat as
        /// an independent measurement makes the filter wildly overconfident.
        /// A genuinely new noisy reading is never bit-identical to the last,
        /// so this drops repeats and nothing else.
        bool skip_repeated = true;
        /// Fuse the odometry position. Off models flight with no
        /// relative-navigation sensor: position and accelerometer bias are then
        /// unobservable, and the filter dead-reckons with a covariance that
        /// grows to say so. The first pose still initialises the filter.
        bool use_position = true;
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
    //! The last reading applied per instrument, for skip_repeated.
    Eigen::Vector2d last_pos_{Eigen::Vector2d::Zero()};
    Eigen::Vector2d last_vel_{Eigen::Vector2d::Zero()};
    double last_yaw_{0.0};
    bool have_last_{false};
};

} // namespace pat_gnc
