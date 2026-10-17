#pragma once
/*! Planar IMU-driven error-state EKF for the chaser base.
 *
 *  The IMU is the PROCESS INPUT, not a measurement: gyro rate and body-frame
 *  specific force are integrated to propagate the state, and the slower
 *  absolute sensors -- star-tracker yaw and a relative-navigation position fix
 *  -- correct it. That way the filter needs no model of what moves the base.
 *  Arm reaction, thrust and contact all appear in the accelerometer directly,
 *  and none of them has to be predicted. This is the loosely coupled inertial
 *  navigation architecture of Groves, "Principles of GNSS, Inertial, and
 *  Multisensor Integrated Navigation Systems", 2nd ed. (2013), ch. 14,
 *  reduced to the plane.
 *
 *  ERROR-STATE (indirect) form, after Maybeck, "Stochastic Models, Estimation
 *  and Control" Vol. 1 (1979), and Sola, "Quaternion kinematics for the
 *  error-state Kalman filter", arXiv:1711.02508 (2017), sections 5-6. The
 *  nominal state is integrated exactly from the IMU; the filter's mean and
 *  covariance describe only the small error about it, which is injected into
 *  the nominal state after each update and then reset to zero. In the plane
 *  attitude is a single angle, so the error angle is additive and the reset
 *  Jacobian is the identity (Sola sec. 6, the ESKF reset, with G = I): the
 *  covariance is unchanged by the reset.
 *
 *  Gyro bias as a state, corrected by an absolute attitude sensor, is the
 *  spacecraft attitude filter of Lefferts, Markley & Shuster, "Kalman
 *  Filtering for Spacecraft Attitude Estimation", JGCD 5(5), 1982, reduced to
 *  one axis. The accelerometer bias is the same construction on the
 *  translational channel.
 *
 *    nominal / error state, 8:  [px, py, th, vx, vy, b_g, b_ax, b_ay]
 *                                world  world  world body  body
 *
 *  Allocation-free after construction: fixed-size Eigen throughout, so every
 *  call is safe on a realtime thread.
 */
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace pat_gnc {

class ImuEkf {
public:
    static constexpr int N = 8;
    using Vec  = Eigen::Matrix<double, N, 1>;
    using Mat  = Eigen::Matrix<double, N, N>;

    //! Index of each state, so no magic numbers appear outside this header.
    enum Idx : int { PX = 0, PY = 1, TH = 2, VX = 3, VY = 4, BG = 5, BAX = 6, BAY = 7 };

    /*! Noise model, in the SAME units and meaning as pat_simulation's
     *  SensorModel ChannelConfig, so a sensor configuration can be copied
     *  across without conversion.
     *
     *  The two *_noise_std are per-SAMPLE white noise sigmas (ChannelConfig's
     *  noise_std). The two *_bias_walk are continuous random-walk densities
     *  in units/sqrt(s) (ChannelConfig's bias_walk_std). That pairing is
     *  Farrenkopf's gyro model -- angle random walk on the rate plus rate
     *  random walk on the bias -- from "Analytic Steady-State Accuracy
     *  Solutions for Two Common Spacecraft Attitude Estimators", JGC 1(4),
     *  1978, applied to both IMU channels. */
    struct Params {
        double gyro_noise_std   = 0.005;   //!< [rad/s] per sample
        double accel_noise_std  = 0.01;    //!< [m/s^2] per sample
        double gyro_bias_walk   = 1e-4;    //!< [rad/s/sqrt(s)]
        double accel_bias_walk  = 1e-3;    //!< [m/s^2/sqrt(s)]

        double yaw_meas_std     = 0.002;   //!< [rad]  star tracker
        double pos_meas_std     = 0.005;   //!< [m]    relative-nav fix
        double vel_meas_std     = 0.01;    //!< [m/s]  optional velocity

        //! Initial 1-sigma uncertainty, applied by initialise().
        double init_pos_std     = 0.01;
        double init_yaw_std     = 0.01;
        double init_vel_std     = 0.05;
        double init_gyro_bias_std  = 0.01;
        double init_accel_bias_std = 0.05;

        //! Longest IMU interval integrated as one step [s]. A gap longer than
        //! this (a dropped stream, a paused clock) is not bridged by
        //! extrapolating one stale sample: the step is refused and counted.
        double max_imu_gap      = 0.1;

        /*! IMU steps of history kept for delayed measurements. A measurement
         *  stamped older than the oldest kept step is dropped and counted.
         *  This is also the worst-case replay length, so it bounds the cost of
         *  one update: size it as (longest sensor latency) x (IMU rate), with
         *  margin, and no larger. 256 at 500 Hz is 0.5 s. */
        std::size_t history_len = 256;

        /*! Innovation gates: a measurement whose normalised innovation squared
         *  exceeds the gate is rejected rather than applied. The NIS of a
         *  consistent filter is chi-square with the measurement's dimension,
         *  so the gate is a chi-square quantile (Bar-Shalom, Li & Kirubarajan
         *  2001, ch. 5). Defaults are the 99.9% points -- 1 in 1000 good
         *  measurements is lost, and a gross outlier never lands. 0 disables
         *  a gate. */
        double gate_yaw = 10.83;   //!< chi-square(1), 99.9%
        double gate_pos = 13.82;   //!< chi-square(2), 99.9%
        double gate_vel = 13.82;   //!< chi-square(2), 99.9%
    };

    struct Diagnostics {
        std::uint64_t imu_samples      = 0;
        std::uint64_t imu_gaps         = 0;   //!< intervals refused, > max_imu_gap
        std::uint64_t imu_out_of_order = 0;
        std::uint64_t yaw_updates      = 0;   //!< applied
        std::uint64_t pos_updates      = 0;
        std::uint64_t vel_updates      = 0;
        std::uint64_t yaw_rejected     = 0;   //!< failed the gate
        std::uint64_t pos_rejected     = 0;
        std::uint64_t vel_rejected     = 0;
        std::uint64_t too_old          = 0;   //!< older than the kept history
        std::uint64_t not_ready        = 0;   //!< arrived before initialise()
        std::uint64_t slot_overflow    = 0;   //!< see kMeasPerStep
        std::uint64_t replays          = 0;   //!< delayed measurements replayed
        std::size_t   max_replay_steps = 0;   //!< longest replay so far
        double last_nis_yaw = 0.0;
        double last_nis_pos = 0.0;
    };

    explicit ImuEkf(const Params& p);

    /*! Start the filter from a pose fix: position and yaw as given, velocity
     *  and both biases zero, with Params' initial sigmas. Until this is called
     *  propagate() only records the IMU sample time and ready() is false. */
    void initialise(double t, double px, double py, double yaw);

    /*! Integrate from the previous IMU sample to this one, taken at time `t`,
     *  using the mean of the two. `accel` is body-frame specific force [m/s^2] (with
     *  gravity disabled this is the proper acceleration), `gyro_z` is body
     *  yaw rate [rad/s]. Both are raw sensor readings; the estimated biases
     *  are removed here. */
    void propagate(double t, const Eigen::Vector2d& accel, double gyro_z);

    /*! The measurement updates. `t_meas` is when the measurement was TAKEN,
     *  not when it arrived.
     *
     *  A measurement nearest the latest IMU time is applied now. One stamped
     *  earlier is applied where it belongs, by state-history recomputation:
     *  the filter returns to the kept step nearest `t_meas`, applies it
     *  there, and re-runs every later IMU step AND every later measurement,
     *  so a late star-tracker fix cannot discard a position fix that arrived
     *  before it. This is the exact recomputation method that Larsen,
     *  Andersen, Ravn & Poulsen, "Incorporation of Time Delayed Measurements
     *  in a Discrete-time Kalman Filter", IEEE CDC 1998, use as their
     *  reference; their extrapolation shortcut is cheaper but approximate, and
     *  with an 8-state filter the exact method is affordable. The measurement
     *  is applied at the kept step nearest it, so its timing error is at most
     *  half an IMU interval.
     *
     *  Each returns the measurement's normalised innovation squared, computed
     *  where it was applied, or a negative value if it was not considered at
     *  all (not ready, or older than the history). A NIS above the gate means
     *  it was rejected. */
    double updateYaw(double t_meas, double yaw);
    //! Relative-navigation position fix, world frame [m].
    double updatePosition(double t_meas, const Eigen::Vector2d& p);
    //! World-frame velocity [m/s]. Flight has no such sensor; off by default
    //! in the navigator.
    double updateVelocity(double t_meas, const Eigen::Vector2d& v);

    bool ready() const noexcept { return initialised_; }
    double time() const noexcept { return t_; }
    const Vec& state() const noexcept { return x_; }
    const Mat& covariance() const noexcept { return P_; }

    //! Bias-corrected yaw rate from the latest gyro sample [rad/s]. The rate
    //! is an input, not a state, so this is the best estimate of it.
    double yawRate() const noexcept { return last_gyro_ - x_(BG); }
    //! Variance of yawRate(): the sample noise plus the bias uncertainty.
    double yawRateVariance() const noexcept;

    const Diagnostics& diagnostics() const noexcept { return diag_; }
    const Params& params() const noexcept { return p_; }

    /*! Measurements one kept step can hold for replay. Two of each kind within
     *  one 2 ms IMU interval would need a sensor faster than the IMU; if it
     *  happens the extra one is applied but not kept, so a later replay
     *  through that step omits it, and slot_overflow counts it. */
    static constexpr int kMeasPerStep = 4;

private:
    enum class Kind : std::uint8_t { Yaw, Pos, Vel };
    struct Meas {
        Kind kind = Kind::Yaw;
        double a = 0.0, b = 0.0;   //!< yaw in a; or x, y
    };

    /*! One kept IMU step: the state just AFTER propagating to `t` and BEFORE
     *  any measurement applied at `t`, the sample taken at `t` (the start of
     *  the next interval), and the measurements applied there. */
    struct Step {
        double t = 0.0;
        Vec x{Vec::Zero()};
        Mat P{Mat::Zero()};
        Eigen::Vector2d accel{Eigen::Vector2d::Zero()};
        double gyro = 0.0;
        //! False when the interval ending here was refused (a gap): the state
        //! carries over unchanged, and a replay must do the same.
        bool integrated = true;
        Meas meas[kMeasPerStep];
        int n_meas = 0;
    };

    //! Propagate x_/P_ across [t0, t1] using the two end samples. No
    //! bookkeeping, so a replay can call it without recounting anything.
    void integrate(double dt, const Eigen::Vector2d& a0, double g0,
                   const Eigen::Vector2d& a1, double g1);

    //! Route a measurement: now, by replay, or not at all.
    double insert(double t_meas, const Meas& m);
    //! Apply one measurement to x_/P_. `first` is false during a replay, so
    //! the counters describe measurements, not recomputations of them.
    double apply(const Meas& m, bool first);

    /*! One-measurement-model update: innovation `y` with Jacobian `H` (rows
     *  M) and noise R. If `gate` > 0 and the NIS exceeds it, nothing is
     *  changed and `accepted` is false. Otherwise a Joseph-form covariance
     *  update, then the error is injected into the nominal state. Returns the
     *  NIS either way. */
    template <int M>
    double correct(const Eigen::Matrix<double, M, 1>& y,
                   const Eigen::Matrix<double, M, N>& H,
                   const Eigen::Matrix<double, M, M>& R, double gate,
                   bool& accepted);

    void pushStep(bool integrated);
    Step& stepAt(std::size_t i);          //!< 0 = oldest kept
    static double wrapAngle(double a) noexcept;

    //! The kept history, a ring sized once in the constructor.
    std::vector<Step> hist_;
    std::size_t hist_head_ = 0;           //!< index of the oldest
    std::size_t hist_size_ = 0;

    Params p_;
    Vec x_{Vec::Zero()};
    Mat P_{Mat::Identity()};
    double t_ = 0.0;
    bool initialised_ = false;
    //! The previous IMU sample: each interval is integrated from its two end
    //! samples, and yawRate() reports the latest.
    double last_gyro_ = 0.0;
    Eigen::Vector2d last_accel_{Eigen::Vector2d::Zero()};
    bool have_sample_ = false;
    Diagnostics diag_;
};

}  // namespace pat_gnc
