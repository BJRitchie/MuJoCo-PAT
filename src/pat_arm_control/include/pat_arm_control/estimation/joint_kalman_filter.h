#pragma once
#include <vector>

#include <Eigen/Dense>

namespace pat_arm_control {

/*! Tuning for JointKalmanFilter. */
struct JointKalmanParams {
    /*! Power spectral density of the white acceleration that drives each
     *  joint's model [(rad/s^2)^2 / Hz]. It is how far the filter believes the
     *  joint can change velocity between readings: larger follows a fast
     *  manoeuvre with less lag and passes more encoder noise, smaller smooths
     *  more and lags more.
     *
     *  3e-3 is set by contact_bench on the arm (tuned SQP-NMPC, flight sensing):
     *  at 1x the command's tick-to-tick torque change halves (39 to 19 mN m)
     *  with no loss of tracking, and at 3x a circle still tracks as well as
     *  with no filter, where 1e-3 lags it into safe mode. */
    double accel_psd = 3e-3;
    double pos_meas_std = 5e-4;   //!< [rad]   encoder position, 1 sigma per reading
    /*! [rad/s] the encoder driver's velocity, 1 sigma per reading. Zero or
     *  negative: the driver gives no velocity, and it is inferred from the
     *  positions alone. */
    double vel_meas_std = 1e-2;
    double init_vel_std = 0.1;    //!< [rad/s] uncertainty of the first velocity
};

/*! One independent Kalman filter per arm joint, for encoder readings.
 *
 *  Each joint is the discretised continuous white-noise-acceleration model of
 *  Bar-Shalom, Li & Kirubarajan, "Estimation with Applications to Tracking and
 *  Navigation" (2001), sec. 6.2.2: state [q, qdot], driven by white
 *  acceleration of density `accel_psd`,
 *
 *      F = [1 h; 0 1]      Q = accel_psd [h^3/3  h^2/2; h^2/2  h]
 *
 *  measured directly in both components (H = I, R = diag(pos^2, vel^2)), or in
 *  position alone when the driver gives no velocity. The joints are filtered
 *  independently: the model does not know the arm's dynamics, so the coupling
 *  between joints appears as acceleration, which accel_psd has to cover.
 *
 *  It improves the velocity most. A velocity reading is the noisiest signal an
 *  encoder driver gives, and every controller here and the disturbance observer
 *  use it; the filter combines it with the change in the (much more precise)
 *  position.
 *
 *  update() allocates nothing: every buffer is sized in the constructor, so it
 *  shares the control path's realtime constraints. */
class JointKalmanFilter {
public:
    /*! `n_joints` filters with the same tuning. Throws std::invalid_argument
     *  on a negative or non-finite setting, or a zero position sigma. */
    JointKalmanFilter(const JointKalmanParams& p, int n_joints);

    /*! One reading of every joint, `h` seconds after the previous one. The
     *  first call initialises each filter at the reading. `qd` is ignored when
     *  vel_meas_std <= 0. Both are n_joints long. Throws std::invalid_argument
     *  on a non-positive `h` after the first call. */
    void update(double h, const Eigen::Ref<const Eigen::VectorXd>& q,
                const Eigen::Ref<const Eigen::VectorXd>& qd);

    bool ready() const noexcept { return initialised_; }
    int size() const noexcept { return static_cast<int>(x_.size()); }
    double position(int i) const { return x_[static_cast<std::size_t>(i)](0); }
    double velocity(int i) const { return x_[static_cast<std::size_t>(i)](1); }
    //! Joint i's covariance [q, qdot].
    const Eigen::Matrix2d& covariance(int i) const { return P_[static_cast<std::size_t>(i)]; }
    const JointKalmanParams& params() const noexcept { return p_; }

    void reset() noexcept { initialised_ = false; }

private:
    JointKalmanParams p_;
    std::vector<Eigen::Vector2d> x_;
    std::vector<Eigen::Matrix2d> P_;
    bool initialised_ = false;
};

}  // namespace pat_arm_control
