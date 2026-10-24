#pragma once
/*! What the controller is told about the plant: the exact state, a seeded noisy
 *  measurement of it, or the flight sensors, raw or through the base state
 *  estimator.
 *
 *  The bench reads the plant's true state every tick. A StateSource turns that
 *  into the `q, v` the controllers and the disturbance observer actually see:
 *
 *    exact   the plant's state, untouched: the benches' behaviour until now
 *    noisy   every measured quantity corrupted independently with seeded noise
 *            (and optionally a delay of whole control ticks): encoders on the
 *            joints, a mocap-like position / attitude / velocity sensor on the
 *            base, every tick
 *    ekf     the joints read as encoders (noisy as above), and the base state
 *            taken from pat_gnc's ImuEkf, fed by seeded IMU, relative-position
 *            and star-tracker measurements at the rates and latencies of the
 *            flight sensing configuration, exactly as nav_bench feeds it
 *    raw     the same flight sensors with no estimator: the last relative-nav
 *            fix as position, the last star-tracker reading as yaw, the gyro as
 *            yaw rate, and the difference between the last two fixes as
 *            velocity
 *
 *  The flight instruments of ekf and raw are the same seeded channels, so the
 *  two see identical sensor noise for one seed.
 *
 *  In every source but `exact`, the joint readings can be passed through a
 *  JointKalmanFilter (`joint_filter`). In the flight sources, `noise_scale`
 *  multiplies every flight sensor error (noise, bias and bias walk) and the
 *  joints', and the filters are told the scaled figures, so they stay told the
 *  truth; `delay_ticks` delays the base state handed over as well as the joints.
 *
 *  The corruption is the same seeded SensorModel the simulator uses, so a
 *  stream depends only on the seed and the channel's name, and two runs of one
 *  configuration are identical. With `exact`, nothing random runs at all.
 *
 *  Scoring always uses the plant's truth, not what the controller was handed;
 *  lastError() reports the difference so the estimator's own accuracy is in the
 *  record.
 */
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "pat_arm_control/estimation/joint_kalman_filter.h"
#include "pat_gnc/navigation/imu_ekf_navigator.hpp"
#include "pat_sensors/sensor_model.hpp"

namespace pat_arm_control {
namespace bench {

class ContactScenario;

enum class StateSourceKind { Exact, Noisy, Ekf, Raw };

/*! 1-sigma of each measured quantity, before the noise scale. The joint-velocity
 *  figure is the one the observer analysis used (0.01 rad/s); the base position,
 *  attitude and the encoder position are the flight profile's nav and star
 *  tracker figures and an assumed encoder. They are assumptions, not datasheet
 *  values, which is why the studies sweep the scale rather than rely on one
 *  setting. */
struct NoiseProfile {
    double joint_pos = 5e-4;   //!< [rad]   joint encoder position
    double joint_vel = 1e-2;   //!< [rad/s] joint velocity
    double base_pos  = 5e-3;   //!< [m]     base x and y
    double base_yaw  = 5e-4;   //!< [rad]   base attitude
    double base_vel  = 5e-3;   //!< [m/s]   base vx and vy
    double base_wz   = 1e-3;   //!< [rad/s] base yaw rate

    //! Set one figure by name (joint_pos, joint_vel, base_pos, base_yaw,
    //! base_vel, base_wz). Throws std::runtime_error on an unknown name.
    void set(const std::string& name, double value);
};

struct StateSourceConfig {
    StateSourceKind kind = StateSourceKind::Exact;
    double noise_scale = 1.0;          //!< multiplies every sigma in `profile`
    int delay_ticks = 0;               //!< measurement delay, in control ticks
    std::uint64_t seed = 0;
    NoiseProfile profile;
    //! ekf / raw: the simulator's base params, its flight sensor overlay (both
    //! node `pat_simulation`) and the estimator's params (node `pat_nav`).
    std::string sim_yaml, sensors_yaml, nav_yaml;
    //! Pass the joint readings through a JointKalmanFilter. Its measurement
    //! sigmas are set from `profile` x `noise_scale`; accel_psd is used as given.
    bool joint_filter = false;
    JointKalmanParams joint_kf;
};

//! The state handed to the controller minus the plant's truth.
struct BaseError {
    double x = 0.0, y = 0.0, yaw = 0.0, vx = 0.0, vy = 0.0, wz = 0.0;
};

class StateSource {
public:
    /*! `n_joints` model joints, the control period and the physics step [s].
     *  Throws std::runtime_error on an unreadable or incomplete YAML. */
    StateSource(const StateSourceConfig& cfg, int n_joints, double control_dt,
                double physics_dt);
    ~StateSource();

    bool exact() const { return cfg_.kind == StateSourceKind::Exact; }
    bool isEkf() const { return cfg_.kind == StateSourceKind::Ekf; }
    //! ekf or raw: the flight instruments are fed every physics step.
    bool isFlight() const {
        return cfg_.kind == StateSourceKind::Ekf || cfg_.kind == StateSourceKind::Raw;
    }

    /*! Call after every physics step with the number of steps completed so far.
     *  Advances the flight instruments (and feeds the estimator) at their own
     *  rates; does nothing unless the source is a flight source. */
    void onSubstep(ContactScenario& scene, long completed_steps);

    /*! The state the controller sees, from the plant's truth. `q` and `v` use the
     *  controller's layout (readControllerState's): base position plus anchor,
     *  base yaw as a quaternion in q(3), q(6), world-frame base velocity. */
    void measure(const Eigen::VectorXd& q_true, const Eigen::VectorXd& v_true,
                 double anchor_x, double anchor_y, Eigen::VectorXd& q, Eigen::VectorXd& v);

    //! What measure() last handed over minus the truth it was given. All zero
    //! for `exact`, and while the estimator has not yet been initialised.
    const BaseError& lastError() const { return err_; }
    //! True once the estimator has taken its first fix (always true otherwise).
    bool ready() const;

private:
    struct Impl;
    StateSourceConfig cfg_;
    std::unique_ptr<Impl> impl_;
    BaseError err_;
};

}  // namespace bench
}  // namespace pat_arm_control
