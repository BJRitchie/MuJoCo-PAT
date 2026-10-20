#pragma once
/*! Parameters every arm control law shares: which joints it owns and their
 *  limits, the end-effector site, the safe mode, the disturbance feedforward
 *  and the reference handling. A law's own parameters derive from this
 *  (NMPCParams : ControllerParams), so a law's code reads both the same way.
 *  Zero ROS.
 */
#include <string>
#include <utility>
#include <vector>

namespace pat_arm_control {

/*! Position/velocity limits for a joint this controller OWNS -- i.e. whose
 *  Δq/qdot rows become box-constrained state. Torque limits are not here:
 *  they apply to every model joint, owned or not, and live in
 *  ControllerParams::model_torque_lims. */
struct JointLimit { std::string name; double q_min, q_max, qd_max; };

struct ControllerParams {
    std::string ee_site_name;     //!< name of the MJCF <site> at the end effector (required)

    // --- Joints and limits (required) ----------------------------------------
    // Position/velocity limits for the OWNED joints.
    std::vector<JointLimit> joint_lims;

    /*! Torque limit [N*m] for EVERY joint in the model, as {name, tau_max}.
     *  The torque constraint and the rollout clamp both cover all model joints
     *  -- a controller rolls the whole model forward, including the arm it
     *  does not own -- so a limit is needed for each one, and this is the
     *  single place they are set. Must name every model joint. */
    std::vector<std::pair<std::string, double>> model_torque_lims;

    // Joint names this controller owns (whose Δq/qdot rows become
    // box-constrained state). Empty ⇒ owns all arm joints.
    std::vector<std::string> owned_joints;

    //! DLS damping for H_g inversion (see FloatingBaseDynamics::pinvDLS).
    //! pinvDLS scales each singular value by s/(s²+λ²), so λ has to sit well
    //! below H_g's smallest singular value or the inverse is distorted rather
    //! than merely regularised. A planar Piper arm's smallest is ~5e-3, where
    //! λ=1e-3 costs ~3.4% and λ=1e-4 costs ~0.03%, still bounding the inverse
    //! at 1/(2λ) near a genuine singularity.
    double Hg_damping  = 1e-4;

    // --- Safe mode -----------------------------------------------------------
    /*! Joint damping gain for safe mode, as a fraction of the torque available
     *  at each joint's rated speed: kd = frac * tau_max / qd_max. Derived the
     *  same way du_max_frac is, and for the same reason — an absolute N*m*s/rad
     *  gain cannot track the torque limits. Sizing it this way also makes safe
     *  mode non-saturating by construction: |tau| <= frac*tau_max whenever
     *  |qdot| <= qd_max. Note the torque limit, not this gain, sets how quickly
     *  the arm can actually stop (H_ii*qdot/tau_max at best). */
    double safe_mode_damping_frac = 1.0;

    /*! Consecutive successful solves required to leave safe mode. Pure
     *  hysteresis against solve-failure chatter — it does NOT ramp the handover,
     *  which steps straight from damping to the full command. */
    int    safe_mode_recovery_ticks = 10;

    /*! Tier 3: consecutive damping ticks after which safe mode LATCHES and stops
     *  auto-recovering, so a law that never regains health cannot flap between
     *  damping and tracking indefinitely. Clearing it is deliberate
     *  (TaskSpaceController::clearLatch) rather than automatic. 0 disables
     *  latching. */
    int    tier3_latch_after_ticks = 500;

    // --- Disturbance feedforward ---------------------------------------------
    /*! How much of the disturbance estimate to cancel, 0 to 1. 0 disables the
     *  feedforward entirely and is the default, so an observer can run and be
     *  scored without touching the command.
     *
     *  A fraction rather than a switch because partial compensation is the
     *  useful bring-up setting: at 0.5 a wrong estimate costs half as much, and
     *  the tracking improvement is still visible. Compensating an estimate is
     *  closing a loop around it, and the estimate has its own lag (1/K) and its
     *  own error floor, so full authority is the last step rather than the
     *  first. */
    double ff_gain = 0.0;

    /*! Saturation on the feedforward, as a fraction of each joint's torque
     *  limit. Sized this way for the same reason safe_mode_damping_frac is: an
     *  absolute N*m bound cannot track per-joint limits that differ (10, 10, 5
     *  N*m here).
     *
     *  This is not optional dressing. An estimator transient goes straight to
     *  the actuators, and the estimate is unbounded by construction -- a
     *  momentum residual during a hard contact is whatever the contact was. The
     *  cap is what makes a bad estimate cost tracking accuracy instead of
     *  costing the arm. */
    double ff_max_frac = 0.3;

    /*! Maximum rate of change of the feedforward [N*m/s]; 0 disables the limit.
     *
     *  Separate from the saturation because they catch different failures: the
     *  cap bounds a wrong steady estimate, the slew bounds a fast one. Contact
     *  makes the residual step within a tick or two, and handing that step
     *  straight to a motor is a torque discontinuity the arm feels. */
    double ff_slew = 0.0;

    // --- Reference -----------------------------------------------------------
    /*! When following a reference trajectory (setReferenceWindow), score each
     *  reference node against the reference at THAT node's time,
     *  t_now + k*spacing, so a law with a horizon knows where the target will
     *  be over all of it.
     *
     *  false scores every node against the reference at t_now -- its velocity
     *  and acceleration still fed forward, but no look-ahead. That is a
     *  reactive controller, kept as the baseline that shows what the preview
     *  buys. With a single static setpoint, or a law with a single node, the
     *  two are identical. */
    bool   reference_preview = true;

    /*! Control period [s] — the interval between ticks. Set by whoever runs
     *  the loop (the node's control_hz, a bench's --hz), never by the YAML.
     *  0 means unknown; a law that needs it says what it falls back to. */
    double control_period = 0.0;
};

}  // namespace pat_arm_control
