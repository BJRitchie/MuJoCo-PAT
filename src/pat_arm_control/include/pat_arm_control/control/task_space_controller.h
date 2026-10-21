#pragma once
/*! Everything a task-space arm control law shares, so a law is only its law.
 *
 *  A law derives from this and implements controlLawImpl(). This supplies:
 *
 *    - the joints it owns and every limit, from the MJCF and ControllerParams;
 *    - the reference: the static setpoint, or a sampled window evaluated at
 *      each of the law's reference nodes (one for a one-step law, N+1 for a
 *      horizon);
 *    - the disturbance feedforward, conditioned (gain, cap, slew) and folded
 *      into the joint-torque mapping every law ends with;
 *    - safe mode: Tier 2 damping, the Tier 3 latch, the recovery hysteresis,
 *      and the guard that turns any exception out of the law into damping.
 *      Tier 1 (replaying a plan) belongs to a law that keeps one.
 *
 *  Allocation-free on the control path, like the laws built on it. Zero ROS.
 */
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "pat_arm_control/control/controller_params.h"
#include "pat_arm_control/control/i_arm_controller.h"
#include "pat_arm_control/dynamics/floating_base_dynamics.h"
#include "pat_arm_control/reference/reference_window.h"

namespace pat_arm_control {

class TaskSpaceController : public FloatingBaseDynamics, public IArmController {
public:
    /*! Loads the model and resolves every joint and limit. Throws
     *  std::runtime_error on a missing site, joint or torque limit, or a
     *  safe-mode or feedforward parameter out of range. */
    TaskSpaceController(const std::string& mjcf_path, const ControllerParams& params);
    ~TaskSpaceController() override = default;

    // --- IArmController: the model and one tick ------------------------------
    int numArmJoints() const override { return FloatingBaseDynamics::numArmJoints(); }
    void fullDynamics(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                      Workspace& ws, Eigen::MatrixXd& M, Eigen::VectorXd& bias,
                      Eigen::MatrixXd& H_bb_inv_H_bm) const override {
        FloatingBaseDynamics::fullDynamics(q, v, ws, M, bias, H_bb_inv_H_bm);
    }

    /*! One full control tick: loads (q,v) into the internal model + mj_forward,
     *  then runs the law. */
    Eigen::VectorXd computeControl(const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& v) override;

    /*! Measured EE-site pose at (q,v). Used by the node to seed the desired
     *  pose on the first tick so the arm holds its start pose until an
     *  external setpoint arrives. */
    EePose currentEePose(const Eigen::VectorXd& q, const Eigen::VectorXd& v) override;

    /*! Safe-mode joint damping: tau = -kd*qdot on the owned joints, zero
     *  elsewhere, clamped per joint. Deliberately depends on nothing but the
     *  measured velocity and constants fixed at construction — no H_g, no
     *  Lambda_inv, no Cv feedforward — because the usual reason for needing it
     *  is that the model pipeline has just produced something non-finite.
     *  Dissipative by construction: tau·qdot = -qdot^T Kd qdot <= 0, so it
     *  removes energy however wrong the model is. Non-finite velocity entries
     *  yield zero torque on that joint rather than propagating NaN to the
     *  actuators. v has the same layout as computeControl's (length 6 +
     *  numArmJoints()). */
    Eigen::VectorXd dampingTorque(const Eigen::VectorXd& v) const override;

    // --- IArmController: the reference ---------------------------------------
    void setDesiredPos(double x, double y, double z) override {
        desiredPos[0] = x; desiredPos[1] = y; desiredPos[2] = z; }
    /*! Desired EE orientation setpoint, as a unit quaternion [w, x, y, z]
     *  (MuJoCo convention). */
    void setDesiredOrient(double w, double x, double y, double z) override {
        desiredQuat[0] = w; desiredQuat[1] = x; desiredQuat[2] = y; desiredQuat[3] = z; }
    void setDesiredVel(double x, double y, double z) {
        desiredVel[0] = x; desiredVel[1] = y; desiredVel[2] = z; }
    void setDesiredAcc(double x, double y, double z) {
        desiredAcc[0] = x; desiredAcc[1] = y; desiredAcc[2] = z; }
    /*! Desired EE angular velocity setpoint, world frame [rad/s]. */
    void setDesiredAngVel(double x, double y, double z) {
        desiredAngVel[0] = x; desiredAngVel[1] = y; desiredAngVel[2] = z; }
    /*! Desired EE angular acceleration setpoint, world frame [rad/s^2]. */
    void setDesiredAngAcc(double x, double y, double z) {
        desiredAngAcc[0] = x; desiredAngAcc[1] = y; desiredAngAcc[2] = z; }

    /*! Follow a sampled reference trajectory instead of the static setpoint.
     *
     *  The window is copied in, so the caller's may change afterwards. Each
     *  tick then samples it at t_now + k*spacing for every reference node k
     *  (see ControllerParams::reference_preview), with t_now from
     *  setReferenceTime. Past the window's end it holds the last position at
     *  rest, and counts those samples (referenceSamplesOutside).
     *  Allocation-free: the window and the per-node references are storage
     *  sized at construction. */
    void setReferenceWindow(const ReferenceWindow& window) override;
    void clearReferenceWindow() noexcept override { have_window_ = false; }
    bool referenceActive() const noexcept { return have_window_; }

    /*! The time [s] the next computeControl's state was measured at, on the
     *  same clock as the window's sample times. The reference is evaluated
     *  there, so a late tick still compares the state with the reference for
     *  the instant it describes. */
    void setReferenceTime(double t_now) noexcept override { t_ref_ = t_now; }

    //! Node references that fell outside the window and were held, and the
    //! ticks that used a window at all. Health counters for the node's report:
    //! a window that keeps running short means the planner is not keeping up.
    long referenceSamplesOutside() const noexcept override { return ref_outside_; }
    long referenceTicks() const noexcept override { return ref_ticks_; }

    // --- IArmController: disturbance feedforward ------------------------------
    /*! Hand the controller the latest disturbance estimate, reduced to joint
     *  space, model-joint order and length numArmJoints().
     *
     *  Called from the control thread once per tick, BEFORE computeControl, so
     *  the command and any torque bounds the law uses take the same value.
     *  Applies ff_gain, then the ff_max_frac saturation, then the ff_slew rate
     *  limit, and keeps the result until the next call -- so a tick that gets no
     *  fresh estimate reuses the last one rather than dropping to zero, which
     *  would itself be a torque step.
     *
     *  `dt` is the control period [s], needed to turn ff_slew's N*m/s into a
     *  per-tick delta. Passed in rather than taken from the parameters because
     *  a law's prediction step is not the control period and conflating them
     *  would make the slew limit silently wrong.
     *
     *  With ff_gain at 0 this records the estimate and changes nothing, so the
     *  observer can be scored against the same run it would act on.
     *  Allocation-free. */
    void setDisturbanceFeedforward(const Eigen::VectorXd& d_joints, double dt) override;

    /*! What the feedforward is actually applying, after gain, saturation and
     *  slew -- not the raw estimate. Exposed so the node can log the value that
     *  reached the command rather than the one that was offered. */
    const Eigen::VectorXd& disturbanceFeedforward() const noexcept { return d_ff_; }

    /*! Ticks on which the saturation or the slew limit bound the feedforward.
     *  If either is non-zero in steady state the estimate is being clipped, and
     *  the compensation is not what the gain says it is. */
    long ffSaturated() const noexcept { return ff_saturated_; }
    long ffSlewLimited() const noexcept { return ff_slew_limited_; }

    // --- IArmController: health -----------------------------------------------
    ControlMode mode() const override { return mode_; }
    /*! Ticks spent in safe mode since construction — a health counter for the
     *  node to log, since a brief excursion and a permanent one look the same
     *  from a single transition message. */
    long safeModeTicks() const override { return safe_mode_ticks_; }
    bool latched() const { return mode_ == ControlMode::Latched; }
    /*! Leave Tier 3. Deliberately explicit: the latch means the law never
     *  recovered on its own, so resuming is an operator decision, not a timeout. */
    void clearLatch();

    //! A law with no plan to replay has no Tier 1, and no solver to report on.
    long tier1Ticks() const override { return 0; }
    long tier1Entries() const override { return 0; }
    int lastQpStatus() const override { return 0; }

    /*! Human-readable timing breakdown of a snapshot: per phase, the mean, the
     *  max and the histogram, plus deadline misses against the tick budget and
     *  peak soft-constraint slack. Builds a string, so call it from a
     *  diagnostics context and never from the control tick. */
    static std::string formatProfile(const ProfileSnapshot& s);

protected:
    /*! Thin guard around controlLawImpl: any exception escaping the control
     *  computation becomes safe mode instead of terminating the process. Wrapped
     *  here rather than around computeControl so loadLiveState's size-mismatch
     *  throw stays fatal — that one is a static configuration error, and if it
     *  fires, v cannot be trusted enough to damp against. */
    Eigen::VectorXd controlLaw(const Eigen::VectorXd& q,
                               const Eigen::VectorXd& v) final;

    //! The law. Live state already loaded; returns joint torques.
    virtual Eigen::VectorXd controlLawImpl(const Eigen::VectorXd& q,
                                           const Eigen::VectorXd& v) = 0;

    // --- Reference nodes ------------------------------------------------------
    /*! The reference one node is scored against: target EE position, velocity
     *  and acceleration (planar), orientation and its rates. Filled once per
     *  tick for every node by sampleNodeReferences(), either from the reference
     *  window at that node's time or, with no window, from the static desired*
     *  setpoint -- identically at every node, which is what makes the setpoint
     *  path unchanged by the trajectory machinery. */
    struct NodeRef {
        double p[2] = {0.0, 0.0};
        double v[2] = {0.0, 0.0};
        double a[2] = {0.0, 0.0};
        double quat[4] = {1.0, 0.0, 0.0, 0.0};   //!< [w, x, y, z]
        double w = 0.0, aw = 0.0;                //!< angular rate / acceleration about Z
    };

    /*! How many reference nodes the law scores, and how far apart in time [s].
     *  Called once from the law's constructor: 1 for a one-step law, N+1 for a
     *  horizon of N steps Ts apart. */
    void setReferenceNodes(int n_nodes, double spacing);

    //! The static setpoint as a NodeRef: the desired* members, verbatim.
    NodeRef staticReference() const;
    //! Fill node_ref_ for this tick. Allocation-free.
    void sampleNodeReferences();

    // --- Planar task space ----------------------------------------------------
    /*! The three task rows [x, y, thetaz] out of the 6-row generalised
     *  Jacobian (or any 6-row EE quantity stacked [lin; ang]): rows 0, 1 and
     *  the angular-Z row 5. `J3` is resized to 3 x J6.cols(), which is free
     *  when it already has that shape. */
    static void planarRows(const Eigen::MatrixXd& J6, Eigen::MatrixXd& J3);

    /*! World-frame angle-axis (log-map) orientation error of the EE about Z, the
     *  one component the planar task has: the reference orientation relative to
     *  `ee_quat` ([w, x, y, z]), rotated into the world frame. */
    static double orientationErrorZ(const double ref_quat[4], const Eigen::Vector4d& ee_quat);

    // --- Torque mapping -------------------------------------------------------
    /*! tau_joints = J^T*tau_task + Cv_joints - d_ff, clamped to
     *  torqueLimsByIndex. The out-parameter form is the one the control path
     *  calls, so it does not allocate; the returning form is for tests and
     *  one-off callers. */
    Eigen::VectorXd finalizeJointTorques(const Eigen::MatrixXd& J,
                                         const Eigen::VectorXd& tau_task,
                                         const Eigen::VectorXd& Cv_joints);
    void finalizeJointTorques(const Eigen::MatrixXd& J,
                              const Eigen::VectorXd& tau_task,
                              const Eigen::VectorXd& Cv_joints,
                              Eigen::VectorXd& tau_joints);

    // --- Safe mode ------------------------------------------------------------
    /*! Enter safe mode: latch the mode, restart the recovery count, drop the
     *  feedforward, and let the law drop whatever it carries between ticks
     *  (onEnterDamping), since that was either produced by a failed tick or
     *  computed around one. */
    void enterDamping();
    //! The law's part of enterDamping(); nothing by default.
    virtual void onEnterDamping() {}

    /*! Tiers 2 and 3 for a tick that produced no usable command: damping, and
     *  the latch once damping has gone on long enough to count as not
     *  recovering. Returns the torque to apply. */
    Eigen::VectorXd failToDamping(const Eigen::VectorXd& v);

    /*! After a GOOD tick: true while safe mode must still hold (the latch, or
     *  fewer than safe_mode_recovery_ticks good ticks in a row), in which case
     *  the caller applies dampingTorque instead of its command. Otherwise
     *  leaves safe mode if it was in it, and returns false. */
    bool holdSafeMode();

    // --- Configuration and limits ---------------------------------------------
    ControllerParams ctrl_params_;
    int ee_site_id_ = -1;

    // Per-joint limits, keyed by joint name as given in joint_lims, parsed
    // in the constructor. Values are validated against the loaded MJCF via
    // FloatingBaseDynamics::jointIndexFromName() at construction time.
    std::unordered_map<std::string, std::pair<double, double>> qlims;       //!< [rad] (q-, q+)
    std::unordered_map<std::string, double>                    vlims;       //!< [rad/s] |qdot| limit
    std::unordered_map<std::string, double>                    torque_lims; //!< [N*m] |tau| limit

    // Same data as qlims/vlims/torque_lims, but indexed by arm-local joint
    // index (0..n_joints_-1, matching q[7+i]/v[6+i]) rather than name — built
    // once in the constructor via jointIndexFromName() so the per-tick control
    // law can look bounds up by index directly instead of re-hashing joint
    // names every control tick.
    // Two distinct position limits, and the distinction matters:
    //
    //   modelQlimsByIndex -- the MJCF's own joint ranges. MECHANICAL: MuJoCo
    //     enforces these with constraint forces, so the plant cannot leave them.
    //     This is what a rollout clamps to, because the rollout clamp exists to
    //     keep the predicted trajectory inside the physically reachable
    //     configuration space (an unreachable one loses Jacobian rank).
    //   qlimsByIndex -- the above, tightened by the configured limits.q_min/q_max.
    //     A SOFTWARE margin, which only a constraint enforces.
    //
    // Clamping a rollout to the software margin instead would satisfy the box
    // by construction: the clamp would teleport the nominal trajectory inside the
    // margin and the multiple-shooting defect would present that teleport to the
    // QP as free dynamics, leaving the constraint with nothing to do.
    std::vector<std::pair<double, double>> modelQlimsByIndex;
    std::vector<std::pair<double, double>> qlimsByIndex;
    std::vector<double>                    vlimsByIndex;
    std::vector<double>                    torqueLimsByIndex;
    std::vector<int>                       ownedJointInds;
    int                                    n_owned = 0;

    //! torqueLimsByIndex as an Eigen vector, for torque-bound arithmetic at
    //! every node. Built once at construction alongside the vector it mirrors,
    //! since neither changes afterwards.
    Eigen::VectorXd tau_lim_vec_;

    //! The disturbance feedforward in force right now: gain, saturation and
    //! slew already applied. Zero unless setDisturbanceFeedforward has been
    //! called with a non-zero ff_gain, and reset to zero on safe-mode entry so
    //! recovery ramps up from nothing rather than resuming a stale value.
    Eigen::VectorXd d_ff_;
    //! Per-joint saturation bound, ff_max_frac * torqueLimsByIndex, built once.
    Eigen::VectorXd ff_max_vec_;
    long            ff_saturated_ = 0;
    long            ff_slew_limited_ = 0;

    // --- The static setpoint --------------------------------------------------
    // Planar laws read x, y and yaw; the remaining components are kept so a
    // setpoint is a full pose and rate.
    double desiredPos[3] = {0.0, 0.0, 0.0};  //!< [m]    EE position setpoint
    double desiredVel[3] = {0.0, 0.0, 0.0};  //!< [m/s]  EE velocity setpoint
    double desiredAcc[3] = {0.0, 0.0, 0.0};  //!< [m/s²] EE acceleration setpoint
    /*! Desired EE orientation. Defaults to the identity quaternion; the node
     *  seeds it from the arm's own start orientation before tracking. */
    double desiredQuat[4]   = {1.0, 0.0, 0.0, 0.0};  //!< [w,x,y,z] EE orientation setpoint
    double desiredAngVel[3] = {0.0, 0.0, 0.0};       //!< [rad/s]   EE angular velocity setpoint
    double desiredAngAcc[3] = {0.0, 0.0, 0.0};       //!< [rad/s²]  EE angular acceleration setpoint

    // --- Reference trajectory -------------------------------------------------
    // The window is ~20 KB of fixed storage, held here so setReferenceWindow
    // copies into memory that already exists; node_ref_ is sized by
    // setReferenceNodes at construction. Neither allocates once the controller
    // is built.
    ReferenceWindow      window_;
    bool                 have_window_ = false;
    double               t_ref_ = 0.0;
    double               ref_spacing_ = 0.0;   //!< [s] between reference nodes
    std::vector<NodeRef> node_ref_;
    long                 ref_outside_ = 0;
    long                 ref_ticks_ = 0;

    // --- Safe mode ------------------------------------------------------------
    // kd_ is sized n_joints_ and nonzero only on owned indices, fixed at
    // construction from the torque and velocity limits. The mode latches on
    // failure and only clears after safe_mode_recovery_ticks consecutive good
    // ticks, so the output cannot chatter between damping and tracking while
    // the law is marginal.
    std::vector<double> kd_;
    ControlMode         mode_ = ControlMode::Nominal;
    int                 consecutive_ok_ = 0;
    long                safe_mode_ticks_ = 0;
    int                 damping_run_ = 0;      //!< consecutive damping ticks, for Tier 3
};

}  // namespace pat_arm_control
