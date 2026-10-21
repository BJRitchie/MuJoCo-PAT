#pragma once
/*! What a node, a bench or a test needs from an arm controller, whatever its
 *  control law: one tick of control, the measured end-effector pose, the
 *  reference to track, the disturbance feedforward, health counters, and
 *  timing diagnostics.
 *
 *  Every law implements it through TaskSpaceController, which supplies
 *  everything here that is not the law itself. Zero ROS.
 *
 *  The types shared by every law (ControlMode, EePose, ProfileSnapshot,
 *  TickTrace) are nested here, so `ArmNMPC::ControlMode` and the like keep
 *  naming them through inheritance.
 */
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "pat_arm_control/common/profiling.h"
#include "pat_arm_control/dynamics/floating_base_dynamics.h"
#include "pat_arm_control/reference/reference_window.h"

namespace pat_arm_control {

class IArmController {
public:
    virtual ~IArmController() = default;

    /*! EE-site pose in the world frame. quat is [w,x,y,z] (MuJoCo convention). */
    struct EePose {
        Eigen::Vector3d pos;
        Eigen::Vector4d quat;
    };

    /*! The graded response to a tick that did not produce a usable command.
     *
     *    Nominal    tracking the reference from a fresh solve
     *    PlanReplay Tier 1 -- executing the last accepted plan, time-shifted
     *               (only a law that keeps a plan has this tier)
     *    Damping    Tier 2 -- tau = -kd*qdot, task abandoned, auto-recovers
     *    Latched    Tier 3 -- as Damping but will not auto-recover */
    enum class ControlMode { Nominal, PlanReplay, Damping, Latched };

    /*! The timing breakdown, copied out in one trivially copyable value so the
     *  thread that owns the counters is not the one formatting them. A law
     *  fills the phases it has; the rest stay at count 0 and are not reported.
     *  See TaskSpaceController::formatProfile. */
    struct ProfileSnapshot {
        PhaseStat tick, dynamics, jdotqdot, lambda, qpsolve;
        PhaseStat rollout_dynamics, rollout_integrate, pinv;
        PhaseStat build_model, defect, trust_region, finalize;
        PhaseStat law;   //!< a closed-form law's own computation (OneStepController)
        Tally     qp_iters, qp_time_ms;
        long      qp_retries    = 0;
        long      spd_fallbacks = 0;
        long      trace_size    = 0;
        long      trace_cap     = 0;
        bool      trace_full    = false;
        double    max_box_slack = 0.0;
        double    max_tau_slack = 0.0;
        int       qp_max_iter   = 0;
    };

    /*! One row per control tick, for plotting the distribution offline.
     *
     *  The report line's histogram is eight fixed bins chosen around a 5 ms
     *  period, which is far too coarse to see the shape of the distribution at
     *  other rates. This keeps every tick instead. Durations are microseconds;
     *  the phase fields are that tick's share, so they sum to roughly `tick_us`
     *  less the unattributed remainder. A law leaves the phases it does not
     *  have at zero. */
    struct TickTrace {
        double t_us;          //!< since the first recorded tick, the x axis
        float tick_us;        //!< the whole control law
        float dynamics;       //!< the linearized path only
        float jdotqdot;       //!< the linearized path only
        float lambda;
        float qpsolve;
        float rollout_dynamics;
        float rollout_integrate;
        float pinv;
        float build_model;
        float defect;
        float trust_region;
        float finalize;
        float hpipm_us;       //!< HPIPM's own reported time, summed over the tick
        int32_t qp_iters;     //!< summed over the tick's solves; -1 if unreported
        int16_t qp_status;    //!< 0 converged, 1 max iter, 2 min step, 3 NaN
        int32_t qp_iters1;    //!< the FIRST solve alone, before any retry
        int16_t qp_status1;   //!< the first solve's status
        int8_t mode;          //!< ControlMode: 0 nominal, 1 replay, 2 damping, 3 latched
        int8_t retried;       //!< 1 if the cold-start retry ran
    };

    // --- The model --------------------------------------------------------- //

    /*! Arm DOF of the loaded model. The q/v the caller feeds must be sized
     *  7 + this / 6 + this. */
    virtual int numArmJoints() const = 0;

    /*! The full-system dynamics at an arbitrary (q, v), for a disturbance
     *  observer on another thread. See FloatingBaseDynamics::fullDynamics. */
    virtual void fullDynamics(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                              FloatingBaseDynamics::Workspace& ws, Eigen::MatrixXd& M,
                              Eigen::VectorXd& bias, Eigen::MatrixXd& H_bb_inv_H_bm) const = 0;

    // --- One tick ---------------------------------------------------------- //

    /*! One full control tick at (q, v). Returns joint torques of length
     *  numArmJoints() in MuJoCo model joint order; joints this controller does
     *  not own come back 0, so a caller that owns a subset extracts its own.
     *  q: length 7 + numArmJoints();  v: length 6 + numArmJoints(). */
    virtual Eigen::VectorXd computeControl(const Eigen::VectorXd& q,
                                           const Eigen::VectorXd& v) = 0;

    //! Measured EE-site pose at (q, v).
    virtual EePose currentEePose(const Eigen::VectorXd& q, const Eigen::VectorXd& v) = 0;

    /*! Safe-mode joint damping at velocity v: dissipative, finite and inside
     *  the torque limits whatever v holds. Public because a node's own
     *  exception handler needs it. */
    virtual Eigen::VectorXd dampingTorque(const Eigen::VectorXd& v) const = 0;

    // --- The reference ----------------------------------------------------- //

    //! The static setpoint, world frame. Z is unused by the planar laws.
    virtual void setDesiredPos(double x, double y, double z) = 0;
    //! Setpoint orientation, unit quaternion [w, x, y, z].
    virtual void setDesiredOrient(double w, double x, double y, double z) = 0;

    //! Follow a sampled reference trajectory instead of the static setpoint.
    virtual void setReferenceWindow(const ReferenceWindow& window) = 0;
    //! Back to the static setpoint.
    virtual void clearReferenceWindow() = 0;
    //! The time [s] the next computeControl's state was measured at, on the
    //! window's clock.
    virtual void setReferenceTime(double t_now) = 0;
    //! Reference samples that fell outside the window, and ticks that used one.
    virtual long referenceSamplesOutside() const = 0;
    virtual long referenceTicks() const = 0;

    // --- Disturbance feedforward ------------------------------------------- //

    //! The latest disturbance estimate in joint space; see
    //! TaskSpaceController::setDisturbanceFeedforward.
    virtual void setDisturbanceFeedforward(const Eigen::VectorXd& d_joints, double dt) = 0;

    // --- Health ------------------------------------------------------------ //

    virtual ControlMode mode() const = 0;
    virtual long safeModeTicks() const = 0;
    virtual long tier1Ticks() const = 0;
    virtual long tier1Entries() const = 0;
    //! The solver's status from the last tick (0 = converged); 0 for a law
    //! with no solver.
    virtual int lastQpStatus() const = 0;
    //! Whether the law solves an optimisation each tick, which decides how the
    //! node words a failure: "solve failed (QP status n)" or "tick failed".
    virtual bool hasSolver() const { return false; }

    // --- Diagnostics ------------------------------------------------------- //

    //! The control period in microseconds, so a tick that overruns it is counted.
    virtual void setTickBudgetUs(double us) = 0;
    virtual void snapshotProfile(ProfileSnapshot& out) const = 0;
    //! Zero the per-interval statistics; call right after reporting.
    virtual void resetProfile() = 0;
    //! Record every tick into a buffer allocated here, once.
    virtual void enableTrace(std::size_t capacity) = 0;
    //! Move the recorded rows into `sink` and resume recording from empty.
    virtual void drainTrace(std::vector<TickTrace>& sink) = 0;
};

}  // namespace pat_arm_control
