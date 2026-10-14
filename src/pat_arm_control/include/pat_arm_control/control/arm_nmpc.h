#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <mujoco/mujoco.h>

#include "pat_arm_control/dynamics/floating_base_dynamics.h"
#include "pat_arm_control/solvers/quad_prob_solver.h"

namespace pat_arm_control {

/*! Position/velocity limits for a joint this controller OWNS -- i.e. whose
 *  Δq/qdot rows become box-constrained state. Torque limits are not here:
 *  they apply to every model joint, owned or not, and live in
 *  NMPCParams::model_torque_lims. */
struct JointLimit { std::string name; double q_min, q_max, qd_max; };

struct NMPCParams {    
    // --- Horizon discretization ------------------------------------------
    double Ts = 0.05;   //!< [s] prediction-step interval (NOT the outer sim dt)
    int    N  = 20;     //!< number of horizon steps

    // --- LSQ weighting on EE position/velocity error ----------------------
    double Qx = 10.0, Qy = 10.0, Qz = 10.0;         //!< EE position error weight
    double Qdotx = 0.0, Qdoty = 0.0, Qdotz = 0.0;   //!< EE velocity error weight

    // --- Control-effort regularization -------------------------------------
    double Rx = 1e-3, Ry = 1e-3, Rz = 1e-3;

    // Off by default: when true, activates full nonlinear control with 
    // no linearisation for the optimiser. 
    bool fullNonlinear = false; 

    double Qwx = 10.0, Qwy = 10.0, Qwz = 10.0;         //!< EE orientation error weight
    double Qdotwx = 0.0, Qdotwy = 0.0, Qdotwz = 0.0;   //!< EE angular velocity error weight
    double Rwx = 1e-3, Rwy = 1e-3, Rwz = 1e-3;         //!< task-space torque effort regularization

    // --- Joint limits (required) --------------------------------------------
    // Position/velocity limits for the OWNED joints.
    std::vector<JointLimit> joint_lims;

    /*! Torque limit [N*m] for EVERY joint in the model, as {name, tau_max}.
     *  The torque constraint and the rollout clamp both cover all model joints
     *  -- this controller rolls the whole model forward, including the arm it
     *  does not own -- so a limit is needed for each one, and this is the
     *  single place they are set. Must name every model joint. */
    std::vector<std::pair<std::string, double>> model_torque_lims;

    // --- Name of owned joints ----------------------------------------------
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

    std::string ee_site_name;     //!< name of the MJCF <site> at the end effector (required)

    // --- QP solver tuning ----------------------------------------------------
    int    qp_max_iter = 50;      //!< HPIPM iteration cap ("iter_max" field)
    double qp_tol      = 1e-6;    //!< HPIPM stationarity convergence tolerance ("tol_stat" field)
    int    sqp_iters = 1;

    double qp_reg_prim = 1e-6;
    int    qp_warm_start = 2;

    double joint_limit_slack_linear    = 1e2;  //!< z: cost per unit of Δq/qdot slack
    double joint_limit_slack_quadratic = 1e3;  //!< Z: cost per unit^2 of Δq/qdot slack
    double torque_slack_linear    = 1e2;
    double torque_slack_quadratic = 1e3;

    //! Soft (penalised) vs hard (infeasible-on-violation) running constraints,
    //! and the guard that makes a hard box usable when the measurement is
    //! already outside it. All construction-time. See the matching fields on
    //! QuadProbSolverParams for the full rationale, including why hard is not
    //! automatically safer.
    bool   soft_joint_limits        = true;
    //! False by default: measured better or equal on every column against soft
    //! torque rows, and soft ones collapse on aarch64 near infeasibility. See
    //! nmpc.yaml's soft_torque_limits and the README's "Constraint configuration".
    bool   soft_torque_limits       = false;
    bool   relax_box_to_contain_x0  = false;
    double relax_box_margin         = 1e-3;

    double rollout_v_clamp_mult = 2.0;

    /*! SQP control trust region, as a fraction of the wrench the arm can
     *  actually produce in its current configuration (see
     *  ArmNMPC::reachableWrenchBound). Derived rather than absolute because an
     *  absolute N / N*m bound cannot track either the configuration or the
     *  torque limits: sized for one, it silently stops binding for the other.
     *  1.0 lets the solution swing across the full reachable range in a single
     *  tick — finite and physically meaningful, but loose; lower it to damp the
     *  SQP step harder. */
    double du_max_frac = 1.0;

    bool   terminal_velocity_constraint = false;
    double terminal_cost_multiplier = 1.0;

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
     *  which steps straight from damping to the full QP solution. */
    int    safe_mode_recovery_ticks = 10;

    /*! Tier 1 budget: ticks the controller may keep executing the last ACCEPTED
     *  plan after a failed solve, before giving up and damping. The solver already
     *  produced N stages of control and only the first was applied, so a single
     *  bad solve does not mean there is nothing sensible to do.
     *
     *  Replaying beats holding the last torque: on a free-floating base a held
     *  torque keeps pushing the shared bus with nothing scheduled to undo it,
     *  whereas the plan was computed as a trajectory that arrives with zero
     *  terminal velocity. It beats damping for a TRANSIENT failure, which is what
     *  the measured ones are (one bad solve, then recovery).
     *
     *  Sized for the transient, not for the horizon: the plan goes stale as the
     *  true state diverges from the one it was computed at, so this is tens of
     *  milliseconds, not the full second the horizon covers. 0 disables Tier 1. */
    int    tier1_max_replay_ticks = 5;

    /*! Tier 3: consecutive damping ticks after which safe mode LATCHES and stops
     *  auto-recovering, so a solver that never regains health cannot flap between
     *  damping and tracking indefinitely. Clearing it is deliberate
     *  (ArmNMPC::clearLatch) rather than automatic. 0 disables latching. */
    int    tier3_latch_after_ticks = 500;

    /*! Control period [s] — the interval between ticks, NOT the horizon step Ts.
     *  Used only to map a plan whose stages are Ts apart onto ticks that are this
     *  far apart during Tier 1 replay. 0 means "one stage per tick", which is only
     *  right when Ts happens to equal the period. */
    double control_period = 0.0;

    NMPCParams() = default;
};

/*! Receding-horizon task-space controller (successive-linearization LTV-MPC,
 *  optional full-nonlinear multiple-shooting SQP) with box constraints on
 *  joint position/velocity solved via acados/HPIPM. Port of VORTEX
 *  ArmConstrainedNMPController. */
class ArmNMPC : public FloatingBaseDynamics {
public:
    explicit ArmNMPC(const std::string& mjcf_path, const NMPCParams& params);
    ~ArmNMPC() override;

    /*! EE-site pose in the world frame. quat is [w,x,y,z] (MuJoCo convention). */
    struct EePose {
        Eigen::Vector3d pos;
        Eigen::Vector4d quat;
    };

    /*! One full control tick: loads (q,v) into the internal model + mj_forward,
     *  then evaluates the task-space law. Returns joint torques of length
     *  numArmJoints() in MuJoCo model joint order; torques for joints absent
     *  from NMPCParams.joint_lims come back clamped to 0, so a caller that owns
     *  only a subset extracts its own entries.
     *  q: length 7 + numArmJoints();  v: length 6 + numArmJoints(). */
    Eigen::VectorXd computeControl(const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& v);

    /*! Measured EE-site pose at (q,v). Used by the node to seed the desired
     *  pose on the first tick so the arm holds its start pose until an
     *  external setpoint arrives. */
    EePose currentEePose(const Eigen::VectorXd& q, const Eigen::VectorXd& v);

    /*! The graded response to a solve that did not produce a usable answer.
     *
     *    Nominal    tracking the setpoint from a fresh solve
     *    PlanReplay Tier 1 -- executing the last accepted plan, time-shifted
     *    Damping    Tier 2 -- tau = -kd*qdot, task abandoned, auto-recovers
     *    Latched    Tier 3 -- as Damping but will not auto-recover
     *
     *  The tiers exist because the response used to be binary: one failed solve
     *  dropped straight to damping and safe_mode_recovery_ticks then held it there,
     *  so a single bad solve cost ~100 ms of abandoned tracking at 100 Hz. */
    enum class ControlMode { Nominal, PlanReplay, Damping, Latched };

    /*! Safe-mode joint damping: tau = -kd*qdot on the owned joints, zero
     *  elsewhere, clamped per joint. Deliberately depends on nothing but the
     *  measured velocity and constants fixed at construction — no H_g, no
     *  Lambda_inv, no Cv feedforward — because the usual reason for needing it
     *  is that the model pipeline has just produced something non-finite.
     *  Dissipative by construction: tau·qdot = -qdot^T Kd qdot <= 0, so it
     *  removes energy however wrong the model is. Non-finite velocity entries
     *  yield zero torque on that joint rather than propagating NaN to the
     *  actuators. Public because the ROS node's own exception handler needs it.
     *  v has the same layout as controlLaw's (length 6 + numArmJoints()). */
    Eigen::VectorXd dampingTorque(const Eigen::VectorXd& v) const;

    ControlMode mode() const { return mode_; }
    /*! Ticks spent in safe mode since construction — a health counter for the
     *  node to log, since a brief excursion and a permanent one look the same
     *  from a single transition message. */
    long safeModeTicks() const { return safe_mode_ticks_; }

    //! Tier counters since construction, for the node's health reporting.
    long tier1Ticks()   const { return tier1_ticks_; }
    long tier1Entries() const { return tier1_entries_; }
    bool latched()      const { return mode_ == ControlMode::Latched; }
    /*! Leave Tier 3. Deliberately explicit: the latch means the solver never
     *  recovered on its own, so resuming is an operator decision, not a timeout. */
    void clearLatch();
    /*! HPIPM status from the most recent solve (0 = converged), for the node to
     *  report when it logs a transition into safe mode. */
    int lastQpStatus() const;

    /*! The control period in microseconds, so a tick that overruns it is counted
     *  as it happens. Set once at construction; survives resetProfile(). */
    void setTickBudgetUs(double us) { tick.budget_us = us; }

    /*! Human-readable timing breakdown: per phase, the mean, the max and the
     *  histogram, plus deadline misses against the tick budget and peak
     *  soft-constraint slack. Builds a string, so call it from a diagnostics
     *  timer and never from the control tick.
     *
     *  snapshotProfile + formatProfile are the same report in two halves, for when
     *  the thread that owns the counters is not the one allowed to format them:
     *  the owner copies them out in one trivially copyable value, and whatever
     *  logs formats its own copy. Reading the live members from another thread
     *  instead would be a data race on every field, and resetProfile() racing a
     *  reader would report a half-cleared window. */
    std::string profileReport() const;
    struct ProfileSnapshot {
        PhaseStat tick, dynamics, jdotqdot, lambda, qpsolve;
        PhaseStat rollout_dynamics, rollout_integrate, pinv;
        PhaseStat build_model, defect, trust_region, finalize;
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
    void snapshotProfile(ProfileSnapshot& out) const;
    static std::string formatProfile(const ProfileSnapshot& s);

    /*! Zero every phase statistic and the slack peaks. Call right after reporting:
     *  the statistics are per-interval, not cumulative, because a running average
     *  over a whole session stops responding to the spike that matters. */
    void resetProfile();

    /*! One row per control tick, for plotting the distribution offline.
     *
     *  The report line's histogram is eight fixed bins chosen around a 5 ms
     *  period, which is far too coarse to see the shape of the distribution at
     *  other rates. This keeps every tick instead. Durations are microseconds;
     *  the phase fields are that tick's share, so they sum to roughly `tick_us`
     *  less the unattributed remainder. */
    struct TickTrace {
        double t_us;          //!< since the first recorded tick, the x axis
        float tick_us;        //!< whole controlLawImpl
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

    /*! Start recording, into a buffer allocated once, here. Recording stops when
     *  it is full rather than growing or wrapping: a trace that silently dropped
     *  its first half would misreport a run's start, and a tick must never
     *  allocate. Capacity 0 disables. */
    void enableTrace(std::size_t capacity);
    const std::vector<TickTrace>& trace() const { return trace_; }
    void clearTrace() { trace_.clear(); }
    //! True once the buffer filled and ticks stopped being recorded. With a caller
    //! that drains regularly this should never become true; if it does, the
    //! capacity is smaller than the drain interval's worth of ticks.
    bool traceFull() const { return trace_cap_ > 0 && trace_.size() >= trace_cap_; }

    /*! Move the recorded rows into `sink` and resume recording from empty.
     *
     *  For a caller that writes them out as the run proceeds, so the trace is not
     *  all held in memory and is not all lost if the process is killed -- which is
     *  what happens under `ros2 launch`, whose shutdown escalates to SIGKILL on a
     *  grace period a large write does not finish inside. Call this from the
     *  diagnostics path, never from the control tick. */
    void drainTrace(std::vector<TickTrace>& sink);

protected:
    /*! Thin guard around controlLawImpl: any exception escaping the control
     *  computation becomes safe mode instead of terminating the process. Wrapped
     *  here rather than around computeControl so loadLiveState's size-mismatch
     *  throw stays fatal — that one is a static configuration error, and if it
     *  fires, v cannot be trusted enough to damp against. */
    Eigen::VectorXd controlLaw(const Eigen::VectorXd& q,
                               const Eigen::VectorXd& v) override;

    Eigen::VectorXd controlLawImpl(const Eigen::VectorXd& q,
                                    const Eigen::VectorXd& v);

protected:
    // === Shared helpers used in the controlLaw function above ======
    // Protected rather than private: these are the extension points a future
    // control law reuses (the same role getDynamics()/dynamicsAt() play in the
    // base class), and the model-fidelity tests drive them directly through a
    // subclass rather than inferring A/B from closed-loop behaviour.

    /*! The only two things the prediction model needs H_g^-1 for:
     *    H_g_inv_Jt = H_g^-1 J^T      (n_joints x d)
     *    Lambda_inv = J H_g_inv_Jt    (d x d)
     *  Both come from ONE Cholesky of H_g and a solve against J^T -- H_g is the
     *  Schur complement of the base block in a positive-definite mass matrix, so
     *  it is positive definite (NmpcDynamics asserts it) and no pseudo-inverse is
     *  called for. If the factorisation is rejected the model is degenerate, and
     *  this falls back to the damped SVD inverse and counts it under
     *  spdFallbacks(). NaN-checked.
     *
     *  Note Lambda_inv is J H_g^-1 J^T, which IS singular at a kinematic
     *  singularity -- that is a property of J, not of H_g, and is not what the
     *  Cholesky here can or should guard. */
    void computeInertiaProducts(const Eigen::MatrixXd& H_g, const Eigen::MatrixXd& J,
                                Eigen::MatrixXd& H_g_inv_Jt,
                                Eigen::MatrixXd& Lambda_inv);

    /*! getDynamics() into H_g/Cv_joints, timed under `dynamics`. */
    void computeGeneralizedInertia(Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints);

    /*! Builds the full augmented (task + Δq/qdot) A/B prediction matrices,
     *  sized off J_task's/Lambda_inv's dimension d (task dim == nu).
     *  H_g_inv_Jt = H_g^-1 J_task^T (n_joints_ x d) maps a task wrench to the
     *  joint acceleration it produces, which is what couples the control into
     *  the Δq/qdot rows. */
    void buildAugmentedModel(const Eigen::MatrixXd& Lambda_inv,
                              const Eigen::VectorXd& bias,
                              const Eigen::MatrixXd& H_g_inv_Jt,
                              Eigen::MatrixXd& A, Eigen::MatrixXd& B);

    /*! Per-channel bound on the task wrench the arm can produce at this
     *  configuration: entry j is the largest |u_j| that, acting alone, keeps
     *  every joint torque J^T u within torqueLimsByIndex. Acting alone is the
     *  caveat — the simultaneous feasible set is smaller by up to a factor of d
     *  — which suits a trust region, whose job is to bound a wild step rather
     *  than to replace the torque constraint. Channels the Jacobian cannot
     *  actuate are capped at a finite value, since these feed QP bounds.
     *
     *  The out-parameter form is what the control path calls, so a per-tick
     *  bound costs no allocation; the returning form is kept for the tests and
     *  any caller that just wants the value. */
    Eigen::VectorXd reachableWrenchBound(const Eigen::MatrixXd& J_task) const;
    void reachableWrenchBound(const Eigen::MatrixXd& J_task,
                               Eigen::VectorXd& bound) const;

    /*! Δq/qdot box bounds on the OWNED joints, offset by current measurement q0. */
    void buildJointBoxBounds(const Eigen::VectorXd& q0,
                              Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j);

    /*! Per-joint torque limit as a general (polytopic) constraint lg<=D*u<=ug. */
    void buildTorqueGeneralConstraint(const Eigen::MatrixXd& J,
                                       const Eigen::VectorXd& Cv_joints,
                                       Eigen::MatrixXd& D, Eigen::VectorXd& lg,
                                       Eigen::VectorXd& ug);

    /*! Solves the single-stage QP (timed under t_qpsolve_us_), writing the
     *  stage-0 control into tau_task_out (size = B.cols()) and recording the
     *  soft-constraint slack on success. Returns false if the solver did not
     *  converge or returned a non-finite solution — the caller's job is then to
     *  enter safe mode, not to substitute an approximation. Q_N: terminal-stage
     *  cost, used in place of Q at the terminal weight; callers pass Q itself
     *  when terminal_cost_multiplier == 1.0. */
    // Non-const refs: qp_->solve() takes raw non-const double* into these
    // (HPIPM's C API doesn't promise not to touch its inputs), matching the
    // pre-refactor call sites, which all passed freshly-built locals anyway.
    bool solveStageQP(Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                       Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
                       Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
                       Eigen::MatrixXd& D, Eigen::VectorXd& lg,
                       Eigen::VectorXd& ug, Eigen::VectorXd& x_aug,
                       Eigen::MatrixXd& Q_N,
                       Eigen::VectorXd& tau_task_out);

    /*! tau_joints = J^T*tau_task + Cv_joints, clamped to torqueLimsByIndex. The
     *  out-parameter form is the one the control path calls; see
     *  reachableWrenchBound for why both exist. */
    Eigen::VectorXd finalizeJointTorques(const Eigen::MatrixXd& J,
                                          const Eigen::VectorXd& tau_task,
                                          const Eigen::VectorXd& Cv_joints);
    void finalizeJointTorques(const Eigen::MatrixXd& J,
                               const Eigen::VectorXd& tau_task,
                               const Eigen::VectorXd& Cv_joints,
                               Eigen::VectorXd& tau_joints);


    // === Full-nonlinear (multiple-shooting SQP) path, used instead of the
    // single-linearization-point buildAugmentedModel/solveQPWithFallback
    // pair above when params_.fullNonlinear is true. See
    // armConstrainedNMPController.cpp's solveNonlinearMPC() doc comment.

    /*! Per-node task-space quantities a control law's TaskSpaceEval callback
     *  produces from that node's rolled-out (q,v)/dynamics -- feeds
     *  buildAugmentedModel/buildTorqueGeneralConstraint exactly like the
     *  linearized path's single node-0 computation does today, just called
     *  once per rollout node instead of once per tick. */
    struct NodeTaskSpace {
        Eigen::VectorXd e, edot;    // task-space error state, dim d
        Eigen::MatrixXd J_task;     // d x n_joints_
        Eigen::VectorXd bias;       // dim d (a_d - h')
    };
    //! Writes into a NodeTaskSpace the caller owns rather than returning one:
    //! NodeTaskSpace carries four dynamically-sized Eigen members, so a by-value
    //! return costs four allocations at every rollout node.
    using TaskSpaceEval = std::function<void(
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
        NodeTaskSpace& out)>;

    /*! TaskSpaceEval implementations. Each control law wraps it in a thin
     *  TaskSpaceEval lambda when calling solveNonlinearMPC()
     *  (a lambda is still needed to bind `this` into the std::function, but
     *  the actual per-dimension logic lives here, named and documented).
     *  The out-parameter form is what the rollout uses; the by-value one remains
     *  for the tests and for single-shot callers. */
    void evalTaskSpace(
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
        NodeTaskSpace& out) const;
    NodeTaskSpace evalTaskSpace(
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6) const;

    Eigen::VectorXd solveNonlinearMPC(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                                    int d, const TaskSpaceEval& evalTaskSpace,
                                    Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
                                    Eigen::MatrixXd& Q_N);

    /*! Enter safe mode: latch the mode, restart the recovery count, and drop the
     *  nominal control trajectory, since whatever it holds was either produced
     *  by a failed solve or linearized around one. */
    void enterDamping();

    /*! Fold the last solve's stage-1 slacks into max_box_slack_/max_tau_slack_.
     *  Stage 1 is where both soft row types coexist (box rows first, then
     *  general/torque rows); the split comes from the solver's own
     *  nsbxAt/nsgAt, since which groups are soft is configurable. */
    void recordSlack();

    //! Fold this tick's total into `tick`. Accumulation only — no I/O, so nothing
    //! on the control path can block writing to a pipe. Reporting is the caller's
    //! job, via profileReport() from a lower-rate context.
    void recordTick(std::chrono::steady_clock::time_point start);

    NMPCParams params_;
    int        ee_site_id_ = -1;

    // Per-joint limits, keyed by joint name as given in joint_lims, parsed
    // in the constructor. Values are validated against the loaded MJCF via
    // FloatingBaseDynamics::jointIndexFromName() at construction time.
    std::unordered_map<std::string, std::pair<double, double>> qlims;       //!< [rad] (q-, q+)
    std::unordered_map<std::string, double>                    vlims;       //!< [rad/s] |qdot| limit
    std::unordered_map<std::string, double>                    torque_lims; //!< [N*m] |tau| limit

    // Same data as qlims/vlims/torque_lims, but indexed by arm-local joint
    // index (0..numJoints-1, matching q[6+i]/getJointPosInMsg(i)) rather
    // than name — built once in the constructor via jointIndexFromName() so
    // the per-tick control law can look bounds up by index directly instead
    // of re-hashing joint names every control tick.
    // Two distinct position limits, and the distinction matters:
    //
    //   modelQlimsByIndex -- the MJCF's own joint ranges. MECHANICAL: MuJoCo
    //     enforces these with constraint forces, so the plant cannot leave them.
    //     This is what the rollout clamps to, because the rollout clamp exists to
    //     keep the predicted trajectory inside the physically reachable
    //     configuration space (an unreachable one loses Jacobian rank).
    //   qlimsByIndex -- the above, tightened by the configured limits.q_min/q_max.
    //     A SOFTWARE margin, which only the QP constraint enforces.
    //
    // Clamping the rollout to the software margin instead would satisfy the box
    // by construction: the clamp would teleport the nominal trajectory inside the
    // margin and the multiple-shooting defect would present that teleport to the
    // QP as free dynamics, leaving the constraint with nothing to do.
    std::vector<std::pair<double, double>> modelQlimsByIndex;
    std::vector<std::pair<double, double>> qlimsByIndex;
    std::vector<double>                    vlimsByIndex;
    std::vector<double>                    torqueLimsByIndex;
    std::vector<int>                       ownedJointInds;
    int                                    n_owned = 0;

    //! torqueLimsByIndex as an Eigen vector, for the torque constraint's bound
    //! arithmetic at every rollout node. Built once at construction alongside the
    //! vector it mirrors, since neither changes afterwards.
    Eigen::VectorXd tau_lim_vec_;

    // Safe mode. kd_ is sized n_joints_ and nonzero only on owned indices,
    // fixed at construction from the torque and velocity limits. The mode
    // latches on failure and only clears after safe_mode_recovery_ticks
    // consecutive good solves, so the output cannot chatter between damping and
    // tracking while the solver is marginal.
    std::vector<double> kd_;
    ControlMode         mode_ = ControlMode::Nominal;
    int                 consecutive_ok_ = 0;
    long                safe_mode_ticks_ = 0;
    int                 damping_run_ = 0;      //!< consecutive damping ticks, for Tier 3

    /*! The graded failure response. Returns the torque to apply for a tick whose
     *  solve produced nothing usable, and sets mode_ accordingly.
     *
     *  J_task / Cv are this tick's, at the CURRENT measurement, and may be null
     *  when the failure happened before they were computed (an exception out of
     *  the rollout). Tier 1 needs them: it reuses only the stale task WRENCH and
     *  re-derives the joint torque from fresh kinematics, so the feedforward stays
     *  current and only the plan is old. Without them it falls through to Tier 2. */
    Eigen::VectorXd onSolveFailure(const Eigen::VectorXd& v,
                                   const Eigen::MatrixXd* J_task,
                                   const Eigen::VectorXd* Cv);

    /*! Advance the nominal control trajectory by ONE TICK of real time, which is
     *  control_period and not one horizon stage. Shifting a whole stage per tick
     *  moves the warm start Ts/control_period times too fast; at Ts = 25 ms and
     *  100 Hz that is 2.5x, so every tick would linearise around a trajectory
     *  15 ms of plan ahead of where the arm is going to be. */
    void shiftNominalControl();

    //! Snapshot u_bar_ as the plan Tier 1 may replay. Only after a validated
    //! success, and after the shift, so element 0 is the NEXT tick's control.
    void acceptPlan();

    /*! Which plan stage a replay tick that is `age_ticks` old should apply, or -1
     *  if the plan is exhausted. Stages are Ts apart and ticks are control_period
     *  apart, so a stage is HELD for the ticks it spans -- not an approximation,
     *  since the prediction model itself assumes u constant across Ts. */
    int planStageForAge(int age_ticks) const;

    // The plan Tier 1 replays. Separate from u_bar_ on purpose: u_bar_ is mutated
    // in place by the trust region, is zeroed by enterDamping(), and after a failed
    // solve holds a partially-updated iterate. acados' internal iterate is no good
    // either -- it may be garbage once a solve has failed.
    std::vector<Eigen::VectorXd> u_accepted_;
    bool                         have_accepted_plan_ = false;
    int                          replay_ticks_  = 0;   //!< ticks into the current replay
    long                         tier1_ticks_   = 0;
    long                         tier1_entries_ = 0;

    // Box/torque-constrained QP solver (acados_c + HPIPM backend).
    std::unique_ptr<quad_prob_solver::QuadProbSolver> qp_;

    // Nominal control trajectory for the full-nonlinear (fullNonlinear=true)
    // path -- N entries of size d (task dim), warm-started/shifted tick to
    // tick by solveNonlinearMPC(). Empty/wrong-sized on the first tick (or
    // after a dimension change), in which case solveNonlinearMPC()
    // reinitializes it to zero. Unused when fullNonlinear is false.
    std::vector<Eigen::VectorXd> u_bar_;

    // Timing breakdown. `tick` is the whole control law; the rest are phases
    // within it, so `tick.sum_us` minus the phases is unattributed remainder --
    // keep that small, or the profile cannot point at the cost.
    //
    // Which phases are populated depends on the path. dynamics/jdotqdot belong to
    // the LINEARIZED path only (fullNonlinear=false); the SQP path evaluates
    // dynamics once per rollout node through dynamicsAt(), which bundles the same
    // work into one mj_forward pass, so it reports under rollout_dynamics instead.
    // A phase with count==0 is simply not on the active path.
    PhaseStat tick;               //!< whole controlLawImpl
    PhaseStat dynamics;           //!< linearized path: getDynamics + H_g inverse
    PhaseStat jdotqdot;           //!< linearized path: jdotQdot6
    PhaseStat lambda;             //!< Lambda_inv product
    PhaseStat qpsolve;            //!< solve() / solveMultiStage()
    PhaseStat rollout_dynamics;   //!< SQP path: dynamicsAt per node
    PhaseStat rollout_integrate;  //!< SQP path: integrateStep per node
    PhaseStat pinv;               //!< the per-node H_g Cholesky + solve for H_g^-1 J^T
    PhaseStat build_model;        //!< buildAugmentedModel per node
    PhaseStat defect;             //!< the b_k defect loop
    PhaseStat trust_region;       //!< du_max clamp loop + reachableWrenchBound
    PhaseStat finalize;           //!< finalizeJointTorques

    //! Interior-point iterations per solve. HPIPM's solve time is very nearly
    //! linear in this, so it is what distinguishes "the arithmetic is slow" from
    //! "this QP was hard" -- and only the second is fixable by a cap.
    Tally qp_iters;
    //! HPIPM's own reported time for the same solve. Paired with qpsolve's measured
    //! duration it localises a slow solve: close together puts the cost inside
    //! ocp_qp_solve, far apart puts it in the per-stage data marshalling around it.
    Tally qp_time_ms;
    long  qp_retries = 0;         //!< solves that needed the cold-start retry
    void  recordQpIters();

    //! Times the H_g Cholesky was rejected and the damped SVD stood in. Expected
    //! to stay 0: a non-zero count means the model's inertia is degenerate, not
    //! that the arm reached an awkward pose.
    long  spd_fallbacks_ = 0;

    //! The rollout's per-node dynamics, held rather than created per node. Only
    //! the current node's is ever live, so one instance serves all N+1.
    NodeDynamics nd_scratch_;
    NodeTaskSpace ts_scratch_;   //!< likewise, for the per-node task-space terms

    //! The multi-stage QP's per-stage data, held across ticks. Sized once for the
    //! horizon; each entry keeps its own buffer, since assigning a same-sized
    //! Eigen object reuses the storage it already has.
    std::vector<Eigen::MatrixXd> A_k_, B_k_, D_k_;
    std::vector<Eigen::VectorXd> b_k_, lg_k_, ug_k_, x_aug_nodes_;

    //! The rollout loop's remaining per-node destinations. Locals here cost an
    //! allocation each at every node; the model and the stage data are written
    //! straight into their A_k_/B_k_ slots rather than through a copy. q_k_/v_k_
    //! carry the rollout's running state across the nodes.
    Eigen::MatrixXd hg_inv_Jt_, lambda_inv_;
    Eigen::VectorXd tau_joints_k_, q_next_, v_next_, q_k_, v_k_;

    //! And the once-per-tick ones: the measurement the box bounds are offset by,
    //! those bounds, node 0's kinematics and the resulting task wrench, plus the
    //! reachable wrench and the trust-region width derived from it.
    Eigen::VectorXd q0_, lbx_j_, ubx_j_, Cv_joints0_, tau_task_result_;
    Eigen::VectorXd wrench_bound_, du_max_;
    Eigen::MatrixXd J_task0_;
    //! Stage and terminal cost. Built from weights fixed at construction, so
    //! rebuilding them per tick allocated for a result that never changes.
    Eigen::MatrixXd Q_, R_, Q_N_;

    //! computeInertiaProducts' workspace: H_g^2 + d^2 I, H_g J^T, and the Cholesky
    //! of the former. Members for the same reason -- once per node, N a tick.
    Eigen::MatrixXd G_, HgJt_;
    Eigen::LLT<Eigen::MatrixXd> hg_llt_;

    //! Stage-control scratch for the SQP path's per-stage reads out of the solver.
    //! A member rather than a local because both loops that use it run N times a
    //! tick, and a fresh VectorXd each pass is N allocations for a 3-vector.
    Eigen::VectorXd uk_;

    // --- Per-tick trace ----------------------------------------------------
    // A tick's phase cost is the GROWTH of each PhaseStat's running sum across
    // it, so no second set of accumulators is needed -- just the previous tick's
    // sums. resetProfile() zeroes these alongside the PhaseStats, which keeps the
    // first delta after a report window correct rather than counting the window's
    // whole total into one tick.
    std::vector<TickTrace> trace_;
    std::size_t            trace_cap_ = 0;
    bool                   trace_t0_set_ = false;
    std::chrono::steady_clock::time_point trace_t0_{};
    struct PhaseSums {
        double dynamics = 0, jdotqdot = 0, lambda = 0, qpsolve = 0;
        double rollout_dynamics = 0, rollout_integrate = 0, pinv = 0;
        double build_model = 0, defect = 0, trust_region = 0, finalize = 0;
        double hpipm_ms = 0;
    } prev_sums_;
    //! Re-baseline the per-tick deltas on the sums as they stand right now.
    void snapshotPhaseSums();

    // Worst soft-constraint slack since the last profile report -- box (joint
    // position/velocity) and general (torque) rows kept apart, since they carry
    // separate penalty weights. Reported alongside the timing line rather than
    // logged per tick: straining is a sustained condition, not a per-tick
    // event, and the loop cannot afford a stderr write at every control period.
    // slack_lo_/slack_hi_ are getSlack()'s destinations, sized once so
    // recordSlack() does not allocate on the control path.
    double max_box_slack_ = 0.0, max_tau_slack_ = 0.0;
    Eigen::VectorXd slack_lo_, slack_hi_;
};

}  // namespace pat_arm_control
