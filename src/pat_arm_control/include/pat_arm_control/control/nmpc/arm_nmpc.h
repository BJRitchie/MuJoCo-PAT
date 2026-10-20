#pragma once

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "pat_arm_control/common/profiling.h"
#include "pat_arm_control/control/controller_params.h"
#include "pat_arm_control/control/nmpc/quad_prob_solver.h"
#include "pat_arm_control/control/task_space_controller.h"

namespace pat_arm_control {

/*! The NMPC's own parameters, on top of the ones every law shares. */
struct NMPCParams : ControllerParams {
    // --- Horizon discretization ------------------------------------------
    double Ts = 0.05;   //!< [s] prediction-step interval (NOT the outer sim dt)
    int    N  = 20;     //!< number of horizon steps

    // --- LSQ weighting, [x, y, theta_z] -----------------------------------
    std::array<double, 3> Q_pos = {10.0, 10.0, 10.0};   //!< EE pose error weight
    std::array<double, 3> Q_vel = {0.0, 0.0, 0.0};      //!< EE velocity error weight
    std::array<double, 3> R     = {1e-3, 1e-3, 1e-3};   //!< task-space effort regularization

    // Off by default: when true, activates full nonlinear control with
    // no linearisation for the optimiser.
    bool fullNonlinear = false;

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
     *  milliseconds, not the full second the horizon covers. 0 disables Tier 1.
     *
     *  Uses ControllerParams::control_period to map a plan whose stages are Ts
     *  apart onto ticks that are that far apart; 0 there means "one stage per
     *  tick", which is only right when Ts happens to equal the period. */
    int    tier1_max_replay_ticks = 5;

    NMPCParams() = default;
};

/*! Receding-horizon task-space controller (successive-linearization LTV-MPC,
 *  optional full-nonlinear multiple-shooting SQP) with box constraints on
 *  joint position/velocity solved via acados/HPIPM. Port of VORTEX
 *  ArmConstrainedNMPController.
 *
 *  Everything that is not the NMPC itself -- limits, reference, feedforward,
 *  Tier 2/3 safe mode -- is TaskSpaceController's. This adds the QP, the SQP
 *  rollout, Tier 1 plan replay, and the per-phase profile and trace. */
class ArmNMPC : public TaskSpaceController {
public:
    explicit ArmNMPC(const std::string& mjcf_path, const NMPCParams& params);
    ~ArmNMPC() override;

    //! Tier 1 counters since construction, for the node's health reporting.
    long tier1Ticks()   const override { return tier1_ticks_; }
    long tier1Entries() const override { return tier1_entries_; }
    /*! HPIPM status from the most recent solve (0 = converged), for the node to
     *  report when it logs a transition into safe mode. */
    int lastQpStatus() const override;

    /*! The control period in microseconds, so a tick that overruns it is counted
     *  as it happens. Set once at construction; survives resetProfile(). */
    void setTickBudgetUs(double us) override { tick.budget_us = us; }

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
    void snapshotProfile(ProfileSnapshot& out) const override;

    /*! Zero every phase statistic and the slack peaks. Call right after reporting:
     *  the statistics are per-interval, not cumulative, because a running average
     *  over a whole session stops responding to the spike that matters. */
    void resetProfile() override;

    /*! Start recording, into a buffer allocated once, here. Recording stops when
     *  it is full rather than growing or wrapping: a trace that silently dropped
     *  its first half would misreport a run's start, and a tick must never
     *  allocate. Capacity 0 disables. */
    void enableTrace(std::size_t capacity) override;
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
    void drainTrace(std::vector<TickTrace>& sink) override;

protected:
    Eigen::VectorXd controlLawImpl(const Eigen::VectorXd& q,
                                   const Eigen::VectorXd& v) override;

    //! A failed or abandoned plan is no warm start: drop it with the mode change.
    void onEnterDamping() override;

    // === Shared helpers used in the control law ======
    // Protected rather than private: these are the extension points a future
    // NMPC variant reuses (the same role getDynamics()/dynamicsAt() play in the
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

    /*! Solves the single-stage QP (timed under qpsolve), writing the stage-0
     *  control into tau_task_out (size = B.cols()) and recording the
     *  soft-constraint slack on success. Returns false if the solver did not
     *  converge or returned a non-finite solution — the caller's job is then to
     *  enter safe mode, not to substitute an approximation. Q_N: terminal-stage
     *  cost, used in place of Q at the terminal weight; callers pass Q itself
     *  when terminal_cost_multiplier == 1.0. */
    // Non-const refs: qp_->solve() takes raw non-const double* into these
    // (HPIPM's C API doesn't promise not to touch its inputs).
    bool solveStageQP(Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                       Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
                       Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
                       Eigen::MatrixXd& D, Eigen::VectorXd& lg,
                       Eigen::VectorXd& ug, Eigen::VectorXd& x_aug,
                       Eigen::MatrixXd& Q_N,
                       Eigen::VectorXd& tau_task_out);

    // === Full-nonlinear (multiple-shooting SQP) path, used instead of the
    // single-linearization-point buildAugmentedModel/solveStageQP pair above
    // when params_.fullNonlinear is true.

    /*! Per-node task-space quantities a TaskSpaceEval callback produces from
     *  that node's rolled-out (q,v)/dynamics -- feeds
     *  buildAugmentedModel/buildTorqueGeneralConstraint exactly like the
     *  linearized path's single node-0 computation does, just called once per
     *  rollout node instead of once per tick. */
    struct NodeTaskSpace {
        Eigen::VectorXd e, edot;    // task-space error state, dim d
        Eigen::MatrixXd J_task;     // d x n_joints_
        Eigen::VectorXd bias;       // dim d (a_d - h')
    };
    //! Writes into a NodeTaskSpace the caller owns rather than returning one:
    //! NodeTaskSpace carries four dynamically-sized Eigen members, so a by-value
    //! return costs four allocations at every rollout node.
    //! `k` is the rollout node, so the callback can score it against
    //! node_ref_[k].
    using TaskSpaceEval = std::function<void(
        int k,
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
        NodeTaskSpace& out)>;

    /*! The planar task-space evaluation behind TaskSpaceEval. The NodeRef form
     *  is what the rollout uses, one reference per node; the others evaluate
     *  against the static setpoint, for the tests and for single-shot callers. */
    void evalTaskSpace(
        const NodeRef& ref,
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6,
        NodeTaskSpace& out) const;
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

    // --- Tier 1: plan replay --------------------------------------------------

    /*! The graded failure response. Returns the torque to apply for a tick whose
     *  solve produced nothing usable, and sets mode_ accordingly.
     *
     *  J_task / Cv are this tick's, at the CURRENT measurement, and may be null
     *  when the failure happened before they were computed. Tier 1 needs them:
     *  it reuses only the stale task WRENCH and re-derives the joint torque from
     *  fresh kinematics, so the feedforward stays current and only the plan is
     *  old. Without them it falls through to Tier 2 (failToDamping). */
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
