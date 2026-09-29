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

#include "pat_arm_nmpc/arm_task_space_controller.h"
#include "pat_arm_nmpc/quad_prob_solver.h"

namespace pat_arm_nmpc {

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

    //! DLS damping for H_g inversion (see ArmTaskSpaceController::pinvDLS).
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

    NMPCParams() = default;
};

/*! Receding-horizon task-space controller (successive-linearization LTV-MPC,
 *  optional full-nonlinear multiple-shooting SQP) with box constraints on
 *  joint position/velocity solved via acados/HPIPM. Port of VORTEX
 *  ArmConstrainedNMPController. */
class ArmNMPC : public ArmTaskSpaceController {
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

    /*! Nominal = tracking the setpoint. Damping = safe mode, commanding
     *  tau = -kd*qdot to bring the arm to rest. */
    enum class ControlMode { Nominal, Damping };

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
    /*! HPIPM status from the most recent solve (0 = converged), for the node to
     *  report when it logs a transition into safe mode. */
    int lastQpStatus() const;

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

    /*! Runs getDynamics() and DLS-inverts H_g (timed under t_dynamics_us_). */
    Eigen::MatrixXd computeGeneralizedInertiaInv(Eigen::VectorXd& Cv_joints);

    /*! Lambda_inv = J*H_g_inv*J^T (timed under t_lambda_us_), NaN-checked. */
    Eigen::MatrixXd computeLambdaInv(const Eigen::MatrixXd& J,
                                      const Eigen::MatrixXd& H_g_inv);

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
     *  actuate are capped at a finite value, since these feed QP bounds. */
    Eigen::VectorXd reachableWrenchBound(const Eigen::MatrixXd& J_task) const;

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
                       int nOwned, Eigen::MatrixXd& Q_N,
                       Eigen::VectorXd& tau_task_out);

    /*! tau_joints = J^T*tau_task + Cv_joints, clamped to torqueLimsByIndex. */
    Eigen::VectorXd finalizeJointTorques(const Eigen::MatrixXd& J,
                                          const Eigen::VectorXd& tau_task,
                                          const Eigen::VectorXd& Cv_joints);


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
    using TaskSpaceEval = std::function<NodeTaskSpace(
        const Eigen::VectorXd& q_node, const Eigen::VectorXd& v_node,
        const Eigen::Vector3d& ee_pos, const Eigen::Vector4d& ee_quat,
        const Eigen::MatrixXd& J_g6, const Eigen::Matrix<double, 6, 1>& jdot_qdot6)>;

    /*! TaskSpaceEval implementations. Each control law wraps it in a thin 
     *  TaskSpaceEval lambda when calling solveNonlinearMPC()
     *  (a lambda is still needed to bind `this` into the std::function, but
     *  the actual per-dimension logic lives here, named and documented). */
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
     *  general/torque rows). */
    void recordSlack(int nOwned);

    // Print the timing info at the end of a control loop
    void printTimingInfo(
        const std::chrono::time_point<std::chrono::high_resolution_clock> start,
        int modulo = 100);

    NMPCParams params_;
    int        ee_site_id_ = -1;

    // Per-joint limits, keyed by joint name as given in joint_lims, parsed
    // in the constructor. Values are validated against the loaded MJCF via
    // ArmTaskSpaceController::jointIndexFromName() at construction time.
    std::unordered_map<std::string, std::pair<double, double>> qlims;       //!< [rad] (q-, q+)
    std::unordered_map<std::string, double>                    vlims;       //!< [rad/s] |qdot| limit
    std::unordered_map<std::string, double>                    torque_lims; //!< [N*m] |tau| limit

    // Same data as qlims/vlims/torque_lims, but indexed by arm-local joint
    // index (0..numJoints-1, matching q[6+i]/getJointPosInMsg(i)) rather
    // than name — built once in the constructor via jointIndexFromName() so
    // the per-tick control law can look bounds up by index directly instead
    // of re-hashing joint names every control tick.
    std::vector<std::pair<double, double>> qlimsByIndex;
    std::vector<double>                    vlimsByIndex;
    std::vector<double>                    torqueLimsByIndex;
    std::vector<int>                       ownedJointInds;
    int                                    n_owned = 0;

    // Safe mode. kd_ is sized n_joints_ and nonzero only on owned indices,
    // fixed at construction from the torque and velocity limits. The mode
    // latches on failure and only clears after safe_mode_recovery_ticks
    // consecutive good solves, so the output cannot chatter between damping and
    // tracking while the solver is marginal.
    std::vector<double> kd_;
    ControlMode         mode_ = ControlMode::Nominal;
    int                 consecutive_ok_ = 0;
    long                safe_mode_ticks_ = 0;

    // Box/torque-constrained QP solver (acados_c + HPIPM backend).
    std::unique_ptr<quad_prob_solver::QuadProbSolver> qp_;

    // Nominal control trajectory for the full-nonlinear (fullNonlinear=true)
    // path -- N entries of size d (task dim), warm-started/shifted tick to
    // tick by solveNonlinearMPC(). Empty/wrong-sized on the first tick (or
    // after a dimension change), in which case solveNonlinearMPC()
    // reinitializes it to zero. Unused when fullNonlinear is false.
    std::vector<Eigen::VectorXd> u_bar_;

    // Variables for investigating timing break-down. dynamics_us_/jdotqdot_us_
    // are only ever incremented by the LINEARIZED path (computeGeneralizedInertiaInv
    // / the explicit jdotQdot(6) call in each controlLaw*) -- solveNonlinearMPC()
    // never touches them, so their n_ counters stay 0 (and the printed average
    // divides by zero) whenever fullNonlinear=true. The nonlinear rollout's
    // own dynamics evaluation (dynamicsAt(), which bundles what the linearized
    // path does as two separate calls -- getDynamics-equivalent AND jdotQdot6
    // -- into one mj_forward pass, specifically to avoid redundant scratch
    // round-trips) and its rollout propagation (integrateStep(), with no
    // linearized-path counterpart at all) get their OWN dedicated timers
    // instead of being force-mapped onto dynamics_us_/jdotqdot_us_, since
    // they're genuinely not the same granularity of work.
    double t_dynamics_us_ = 0, t_lambda_us_ = 0, t_jdotqdot_us_ = 0, t_qpsolve_us_ = 0, t_total_us_ = 0;
    long   n_dynamics_ = 0, n_lambda_ = 0, n_jdotqdot_ = 0, n_qpsolve_ = 0, n_total_ = 0;
    double t_rollout_dynamics_us_ = 0, t_rollout_integrate_us_ = 0;
    long   n_rollout_dynamics_ = 0, n_rollout_integrate_ = 0;

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

}  // namespace pat_arm_nmpc
