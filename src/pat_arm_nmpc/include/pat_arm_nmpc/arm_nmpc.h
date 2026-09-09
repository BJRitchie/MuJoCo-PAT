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
// #include "pat_arm_nmpc/quad_prob_solver.h"

namespace pat_arm_nmpc {

struct JointLimit { std::string name; double q_min, q_max, qd_max, tau_max; };

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
    // Per-joint limits  
    std::vector<JointLimit> joint_lims;

    // --- Name of owned joints ----------------------------------------------
    // Joint names owned by this controller, whitespace-seperated: 
    //  "<j0_name> <j1_name> ... <jN_name>" 
    // 
    // If this field is left empty, controller assumes ownership 
    // of all joints 
    std::vector<std::string> owned_joints; 

    double Hg_damping  = 0.001;   //!< DLS damping for H_g inversion (see ArmTaskSpaceController::pinvDLS)

    std::string ee_site_name;     //!< name of the MJCF <site> at the end effector (required)

    double tau_max = 100.0;   //!< [N*m] per-joint torque saturation (post-hoc clamp, see .cpp)

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
    double du_max_x = 50.0, du_max_y = 50.0, du_max_z = 50.0;
    double du_max_wx = 5.0, du_max_wy = 5.0, du_max_wz = 5.0;

    bool   terminal_velocity_constraint = false;
    double terminal_cost_multiplier = 1.0;

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

protected:
    Eigen::VectorXd controlLaw(const Eigen::VectorXd& q,
                               const Eigen::VectorXd& v) override;

private:
    // === Shared helpers used in the controlLaw function above ======

    /*! Runs getDynamics() and DLS-inverts H_g (timed under t_dynamics_us_). */
    Eigen::MatrixXd computeGeneralizedInertiaInv(Eigen::VectorXd& Cv_joints);

    /*! Lambda_inv = J*H_g_inv*J^T (timed under t_lambda_us_), NaN-checked. */
    Eigen::MatrixXd computeLambdaInv(const Eigen::MatrixXd& J,
                                      const Eigen::MatrixXd& H_g_inv);

    /*! Builds the full augmented (task + Δq/qdot) A/B prediction matrices,
     *  sized off J_task's/Lambda_inv's dimension d (task dim == nu). */
    void buildAugmentedModel(const Eigen::MatrixXd& J_task,
                              const Eigen::MatrixXd& Lambda_inv,
                              const Eigen::VectorXd& bias,
                              const Eigen::VectorXd& v_d_task,
                              Eigen::MatrixXd& A, Eigen::MatrixXd& B);

    /*! Δq/qdot box bounds on the OWNED joints, offset by current measurement q0. */
    void buildJointBoxBounds(const Eigen::VectorXd& q0,
                              Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j);

    /*! Per-joint torque limit as a general (polytopic) constraint lg<=D*u<=ug. */
    void buildTorqueGeneralConstraint(const Eigen::MatrixXd& J,
                                       const Eigen::VectorXd& Cv_joints,
                                       Eigen::MatrixXd& D, Eigen::VectorXd& lg,
                                       Eigen::VectorXd& ug);

    /*! Solves the QP (timed under t_qpsolve_us_); on failure/non-convergence
     *  falls back to an unconstrained backward-Riccati recursion; on success
     *  logs a diagnostic if the soft box/torque slack is straining. Returns
     *  tau_task (size = B.cols()). Q_N: terminal-stage cost, used in place
     *  of Q for both the QP's terminal weight and the Riccati fallback's
     *  terminal cost -- callers pass Q itself when there's no separate
     *  terminal weight (NMPCParams::terminal_cost_multiplier == 1.0). */
    // Non-const refs: qp_->solve() takes raw non-const double* into these
    // (HPIPM's C API doesn't promise not to touch its inputs), matching the
    // pre-refactor call sites, which all passed freshly-built locals anyway.
    Eigen::VectorXd solveQPWithFallback(Eigen::MatrixXd& A, Eigen::MatrixXd& B,
                                         Eigen::MatrixXd& Q, Eigen::MatrixXd& R,
                                         Eigen::VectorXd& lbx_j, Eigen::VectorXd& ubx_j,
                                         Eigen::MatrixXd& D, Eigen::VectorXd& lg,
                                         Eigen::VectorXd& ug, Eigen::VectorXd& x_aug,
                                         int nOwned, Eigen::MatrixXd& Q_N);

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
        Eigen::VectorXd v_d_task;   // dim d
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
                                    const Eigen::VectorXd& du_max,
                                    Eigen::MatrixXd& Q_N);

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
    
    // TODO(port) quad prob solver over
    // std::unique_ptr<QuadProbSolver> qp_;

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
};

}  // namespace pat_arm_nmpc
