#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

/*! MuJoCo's mjData, forward-declared exactly as <mujoco/mjdata.h> declares it
 *  so the dynamics-helper signatures can take an mjData* without pulling the
 *  real MuJoCo headers into this file. They are included only in the .cpp. */
struct mjData_;
typedef struct mjData_ mjData;

namespace pat_arm_nmpc {

/*! Opaque MuJoCo model + data bundle. Fully defined in the .cpp so raw MuJoCo
 *  headers never leak into this heade
 r (relic from the Basilisk port). */
struct MjImpl;

/*! Opaque scratch mjData bundle — a SEPARATE mjData sharing MjImpl's mjModel,
 *  used to evaluate dynamics at hypothetical (predicted / rolled-out) states
 *  without perturbing the live tracked state. Does not own the mjModel*. */
struct MjScratchData;


/*! Bundled dynamics quantities at one rollout node, returned by
*  dynamicsAt() below. */
struct NodeDynamics {
    Eigen::Vector3d           ee_pos;       //!< world frame [m]
    Eigen::Vector4d           ee_quat;      //!< [w,x,y,z], world frame
    Eigen::MatrixXd           H_g;          //!< n_joints_ x n_joints_
    Eigen::VectorXd           Cv_joints;    //!< n_joints_
    Eigen::MatrixXd           J_g6;         //!< 6 x n_joints_ (rows [0:3) lin, [3:6) ang)
    Eigen::Matrix<double,6,1> jdot_qdot6;   //!< GJM-corrected bias accel, 6-vector
};

// Simple RAII stopwatch — accumulates elapsed microseconds into a running
// sum/count so per-tick printf overhead doesn't distort the measurement.
struct ScopedTimer {
    ScopedTimer(double& accumUs, long& count)
        : accumUs_(accumUs), count_(count)
        , start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto stop = std::chrono::high_resolution_clock::now();
        accumUs_ += std::chrono::duration<double, std::micro>(stop - start_).count();
        ++count_;
    }
    double& accumUs_;
    long&   count_;
    std::chrono::high_resolution_clock::time_point start_;
};

/*! Base class for a task-space arm controller.
 *
 *  Loads its own MJCF and uses MuJoCo purely as a rigid-body dynamics library
 *  (mass matrix, generalized Jacobian, bias forces) — it is NOT the simulator.
 *  Subclasses implement controlLaw(). Ported from VORTEX ArmTaskSpaceController
 *  with the Basilisk SysModel / message-port plumbing removed; ROS I/O lives
 *  in the node wrapper, never here. */
class ArmTaskSpaceController {
public:
    explicit ArmTaskSpaceController(const std::string& mjcf_path);
    virtual ~ArmTaskSpaceController();

    ArmTaskSpaceController(const ArmTaskSpaceController&) = delete;
    ArmTaskSpaceController& operator=(const ArmTaskSpaceController&) = delete;

    /*! Arm DOF of the loaded model (mjModel.nv - 6). Valid immediately after
     *  construction. The q/v the caller feeds must be sized 7+this / 6+this. */
    int numArmJoints() const { return n_joints_; }

    /*! Number of active joints to control (≤ MAX_JOINTS).
     *  Must equal the URDF arm DOF and be set before Reset(). */
    int numJoints = 0;

    // Setter functions
    void setDesiredPos(double x, double y, double z) {
        desiredPos[0] = x; desiredPos[1] = y; desiredPos[2] = z; }
    void setDesiredVel(double x, double y, double z) {
        desiredVel[0] = x; desiredVel[1] = y; desiredVel[2] = z; }
    void setDesiredAcc(double x, double y, double z) {
        desiredAcc[0] = x; desiredAcc[1] = y; desiredAcc[2] = z; }

    /*! Desired EE orientation setpoint, as a unit quaternion [w, x, y, z]
     *  (MuJoCo convention). Only consumed by subclasses that opt into
     *  orientation tracking (e.g. ArmNMPController with
     *  NMPCParams.trackOrientation = true) — subclasses that never read
     *  desiredQuat are unaffected. */
    void setDesiredOrient(double w, double x, double y, double z) {
        desiredQuat[0] = w; desiredQuat[1] = x; desiredQuat[2] = y; desiredQuat[3] = z; }
    /*! Desired EE angular velocity setpoint, world frame [rad/s]. */
    void setDesiredAngVel(double x, double y, double z) {
        desiredAngVel[0] = x; desiredAngVel[1] = y; desiredAngVel[2] = z; }
    /*! Desired EE angular acceleration setpoint, world frame [rad/s^2]. */
    void setDesiredAngAcc(double x, double y, double z) {
        desiredAngAcc[0] = x; desiredAngAcc[1] = y; desiredAngAcc[2] = z; }

protected:
    /*! Subclass control law.
     *  @param q  configuration vector  (nq)
     *  @param v  velocity vector       (nv)
     *  @return   joint torques [N*m], length n_joints_ */
    virtual Eigen::VectorXd controlLaw(const Eigen::VectorXd& q,
                                       const Eigen::VectorXd& v) = 0;

    /*! Copy q (nq) / v (nv) into the LIVE mj_->data and run mj_forward, so the
     *  dynamics helpers below (which read mj_->data by default) reflect the
     *  given state. This is the de-Basilisk'd replacement for VORTEX's
     *  UpdateState(). Throws if q/v are not exactly nq/nv long.
     *  q layout: [base_pos(3), base_quat(4, w,x,y,z), joint_angles(n_joints_)];
     *  v layout: [base_linvel(3), base_angvel(3), joint_vels(n_joints_)]. */
    void loadLiveState(const Eigen::VectorXd& q, const Eigen::VectorXd& v);

    /*! Resolve a MuJoCo site name to its id. Throws if not found — use this
     *  instead of raw mj_name2id so a typo'd or missing site fails loudly at
     *  construction time, not as a silent -1 read later. */
    int siteIdFromName(const std::string& name) const;

    /*! Resolve a MuJoCo joint name to its ARM-LOCAL index — i.e. the same
     *  index space as getJointPosInMsg(i)/getJointVelInMsg(i)/q[6+i]/v[6+i]
     *  (0 = first arm joint after the 6 free-floating base DOFs), not the
     *  raw MuJoCo joint id. Throws if the name doesn't resolve to a
     *  single-DOF (hinge/slide) joint within [0, n_joints_) — use this
     *  instead of raw mj_name2id so a typo'd joint name (e.g. in a
     *  user-supplied joint-limit string) fails loudly at construction time
     *  rather than silently never being constrained. */
    int jointIndexFromName(const std::string& name) const;

    // --- Dynamics helpers (valid after mj_forward has run) ---------------
    // All take an optional trailing mjData* -- defaults to the live mj_->data
    // when omitted (every existing call site), but lets dynamicsAt() below
    // evaluate the exact same quantities against the scratch buffer instead,
    // without duplicating any of this logic.

    /*! End-effector position in world frame [m]. */
    Eigen::Vector3d  eePosition(int site_id, mjData* data = nullptr) const;

    /*! End-effector orientation in world frame, unit quaternion [w, x, y, z]
     *  (MuJoCo convention), read from site_xmat. */
    Eigen::Vector4d  eeOrientation(int site_id, mjData* data = nullptr) const;

    /*! Generalised Jacobian  J_g = J_m − J_b H_bb⁻¹ H_bm  (3 × n_joints_). */
    Eigen::Matrix3Xd gjm(int site_id, mjData* data = nullptr) const;

    /*! Generalised Jacobian, position AND orientation stacked
     *  (6 × n_joints_): rows [0:3) translational (same as gjm()),
     *  rows [3:6) rotational. Same base-coupling correction applied to both
     *  blocks since J_b^-1 H_bb H_bm correction is linear per-row. */
    Eigen::MatrixXd gjm6(int site_id, mjData* data = nullptr) const;

    /*! Joint generalised inertia H_g and generalised Coriolis / centrifugal
     *  bias Cv_joints = c_m - H_bm^T H_bb^-1 c_b (base eliminated, so that
     *  H_g qddot + Cv_joints = tau holds for the unactuated floating base). */
    void getDynamics(Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints,
                      mjData* data = nullptr) const;

    /*! GJM-corrected bias acceleration  h′ = J̇q̇  in world frame [m/s²]. */
    Eigen::Vector3d  jdotQdot(int site_id, mjData* data = nullptr) const;

    /*! GJM-corrected bias acceleration, position AND orientation stacked
     *  (6-vector): [0:3) linear [m/s^2], [3:6) angular [rad/s^2]. */
    Eigen::Matrix<double, 6, 1> jdotQdot6(int site_id, mjData* data = nullptr) const;

    /*! Full nonlinear effects vector (gravity + Coriolis).
     *  Elements [0:6] are base DOFs; [6:6+n_joints_] are joint DOFs. */
    const Eigen::VectorXd& nle() const;   // returns qfrc_bias

    /*! Evaluates the same quantities eePosition()/eeOrientation()/
     *  getDynamics()/gjm6()/jdotQdot6() return, but at an arbitrary
     *  hypothetical (q, v) rather than the live measured state -- sets
     *  qpos/qvel on a cached scratch mjData (kinematics + inertia only, via
     *  mj_forward -- NOT mj_step, no contacts/integration), then reads
     *  everything out in one pass. Never touches mj_->data / the live
     *  UpdateState() state -- safe to call repeatedly from a subclass's SQP
     *  rollout without perturbing the next tick's live dynamics.
     *  q (nq = 7+n_joints_), v (nv = 6+n_joints_), same layout as
     *  controlLaw's own q/v arguments. */
    NodeDynamics dynamicsAt(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                             int site_id) const;

    /*! Rolls the FULL coupled floating-base + joint state forward one Ts
     *  step under an applied joint torque, via MuJoCo's own forward
     *  dynamics solve on the scratch buffer (mj_forward with qfrc_applied
     *  set on the joint rows only -- base rows get zero direct actuation,
     *  same as the real system), then semi-implicit-Euler-integrates
     *  (mj_integratePos() for q, which correctly handles the free joint's
     *  quaternion component). tau_joints_generalized is the torque the real
     *  plant would actually receive (n_joints_) -- i.e. INCLUDING the Cv_joints
     *  feedforward, J_task^T*tau_task + Cv_joints, as finalizeJointTorques()
     *  applies it. mj_forward supplies the true bias/Coriolis/gravity forces
     *  itself, and the feedforward cancels them (it is not double-counted);
     *  omitting it predicts a plant with no Coriolis compensation, which is
     *  not what runs. Never touches mj_->data. */
    void integrateStep(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                        const Eigen::VectorXd& tau_joints_generalized, double Ts,
                        Eigen::VectorXd& q_next, Eigen::VectorXd& v_next) const;

    static Eigen::Vector3d sat(const Eigen::Vector3d& x);

    /*! Damped Least-Squares (Tikhonov) Pseudo-Inverse. When "name" is set, will 
        log warnings when matrix is ill-conditioned */ 
    static Eigen::MatrixXd pinvDLS(
        const Eigen::MatrixXd& A, double damping, const std::string& name = "");

    /*! Log a BSK_ERROR and throw if hasNaN is true. */
    void checkNaN(bool hasNaN, const char* label);

    // --- Protected state (readable by subclass control laws) ------------------
    std::unique_ptr<MjImpl> mj_;     //!< mujoco model + data
    // Lazily created on first dynamicsAt()/integrateStep() call (mutable
    // since both are const methods) and reused for the controller's
    // lifetime -- never recreated per call/node/tick. Shares mj_->model,
    // does not own it.
    mutable std::unique_ptr<MjScratchData> mj_scratch_;
    int      n_joints_ = 0;          //!< arm DOF from URDF (nv − 6)
    uint64_t prevTime_ = 0;

    /*! Desired task-space trajectory — update each timestep or leave at zero. */
    double desiredPos[3] = {0.0, 0.0, 0.0};  //!< [m]    EE position setpoint
    double desiredVel[3] = {0.0, 0.0, 0.0};  //!< [m/s]  EE velocity setpoint
    double desiredAcc[3] = {0.0, 0.0, 0.0};  //!< [m/s²] EE acceleration setpoint

    /*! Desired EE orientation trajectory — only meaningful to subclasses
     *  that opt into orientation tracking. desiredQuat defaults to the
     *  identity quaternion; a subclass must read the arm's own initial
     *  orientation (or have the scenario call setDesiredOrient) before
     *  using it as an error reference. */
    double desiredQuat[4]   = {1.0, 0.0, 0.0, 0.0};  //!< [w,x,y,z] EE orientation setpoint
    double desiredAngVel[3] = {0.0, 0.0, 0.0};       //!< [rad/s]   EE angular velocity setpoint
    double desiredAngAcc[3] = {0.0, 0.0, 0.0};       //!< [rad/s²]  EE angular acceleration setpoint
};

}  // namespace pat_arm_nmpc
