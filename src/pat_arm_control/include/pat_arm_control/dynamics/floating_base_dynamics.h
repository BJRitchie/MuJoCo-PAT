#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <mujoco/mujoco.h>

/*! MuJoCo's mjData, forward-declared exactly as <mujoco/mjdata.h> declares it
 *  so the dynamics-helper signatures can take an mjData* without pulling the
 *  real MuJoCo headers into this file. They are included only in the .cpp. */
struct mjData_;
typedef struct mjData_ mjData;

namespace pat_arm_control {

/*! Opaque MuJoCo model + data bundle. Fully defined in the .cpp so the MuJoCo
 *  data structures stay out of this header. */
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

/*! The plant model a task-space arm controller is built on.
 *
 *  Loads its own MJCF and uses MuJoCo purely as a rigid-body dynamics library
 *  (mass matrix, generalized Jacobian, bias forces) — it is NOT the simulator.
 *  TaskSpaceController derives from it and implements controlLaw(); a control
 *  law derives from that. Ported from VORTEX FloatingBaseDynamics with the
 *  Basilisk SysModel / message-port plumbing removed; ROS I/O lives in the
 *  node wrapper, never here. */
class FloatingBaseDynamics {
public:
    explicit FloatingBaseDynamics(const std::string& mjcf_path);
    virtual ~FloatingBaseDynamics();

    FloatingBaseDynamics(const FloatingBaseDynamics&) = delete;
    FloatingBaseDynamics& operator=(const FloatingBaseDynamics&) = delete;

    /*! Arm DOF of the loaded model (mjModel.nv - 6). Valid immediately after
     *  construction. The q/v the caller feeds must be sized 7+this / 6+this. */
    int numArmJoints() const { return n_joints_; }

    /*! Full-system DOF count (6 base + arm joints), i.e. mjModel.nv. */
    int numDof() const { return 6 + n_joints_; }

    /*! Configuration size, i.e. mjModel.nq (7 + arm joints: the free joint
     *  carries a quaternion). */
    int numConfig() const { return 7 + n_joints_; }

    /*! Per-thread workspace for fullDynamics() below.
     *
     *  A distinct type rather than a comment, because the requirement it
     *  encodes is easy to overlook and expensive to get wrong. dynamicsAt() and
     *  integrateStep() evaluate hypothetical states on MUTABLE MEMBERS
     *  (mj_scratch_, be_scratch_) that the rollout reuses ~N times a tick, so
     *  they cannot be called from two threads at once. mjModel, by contrast, is
     *  read-only once loaded and safe to share. One Workspace per thread is
     *  therefore all a second thread needs -- no second model, no second MJCF
     *  parse, a few hundred kB of mjData rather than a duplicate of everything.
     *
     *  Sized on first use and reused thereafter, so steady-state calls do not
     *  allocate. Must not be shared between threads, which is what deleting the
     *  copy enforces.
     */
    //! Opaque bundle behind Workspace: this thread's mjData plus its own
    //! BaseElim. Nested rather than free so that its definition in the .cpp has
    //! member access to BaseElim, which stays protected.
    struct WorkspaceImpl;

    class Workspace {
    public:
        Workspace();
        ~Workspace();
        Workspace(const Workspace&)            = delete;
        Workspace& operator=(const Workspace&) = delete;

    private:
        friend class FloatingBaseDynamics;
        std::unique_ptr<WorkspaceImpl> impl_;
    };

    /*! The full-system dynamics at an arbitrary (q, v), which is what a
     *  disturbance observer needs and what nothing else exposed.
     *
     *  Runs mj_forward on `ws`'s own mjData -- kinematics and inertia, no
     *  contacts, no integration -- and never touches the live state or the
     *  rollout's scratch, so it is safe to call from another thread provided
     *  that thread owns `ws`.
     *
     *  @param q                nq, same layout as controlLaw's
     *  @param v                nv
     *  @param ws               this thread's workspace
     *  @param M                out, nv x nv full mass matrix
     *  @param bias             out, nv, MuJoCo's qfrc_bias. With gravity
     *                          disabled this is exactly C(q,qdot) qdot
     *  @param H_bb_inv_H_bm    out, 6 x n_joints, the cached Schur term a
     *                          full-system estimate is reduced to joint space
     *                          with
     *
     *  Outputs are resized only if they are the wrong size, so a caller that
     *  reuses them allocates nothing after the first call.
     */
    void fullDynamics(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                      Workspace& ws, Eigen::MatrixXd& M, Eigen::VectorXd& bias,
                      Eigen::MatrixXd& H_bb_inv_H_bm) const;

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

    /*! Per-joint position range from the loaded MJCF, indexed arm-locally (the
     *  same index space jointIndexFromName() returns). Joints the model leaves
     *  unlimited come back as {-M_PI, M_PI}: a finite fallback, because these
     *  feed QP box bounds and HPIPM's factorization does not tolerate bounds
     *  spanning many orders of magnitude.
     *
     *  The model is the authority. A controller owning one arm still rolls the
     *  WHOLE model forward, and MuJoCo enforces every joint's range with
     *  constraint forces that the linearized prediction model has no term for
     *  — so a rollout allowed past a range it does not know about diverges
     *  against forces the QP cannot see. */
    std::vector<std::pair<double, double>> modelJointRanges() const;

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

    /*! The work that getDynamics(), gjm6() and jdotQdot6() all need and all used
     *  to redo: the mass matrix, the factorisation of its base block, and the site
     *  Jacobian. Every one of those three reductions is of the form
     *  X_m - X_b H_bb^-1 H_bm, so computed once per state it serves all three --
     *  three mj_fullM calls, three identical H_bb factorisations and two
     *  mj_jacSite calls become one of each. */
    struct BaseElim {
        Eigen::MatrixXd M;                      //!< nv x nv, dense symmetric
        Eigen::MatrixXd H_bm;                   //!< M.topRightCorner(6, n_joints_)
        Eigen::LDLT<Eigen::MatrixXd> H_bb_ldlt; //!< of M.topLeftCorner(6, 6)
        //! H_bb^-1 H_bm, the term both getDynamics() and gjm6() reduce by. Solved
        //! once here rather than once in each of them, which is a 6x6 solve with
        //! n_joints right-hand sides saved per rollout node.
        Eigen::MatrixXd H_bb_inv_H_bm;
        //! mj_jacSite fills these as ROW-major 3 x nv buffers; declaring them
        //! column-major silently reinterprets the strides whenever nv != 3.
        Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp, jacr;
    };

    /*! Per-node workspace for the dynamics helpers.
     *
     *  The rollout calls those helpers ~N times a tick, so a local Eigen temporary
     *  inside any of them is N allocations per tick. These are held instead, sized
     *  on first use and reused thereafter, which is what keeps the control path
     *  free of the allocator -- the hazard being glibc's arena lock and the page
     *  faults behind a fresh mapping, both unbounded under PREEMPT_RT. */
    struct NodeScratch {
        Eigen::MatrixXd J_full;        //!< 6 x nv, gjm6's stacked site Jacobian
        Eigen::VectorXd c_b, Cv_base;  //!< base rows of qfrc_bias, 6
        Eigen::VectorXd hbb_solve_vec; //!< H_bb^-1 applied to a 6-vector
        Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp_dot, jacr_dot;
        Eigen::VectorXd qvel;          //!< nv
        Eigen::MatrixXd J_0;           //!< 6 x 6, base block of the stacked Jacobian
        Eigen::VectorXd v_mid;         //!< nv, integrateStep's mid-interval velocity
    };

    /*! Size every NodeScratch buffer for this model. Called from the constructor,
     *  so nothing on the control path has to check. */
    void initNodeScratch();

    /*! Fill be.M / be.H_bm / be.H_bb_ldlt from `data` at its current qpos. */
    void computeBaseElim(mjData* data, BaseElim& be) const;

    /*! Fill be.jacp / be.jacr for one site. Separate from computeBaseElim because
     *  getDynamics needs the mass matrix and no Jacobian. */
    void computeSiteJacobian(int site_id, mjData* data, BaseElim& be) const;

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

    // The same three, reusing a BaseElim the caller already built. The
    // no-BaseElim overloads above build one and delegate here, so there is a
    // single implementation of each reduction. Each takes only what it still
    // needs beyond `be`: the Jacobian is in `be`, the bias and velocity are not.
    // gjm6 also takes an out-parameter form, because returning by value costs an
    // allocation per rollout node for a matrix whose size never changes.
    void gjm6(const BaseElim& be, Eigen::MatrixXd& out) const;
    Eigen::MatrixXd gjm6(const BaseElim& be) const;
    void getDynamics(Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints,
                     mjData* data, const BaseElim& be) const;
    Eigen::Matrix<double, 6, 1> jdotQdot6(int site_id, mjData* data,
                                          const BaseElim& be) const;

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

    /*! The same, written into a NodeDynamics the caller owns. The rollout uses
     *  this and reuses one instance: NodeDynamics carries three dynamically-sized
     *  Eigen members, so returning it by value allocates per node. Once `out` has
     *  been through this one, every later call writes into buffers that are already
     *  the right size and allocates nothing. */
    void dynamicsAt(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                    int site_id, NodeDynamics& out) const;

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

    /*! Solve A X = B by Cholesky, for a symmetric positive-definite A.
     *
     *  Returns false, leaving X untouched, if A is not positive definite -- which
     *  for a generalised inertia means the model is degenerate (a massless link),
     *  not that the arm is near a kinematic singularity. A singular Jacobian shows
     *  up in Lambda, not here.
     *
     *  Prefer this to inverting: it is a factorisation and a triangular solve
     *  rather than a full inverse plus a product, it never squares the condition
     *  number, and the caller never needs the inverse itself -- every use of
     *  H_g^-1 in this package is a product with something. */
    static bool solveSpd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                         Eigen::MatrixXd& X);

    /*! The same, factorising into a decomposition the caller owns. On the control
     *  path that matters: an `Eigen::LLT` constructed per call allocates its own
     *  storage, and the rollout calls this once per node. */
    static bool solveSpd(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                         Eigen::MatrixXd& X, Eigen::LLT<Eigen::MatrixXd>& llt);

    /*! Damped-least-squares (Tikhonov) pseudo-inverse via SVD: each singular value
     *  is scaled by s/(s^2 + damping^2), so the result stays bounded as s -> 0.
     *  The fallback for a matrix solveSpd rejects, and the reference the tests
     *  check that solve against. */
    static Eigen::MatrixXd pinvDLS(const Eigen::MatrixXd& A, double damping);

    /*! Throw std::runtime_error naming `label` if hasNaN is true. The control
     *  law's guard turns that into safe mode. */
    void checkNaN(bool hasNaN, const char* label);

    /*! Redirect MuJoCo's error and warning hooks away from their defaults, which
     *  are a file append (MUJOCO_LOG.TXT) and exit() respectively -- neither
     *  acceptable from inside a control tick. Installed by this class's
     *  constructor; exposed so a test can arm it explicitly.
     *
     *  The hooks are process-global in libmujoco, so this affects every MuJoCo
     *  user in the process and the counters below are shared. A warning is
     *  recorded and execution continues; an error throws, which ArmNMPC's guard
     *  turns into safe-mode damping. */
    static void installMujocoHooks();

    //! Counts since process start, and the most recent message of each kind.
    //! Read these off the tick -- they are what replaces MuJoCo's own logging.
    static long mjWarningCount();
    static long mjErrorCount();
    static const char* lastMjWarning();
    static const char* lastMjError();

    // --- Protected state (readable by subclass control laws) ------------------
    std::unique_ptr<MjImpl> mj_;     //!< mujoco model + data
    // Lazily created on first dynamicsAt()/integrateStep() call (mutable
    // since both are const methods) and reused for the controller's
    // lifetime -- never recreated per call/node/tick. Shares mj_->model,
    // does not own it.
    mutable std::unique_ptr<MjScratchData> mj_scratch_;
    //! dynamicsAt()'s BaseElim, held rather than created per call for the same
    //! reason as mj_scratch_: the rollout calls it ~N times a tick.
    mutable BaseElim be_scratch_;
    //! The dynamics helpers' Eigen temporaries, for the same reason again. Sized
    //! once by initNodeScratch() at construction.
    mutable NodeScratch ns_;
    int      n_joints_ = 0;          //!< arm DOF of the model (nv − 6)
};

}  // namespace pat_arm_control
