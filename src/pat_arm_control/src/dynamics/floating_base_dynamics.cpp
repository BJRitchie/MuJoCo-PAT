#include "pat_arm_control/dynamics/floating_base_dynamics.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace pat_arm_control {

// === MuJoCo error / warning hooks ===========================================
// MuJoCo's defaults are both unacceptable inside a control tick: mju_warning
// opens and appends to MUJOCO_LOG.TXT (a file write of unbounded latency, from a
// path that fires precisely when the solver is struggling), and mju_error prints
// and calls exit(), killing a process that is mid-motion with a torque latched in
// the actuators.
//
// The hooks are global C function pointers, so the state behind them is global
// too. Plain counters and a fixed buffer, no allocation and no I/O: a hook can
// fire from anywhere inside mj_forward.
namespace {
std::atomic<long> g_mj_warnings{0};
std::atomic<long> g_mj_errors{0};
constexpr std::size_t kMjMsgLen = 192;
char g_mj_last_warning[kMjMsgLen] = {};
char g_mj_last_error[kMjMsgLen]   = {};

void copyMsg(char* dst, const char* msg) {
    // std::snprintf rather than a std::string: bounded, and it does not allocate
    // for a plain "%s".
    std::snprintf(dst, kMjMsgLen, "%s", msg ? msg : "(null)");
}

void patMjWarning(const char* msg) {
    g_mj_warnings.fetch_add(1, std::memory_order_relaxed);
    copyMsg(g_mj_last_warning, msg);
}

void patMjError(const char* msg) {
    g_mj_errors.fetch_add(1, std::memory_order_relaxed);
    copyMsg(g_mj_last_error, msg);
    // MuJoCo requires this not to return -- it calls exit() if it does. Throwing
    // hands control to ArmNMPC's existing guard, which turns it into safe-mode
    // joint damping: the same trade the QP-failure path already makes, a recorded
    // survivable event instead of a dead process. Unwinding is not realtime, but
    // this only runs when the alternative is termination.
    throw std::runtime_error(std::string("MuJoCo error: ") + g_mj_last_error);
}
}  // namespace

void FloatingBaseDynamics::installMujocoHooks() {
    mju_user_warning = &patMjWarning;
    mju_user_error   = &patMjError;
}

long FloatingBaseDynamics::mjWarningCount() {
    return g_mj_warnings.load(std::memory_order_relaxed);
}
long FloatingBaseDynamics::mjErrorCount() {
    return g_mj_errors.load(std::memory_order_relaxed);
}
const char* FloatingBaseDynamics::lastMjWarning() { return g_mj_last_warning; }
const char* FloatingBaseDynamics::lastMjError()   { return g_mj_last_error; }

// === MjImpl — MuJoCo model + data bundle ====================================
struct MjImpl {
    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    ~MjImpl() {
        if (data)  mj_deleteData(data);
        if (model) mj_deleteModel(model);
    }
};

// === MjScratchData — second mjData sharing mj_->model, for dynamicsAt()/ ===
// === integrateStep() hypothetical-state evaluation, never touching mj_->data
struct MjScratchData {
    mjData* data = nullptr;   // does NOT own the mjModel*
    ~MjScratchData() {
        if (data) mj_deleteData(data);
    }
};

// === WorkspaceImpl — one thread's mjData + base-elimination buffer ==========
// Separate from MjScratchData because it carries the BaseElim too: the point of
// a Workspace is that a second thread shares NOTHING mutable with the rollout,
// and the rollout's be_scratch_ is a member of the controller.
struct FloatingBaseDynamics::WorkspaceImpl {
    mjData* data = nullptr;   // does NOT own the mjModel*
    BaseElim be;
    ~WorkspaceImpl() {
        if (data) mj_deleteData(data);
    }
};

FloatingBaseDynamics::Workspace::Workspace() = default;
FloatingBaseDynamics::Workspace::~Workspace() = default;

// === Construction ============================================================
FloatingBaseDynamics::FloatingBaseDynamics(
    const std::string& mjcf_path)
    : mj_(std::make_unique<MjImpl>())
{
    // Before the first MuJoCo call, so even a load failure is reported through the
    // hooks rather than MUJOCO_LOG.TXT.
    installMujocoHooks();

    char error[1000] = "";
    mj_->model = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
    if (!mj_->model) {
        throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }

    // TODO - should constrained joints be allowed 
    if (mj_->model->jnt_type[0] != mjJNT_FREE) {
        throw std::runtime_error(
            "FloatingBaseDynamics: root joint is not a free joint — "
            "check the MJCF/URDF root <joint type=\"floating\">.");
    }

    // (TODO - should this be disabled?) Disable gravity 
    mj_->model->opt.gravity[0] = 0;
    mj_->model->opt.gravity[1] = 0;
    mj_->model->opt.gravity[2] = 0;

    mj_->data = mj_makeData(mj_->model);
    n_joints_ = mj_->model->nv - 6;

    initNodeScratch();   // needs the model and n_joints_, so last
}

FloatingBaseDynamics::~FloatingBaseDynamics() = default;

// === Live-state load (de-Basilisk'd UpdateState) =============================
void FloatingBaseDynamics::loadLiveState(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    const int nq = mj_->model->nq;
    const int nv = mj_->model->nv;
    if (q.size() != nq || v.size() != nv) {
        throw std::runtime_error(
            "FloatingBaseDynamics::loadLiveState: q/v size mismatch — got q=" +
            std::to_string(q.size()) + " v=" + std::to_string(v.size()) +
            ", model needs nq=" + std::to_string(nq) + " nv=" + std::to_string(nv));
    }
    for (int i = 0; i < nq; ++i) mj_->data->qpos[i] = q[i];
    for (int i = 0; i < nv; ++i) mj_->data->qvel[i] = v[i];
    mj_forward(mj_->model, mj_->data);
}

// === Protected dynamics helpers =========================================================
int FloatingBaseDynamics::siteIdFromName(const std::string& name) const
{
    int id = mj_name2id(mj_->model, mjOBJ_SITE, name.c_str());
    if (id < 0) {
        throw std::runtime_error(
            "FloatingBaseDynamics: no <site> named \"" + name +
            "\" found in the MJCF. Check spelling and that the site is "
            "defined in the loaded model file.");
    }
    return id;
}

std::vector<std::pair<double, double>> FloatingBaseDynamics::modelJointRanges() const
{
    std::vector<std::pair<double, double>> ranges(
        n_joints_, std::make_pair(-M_PI, M_PI));

    for (int j = 0; j < mj_->model->njnt; ++j) {
        const int type = mj_->model->jnt_type[j];

        // skip the base freejoint
        if (type != mjJNT_HINGE && type != mjJNT_SLIDE) continue;   

        const int idx = mj_->model->jnt_dofadr[j] - 6;
        if (idx < 0 || idx >= n_joints_) continue;
        if (!mj_->model->jnt_limited[j]) continue; // keep the finite fallback
        ranges[idx] = std::make_pair(mj_->model->jnt_range[2 * j],
                                      mj_->model->jnt_range[2 * j + 1]);
    }
    return ranges;
}

int FloatingBaseDynamics::jointIndexFromName(const std::string& name) const
{
    int jnt_id = mj_name2id(mj_->model, mjOBJ_JOINT, name.c_str());
    if (jnt_id < 0) {
        throw std::runtime_error(
            "FloatingBaseDynamics: no <joint> named \"" + name +
            "\" found in the MJCF. Check spelling and that the joint is "
            "defined in the loaded model file.");
    }
    int jnt_type = mj_->model->jnt_type[jnt_id];
    if (jnt_type != mjJNT_HINGE && jnt_type != mjJNT_SLIDE) {
        throw std::runtime_error(
            "FloatingBaseDynamics: joint \"" + name + "\" is not a "
            "single-DOF hinge/slide joint — cannot map it to an arm-local "
            "control index.");
    }
    // jnt_dofadr is the joint's offset into the FULL qvel vector (base free
    // joint occupies dofs [0:6)); subtract 6 to land in the same arm-local
    // index space as getJointPosInMsg(i)/q[6+i]/v[6+i].
    int idx = mj_->model->jnt_dofadr[jnt_id] - 6;
    if (idx < 0 || idx >= n_joints_) {
        throw std::runtime_error(
            "FloatingBaseDynamics: joint \"" + name + "\" resolved to "
            "arm-local index " + std::to_string(idx) + ", outside "
            "[0, n_joints_) — is it part of the free-floating base or a "
            "different kinematic branch?");
    }
    return idx;
}

Eigen::Vector3d FloatingBaseDynamics::eePosition(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const mjtNum* p = d->site_xpos + 3 * site_id;
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

Eigen::Vector4d FloatingBaseDynamics::eeOrientation(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    mjtNum quat[4];
    mju_mat2Quat(quat, d->site_xmat + 9 * site_id);
    return Eigen::Vector4d(quat[0], quat[1], quat[2], quat[3]);
}

void FloatingBaseDynamics::initNodeScratch()
{
    const int nv = mj_->model->nv;
    ns_.J_full.resize(6, nv);
    ns_.c_b.resize(6);
    ns_.Cv_base.resize(6);
    ns_.hbb_solve_vec.resize(6);
    ns_.jacp_dot.resize(3, nv);
    ns_.jacr_dot.resize(3, nv);
    ns_.qvel.resize(nv);
    ns_.J_0.resize(6, 6);
    ns_.v_mid.resize(nv);
}

void FloatingBaseDynamics::computeBaseElim(mjData* data, BaseElim& be) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;

    be.M.resize(nv, nv);
    mj_fullM(mj_->model, be.M.data(), d->qM);       // already dense + symmetric
    be.H_bm = be.M.topRightCorner(6, n_joints_);
    be.H_bb_ldlt.compute(be.M.topLeftCorner(6, 6));
    be.H_bb_inv_H_bm = be.H_bb_ldlt.solve(be.H_bm);
}

void FloatingBaseDynamics::computeSiteJacobian(
    int site_id, mjData* data, BaseElim& be) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    be.jacp.resize(3, nv);
    be.jacr.resize(3, nv);
    mj_jacSite(mj_->model, d, be.jacp.data(), be.jacr.data(), site_id);
}

void FloatingBaseDynamics::getDynamics(
    Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints, mjData* data) const
{
    BaseElim be;
    computeBaseElim(data, be);
    getDynamics(H_g, Cv_joints, data, be);
}

void FloatingBaseDynamics::getDynamics(
    Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints, mjData* data,
    const BaseElim& be) const
{
    mjData* d = data ? data : mj_->data;

    // Generalised inertia:  H_g = H_mm − H_bm^T H_bb^{−1} H_bm
    H_g.resize(n_joints_, n_joints_);
    H_g.noalias() = be.M.bottomRightCorner(n_joints_, n_joints_)
                    - be.H_bm.transpose() * be.H_bb_inv_H_bm;

    // Generalised bias, same base elimination as H_g (unactuated base):
    //   Cv_g = c_m − H_bm^T H_bb^{−1} c_b
    // The raw joint slice c_m alone is not the reduced-system bias -- it
    // omits the base Coriolis reaction, so the feedforward under-cancels and
    // H_g q̈ = J^T τ no longer holds (consistent with H_g, gjm6 and
    // jdotQdot6, which all apply this same elimination).
    for (int i = 0; i < 6; ++i) ns_.c_b[i] = d->qfrc_bias[i];
    Cv_joints.resize(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
        Cv_joints[i] = d->qfrc_bias[6 + i];
    }
    ns_.hbb_solve_vec = be.H_bb_ldlt.solve(ns_.c_b);
    Cv_joints.noalias() -= be.H_bm.transpose() * ns_.hbb_solve_vec;
}

Eigen::Matrix3Xd FloatingBaseDynamics::gjm(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    // mj_jacSite fills jacp/jacr as row-major 3 x nv buffers (row0 = d/dq of x,
    // row1 = y, row2 = z). Eigen::MatrixXd is column-major by default, so it
    // must be declared RowMajor here or the buffer gets reinterpreted with the
    // wrong strides whenever nv != 3.
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacr(3, nv);  // jacr unused here but required by API
    mj_jacSite(mj_->model, d, jacp.data(), jacr.data(), site_id);

    Eigen::MatrixXd J_b = jacp.leftCols(6);
    Eigen::MatrixXd J_m = jacp.rightCols(n_joints_);

    Eigen::MatrixXd M(nv, nv);
    mj_fullM(mj_->model, M.data(), d->qM);  // already dense + symmetric

    return J_m - J_b * M.topLeftCorner(6, 6).ldlt().solve(M.topRightCorner(6, n_joints_));
}

Eigen::MatrixXd FloatingBaseDynamics::gjm6(int site_id, mjData* data) const
{
    BaseElim be;
    computeBaseElim(data, be);
    computeSiteJacobian(site_id, data, be);
    return gjm6(be);
}

void FloatingBaseDynamics::gjm6(const BaseElim& be, Eigen::MatrixXd& out) const
{
    ns_.J_full.topRows(3)    = be.jacp;
    ns_.J_full.bottomRows(3) = be.jacr;

    // Same base-coupling correction as gjm(), applied to both the
    // translational and rotational rows at once — the correction
    // J_b H_bb^-1 H_bm is linear per-row, so stacking jacp/jacr first and
    // correcting once is equivalent to correcting each block separately.
    out.resize(6, n_joints_);
    out.noalias() = ns_.J_full.rightCols(n_joints_)
                    - ns_.J_full.leftCols(6) * be.H_bb_inv_H_bm;
}

Eigen::MatrixXd FloatingBaseDynamics::gjm6(const BaseElim& be) const
{
    Eigen::MatrixXd out;
    gjm6(be, out);
    return out;
}

Eigen::Vector3d FloatingBaseDynamics::jdotQdot(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    // Same row-major layout as mj_jacSite — see gjm() above.
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp_dot(3, nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacr_dot(3, nv);
    // mj_jacDot requires MuJoCo >= 2.3.3. It analytically differentiates the
    // site Jacobian
    mj_jacDot(mj_->model, d, jacp_dot.data(), jacr_dot.data(),
              d->site_xpos + 3 * site_id, mj_->model->site_bodyid[site_id]);

    Eigen::VectorXd qvel(nv);
    for (int i = 0; i < nv; ++i) qvel[i] = d->qvel[i];

    Eigen::Vector3d bias = jacp_dot * qvel;

    // GJM correction: subtract base Coriolis pseudo-acceleration
    //   h' = bias - J_0 H_bb^{-1} Cv_base
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv), jacr(3, nv);
    mj_jacSite(mj_->model, d, jacp.data(), jacr.data(), site_id);
    Eigen::MatrixXd J_0 = jacp.leftCols(6);

    Eigen::MatrixXd M(nv, nv);
    mj_fullM(mj_->model, M.data(), d->qM);

    Eigen::VectorXd Cv_base(6);
    for (int i = 0; i < 6; ++i) Cv_base[i] = d->qfrc_bias[i];

    return bias - J_0 * M.topLeftCorner(6, 6).ldlt().solve(Cv_base);
}

Eigen::Matrix<double, 6, 1> FloatingBaseDynamics::jdotQdot6(int site_id, mjData* data) const
{
    BaseElim be;
    computeBaseElim(data, be);
    computeSiteJacobian(site_id, data, be);
    return jdotQdot6(site_id, data, be);
}

Eigen::Matrix<double, 6, 1> FloatingBaseDynamics::jdotQdot6(
    int site_id, mjData* data, const BaseElim& be) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    mj_jacDot(mj_->model, d, ns_.jacp_dot.data(), ns_.jacr_dot.data(),
              d->site_xpos + 3 * site_id, mj_->model->site_bodyid[site_id]);

    for (int i = 0; i < nv; ++i) ns_.qvel[i] = d->qvel[i];

    ns_.J_0.topRows(3)    = be.jacp.leftCols(6);
    ns_.J_0.bottomRows(3) = be.jacr.leftCols(6);

    for (int i = 0; i < 6; ++i) ns_.Cv_base[i] = d->qfrc_bias[i];

    // The return is fixed-size, so it costs nothing; everything feeding it comes
    // out of the held scratch.
    Eigen::Matrix<double, 6, 1> bias;
    bias << ns_.jacp_dot * ns_.qvel, ns_.jacr_dot * ns_.qvel;

    ns_.hbb_solve_vec = be.H_bb_ldlt.solve(ns_.Cv_base);
    return bias - ns_.J_0 * ns_.hbb_solve_vec;
}

NodeDynamics FloatingBaseDynamics::dynamicsAt(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v, int site_id) const
{
    NodeDynamics nd;
    dynamicsAt(q, v, site_id, nd);
    return nd;
}

void FloatingBaseDynamics::dynamicsAt(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v, int site_id,
    NodeDynamics& out) const
{
    if (!mj_scratch_) {
        mj_scratch_ = std::make_unique<MjScratchData>();
        mj_scratch_->data = mj_makeData(mj_->model);
    }
    mjData* d = mj_scratch_->data;

    for (int i = 0; i < mj_->model->nq; ++i) d->qpos[i] = q[i];
    for (int i = 0; i < mj_->model->nv; ++i) d->qvel[i] = v[i];
    mj_forward(mj_->model, d);

    // One mass matrix, one base factorisation and one site Jacobian for all three
    // reductions. be_scratch_ is a member so the rollout's ~N calls per tick reuse
    // its buffers instead of reallocating them at every node.
    computeBaseElim(d, be_scratch_);
    computeSiteJacobian(site_id, d, be_scratch_);

    out.ee_pos  = eePosition(site_id, d);
    out.ee_quat = eeOrientation(site_id, d);
    getDynamics(out.H_g, out.Cv_joints, d, be_scratch_);
    gjm6(be_scratch_, out.J_g6);
    out.jdot_qdot6 = jdotQdot6(site_id, d, be_scratch_);
}

void FloatingBaseDynamics::fullDynamics(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v, Workspace& ws,
    Eigen::MatrixXd& M, Eigen::VectorXd& bias,
    Eigen::MatrixXd& H_bb_inv_H_bm) const
{
    const int nq = mj_->model->nq;
    const int nv = mj_->model->nv;
    if (q.size() != nq || v.size() != nv) {
        throw std::runtime_error(
            "fullDynamics: q must be nq and v must be nv for this model");
    }

    // Lazily created, like the rollout's own scratch, so constructing a
    // Workspace is cheap and the mjData is only paid for if it is used.
    if (!ws.impl_) {
        ws.impl_ = std::make_unique<WorkspaceImpl>();
        ws.impl_->data = mj_makeData(mj_->model);
    }
    mjData* d = ws.impl_->data;

    for (int i = 0; i < nq; ++i) d->qpos[i] = q[i];
    for (int i = 0; i < nv; ++i) d->qvel[i] = v[i];
    mj_forward(mj_->model, d);

    computeBaseElim(d, ws.impl_->be);

    // Resize only when wrong, so a caller reusing its outputs allocates nothing
    // after the first call -- the observer runs at the control rate.
    if (M.rows() != nv || M.cols() != nv) M.resize(nv, nv);
    M = ws.impl_->be.M;
    if (bias.size() != nv) bias.resize(nv);
    for (int i = 0; i < nv; ++i) bias[i] = d->qfrc_bias[i];
    if (H_bb_inv_H_bm.rows() != 6 || H_bb_inv_H_bm.cols() != n_joints_) {
        H_bb_inv_H_bm.resize(6, n_joints_);
    }
    H_bb_inv_H_bm = ws.impl_->be.H_bb_inv_H_bm;
}

void FloatingBaseDynamics::integrateStep(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v,
    const Eigen::VectorXd& tau_joints_generalized, double Ts,
    Eigen::VectorXd& q_next, Eigen::VectorXd& v_next) const
{
    if (!mj_scratch_) {
        mj_scratch_ = std::make_unique<MjScratchData>();
        mj_scratch_->data = mj_makeData(mj_->model);
    }
    mjData* d = mj_scratch_->data;
    const int nq = mj_->model->nq;
    const int nv = mj_->model->nv;

    for (int i = 0; i < nq; ++i) d->qpos[i] = q[i];
    for (int i = 0; i < nv; ++i) d->qvel[i] = v[i];

    // Joint torque as the real plant receives it (the caller includes the
    // Cv_joints feedforward) -- no direct actuation on the free floating
    // base. mj_forward computes the true coupled bias/Coriolis/gravity forces
    // for the FULL system itself as part of solving qacc; the feedforward in
    // qfrc_applied cancels them, exactly as it does on the real plant.
    for (int i = 0; i < 6; ++i) d->qfrc_applied[i] = 0.0;
    for (int i = 0; i < n_joints_; ++i) d->qfrc_applied[6 + i] = tau_joints_generalized[i];

    mj_forward(mj_->model, d);   // solves the full coupled base+joint qacc

    v_next.resize(nv);
    for (int i = 0; i < nv; ++i) {
        v_next[i]    = v[i] + Ts * d->qacc[i];
        ns_.v_mid[i] = v[i] + 0.5 * Ts * d->qacc[i];
    }

    q_next = q;
    // Position advances with the MID-interval velocity, i.e. exact
    // constant-acceleration integration (dq = Ts*v + Ts^2/2*a). Using v_next
    // (semi-implicit Euler) gives dq = Ts*v + Ts^2*a -- double the
    // acceleration's contribution, inconsistent with both the real plant
    // (constant torque over Ts) and the QP model's B (-Ts^2/2 * Lambda_inv),
    // and worse the larger Ts is.
    // mj_integratePos correctly integrates the free joint's quaternion
    // component via the exponential map -- naive q + Ts*v is only valid for
    // the hinge/slide joint rows and would corrupt the base orientation.
    mj_integratePos(mj_->model, q_next.data(), ns_.v_mid.data(), Ts);
}

Eigen::Vector3d FloatingBaseDynamics::sat(const Eigen::Vector3d& x)
{
    return x.cwiseMax(-1.0).cwiseMin(1.0);
}

bool FloatingBaseDynamics::solveSpd(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X)
{
    // Via LLT, not LDLT. 
    // Both are Cholesky factorisations -- they rewrite a symmetric A as a product
    // of triangular factors, after which solving A X = B is two triangular
    // substitutions instead of an inverse:
    //   LLT  gives A = L L^T, and exists ONLY if A is positive definite, so the
    //        factorisation failing is itself the positive-definiteness test.
    //   LDLT gives A = L D L^T with a diagonal D, which also covers semidefinite
    //        and indefinite A -- it succeeds on matrices LLT rejects.
    // LDLT is the more forgiving of the two, which is exactly why it is wrong
    // here: it would factorise a degenerate inertia without complaint and hide
    // the condition the caller is asking about.
    Eigen::LLT<Eigen::MatrixXd> llt;
    return solveSpd(A, B, X, llt);
}

bool FloatingBaseDynamics::solveSpd(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X,
    Eigen::LLT<Eigen::MatrixXd>& llt)
{
    llt.compute(A);
    if (llt.info() != Eigen::Success) return false;
    X = llt.solve(B);
    return true;
}

Eigen::MatrixXd FloatingBaseDynamics::pinvDLS(
    const Eigen::MatrixXd& A, double damping)
{
    // Thin, not Full: for a non-square A (m x n) the thin factors are
    // U: m x k, V: n x k, sigma: k  with k = min(m,n), so
    // V * diag(sigma_inv) * U^T is a well-formed n x m pseudo-inverse.
    // Full U/V make that product dimension-mismatch — it aborts under live
    // Eigen asserts and silently computes a wrong result under NDEBUG.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Damped least-squares (Tikhonov-regularized) pseudoinverse:
    // s / (s^2 + lambda^2) instead of 1/s — bounded even as s -> 0.
    Eigen::VectorXd sv_inv = svd.singularValues().unaryExpr(
        [damping](double s) { return s / (s*s + damping*damping); });

    return svd.matrixV() * sv_inv.asDiagonal() * svd.matrixU().transpose();
}

void FloatingBaseDynamics::checkNaN(bool hasNaN, const char* label)
{
    if (hasNaN) {
        throw std::runtime_error(std::string("NaN in ") + label);
    }
}

const Eigen::VectorXd& FloatingBaseDynamics::nle() const
{
    static thread_local Eigen::VectorXd bias;
    bias.resize(mj_->model->nv);
    for (int i = 0; i < mj_->model->nv; ++i) bias[i] = mj_->data->qfrc_bias[i];
    return bias;
}

}


