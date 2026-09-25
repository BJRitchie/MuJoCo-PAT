#include "pat_arm_nmpc/arm_task_space_controller.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pat_arm_nmpc {

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

// === Construction ============================================================
ArmTaskSpaceController::ArmTaskSpaceController(
    const std::string& mjcf_path)
    : mj_(std::make_unique<MjImpl>())
{
    char error[1000] = "";
    mj_->model = mj_loadXML(mjcf_path.c_str(), nullptr, error, sizeof(error));
    if (!mj_->model) {
        throw std::runtime_error(std::string("mj_loadXML failed: ") + error);
    }

    // TODO - should constrained joints be allowed 
    if (mj_->model->jnt_type[0] != mjJNT_FREE) {
        throw std::runtime_error(
            "ArmTaskSpaceController: root joint is not a free joint — "
            "check the MJCF/URDF root <joint type=\"floating\">.");
    }

    // (TODO - should this be disabled?) Disable gravity 
    mj_->model->opt.gravity[0] = 0;
    mj_->model->opt.gravity[1] = 0;
    mj_->model->opt.gravity[2] = 0;

    mj_->data = mj_makeData(mj_->model);
    n_joints_ = mj_->model->nv - 6;
}

ArmTaskSpaceController::~ArmTaskSpaceController() = default;

// === Live-state load (de-Basilisk'd UpdateState) =============================
void ArmTaskSpaceController::loadLiveState(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v)
{
    const int nq = mj_->model->nq;
    const int nv = mj_->model->nv;
    if (q.size() != nq || v.size() != nv) {
        throw std::runtime_error(
            "ArmTaskSpaceController::loadLiveState: q/v size mismatch — got q=" +
            std::to_string(q.size()) + " v=" + std::to_string(v.size()) +
            ", model needs nq=" + std::to_string(nq) + " nv=" + std::to_string(nv));
    }
    for (int i = 0; i < nq; ++i) mj_->data->qpos[i] = q[i];
    for (int i = 0; i < nv; ++i) mj_->data->qvel[i] = v[i];
    mj_forward(mj_->model, mj_->data);
}

// === Protected dynamics helpers =========================================================
int ArmTaskSpaceController::siteIdFromName(const std::string& name) const
{
    int id = mj_name2id(mj_->model, mjOBJ_SITE, name.c_str());
    if (id < 0) {
        throw std::runtime_error(
            "ArmTaskSpaceController: no <site> named \"" + name +
            "\" found in the MJCF. Check spelling and that the site is "
            "defined in the loaded model file.");
    }
    return id;
}

int ArmTaskSpaceController::jointIndexFromName(const std::string& name) const
{
    int jnt_id = mj_name2id(mj_->model, mjOBJ_JOINT, name.c_str());
    if (jnt_id < 0) {
        throw std::runtime_error(
            "ArmTaskSpaceController: no <joint> named \"" + name +
            "\" found in the MJCF. Check spelling and that the joint is "
            "defined in the loaded model file.");
    }
    int jnt_type = mj_->model->jnt_type[jnt_id];
    if (jnt_type != mjJNT_HINGE && jnt_type != mjJNT_SLIDE) {
        throw std::runtime_error(
            "ArmTaskSpaceController: joint \"" + name + "\" is not a "
            "single-DOF hinge/slide joint — cannot map it to an arm-local "
            "control index.");
    }
    // jnt_dofadr is the joint's offset into the FULL qvel vector (base free
    // joint occupies dofs [0:6)); subtract 6 to land in the same arm-local
    // index space as getJointPosInMsg(i)/q[6+i]/v[6+i].
    int idx = mj_->model->jnt_dofadr[jnt_id] - 6;
    if (idx < 0 || idx >= n_joints_) {
        throw std::runtime_error(
            "ArmTaskSpaceController: joint \"" + name + "\" resolved to "
            "arm-local index " + std::to_string(idx) + ", outside "
            "[0, n_joints_) — is it part of the free-floating base or a "
            "different kinematic branch?");
    }
    return idx;
}

Eigen::Vector3d ArmTaskSpaceController::eePosition(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const mjtNum* p = d->site_xpos + 3 * site_id;
    return Eigen::Vector3d(p[0], p[1], p[2]);
}

Eigen::Vector4d ArmTaskSpaceController::eeOrientation(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    mjtNum quat[4];
    mju_mat2Quat(quat, d->site_xmat + 9 * site_id);
    return Eigen::Vector4d(quat[0], quat[1], quat[2], quat[3]);
}

void ArmTaskSpaceController::getDynamics(
    Eigen::MatrixXd& H_g, Eigen::VectorXd& Cv_joints, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    Eigen::MatrixXd M(nv, nv);
    mj_fullM(mj_->model, M.data(), d->qM);

    Eigen::MatrixXd H_bb = M.topLeftCorner(6, 6);
    Eigen::MatrixXd H_bm = M.topRightCorner(6, n_joints_);
    Eigen::MatrixXd H_mm = M.bottomRightCorner(n_joints_, n_joints_);

    // Generalised inertia:  H_g = H_mm − H_bm^T H_bb^{−1} H_bm
    const auto H_bb_ldlt = H_bb.ldlt();
    H_g = H_mm - H_bm.transpose() * H_bb_ldlt.solve(H_bm);

    // Generalised bias, same base elimination as H_g (unactuated base):
    //   Cv_g = c_m − H_bm^T H_bb^{−1} c_b
    // The raw joint slice c_m alone is not the reduced-system bias -- it
    // omits the base Coriolis reaction, so the feedforward under-cancels and
    // H_g q̈ = J^T τ no longer holds (consistent with H_g, gjm6 and
    // jdotQdot6, which all apply this same elimination).
    Eigen::VectorXd c_b(6);
    for (int i = 0; i < 6; ++i) c_b[i] = d->qfrc_bias[i];
    Cv_joints.resize(n_joints_);
    for (int i = 0; i < n_joints_; ++i) {
        Cv_joints[i] = d->qfrc_bias[6 + i];
    }
    Cv_joints -= H_bm.transpose() * H_bb_ldlt.solve(c_b);
}

Eigen::Matrix3Xd ArmTaskSpaceController::gjm(int site_id, mjData* data) const
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

Eigen::MatrixXd ArmTaskSpaceController::gjm6(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    // Same row-major layout requirement as gjm() above.
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacr(3, nv);
    mj_jacSite(mj_->model, d, jacp.data(), jacr.data(), site_id);

    Eigen::MatrixXd J_full(6, nv);
    J_full.topRows(3)    = jacp;
    J_full.bottomRows(3) = jacr;

    Eigen::MatrixXd J_b = J_full.leftCols(6);
    Eigen::MatrixXd J_m = J_full.rightCols(n_joints_);

    Eigen::MatrixXd M(nv, nv);
    mj_fullM(mj_->model, M.data(), d->qM);

    // Same base-coupling correction as gjm(), applied to both the
    // translational and rotational rows at once — the correction
    // J_b H_bb^-1 H_bm is linear per-row, so stacking jacp/jacr first and
    // correcting once is equivalent to correcting each block separately.
    return J_m - J_b * M.topLeftCorner(6, 6).ldlt().solve(M.topRightCorner(6, n_joints_));
}

Eigen::Vector3d ArmTaskSpaceController::jdotQdot(int site_id, mjData* data) const
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

Eigen::Matrix<double, 6, 1> ArmTaskSpaceController::jdotQdot6(int site_id, mjData* data) const
{
    mjData* d = data ? data : mj_->data;
    const int nv = mj_->model->nv;
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp_dot(3, nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacr_dot(3, nv);
    mj_jacDot(mj_->model, d, jacp_dot.data(), jacr_dot.data(),
              d->site_xpos + 3 * site_id, mj_->model->site_bodyid[site_id]);

    Eigen::VectorXd qvel(nv);
    for (int i = 0; i < nv; ++i) qvel[i] = d->qvel[i];

    Eigen::Vector3d bias_lin = jacp_dot * qvel;
    Eigen::Vector3d bias_ang = jacr_dot * qvel;

    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv), jacr(3, nv);
    mj_jacSite(mj_->model, d, jacp.data(), jacr.data(), site_id);
    Eigen::MatrixXd J_0(6, 6);
    J_0.topRows(3)    = jacp.leftCols(6);
    J_0.bottomRows(3) = jacr.leftCols(6);

    Eigen::MatrixXd M(nv, nv);
    mj_fullM(mj_->model, M.data(), d->qM);

    Eigen::VectorXd Cv_base(6);
    for (int i = 0; i < 6; ++i) Cv_base[i] = d->qfrc_bias[i];

    Eigen::Matrix<double, 6, 1> bias;
    bias << bias_lin, bias_ang;

    return bias - J_0 * M.topLeftCorner(6, 6).ldlt().solve(Cv_base);
}

NodeDynamics ArmTaskSpaceController::dynamicsAt(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v, int site_id) const
{
    if (!mj_scratch_) {
        mj_scratch_ = std::make_unique<MjScratchData>();
        mj_scratch_->data = mj_makeData(mj_->model);
    }
    mjData* d = mj_scratch_->data;

    for (int i = 0; i < mj_->model->nq; ++i) d->qpos[i] = q[i];
    for (int i = 0; i < mj_->model->nv; ++i) d->qvel[i] = v[i];
    mj_forward(mj_->model, d);

    NodeDynamics nd;
    nd.ee_pos  = eePosition(site_id, d);
    nd.ee_quat = eeOrientation(site_id, d);
    getDynamics(nd.H_g, nd.Cv_joints, d);
    nd.J_g6       = gjm6(site_id, d);
    nd.jdot_qdot6 = jdotQdot6(site_id, d);
    return nd;
}

void ArmTaskSpaceController::integrateStep(
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
    Eigen::VectorXd v_mid(nv);
    for (int i = 0; i < nv; ++i) {
        v_next[i] = v[i] + Ts * d->qacc[i];
        v_mid[i]  = v[i] + 0.5 * Ts * d->qacc[i];
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
    mj_integratePos(mj_->model, q_next.data(), v_mid.data(), Ts);
}

Eigen::Vector3d ArmTaskSpaceController::sat(const Eigen::Vector3d& x)
{
    return x.cwiseMax(-1.0).cwiseMin(1.0);
}

Eigen::MatrixXd ArmTaskSpaceController::pinvDLS(
    const Eigen::MatrixXd& A, double damping, const std::string& name)
{
    // Thin, not Full: for a non-square A (m x n) the thin factors are
    // U: m x k, V: n x k, sigma: k  with k = min(m,n), so
    // V * diag(sigma_inv) * U^T is a well-formed n x m pseudo-inverse.
    // Full U/V make that product dimension-mismatch — it aborts under live
    // Eigen asserts and silently computes a wrong result under NDEBUG
    // (which is how it slipped through the Basilisk build).
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    double max_sv = svd.singularValues()(0);
    double min_sv = svd.singularValues()(svd.singularValues().size() - 1);

    // Damped least-squares (Tikhonov-regularized) pseudoinverse:
    // s / (s^2 + lambda^2) instead of 1/s — bounded even as s -> 0.
    Eigen::VectorXd sv_inv = svd.singularValues().unaryExpr(
        [damping](double s) { return s / (s*s + damping*damping); });

    double cond = max_sv / std::max(min_sv, 1e-12);
    if (cond > 1e4 && !name.empty()) {
        std::cout << "[WARN] " << name << " ill-conditioned, cond=" << cond
                   << ", sv=" << svd.singularValues().transpose() << "\n";
    }

    return svd.matrixV() * sv_inv.asDiagonal() * svd.matrixU().transpose();
}

void ArmTaskSpaceController::checkNaN(bool hasNaN, const char* label)
{
    if (hasNaN) {
        throw std::runtime_error(std::string("NaN in ") + label);
    }
}

const Eigen::VectorXd& ArmTaskSpaceController::nle() const
{
    static thread_local Eigen::VectorXd bias;
    bias.resize(mj_->model->nv);
    for (int i = 0; i < mj_->model->nv; ++i) bias[i] = mj_->data->qfrc_bias[i];
    return bias;
}

}


