#include "contact_scenario.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <mujoco/mujoco.h>

namespace pat_arm_control {
namespace bench {
namespace {

int jointId(const mjModel* m, const std::string& n) {
    const int id = mj_name2id(m, mjOBJ_JOINT, n.c_str());
    if (id < 0) throw std::runtime_error("scene has no joint '" + n + "'");
    return id;
}

int actuatorId(const mjModel* m, const std::string& n) {
    const int id = mj_name2id(m, mjOBJ_ACTUATOR, n.c_str());
    if (id < 0) throw std::runtime_error("scene has no actuator '" + n + "'");
    return id;
}

/*! joint2_L -> motor2_L, the MJCF's own pairing convention: every arm joint has
 *  a <motor> of the same number and side. Derived rather than configured so the
 *  two lists cannot fall out of step. */
std::string motorFor(const std::string& joint) {
    const auto pos = joint.find("joint");
    if (pos == std::string::npos) {
        throw std::runtime_error("'" + joint + "' is not joint<N>_<side>, so its "
                                 "paired motor cannot be derived");
    }
    std::string m = joint;
    m.replace(pos, 5, "motor");
    return m;
}

double peakOf(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return *std::max_element(v.begin(), v.end(), [](double a, double b) {
        return std::fabs(a) < std::fabs(b);
    });
}

}  // namespace

ContactScenario::ContactScenario(const Config& cfg) : cfg_(cfg) {
    if (cfg_.model_joints.empty()) {
        throw std::runtime_error("ContactScenario: model_joints is required");
    }
    if (cfg_.arm_home.size() != cfg_.model_joints.size()) {
        throw std::runtime_error("ContactScenario: arm_home needs one entry per "
                                 "model joint");
    }

    char err[1000] = "";
    m_ = mj_loadXML(cfg_.scene_mjcf.c_str(), nullptr, err, sizeof(err));
    if (!m_) throw std::runtime_error(std::string("mj_loadXML: ") + err);
    d_ = mj_makeData(m_);

    joints_.reserve(cfg_.model_joints.size());
    for (const auto& jn : cfg_.model_joints) {
        const int jid = jointId(m_, jn);
        JointRef r;
        r.qpos = m_->jnt_qposadr[jid];
        r.dof  = m_->jnt_dofadr[jid];
        r.act  = actuatorId(m_, motorFor(jn));
        joints_.push_back(r);
    }
    cx_  = m_->jnt_qposadr[jointId(m_, "chaser_x")];
    cy_  = m_->jnt_qposadr[jointId(m_, "chaser_y")];
    cw_  = m_->jnt_qposadr[jointId(m_, "chaser_yaw")];
    cxd_ = m_->jnt_dofadr[jointId(m_, "chaser_x")];
    cyd_ = m_->jnt_dofadr[jointId(m_, "chaser_y")];
    cwd_ = m_->jnt_dofadr[jointId(m_, "chaser_yaw")];
    tx_  = m_->jnt_qposadr[jointId(m_, "target_x")];
    ty_  = m_->jnt_qposadr[jointId(m_, "target_y")];

    // Damping is a model constant, so it is set once here rather than on every
    // reset. Zero leaves the body free at whatever the MJCF ships.
    const auto damp = [this](int dof, double value) {
        if (value > 0.0) m_->dof_damping[dof] = value;
    };
    damp(cxd_, cfg_.base_damping[0]);
    damp(cyd_, cfg_.base_damping[1]);
    damp(cwd_, cfg_.base_damping[2]);
    damp(m_->jnt_dofadr[jointId(m_, "target_x")],   cfg_.target_damping[0]);
    damp(m_->jnt_dofadr[jointId(m_, "target_y")],   cfg_.target_damping[1]);
    damp(m_->jnt_dofadr[jointId(m_, "target_yaw")], cfg_.target_damping[2]);

    reset();
}

ContactScenario::~ContactScenario() {
    if (d_) mj_deleteData(d_);
    if (m_) mj_deleteModel(m_);
}

void ContactScenario::reset() {
    mj_resetData(m_, d_);
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        d_->qpos[joints_[i].qpos] = cfg_.arm_home[i];
    }
    d_->qpos[tx_] = cfg_.target_x;
    d_->qpos[ty_] = cfg_.target_y;
    // So every derived quantity, contacts included, matches the pose that was
    // just written rather than the one mj_resetData left behind.
    mj_forward(m_, d_);
}

void ContactScenario::applyArmTorque(const Eigen::VectorXd& tau_joints) {
    if (tau_joints.size() != static_cast<Eigen::Index>(joints_.size())) {
        throw std::runtime_error("applyArmTorque: expected one torque per model "
                                 "joint");
    }
    for (std::size_t i = 0; i < joints_.size(); ++i) {
        d_->ctrl[joints_[i].act] = tau_joints(static_cast<Eigen::Index>(i));
    }
}

void ContactScenario::step(int n_steps) {
    for (int i = 0; i < n_steps; ++i) mj_step(m_, d_);
}

double ContactScenario::time() const { return d_->time; }
double ContactScenario::timestep() const { return m_->opt.timestep; }
int    ContactScenario::contactCount() const { return d_->ncon; }

std::string ContactScenario::contactPairs() const {
    std::string out;
    for (int i = 0; i < d_->ncon; ++i) {
        const auto name = [this](int gid) -> std::string {
            const char* n = mj_id2name(m_, mjOBJ_GEOM, gid);
            return n ? n : ("geom" + std::to_string(gid));
        };
        if (!out.empty()) out += ", ";
        out += name(d_->contact[i].geom1) + "--" + name(d_->contact[i].geom2);
    }
    return out;
}

void ContactScenario::armConstraintForce(Eigen::VectorXd& out) const {
    const auto n = static_cast<Eigen::Index>(joints_.size());
    if (out.size() != n) out.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        out(i) = d_->qfrc_constraint[joints_[static_cast<std::size_t>(i)].dof];
    }
}

void ContactScenario::readControllerState(double anchor_x, double anchor_y,
                                          Eigen::VectorXd& q,
                                          Eigen::VectorXd& v) const {
    const auto n = static_cast<Eigen::Index>(joints_.size());
    if (q.size() != 7 + n) q.resize(7 + n);
    if (v.size() != 6 + n) v.resize(6 + n);

    // The controller's base is a freejoint, so its first seven q entries are a
    // position and a quaternion where the scene has three planar joint values.
    // Only yaw is populated: the table is planar, and z, roll and pitch are
    // identically zero in both models.
    q(0) = d_->qpos[cx_] + anchor_x;
    q(1) = d_->qpos[cy_] + anchor_y;
    q(2) = 0.0;
    const double yaw = d_->qpos[cw_];
    q(3) = std::cos(yaw / 2.0);
    q(4) = 0.0;
    q(5) = 0.0;
    q(6) = std::sin(yaw / 2.0);

    v(0) = d_->qvel[cxd_];
    v(1) = d_->qvel[cyd_];
    v(2) = 0.0;
    v(3) = 0.0;
    v(4) = 0.0;
    v(5) = d_->qvel[cwd_];

    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& r = joints_[static_cast<std::size_t>(i)];
        q(7 + i) = d_->qpos[r.qpos];
        v(6 + i) = d_->qvel[r.dof];
    }
}

double JointScore::ratio() const {
    return truth_peak != 0.0 ? est_peak / truth_peak : std::nan("");
}

JointScore scoreJoint(const std::vector<ContactRow>& rows, int joint) {
    JointScore s;
    const auto j = static_cast<std::size_t>(joint);
    std::vector<double> tr, es;
    for (const auto& r : rows) {
        if (r.ncon <= 0) continue;            // quiet ticks are not the contact
        if (j >= r.truth.size()) continue;
        tr.push_back(r.truth[j]);
        es.push_back(r.raw[j]);
    }
    s.n_contact = static_cast<int>(tr.size());
    if (tr.empty()) return s;

    s.truth_peak = peakOf(tr);
    s.est_peak   = peakOf(es);

    const double n = static_cast<double>(tr.size());
    for (std::size_t i = 0; i < tr.size(); ++i) {
        s.truth_mean += tr[i];
        s.est_mean   += es[i];
    }
    s.truth_mean /= n;
    s.est_mean   /= n;

    double st = 0.0, se = 0.0, cov = 0.0, sq = 0.0;
    for (std::size_t i = 0; i < tr.size(); ++i) {
        const double a = tr[i] - s.truth_mean;
        const double b = es[i] - s.est_mean;
        st  += a * a;
        se  += b * b;
        cov += a * b;
        sq  += (tr[i] - es[i]) * (tr[i] - es[i]);
    }
    s.corr = (st > 1e-18 && se > 1e-18) ? cov / std::sqrt(st * se) : std::nan("");
    s.rms_err = std::sqrt(sq / n);
    return s;
}

}  // namespace bench
}  // namespace pat_arm_control
