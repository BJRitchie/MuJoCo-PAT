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
    chaser_body_ = m_->jnt_bodyid[jointId(m_, "chaser_x")];
    jacp_.resize(static_cast<std::size_t>(3 * m_->nv));
    jacr_.resize(static_cast<std::size_t>(3 * m_->nv));
    mv_.resize(static_cast<std::size_t>(m_->nv));

    // Damping is a model constant, so it is set once here rather than on every
    // reset. A negative value keeps whatever the MJCF ships; zero is frictionless.
    const auto damp = [this](int dof, double value) {
        if (value >= 0.0) m_->dof_damping[dof] = value;
    };
    damp(cxd_, cfg_.base_damping[0]);
    damp(cyd_, cfg_.base_damping[1]);
    damp(cwd_, cfg_.base_damping[2]);
    damp(m_->jnt_dofadr[jointId(m_, "target_x")],   cfg_.target_damping[0]);
    damp(m_->jnt_dofadr[jointId(m_, "target_y")],   cfg_.target_damping[1]);
    damp(m_->jnt_dofadr[jointId(m_, "target_yaw")], cfg_.target_damping[2]);

    applyMassScale();

    reset();
}

void ContactScenario::applyMassScale() {
    if (cfg_.arm_mass_scale == 1.0) return;
    if (!(cfg_.arm_mass_scale > 0.0)) {
        throw std::runtime_error("ContactScenario: arm_mass_scale must be "
                                 "positive");
    }
    const double s = cfg_.arm_mass_scale;
    // The bodies to perturb are exactly the ones the model joints move, taken
    // from jnt_bodyid rather than from a name list, so adding an arm segment to
    // the MJCF needs no change here. Mass and rotational inertia scale together
    // because a denser link of the same geometry is what is being modelled; the
    // centre of mass does not move.
    for (const auto& jn : cfg_.model_joints) {
        const int bid = m_->jnt_bodyid[jointId(m_, jn)];
        m_->body_mass[bid] *= s;
        for (int k = 0; k < 3; ++k) m_->body_inertia[3 * bid + k] *= s;
    }
    // body_mass and body_inertia feed derived constants, subtree masses among
    // them, so those have to be rebuilt or the model is internally
    // inconsistent. reset() then runs mj_forward to put mjData back in step.
    mj_setConst(m_, d_);
}

double ContactScenario::armMass() const {
    double total = 0.0;
    for (const auto& jn : cfg_.model_joints) {
        total += m_->body_mass[m_->jnt_bodyid[jointId(m_, jn)]];
    }
    return total;
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

void ContactScenario::setJointDisturbance(int model_joint, double torque) {
    if (model_joint < 0 || model_joint >= static_cast<int>(joints_.size())) {
        throw std::runtime_error("setJointDisturbance: no such model joint");
    }
    d_->qfrc_applied[joints_[static_cast<std::size_t>(model_joint)].dof] = torque;
}

void ContactScenario::step(int n_steps) {
    for (int i = 0; i < n_steps; ++i) mj_step(m_, d_);
}

void ContactScenario::refreshDerived() { mj_forward(m_, d_); }

void ContactScenario::readImu(double& ax, double& ay, double& gz) {
    const auto adr = [this](const char* name) {
        const int id = mj_name2id(m_, mjOBJ_SENSOR, name);
        if (id < 0) {
            throw std::runtime_error(std::string("the scene has no sensor '") + name + "'");
        }
        return m_->sensor_adr[id];
    };
    mj_forward(m_, d_);
    const int a = adr("chaser_accel"), g = adr("chaser_gyro");
    ax = d_->sensordata[a];
    ay = d_->sensordata[a + 1];
    gz = d_->sensordata[g + 2];
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

void ContactScenario::readControllerAccel(Eigen::VectorXd& a) const {
    const auto n = static_cast<Eigen::Index>(joints_.size());
    if (a.size() != 6 + n) a.resize(6 + n);
    a.setZero();
    a(0) = d_->qacc[cxd_];
    a(1) = d_->qacc[cyd_];
    a(5) = d_->qacc[cwd_];
    for (Eigen::Index i = 0; i < n; ++i) {
        a(6 + i) = d_->qacc[joints_[static_cast<std::size_t>(i)].dof];
    }
}

std::array<double, 3> ContactScenario::baseVelocity() const {
    return {{d_->qvel[cxd_], d_->qvel[cyd_], d_->qvel[cwd_]}};
}

std::array<double, 3> ContactScenario::basePose() const {
    return {{d_->qpos[cx_], d_->qpos[cy_], d_->qpos[cw_]}};
}

void ContactScenario::chaserMomentum(PlanarMomentum& system, PlanarMomentum& bus) const {
    const int b = chaser_body_;
    // subtree_linvel is the subtree's centre-of-mass velocity and
    // subtree_angmom its angular momentum about that centre of mass.
    mj_subtreeVel(m_, d_);
    const double m_sys = m_->body_subtreemass[b];
    system.px = m_sys * d_->subtree_linvel[3 * b];
    system.py = m_sys * d_->subtree_linvel[3 * b + 1];
    system.lz = d_->subtree_angmom[3 * b + 2];

    // The bus body alone: its own centre-of-mass velocity and spin, moved to
    // the chaser's centre of mass.
    mj_jacBodyCom(m_, d_, jacp_.data(), jacr_.data(), b);
    double vx = 0.0, vy = 0.0, wz = 0.0;
    for (int i = 0; i < m_->nv; ++i) {
        vx += jacp_[static_cast<std::size_t>(i)] * d_->qvel[i];
        vy += jacp_[static_cast<std::size_t>(m_->nv + i)] * d_->qvel[i];
        wz += jacr_[static_cast<std::size_t>(2 * m_->nv + i)] * d_->qvel[i];
    }
    const double m_bus = m_->body_mass[b];
    // World-frame Izz from the principal inertia: sum of I_k * R(2,k)^2, with
    // ximat the row-major rotation of the inertial frame.
    double izz = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double r2k = d_->ximat[9 * b + 6 + k];
        izz += m_->body_inertia[3 * b + k] * r2k * r2k;
    }
    const double rx = d_->xipos[3 * b]     - d_->subtree_com[3 * b];
    const double ry = d_->xipos[3 * b + 1] - d_->subtree_com[3 * b + 1];
    bus.px = m_bus * vx;
    bus.py = m_bus * vy;
    bus.lz = izz * wz + m_bus * (rx * vy - ry * vx);
}

double ContactScenario::chaserKineticEnergy() const {
    // M is block-diagonal between the chaser and the target (separate trees),
    // so summing over the chaser's DOFs alone leaves the target out exactly.
    mj_mulM(m_, d_, mv_.data(), d_->qvel);
    double ke = 0.0;
    for (int i = 0; i < m_->nv; ++i) {
        if (m_->body_rootid[m_->dof_bodyid[i]] == chaser_body_) {
            ke += d_->qvel[i] * mv_[static_cast<std::size_t>(i)];
        }
    }
    return 0.5 * ke;
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

/*! Below this a truth signal is numerical dust, not a disturbance.
 *
 *  A bare "!= 0.0" guard is not enough: a baseline run with a correct model and
 *  nothing touching leaves a truth of order 1e-14, and dividing by it produced
 *  ratios of 5e13 that look like catastrophic failure rather than like the
 *  absence of a signal. Far below any disturbance worth estimating -- the
 *  observer's own measured error floor is 8e-5 -- so a real signal is never
 *  suppressed by it. */
constexpr double kSignalFloor = 1e-6;

double JointScore::ratio() const {
    return std::fabs(truth_peak) > kSignalFloor ? est_peak / truth_peak
                                                : std::nan("");
}

double JointScore::amplitudeRatio() const {
    return truth_rms > kSignalFloor ? est_rms / truth_rms : std::nan("");
}

namespace {

/*! Peak, mean, correlation and RMS error of a paired truth/estimate series.
 *
 *  Shared by both scorers so the two windows cannot drift in how they compute
 *  the same statistics; only the choice of which ticks to include differs. */
JointScore statsOf(const std::vector<double>& tr, const std::vector<double>& es) {
    JointScore s;
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

    double st = 0.0, se = 0.0, cov = 0.0, sq = 0.0, mt = 0.0, me = 0.0;
    for (std::size_t i = 0; i < tr.size(); ++i) {
        const double a = tr[i] - s.truth_mean;
        const double b = es[i] - s.est_mean;
        st  += a * a;
        se  += b * b;
        cov += a * b;
        sq  += (tr[i] - es[i]) * (tr[i] - es[i]);
        mt  += tr[i] * tr[i];
        me  += es[i] * es[i];
    }
    s.corr = (st > 1e-18 && se > 1e-18) ? cov / std::sqrt(st * se) : std::nan("");
    s.rms_err   = std::sqrt(sq / n);
    s.truth_rms = std::sqrt(mt / n);
    s.est_rms   = std::sqrt(me / n);
    return s;
}

}  // namespace

JointScore scoreJoint(const std::vector<ContactRow>& rows, int joint) {
    const auto j = static_cast<std::size_t>(joint);
    std::vector<double> tr, es;
    for (const auto& r : rows) {
        if (r.ncon <= 0) continue;            // quiet ticks are not the contact
        if (j >= r.truth.size()) continue;
        tr.push_back(r.truth[j]);
        es.push_back(r.raw[j]);
    }
    return statsOf(tr, es);
}

JointScore scoreJointResidual(const std::vector<ContactRow>& rows, int joint,
                              std::size_t from) {
    const auto j = static_cast<std::size_t>(joint);
    std::vector<double> tr, es;
    for (std::size_t i = from; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (j >= r.resid.size()) continue;
        tr.push_back(r.resid[j]);
        es.push_back(r.raw[j]);
    }
    return statsOf(tr, es);
}

}  // namespace bench
}  // namespace pat_arm_control
