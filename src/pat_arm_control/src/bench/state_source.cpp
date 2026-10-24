#include "state_source.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

#include "bench_yaml.h"
#include "contact_scenario.h"

namespace pat_arm_control {
namespace bench {

void NoiseProfile::set(const std::string& name, double value) {
    if      (name == "joint_pos") joint_pos = value;
    else if (name == "joint_vel") joint_vel = value;
    else if (name == "base_pos")  base_pos  = value;
    else if (name == "base_yaw")  base_yaw  = value;
    else if (name == "base_vel")  base_vel  = value;
    else if (name == "base_wz")   base_wz   = value;
    else throw std::runtime_error(
        "unknown noise figure '" + name +
        "' (wanted joint_pos, joint_vel, base_pos, base_yaw, base_vel or base_wz)");
}

namespace {

// ===================== The estimator's configuration ====================== //
// nav_bench builds the same objects from the same files; its loaders are private
// to that executable, so these are the same lookups for this one. Every key the
// loaders do not know is a hard error, so a knob added to nav.yaml says so.

double num(const Flat& f, const std::string& k, double def) {
    const auto it = f.find(k);
    return it == f.end() ? def : it->second.as<double>();
}

//! One node's ros__parameters block, flattened and merged over `into`.
void loadNodeParams(const std::string& path, const std::string& node, Flat& into) {
    const YAML::Node root = YAML::LoadFile(path);
    if (!root[node] || !root[node]["ros__parameters"]) {
        throw std::runtime_error(path + " has no " + node + ".ros__parameters block");
    }
    flatten(root[node]["ros__parameters"], "", into);
}

//! A sensor channel's eight fields, defaulting to perfect.
pat_sensors::ChannelConfig channelFrom(const Flat& f, const std::string& name) {
    const std::string p = "sensors." + name + ".";
    pat_sensors::ChannelConfig c;
    c.noise_std     = num(f, p + "noise_std", 0.0);
    c.bias          = num(f, p + "bias", 0.0);
    c.bias_walk_std = num(f, p + "bias_walk_std", 0.0);
    c.scale_error   = num(f, p + "scale_error", 0.0);
    c.quantisation  = num(f, p + "quantisation", 0.0);
    c.delay_s       = num(f, p + "delay_s", 0.0);
    c.dropout_prob  = num(f, p + "dropout_prob", 0.0);
    c.rate_hz       = num(f, p + "rate_hz", 0.0);
    return c;
}

pat_gnc::ImuEkfNavigator::Config navConfigFrom(const Flat& f) {
    pat_gnc::ImuEkfNavigator::Config c;
    pat_gnc::ImuEkf::Params& e = c.ekf;
    std::set<std::string> used;
    const auto d = [&](const std::string& k, double def) {
        used.insert("imu_ekf." + k);
        return num(f, "imu_ekf." + k, def);
    };
    const auto b = [&](const std::string& k, bool def) {
        used.insert("imu_ekf." + k);
        const auto it = f.find("imu_ekf." + k);
        return it == f.end() ? def : it->second.as<bool>();
    };
    e.gyro_noise_std      = d("gyro_noise_std", e.gyro_noise_std);
    e.accel_noise_std     = d("accel_noise_std", e.accel_noise_std);
    e.gyro_bias_walk      = d("gyro_bias_walk", e.gyro_bias_walk);
    e.accel_bias_walk     = d("accel_bias_walk", e.accel_bias_walk);
    e.yaw_meas_std        = d("yaw_meas_std", e.yaw_meas_std);
    e.pos_meas_std        = d("pos_meas_std", e.pos_meas_std);
    e.vel_meas_std        = d("vel_meas_std", e.vel_meas_std);
    e.init_pos_std        = d("init_pos_std", e.init_pos_std);
    e.init_yaw_std        = d("init_yaw_std", e.init_yaw_std);
    e.init_vel_std        = d("init_vel_std", e.init_vel_std);
    e.init_gyro_bias_std  = d("init_gyro_bias_std", e.init_gyro_bias_std);
    e.init_accel_bias_std = d("init_accel_bias_std", e.init_accel_bias_std);
    e.max_imu_gap         = d("max_imu_gap", e.max_imu_gap);
    e.gate_yaw            = d("gate_yaw", e.gate_yaw);
    e.gate_pos            = d("gate_pos", e.gate_pos);
    e.gate_vel            = d("gate_vel", e.gate_vel);
    e.history_len = static_cast<std::size_t>(d("history_len", static_cast<double>(e.history_len)));
    c.latency_yaw   = d("latency_yaw", c.latency_yaw);
    c.latency_pos   = d("latency_pos", c.latency_pos);
    c.latency_vel   = d("latency_vel", c.latency_vel);
    c.use_twist     = b("use_twist", c.use_twist);
    c.skip_repeated = b("skip_repeated", c.skip_repeated);
    c.use_position  = b("use_position", c.use_position);
    for (const auto& kv : f) {
        if (kv.first.rfind("imu_ekf.", 0) == 0 && !used.count(kv.first)) {
            throw std::runtime_error("the bench does not know '" + kv.first +
                                     "' -- teach navConfigFrom() it");
        }
    }
    return c;
}

//! A flight sensor channel with its error (noise, bias, bias walk) scaled.
pat_sensors::ChannelConfig scaled(pat_sensors::ChannelConfig c, double s) {
    c.noise_std *= s;
    c.bias *= s;
    c.bias_walk_std *= s;
    return c;
}

}  // namespace

struct StateSource::Impl {
    // Joint encoders (every non-exact source) and, for `noisy`, the base sensors.
    pat_sensors::SensorModel meas;
    std::vector<int> h_jpos, h_jvel;
    int h_bx = -1, h_by = -1, h_byaw = -1, h_bvx = -1, h_bvy = -1, h_bwz = -1;

    // The joint readings of one tick, and the filter they may go through.
    Eigen::VectorXd jq, jv;
    std::unique_ptr<JointKalmanFilter> jkf;
    double control_dt = 0.0;

    // The flight instruments (ekf, raw) and the estimator they feed (ekf).
    std::unique_ptr<pat_gnc::ImuEkfNavigator> nav;
    pat_sensors::SensorModel imu_model;
    int h_acc = -1, h_gyro = -1, h_nav = -1;
    int imu_every = 1, pub_every = 1;

    // raw: the flight readings as they last arrived, and the previous distinct
    // position fix, for the velocity.
    double raw_x = 0.0, raw_y = 0.0, raw_yaw = 0.0, raw_wz = 0.0;
    double raw_vx = 0.0, raw_vy = 0.0;
    bool have_fix = false;
    double fix_t = 0.0, fix_x = 0.0, fix_y = 0.0;

    // ekf / raw with a delay: the base state handed over, delay_ticks deep.
    std::vector<std::array<double, 6>> ring;
    std::size_t ring_next = 0;
    bool ring_primed = false;
};

StateSource::StateSource(const StateSourceConfig& cfg, int n_joints, double control_dt,
                         double physics_dt)
    : cfg_(cfg), impl_(std::make_unique<Impl>()) {
    if (cfg_.kind == StateSourceKind::Exact) return;
    if (cfg_.noise_scale < 0.0 || cfg_.delay_ticks < 0) {
        throw std::runtime_error("the noise scale and the delay cannot be negative");
    }
    Impl& m = *impl_;

    // Joint encoders and, for `noisy`, the base sensors: one channel each,
    // advanced once per control tick, so the delay is in control ticks.
    m.meas = pat_sensors::SensorModel(control_dt, cfg_.seed);
    const double s = cfg_.noise_scale;
    const auto channel = [&](const std::string& name, double sigma) {
        pat_sensors::ChannelConfig c;
        c.noise_std = sigma * s;
        c.delay_s   = cfg_.delay_ticks * control_dt;
        return m.meas.addChannel(name, c, control_dt);
    };
    for (int i = 0; i < n_joints; ++i) {
        m.h_jpos.push_back(channel("joint_pos_" + std::to_string(i), cfg_.profile.joint_pos));
        m.h_jvel.push_back(channel("joint_vel_" + std::to_string(i), cfg_.profile.joint_vel));
    }
    m.control_dt = control_dt;
    m.jq = Eigen::VectorXd::Zero(n_joints);
    m.jv = Eigen::VectorXd::Zero(n_joints);
    if (cfg_.joint_filter) {
        if (!(s > 0.0)) {
            throw std::runtime_error("the joint filter needs encoder noise to weigh "
                                     "(a noise scale above 0)");
        }
        // Told the encoders' true sigmas, as the base estimator is told its sensors'.
        JointKalmanParams jp = cfg_.joint_kf;
        jp.pos_meas_std = cfg_.profile.joint_pos * s;
        jp.vel_meas_std = cfg_.profile.joint_vel * s;
        m.jkf = std::make_unique<JointKalmanFilter>(jp, n_joints);
    }
    if (cfg_.kind == StateSourceKind::Noisy) {
        m.h_bx   = channel("base_x",   cfg_.profile.base_pos);
        m.h_by   = channel("base_y",   cfg_.profile.base_pos);
        m.h_byaw = channel("base_yaw", cfg_.profile.base_yaw);
        m.h_bvx  = channel("base_vx",  cfg_.profile.base_vel);
        m.h_bvy  = channel("base_vy",  cfg_.profile.base_vel);
        m.h_bwz  = channel("base_wz",  cfg_.profile.base_wz);
        return;
    }

    // ---- ekf / raw: the flight instruments ----------------------------------
    if (cfg_.sim_yaml.empty() || cfg_.sensors_yaml.empty() || cfg_.nav_yaml.empty()) {
        throw std::runtime_error(
            "the flight state sources need the simulator's params, the flight sensor "
            "overlay and the estimator's params (not found on AMENT_PREFIX_PATH?)");
    }
    if (cfg_.kind == StateSourceKind::Ekf && !(s > 0.0)) {
        throw std::runtime_error("the ekf state source needs sensor noise to weigh "
                                 "(a noise scale above 0)");
    }
    Flat sim, nav;
    loadNodeParams(cfg_.sim_yaml, "pat_simulation", sim);
    loadNodeParams(cfg_.sensors_yaml, "pat_simulation", sim);
    loadNodeParams(cfg_.nav_yaml, "pat_nav", nav);

    const double sim_hz = 1.0 / physics_dt;
    const double pub_hz = num(sim, "pub_rate_hz", 100.0);
    const double imu_hz = num(sim, "imu_rate_hz", pub_hz);
    m.imu_every = std::max(1, static_cast<int>(std::lround(sim_hz / imu_hz)));
    m.pub_every = std::max(1, static_cast<int>(std::lround(sim_hz / pub_hz)));

    // The node's sensor channels: the same names (so the same random stream for a
    // given seed), the periods they are actually advanced at, and the same
    // consecutive groups, as nav_bench builds them. Every error is scaled by the
    // noise scale.
    m.imu_model = pat_sensors::SensorModel(physics_dt, cfg_.seed);
    const double imu_dt = m.imu_every * physics_dt;
    const double pub_dt = 1.0 / pub_hz;
    const auto flight = [&](const std::string& name, double dt) {
        return m.imu_model.addChannel(name, scaled(channelFrom(sim, name), s), dt);
    };
    m.h_acc = flight("imu_accel_x", imu_dt);
    flight("imu_accel_y", imu_dt);
    m.h_gyro = flight("imu_gyro_z", imu_dt);
    m.h_nav = flight("nav_x", pub_dt);
    flight("nav_y", pub_dt);
    flight("star_yaw", pub_dt);

    if (cfg_.delay_ticks > 0) {
        m.ring.assign(static_cast<std::size_t>(cfg_.delay_ticks) + 1, {});
    }
    if (cfg_.kind != StateSourceKind::Ekf) return;

    // The estimator, told the same scaled errors its sensors have.
    auto nc = navConfigFrom(nav);
    pat_gnc::ImuEkf::Params& e = nc.ekf;
    for (double* f : {&e.gyro_noise_std, &e.accel_noise_std, &e.gyro_bias_walk,
                      &e.accel_bias_walk, &e.yaw_meas_std, &e.pos_meas_std,
                      &e.vel_meas_std, &e.init_gyro_bias_std, &e.init_accel_bias_std}) {
        *f *= s;
    }
    m.nav = std::make_unique<pat_gnc::ImuEkfNavigator>(nc);
}

StateSource::~StateSource() = default;

bool StateSource::ready() const {
    return cfg_.kind != StateSourceKind::Ekf || impl_->nav->filter().ready();
}

void StateSource::onSubstep(ContactScenario& scene, long k) {
    if (!isFlight()) return;
    Impl& m = *impl_;
    const double t = scene.time();

    // The IMU, at its own rate, with the odometry of the same instant fed after
    // it: the order nav_node's timestamp merge produces.
    if (k % m.imu_every == 0) {
        double a[2], g;
        scene.readImu(a[0], a[1], g);
        double a_out[2];
        m.imu_model.applyVector(m.h_acc, a, a_out, 2);
        const double g_out = m.imu_model.apply(m.h_gyro, g);
        if (m.nav) m.nav->onImu(t, {a_out[0], a_out[1]}, g_out);
        m.raw_wz = g_out;
    }
    if (k % m.pub_every == 0) {
        const auto b = scene.basePose();
        const double pose_truth[3] = {b[0], b[1], b[2]};
        double pose[3];
        m.imu_model.applyVector(m.h_nav, pose_truth, pose, 3);
        if (m.nav) {
            m.nav->onOdometry(t, {pose[0], pose[1], pose[2]}, {0.0, 0.0, 0.0});
            return;
        }
        // raw: take the readings as they arrive. A 2 Hz fix is
        // republished on every odometry tick, so a new fix is one whose value
        // changed; the velocity is the difference between the last two, over the
        // time between their arrivals.
        m.raw_x = pose[0];
        m.raw_y = pose[1];
        m.raw_yaw = pose[2];
        if (!m.have_fix) {
            m.have_fix = true;
        } else if (pose[0] != m.fix_x || pose[1] != m.fix_y) {
            const double dt = t - m.fix_t;
            m.raw_vx = (pose[0] - m.fix_x) / dt;
            m.raw_vy = (pose[1] - m.fix_y) / dt;
        } else {
            return;
        }
        m.fix_t = t;
        m.fix_x = pose[0];
        m.fix_y = pose[1];
    }
}

void StateSource::measure(const Eigen::VectorXd& q_true, const Eigen::VectorXd& v_true,
                          double anchor_x, double anchor_y, Eigen::VectorXd& q,
                          Eigen::VectorXd& v) {
    q = q_true;
    v = v_true;
    err_ = BaseError{};
    if (cfg_.kind == StateSourceKind::Exact) return;
    Impl& m = *impl_;

    // Joints: encoders, in every non-exact source, optionally filtered.
    for (std::size_t i = 0; i < m.h_jpos.size(); ++i) {
        const auto ii = static_cast<Eigen::Index>(i);
        m.jq(ii) = m.meas.apply(m.h_jpos[i], q_true(7 + ii));
        m.jv(ii) = m.meas.apply(m.h_jvel[i], v_true(6 + ii));
    }
    if (m.jkf) {
        m.jkf->update(m.control_dt, m.jq, m.jv);
        for (int i = 0; i < m.jkf->size(); ++i) {
            m.jq(i) = m.jkf->position(i);
            m.jv(i) = m.jkf->velocity(i);
        }
    }
    const auto nj = static_cast<Eigen::Index>(m.h_jpos.size());
    q.segment(7, nj) = m.jq;
    v.segment(6, nj) = m.jv;

    const double yaw_true = 2.0 * std::atan2(q_true(6), q_true(3));
    double x, y, yaw, vx, vy, wz;
    if (cfg_.kind == StateSourceKind::Noisy) {
        x   = m.meas.apply(m.h_bx, q_true(0));
        y   = m.meas.apply(m.h_by, q_true(1));
        yaw = m.meas.apply(m.h_byaw, yaw_true);
        vx  = m.meas.apply(m.h_bvx, v_true(0));
        vy  = m.meas.apply(m.h_bvy, v_true(1));
        wz  = m.meas.apply(m.h_bwz, v_true(5));
    } else {
        std::array<double, 6> b;
        if (cfg_.kind == StateSourceKind::Raw) {
            b = {m.raw_x + anchor_x, m.raw_y + anchor_y, m.raw_yaw,
                 m.raw_vx, m.raw_vy, m.raw_wz};
        } else if (m.nav && m.nav->ready()) {
            const auto s = m.nav->state();
            b = {s(0) + anchor_x, s(1) + anchor_y, s(2), s(3), s(4), s(5)};
        } else {
            // Until the filter has taken its first fix there is no estimate to
            // hand over, and the node would not be commanding yet either: the
            // exact state stands in, so the run starts from a known pose.
            b = {q_true(0), q_true(1), yaw_true, v_true(0), v_true(1), v_true(5)};
        }
        // A delay of N ticks hands over what was measured N ticks ago; the
        // ring starts full of the first state, as a sensor's delay line does.
        if (!m.ring.empty()) {
            if (!m.ring_primed) {
                std::fill(m.ring.begin(), m.ring.end(), b);
                m.ring_primed = true;
            }
            m.ring[m.ring_next] = b;
            m.ring_next = (m.ring_next + 1) % m.ring.size();
            b = m.ring[m.ring_next];
        }
        const bool estimating = (m.nav && m.nav->ready()) || cfg_.kind == StateSourceKind::Raw;
        if (!estimating && m.ring.empty()) return;
        x = b[0];
        y = b[1];
        yaw = b[2];
        vx = b[3];
        vy = b[4];
        wz = b[5];
    }
    q(0) = x;
    q(1) = y;
    q(3) = std::cos(yaw / 2.0);
    q(6) = std::sin(yaw / 2.0);
    v(0) = vx;
    v(1) = vy;
    v(5) = wz;

    err_.x   = x - q_true(0);
    err_.y   = y - q_true(1);
    err_.yaw = std::remainder(yaw - yaw_true, 2.0 * M_PI);
    err_.vx  = vx - v_true(0);
    err_.vy  = vy - v_true(1);
    err_.wz  = wz - v_true(5);
}

}  // namespace bench
}  // namespace pat_arm_control
