/*! nav_bench -- scores pat_gnc's base state estimator against MuJoCo truth,
 *  deterministically.
 *
 *  ============================ WHAT IT DOES ============================
 *
 *  Runs the scene MJCF as the plant, corrupts its IMU, star-tracker and
 *  relative-nav readings through the SAME SensorModel the simulator node uses,
 *  feeds them to the SAME ImuEkfNavigator nav_node runs, and compares the
 *  estimate with the plant's true state at every odometry tick. The truth
 *  includes the sensor biases actually in force, which only the simulator
 *  knows, so the bias estimates are scored too.
 *
 *  ========================= WHY IT IS OFFLINE ==========================
 *
 *  Same reason as contact_bench and nmpc_bench: one process, no ROS, no DDS,
 *  no wall clock, and every random draw from a fixed seed. Two runs of one
 *  configuration produce identical output, so a change in a number is a
 *  change in the filter or the configuration, never in the run. That is also
 *  what makes the Monte-Carlo consistency test meaningful: each seed is a
 *  reproducible, independent realisation of the same sensors.
 *
 *  ========================== HOW IT IS WIRED ===========================
 *
 *    configuration  simulation.yaml, then a sensor overlay on top of it
 *                   (sensors_flight.yaml by default) -- the same two files, in
 *                   the same order, the launch file gives the node -- and
 *                   nav_flight.yaml for the estimator.
 *    Plant          the scene, with the target parked out of the way, the arms
 *                   held at their start pose by a soft PD (so they move under
 *                   base acceleration, as they would, without hitting their
 *                   stops), and a smooth scripted wrench on the chaser so it
 *                   translates and turns throughout.
 *    the loop       every physics step: wrench, step. Every imu_rate_hz step:
 *                   read MuJoCo's accelerometer and gyro, corrupt, propagate.
 *                   Every pub_rate_hz step: read the true pose, corrupt the
 *                   position and yaw channels, update; record estimate and
 *                   truth together.
 *
 *  The sensor channels are advanced at the period the node advances them at
 *  (IMU at imu_rate_hz, odometry at pub_rate_hz), and odometry is stamped
 *  with the publish time exactly as the node stamps it, so a configured
 *  latency means here what it means live.
 *
 *  ========================= READING THE OUTPUT =========================
 *
 *    error rms / peak   estimate minus truth, after the settle time
 *    bias est / true    the filter's bias estimates against the biases the
 *                       sensor model actually applied
 *    NEES               the error normalised by the filter's own covariance.
 *                       8 on average if the covariance is honest; higher is
 *                       overconfident, lower pessimistic. With --seeds N the
 *                       per-instant average over seeds is checked against the
 *                       chi-square 95% band
 *    NIS                the same for each sensor's innovations: 1 for yaw, 2
 *                       for position, if the filter predicts them honestly
 *
 *  ================================ USE =================================
 *
 *    nav_bench
 *    nav_bench --seeds 20
 *    nav_bench --no-position --seconds 60
 *    nav_bench --latency-scale 0.5
 *    nav_bench --csv /tmp/nav.csv
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "pat_gnc/navigation/imu_ekf_navigator.hpp"
#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/sensors/sensor_model.hpp"

namespace {

using pat_gnc::ImuEkf;
using pat_gnc::ImuEkfNavigator;
using pat_simulation::ChannelConfig;
using pat_simulation::MuJoCoSim;
using pat_simulation::SensorModel;

// ============================ File lookup ================================ //

/*! `<prefix>/share/<pkg>/<rel>` for the first prefix in AMENT_PREFIX_PATH
 *  that has it. colcon gives each package its own prefix, so this package's
 *  CMAKE_INSTALL_PREFIX does not contain pat_gnc's share directory; scanning
 *  the overlay finds it either way. Reading an environment variable is not a
 *  ROS dependency. Returns "" if nothing matches. */
std::string findShare(const std::string& pkg, const std::string& rel) {
    if (const char* paths = std::getenv("AMENT_PREFIX_PATH")) {
        std::stringstream ss(paths);
        std::string prefix;
        while (std::getline(ss, prefix, ':')) {
            if (prefix.empty()) continue;
            const std::string c = prefix + "/share/" + pkg + "/" + rel;
            if (std::ifstream(c).good()) return c;
        }
    }
    return "";
}

// ============================ Configuration ============================== //

using Flat = std::map<std::string, YAML::Node>;

//! Nested YAML maps -> the dotted keys ROS parameter names use.
void flatten(const YAML::Node& n, const std::string& prefix, Flat& out) {
    if (n.IsMap()) {
        for (const auto& kv : n) {
            const std::string k = kv.first.as<std::string>();
            flatten(kv.second, prefix.empty() ? k : prefix + "." + k, out);
        }
    } else {
        out[prefix] = n;
    }
}

/*! One node's ros__parameters block from a ROS params file, flattened, and
 *  merged over `into` -- so loading a base file and then an overlay leaves the
 *  overlay's values wherever both set a key, as ROS does. */
void loadParams(const std::string& path, const std::string& node, Flat& into) {
    const YAML::Node root = YAML::LoadFile(path);
    if (!root[node] || !root[node]["ros__parameters"]) {
        throw std::runtime_error(path + " has no " + node + ".ros__parameters block");
    }
    flatten(root[node]["ros__parameters"], "", into);
}

double num(const Flat& f, const std::string& k, double def) {
    const auto it = f.find(k);
    return it == f.end() ? def : it->second.as<double>();
}

//! A sensor channel's eight fields, defaulting to perfect -- what the node's
//! declareChannel does with an unset parameter.
ChannelConfig channelFrom(const Flat& f, const std::string& name) {
    const std::string p = "sensors." + name + ".";
    ChannelConfig c;
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

/*! The estimator's configuration from nav_node's params file.
 *
 *  Mirrors pat_gnc's navigator_factory.hpp, which reads the same keys through
 *  ROS parameters. Any imu_ekf.* key in the file that is not mapped here is a
 *  hard error, so a knob added to nav.yaml without teaching it to the bench
 *  says so instead of being silently measured at its default. */
ImuEkfNavigator::Config navConfigFrom(const Flat& f, double& imu_hz) {
    ImuEkfNavigator::Config c;
    ImuEkf::Params& e = c.ekf;
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
            throw std::runtime_error("nav_bench does not know '" + kv.first +
                                     "' -- teach navConfigFrom() it");
        }
    }
    imu_hz = num(f, "imu_hz", 100.0);
    return c;
}

// =============================== Options ================================= //

struct Options {
    std::string scene   = findShare("pat_simulation",
                                    "models/environment/air_bearing_table.xml");
    std::string sim_cfg = findShare("pat_simulation", "config/simulation.yaml");
    std::string sensors = findShare("pat_simulation", "config/sensors_flight.yaml");
    std::string nav     = findShare("pat_gnc", "config/nav_flight.yaml");
    std::string csv;
    double seconds = 60.0;
    double settle  = 10.0;     //!< excluded from scoring: filter converging
    int    seeds   = 1;
    long   seed    = -1;       //!< -1: the sensor_seed in simulation.yaml
    bool   no_position = false;
    //! Multiplies the latencies the FILTER is told, not the sensors' real
    //! ones: 1 is honest, anything else is a mis-specified latency.
    double latency_scale = 1.0;
};

void usage() {
    std::cout <<
R"(nav_bench - deterministic scoring of pat_gnc's IMU-driven estimator against
            MuJoCo truth. No ROS, no DDS, no wall clock.

  --scene PATH         scene MJCF (default: the installed air_bearing_table.xml)
  --sim-config PATH    base simulator params (default: installed simulation.yaml)
  --sensors PATH       sensor overlay (default: installed sensors_flight.yaml)
  --nav PATH           estimator params (default: installed nav_flight.yaml)
  --seconds S          simulated seconds per run (default 60)
  --settle S           seconds excluded from scoring at the start (default 10)
  --seeds N            Monte-Carlo runs, each with its own sensor seed, for the
                       NEES consistency test (default 1)
  --seed S             first sensor seed (default: simulation.yaml's sensor_seed)
  --no-position        fuse no position fix: flight without relative navigation.
                       Position and accelerometer bias become unobservable
  --latency-scale F    tell the filter F x the true sensor latencies (default 1)
  --csv PATH           write the first run's per-tick series
)";
}

bool parseArgs(int argc, char** argv, Options& o, int& code) {
    code = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        try {
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("needs a value");
                return argv[++i];
            };
            if      (a == "--scene")         o.scene = next();
            else if (a == "--sim-config")    o.sim_cfg = next();
            else if (a == "--sensors")       o.sensors = next();
            else if (a == "--nav")           o.nav = next();
            else if (a == "--csv")           o.csv = next();
            else if (a == "--seconds")       o.seconds = std::stod(next());
            else if (a == "--settle")        o.settle = std::stod(next());
            else if (a == "--seeds")         o.seeds = std::stoi(next());
            else if (a == "--seed")          o.seed = std::stol(next());
            else if (a == "--no-position")   o.no_position = true;
            else if (a == "--latency-scale") o.latency_scale = std::stod(next());
            else if (a == "-h" || a == "--help") { usage(); return false; }
            else {
                std::cerr << "unknown argument: " << a << "\n";
                usage();
                code = 2;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "nav_bench: " << a << ": " << e.what() << "\n";
            code = 2;
            return false;
        }
    }
    if (o.seconds <= o.settle || o.seeds < 1 || o.latency_scale < 0.0) {
        std::cerr << "nav_bench: need seconds > settle, seeds >= 1, latency-scale >= 0\n";
        code = 2;
        return false;
    }
    for (const auto* p : {&o.scene, &o.sim_cfg, &o.sensors, &o.nav}) {
        if (p->empty()) {
            std::cerr << "nav_bench: a default file was not found on AMENT_PREFIX_PATH "
                         "(source the overlay, or pass it explicitly)\n";
            code = 2;
            return false;
        }
    }
    return true;
}

// ================================ Plant ================================== //

/*! The scene as a plant: the chaser driven by a smooth scripted wrench, the
 *  arms held softly at their start pose, the target parked out of reach. */
class Plant {
public:
    Plant(const std::string& scene, const Flat& sim) : sim_(scene, /*visualise=*/false) {
        arm_joints_ = sim.at("arm_joint_names").as<std::vector<std::string>>();
        const auto acts = sim.at("arm_actuator_names").as<std::vector<std::string>>();
        for (const auto& a : acts) arm_acts_.push_back(sim_.actuatorId(a));
        arm_home_ = sim.at("initial_arm_qpos").as<std::vector<double>>();
        if (arm_home_.size() != arm_joints_.size() || acts.size() != arm_joints_.size()) {
            throw std::runtime_error("simulation.yaml: arm joint, actuator and home lists differ in length");
        }
        // The MJCF's all-zero arm pose self-collides at the real mount, so the
        // configured home pose is applied before anything runs.
        sim_.setJointPositions(arm_joints_, arm_home_);
        // Out of reach, so no contact can put an impulse into the base that the
        // scenario did not script.
        sim_.setJointPositions({"target_x", "target_y"}, {2.0, 2.0});
        chaser_ = sim_.bodyId("chaser");
        ctrl_.assign(static_cast<std::size_t>(sim_.model()->nu), 0.0);
        accel_ = sim_.sensorId("chaser_accel");
        gyro_ = sim_.sensorId("chaser_gyro");
    }

    double time() const { return sim_.time(); }
    double dt() const { return sim_.dt(); }

    /*! One physics step under the scripted wrench.
     *
     *  Several incommensurate sinusoids, so the base translates and turns
     *  through the whole run and attitude, velocity and both biases are all
     *  exercised. World-frame force (N) and torque (N*m) on a 49 kg bus with
     *  yaw inertia ~2.5 kg*m^2.
     *
     *  Cosines, not sines: starting from rest, A*cos(w t) integrates to a
     *  velocity A/(m w) sin(w t) that averages to zero, so the base oscillates
     *  -- about +/-16 cm and +/-1.4 rad -- instead of drifting off at the mean
     *  velocity a sine leaves behind. */
    void step() {
        const double t = sim_.time();
        const double fx = 2.0 * std::cos(0.5 * t) + 1.2 * std::cos(1.3 * t);
        const double fy = 1.5 * std::cos(0.37 * t) + 0.8 * std::cos(1.1 * t);
        const double tz = 0.3 * std::cos(0.29 * t) + 0.15 * std::cos(0.9 * t);
        sim_.setExternalWrench(chaser_, fx, fy, tz);

        // A soft PD about the home pose, clipped to the motors' 0.25 N*m. The
        // arms still swing under base acceleration -- their reaction is part
        // of what the accelerometer sees -- but stay off their stops.
        const auto js = sim_.getJointStates(arm_joints_);
        for (std::size_t i = 0; i < arm_joints_.size(); ++i) {
            const double tau = -1.0 * (js.pos[i] - arm_home_[i]) - 0.2 * js.vel[i];
            ctrl_[static_cast<std::size_t>(arm_acts_[i])] = std::clamp(tau, -0.25, 0.25);
        }
        sim_.step(ctrl_);
    }

    //! MuJoCo's accelerometer (x, y) and gyro (z), on the current state.
    void imu(double& ax, double& ay, double& gz) {
        sim_.refreshSensors();
        const double* a = sim_.sensorData(accel_);
        const double* g = sim_.sensorData(gyro_);
        ax = a[0];
        ay = a[1];
        gz = g[2];
    }

    pat_simulation::PlanarState chaser() const { return sim_.getChaserState(); }

private:
    MuJoCoSim sim_;
    std::vector<std::string> arm_joints_;
    std::vector<int> arm_acts_;
    std::vector<double> arm_home_;
    std::vector<double> ctrl_;
    int chaser_ = -1, accel_ = -1, gyro_ = -1;
};

// ================================ Trial ================================== //

//! One odometry tick of the record.
struct Row {
    double t = 0.0;
    ImuEkf::Vec truth{ImuEkf::Vec::Zero()};   //!< in the filter's 8-state layout
    ImuEkf::Vec est{ImuEkf::Vec::Zero()};
    ImuEkf::Vec sd{ImuEkf::Vec::Zero()};      //!< filter 1-sigma per state
    double nees = 0.0;
};

struct Trial {
    std::vector<Row> rows;
    std::vector<double> nis_yaw, nis_pos;
    ImuEkf::Diagnostics diag;
};

double wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

/*! Run one realisation with sensor seed `seed`.
 *
 *  Order per physics step mirrors the node: the IMU is read from the step
 *  loop at imu_rate_hz, odometry at pub_rate_hz with the IMU of the same
 *  instant fed first -- the order nav_node's timestamp merge produces. */
Trial runTrial(const Options& o, const Flat& sim, ImuEkfNavigator::Config nav_cfg,
               std::uint64_t seed) {
    Plant plant(o.scene, sim);

    const double sim_hz = num(sim, "sim_rate_hz", 500.0);
    const double pub_hz = num(sim, "pub_rate_hz", 100.0);
    const double imu_hz = num(sim, "imu_rate_hz", pub_hz);
    const int imu_every = std::max(1, static_cast<int>(std::lround(sim_hz / imu_hz)));
    const int pub_every = std::max(1, static_cast<int>(std::lround(sim_hz / pub_hz)));

    // The node's sensor channels, built with the same names (so the same
    // per-channel random streams for a given seed), the same periods, and in
    // the same consecutive groups.
    SensorModel sensors(plant.dt(), seed);
    const double imu_dt = imu_every * plant.dt();
    const double pub_dt = 1.0 / pub_hz;
    const int h_acc = sensors.addChannel("imu_accel_x", channelFrom(sim, "imu_accel_x"), imu_dt);
    sensors.addChannel("imu_accel_y", channelFrom(sim, "imu_accel_y"), imu_dt);
    const int h_gyro = sensors.addChannel("imu_gyro_z", channelFrom(sim, "imu_gyro_z"), imu_dt);
    const int h_nav = sensors.addChannel("nav_x", channelFrom(sim, "nav_x"), pub_dt);
    sensors.addChannel("nav_y", channelFrom(sim, "nav_y"), pub_dt);
    sensors.addChannel("star_yaw", channelFrom(sim, "star_yaw"), pub_dt);

    if (o.no_position) nav_cfg.use_position = false;
    nav_cfg.latency_yaw *= o.latency_scale;
    nav_cfg.latency_pos *= o.latency_scale;
    ImuEkfNavigator nav(nav_cfg);

    Trial tr;
    const int steps = static_cast<int>(std::lround(o.seconds * sim_hz));
    tr.rows.reserve(static_cast<std::size_t>(steps / pub_every + 1));
    std::uint64_t seen_yaw = 0, seen_pos = 0;

    for (int k = 1; k <= steps; ++k) {
        plant.step();
        const double t = plant.time();

        if (k % imu_every == 0) {
            double a[2], g;
            plant.imu(a[0], a[1], g);
            double a_out[2];
            sensors.applyVector(h_acc, a, a_out, 2);
            nav.onImu(t, {a_out[0], a_out[1]}, sensors.apply(h_gyro, g));
        }
        if (k % pub_every != 0) continue;

        const auto s = plant.chaser();
        const double pose_truth[3] = {s.x, s.y, s.theta};
        double pose[3];
        sensors.applyVector(h_nav, pose_truth, pose, 3);
        nav.onOdometry(t, {pose[0], pose[1], pose[2]}, {0.0, 0.0, 0.0});

        const auto& f = nav.filter();
        // Innovations as they were applied -- read off the diagnostics after
        // each update, since the navigator does not return them.
        const auto& dg = f.diagnostics();
        if (dg.yaw_updates + dg.yaw_rejected > seen_yaw) {
            tr.nis_yaw.push_back(dg.last_nis_yaw);
            seen_yaw = dg.yaw_updates + dg.yaw_rejected;
        }
        if (dg.pos_updates + dg.pos_rejected > seen_pos) {
            tr.nis_pos.push_back(dg.last_nis_pos);
            seen_pos = dg.pos_updates + dg.pos_rejected;
        }
        if (!f.ready()) continue;

        Row r;
        r.t = t;
        // Truth in the filter's layout. The biases are the ones the sensor
        // model has in force right now, constant plus walk, in the IMU's body
        // frame -- exactly what the filter's bias states estimate.
        r.truth << s.x, s.y, s.theta, s.xdot, s.ydot,
            sensors.channel(h_gyro).bias(),
            sensors.channel(h_acc).bias(), sensors.channel(h_acc + 1).bias();
        r.est = f.state();
        r.sd = f.covariance().diagonal().cwiseSqrt();
        ImuEkf::Vec e = r.truth - r.est;
        e(ImuEkf::TH) = wrap(e(ImuEkf::TH));
        r.nees = e.dot(f.covariance().ldlt().solve(e));
        tr.rows.push_back(r);
    }
    tr.diag = nav.filter().diagnostics();
    return tr;
}

// =============================== Scoring ================================= //

double chi2Quantile(double z, double k) {
    // Wilson-Hilferty: well under 1% error at the dof used here.
    const double a = 2.0 / (9.0 * k);
    return k * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return std::nan("");
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

void reportTrial(const Options& o, const Trial& tr) {
    const char* names[ImuEkf::N] = {"x [m]", "y [m]", "yaw [rad]", "vx [m/s]", "vy [m/s]",
                                    "gyro bias", "accel bias x", "accel bias y"};
    std::printf("\n  %-14s %12s %12s %12s %12s %12s\n", "state", "error rms", "error peak",
                "filter sd", "final est", "final true");
    for (int i = 0; i < ImuEkf::N; ++i) {
        double sq = 0.0, peak = 0.0, sd = 0.0;
        int n = 0;
        for (const auto& r : tr.rows) {
            if (r.t < o.settle) continue;
            double e = r.truth(i) - r.est(i);
            if (i == ImuEkf::TH) e = wrap(e);
            sq += e * e;
            peak = std::max(peak, std::fabs(e));
            sd += r.sd(i);
            ++n;
        }
        const Row& last = tr.rows.back();
        std::printf("  %-14s %12.5f %12.5f %12.5f %12.5f %12.5f\n", names[i],
                    n ? std::sqrt(sq / n) : 0.0, peak, n ? sd / n : 0.0,
                    last.est(i), last.truth(i));
    }
    std::vector<double> nees;
    for (const auto& r : tr.rows) if (r.t >= o.settle) nees.push_back(r.nees);
    std::printf("\n  NEES mean %.2f (8 if the covariance is honest)   NIS mean: yaw %.2f "
                "(1)  pos %.2f (2)\n", mean(nees), mean(tr.nis_yaw), mean(tr.nis_pos));
    const auto& d = tr.diag;
    std::printf("  updates: yaw %lu applied / %lu rejected, pos %lu / %lu, too_old %lu, "
                "replays %lu (max %zu steps), imu %lu (gaps %lu)\n",
                static_cast<unsigned long>(d.yaw_updates), static_cast<unsigned long>(d.yaw_rejected),
                static_cast<unsigned long>(d.pos_updates), static_cast<unsigned long>(d.pos_rejected),
                static_cast<unsigned long>(d.too_old), static_cast<unsigned long>(d.replays),
                d.max_replay_steps, static_cast<unsigned long>(d.imu_samples),
                static_cast<unsigned long>(d.imu_gaps));
}

/*! The Monte-Carlo consistency test: per instant, the NEES averaged over the
 *  seeds is chi-square(8M)/M if the filter's covariance is right. */
void reportConsistency(const Options& o, const std::vector<Trial>& trials) {
    const std::size_t n = trials.front().rows.size();
    for (const auto& t : trials) {
        if (t.rows.size() != n) {
            std::printf("\n  runs have different lengths; consistency not scored\n");
            return;
        }
    }
    const double M = static_cast<double>(trials.size());
    const double lo = chi2Quantile(-1.96, 8.0 * M) / M;
    const double hi = chi2Quantile(+1.96, 8.0 * M) / M;
    int inside = 0, counted = 0;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (trials.front().rows[i].t < o.settle) continue;
        double avg = 0.0;
        for (const auto& t : trials) avg += t.rows[i].nees;
        avg /= M;
        sum += avg;
        ++counted;
        if (avg >= lo && avg <= hi) ++inside;
    }
    std::vector<double> nis_y, nis_p;
    for (const auto& t : trials) {
        nis_y.insert(nis_y.end(), t.nis_yaw.begin(), t.nis_yaw.end());
        nis_p.insert(nis_p.end(), t.nis_pos.begin(), t.nis_pos.end());
    }
    std::printf("\nconsistency over %d seeds: NEES averaged per instant is in the 95%% band "
                "[%.2f, %.2f]\n  on %.1f%% of instants (about 95%% expected for an honest "
                "filter); overall mean %.2f (8)\n  NIS mean over all runs: yaw %.2f (1)  pos %.2f (2)\n",
                static_cast<int>(trials.size()), lo, hi,
                counted ? 100.0 * inside / counted : 0.0, counted ? sum / counted : 0.0,
                mean(nis_y), mean(nis_p));
}

void writeCsv(const std::string& path, const Trial& tr) {
    std::ofstream os(path);
    if (!os) throw std::runtime_error("cannot write " + path);
    const char* s[ImuEkf::N] = {"x", "y", "yaw", "vx", "vy", "bg", "bax", "bay"};
    os << "t";
    for (const char* k : s) os << ",true_" << k << ",est_" << k << ",sd_" << k;
    os << ",nees\n";
    for (const auto& r : tr.rows) {
        os << r.t;
        for (int i = 0; i < ImuEkf::N; ++i) os << ',' << r.truth(i) << ',' << r.est(i) << ',' << r.sd(i);
        os << ',' << r.nees << '\n';
    }
    std::printf("\n  wrote %zu rows to %s\n", tr.rows.size(), path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    int code = 0;
    if (!parseArgs(argc, argv, o, code)) return code;
    try {
        // simulation.yaml, then the overlay on top -- the order the launch file
        // gives the node.
        Flat sim;
        loadParams(o.sim_cfg, "pat_simulation", sim);
        loadParams(o.sensors, "pat_simulation", sim);
        Flat navf;
        loadParams(o.nav, "pat_nav", navf);
        double nav_imu_hz = 0.0;
        const ImuEkfNavigator::Config nav_cfg = navConfigFrom(navf, nav_imu_hz);

        const long seed0 = o.seed >= 0 ? o.seed
                                       : static_cast<long>(num(sim, "sensor_seed", 20251014));
        std::printf("nav_bench: %.0f s per run, settle %.0f s, %d seed(s) from %ld%s, "
                    "latency x%.2f\n  sensors %s\n  nav     %s\n",
                    o.seconds, o.settle, o.seeds, seed0,
                    o.no_position ? ", NO position fix" : "", o.latency_scale,
                    o.sensors.c_str(), o.nav.c_str());
        const double sim_imu_hz = num(sim, "imu_rate_hz", num(sim, "pub_rate_hz", 100.0));
        if (std::fabs(sim_imu_hz - nav_imu_hz) > 1e-9) {
            std::printf("  NOTE: the simulator's imu_rate_hz (%.0f) and the estimator's imu_hz "
                        "(%.0f) differ\n", sim_imu_hz, nav_imu_hz);
        }

        std::vector<Trial> trials;
        for (int i = 0; i < o.seeds; ++i) {
            trials.push_back(runTrial(o, sim, nav_cfg, static_cast<std::uint64_t>(seed0 + i)));
        }
        std::printf("\nrun 1 (seed %ld):", seed0);
        reportTrial(o, trials.front());
        if (o.seeds > 1) reportConsistency(o, trials);
        if (!o.csv.empty()) writeCsv(o.csv, trials.front());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "nav_bench: " << e.what() << "\n";
        return 1;
    }
}
