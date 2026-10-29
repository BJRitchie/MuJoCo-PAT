/*! What the controller is told about the plant: exact, noisy and delayed. The
 *  statistics are checked against the requested sigmas, the streams against
 *  their seeds, and the delay against whole control ticks, so a study that sweeps
 *  the noise level is sweeping what it says it is.
 *
 *  The flight sources (ekf, raw) need a plant to feed them and are exercised by
 *  running the bench; here it is only checked that they refuse to start without
 *  their inputs. The joint filter needs no plant and is checked here.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "state_source.h"

namespace pat_arm_control {
namespace {
using bench::NoiseProfile;
using bench::StateSource;
using bench::StateSourceConfig;
using bench::StateSourceKind;

constexpr int kJoints = 6;
constexpr double kDt = 0.01;

//! A resting plant state at a known pose, in the controller's layout.
void truthAt(double x, double y, double yaw, Eigen::VectorXd& q, Eigen::VectorXd& v) {
    q = Eigen::VectorXd::Zero(7 + kJoints);
    v = Eigen::VectorXd::Zero(6 + kJoints);
    q(0) = x;  q(1) = y;
    q(3) = std::cos(yaw / 2.0);
    q(6) = std::sin(yaw / 2.0);
    for (int i = 0; i < kJoints; ++i) q(7 + i) = 0.3 * (i + 1);
}

StateSourceConfig noisy(double scale = 1.0, int delay = 0, std::uint64_t seed = 1) {
    StateSourceConfig c;
    c.kind = StateSourceKind::Noisy;
    c.noise_scale = scale;
    c.delay_ticks = delay;
    c.seed = seed;
    return c;
}

double stddev(const std::vector<double>& x) {
    double m = 0.0;
    for (double v : x) m += v;
    m /= static_cast<double>(x.size());
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(x.size()));
}

TEST(StateSource, ExactHandsOverThePlantUntouched) {
    StateSourceConfig c;   // exact
    StateSource s(c, kJoints, kDt, 0.002);
    Eigen::VectorXd qt, vt, q, v;
    truthAt(0.1, -0.2, 0.7, qt, vt);
    vt(5) = 0.05;
    s.measure(qt, vt, -0.5, 0.0, q, v);
    EXPECT_EQ(q, qt);
    EXPECT_EQ(v, vt);
    EXPECT_DOUBLE_EQ(s.lastError().x, 0.0);
    EXPECT_TRUE(s.exact());
}

TEST(StateSource, NoiseHasTheRequestedSigmaOnEveryChannel) {
    StateSource s(noisy(), kJoints, kDt, 0.002);
    NoiseProfile p;   // the defaults the source was built with
    Eigen::VectorXd qt, vt, q, v;
    truthAt(0.1, -0.2, 0.7, qt, vt);

    const int n = 20000;
    std::vector<double> ex, ey, eyaw, evx, ewz, ej0, ejv0;
    for (int k = 0; k < n; ++k) {
        s.measure(qt, vt, 0.0, 0.0, q, v);
        ex.push_back(q(0) - qt(0));
        ey.push_back(q(1) - qt(1));
        eyaw.push_back(s.lastError().yaw);
        evx.push_back(v(0) - vt(0));
        ewz.push_back(v(5) - vt(5));
        ej0.push_back(q(7) - qt(7));
        ejv0.push_back(v(6) - vt(6));
    }
    // Standard error of a sample sigma over n draws is sigma / sqrt(2n): 0.5%.
    const double tol = 0.05;
    EXPECT_NEAR(stddev(ex) / p.base_pos, 1.0, tol);
    EXPECT_NEAR(stddev(ey) / p.base_pos, 1.0, tol);
    EXPECT_NEAR(stddev(eyaw) / p.base_yaw, 1.0, tol);
    EXPECT_NEAR(stddev(evx) / p.base_vel, 1.0, tol);
    EXPECT_NEAR(stddev(ewz) / p.base_wz, 1.0, tol);
    EXPECT_NEAR(stddev(ej0) / p.joint_pos, 1.0, tol);
    EXPECT_NEAR(stddev(ejv0) / p.joint_vel, 1.0, tol);
}

TEST(StateSource, TheNoiseScaleMultipliesEverySigma) {
    Eigen::VectorXd qt, vt, q, v;
    truthAt(0.0, 0.0, 0.0, qt, vt);
    const auto sigma = [&](double scale) {
        StateSource s(noisy(scale), kJoints, kDt, 0.002);
        std::vector<double> e;
        for (int k = 0; k < 5000; ++k) {
            s.measure(qt, vt, 0.0, 0.0, q, v);
            e.push_back(v(6) - vt(6));
        }
        return stddev(e);
    };
    EXPECT_NEAR(sigma(4.0) / sigma(1.0), 4.0, 0.2);
    EXPECT_EQ(sigma(0.0), 0.0);   // no noise at all at scale 0
}

TEST(StateSource, ASeedReproducesTheStreamAndAnotherSeedChangesIt) {
    Eigen::VectorXd qt, vt, qa, va, qb, vb, qc, vc;
    truthAt(0.1, 0.2, 0.3, qt, vt);
    StateSource a(noisy(1.0, 0, 7), kJoints, kDt, 0.002);
    StateSource b(noisy(1.0, 0, 7), kJoints, kDt, 0.002);
    StateSource c(noisy(1.0, 0, 8), kJoints, kDt, 0.002);
    for (int k = 0; k < 50; ++k) {
        a.measure(qt, vt, 0.0, 0.0, qa, va);
        b.measure(qt, vt, 0.0, 0.0, qb, vb);
        c.measure(qt, vt, 0.0, 0.0, qc, vc);
        EXPECT_EQ(qa, qb);
        EXPECT_EQ(va, vb);
    }
    EXPECT_NE(qa, qc);
}

TEST(StateSource, ADelayIsExactlyThatManyControlTicks) {
    // No noise, a ramp in x: what comes out is the ramp, N ticks late.
    const int delay = 4;
    StateSource s(noisy(0.0, delay), kJoints, kDt, 0.002);
    Eigen::VectorXd qt, vt, q, v;
    std::vector<double> out;
    for (int k = 0; k < 30; ++k) {
        truthAt(0.001 * k, 0.0, 0.0, qt, vt);
        s.measure(qt, vt, 0.0, 0.0, q, v);
        out.push_back(q(0));
    }
    for (int k = delay; k < 30; ++k) EXPECT_NEAR(out[k], 0.001 * (k - delay), 1e-12) << k;
    // Before the ring fills, the first reading is held: no step from zero.
    for (int k = 0; k < delay; ++k) EXPECT_NEAR(out[k], 0.0, 1e-12) << k;
}

TEST(StateSource, YawNoiseIsAppliedToTheAngleNotTheQuaternionParts) {
    // The returned quaternion must stay unit length whatever the noise.
    StateSource s(noisy(8.0), kJoints, kDt, 0.002);
    Eigen::VectorXd qt, vt, q, v;
    truthAt(0.0, 0.0, 1.0, qt, vt);
    for (int k = 0; k < 200; ++k) {
        s.measure(qt, vt, 0.0, 0.0, q, v);
        EXPECT_NEAR(q(3) * q(3) + q(6) * q(6), 1.0, 1e-12);
    }
}

TEST(StateSource, TheFlightSourcesRefuseToStartWithoutTheirInputs) {
    StateSourceConfig c;
    c.kind = StateSourceKind::Ekf;   // no YAML paths
    EXPECT_THROW(StateSource(c, kJoints, kDt, 0.002), std::runtime_error);
    c.kind = StateSourceKind::Raw;
    EXPECT_THROW(StateSource(c, kJoints, kDt, 0.002), std::runtime_error);
}

TEST(StateSource, TheJointFilterCutsTheJointVelocityNoise) {
    // A joint at rest, read through the same seeded encoders with and without
    // the filter: the velocity handed over is several times less noisy with it.
    Eigen::VectorXd qt, vt, q, v;
    truthAt(0.0, 0.0, 0.0, qt, vt);
    const auto velocity_sigma = [&](bool filter) {
        StateSourceConfig c = noisy();
        c.joint_filter = filter;
        c.joint_kf.accel_psd = 1e-4;
        StateSource s(c, kJoints, kDt, 0.002);
        std::vector<double> e;
        for (int k = 0; k < 3000; ++k) {
            s.measure(qt, vt, 0.0, 0.0, q, v);
            if (k >= 200) e.push_back(v(6) - vt(6));
        }
        return stddev(e);
    };
    EXPECT_LT(velocity_sigma(true), velocity_sigma(false) / 3.0);
}

TEST(StateSource, TheJointFilterNeedsEncoderNoiseToWeigh) {
    StateSourceConfig c = noisy(0.0);
    c.joint_filter = true;
    EXPECT_THROW(StateSource(c, kJoints, kDt, 0.002), std::runtime_error);
}

TEST(StateSource, ANoiseFigureIsSetByNameAndAnUnknownOneIsRefused) {
    NoiseProfile p;
    p.set("joint_vel", 0.05);
    EXPECT_DOUBLE_EQ(p.joint_vel, 0.05);
    EXPECT_THROW(p.set("joint_velocity", 0.05), std::runtime_error);
}

// The live simulator's arm encoders must be the ones the bench hands its
// controllers, or a live run and a bench run are not given the same measurements.
TEST(StateSource, TheSimulatorsEncoderProfileIsTheBenchsNoiseProfile) {
    const YAML::Node root = YAML::LoadFile(PAT_SIM_ENCODER_PROFILE);
    const YAML::Node sensors = root["pat_simulation"]["ros__parameters"]["sensors"];
    const NoiseProfile bench;
    int seen = 0;
    for (const char* kind : {"joint_pos", "joint_vel"}) {
        ASSERT_TRUE(sensors[kind]) << kind;
        for (const auto& joint : sensors[kind]) {
            const YAML::Node c = joint.second;
            const double want = std::string(kind) == "joint_pos" ? bench.joint_pos
                                                                 : bench.joint_vel;
            EXPECT_DOUBLE_EQ(c["noise_std"].as<double>(), want) << kind << " " << joint.first;
            // White noise only: nothing the bench does not model.
            for (const char* other : {"bias", "bias_walk_std", "scale_error", "quantisation",
                                      "rate_hz", "dropout_prob", "delay_s"}) {
                EXPECT_DOUBLE_EQ(c[other].as<double>(), 0.0) << kind << " " << other;
            }
            ++seen;
        }
    }
    EXPECT_EQ(seen, 12);   // two per arm joint, six joints
}

TEST(StateSource, ANegativeScaleOrDelayIsRefused) {
    EXPECT_THROW(StateSource(noisy(-1.0), kJoints, kDt, 0.002), std::runtime_error);
    EXPECT_THROW(StateSource(noisy(1.0, -1), kJoints, kDt, 0.002), std::runtime_error);
}

}  // namespace
}  // namespace pat_arm_control
