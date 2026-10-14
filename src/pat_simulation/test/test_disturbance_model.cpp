#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <mujoco/mujoco.h>

#include "pat_simulation/mujoco_sim.hpp"
#include "pat_simulation/sensors/disturbance_model.hpp"

using pat_simulation::DisturbanceModel;
using pat_simulation::MuJoCoSim;
using pat_simulation::WrenchProfile;

namespace {

constexpr double kDt = 0.002;

const std::vector<std::string> kArmJoints = {
    "joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"};
const std::vector<double> kArmHome = {
    0.6108652382, -0.2617993878, 0.0, 0.6108652382, -0.2617993878, 0.0};

MuJoCoSim makeSim() { return MuJoCoSim(PAT_SIM_TEST_MJCF, false); }

std::vector<double> zeroCtrl(MuJoCoSim& sim) {
    return std::vector<double>(static_cast<std::size_t>(sim.model()->nu), 0.0);
}

} // namespace

// ---------------------------------------------------------------------------
// The profiles, in isolation.
// ---------------------------------------------------------------------------

TEST(DisturbanceProfile, NoneIsExactlyZeroAndInactive) {
    DisturbanceModel d(WrenchProfile{}, kDt, 1);
    EXPECT_FALSE(d.active());
    EXPECT_EQ(d.describe(), "none");
    double w[3] = {9, 9, 9};
    d.wrenchAt(5.0, w);
    EXPECT_EQ(w[0], 0.0);
    EXPECT_EQ(w[1], 0.0);
    EXPECT_EQ(w[2], 0.0);
}

TEST(DisturbanceProfile, StepHoldsItsAmplitudeInsideItsWindow) {
    WrenchProfile p;
    p.type = "step";
    p.fx = 0.5;
    p.tz = -0.02;
    p.start_s = 1.0;
    p.duration_s = 2.0;
    DisturbanceModel d(p, kDt, 1);
    ASSERT_TRUE(d.active());

    double w[3];
    d.wrenchAt(0.5, w);                      // before
    EXPECT_EQ(w[0], 0.0);
    d.wrenchAt(1.0, w);                      // at the leading edge
    EXPECT_DOUBLE_EQ(w[0], 0.5);
    EXPECT_DOUBLE_EQ(w[1], 0.0);
    EXPECT_DOUBLE_EQ(w[2], -0.02);
    d.wrenchAt(2.9, w);                      // inside
    EXPECT_DOUBLE_EQ(w[0], 0.5);
    d.wrenchAt(3.0, w);                      // the window is half-open
    EXPECT_EQ(w[0], 0.0);
    d.wrenchAt(10.0, w);                     // long after
    EXPECT_EQ(w[0], 0.0);
}

TEST(DisturbanceProfile, ZeroDurationMeansIndefinitely) {
    WrenchProfile p;
    p.type = "step";
    p.fy = 1.0;
    p.start_s = 0.5;
    DisturbanceModel d(p, kDt, 1);
    double w[3];
    d.wrenchAt(0.4, w);
    EXPECT_EQ(w[1], 0.0);
    d.wrenchAt(1e6, w);
    EXPECT_DOUBLE_EQ(w[1], 1.0);
}

TEST(DisturbanceProfile, SinusoidStartsAtZeroAndTracksItsPhase) {
    WrenchProfile p;
    p.type = "sinusoid";
    p.fx = 2.0;
    p.freq_hz = 0.5;
    p.start_s = 1.0;
    DisturbanceModel d(p, kDt, 1);

    double w[3];
    // Phase is measured from start_s, so switching on does not jump to the
    // amplitude -- a step disguised as a sinusoid would excite the whole
    // spectrum rather than the one frequency asked for.
    d.wrenchAt(1.0, w);
    EXPECT_NEAR(w[0], 0.0, 1e-12);
    d.wrenchAt(1.5, w);                      // quarter period of 0.5 Hz
    EXPECT_NEAR(w[0], 2.0, 1e-12);
    d.wrenchAt(2.0, w);                      // half period
    EXPECT_NEAR(w[0], 0.0, 1e-12);
    d.wrenchAt(2.5, w);                      // three quarters
    EXPECT_NEAR(w[0], -2.0, 1e-12);
}

TEST(DisturbanceProfile, RandomWalkOnlyTouchesTheAxesItWasGiven) {
    WrenchProfile p;
    p.type = "random_walk";
    p.fx = 1.0;
    p.fy = 0.0;      // deliberately unused
    p.tz = 0.0;      // deliberately unused
    p.walk_std = 0.5;
    DisturbanceModel d(p, kDt, 7);

    double w[3];
    bool fx_moved = false;
    for (int i = 0; i < 2000; ++i) {
        d.wrenchAt(i * kDt, w);
        if (std::fabs(w[0]) > 1e-9) fx_moved = true;
        // An axis with zero amplitude must stay exactly zero rather than
        // picking up noise, or "disturb x only" would silently disturb y too.
        EXPECT_EQ(w[1], 0.0) << "at i=" << i;
        EXPECT_EQ(w[2], 0.0) << "at i=" << i;
    }
    EXPECT_TRUE(fx_moved);
}

TEST(DisturbanceProfile, RandomWalkIsReproducibleFromItsSeed) {
    WrenchProfile p;
    p.type = "random_walk";
    p.fx = 1.0;
    p.walk_std = 0.3;

    auto run = [&](std::uint64_t seed) {
        DisturbanceModel d(p, kDt, seed);
        std::vector<double> v;
        double w[3];
        for (int i = 0; i < 500; ++i) {
            d.wrenchAt(i * kDt, w);
            v.push_back(w[0]);
        }
        return v;
    };
    EXPECT_EQ(run(11), run(11));
    EXPECT_NE(run(11), run(12));
}

TEST(DisturbanceProfile, RejectsBadConfiguration) {
    WrenchProfile bad_type;
    bad_type.type = "sawtooth";
    EXPECT_THROW(DisturbanceModel(bad_type, kDt, 1), std::invalid_argument);

    WrenchProfile no_freq;
    no_freq.type = "sinusoid";
    no_freq.fx = 1.0;
    EXPECT_THROW(DisturbanceModel(no_freq, kDt, 1), std::invalid_argument);

    WrenchProfile no_walk;
    no_walk.type = "random_walk";
    no_walk.fx = 1.0;
    EXPECT_THROW(DisturbanceModel(no_walk, kDt, 1), std::invalid_argument);

    WrenchProfile neg;
    neg.type = "step";
    neg.duration_s = -1.0;
    EXPECT_THROW(DisturbanceModel(neg, kDt, 1), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Against the simulator: the wrench has to actually do something, and the
// amount it does has to be recoverable -- otherwise it is useless as the
// instrument for scoring an observer.
// ---------------------------------------------------------------------------

TEST(DisturbanceInjection, AWrenchAcceleratesTheBodyByForceOverMass) {
    MuJoCoSim sim = makeSim();
    sim.setJointPositions(kArmJoints, kArmHome);
    const int chaser = sim.bodyId("chaser");
    const double mass = sim.model()->body_subtreemass[chaser];
    ASSERT_GT(mass, 10.0);

    // A known steady push, and nothing else acting.
    constexpr double kFx = 2.0;
    auto ctrl = zeroCtrl(sim);
    const double before = sim.getChaserState().xdot;
    constexpr int kSteps = 500;              // 1 s
    for (int i = 0; i < kSteps; ++i) {
        sim.setExternalWrench(chaser, kFx, 0.0, 0.0);
        sim.step(ctrl);
    }
    const double after = sim.getChaserState().xdot;

    // dv = F/m * t, less what the table's damping took out. chaser_x ships
    // damping=0.05 in the standalone MJCF, which at these speeds removes a
    // few percent, so this is checked to 10% rather than exactly.
    const double expected = kFx / mass * (kSteps * sim.dt());
    EXPECT_GT(after - before, 0.0);
    EXPECT_NEAR(after - before, expected, 0.1 * expected);
}

TEST(DisturbanceInjection, ClearingTheWrenchStopsIt) {
    MuJoCoSim sim = makeSim();
    sim.setJointPositions(kArmJoints, kArmHome);
    const int chaser = sim.bodyId("chaser");
    auto ctrl = zeroCtrl(sim);

    sim.setExternalWrench(chaser, 3.0, 0.0, 0.0);
    for (int i = 0; i < 100; ++i) sim.step(ctrl);
    const double pushed = sim.getChaserState().xdot;
    ASSERT_GT(pushed, 1e-4);

    // xfrc_applied persists until overwritten, which is exactly why the node
    // rewrites it every step; clearing has to actually clear.
    sim.clearExternalWrenches();
    for (int i = 0; i < 100; ++i) sim.step(ctrl);
    const double coasted = sim.getChaserState().xdot;
    EXPECT_LT(coasted, pushed) << "velocity should only decay once unforced";
}

TEST(DisturbanceInjection, NeverWritingAWrenchLeavesTheTrajectoryUntouched) {
    // The default path. If this drifts, every trajectory recorded before the
    // disturbance machinery existed becomes unreproducible.
    auto drive = [](bool touch_wrench) {
        MuJoCoSim sim = makeSim();
        sim.setJointPositions(kArmJoints, kArmHome);
        auto ctrl = zeroCtrl(sim);
        ctrl[sim.actuatorId("motor2_L")] = 3.0;
        const int chaser = sim.bodyId("chaser");
        for (int i = 0; i < 500; ++i) {
            if (touch_wrench) sim.setExternalWrench(chaser, 0.0, 0.0, 0.0);
            sim.step(ctrl);
        }
        return sim.getChaserState();
    };
    const auto never = drive(false);
    const auto zeroed = drive(true);
    // Writing an explicit zero must be indistinguishable from not writing.
    EXPECT_DOUBLE_EQ(never.x, zeroed.x);
    EXPECT_DOUBLE_EQ(never.theta, zeroed.theta);
    EXPECT_DOUBLE_EQ(never.xdot, zeroed.xdot);
}

// ---------------------------------------------------------------------------
// Contact ground truth and model error.
// ---------------------------------------------------------------------------

TEST(ContactTruth, ConstraintForceIsZeroApartFromContact) {
    MuJoCoSim sim = makeSim();
    const int nv = sim.nv();
    ASSERT_GT(nv, 0);

    // Clear of contact: the constraint force is the solver reporting nothing
    // to resolve, so it must be identically zero.
    sim.setJointPositions(kArmJoints, kArmHome);
    sim.step(zeroCtrl(sim));
    sim.refreshSensors();
    const double* q = sim.constraintForce();
    double norm = 0.0;
    for (int i = 0; i < nv; ++i) norm += q[i] * q[i];
    EXPECT_NEAR(std::sqrt(norm), 0.0, 1e-9);

    // In contact it must not be: the MJCF's default arm pose self-collides,
    // which is the signal a contact estimate gets scored against.
    MuJoCoSim touching = makeSim();
    touching.step(zeroCtrl(touching));
    touching.refreshSensors();
    const double* qc = touching.constraintForce();
    double cnorm = 0.0;
    for (int i = 0; i < nv; ++i) cnorm += qc[i] * qc[i];
    EXPECT_GT(std::sqrt(cnorm), 1e-6);
    EXPECT_GT(touching.data()->ncon, 0);
}

TEST(ModelError, MassScalingChangesTheResponseToAKnownForce) {
    // The point of mass error: the same commanded force produces a different
    // acceleration, so a controller carrying the unscaled model is wrong by a
    // knowable amount.
    auto push = [](double factor) {
        MuJoCoSim sim = makeSim();
        sim.setJointPositions(kArmJoints, kArmHome);
        if (factor != 1.0) sim.setBodyMassScale({"chaser"}, {factor});
        const int chaser = sim.bodyId("chaser");
        auto ctrl = zeroCtrl(sim);
        for (int i = 0; i < 250; ++i) {
            sim.setExternalWrench(chaser, 2.0, 0.0, 0.0);
            sim.step(ctrl);
        }
        return sim.getChaserState().xdot;
    };

    const double v_nominal = push(1.0);
    const double v_heavy   = push(1.5);
    ASSERT_GT(v_nominal, 1e-5);

    // Only the bus body is scaled, not the arms, so the subtree mass does not
    // rise by the full 1.5x -- hence a directional check plus a loose bound
    // rather than a precise ratio.
    EXPECT_LT(v_heavy, v_nominal) << "a heavier bus must accelerate less";
    EXPECT_GT(v_heavy, 0.5 * v_nominal);
}

TEST(ModelError, MassScalingValidatesItsInput) {
    MuJoCoSim sim = makeSim();
    EXPECT_THROW(sim.setBodyMassScale({"chaser"}, {1.0, 2.0}), std::runtime_error);
    EXPECT_THROW(sim.setBodyMassScale({"no_such_body"}, {1.0}), std::runtime_error);
    EXPECT_THROW(sim.setBodyMassScale({"chaser"}, {0.0}), std::runtime_error);
    EXPECT_THROW(sim.setBodyMassScale({"chaser"}, {-1.0}), std::runtime_error);
}

TEST(ModelError, MassScalingIsReflectedInTheModelsOwnMass) {
    MuJoCoSim sim = makeSim();
    const int chaser = sim.bodyId("chaser");
    const double m0 = sim.model()->body_mass[chaser];
    const double i0 = sim.model()->body_inertia[3 * chaser];
    ASSERT_GT(m0, 0.0);

    sim.setBodyMassScale({"chaser"}, {2.0});

    // Mass and inertia scale together: this is a density change, not a body
    // that got heavier without getting harder to spin.
    EXPECT_DOUBLE_EQ(sim.model()->body_mass[chaser], 2.0 * m0);
    EXPECT_DOUBLE_EQ(sim.model()->body_inertia[3 * chaser], 2.0 * i0);
    // And the derived constants were rebuilt, not left stale.
    EXPECT_GT(sim.model()->body_subtreemass[chaser], m0);
}
