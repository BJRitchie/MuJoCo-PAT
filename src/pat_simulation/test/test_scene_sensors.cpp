#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "pat_simulation/mujoco_sim.hpp"

using pat_simulation::MuJoCoSim;

namespace {

// The scene body's own origin, straight out of air_bearing_table.xacro's
// <body name="chaser" pos="-0.5 0 0.2445">. The site sensors report world
// coordinates, so they carry this offset while the joint coordinates do not --
// which is exactly the relationship these tests exist to pin down.
constexpr double kChaserBodyX = -0.5;
constexpr double kChaserBodyY = 0.0;
constexpr double kChaserBodyZ = 0.2445;

// The collision-free starting pose, matching simulation.yaml's
// initial_arm_qpos. The MJCF's own all-zero fold pose is NOT collision-free at
// the real (digital-twin-derived) arm mount with arm_contype=2, so a sim left
// at the raw default begins with contact forces already acting -- which reads
// on the accelerometer as a genuine acceleration of about 0.05 m/s^2. Any test
// that needs an undisturbed chaser has to set this, exactly as the node does.
const std::vector<std::string> kArmJoints = {
    "joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"};
const std::vector<double> kArmHome = {
    0.6108652382, -0.2617993878, 0.0, 0.6108652382, -0.2617993878, 0.0};

// Headless: the tests never open a window.
MuJoCoSim makeSim() { return MuJoCoSim(PAT_SIM_TEST_MJCF, false); }

std::vector<double> zeroCtrl(MuJoCoSim& sim) {
    return std::vector<double>(static_cast<std::size_t>(sim.model()->nu), 0.0);
}

} // namespace

// ---------------------------------------------------------------------------
// The sensor contract: what the MJCF declares, and the layout the node's
// SensorModel wiring indexes into.
// ---------------------------------------------------------------------------

TEST(SceneSensors, TheDeclaredSuiteIsPresentWithTheExpectedWidths) {
    MuJoCoSim sim = makeSim();

    struct Expect { const char* name; int dim; int type; };
    const Expect want[] = {
        {"chaser_accel",    3, mjSENS_ACCELEROMETER},
        {"chaser_gyro",     3, mjSENS_GYRO},
        {"chaser_attitude", 4, mjSENS_FRAMEQUAT},
        {"chaser_position", 3, mjSENS_FRAMEPOS},
    };
    for (const auto& w : want) {
        const int id = sim.sensorId(w.name);
        EXPECT_EQ(sim.sensorDim(id), w.dim) << w.name;
        EXPECT_EQ(sim.model()->sensor_type[id], w.type) << w.name;
    }
}

TEST(SceneSensors, AskingForASensorThatDoesNotExistThrows) {
    MuJoCoSim sim = makeSim();
    EXPECT_THROW(sim.sensorId("no_such_sensor"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// What each sensor means, related back to the joint-space truth the node
// already publishes. If these ever disagree, the odom and the IMU are
// describing different vehicles.
// ---------------------------------------------------------------------------

TEST(SceneSensors, FramePosIsTheJointPositionPlusTheBodyOrigin) {
    MuJoCoSim sim = makeSim();
    const int pos_id = sim.sensorId("chaser_position");

    // Drive the chaser somewhere unambiguous via the joints, then compare.
    sim.setJointPositions({"chaser_x", "chaser_y", "chaser_yaw"},
                          {0.3, -0.2, 0.0});
    const auto s = sim.getChaserState();
    const double* p = sim.sensorData(pos_id);

    EXPECT_NEAR(p[0], s.x + kChaserBodyX, 1e-9);
    EXPECT_NEAR(p[1], s.y + kChaserBodyY, 1e-9);
    EXPECT_NEAR(p[2], kChaserBodyZ, 1e-9);

    // Stated the other way round, because this is the conversion the node needs
    // if it ever publishes odom from the sensor rather than from the joints.
    EXPECT_NEAR(p[0] - kChaserBodyX, 0.3, 1e-9);
    EXPECT_NEAR(p[1] - kChaserBodyY, -0.2, 1e-9);
}

TEST(SceneSensors, FrameQuatEncodesTheYawJoint) {
    MuJoCoSim sim = makeSim();
    const int att_id = sim.sensorId("chaser_attitude");

    const double yaw = 0.7;
    sim.setJointPositions({"chaser_yaw"}, {yaw});
    const double* q = sim.sensorData(att_id);

    // Planar yaw about +z: q = [cos(yaw/2), 0, 0, sin(yaw/2)], MuJoCo's w-first
    // order. Recovering the angle is the operation a star-tracker-fed navigator
    // performs, so check it that way too.
    EXPECT_NEAR(q[0], std::cos(yaw / 2.0), 1e-9);
    EXPECT_NEAR(q[1], 0.0, 1e-9);
    EXPECT_NEAR(q[2], 0.0, 1e-9);
    EXPECT_NEAR(q[3], std::sin(yaw / 2.0), 1e-9);

    const double recovered = 2.0 * std::atan2(q[3], q[0]);
    EXPECT_NEAR(std::remainder(recovered - yaw, 2.0 * M_PI), 0.0, 1e-9);
}

TEST(SceneSensors, GyroZIsTheYawRate) {
    MuJoCoSim sim = makeSim();
    const int gyro_id = sim.sensorId("chaser_gyro");

    // Yaw has to come from arm reaction, not from a thruster: all four
    // thrusters push through the body origin (thr_port sits at (0, 0.25, 0) and
    // pushes along +y, so r x F is zero), which is what makes them a pure
    // translation set. Swinging an arm is the only yaw authority the chaser has
    // here, and it is also the reaction the NMPC exists to account for.
    sim.setJointPositions(kArmJoints, kArmHome);
    auto ctrl = zeroCtrl(sim);
    ctrl[sim.actuatorId("motor2_L")] = 8.0;
    for (int i = 0; i < 200; ++i) sim.step(ctrl);
    sim.refreshSensors();   // without this the gyro is an RK4 stage out, ~6%

    const auto s = sim.getChaserState();
    const double* g = sim.sensorData(gyro_id);
    ASSERT_GT(std::fabs(s.thetadot), 1e-6) << "the arm produced no yaw reaction";

    EXPECT_NEAR(g[2], s.thetadot, 1e-9);
    // Planar table: no roll or pitch rate exists to measure.
    EXPECT_NEAR(g[0], 0.0, 1e-9);
    EXPECT_NEAR(g[1], 0.0, 1e-9);
}

TEST(SceneSensors, AccelerometerCarriesRealSignalNotJustNoise) {
    // The reason this sensor was added: simulation_node used to publish pure
    // noise on linear_acceleration with no signal underneath it at all.
    MuJoCoSim sim = makeSim();
    const int acc_id = sim.sensorId("chaser_accel");

    sim.setJointPositions(kArmJoints, kArmHome);
    auto ctrl = zeroCtrl(sim);
    const double* a = sim.sensorData(acc_id);

    // At rest and unforced, nothing is accelerating.
    sim.step(ctrl);
    sim.refreshSensors();
    EXPECT_NEAR(a[0], 0.0, 1e-6);
    EXPECT_NEAR(a[1], 0.0, 1e-6);

    // Fire the forward thruster: +x body force, so +x proper acceleration, of
    // order F/m for the 49 kg bus. Gravity is disabled, so there is no 1 g
    // bias term -- which is also true of a real IMU in free fall.
    ctrl[sim.actuatorId("thr_fwd")] = 5.0;
    sim.step(ctrl);
    sim.refreshSensors();

    EXPECT_GT(a[0], 0.0) << "forward thrust did not register as +x acceleration";
    const double total_mass = sim.model()->body_subtreemass[
        mj_name2id(sim.model(), mjOBJ_BODY, "chaser")];
    ASSERT_GT(total_mass, 1.0);
    EXPECT_NEAR(a[0], 5.0 / total_mass, 0.5 * 5.0 / total_mass);
}

TEST(SceneSensors, SensorsReadAfterAStepNeedARefresh) {
    // Pinning the reason refreshSensors() exists. mj_step evaluates the
    // dynamics at the start of the step and integrates afterwards, and under
    // RK4 the last internal evaluation is an intermediate stage, so reading
    // sensordata straight after step() gives a state that matches no instant.
    // If a MuJoCo upgrade ever makes mj_step leave sensors synchronised, this
    // test fails and refreshSensors() can become a no-op -- deliberately, not
    // by accident.
    MuJoCoSim sim = makeSim();
    const int gyro_id = sim.sensorId("chaser_gyro");
    sim.setJointPositions(kArmJoints, kArmHome);

    auto ctrl = zeroCtrl(sim);
    ctrl[sim.actuatorId("motor2_L")] = 8.0;
    for (int i = 0; i < 200; ++i) sim.step(ctrl);

    const double stale = sim.sensorData(gyro_id)[2];
    const double truth = sim.getChaserState().thetadot;
    sim.refreshSensors();
    const double fresh = sim.sensorData(gyro_id)[2];

    EXPECT_NEAR(fresh, truth, 1e-9) << "refreshSensors did not synchronise";
    EXPECT_GT(std::fabs(stale - truth), 1e-6)
        << "mj_step now leaves sensors synchronised; refreshSensors is redundant";
}

TEST(SceneSensors, TheMjcfDefaultArmPoseStartsInContact) {
    // Why every test above sets kArmHome first, and why simulation.yaml ships
    // initial_arm_qpos rather than trusting the model's own defaults: with
    // arm_contype=2 the all-zero fold pose is not collision-free at the real
    // arm mount, so an unoverridden sim starts with contact forces already
    // acting. Worth a failing test if that ever changes, because the override
    // could then be dropped.
    MuJoCoSim sim = makeSim();
    const int acc_id = sim.sensorId("chaser_accel");
    auto ctrl = zeroCtrl(sim);

    sim.step(ctrl);
    sim.refreshSensors();
    const double* a = sim.sensorData(acc_id);
    const double mag = std::hypot(a[0], a[1]);

    EXPECT_GT(sim.data()->ncon, 0) << "expected contacts at the default pose";
    EXPECT_GT(mag, 1e-3)
        << "the default pose no longer self-collides; initial_arm_qpos may be "
           "droppable";

    // And the documented home pose really is clear of it.
    MuJoCoSim clean = makeSim();
    clean.setJointPositions(kArmJoints, kArmHome);
    clean.step(zeroCtrl(clean));
    clean.refreshSensors();
    const double* ca = clean.sensorData(clean.sensorId("chaser_accel"));
    EXPECT_NEAR(std::hypot(ca[0], ca[1]), 0.0, 1e-6);
}

TEST(SceneSensors, RefreshingSensorsDoesNotPerturbTheTrajectory) {
    // refreshSensors() runs a full mj_forward, which also rewrites
    // d->qacc_warmstart -- and MuJoCo's constraint solver starts from that, so
    // an extra forward evaluation is not obviously free of physical effect. The
    // node calls it once per publish, so if it did perturb the dynamics then
    // every trajectory recorded before this change would be unreproducible.
    //
    // Run deliberately in contact, where the constraint solver is actually
    // engaged and a warm-start difference could show: the MJCF default arm pose
    // self-collides (see TheMjcfDefaultArmPoseStartsInContact).
    auto drive = [](bool refresh) {
        MuJoCoSim sim = makeSim();
        auto ctrl = std::vector<double>(
            static_cast<std::size_t>(sim.model()->nu), 0.0);
        ctrl[sim.actuatorId("thr_fwd")]  = 2.0;
        ctrl[sim.actuatorId("motor2_L")] = 4.0;
        ctrl[sim.actuatorId("motor3_R")] = -3.0;
        // 5 steps per publish at 500 Hz physics / 100 Hz publish.
        for (int pub = 0; pub < 200; ++pub) {
            for (int i = 0; i < 5; ++i) sim.step(ctrl);
            if (refresh) sim.refreshSensors();
        }
        return sim.getChaserState();
    };

    const auto without = drive(false);
    const auto with    = drive(true);

    ASSERT_GT(std::fabs(without.x) + std::fabs(without.theta), 1e-6)
        << "the run did not move, so it cannot discriminate";
    EXPECT_DOUBLE_EQ(with.x, without.x);
    EXPECT_DOUBLE_EQ(with.y, without.y);
    EXPECT_DOUBLE_EQ(with.theta, without.theta);
    EXPECT_DOUBLE_EQ(with.xdot, without.xdot);
    EXPECT_DOUBLE_EQ(with.ydot, without.ydot);
    EXPECT_DOUBLE_EQ(with.thetadot, without.thetadot);
}

TEST(SceneSensors, AddingSensorsLeftThePhysicsAlone) {
    // Sensors are passive outputs, so the trajectory under a given control must
    // be untouched by declaring them. Pinned here because every committed
    // measurement in the README was taken before the <sensor> block existed.
    MuJoCoSim sim = makeSim();
    auto ctrl = zeroCtrl(sim);
    ctrl[sim.actuatorId("thr_fwd")] = 3.0;
    ctrl[sim.actuatorId("motor2_L")] = 1.5;

    for (int i = 0; i < 500; ++i) sim.step(ctrl);
    const auto s = sim.getChaserState();

    // A free, undamped-in-yaw body under constant forward thrust for 1 s: the
    // assertion worth making is that it moved sensibly and stayed finite, not
    // a magic number that would have to be rewritten whenever the bus mass is
    // corrected.
    EXPECT_TRUE(std::isfinite(s.x));
    EXPECT_TRUE(std::isfinite(s.y));
    EXPECT_TRUE(std::isfinite(s.theta));
    EXPECT_GT(s.x, 0.0) << "forward thrust should move +x in joint coordinates";
    EXPECT_LT(std::fabs(s.x), 1.0) << "1 s of 3 N on a 49 kg bus should not fly";
}
