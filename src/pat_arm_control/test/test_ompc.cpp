/*! The one-step MPC law (ArmOMPC): its parameters, that it holds a pose it is
 *  already at, that it drives the real planar model to a setpoint, and that its
 *  tick stays off the heap.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "bench_yaml.h"
#include "pat_arm_control/control/ompc/arm_ompc.h"
#include "pat_arm_control/control/ompc/ompc_config.h"
#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

// Re-exports what a test needs from the protected side.
class OmpcProbe : public ArmOMPC {
public:
    OmpcProbe(const std::string& mjcf, const OMPCParams& p) : ArmOMPC(mjcf, p) {}
    using ArmOMPC::integrateStep;
    using ArmOMPC::n_joints_;
    using ArmOMPC::ownedJointInds;
};

//! The left arm on the live limits, with the OMPC's own defaults.
OMPCParams ompcParams()
{
    OMPCParams p;
    static_cast<ControllerParams&>(p) = liveParams();
    p.control_period = 0.010;
    return p;
}

State restState(int n_joints)
{
    State s = referenceState(n_joints);
    s.v.setZero();
    return s;
}

TEST(Ompc, HoldsThePoseItIsAlreadyAt)
{
    OmpcProbe c(kMjcf, ompcParams());
    const State s = restState(c.n_joints_);
    const ArmOMPC::EePose here = c.currentEePose(s.q, s.v);
    c.setDesiredPos(here.pos.x(), here.pos.y(), here.pos.z());
    c.setDesiredOrient(here.quat(0), here.quat(1), here.quat(2), here.quat(3));

    // Zero error and zero velocity: the task force is zero, and at rest the
    // Coriolis term is zero too, so no torque.
    const Eigen::VectorXd tau = c.computeControl(s.q, s.v);
    EXPECT_LT(tau.cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_EQ(c.mode(), ArmOMPC::ControlMode::Nominal);
}

TEST(Ompc, DrivesTheEndEffectorToASetpointOnTheRealModel)
{
    OmpcProbe c(kMjcf, ompcParams());
    State s = restState(c.n_joints_);
    const ArmOMPC::EePose start = c.currentEePose(s.q, s.v);
    const double tx = start.pos.x() + 0.06, ty = start.pos.y() + 0.06;
    c.setDesiredPos(tx, ty, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    const double dt = 0.010;
    const double err0 = std::hypot(tx - start.pos.x(), ty - start.pos.y());
    double err = err0, err_max = err0;
    for (int t = 0; t < 1000; ++t) {
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        ASSERT_TRUE(tau_all.allFinite());
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;
        const ArmOMPC::EePose now = c.currentEePose(s.q, s.v);
        err = std::hypot(tx - now.pos.x(), ty - now.pos.y());
        err_max = std::max(err_max, err);
    }
    std::printf("[ompc] step response: start %.4f m, end %.4f m, worst %.4f m\n",
                err0, err, err_max);
    EXPECT_EQ(c.mode(), ArmOMPC::ControlMode::Nominal);
    EXPECT_LT(err, 0.25 * err0) << "start error " << err0 << " m, end error " << err << " m";
    // No overshoot beyond the start offset: the law is meant to approach, not ring.
    EXPECT_LT(err_max, 1.5 * err0);
}

TEST(Ompc, ARefusedParameterSetNamesTheProblem)
{
    {
        OMPCParams p = ompcParams();
        p.Ts = 0.0;
        EXPECT_THROW(ArmOMPC(kMjcf, p), std::runtime_error);
    }
    {
        // Q and R both zero on an axis: the gain matrix is singular there.
        OMPCParams p = ompcParams();
        p.Q = {0.0, 1.0, 1.0};
        p.R = {0.0, 1.0, 1.0};
        EXPECT_THROW(ArmOMPC(kMjcf, p), std::runtime_error);
    }
    {
        OMPCParams p = ompcParams();
        p.R = {1.0, -1.0, 1.0};
        EXPECT_THROW(ArmOMPC(kMjcf, p), std::runtime_error);
    }
}

TEST(Ompc, ControlTickAllocationCountIsBounded)
{
    constexpr long kTicks = 10;
    constexpr long kMaxPerTick = 8;   // the returned vector is the one expected

    OmpcProbe c(kMjcf, ompcParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 3; ++i) c.computeControl(s.q, s.v);   // one-off sizing

    long allocs = 0;
    {
        AllocScope scope;
        for (long i = 0; i < kTicks; ++i) c.computeControl(s.q, s.v);
        allocs = scope.n();
    }
    const double per_tick = static_cast<double>(allocs) / kTicks;
    std::printf("[alloc] ompc: %.0f heap allocations per control tick\n", per_tick);
    EXPECT_GT(allocs, 0) << "the malloc interposer is not being reached";
    EXPECT_LT(per_tick, kMaxPerTick);
}

TEST(Ompc, ProfileHasNoSolverAndTheTraceRecordsEveryTick)
{
    OmpcProbe c(kMjcf, ompcParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);
    c.setTickBudgetUs(10000.0);
    c.enableTrace(64);
    for (int i = 0; i < 20; ++i) c.computeControl(s.q, s.v);

    ArmOMPC::ProfileSnapshot snap;
    c.snapshotProfile(snap);
    EXPECT_EQ(snap.tick.count, 20);
    EXPECT_EQ(snap.qp_max_iter, 0);
    EXPECT_EQ(snap.trace_size, 20);
    const std::string report = ArmOMPC::formatProfile(snap);
    EXPECT_EQ(report.find("peak slack"), std::string::npos) << report;
    EXPECT_FALSE(c.hasSolver());

    std::vector<ArmOMPC::TickTrace> rows;
    c.drainTrace(rows);
    EXPECT_EQ(rows.size(), 20u);
    c.resetProfile();
    c.snapshotProfile(snap);
    EXPECT_EQ(snap.tick.count, 0);
}

TEST(Ompc, TheShippedFileLoadsForBothArms)
{
    // What the node reads but no loader maps: the node's own runtime settings.
    const std::set<std::string> node_only = {
        "control_hz", "arm_side", "rt_priority", "lock_memory", "rt_cpus",
        "rt_priority_observer", "rt_cpus_observer", "trace_file",
        "trace_capacity", "pub_rate_hz"};
    const YAML::Node common = YAML::LoadFile(PAT_NMPC_TEST_COMMON);
    const YAML::Node law    = YAML::LoadFile(PAT_OMPC_TEST_PARAMS);

    for (const std::string arm : {"left", "right"}) {
        bench::Flat f;
        bench::flatten(common["/**"]["ros__parameters"], "", f);
        bench::flatten(law["/**"]["ros__parameters"], "", f);
        bench::flatten(law["pat_arm_ompc_" + arm]["ros__parameters"], "", f);

        bench::YamlParamSource src(f);
        const ArmJoints joints = loadArmJoints(src);
        const OMPCParams p = loadOmpcParams(src, joints);
        // The rest of what every law's node reads from the shared file.
        loadObserverConfig(src);
        loadBaseAnchor(src);
        src.getString("mjcf_path", "");

        EXPECT_EQ(joints.owned.size(), 3u) << arm;
        EXPECT_EQ(joints.model.size(), 6u) << arm;
        EXPECT_EQ(p.joint_lims.size(), 3u) << arm;
        EXPECT_EQ(p.ee_site_name, arm == "left" ? "ee_site_L" : "ee_site_R");
        EXPECT_DOUBLE_EQ(p.Ts, 1.0);
        EXPECT_DOUBLE_EQ(p.Q[0], 1.0e5);
        EXPECT_DOUBLE_EQ(p.R[2], 1.0);
        EXPECT_DOUBLE_EQ(p.damping, 1.0e-4);

        // The drift guard: a key in the file that nothing read.
        for (const auto& [k, v] : f) {
            (void)v;
            EXPECT_TRUE(src.used().count(k) || node_only.count(k))
                << "unmapped key in ompc.yaml / arm_common.yaml: " << k;
        }
    }
}

TEST(Ompc, ALoadedParameterReachesItsField)
{
    bench::Flat f;
    YAML::Node n = YAML::Load(R"(
joint_names: [joint2_L]
model_joint_names: [joint2_L, joint3_L]
model_tau_max: [0.25, 0.25]
limits: {q_min: [0.0], q_max: [1.0], qd_max: [1.0]}
ompc: {Ts: 0.7, Q: [1.0, 2.0, 3.0], R: [4.0, 5.0, 6.0], damping: 0.5, Hg_damping: 0.25}
)");
    bench::flatten(n, "", f);
    bench::YamlParamSource src(f);
    const ArmJoints joints = loadArmJoints(src);
    const OMPCParams p = loadOmpcParams(src, joints);
    EXPECT_DOUBLE_EQ(p.Ts, 0.7);
    EXPECT_EQ(p.Q, (std::array<double, 3>{1.0, 2.0, 3.0}));
    EXPECT_EQ(p.R, (std::array<double, 3>{4.0, 5.0, 6.0}));
    EXPECT_DOUBLE_EQ(p.damping, 0.5);
    EXPECT_DOUBLE_EQ(p.Hg_damping, 0.25);

    bench::Flat bad;
    bench::flatten(YAML::Load("joint_names: [a]\nmodel_joint_names: [a]\n"
                              "model_tau_max: [1.0]\n"
                              "limits: {q_min: [0.0], q_max: [1.0], qd_max: [1.0]}\n"
                              "ompc: {Q: [1.0, 2.0]}\n"), "", bad);
    bench::YamlParamSource bsrc(bad);
    EXPECT_THROW(loadOmpcParams(bsrc, loadArmJoints(bsrc)), std::runtime_error);
}

}  // namespace
}  // namespace pat_arm_control
