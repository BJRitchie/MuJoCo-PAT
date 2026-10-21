/*! The sliding-mode law (ArmSMC): its parameters, that it holds a pose it is
 *  already at, that it drives the real planar model to a setpoint and rejects a
 *  constant disturbance, and that its signs are the right ones.
 *
 *  The signs are the point of the closed-loop tests: the law is derived, not
 *  copied from VORTEX (whose signs cannot give a stable surface), so the
 *  evidence it is right is the plant converging under it and diverging under
 *  its negation.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "bench_yaml.h"
#include "pat_arm_control/control/smc/arm_smc.h"
#include "pat_arm_control/control/smc/smc_config.h"
#include "support/alloc_trace.h"
#include "support/nmpc_fixtures.h"

namespace pat_arm_control {
namespace {
using namespace test_support;

class SmcProbe : public ArmSMC {
public:
    SmcProbe(const std::string& mjcf, const SMCParams& p) : ArmSMC(mjcf, p) {}
    using ArmSMC::integrateStep;
    using ArmSMC::n_joints_;
    using ArmSMC::ownedJointInds;
    using ArmSMC::pinvDLS;
};

//! The same law with the opposite sign: what a sign error would do.
class NegatedSmc : public SmcProbe {
public:
    using SmcProbe::SmcProbe;
protected:
    Eigen::Vector3d taskForce(const TaskState& s) override {
        return -ArmSMC::taskForce(s);
    }
};

SMCParams smcParams()
{
    SMCParams p;
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

struct StepResult {
    double err0 = 0.0, err = 0.0, err_max = 0.0;
    bool finite = true;
    ArmSMC::ControlMode mode = ArmSMC::ControlMode::Nominal;
};

/*! A 6 cm step in x and y from rest, for `ticks` at 100 Hz. `disturbance` is a
 *  constant extra torque [N*m] on the first owned joint that the controller
 *  does not know about. */
template <class Law>
StepResult runStep(Law& c, int ticks, double disturbance = 0.0)
{
    State s = restState(c.n_joints_);
    const ArmSMC::EePose start = c.currentEePose(s.q, s.v);
    const double tx = start.pos.x() + 0.06, ty = start.pos.y() + 0.06;
    c.setDesiredPos(tx, ty, start.pos.z());
    c.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    StepResult r;
    r.err0 = r.err = r.err_max = std::hypot(tx - start.pos.x(), ty - start.pos.y());
    const double dt = 0.010;
    for (int t = 0; t < ticks; ++t) {
        const Eigen::VectorXd tau_all = c.computeControl(s.q, s.v);
        if (!tau_all.allFinite()) { r.finite = false; break; }
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(c.n_joints_);
        for (int idx : c.ownedJointInds) tau[idx] = tau_all[idx];
        tau[c.ownedJointInds[0]] += disturbance;
        Eigen::VectorXd q1, v1;
        c.integrateStep(s.q, s.v, tau, dt, q1, v1);
        s.q = q1;
        s.v = v1;
        const ArmSMC::EePose now = c.currentEePose(s.q, s.v);
        r.err = std::hypot(tx - now.pos.x(), ty - now.pos.y());
        r.err_max = std::max(r.err_max, r.err);
        if (!std::isfinite(r.err)) { r.finite = false; break; }
    }
    r.mode = c.mode();
    return r;
}

TEST(Smc, HoldsThePoseItIsAlreadyAt)
{
    SmcProbe c(kMjcf, smcParams());
    const State s = restState(c.n_joints_);
    const ArmSMC::EePose here = c.currentEePose(s.q, s.v);
    c.setDesiredPos(here.pos.x(), here.pos.y(), here.pos.z());
    c.setDesiredOrient(here.quat(0), here.quat(1), here.quat(2), here.quat(3));

    // On the surface (sigma = 0) the switching term is zero, and at rest the
    // feedforward and the bias are zero: no torque.
    const Eigen::VectorXd tau = c.computeControl(s.q, s.v);
    EXPECT_LT(tau.cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_EQ(c.mode(), ArmSMC::ControlMode::Nominal);
}

TEST(Smc, DrivesTheEndEffectorToASetpointOnTheRealModel)
{
    SmcProbe c(kMjcf, smcParams());
    const StepResult r = runStep(c, 1000);
    std::printf("[smc] step response: start %.4f m, end %.4f m, worst %.4f m\n",
                r.err0, r.err, r.err_max);
    ASSERT_TRUE(r.finite);
    EXPECT_EQ(r.mode, ArmSMC::ControlMode::Nominal);
    EXPECT_LT(r.err, 0.25 * r.err0);
    EXPECT_LT(r.err_max, 1.5 * r.err0);
}

TEST(Smc, RejectsAConstantDisturbanceItDoesNotKnowAbout)
{
    // 0.05 N*m, a fifth of the torque limit, on one joint, unmodelled.
    SmcProbe c(kMjcf, smcParams());
    const StepResult r = runStep(c, 1000, 0.05);
    std::printf("[smc] with a 0.05 N*m disturbance: start %.4f m, end %.4f m\n",
                r.err0, r.err);
    ASSERT_TRUE(r.finite);
    EXPECT_LT(r.err, 0.25 * r.err0);
}

TEST(Smc, TheNegatedLawDoesNotConverge)
{
    // The guard on the signs: the same law with the opposite sign must be
    // worse than the start, or the convergence test above proves nothing about
    // which way the force points.
    NegatedSmc c(kMjcf, smcParams());
    const StepResult r = runStep(c, 300);
    std::printf("[smc] negated law: start %.4f m, end %.4f m\n", r.err0, r.err);
    EXPECT_TRUE(!r.finite || r.err > r.err0);
}

TEST(Smc, TheTaskInertiaIsTheDampedPseudoInverse)
{
    // The law forms L = (L^-2 + d^2 I)^-1 L^-1 as a 3x3 solve; it has to be the
    // SVD damped pseudo-inverse the VORTEX law (and the tests' pinvDLS) mean.
    SmcProbe c(kMjcf, smcParams());
    Eigen::Matrix3d R = Eigen::Matrix3d::Random();
    Eigen::Matrix3d Lam_inv = R * R.transpose();
    Lam_inv(2, 2) *= 1e-6;   // a near-singular axis, where the damping matters
    const double d = 0.02;
    Eigen::Matrix3d M = Lam_inv * Lam_inv;
    M.diagonal().array() += d * d;
    const Eigen::Matrix3d direct = M.ldlt().solve(Lam_inv);
    const Eigen::MatrixXd svd = SmcProbe::pinvDLS(Lam_inv, d);
    EXPECT_LT((direct - svd).cwiseAbs().maxCoeff(), 1e-9);
}

TEST(Smc, ARefusedParameterSetNamesTheProblem)
{
    {
        SMCParams p = smcParams();
        p.epsilon = 0.0;
        EXPECT_THROW(ArmSMC(kMjcf, p), std::runtime_error);
    }
    {
        SMCParams p = smcParams();
        p.lam = {3.0, -1.0, 3.0};
        EXPECT_THROW(ArmSMC(kMjcf, p), std::runtime_error);
    }
    {
        SMCParams p = smcParams();
        p.k2 = std::nan("");
        EXPECT_THROW(ArmSMC(kMjcf, p), std::runtime_error);
    }
}

TEST(Smc, ControlTickAllocationCountIsBounded)
{
    constexpr long kTicks = 10;
    constexpr long kMaxPerTick = 8;   // the returned vector is the one expected

    SmcProbe c(kMjcf, smcParams());
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
    std::printf("[alloc] smc: %.0f heap allocations per control tick\n", per_tick);
    EXPECT_GT(allocs, 0) << "the malloc interposer is not being reached";
    EXPECT_LT(per_tick, kMaxPerTick);
}

/*! Where a steady-state tick's allocations come from, by call site: run with
 *  --gtest_also_run_disabled_tests, then resolve each offset with
 *  `addr2line -f -C -i -e <object> <offset>`. The same study as the NMPC's
 *  DISABLED_AllocStudy.Attribution, on the closed-form law. */
TEST(DISABLED_SmcAllocStudy, Attribution)
{
    SmcProbe c(kMjcf, smcParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 5; ++i) c.computeControl(s.q, s.v);

    AllocTrace trace;
    c.computeControl(s.q, s.v);
    trace.stop();
    std::printf("\none steady-state tick: %ld allocations\n", trace.n());
    std::printf("%-40s %18s %9s\n", "object", "offset", "allocs");
    for (const auto& st : trace.bySite()) {
        std::printf("%-40s %#18" PRIxPTR " %9ld\n",
                    st.object.c_str(), st.offset, st.count);
    }
}

TEST(Smc, ProfileHasNoSolverAndReportsTheLawPhase)
{
    SmcProbe c(kMjcf, smcParams());
    const State s = referenceState(c.n_joints_);
    c.setDesiredPos(0.30, 0.25, 0.0);
    c.setDesiredOrient(1.0, 0.0, 0.0, 0.0);
    c.setTickBudgetUs(10000.0);
    c.enableTrace(64);
    for (int i = 0; i < 20; ++i) c.computeControl(s.q, s.v);

    ArmSMC::ProfileSnapshot snap;
    c.snapshotProfile(snap);
    EXPECT_EQ(snap.tick.count, 20);
    EXPECT_EQ(snap.law.count, 20);
    EXPECT_EQ(snap.qp_max_iter, 0);
    const std::string report = ArmSMC::formatProfile(snap);
    EXPECT_EQ(report.find("peak slack"), std::string::npos) << report;
    EXPECT_NE(report.find("law(n=20"), std::string::npos) << report;
    EXPECT_FALSE(c.hasSolver());

    std::vector<ArmSMC::TickTrace> rows;
    c.drainTrace(rows);
    EXPECT_EQ(rows.size(), 20u);
}

TEST(Smc, TheShippedFileLoadsForBothArms)
{
    // What the node reads but no loader maps: the node's own runtime settings.
    const std::set<std::string> node_only = {
        "control_hz", "arm_side", "rt_priority", "lock_memory", "rt_cpus",
        "rt_priority_observer", "rt_cpus_observer", "trace_file",
        "trace_capacity", "pub_rate_hz"};
    const YAML::Node common = YAML::LoadFile(PAT_NMPC_TEST_COMMON);
    const YAML::Node law    = YAML::LoadFile(PAT_SMC_TEST_PARAMS);

    for (const std::string arm : {"left", "right"}) {
        bench::Flat f;
        bench::flatten(common["/**"]["ros__parameters"], "", f);
        bench::flatten(law["/**"]["ros__parameters"], "", f);
        bench::flatten(law["pat_arm_smc_" + arm]["ros__parameters"], "", f);

        bench::YamlParamSource src(f);
        const ArmJoints joints = loadArmJoints(src);
        const SMCParams p = loadSmcParams(src, joints);
        loadObserverConfig(src);
        loadBaseAnchor(src);
        src.getString("mjcf_path", "");

        EXPECT_EQ(joints.owned.size(), 3u) << arm;
        EXPECT_EQ(joints.model.size(), 6u) << arm;
        EXPECT_EQ(p.joint_lims.size(), 3u) << arm;
        EXPECT_EQ(p.ee_site_name, arm == "left" ? "ee_site_L" : "ee_site_R");
        EXPECT_DOUBLE_EQ(p.lam[0], 3.0);
        EXPECT_DOUBLE_EQ(p.alpha[2], 0.5);
        EXPECT_DOUBLE_EQ(p.epsilon, 0.05);
        EXPECT_DOUBLE_EQ(p.lambda_damping, 0.02);

        // The drift guard: a key in the file that nothing read.
        for (const auto& [k, v] : f) {
            (void)v;
            EXPECT_TRUE(src.used().count(k) || node_only.count(k))
                << "unmapped key in smc.yaml / arm_common.yaml: " << k;
        }
    }
}

TEST(Smc, ALoadedParameterReachesItsField)
{
    bench::Flat f;
    bench::flatten(YAML::Load(R"(
joint_names: [joint2_L]
model_joint_names: [joint2_L, joint3_L]
model_tau_max: [0.25, 0.25]
limits: {q_min: [0.0], q_max: [1.0], qd_max: [1.0]}
smc: {lam: [1.0, 2.0, 3.0], alpha: [4.0, 5.0, 6.0], epsilon: 0.7, pi_bound: 0.11,
      k2: 0.22, b2: 0.33, tau_bound: 0.44, lambda_damping: 0.55, Hg_damping: 0.66}
)"), "", f);
    bench::YamlParamSource src(f);
    const SMCParams p = loadSmcParams(src, loadArmJoints(src));
    EXPECT_EQ(p.lam, (std::array<double, 3>{1.0, 2.0, 3.0}));
    EXPECT_EQ(p.alpha, (std::array<double, 3>{4.0, 5.0, 6.0}));
    EXPECT_DOUBLE_EQ(p.epsilon, 0.7);
    EXPECT_DOUBLE_EQ(p.pi_bound, 0.11);
    EXPECT_DOUBLE_EQ(p.k2, 0.22);
    EXPECT_DOUBLE_EQ(p.b2, 0.33);
    EXPECT_DOUBLE_EQ(p.tau_bound, 0.44);
    EXPECT_DOUBLE_EQ(p.lambda_damping, 0.55);
    EXPECT_DOUBLE_EQ(p.Hg_damping, 0.66);

    bench::Flat bad;
    bench::flatten(YAML::Load("joint_names: [a]\nmodel_joint_names: [a]\n"
                              "model_tau_max: [1.0]\n"
                              "limits: {q_min: [0.0], q_max: [1.0], qd_max: [1.0]}\n"
                              "smc: {lam: [1.0]}\n"), "", bad);
    bench::YamlParamSource bsrc(bad);
    EXPECT_THROW(loadSmcParams(bsrc, loadArmJoints(bsrc)), std::runtime_error);
}

}  // namespace
}  // namespace pat_arm_control
