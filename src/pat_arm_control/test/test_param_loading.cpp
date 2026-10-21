/*! The configuration loaders: nmpc.yaml's keys -> the controller's parameters.
 *
 *  The node (through ROS parameters) and the benches (through yaml-cpp) both
 *  run these loaders, so this is the one place the mapping is pinned. Each key
 *  gets a value no other key has, so a key wired to the wrong field, or two
 *  keys wired to one, fails here rather than quietly changing what runs.
 */
#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

#include "bench_yaml.h"

namespace pat_arm_control {
namespace {

using bench::loadConfig;
using bench::Loaded;

//! Write `body` to a fresh file and return its path.
std::string writeYaml(const std::string& name, const std::string& body) {
    const std::string path = ::testing::TempDir() + name;
    std::ofstream(path) << body;
    return path;
}

Loaded load(const std::vector<std::string>& paths, const std::string& arm = "left",
            const std::set<std::string>& not_ours = {}) {
    return loadConfig(paths, arm, "model.xml", "default.xml", not_ours, "test");
}

Loaded load(const std::string& path, const std::string& arm = "left",
            const std::set<std::string>& not_ours = {}) {
    return load(std::vector<std::string>{path}, arm, not_ours);
}

/*! Every key the loaders map, each with a distinct value. Shared settings in
 *  the wildcard block, per-arm ones in the node block, as nmpc.yaml has them. */
const char* kEveryKey = R"(
/**:
  ros__parameters:
    model_joint_names: [a1, a2, a3, b1, b2, b3]
    model_tau_max: [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    base_anchor_x: -0.51
    base_anchor_y: 0.52
    safe_mode:
      damping_frac: 0.61
      recovery_ticks: 62
      tier1_max_replay_ticks: 63
      tier3_latch_after_ticks: 64
    observer:
      hz: 71.0
      gain: 72.0
      gain_base: 73.0
      mdot_clamp: 74.0
      estimate_clamp: 75.0
      ff_gain: 0.76
      ff_max_frac: 0.77
      ff_slew: 78.0
    nmpc:
      Ts: 0.081
      N: 82
      full_nonlinear: false
      sqp_iters: 83
      Hg_damping: 0.084
      rollout_v_clamp_mult: 8.5
      du_max_frac: 0.86
      terminal_velocity_constraint: true
      terminal_cost_multiplier: 87.0
      reference_preview: false
    weights:
      Q_pos:    [1.1, 1.2, 1.3]
      Q_vel:    [2.1, 2.2, 2.3]
      R:        [3.1, 3.2, 3.3]
    qp:
      max_iter: 91
      tol: 0.092
      reg_prim: 0.093
      warm_start: 1
      joint_slack_linear: 95.0
      joint_slack_quadratic: 96.0
      torque_slack_linear: 97.0
      torque_slack_quadratic: 98.0
      soft_joint_limits: false
      soft_torque_limits: true
      relax_box_to_contain_x0: true
      relax_box_margin: 0.099
pat_arm_nmpc_left:
  ros__parameters:
    ee_site_name: site_left
    joint_names:  [a1, a2, a3]
    owned_joints: [a1, a2, a3]
    limits:
      q_min:  [-1.1, -1.2, -1.3]
      q_max:  [ 1.1,  1.2,  1.3]
      qd_max: [ 2.1,  2.2,  2.3]
)";

TEST(ParamLoading, EveryKeyReachesItsOwnField) {
    const Loaded L = load(writeYaml("every_key.yaml", kEveryKey));
    const NMPCParams& p = L.params;

    EXPECT_EQ(p.ee_site_name, "site_left");
    EXPECT_EQ(p.owned_joints, (std::vector<std::string>{"a1", "a2", "a3"}));
    EXPECT_EQ(L.joints.owned, (std::vector<std::string>{"a1", "a2", "a3"}));
    EXPECT_EQ(L.joints.model.size(), 6u);

    EXPECT_DOUBLE_EQ(p.Ts, 0.081);
    EXPECT_EQ(p.N, 82);
    EXPECT_FALSE(p.fullNonlinear);
    EXPECT_EQ(p.sqp_iters, 83);
    EXPECT_DOUBLE_EQ(p.Hg_damping, 0.084);
    EXPECT_DOUBLE_EQ(p.rollout_v_clamp_mult, 8.5);
    EXPECT_DOUBLE_EQ(p.du_max_frac, 0.86);
    EXPECT_TRUE(p.terminal_velocity_constraint);
    EXPECT_DOUBLE_EQ(p.terminal_cost_multiplier, 87.0);
    EXPECT_FALSE(p.reference_preview);

    EXPECT_DOUBLE_EQ(p.safe_mode_damping_frac, 0.61);
    EXPECT_EQ(p.safe_mode_recovery_ticks, 62);
    EXPECT_EQ(p.tier1_max_replay_ticks, 63);
    EXPECT_EQ(p.tier3_latch_after_ticks, 64);

    EXPECT_DOUBLE_EQ(p.ff_gain, 0.76);
    EXPECT_DOUBLE_EQ(p.ff_max_frac, 0.77);
    EXPECT_DOUBLE_EQ(p.ff_slew, 78.0);
    EXPECT_DOUBLE_EQ(L.observer.hz, 71.0);
    EXPECT_DOUBLE_EQ(L.observer.gain, 72.0);
    EXPECT_DOUBLE_EQ(L.observer.gain_base, 73.0);
    EXPECT_DOUBLE_EQ(L.observer.mdot_clamp, 74.0);
    EXPECT_DOUBLE_EQ(L.observer.estimate_clamp, 75.0);

    EXPECT_DOUBLE_EQ(L.anchor.x, -0.51);
    EXPECT_DOUBLE_EQ(L.anchor.y, 0.52);

    // The planar weights, [x, y, θz] each, in order.
    EXPECT_EQ(p.Q_pos, (std::array<double, 3>{1.1, 1.2, 1.3}));
    EXPECT_EQ(p.Q_vel, (std::array<double, 3>{2.1, 2.2, 2.3}));
    EXPECT_EQ(p.R,     (std::array<double, 3>{3.1, 3.2, 3.3}));

    EXPECT_EQ(p.qp_max_iter, 91);
    EXPECT_DOUBLE_EQ(p.qp_tol, 0.092);
    EXPECT_DOUBLE_EQ(p.qp_reg_prim, 0.093);
    EXPECT_EQ(p.qp_warm_start, 1);
    EXPECT_DOUBLE_EQ(p.joint_limit_slack_linear, 95.0);
    EXPECT_DOUBLE_EQ(p.joint_limit_slack_quadratic, 96.0);
    EXPECT_DOUBLE_EQ(p.torque_slack_linear, 97.0);
    EXPECT_DOUBLE_EQ(p.torque_slack_quadratic, 98.0);
    EXPECT_FALSE(p.soft_joint_limits);
    EXPECT_TRUE(p.soft_torque_limits);
    EXPECT_TRUE(p.relax_box_to_contain_x0);
    EXPECT_DOUBLE_EQ(p.relax_box_margin, 0.099);

    ASSERT_EQ(p.joint_lims.size(), 3u);
    EXPECT_EQ(p.joint_lims[1].name, "a2");
    EXPECT_DOUBLE_EQ(p.joint_lims[1].q_min, -1.2);
    EXPECT_DOUBLE_EQ(p.joint_lims[1].q_max, 1.2);
    EXPECT_DOUBLE_EQ(p.joint_lims[1].qd_max, 2.2);
    ASSERT_EQ(p.model_torque_lims.size(), 6u);
    EXPECT_EQ(p.model_torque_lims[4].first, "b2");
    EXPECT_DOUBLE_EQ(p.model_torque_lims[4].second, 0.15);

    // The control period is the loop's, never the file's.
    EXPECT_DOUBLE_EQ(p.control_period, NMPCParams{}.control_period);
    EXPECT_EQ(L.mjcf, "model.xml");
}

TEST(ParamLoading, TheObserverTypeIsItsOwnKey) {
    const std::string path = writeYaml("observer_type.yaml", R"(
/**:
  ros__parameters:
    model_joint_names: [a1]
    model_tau_max: [0.1]
    observer: momentum
pat_arm_nmpc_left:
  ros__parameters:
    joint_names: [a1]
    limits: {q_min: [-1.0], q_max: [1.0], qd_max: [1.0]}
)");
    EXPECT_EQ(load(path).observer.type, "momentum");
}

TEST(ParamLoading, ThePerNodeBlockWinsOverTheShared) {
    const std::string path = writeYaml("override.yaml", R"(
/**:
  ros__parameters:
    model_joint_names: [a1]
    model_tau_max: [0.1]
    nmpc: {N: 10}
pat_arm_nmpc_left:
  ros__parameters:
    joint_names: [a1]
    nmpc: {N: 20}
    limits: {q_min: [-1.0], q_max: [1.0], qd_max: [1.0]}
)");
    EXPECT_EQ(load(path).params.N, 20);
}

TEST(ParamLoading, TheShippedFileLoadsForBothArms) {
    // What the node reads but no loader maps: the node's own runtime settings.
    const std::set<std::string> node_only = {
        "control_hz", "arm_side", "rt_priority", "lock_memory", "rt_cpus",
        "rt_priority_observer", "rt_cpus_observer", "trace_file",
        "trace_capacity", "pub_rate_hz"};
    for (const std::string arm : {"left", "right"}) {
        const Loaded L = load(std::vector<std::string>{PAT_NMPC_TEST_COMMON,
                                                       PAT_NMPC_TEST_PARAMS},
                              arm, node_only);
        EXPECT_EQ(L.joints.owned.size(), 3u) << arm;
        EXPECT_EQ(L.joints.model.size(), 6u) << arm;
        EXPECT_EQ(L.params.joint_lims.size(), 3u) << arm;
        EXPECT_EQ(L.params.ee_site_name, arm == "left" ? "ee_site_L" : "ee_site_R");
    }
}

TEST(ParamLoading, EveryLawsShippedFileLoadsThroughTheBenchLoader) {
    // The benches pick a law's node block and loader by name. Each law's shipped
    // file, with arm_common.yaml, has to load for both arms with every key
    // mapped -- the same check the NMPC's file gets above.
    const std::set<std::string> node_only = {
        "control_hz", "arm_side", "rt_priority", "lock_memory", "rt_cpus",
        "rt_priority_observer", "rt_cpus_observer", "trace_file",
        "trace_capacity", "pub_rate_hz"};
    const struct { const char* law; const char* file; } laws[] = {
        {"nmpc", PAT_NMPC_TEST_PARAMS},
        {"ompc", PAT_OMPC_TEST_PARAMS},
        {"smc",  PAT_SMC_TEST_PARAMS}};
    for (const auto& l : laws) {
        for (const std::string arm : {"left", "right"}) {
            const Loaded L = loadConfig(
                std::vector<std::string>{PAT_NMPC_TEST_COMMON, l.file}, arm,
                "model.xml", "default.xml", node_only, "test", l.law);
            EXPECT_EQ(L.law, l.law);
            EXPECT_EQ(L.common().joint_lims.size(), 3u) << l.law << " " << arm;
            EXPECT_EQ(L.common().ee_site_name, arm == "left" ? "ee_site_L" : "ee_site_R")
                << l.law << " " << arm;
        }
    }
}

TEST(ParamLoading, AnUnknownLawIsRefused) {
    EXPECT_THROW(loadConfig(std::vector<std::string>{PAT_NMPC_TEST_COMMON,
                                                     PAT_NMPC_TEST_PARAMS},
                            "left", "model.xml", "default.xml", {}, "test", "lqr"),
                 std::runtime_error);
}

TEST(ParamLoading, AKeyNothingMapsIsRefused) {
    const std::string path = writeYaml("unknown.yaml", R"(
/**:
  ros__parameters:
    model_joint_names: [a1]
    model_tau_max: [0.1]
    nmpc: {N: 10, horizn: 3}
pat_arm_nmpc_left:
  ros__parameters:
    joint_names: [a1]
    limits: {q_min: [-1.0], q_max: [1.0], qd_max: [1.0]}
)");
    try {
        load(path);
        FAIL() << "a misspelt key was accepted";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("nmpc.horizn"), std::string::npos)
            << e.what();
    }
}

TEST(ParamLoading, MisshapenValuesAreRefused) {
    auto withNode = [](const std::string& shared, const std::string& node) {
        return "/**:\n  ros__parameters:\n" + shared +
               "pat_arm_nmpc_left:\n  ros__parameters:\n" + node;
    };
    const std::string model = "    model_joint_names: [a1, a2]\n"
                              "    model_tau_max: [0.1, 0.1]\n";
    const std::string limits = "    limits: {q_min: [-1.0], q_max: [1.0], qd_max: [1.0]}\n";
    const std::string good_node = "    joint_names: [a1]\n" + limits;

    // A weight vector that is not [x, y, thetaz].
    EXPECT_THROW(load(writeYaml("w.yaml", withNode(
                     model + "    weights: {Q_pos: [1.0, 2.0]}\n", good_node))),
                 std::runtime_error);
    // A torque limit missing for one model joint.
    EXPECT_THROW(load(writeYaml("t.yaml", withNode(
                     "    model_joint_names: [a1, a2]\n    model_tau_max: [0.1]\n",
                     good_node))),
                 std::runtime_error);
    // A limit array not one per owned joint.
    EXPECT_THROW(load(writeYaml("l.yaml", withNode(model,
                     "    joint_names: [a1, a2]\n" + limits))),
                 std::runtime_error);
    // q_min above q_max.
    EXPECT_THROW(load(writeYaml("q.yaml", withNode(model,
                     "    joint_names: [a1]\n"
                     "    limits: {q_min: [1.0], q_max: [-1.0], qd_max: [1.0]}\n"))),
                 std::runtime_error);
    // An owned joint the model does not have.
    EXPECT_THROW(load(writeYaml("j.yaml", withNode(model,
                     "    joint_names: [c9]\n" + limits))),
                 std::runtime_error);
}

}  // namespace
}  // namespace pat_arm_control
