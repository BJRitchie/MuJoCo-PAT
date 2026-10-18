#include "bench_yaml.h"

#include <stdexcept>

namespace pat_arm_control {
namespace bench {

void flatten(const YAML::Node& node, const std::string& prefix, Flat& out) {
    if (node.IsMap()) {
        for (const auto& kv : node) {
            flatten(kv.second, prefix.empty()
                                   ? kv.first.as<std::string>()
                                   : prefix + "." + kv.first.as<std::string>(), out);
        }
    } else {
        out[prefix] = node;
    }
}

double asDouble(const Flat& f, const std::string& k, double def,
                std::set<std::string>& used) {
    used.insert(k);
    auto it = f.find(k);
    return it == f.end() ? def : it->second.as<double>();
}
int asInt(const Flat& f, const std::string& k, int def, std::set<std::string>& used) {
    used.insert(k);
    auto it = f.find(k);
    return it == f.end() ? def : it->second.as<int>();
}
bool asBool(const Flat& f, const std::string& k, bool def, std::set<std::string>& used) {
    used.insert(k);
    auto it = f.find(k);
    return it == f.end() ? def : it->second.as<bool>();
}
std::vector<double> asVec(const Flat& f, const std::string& k,
                          std::vector<double> def, std::set<std::string>& used) {
    used.insert(k);
    auto it = f.find(k);
    if (it == f.end()) return def;
    std::vector<double> v;
    for (const auto& e : it->second) v.push_back(e.as<double>());
    return v;
}
std::vector<std::string> asStrVec(const Flat& f, const std::string& k,
                                  std::set<std::string>& used) {
    used.insert(k);
    std::vector<std::string> v;
    auto it = f.find(k);
    if (it == f.end()) return v;
    for (const auto& e : it->second) v.push_back(e.as<std::string>());
    return v;
}

Loaded loadConfig(const std::string& yaml_path, const std::string& arm,
                  const std::string& mjcf_override,
                  const std::string& default_model,
                  const std::set<std::string>& not_ours,
                  const std::string& tool) {
    const YAML::Node root = YAML::LoadFile(yaml_path);

    // The shared block plus this arm's, merged the way ROS merges them: the
    // per-node block wins. "/**" is the wildcard every node loads.
    Flat f;
    const char* shared = "/**";
    if (root[shared] && root[shared]["ros__parameters"])
        flatten(root[shared]["ros__parameters"], "", f);
    const std::string node_name = "pat_arm_nmpc_" + arm;
    if (!root[node_name] || !root[node_name]["ros__parameters"]) {
        throw std::runtime_error("no '" + node_name + ": ros__parameters:' block in " +
                                 yaml_path);
    }
    flatten(root[node_name]["ros__parameters"], "", f);

    std::set<std::string> used;
    Loaded L;
    NMPCParams& p = L.params;

    L.ee_site = "arm_ee_site";
    used.insert("ee_site_name");
    if (auto it = f.find("ee_site_name"); it != f.end())
        L.ee_site = it->second.as<std::string>();
    p.ee_site_name = L.ee_site;

    p.owned_joints = asStrVec(f, "owned_joints", used);

    p.Ts            = asDouble(f, "nmpc.Ts", p.Ts, used);
    p.N             = asInt(f, "nmpc.N", p.N, used);
    p.fullNonlinear = asBool(f, "nmpc.full_nonlinear", p.fullNonlinear, used);
    p.sqp_iters     = asInt(f, "nmpc.sqp_iters", p.sqp_iters, used);
    p.Hg_damping    = asDouble(f, "nmpc.Hg_damping", p.Hg_damping, used);
    p.rollout_v_clamp_mult =
        asDouble(f, "nmpc.rollout_v_clamp_mult", p.rollout_v_clamp_mult, used);
    p.du_max_frac   = asDouble(f, "nmpc.du_max_frac", p.du_max_frac, used);
    p.terminal_velocity_constraint =
        asBool(f, "nmpc.terminal_velocity_constraint", p.terminal_velocity_constraint, used);
    p.terminal_cost_multiplier =
        asDouble(f, "nmpc.terminal_cost_multiplier", p.terminal_cost_multiplier, used);
    p.reference_preview =
        asBool(f, "nmpc.reference_preview", p.reference_preview, used);

    p.safe_mode_damping_frac =
        asDouble(f, "safe_mode.damping_frac", p.safe_mode_damping_frac, used);
    p.safe_mode_recovery_ticks =
        asInt(f, "safe_mode.recovery_ticks", p.safe_mode_recovery_ticks, used);
    p.ff_gain     = asDouble(f, "observer.ff_gain", p.ff_gain, used);
    p.ff_max_frac = asDouble(f, "observer.ff_max_frac", p.ff_max_frac, used);
    p.ff_slew     = asDouble(f, "observer.ff_slew", p.ff_slew, used);
    p.tier1_max_replay_ticks =
        asInt(f, "safe_mode.tier1_max_replay_ticks", p.tier1_max_replay_ticks, used);
    p.tier3_latch_after_ticks =
        asInt(f, "safe_mode.tier3_latch_after_ticks", p.tier3_latch_after_ticks, used);

    // The observer's own knobs. Mapped even for a bench that does not run one,
    // so there is a single place the key names live.
    used.insert("observer");
    if (auto it = f.find("observer"); it != f.end())
        L.observer_type = it->second.as<std::string>();
    L.observer_hz   = asDouble(f, "observer.hz", L.observer_hz, used);
    L.observer_gain = asDouble(f, "observer.gain", L.observer_gain, used);
    L.observer_gain_base =
        asDouble(f, "observer.gain_base", L.observer_gain_base, used);
    L.observer_mdot_clamp =
        asDouble(f, "observer.mdot_clamp", L.observer_mdot_clamp, used);
    L.observer_estimate_clamp =
        asDouble(f, "observer.estimate_clamp", L.observer_estimate_clamp, used);

    L.base_anchor_x = asDouble(f, "base_anchor_x", L.base_anchor_x, used);
    L.base_anchor_y = asDouble(f, "base_anchor_y", L.base_anchor_y, used);

    // Weights are [x, y, thetaz] 3-vectors, and the planar law reads the yaw
    // channel through Qwz/Qdotwz/Rwz -- the same remapping the node does, and for
    // the same reason: without it a tuned yaw weight is silently ignored.
    const auto q_pos = asVec(f, "weights.Q_pos", {p.Qx, p.Qy, p.Qz}, used);
    const auto q_vel = asVec(f, "weights.Q_vel", {p.Qdotx, p.Qdoty, p.Qdotz}, used);
    const auto r_tr  = asVec(f, "weights.R",     {p.Rx, p.Ry, p.Rz}, used);
    const auto q_ori    = asVec(f, "weights.Q_ori",    {p.Qwx, p.Qwy, p.Qwz}, used);
    const auto q_angvel = asVec(f, "weights.Q_angvel", {p.Qdotwx, p.Qdotwy, p.Qdotwz}, used);
    const auto r_ori    = asVec(f, "weights.R_ori",    {p.Rwx, p.Rwy, p.Rwz}, used);
    p.Qx = q_pos[0];    p.Qy = q_pos[1];    p.Qz = q_pos[2];
    p.Qdotx = q_vel[0]; p.Qdoty = q_vel[1]; p.Qdotz = q_vel[2];
    p.Rx = r_tr[0];     p.Ry = r_tr[1];     p.Rz = r_tr[2];
    p.Qwx = q_ori[0];   p.Qwy = q_ori[1];
    p.Qdotwx = q_angvel[0]; p.Qdotwy = q_angvel[1];
    p.Rwx = r_ori[0];   p.Rwy = r_ori[1];
    p.Qwz    = q_pos[2];
    p.Qdotwz = q_vel[2];
    p.Rwz    = r_tr[2];

    p.qp_max_iter   = asInt(f, "qp.max_iter", p.qp_max_iter, used);
    p.qp_tol        = asDouble(f, "qp.tol", p.qp_tol, used);
    p.qp_reg_prim   = asDouble(f, "qp.reg_prim", p.qp_reg_prim, used);
    p.qp_warm_start = asInt(f, "qp.warm_start", p.qp_warm_start, used);
    p.joint_limit_slack_linear =
        asDouble(f, "qp.joint_slack_linear", p.joint_limit_slack_linear, used);
    p.joint_limit_slack_quadratic =
        asDouble(f, "qp.joint_slack_quadratic", p.joint_limit_slack_quadratic, used);
    p.torque_slack_linear =
        asDouble(f, "qp.torque_slack_linear", p.torque_slack_linear, used);
    p.torque_slack_quadratic =
        asDouble(f, "qp.torque_slack_quadratic", p.torque_slack_quadratic, used);
    p.soft_joint_limits  = asBool(f, "qp.soft_joint_limits", p.soft_joint_limits, used);
    p.soft_torque_limits = asBool(f, "qp.soft_torque_limits", p.soft_torque_limits, used);
    p.relax_box_to_contain_x0 =
        asBool(f, "qp.relax_box_to_contain_x0", p.relax_box_to_contain_x0, used);
    p.relax_box_margin = asDouble(f, "qp.relax_box_margin", p.relax_box_margin, used);

    // Per-joint limits for the owned joints, and a torque limit for EVERY model
    // joint -- the rollout clamps all of them, including the arm this controller
    // does not own, so model_tau_max has to cover the lot.
    const auto q_min  = asVec(f, "limits.q_min", {}, used);
    const auto q_max  = asVec(f, "limits.q_max", {}, used);
    const auto qd_max = asVec(f, "limits.qd_max", {}, used);
    if (q_min.size() != p.owned_joints.size() || q_max.size() != q_min.size() ||
        qd_max.size() != q_min.size()) {
        throw std::runtime_error("limits.q_min/q_max/qd_max must each have one entry "
                                 "per owned joint");
    }
    for (std::size_t i = 0; i < p.owned_joints.size(); ++i) {
        pat_arm_control::JointLimit jl;
        jl.name   = p.owned_joints[i];
        jl.q_min  = q_min[i];
        jl.q_max  = q_max[i];
        jl.qd_max = qd_max[i];
        p.joint_lims.push_back(jl);
    }

    // One torque limit per model joint, same contract as the node's: the rollout
    // clamps every model joint, including the other arm's, so this array covers
    // all of them and its length has to match model_joint_names exactly.
    L.model_joints       = asStrVec(f, "model_joint_names", used);
    const auto model_tau = asVec(f, "model_tau_max", {}, used);
    if (L.model_joints.empty()) {
        throw std::runtime_error("model_joint_names is required");
    }
    if (model_tau.size() != L.model_joints.size()) {
        throw std::runtime_error(
            "model_tau_max: expected " + std::to_string(L.model_joints.size()) +
            " entries (one per model_joint_names), got " +
            std::to_string(model_tau.size()));
    }
    for (std::size_t i = 0; i < L.model_joints.size(); ++i)
        p.model_torque_lims.emplace_back(L.model_joints[i], model_tau[i]);

    // --model wins, then the params file, then the installed model. nmpc.yaml
    // ships mjcf_path as "" because the launch file substitutes it from
    // FindPackageShare, so in practice the compiled-in default is what gets used.
    L.mjcf = mjcf_override;
    if (L.mjcf.empty()) {
        used.insert("mjcf_path");
        if (auto it = f.find("mjcf_path"); it != f.end())
            L.mjcf = it->second.as<std::string>();
    }
    if (L.mjcf.empty()) L.mjcf = default_model;

    // The drift guard: anything in the file this binary neither mapped nor
    // knowingly skipped. A new knob in nmpc.yaml lands here rather than being
    // silently dropped, which would make the bench measure a configuration
    // nobody deploys.
    std::vector<std::string> unknown;
    for (const auto& [k, v] : f) {
        (void)v;
        if (!used.count(k) && !not_ours.count(k)) unknown.push_back(k);
    }
    if (!unknown.empty()) {
        std::string msg = tool + " does not know these keys in " + yaml_path + ":";
        for (const auto& k : unknown) msg += "\n    " + k;
        msg += "\n  Map them in bench_yaml.cpp's loadConfig, or add them to the "
               "tool's not_ours set if they cannot affect what it measures.";
        throw std::runtime_error(msg);
    }
    return L;
}

}  // namespace bench
}  // namespace pat_arm_control
