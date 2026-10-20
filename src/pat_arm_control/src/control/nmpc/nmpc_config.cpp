#include "pat_arm_control/control/nmpc/nmpc_config.h"

#include <array>
#include <stdexcept>
#include <string>

namespace pat_arm_control {

namespace {

//! A [x, y, thetaz] weight vector; exactly three entries.
std::array<double, 3> vec3(ParamSource& src, const std::string& key,
                           std::array<double, 3> def) {
    const auto v = src.getDoubles(key, {def[0], def[1], def[2]});
    if (v.size() != 3)
        throw std::runtime_error(key + ": expected 3 entries, got " +
                                 std::to_string(v.size()));
    return {v[0], v[1], v[2]};
}

}  // namespace

NMPCParams loadNmpcParams(ParamSource& src, const ArmJoints& joints) {
    NMPCParams p;
    loadControllerParams(src, joints, p);

    p.Ts            = src.getDouble("nmpc.Ts", p.Ts);
    p.N             = src.getInt("nmpc.N", p.N);
    p.fullNonlinear = src.getBool("nmpc.full_nonlinear", p.fullNonlinear);
    p.sqp_iters     = src.getInt("nmpc.sqp_iters", p.sqp_iters);
    p.Hg_damping    = src.getDouble("nmpc.Hg_damping", p.Hg_damping);
    p.tier1_max_replay_ticks =
        src.getInt("safe_mode.tier1_max_replay_ticks", p.tier1_max_replay_ticks);

    p.rollout_v_clamp_mult =
        src.getDouble("nmpc.rollout_v_clamp_mult", p.rollout_v_clamp_mult);
    p.terminal_velocity_constraint =
        src.getBool("nmpc.terminal_velocity_constraint", p.terminal_velocity_constraint);
    p.terminal_cost_multiplier =
        src.getDouble("nmpc.terminal_cost_multiplier", p.terminal_cost_multiplier);
    p.reference_preview = src.getBool("nmpc.reference_preview", p.reference_preview);

    // [x, y, θz] each: the planar law's three task channels.
    p.Q_pos = vec3(src, "weights.Q_pos", p.Q_pos);
    p.Q_vel = vec3(src, "weights.Q_vel", p.Q_vel);
    p.R     = vec3(src, "weights.R",     p.R);
    p.du_max_frac = src.getDouble("nmpc.du_max_frac", p.du_max_frac);

    p.qp_max_iter    = src.getInt("qp.max_iter", p.qp_max_iter);
    p.qp_tol         = src.getDouble("qp.tol", p.qp_tol);
    p.qp_reg_prim    = src.getDouble("qp.reg_prim", p.qp_reg_prim);
    p.qp_warm_start  = src.getInt("qp.warm_start", p.qp_warm_start);
    p.joint_limit_slack_linear =
        src.getDouble("qp.joint_slack_linear", p.joint_limit_slack_linear);
    p.joint_limit_slack_quadratic =
        src.getDouble("qp.joint_slack_quadratic", p.joint_limit_slack_quadratic);
    p.torque_slack_linear =
        src.getDouble("qp.torque_slack_linear", p.torque_slack_linear);
    p.torque_slack_quadratic =
        src.getDouble("qp.torque_slack_quadratic", p.torque_slack_quadratic);
    p.soft_joint_limits  = src.getBool("qp.soft_joint_limits", p.soft_joint_limits);
    p.soft_torque_limits = src.getBool("qp.soft_torque_limits", p.soft_torque_limits);
    p.relax_box_to_contain_x0 =
        src.getBool("qp.relax_box_to_contain_x0", p.relax_box_to_contain_x0);
    p.relax_box_margin = src.getDouble("qp.relax_box_margin", p.relax_box_margin);

    return p;
}

}  // namespace pat_arm_control
