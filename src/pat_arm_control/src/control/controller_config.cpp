#include "pat_arm_control/control/controller_config.h"

#include <algorithm>
#include <stdexcept>

namespace pat_arm_control {

namespace {

//! A per-joint array, one entry per name in `names`.
std::vector<double> perJoint(ParamSource& src, const std::string& key,
                             const std::vector<std::string>& names,
                             const char* names_key) {
    const auto v = src.getDoubles(key, {});
    if (v.size() != names.size())
        throw std::runtime_error(
            key + ": expected " + std::to_string(names.size()) + " entries (one per " +
            names_key + "), got " + std::to_string(v.size()));
    return v;
}

}  // namespace

ArmJoints loadArmJoints(ParamSource& src) {
    ArmJoints j;
    j.owned = src.getStrings("joint_names", {});
    j.model = src.getStrings("model_joint_names", {});
    if (j.owned.empty())
        throw std::runtime_error("'joint_names' must be non-empty");
    if (j.model.empty())
        throw std::runtime_error("'model_joint_names' must be non-empty");
    for (const auto& name : j.owned) {
        if (std::find(j.model.begin(), j.model.end(), name) == j.model.end())
            throw std::runtime_error("joint '" + name + "' not found in model_joint_names");
    }
    return j;
}

void loadControllerParams(ParamSource& src, const ArmJoints& joints, ControllerParams& p) {
    p.ee_site_name  = src.getString("ee_site_name", "arm_ee_site");
    p.owned_joints  = src.getStrings("owned_joints", {});

    p.safe_mode_damping_frac =
        src.getDouble("safe_mode.damping_frac", p.safe_mode_damping_frac);
    p.safe_mode_recovery_ticks =
        src.getInt("safe_mode.recovery_ticks", p.safe_mode_recovery_ticks);
    p.tier3_latch_after_ticks =
        src.getInt("safe_mode.tier3_latch_after_ticks", p.tier3_latch_after_ticks);

    // The disturbance feedforward. Named under observer.* because it is
    // meaningless without an observer to feed it, and 0 by default so enabling
    // the observer still changes nothing about the command.
    p.ff_gain     = src.getDouble("observer.ff_gain", p.ff_gain);
    p.ff_max_frac = src.getDouble("observer.ff_max_frac", p.ff_max_frac);
    p.ff_slew     = src.getDouble("observer.ff_slew", p.ff_slew);

    // Per-joint position/velocity limits: parallel arrays, entry i belongs to
    // joints.owned[i] -- this instance's OWN joints, the ones it box-constrains.
    const auto q_min  = perJoint(src, "limits.q_min", joints.owned, "joint_names");
    const auto q_max  = perJoint(src, "limits.q_max", joints.owned, "joint_names");
    const auto qd_max = perJoint(src, "limits.qd_max", joints.owned, "joint_names");
    p.joint_lims.clear();
    p.joint_lims.reserve(joints.owned.size());
    for (std::size_t i = 0; i < joints.owned.size(); ++i) {
        if (q_min[i] > q_max[i])
            throw std::runtime_error("limits: q_min > q_max for joint '" +
                                     joints.owned[i] + "'");
        p.joint_lims.push_back({joints.owned[i], q_min[i], q_max[i], qd_max[i]});
    }

    // Torque limits are a property of the model's actuators, not of which arm
    // this instance drives, so they are set once for every model joint in the
    // shared parameter block — one array, one place, every instance.
    const auto model_tau_max =
        perJoint(src, "model_tau_max", joints.model, "model_joint_names");
    p.model_torque_lims.clear();
    p.model_torque_lims.reserve(joints.model.size());
    for (std::size_t i = 0; i < joints.model.size(); ++i)
        p.model_torque_lims.emplace_back(joints.model[i], model_tau_max[i]);
}

ObserverConfig loadObserverConfig(ParamSource& src) {
    ObserverConfig o;
    o.type           = src.getString("observer", o.type);
    o.hz             = src.getDouble("observer.hz", o.hz);
    o.gain           = src.getDouble("observer.gain", o.gain);
    o.gain_base      = src.getDouble("observer.gain_base", o.gain_base);
    o.mdot_clamp     = src.getDouble("observer.mdot_clamp", o.mdot_clamp);
    o.estimate_clamp = src.getDouble("observer.estimate_clamp", o.estimate_clamp);
    return o;
}

BaseAnchor loadBaseAnchor(ParamSource& src) {
    BaseAnchor a;
    a.x = src.getDouble("base_anchor_x", a.x);
    a.y = src.getDouble("base_anchor_y", a.y);
    return a;
}

}  // namespace pat_arm_control
