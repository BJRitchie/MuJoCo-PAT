#pragma once
/*! Configuration every arm controller shares, whatever its control law: which
 *  joints it drives, the disturbance observer beside it, and the frame
 *  correction on the base state. Read from a ParamSource, so the node and the
 *  benches load it through the same code. Zero ROS.
 */
#include <array>
#include <string>
#include <vector>

#include "pat_arm_control/common/param_source.h"
#include "pat_arm_control/control/controller_params.h"

namespace pat_arm_control {

/*! The joints a controller instance works with, by MuJoCo name. */
struct ArmJoints {
    //! This instance's own joints (`joint_names`): the ones it commands, and
    //! the order its per-joint limit arrays follow.
    std::vector<std::string> owned;
    //! EVERY model joint in MuJoCo body-tree order (`model_joint_names`). The
    //! controller rolls the whole model forward, so per-model arrays such as
    //! the torque limits follow this order.
    std::vector<std::string> model;
};

/*! `joint_names` and `model_joint_names`. Throws std::runtime_error if either
 *  is empty or an owned joint is not a model joint. */
ArmJoints loadArmJoints(ParamSource& src);

/*! The ControllerParams fields every law reads under the same keys:
 *  `ee_site_name`, `owned_joints`, `safe_mode.{damping_frac, recovery_ticks,
 *  tier3_latch_after_ticks}`, `observer.ff_*`, the per-joint `limits.*` for
 *  `joints.owned` and `model_tau_max` for `joints.model`. A law's own loader
 *  calls this first and then reads the rest, including the ControllerParams
 *  fields whose key belongs to the law's block (NMPC's `nmpc.Hg_damping`,
 *  `nmpc.reference_preview`). control_period is left alone.
 *
 *  Throws std::runtime_error on a limit array whose length does not match its
 *  joint list, or q_min > q_max. */
void loadControllerParams(ParamSource& src, const ArmJoints& joints, ControllerParams& p);

/*! A [x, y, thetaz] triple (a weight or a gain per planar task channel);
 *  exactly three entries, or std::runtime_error naming the key. */
std::array<double, 3> loadTriple(ParamSource& src, const std::string& key,
                                 std::array<double, 3> def);

/*! The disturbance observer that runs beside the controller (`observer.*`).
 *  Its feedforward gains are controller parameters, not part of this. */
struct ObserverConfig {
    std::string type       = "none";   //!< "none" or "momentum"
    double hz              = 100.0;    //!< [Hz] its own thread's rate
    double gain            = 20.0;     //!< [1/s] joint rows
    double gain_base       = -1.0;     //!< [1/s] base rows; < 0 means "same as gain"
    double mdot_clamp      = 0.0;      //!< 0 disables
    double estimate_clamp  = 0.0;      //!< 0 disables
};

ObserverConfig loadObserverConfig(ParamSource& src);

/*! The world-frame position of the chaser body's anchor in this MJCF
 *  (`base_anchor_x`/`_y`): odom reports x/y relative to it, the model
 *  expects world coordinates. 0, 0 for a model with no such offset. */
struct BaseAnchor {
    double x = 0.0;
    double y = 0.0;
};

BaseAnchor loadBaseAnchor(ParamSource& src);

}  // namespace pat_arm_control
