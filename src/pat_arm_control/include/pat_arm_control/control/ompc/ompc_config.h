#pragma once
/*! ompc.yaml's keys -> OMPCParams: the one mapping the node and the benches
 *  both use. Zero ROS. */
#include "pat_arm_control/common/param_source.h"
#include "pat_arm_control/control/controller_config.h"
#include "pat_arm_control/control/ompc/arm_ompc.h"

namespace pat_arm_control {

/*! The shared fields through loadControllerParams (per-joint limits for
 *  `joints.owned`, a torque limit for each of `joints.model`, safe mode,
 *  feedforward), then the OMPC's own, under `ompc.*`:
 *
 *    ompc.Ts, ompc.Q, ompc.R, ompc.damping, ompc.Hg_damping
 *
 *  control_period is left at its default: it is the control loop's period,
 *  owned by whoever runs the loop. Throws std::runtime_error on a weight
 *  that is not 3 long, or a limit array that does not match its joint list. */
OMPCParams loadOmpcParams(ParamSource& src, const ArmJoints& joints);

}  // namespace pat_arm_control
