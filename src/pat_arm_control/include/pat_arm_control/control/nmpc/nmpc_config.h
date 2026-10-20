#pragma once
/*! nmpc.yaml's keys -> NMPCParams: the one mapping the node and the benches
 *  both use. Zero ROS. */
#include "pat_arm_control/common/param_source.h"
#include "pat_arm_control/control/controller_config.h"
#include "pat_arm_control/control/nmpc/arm_nmpc.h"

namespace pat_arm_control {

/*! Every NMPCParams field the config can set: the shared ones through
 *  loadControllerParams (per-joint limits for `joints.owned`, a torque limit
 *  for each of `joints.model`, safe mode, feedforward), then the NMPC's own.
 *
 *  control_period is left at its default: it is the control loop's period,
 *  owned by whoever runs the loop (the node's control_hz, a bench's --hz).
 *
 *  Throws std::runtime_error on a weight vector that is not 3 long, a limit
 *  array whose length does not match its joint list, or q_min > q_max. */
NMPCParams loadNmpcParams(ParamSource& src, const ArmJoints& joints);

}  // namespace pat_arm_control
