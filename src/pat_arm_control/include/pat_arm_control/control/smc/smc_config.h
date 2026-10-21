#pragma once
/*! smc.yaml's keys -> SMCParams: the one mapping the node and the benches
 *  both use. Zero ROS. */
#include "pat_arm_control/common/param_source.h"
#include "pat_arm_control/control/controller_config.h"
#include "pat_arm_control/control/smc/arm_smc.h"

namespace pat_arm_control {

/*! The shared fields through loadControllerParams (per-joint limits for
 *  `joints.owned`, a torque limit for each of `joints.model`, safe mode,
 *  feedforward), then the SMC's own, under `smc.*`:
 *
 *    smc.lam, smc.alpha, smc.epsilon, smc.pi_bound, smc.k2, smc.b2,
 *    smc.tau_bound, smc.lambda_damping, smc.Hg_damping
 *
 *  control_period is left at its default: it is the control loop's period,
 *  owned by whoever runs the loop. Throws std::runtime_error on a triple that
 *  is not 3 long, or a limit array that does not match its joint list. */
SMCParams loadSmcParams(ParamSource& src, const ArmJoints& joints);

}  // namespace pat_arm_control
