#include "pat_arm_control/control/smc/smc_config.h"

namespace pat_arm_control {

SMCParams loadSmcParams(ParamSource& src, const ArmJoints& joints) {
    SMCParams p;
    loadControllerParams(src, joints, p);

    p.epsilon        = src.getDouble("smc.epsilon", p.epsilon);
    p.pi_bound       = src.getDouble("smc.pi_bound", p.pi_bound);
    p.k2             = src.getDouble("smc.k2", p.k2);
    p.b2             = src.getDouble("smc.b2", p.b2);
    p.tau_bound      = src.getDouble("smc.tau_bound", p.tau_bound);
    p.lambda_damping = src.getDouble("smc.lambda_damping", p.lambda_damping);
    p.Hg_damping     = src.getDouble("smc.Hg_damping", p.Hg_damping);

    // [x, y, θz] each: the planar law's three task channels.
    p.lam   = loadTriple(src, "smc.lam",   p.lam);
    p.alpha = loadTriple(src, "smc.alpha", p.alpha);
    return p;
}

}  // namespace pat_arm_control
