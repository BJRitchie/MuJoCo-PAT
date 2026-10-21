#include "pat_arm_control/control/ompc/ompc_config.h"

namespace pat_arm_control {

OMPCParams loadOmpcParams(ParamSource& src, const ArmJoints& joints) {
    OMPCParams p;
    loadControllerParams(src, joints, p);

    p.Ts         = src.getDouble("ompc.Ts", p.Ts);
    p.damping    = src.getDouble("ompc.damping", p.damping);
    p.Hg_damping = src.getDouble("ompc.Hg_damping", p.Hg_damping);

    // [x, y, θz] each: the planar law's three task channels.
    p.Q = loadTriple(src, "ompc.Q", p.Q);
    p.R = loadTriple(src, "ompc.R", p.R);
    return p;
}

}  // namespace pat_arm_control
