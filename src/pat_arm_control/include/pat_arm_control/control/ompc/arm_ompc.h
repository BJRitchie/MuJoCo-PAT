#pragma once
/*! One-step-ahead model-predictive task-space controller (OMPC), after Shi et
 *  al. 2017, Eq. 38. Port of VORTEX armOMPController, planar: the task is
 *  [x, y, thetaz] (rows 0, 1 and the angular-Z row of the generalised
 *  Jacobian) instead of VORTEX's (x, y, z) position.
 *
 *  Not a horizon controller: it predicts the task-space error one step Ts
 *  ahead from the current linearisation and minimises, in closed form,
 *
 *      J = 1/2 ||e(t+Ts)||^2_Q + 1/2 ||tau_task||^2_R .
 *
 *  With e = p_d - p (the convention everywhere in this package) and the
 *  reduced dynamics H_g qdd + Cv = tau, commanding tau = J^T tau_task + Cv
 *  gives  xdd = L^-1 tau_task + Jdot*qdot  with  L^-1 = J H_g^-1 J^T, so
 *
 *      e(t+Ts) = C - (Ts^2/2) L^-1 tau_task
 *      C       = e + Ts*edot - (Ts^2/2) Jdot*qdot + (Ts^2/2) a_d
 *
 *  and the minimiser is
 *
 *      tau_task = A^-1 B C
 *      A = R + (Ts^2/2)^2 L^-T Q L^-1,   B = (Ts^2/2) L^-T Q .
 *
 *  Differences from VORTEX:
 *    - Q and R are [x, y, thetaz] triples, not scalar per axis.
 *    - H_g^-1 J^T comes from a Cholesky solve (falling back to the damped
 *      pseudo-inverse if H_g is not positive definite), not a plain inverse.
 *    - The defaults are the ones VORTEX's scenario ran (Q = 1e5, R = 1,
 *      Ts = 1); VORTEX's own defaults of Q = R = 0 make A singular.
 *    - The disturbance feedforward, the per-joint torque clamp and safe mode
 *      come from TaskSpaceController.
 *
 *  Stateless between ticks. There is no constraint handling: the torque limit
 *  is a clamp, so a command that wants more than the limit is saturated, not
 *  re-solved.
 */
#include <array>
#include <string>

#include <Eigen/Dense>

#include "pat_arm_control/control/controller_params.h"
#include "pat_arm_control/control/one_step_controller.h"

namespace pat_arm_control {

struct OMPCParams : ControllerParams {
    //! [s] prediction step. Not the control period: it sets how far ahead the
    //! error is predicted, and with it how hard the law pushes.
    double Ts = 1.0;
    //! Weight on the predicted error, [x, y, thetaz].
    std::array<double, 3> Q = {1e5, 1e5, 1e5};
    //! Weight on the task-space force, [x, y, thetaz].
    std::array<double, 3> R = {1.0, 1.0, 1.0};
    //! Added to the diagonal of L^-1 = J H_g^-1 J^T, keeping it invertible
    //! near a kinematic singularity.
    double damping = 1e-4;

    OMPCParams() = default;
};

class ArmOMPC : public OneStepController {
public:
    /*! Throws std::runtime_error if Ts <= 0, a weight is negative or
     *  non-finite, an axis has both Q and R zero (A would be singular), or
     *  damping is negative. */
    ArmOMPC(const std::string& mjcf_path, const OMPCParams& params);

protected:
    Eigen::Vector3d taskForce(const TaskState& s) override;

    OMPCParams params_;
};

}  // namespace pat_arm_control
