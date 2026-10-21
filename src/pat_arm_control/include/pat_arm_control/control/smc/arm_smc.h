#pragma once
/*! Sliding-mode task-space controller (SMC), after Shi et al. 2017, Eq. 26/27.
 *  Port of VORTEX armSlidingModeController, planar: the task is
 *  [x, y, thetaz] instead of VORTEX's (x, y, z) position, and the signs are
 *  re-derived rather than copied (see below).
 *
 *  With e = p_d - p (the convention everywhere in this package), the sliding
 *  surface is
 *
 *      sigma = edot + Lam*e                         (Lam = diag(lam))
 *
 *  and its derivative, using xdd = L^-1 tau_task + Jdot*qdot with
 *  L^-1 = J H_g^-1 J^T, is
 *
 *      sigma_dot = a_d - xdd + Lam*edot .
 *
 *  Commanding the task acceleration
 *
 *      xdd* = a_d + Lam*edot + rho * sat(sigma/eps)
 *
 *  gives sigma_dot = -rho * sat(sigma/eps), so sigma*sigma_dot <= 0: the state
 *  is driven to the surface and slides to the target along it. The task force
 *  that produces it, with L = (J H_g^-1 J^T)^-1 the task-space inertia, is
 *
 *      tau_task = L * (xdd* - Jdot*qdot) ,
 *
 *  and the joints are commanded tau = J^T tau_task + Cv (Cv cancels the
 *  Coriolis term, so H_g qdd = J^T tau_task).
 *
 *  The robustness gain follows Shi Eq. 27, a scalar plus a per-axis margin:
 *
 *      rho_s = (k2*b2 + 1) |H_g^-1| |Cv| |v_ee|
 *            + k2*pi_bound * |H_g^-1| |H_g| |H_g^-1| * tau_bound
 *      rho   = rho_s + alpha          (per axis)
 *
 *  with Frobenius norms.
 *
 *  Differences from VORTEX, which the closed-loop tests are the check on:
 *    - VORTEX returns -tau_joints, including the +Cv term, and writes
 *      tau_star = -Lam*z2 + a_d - h' - rho*sat(sigma/eps). With z = p_d - p
 *      the signs on Lam*z2 and rho*sat cannot be made to give sigma_dot =
 *      -rho*sat(.) by any overall sign, so the law above is derived instead.
 *    - rho_s uses |v_ee|; VORTEX's (z2 + v_d) is 2*v_d - v_ee, which is not
 *      the end-effector velocity it stands in for.
 *    - H_g^-1 is exact (a Cholesky inverse), not damped by Hg_damping, which
 *      distorted it; L uses the damped pseudo-inverse with lambda_damping,
 *      as VORTEX does.
 *    - tau_bound, the force bound in rho_s, defaults to the largest model
 *      torque limit rather than VORTEX's 100, which has no meaning at this
 *      arm's 0.25 N*m.
 *    - Per-axis lam and alpha triples.
 *
 *  Stateless between ticks, and, as a one-step law, has no constraint
 *  handling: the torque limit is a clamp.
 */
#include <array>
#include <string>

#include <Eigen/Dense>

#include "pat_arm_control/control/controller_params.h"
#include "pat_arm_control/control/one_step_controller.h"

namespace pat_arm_control {

struct SMCParams : ControllerParams {
    //! Surface slope per axis [x, y, thetaz] [1/s]: how fast the error decays
    //! once on the surface.
    std::array<double, 3> lam   = {3.0, 3.0, 3.0};
    //! Added to rho per axis [m/s^2, rad/s^2]: a floor on the switching gain.
    std::array<double, 3> alpha = {0.5, 0.5, 0.5};
    //! Boundary-layer width, in units of sigma. Inside it the switching term is
    //! linear instead of a sign, trading chatter for a steady-state band.
    double epsilon = 0.05;
    double pi_bound = 0.3;   //!< Shi's bound on the model-error ratio
    double k2 = 1.3;
    double b2 = 1.3;
    //! Force bound in rho_s [N*m]; 0 means the largest model torque limit.
    double tau_bound = 0.0;
    //! DLS damping for the task-space inertia L = pinv(J H_g^-1 J^T).
    double lambda_damping = 0.02;

    SMCParams() = default;
};

class ArmSMC : public OneStepController {
public:
    /*! Throws std::runtime_error on a negative or non-finite gain, or
     *  epsilon <= 0. */
    ArmSMC(const std::string& mjcf_path, const SMCParams& params);

protected:
    Eigen::Vector3d taskForce(const TaskState& s) override;

    SMCParams params_;

private:
    double tau_bound_ = 0.0;     //!< params_.tau_bound with 0 resolved
    Eigen::MatrixXd Hinv_;       //!< H_g^-1, sized at construction
};

}  // namespace pat_arm_control
