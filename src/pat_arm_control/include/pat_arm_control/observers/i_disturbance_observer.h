#pragma once
#include <Eigen/Dense>

namespace pat_arm_control {

/*! Abstract interface for estimating the generalised force the model does not
 *  account for: unmodelled link inertia, air-bearing drag, contact reaction.
 *
 *  Add a new estimator by implementing this and selecting it with the
 *  `observer` parameter, the same way pat_gnc picks a navigator.
 *
 *  Everything here is in FULL-system generalised coordinates, nv long, with
 *  the 6 floating-base DOFs first and the arm joints after. The full system is
 *  deliberate rather than convenient: the identity the momentum observer rests
 *  on, Mdot = C + C^T, holds for the full mass matrix and does NOT carry to the
 *  base-eliminated H_g, whose Schur complement is itself configuration
 *  dependent. Estimating on the full system and reducing the ANSWER (a linear
 *  operation, see reduceToJoints) avoids needing the identity in reduced form
 *  at all. It is also the more useful formulation: the base rows of the
 *  estimate are the net external wrench on the base, which is a physically
 *  meaningful quantity on its own -- it is what drag and contact reaction
 *  produce -- and the reduced form throws it away.
 */
class IDisturbanceObserver {
public:
    virtual ~IDisturbanceObserver() = default;

    /*! One observer step.
     *
     *  Must be called once per period, in order, with quantities evaluated at
     *  the SAME state -- the estimator differentiates the mass matrix between
     *  calls, so a skipped or repeated call is a spurious disturbance.
     *
     *  @param M     full mass matrix, nv x nv, at the current configuration
     *  @param bias  MuJoCo's qfrc_bias, nv. With gravity disabled this is
     *               exactly C(q,qdot) qdot, which is what the maths needs; if
     *               gravity were ever re-enabled the gravity term would have to
     *               come out of it first.
     *  @param v     generalised velocity, nv
     *  @param tau   APPLIED generalised force, nv, base rows zero. The applied
     *               torque, not the planned one: the controller clamps per
     *               joint before publishing, and feeding the pre-clamp value
     *               makes the clamped difference look like a disturbance.
     */
    virtual void update(const Eigen::MatrixXd& M,
                        const Eigen::VectorXd& bias,
                        const Eigen::VectorXd& v,
                        const Eigen::VectorXd& tau) = 0;

    /*! The current estimate, full system, nv. Zero before the first update. */
    virtual const Eigen::VectorXd& estimate() const noexcept = 0;

    /*! Reduce a full-system estimate to the joint rows the controller can
     *  actually act on:
     *
     *      d_joints = d_m - H_bm^T H_bb^-1 d_b
     *               = d_m - (H_bb^-1 H_bm)^T d_b
     *
     *  the second form because H_bb is symmetric. This is the same reduction
     *  that already produces Cv_joints from qfrc_bias, so the caller passes
     *  FloatingBaseDynamics::BaseElim::H_bb_inv_H_bm straight in.
     *
     *  @param H_bb_inv_H_bm  6 x n_joints
     *  @param out            n_joints; resized if needed
     */
    virtual void reduceToJoints(const Eigen::MatrixXd& H_bb_inv_H_bm,
                                Eigen::VectorXd& out) const = 0;

    /*! Back to a zero estimate with no history. Called on safe-mode entry and
     *  whenever the input stream has a gap, since a gap breaks the
     *  once-per-period contract that update() relies on. */
    virtual void reset() = 0;

    /*! True once the estimate is meaningful, i.e. at least two updates have
     *  been seen -- the first establishes the integrator's initial condition
     *  and the mass matrix the next one differentiates against. */
    virtual bool ready() const noexcept = 0;
};

} // namespace pat_arm_control
