#pragma once
#include <string>

#include <Eigen/Dense>

#include "pat_arm_control/observers/i_disturbance_observer.h"

namespace pat_arm_control {

/*! Tuning for MomentumObserver. */
struct MomentumObserverParams {
    int    nv = 0;        //!< full-system DOF count (6 + arm joints)
    int    n_joints = 0;  //!< arm joints, i.e. nv - 6
    double dt = 0.01;     //!< observer period [s]; must match the call rate

    /*! Per-DOF observer gain K, nv long. Each entry sets a first-order
     *  response: the estimate approaches the true disturbance with time
     *  constant 1/K on that DOF, so K = 20 means a ~50 ms lag. Raising it
     *  tracks contact faster and passes more measurement noise through.
     *  A single value is broadcast to every DOF. */
    Eigen::VectorXd gain;

    /*! Clamp on |Mdot| entries, per second. 0 disables.
     *
     *  Mdot is a divided difference of the mass matrix, so a velocity glitch
     *  or a dropped period produces one enormous Mdot and a corresponding
     *  spike in the estimate. Clamping bounds that without filtering the
     *  signal itself. */
    double mdot_clamp = 0.0;

    /*! Clamp on the magnitude of each estimate entry. 0 disables. The command
     *  path saturates separately; this bounds the state, so a transient cannot
     *  make the integrator diverge. */
    double estimate_clamp = 0.0;
};

/*! De Luca & Mattone generalised-momentum disturbance observer.
 *
 *  Estimates the unmodelled generalised force d in
 *
 *      M(q) qddot + C(q,qdot) qdot = tau + d
 *
 *  without ever differentiating a measured velocity. That is the whole reason
 *  to prefer it here: encoder-derived velocity is the noisiest signal
 *  available, and a linear extended state observer's disturbance channel
 *  differentiates twice, with noise scaling as the cube of its bandwidth.
 *
 *  With generalised momentum p = M qdot,
 *
 *      pdot = M qddot + Mdot qdot = (tau + d - C qdot) + Mdot qdot
 *
 *  and the identity Mdot = C + C^T collapses that to
 *
 *      pdot = tau + d + C^T qdot      =>      d = pdot - tau - C^T qdot
 *
 *  which still needs qddot through pdot. Integrating instead removes it --
 *  the residual
 *
 *      r = K [ p - integral(tau + C^T qdot + r) ]
 *
 *  satisfies rdot = K(d - r), a first-order low-pass of the true disturbance,
 *  with no derivative of a measurement anywhere in it.
 *
 *  Per step of length h, with C^T qdot recovered as Mdot qdot - C qdot and
 *  C qdot read straight off MuJoCo's qfrc_bias:
 *
 *      Mdot    ~ (M_k - M_{k-1}) / h
 *      CTv     = Mdot v - bias
 *      I_k     = I_{k-1} + h (tau_k + CTv + r_{k-1})
 *      r_k     = K . (M_k v_k - I_k)
 *
 *  The integrator is initialised to p on the first update, so the estimate
 *  starts at exactly zero rather than at whatever the momentum happens to be.
 *
 *  MuJoCo exposes no analytic Mdot -- its derivative API (mjd_transitionFD,
 *  mjd_inverseFD) is finite difference throughout -- so the divided difference
 *  above is ours. It is free, because M is computed every period anyway.
 *
 *  update() allocates nothing: every buffer is sized in the constructor. The
 *  observer shares the control path's realtime constraints.
 */
class MomentumObserver final : public IDisturbanceObserver {
public:
    /*! Throws std::invalid_argument on an inconsistent size, a non-positive
     *  dt, or a negative gain. */
    explicit MomentumObserver(const MomentumObserverParams& p);

    void update(const Eigen::MatrixXd& M,
                const Eigen::VectorXd& bias,
                const Eigen::VectorXd& v,
                const Eigen::VectorXd& tau) override;

    const Eigen::VectorXd& estimate() const noexcept override { return r_; }

    void reduceToJoints(const Eigen::MatrixXd& H_bb_inv_H_bm,
                        Eigen::VectorXd& out) const override;

    void reset() override;

    bool ready() const noexcept override { return updates_ >= 2; }

    /*! Updates seen since the last reset. */
    long updates() const noexcept { return updates_; }

    /*! Largest |Mdot| entry seen since the last reset, and how many steps hit
     *  mdot_clamp. Reported so "the clamp is doing something" is a measured
     *  number rather than a guess at tuning time. */
    double maxMdot() const noexcept { return max_mdot_; }
    long   mdotClamped() const noexcept { return mdot_clamped_; }
    long   estimateClamped() const noexcept { return est_clamped_; }

    const MomentumObserverParams& params() const noexcept { return p_; }

private:
    MomentumObserverParams p_;

    // State carried between steps.
    Eigen::VectorXd r_;        //!< the residual, nv -- the estimate itself
    Eigen::VectorXd integral_; //!< I, nv
    Eigen::MatrixXd M_prev_;   //!< nv x nv, for the divided difference
    long updates_ = 0;

    // Preallocated scratch, so update() never reaches the allocator.
    Eigen::MatrixXd Mdot_;
    Eigen::VectorXd p_mom_, CTv_, accum_;

    // Diagnostics.
    double max_mdot_ = 0.0;
    long   mdot_clamped_ = 0;
    long   est_clamped_ = 0;
};

} // namespace pat_arm_control
