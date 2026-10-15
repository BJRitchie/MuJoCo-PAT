#include "pat_arm_control/observers/momentum_observer.h"

#include <cmath>
#include <stdexcept>

namespace pat_arm_control {

MomentumObserver::MomentumObserver(const MomentumObserverParams& p) : p_(p) {
    if (p_.nv <= 6) {
        throw std::invalid_argument(
            "MomentumObserver: nv must exceed the 6 floating-base DOFs");
    }
    if (p_.n_joints != p_.nv - 6) {
        throw std::invalid_argument(
            "MomentumObserver: n_joints must equal nv - 6");
    }
    if (!(p_.dt > 0.0)) {
        throw std::invalid_argument("MomentumObserver: dt must be positive");
    }
    if (p_.mdot_clamp < 0.0 || p_.estimate_clamp < 0.0) {
        throw std::invalid_argument("MomentumObserver: clamps must not be negative");
    }

    // A single gain is broadcast to every DOF; anything else must be nv long.
    if (p_.gain.size() == 1) {
        p_.gain = Eigen::VectorXd::Constant(p_.nv, p_.gain(0));
    } else if (p_.gain.size() != p_.nv) {
        throw std::invalid_argument(
            "MomentumObserver: gain must have 1 or nv entries");
    }
    for (int i = 0; i < p_.gain.size(); ++i) {
        if (p_.gain(i) < 0.0) {
            throw std::invalid_argument("MomentumObserver: gains must not be negative");
        }
    }

    r_.setZero(p_.nv);
    integral_.setZero(p_.nv);
    M_prev_.setZero(p_.nv, p_.nv);
    Mdot_.setZero(p_.nv, p_.nv);
    p_mom_.setZero(p_.nv);
    CTv_.setZero(p_.nv);
    accum_.setZero(p_.nv);
}

void MomentumObserver::update(const Eigen::MatrixXd& M,
                              const Eigen::VectorXd& bias,
                              const Eigen::VectorXd& v,
                              const Eigen::VectorXd& tau) {
    if (M.rows() != p_.nv || M.cols() != p_.nv || bias.size() != p_.nv ||
        v.size() != p_.nv || tau.size() != p_.nv) {
        throw std::invalid_argument("MomentumObserver::update: size mismatch");
    }

    // Generalised momentum at this state.
    p_mom_.noalias() = M * v;

    if (updates_ == 0) {
        // Seed the integrator with the momentum, so the residual starts at
        // exactly zero. Starting it at zero instead would make r_0 = K p_0,
        // reporting the vehicle's own momentum as a disturbance.
        integral_ = p_mom_;
        r_.setZero();
        M_prev_ = M;
        ++updates_;
        return;
    }

    // Mdot by divided difference. MuJoCo offers no analytic form, and this is
    // free: M is computed every period regardless.
    Mdot_ = (M - M_prev_) / p_.dt;

    // Recorded whether or not a clamp is configured, so tuning can see what the
    // clamp would have had to catch before anyone switches it on.
    const double peak_mdot = Mdot_.cwiseAbs().maxCoeff();
    if (peak_mdot > max_mdot_) max_mdot_ = peak_mdot;
    if (p_.mdot_clamp > 0.0 && peak_mdot > p_.mdot_clamp) {
        ++mdot_clamped_;
        Mdot_ = Mdot_.cwiseMax(-p_.mdot_clamp).cwiseMin(p_.mdot_clamp);
    }

    // C^T qdot = Mdot qdot - C qdot, using the identity Mdot = C + C^T. With
    // gravity disabled MuJoCo's qfrc_bias is exactly C qdot.
    CTv_.noalias() = Mdot_ * v;
    CTv_ -= bias;

    // I_k = I_{k-1} + h (tau_k + C^T qdot + r_{k-1}); r_ is still r_{k-1} here.
    accum_ = tau;
    accum_ += CTv_;
    accum_ += r_;
    integral_ += p_.dt * accum_;

    // r_k = K . (p_k - I_k), which satisfies rdot = K(d - r).
    r_ = p_.gain.cwiseProduct(p_mom_ - integral_);

    if (p_.estimate_clamp > 0.0) {
        if (r_.cwiseAbs().maxCoeff() > p_.estimate_clamp) {
            ++est_clamped_;
            r_ = r_.cwiseMax(-p_.estimate_clamp).cwiseMin(p_.estimate_clamp);
            // Hold the integrator consistent with the clamped residual, or it
            // keeps winding up while the output is pinned and the estimate
            // takes just as long to come back down as it spent saturated.
            integral_ = p_mom_ - r_.cwiseQuotient(
                p_.gain.cwiseMax(1e-12));
        }
    }

    M_prev_ = M;
    ++updates_;
}

void MomentumObserver::reduceToJoints(const Eigen::MatrixXd& H_bb_inv_H_bm,
                                      Eigen::VectorXd& out) const {
    if (H_bb_inv_H_bm.rows() != 6 || H_bb_inv_H_bm.cols() != p_.n_joints) {
        throw std::invalid_argument(
            "MomentumObserver::reduceToJoints: H_bb_inv_H_bm must be 6 x n_joints");
    }
    if (out.size() != p_.n_joints) out.resize(p_.n_joints);
    // d_joints = d_m - (H_bb^-1 H_bm)^T d_b
    out.noalias() = H_bb_inv_H_bm.transpose() * r_.head(6);
    out = r_.tail(p_.n_joints) - out;
}

void MomentumObserver::reset() {
    r_.setZero();
    integral_.setZero();
    M_prev_.setZero();
    updates_ = 0;
    max_mdot_ = 0.0;
    mdot_clamped_ = 0;
    est_clamped_ = 0;
}

} // namespace pat_arm_control
