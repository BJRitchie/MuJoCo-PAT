#include "pat_arm_control/control/smc/arm_smc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pat_arm_control {

namespace {
bool nonNegative(double x) { return x >= 0.0 && std::isfinite(x); }
}  // namespace

ArmSMC::ArmSMC(const std::string& mjcf_path, const SMCParams& params)
    : OneStepController(mjcf_path, params)
    , params_(params) {
    for (int i = 0; i < 3; ++i) {
        if (!nonNegative(params_.lam[i]) || !nonNegative(params_.alpha[i])) {
            throw std::runtime_error("ArmSMC: lam and alpha must be finite and >= 0");
        }
    }
    if (!(params_.epsilon > 0.0) || !std::isfinite(params_.epsilon)) {
        throw std::runtime_error("ArmSMC: epsilon must be > 0");
    }
    if (!nonNegative(params_.pi_bound) || !nonNegative(params_.k2) ||
        !nonNegative(params_.b2) || !nonNegative(params_.tau_bound) ||
        !nonNegative(params_.lambda_damping)) {
        throw std::runtime_error(
            "ArmSMC: pi_bound, k2, b2, tau_bound and lambda_damping must be finite and >= 0");
    }

    tau_bound_ = params_.tau_bound;
    if (tau_bound_ == 0.0) {
        for (const double lim : torqueLimsByIndex) tau_bound_ = std::max(tau_bound_, lim);
    }
    Hinv_.resize(n_joints_, n_joints_);
}

Eigen::Vector3d ArmSMC::taskForce(const TaskState& s)
{
    // --- Robustness gain (Shi Eq. 27) ----------------------------------------
    hgInverse(Hinv_);
    const double hinv = Hinv_.norm();
    const double rho_s =
        (params_.k2 * params_.b2 + 1.0) * hinv * coriolis().norm() * s.v_ee.norm()
        + params_.k2 * params_.pi_bound * hinv * hgMatrix().norm() * hinv * tau_bound_;
    const Eigen::Vector3d rho =
        Eigen::Vector3d::Constant(rho_s) +
        Eigen::Vector3d(params_.alpha[0], params_.alpha[1], params_.alpha[2]);

    // --- Surface and the commanded task acceleration -------------------------
    const Eigen::Vector3d lam(params_.lam[0], params_.lam[1], params_.lam[2]);
    const Eigen::Vector3d sigma = s.edot + lam.cwiseProduct(s.e);
    const Eigen::Vector3d xdd_star =
        s.a_d + lam.cwiseProduct(s.edot) + rho.cwiseProduct(sat(sigma / params_.epsilon));

    // --- Task-space inertia, L = pinv_DLS(L^-1) ------------------------------
    // For the symmetric positive semi-definite L^-1 the damped pseudo-inverse
    // is (L^-2 + d^2 I)^-1 L^-1: the same thing as the SVD form, as a 3x3
    // solve with no allocation.
    const double d2 = params_.lambda_damping * params_.lambda_damping;
    Eigen::Matrix3d M = s.Lam_inv * s.Lam_inv;
    M.diagonal().array() += d2;
    const Eigen::Matrix3d Lam_op = M.ldlt().solve(s.Lam_inv);
    checkNaN(Lam_op.hasNaN(), "Lambda");

    return Lam_op * (xdd_star - s.hp);
}

}  // namespace pat_arm_control
