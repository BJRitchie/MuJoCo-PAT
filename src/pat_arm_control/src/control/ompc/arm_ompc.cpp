#include "pat_arm_control/control/ompc/arm_ompc.h"

#include <cmath>
#include <stdexcept>

namespace pat_arm_control {

ArmOMPC::ArmOMPC(const std::string& mjcf_path, const OMPCParams& params)
    : OneStepController(mjcf_path, params)
    , params_(params) {
    if (!(params_.Ts > 0.0) || !std::isfinite(params_.Ts)) {
        throw std::runtime_error("ArmOMPC: OMPCParams.Ts must be > 0");
    }
    if (!(params_.damping >= 0.0) || !std::isfinite(params_.damping)) {
        throw std::runtime_error("ArmOMPC: OMPCParams.damping must be >= 0");
    }
    for (int i = 0; i < 3; ++i) {
        const double q = params_.Q[i], r = params_.R[i];
        if (!(q >= 0.0) || !(r >= 0.0) || !std::isfinite(q) || !std::isfinite(r)) {
            throw std::runtime_error("ArmOMPC: Q and R must be finite and >= 0");
        }
        // With both zero on an axis, A has a zero row and the gain solve is
        // singular: the command on that axis would be garbage, not zero.
        if (q == 0.0 && r == 0.0) {
            throw std::runtime_error(
                "ArmOMPC: Q and R are both 0 on an axis, so the gain matrix is singular");
        }
    }
}

Eigen::Vector3d ArmOMPC::taskForce(const TaskState& s)
{
    Eigen::Matrix3d Lam_inv = s.Lam_inv;
    Lam_inv.diagonal().array() += params_.damping;

    const double Ts2_2 = 0.5 * params_.Ts * params_.Ts;
    const Eigen::Vector3d Qd(params_.Q[0], params_.Q[1], params_.Q[2]);
    const Eigen::Vector3d Rd(params_.R[0], params_.R[1], params_.R[2]);
    const Eigen::Vector3d C = s.e + params_.Ts * s.edot - Ts2_2 * s.hp + Ts2_2 * s.a_d;

    // B = Ts2_2 * L^-T Q ;  A = R + Ts2_2 * B * L^-1  (the same as
    // R + Ts2_2^2 L^-T Q L^-1, with one product shared).
    const Eigen::Matrix3d B = Ts2_2 * Lam_inv.transpose() * Qd.asDiagonal();
    const Eigen::Matrix3d A = Eigen::Matrix3d(Rd.asDiagonal()) + Ts2_2 * B * Lam_inv;
    checkNaN(A.hasNaN(), "A_term");

    return A.ldlt().solve(B * C);
}

}  // namespace pat_arm_control
