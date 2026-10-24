#include "pat_arm_control/estimation/joint_kalman_filter.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace pat_arm_control {

JointKalmanFilter::JointKalmanFilter(const JointKalmanParams& p, int n_joints)
    : p_(p) {
    const auto bad = [](double v) { return !std::isfinite(v) || v < 0.0; };
    if (n_joints <= 0) throw std::invalid_argument("JointKalmanFilter: no joints");
    if (bad(p.accel_psd) || bad(p.init_vel_std) || !std::isfinite(p.vel_meas_std) ||
        !std::isfinite(p.pos_meas_std) || !(p.pos_meas_std > 0.0)) {
        throw std::invalid_argument(
            "JointKalmanFilter: accel_psd and init_vel_std must be finite and >= 0, "
            "pos_meas_std finite and > 0");
    }
    x_.assign(static_cast<std::size_t>(n_joints), Eigen::Vector2d::Zero());
    P_.assign(static_cast<std::size_t>(n_joints), Eigen::Matrix2d::Zero());
}

void JointKalmanFilter::update(double h, const Eigen::Ref<const Eigen::VectorXd>& q,
                               const Eigen::Ref<const Eigen::VectorXd>& qd) {
    const std::size_t n = x_.size();
    const bool use_vel = p_.vel_meas_std > 0.0;
    if (static_cast<std::size_t>(q.size()) != n ||
        (use_vel && static_cast<std::size_t>(qd.size()) != n)) {
        throw std::invalid_argument("JointKalmanFilter: expected " + std::to_string(n) +
                                    " joints");
    }
    const double rp = p_.pos_meas_std * p_.pos_meas_std;
    const double rv = p_.vel_meas_std * p_.vel_meas_std;

    if (!initialised_) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto ii = static_cast<Eigen::Index>(i);
            x_[i] << q(ii), use_vel ? qd(ii) : 0.0;
            const double v0 = use_vel ? rv : p_.init_vel_std * p_.init_vel_std;
            P_[i] << rp, 0.0, 0.0, v0;
        }
        initialised_ = true;
        return;
    }
    if (!(h > 0.0)) throw std::invalid_argument("JointKalmanFilter: h must be > 0");

    Eigen::Matrix2d F;
    F << 1.0, h, 0.0, 1.0;
    Eigen::Matrix2d Q;
    Q << h * h * h / 3.0, h * h / 2.0, h * h / 2.0, h;
    Q *= p_.accel_psd;

    for (std::size_t i = 0; i < n; ++i) {
        const auto ii = static_cast<Eigen::Index>(i);
        Eigen::Vector2d& x = x_[i];
        Eigen::Matrix2d& P = P_[i];
        x = F * x;
        P = F * P * F.transpose() + Q;

        if (use_vel) {
            // H = I: the innovation is the reading minus the prediction.
            Eigen::Matrix2d S = P;
            S(0, 0) += rp;
            S(1, 1) += rv;
            const Eigen::Matrix2d K = P * S.inverse();
            const Eigen::Vector2d y(q(ii) - x(0), qd(ii) - x(1));
            x += K * y;
            // Joseph form: stays symmetric and positive definite in rounding.
            const Eigen::Matrix2d IKH = Eigen::Matrix2d::Identity() - K;
            Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
            R(0, 0) = rp;
            R(1, 1) = rv;
            P = IKH * P * IKH.transpose() + K * R * K.transpose();
        } else {
            const double s = P(0, 0) + rp;
            const Eigen::Vector2d K = P.col(0) / s;
            x += K * (q(ii) - x(0));
            Eigen::Matrix2d IKH = Eigen::Matrix2d::Identity();
            IKH.col(0) -= K;
            P = IKH * P * IKH.transpose() + rp * K * K.transpose();
        }
    }
}

}  // namespace pat_arm_control
