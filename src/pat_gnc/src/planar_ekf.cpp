#include "pat_gnc/navigation/planar_ekf.hpp"

namespace pat_gnc {

PlanarEKF::PlanarEKF(const Params& p) : p_(p) {
    F_ = StateMat::Identity();
    F_.topRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * p_.dt;
    H_ = MeasMat::Zero();
    H_.leftCols<3>() = Eigen::Matrix3d::Identity();
}

void PlanarEKF::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + p_.Q;
}

PlanarEKF::MeasVec PlanarEKF::update(const MeasVec& z) {
    MeasVec innov = z - H_ * x_;
    innov(2) = wrapAngle(innov(2));
    const NoiseMat S = H_ * P_ * H_.transpose() + p_.R;
    const GainMat  K = P_ * H_.transpose() * S.inverse();
    x_ += K * innov;
    x_(2) = wrapAngle(x_(2));
    P_ = (StateMat::Identity() - K * H_) * P_;
    return innov;
}

void PlanarEKF::reset(const StateVec& x0, const StateMat& P0) noexcept { x_ = x0; P_ = P0; }
void PlanarEKF::resetToMeasurement(const MeasVec& z) noexcept {
    x_.setZero(); x_.head<3>() = z; P_ = StateMat::Identity();
}
double PlanarEKF::wrapAngle(double a) noexcept { return std::remainder(a, 2.0 * M_PI); }

} // namespace pat_gnc
