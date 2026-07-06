#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace pat_gnc {

/// EKF for state [px, py, theta, vx, vy, omega].
/// Measurement: [px, py, theta] from motion capture or vision.
class PlanarEKF {
public:
    using StateVec = Eigen::Matrix<double, 6, 1>;
    using StateMat = Eigen::Matrix<double, 6, 6>;
    using MeasVec  = Eigen::Matrix<double, 3, 1>;
    using MeasMat  = Eigen::Matrix<double, 3, 6>;
    using NoiseMat = Eigen::Matrix<double, 3, 3>;
    using GainMat  = Eigen::Matrix<double, 6, 3>;

    struct Params {
        double   dt{0.02};
        StateMat Q{StateMat::Identity() * 1.0e-4};
        NoiseMat R{NoiseMat::Identity() * 1.0e-3};
    };

    explicit PlanarEKF(const Params& p);
    void    predict();
    MeasVec update(const MeasVec& z);
    void    reset(const StateVec& x0, const StateMat& P0) noexcept;
    void    resetToMeasurement(const MeasVec& z) noexcept;
    StateVec state()      const noexcept { return x_; }
    StateMat covariance() const noexcept { return P_; }

private:
    Params   p_;
    StateVec x_{StateVec::Zero()};
    StateMat P_{StateMat::Identity()};
    StateMat F_;
    MeasMat  H_;
    static double wrapAngle(double a) noexcept;
};

} // namespace pat_gnc
