#include <gtest/gtest.h>
#include "pat_gnc/control/pid_controller.hpp"
#include "pat_gnc/navigation/planar_ekf.hpp"
using Eigen::Vector3d;

TEST(PIDController, ZeroErrorZeroOutput) {
    pat_gnc::PIDController::Gains g; g.kp = g.ki = g.kd = Vector3d::Ones();
    pat_gnc::PIDController pid(g, 0.1);
    EXPECT_NEAR(pid.compute(Vector3d::Zero()).norm(), 0.0, 1e-12);
}
TEST(PIDController, ProportionalResponse) {
    pat_gnc::PIDController::Gains g;
    g.kp = {2.0,3.0,1.0}; g.ki = g.kd = Vector3d::Zero();
    pat_gnc::PIDController pid(g, 0.1);
    auto u = pid.compute(Vector3d::Ones());
    EXPECT_NEAR(u(0),2.0,1e-10); EXPECT_NEAR(u(1),3.0,1e-10);
}
TEST(PIDController, SaturationRespected) {
    pat_gnc::PIDController::Gains g;
    g.kp = Vector3d::Constant(1000); g.ki = g.kd = Vector3d::Zero();
    g.u_max = Vector3d::Constant(5.0);
    pat_gnc::PIDController pid(g, 0.1);
    EXPECT_LE(pid.compute(Vector3d::Ones()).cwiseAbs().maxCoeff(), 5.0+1e-10);
}
TEST(PlanarEKF, PredictIncreasesUncertainty) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    pat_gnc::PlanarEKF ekf(p);
    double tr = ekf.covariance().trace();
    ekf.predict();
    EXPECT_GT(ekf.covariance().trace(), tr);
}
TEST(PlanarEKF, UpdateReducesUncertainty) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    pat_gnc::PlanarEKF ekf(p); ekf.predict();
    double tr = ekf.covariance().trace();
    ekf.update({0.001, 0.001, 0.0001});
    EXPECT_LT(ekf.covariance().trace(), tr);
}
TEST(PlanarEKF, AngleWrapsCorrectly) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    pat_gnc::PlanarEKF ekf(p);
    ekf.resetToMeasurement({0.0, 0.0, M_PI - 0.1});
    ekf.update({0.0, 0.0, -M_PI + 0.1});
    EXPECT_LE(std::abs(ekf.state()(2)), M_PI + 1e-9);
}
