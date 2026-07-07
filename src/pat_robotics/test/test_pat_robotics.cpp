#include <gtest/gtest.h>
#include <cmath>
#include "pat_robotics/control/joint_pid.hpp"
#include "pat_robotics/control/arm_controller.hpp"

TEST(JointPID, ZeroErrorZeroOutput) {
    pat_robotics::JointPID::Gains g{1.0, 1.0, 1.0, 100.0};
    pat_robotics::JointPID pid(g, 0.1);
    EXPECT_NEAR(pid.compute(0.0), 0.0, 1e-12);
}
TEST(JointPID, ProportionalResponse) {
    pat_robotics::JointPID::Gains g{2.0, 0.0, 0.0, 100.0};
    pat_robotics::JointPID pid(g, 0.1);
    EXPECT_NEAR(pid.compute(1.0), 2.0, 1e-10);
}
TEST(JointPID, SaturationRespected) {
    pat_robotics::JointPID::Gains g{1000.0, 0.0, 0.0, 5.0};
    pat_robotics::JointPID pid(g, 0.1);
    EXPECT_LE(std::abs(pid.compute(1.0)), 5.0 + 1e-10);
}
TEST(ArmController, ConvergesTowardSetpointDirection) {
    std::vector<pat_robotics::JointPID::Gains> g{
        {5.0, 0.0, 0.0, 10.0}, {5.0, 0.0, 0.0, 10.0}, {5.0, 0.0, 0.0, 10.0}};
    pat_robotics::ArmController ctl(g, 0.01);
    std::vector<double> q{0.0, 0.0, 0.0}, q_des{0.5, -0.3, 0.2};
    auto u = ctl.compute(q, q_des);
    ASSERT_EQ(u.size(), 3u);
    EXPECT_GT(u[0], 0.0);
    EXPECT_LT(u[1], 0.0);
    EXPECT_GT(u[2], 0.0);
}
TEST(ArmController, WrapsAngleErrorNearPi) {
    std::vector<pat_robotics::JointPID::Gains> g{
        {1.0, 0.0, 0.0, 100.0}, {1.0, 0.0, 0.0, 100.0}, {1.0, 0.0, 0.0, 100.0}};
    pat_robotics::ArmController ctl(g, 0.01);
    std::vector<double> q{M_PI - 0.1, 0.0, 0.0}, q_des{-M_PI + 0.1, 0.0, 0.0};
    auto u = ctl.compute(q, q_des);
    // Short way around (~0.2 rad error), not the long way (~2*pi worth).
    EXPECT_LT(std::abs(u[0]), 1.0);
}
TEST(ArmController, IsSizedFromGainsNotHardcoded) {
    std::vector<pat_robotics::JointPID::Gains> g{
        {1.0, 0.0, 0.0, 100.0}, {1.0, 0.0, 0.0, 100.0}};
    pat_robotics::ArmController ctl(g, 0.01);
    EXPECT_EQ(ctl.numJoints(), 2u);
}
int main(int argc, char** argv) { testing::InitGoogleTest(&argc, argv); return RUN_ALL_TESTS(); }
