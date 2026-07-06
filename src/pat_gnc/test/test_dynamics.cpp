#include <gtest/gtest.h>
#include <cmath>
#include "pat_gnc/planar_dynamics.hpp"
using pat_gnc::PlanarDynamics;

TEST(PlanarDynamics, ZeroForceConstantVelocity) {
    PlanarDynamics d;
    PlanarDynamics::State x; x << 0,0,0,0.5,0.2,0.1;
    auto xn = d.step(x, PlanarDynamics::Input::Zero(), 0.1);
    EXPECT_NEAR(xn(3), 0.5, 1e-9); EXPECT_NEAR(xn(4), 0.2, 1e-9);
}
TEST(PlanarDynamics, ForwardForceAcceleratesX) {
    PlanarDynamics d;
    auto xn = d.step(PlanarDynamics::State::Zero(), {5.0,0.0,0.0}, 0.1);
    EXPECT_GT(xn(3), 0.0); EXPECT_NEAR(xn(4), 0.0, 1e-9);
}
TEST(PlanarDynamics, TorqueRotates) {
    PlanarDynamics d;
    auto xn = d.step(PlanarDynamics::State::Zero(), {0.0,0.0,1.0}, 0.1);
    EXPECT_GT(xn(5), 0.0);
}
TEST(PlanarDynamics, RelativeStateSelfIsZero) {
    PlanarDynamics::State s; s << 1,2,0.5,0.1,0.2,0.05;
    EXPECT_NEAR(PlanarDynamics::relativeState(s,s).norm(), 0.0, 1e-12);
}
int main(int argc, char** argv) { testing::InitGoogleTest(&argc, argv); return RUN_ALL_TESTS(); }
