#include <gtest/gtest.h>
#include "pat_gnc/navigation/direct_navigator.hpp"
#include "pat_gnc/navigation/ekf_navigator.hpp"

TEST(DirectNavigator, StoresPoseAndTwist) {
    pat_gnc::DirectNavigator nav;
    nav.onOdometry({1.0, 2.0, 0.5}, {0.1, 0.2, 0.05});
    const auto s = nav.state();
    EXPECT_NEAR(s(0), 1.0, 1e-10); EXPECT_NEAR(s(3), 0.1, 1e-10);
}
TEST(DirectNavigator, ImuIsNoop) {
    pat_gnc::DirectNavigator nav;
    nav.onOdometry({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    nav.onImu(99.9);
    EXPECT_NEAR(nav.state()(5), 0.0, 1e-10);
}
TEST(EkfNavigator, MovesStateTowardMeasurement) {
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    pat_gnc::EkfNavigator nav(p);
    nav.onOdometry({1.0, 2.0, 0.5}, {});
    EXPECT_NEAR(nav.state()(0), 1.0, 0.1);
}
TEST(NavigatorInterface, BothImplementINavigator) {
    std::unique_ptr<pat_gnc::INavigator> nav1 =
        std::make_unique<pat_gnc::DirectNavigator>();
    pat_gnc::PlanarEKF::Params p; p.dt = 0.02;
    std::unique_ptr<pat_gnc::INavigator> nav2 =
        std::make_unique<pat_gnc::EkfNavigator>(p);
    nav1->onOdometry({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    nav2->onOdometry({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    EXPECT_NEAR(nav1->state()(0), 1.0, 0.1);
    EXPECT_NEAR(nav2->state()(0), 1.0, 0.1);
}
