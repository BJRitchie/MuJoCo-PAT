#include <gtest/gtest.h>
#include "pat_telemetry/telemetry_aggregator.hpp"
using pat_telemetry::TelemetryAggregator;

TEST(TelemetryAggregator, MergeByNameSameOrder) {
    std::vector<std::string> names_out{"a", "b", "c"};
    std::vector<std::string> names{"a", "b", "c"};
    std::vector<double> values{1.0, 2.0, 3.0};
    auto out = TelemetryAggregator::mergeByName(names_out, names, values);
    EXPECT_EQ(out, (std::vector<double>{1.0, 2.0, 3.0}));
}

TEST(TelemetryAggregator, MergeByNameReorderedInput) {
    std::vector<std::string> names_out{"a", "b", "c"};
    std::vector<std::string> names{"c", "a", "b"};
    std::vector<double> values{30.0, 10.0, 20.0};
    auto out = TelemetryAggregator::mergeByName(names_out, names, values);
    EXPECT_EQ(out, (std::vector<double>{10.0, 20.0, 30.0}));
}

TEST(TelemetryAggregator, MergeByNameMissingNameKeepsFallback) {
    std::vector<std::string> names_out{"a", "b", "c"};
    std::vector<std::string> names{"a", "c"};
    std::vector<double> values{1.0, 3.0};
    auto out = TelemetryAggregator::mergeByName(names_out, names, values, -1.0);
    EXPECT_EQ(out, (std::vector<double>{1.0, -1.0, 3.0}));
}

TEST(TelemetryAggregator, MergeByNameEmptyInputAllFallback) {
    std::vector<std::string> names_out{"a", "b"};
    auto out = TelemetryAggregator::mergeByName(names_out, {}, {}, 0.0);
    EXPECT_EQ(out, (std::vector<double>{0.0, 0.0}));
}

TEST(TelemetryAggregator, PlanarDistanceZeroWhenSame) {
    EXPECT_NEAR(TelemetryAggregator::planarDistance(1.0, 2.0, 1.0, 2.0), 0.0, 1e-12);
}

TEST(TelemetryAggregator, PlanarDistanceKnownValue) {
    // 3-4-5 triangle
    EXPECT_NEAR(TelemetryAggregator::planarDistance(0.0, 0.0, 3.0, 4.0), 5.0, 1e-9);
}

TEST(TelemetryAggregator, UpdateByNameAppendsNewNames) {
    std::vector<std::string> names;
    std::vector<double> values;
    TelemetryAggregator::updateByName(names, values, {"a", "b"}, {1.0, 2.0});
    EXPECT_EQ(names, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(values, (std::vector<double>{1.0, 2.0}));
}

TEST(TelemetryAggregator, UpdateByNamePreservesOtherSideAcrossAlternatingMessages) {
    // Mirrors /chaser/arm/torque_command: arm_nmpc_left and arm_nmpc_right
    // each publish only their own 3 joints, alternating on the shared topic.
    std::vector<std::string> names;
    std::vector<double> values;
    TelemetryAggregator::updateByName(names, values, {"joint2_L", "joint3_L"}, {1.0, 2.0});
    TelemetryAggregator::updateByName(names, values, {"joint2_R", "joint3_R"}, {3.0, 4.0});
    // A second L-side message must update L's values without dropping R's.
    TelemetryAggregator::updateByName(names, values, {"joint2_L", "joint3_L"}, {10.0, 20.0});

    auto merged = TelemetryAggregator::mergeByName(
        {"joint2_L", "joint3_L", "joint2_R", "joint3_R"}, names, values, -1.0);
    EXPECT_EQ(merged, (std::vector<double>{10.0, 20.0, 3.0, 4.0}));
}

int main(int argc, char** argv) { testing::InitGoogleTest(&argc, argv); return RUN_ALL_TESTS(); }
