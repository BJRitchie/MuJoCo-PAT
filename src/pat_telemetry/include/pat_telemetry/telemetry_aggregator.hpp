#pragma once

#include <string>
#include <vector>

namespace pat_telemetry
{

// Zero-ROS-dependency helpers backing telemetry_node. Kept separate so the
// name-matching and distance logic is unit-testable without spinning up ROS.
class TelemetryAggregator
{
public:
    // Merges two name-keyed value arrays (e.g. joint_states' position/
    // velocity and torque_command's effort) into one array ordered by
    // `names_out` - matched by NAME, not array position, same requirement
    // CLAUDE.md calls out for every other joint_states/torque_command
    // consumer in this repo (see ArmNMPCNode::applyNamedValues). A name
    // present in `names_out` but missing from `names`/`values` keeps
    // `fallback` in the result.
    static std::vector<double> mergeByName(
        const std::vector<std::string>& names_out,
        const std::vector<std::string>& names,
        const std::vector<double>& values,
        double fallback = 0.0);

    // Planar (x,y) Euclidean distance - used for EE pose vs. EE setpoint.
    static double planarDistance(double x1, double y1, double x2, double y2);

    // Merges (names_in, values_in) into a persistent (names, values) cache
    // IN PLACE - existing names are updated, new names appended. Needed
    // because /chaser/arm/torque_command carries only one side's joints per
    // message (each pat_arm_nmpc instance publishes independently onto the
    // shared topic, per CLAUDE.md) - a wholesale cache replacement on every
    // message would lose the other side's last-known torque.
    static void updateByName(
        std::vector<std::string>& names,
        std::vector<double>& values,
        const std::vector<std::string>& names_in,
        const std::vector<double>& values_in);
};

}  // namespace pat_telemetry
