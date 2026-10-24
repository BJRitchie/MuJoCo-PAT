#pragma once
// The sensor model moved to the pat_sensors package so the offline benches can
// use it without depending on the simulator. This header keeps the old include
// path and the pat_simulation names working for the simulator's own code.
#include "pat_sensors/sensor_model.hpp"

namespace pat_simulation {
using pat_sensors::ChannelConfig;
using pat_sensors::SensorChannel;
using pat_sensors::SensorModel;
using pat_sensors::channelSeed;
}  // namespace pat_simulation
