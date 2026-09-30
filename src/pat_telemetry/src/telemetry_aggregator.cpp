#include "pat_telemetry/telemetry_aggregator.hpp"

#include <cmath>

namespace pat_telemetry
{

std::vector<double> TelemetryAggregator::mergeByName(
    const std::vector<std::string>& names_out,
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    double fallback)
{
    std::vector<double> out(names_out.size(), fallback);
    for (size_t k = 0; k < names.size() && k < values.size(); ++k) {
        for (size_t i = 0; i < names_out.size(); ++i) {
            if (names[k] == names_out[i]) { out[i] = values[k]; break; }
        }
    }
    return out;
}

double TelemetryAggregator::planarDistance(double x1, double y1, double x2, double y2)
{
    return std::hypot(x2 - x1, y2 - y1);
}

void TelemetryAggregator::updateByName(
    std::vector<std::string>& names,
    std::vector<double>& values,
    const std::vector<std::string>& names_in,
    const std::vector<double>& values_in)
{
    for (size_t k = 0; k < names_in.size() && k < values_in.size(); ++k) {
        bool found = false;
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == names_in[k]) { values[i] = values_in[k]; found = true; break; }
        }
        if (!found) { names.push_back(names_in[k]); values.push_back(values_in[k]); }
    }
}

}  // namespace pat_telemetry
