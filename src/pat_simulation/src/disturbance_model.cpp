#include "pat_simulation/sensors/disturbance_model.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pat_simulation {

DisturbanceModel::DisturbanceModel(const WrenchProfile& profile, double dt,
                                   std::uint64_t seed)
    : profile_(profile), dt_(dt), rng_(seed) {
    const std::string& t = profile_.type;
    if (t != "none" && t != "" && t != "step" && t != "sinusoid" &&
        t != "random_walk") {
        throw std::invalid_argument(
            "DisturbanceModel: unknown profile type '" + t +
            "' (expected none, step, sinusoid or random_walk)");
    }
    if (profile_.duration_s < 0.0) {
        throw std::invalid_argument("DisturbanceModel: negative duration_s");
    }
    if (profile_.start_s < 0.0) {
        throw std::invalid_argument("DisturbanceModel: negative start_s");
    }
    if (t == "sinusoid" && profile_.freq_hz <= 0.0) {
        throw std::invalid_argument(
            "DisturbanceModel: a sinusoid needs a positive freq_hz");
    }
    if (t == "random_walk" && profile_.walk_std <= 0.0) {
        throw std::invalid_argument(
            "DisturbanceModel: a random_walk needs a positive walk_std");
    }
    walk_step_ = profile_.walk_std * std::sqrt(dt_);
}

bool DisturbanceModel::inWindow(double t) const noexcept {
    if (t < profile_.start_s) return false;
    if (profile_.duration_s > 0.0 && t >= profile_.start_s + profile_.duration_s) {
        return false;
    }
    return true;
}

void DisturbanceModel::wrenchAt(double t, double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    if (!profile_.active()) return;

    const double amp[3] = {profile_.fx, profile_.fy, profile_.tz};

    if (profile_.type == "random_walk") {
        // The walk advances whenever the profile is in its window, so that
        // leaving the window and coming back does not reset the drift -- a
        // physical disturbance that stops being applied has still drifted.
        if (inWindow(t)) {
            for (int i = 0; i < 3; ++i) {
                if (amp[i] == 0.0) continue;   // an unused axis stays exactly 0
                walk_[i] += walk_step_ * gauss_(rng_);
                out[i] = amp[i] * walk_[i];
            }
        }
        return;
    }

    if (!inWindow(t)) return;

    if (profile_.type == "step") {
        out[0] = amp[0];
        out[1] = amp[1];
        out[2] = amp[2];
    } else if (profile_.type == "sinusoid") {
        // Phase measured from start_s, so the wrench begins at zero rather
        // than jumping to the amplitude the instant it switches on.
        const double phase = 2.0 * M_PI * profile_.freq_hz * (t - profile_.start_s);
        const double s = std::sin(phase);
        out[0] = amp[0] * s;
        out[1] = amp[1] * s;
        out[2] = amp[2] * s;
    }
}

std::string DisturbanceModel::describe() const {
    if (!profile_.active()) return "none";
    std::ostringstream os;
    os << profile_.type << " fx=" << profile_.fx << " fy=" << profile_.fy
       << " tz=" << profile_.tz;
    if (profile_.start_s > 0.0)    os << " start=" << profile_.start_s << "s";
    if (profile_.duration_s > 0.0) os << " for " << profile_.duration_s << "s";
    else                           os << " indefinitely";
    if (profile_.type == "sinusoid")    os << " at " << profile_.freq_hz << "Hz";
    if (profile_.type == "random_walk") os << " walk=" << profile_.walk_std
                                           << "/sqrt(s)";
    return os.str();
}

} // namespace pat_simulation
