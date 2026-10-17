#include "pat_simulation/sensors/sensor_model.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace pat_simulation {

std::uint64_t channelSeed(const std::string& name, std::uint64_t master) {
    // FNV-1a, 64-bit.
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : name) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ULL;
    }
    // Mix in the master seed so one seed change moves every channel, while two
    // different channels under the same seed stay independent.
    h ^= master + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

SensorChannel::SensorChannel(std::string name, const ChannelConfig& cfg,
                             double dt, std::uint64_t seed)
    : name_(std::move(name)), cfg_(cfg), dt_(dt), perfect_(cfg.perfect()),
      rng_(channelSeed(name_, seed)), bias_now_(cfg.bias) {
    walk_step_ = cfg_.bias_walk_std * std::sqrt(dt_);

    // Round the delay to whole steps. A delay shorter than half a step is no
    // delay: at 500 Hz the granularity is 2 ms and pretending otherwise would
    // claim a resolution the simulation does not have.
    delay_steps_ = static_cast<int>(std::lround(cfg_.delay_s / dt_));
    if (delay_steps_ > 0) {
        // delay_steps_ samples have to be in flight, plus the one being written.
        ring_.assign(static_cast<std::size_t>(delay_steps_) + 1, 0.0);
    }

    if (cfg_.rate_hz > 0.0) {
        rate_steps_ = static_cast<int>(std::lround(1.0 / (cfg_.rate_hz * dt_)));
        if (rate_steps_ < 1) rate_steps_ = 1;   // faster than the sim: every step
    }
}

double SensorChannel::apply(double truth) {
    // The whole point of the zero-config case: no arithmetic, no RNG draw, so
    // the output is bit-for-bit the input and no other channel's stream moves.
    if (perfect_) return truth;

    // 1. Bias random walk, one step. Done before the due-check so the walk
    //    advances with wall time rather than with the sensor's own rate -- bias
    //    drifts whether or not the sensor happened to read this step.
    if (walk_step_ != 0.0) bias_now_ += walk_step_ * gauss_(rng_);

    // 2. Is the sensor due to produce a reading this step?
    const bool due = (step_ % rate_steps_) == 0;
    ++step_;

    double sample;
    if (!due && have_held_) {
        sample = held_;
    } else {
        const bool dropped = cfg_.dropout_prob > 0.0 &&
                             uniform_(rng_) < cfg_.dropout_prob;
        if (dropped && have_held_) {
            sample = held_;
        } else {
            sample = truth;
            if (cfg_.scale_error != 0.0) sample *= (1.0 + cfg_.scale_error);
            if (bias_now_ != 0.0)        sample += bias_now_;
            if (cfg_.noise_std != 0.0)   sample += cfg_.noise_std * gauss_(rng_);
            if (cfg_.quantisation > 0.0) {
                sample = std::round(sample / cfg_.quantisation) * cfg_.quantisation;
            }
            held_      = sample;
            have_held_ = true;
        }
    }

    // 3. Transport delay. The ring is pre-sized, so this is index arithmetic.
    if (delay_steps_ == 0) return sample;
    if (!ring_primed_) {
        // Prime with the first reading rather than leaving the zeros the
        // constructor put there. An unprimed ring reports 0 for its first
        // delay_steps_ steps, and a step from 0 to the true value is
        // indistinguishable from a real disturbance to anything watching the
        // signal -- which is precisely what this model exists to be watched by.
        for (auto& v : ring_) v = sample;
        ring_primed_ = true;
    }
    ring_[static_cast<std::size_t>(ring_pos_)] = sample;
    ring_pos_ = (ring_pos_ + 1) % static_cast<int>(ring_.size());
    return ring_[static_cast<std::size_t>(ring_pos_)];
}

void SensorChannel::reset() {
    bias_now_  = cfg_.bias;
    step_      = 0;
    held_      = 0.0;
    have_held_ = false;
    ring_pos_  = 0;
    ring_primed_ = false;
    for (auto& v : ring_) v = 0.0;
    // normal_distribution generates values in pairs and caches the spare, so
    // without this a reset run would start mid-pair and diverge from the
    // original stream.
    gauss_.reset();
    uniform_.reset();
}

int SensorModel::addChannel(const std::string& name, const ChannelConfig& cfg) {
    return addChannel(name, cfg, dt_);
}

int SensorModel::addChannel(const std::string& name, const ChannelConfig& cfg,
                            double dt) {
    if (!(dt > 0.0)) {
        throw std::invalid_argument("SensorModel: channel '" + name +
                                    "' needs a positive call period");
    }
    if (handle(name) >= 0) {
        throw std::invalid_argument("SensorModel: duplicate channel '" + name + "'");
    }
    const double nonneg[] = {cfg.noise_std, cfg.bias_walk_std, cfg.quantisation,
                             cfg.delay_s,   cfg.dropout_prob,  cfg.rate_hz};
    for (double v : nonneg) {
        if (v < 0.0) {
            throw std::invalid_argument("SensorModel: negative parameter on channel '" +
                                        name + "'");
        }
    }
    if (cfg.dropout_prob > 1.0) {
        throw std::invalid_argument("SensorModel: dropout_prob > 1 on channel '" +
                                    name + "'");
    }
    channels_.emplace_back(name, cfg, dt, seed_);
    return static_cast<int>(channels_.size()) - 1;
}

int SensorModel::handle(const std::string& name) const {
    for (std::size_t i = 0; i < channels_.size(); ++i) {
        if (channels_[i].name() == name) return static_cast<int>(i);
    }
    return -1;
}

void SensorModel::applyVector(int h, const double* truth, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = channels_.at(h + i).apply(truth[i]);
}

void SensorModel::reset() {
    for (auto& c : channels_) c.reset();
}

bool SensorModel::perfect() const {
    for (const auto& c : channels_) {
        if (!c.config().perfect()) return false;
    }
    return true;
}

std::string SensorModel::report() const {
    std::ostringstream os;
    for (const auto& c : channels_) {
        const ChannelConfig& k = c.config();
        if (k.perfect()) continue;
        os << "\n  " << c.name() << ':';
        if (k.noise_std != 0.0)     os << " sigma=" << k.noise_std;
        if (k.bias != 0.0)          os << " bias=" << k.bias;
        if (k.bias_walk_std != 0.0) os << " walk=" << k.bias_walk_std << "/sqrt(s)";
        if (k.scale_error != 0.0)   os << " scale=" << k.scale_error;
        if (k.quantisation != 0.0)  os << " lsb=" << k.quantisation;
        if (k.delay_s != 0.0)       os << " delay=" << c.delaySteps() << " steps";
        if (k.dropout_prob != 0.0)  os << " dropout=" << k.dropout_prob;
        if (k.rate_hz != 0.0)       os << " rate=" << k.rate_hz << "Hz";
    }
    return os.str();
}

} // namespace pat_simulation
