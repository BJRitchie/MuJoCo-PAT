#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace pat_simulation {

/// How one scalar sensor channel departs from the truth.
///
/// Every field is an imperfection, so an all-zero config is an exact
/// pass-through: apply() returns its input bit-for-bit and never draws from the
/// RNG. That is deliberate and load-bearing. Every number in the README's
/// realtime section, the constraint ladder and the warm-start decision was
/// measured against exact measurements, so introducing this model must not move
/// any of them unless someone asks for an imperfection by name.
struct ChannelConfig {
    double noise_std{0.0};      ///< additive white Gaussian sigma, channel units
    double bias{0.0};           ///< constant offset, channel units
    double bias_walk_std{0.0};  ///< random walk driving the bias, units/sqrt(s)
    double scale_error{0.0};    ///< multiplicative: out = (1 + scale_error) * in
    double quantisation{0.0};   ///< LSB size, channel units; 0 disables
    double delay_s{0.0};        ///< transport delay, rounded to whole update steps
    double dropout_prob{0.0};   ///< per-sample chance the reading is not refreshed
    double rate_hz{0.0};        ///< the sensor's own update rate; 0 = every step

    /// True when this channel is an exact pass-through.
    bool perfect() const noexcept {
        return noise_std == 0.0 && bias == 0.0 && bias_walk_std == 0.0 &&
               scale_error == 0.0 && quantisation == 0.0 && delay_s == 0.0 &&
               dropout_prob == 0.0 && rate_hz == 0.0;
    }

    bool operator==(const ChannelConfig& o) const noexcept {
        return noise_std == o.noise_std && bias == o.bias &&
               bias_walk_std == o.bias_walk_std && scale_error == o.scale_error &&
               quantisation == o.quantisation && delay_s == o.delay_s &&
               dropout_prob == o.dropout_prob && rate_hz == o.rate_hz;
    }
    bool operator!=(const ChannelConfig& o) const noexcept { return !(*this == o); }
};

/// One scalar measurement channel: truth in, corrupted reading out.
///
/// The chain, in the order a real sensor applies it:
///
///   truth -> scale error -> bias (constant + random walk) -> white noise
///         -> quantisation -> rate limit / dropout (sample-and-hold)
///         -> transport delay
///
/// Rate limit and dropout act at the sampling point, because that is where a
/// real sensor either produces a reading or does not; the delay then transports
/// whatever it produced. A held sample is the previous reading, which is what a
/// consumer holding its last-known value actually sees.
///
/// apply() allocates nothing: the delay ring is sized once in the constructor.
class SensorChannel {
public:
    SensorChannel() = default;

    /// `dt` is the period at which apply() will be called. Every time-based
    /// setting -- delay_s, rate_hz, bias_walk_std -- is converted to calls of
    /// apply() through it, so it must be the real call period: a channel
    /// advanced once per 10 ms publish but built with the 2 ms physics step
    /// delays 5x too long, samples 5x too slowly and walks at sqrt(1/5) of its
    /// density.
    /// `seed` is the master seed; this channel derives its own stream from it
    /// and `name` (see kFnvOffset below for why the name and not an index).
    SensorChannel(std::string name, const ChannelConfig& cfg, double dt,
                  std::uint64_t seed);

    /// Advance one step and return the reading corresponding to `truth`.
    double apply(double truth);

    /// Back to the constructed state: zero walk, empty delay ring, no held
    /// sample. Does NOT reseed, so a reset run continues the same stream.
    void reset();

    const std::string&  name()   const noexcept { return name_; }
    const ChannelConfig& config() const noexcept { return cfg_; }

    /// The bias in force right now (constant + accumulated walk). Exposed so a
    /// test can check the walk statistically, and so the sim-only truth topic
    /// can report what was actually injected.
    double bias() const noexcept { return bias_now_; }

    /// How many whole steps of delay the ring is holding.
    int delaySteps() const noexcept { return delay_steps_; }

private:
    std::string   name_;
    ChannelConfig cfg_;
    double        dt_{0.0};
    bool          perfect_{true};

    std::mt19937_64                   rng_;
    std::normal_distribution<double>  gauss_{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};

    double bias_now_{0.0};
    double walk_step_{0.0};   ///< bias_walk_std * sqrt(dt), precomputed

    // Rate limiting. Counted in steps rather than accumulated seconds so a long
    // run cannot drift: at 500 Hz a 5 Hz sensor is exactly every 100th step.
    int step_{0};
    int rate_steps_{1};
    double held_{0.0};
    bool   have_held_{false};

    // Delay ring. delay_steps_ == 0 bypasses it entirely. Primed with the first
    // reading on first use, not with zeros -- see apply().
    std::vector<double> ring_;
    int  ring_pos_{0};
    int  delay_steps_{0};
    bool ring_primed_{false};
};

/// A named set of channels sharing one master seed.
///
/// Lookup is by name and resolved to an index once, at registration, so the
/// per-step path is a vector index and never a string compare.
class SensorModel {
public:
    SensorModel() = default;
    SensorModel(double dt, std::uint64_t seed) : dt_(dt), seed_(seed) {}

    /// Register a channel and return its handle. Handles are stable; names must
    /// be unique. Throws std::invalid_argument on a duplicate name, a negative
    /// parameter, or a dropout probability outside [0, 1].
    int addChannel(const std::string& name, const ChannelConfig& cfg);

    /// As above, for a channel advanced at its own period `dt` rather than
    /// the model's. Channels published at different rates -- an IMU faster
    /// than odometry, say -- each need the period they are actually called at.
    int addChannel(const std::string& name, const ChannelConfig& cfg, double dt);

    /// Handle for a registered name, or -1 if there is none.
    int handle(const std::string& name) const;

    /// Corrupt one reading. `h` must come from addChannel().
    double apply(int h, double truth) { return channels_.at(h).apply(truth); }

    /// Corrupt `n` readings of a vector-valued sensor whose channels were
    /// registered consecutively (accelerometer x/y/z, say). `h` is the first
    /// handle; truth and out may alias.
    void applyVector(int h, const double* truth, double* out, int n);

    void reset();

    /// True when every registered channel is an exact pass-through, i.e. the
    /// whole model is a no-op. The node logs this so a trial's provenance says
    /// plainly whether measurements were perfect.
    bool perfect() const;

    std::uint64_t seed() const noexcept { return seed_; }
    std::size_t   size() const noexcept { return channels_.size(); }
    const SensorChannel& channel(int h) const { return channels_.at(h); }

    /// One line per imperfect channel, for the startup log. Empty when perfect.
    std::string report() const;

private:
    double        dt_{0.002};
    std::uint64_t seed_{0};
    std::vector<SensorChannel> channels_;
};

/// FNV-1a over the channel name, mixed with the master seed.
///
/// A name hash rather than a registration index, so each channel's stream
/// depends only on its own name and the master seed. Adding a channel, or
/// reordering the config, then leaves every other channel's realised sequence
/// untouched -- which is what makes an A/B run mean anything: turning the
/// accelerometer's noise on does not also change the gyro's draws. std::hash is
/// not used because it is not stable across standard library versions, and a
/// trial recorded today has to be reproducible later.
std::uint64_t channelSeed(const std::string& name, std::uint64_t master);

} // namespace pat_simulation
