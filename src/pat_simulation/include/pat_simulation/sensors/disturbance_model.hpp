#pragma once
#include <cstdint>
#include <random>
#include <string>

namespace pat_simulation {

/// A planar external wrench to apply to one body, as a function of time.
///
/// This is the instrument the disturbance observer is scored against: the
/// injected wrench is known exactly, so "the observer recovered 0.42 of the
/// 0.50 N we applied" is a measurement rather than an impression. Contact
/// forces are the other source, and those are read back from MuJoCo's own
/// d->qfrc_constraint instead -- they cannot be commanded.
///
/// Planar only (fx, fy, tz), matching the table: a z force or an x/y torque
/// would be absorbed by the slide/hinge joints and measure nothing.
struct WrenchProfile {
    /// "none", "step", "sinusoid" or "random_walk". Anything else throws.
    std::string type{"none"};

    // Amplitude. For "random_walk" these scale each axis's walk, so an axis
    // left at zero stays exactly zero rather than picking up noise.
    double fx{0.0};   ///< [N], world frame
    double fy{0.0};   ///< [N], world frame
    double tz{0.0};   ///< [N*m], about world z

    double start_s{0.0};     ///< when it begins; before this the wrench is zero
    double duration_s{0.0};  ///< how long it lasts; <= 0 means indefinitely
    double freq_hz{0.0};     ///< "sinusoid" only
    double walk_std{0.0};    ///< "random_walk" only, per sqrt(s), scaled by f*/tz

    bool active() const noexcept { return type != "none" && type != ""; }
};

/// Evaluates a WrenchProfile step by step.
///
/// Deliberately not allocation-free-critical: this runs in the simulator, not
/// in a control loop.
class DisturbanceModel {
public:
    DisturbanceModel() = default;

    /// `dt` is the physics step. Throws std::invalid_argument on an unknown
    /// profile type, a negative duration, or a sinusoid with no frequency.
    DisturbanceModel(const WrenchProfile& profile, double dt, std::uint64_t seed);

    /// The wrench at simulation time `t`, in the world frame. Advances the
    /// random walk, so call it exactly once per step and in order.
    /// `out` receives {fx, fy, tz}.
    void wrenchAt(double t, double out[3]);

    /// True when the profile does nothing at all, so the caller can skip
    /// touching xfrc_applied entirely.
    bool active() const noexcept { return profile_.active(); }

    const WrenchProfile& profile() const noexcept { return profile_; }

    /// One line describing the profile, for the startup log.
    std::string describe() const;

private:
    WrenchProfile profile_;
    double dt_{0.002};
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};
    double walk_[3]{0.0, 0.0, 0.0};
    double walk_step_{0.0};

    /// Is `t` inside the profile's window?
    bool inWindow(double t) const noexcept;
};

} // namespace pat_simulation
