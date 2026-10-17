#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pat_simulation/sensors/sensor_model.hpp"

using pat_simulation::ChannelConfig;
using pat_simulation::SensorChannel;
using pat_simulation::SensorModel;

namespace {

constexpr double kDt = 0.002;   // the sim's 500 Hz step

double sampleStd(const std::vector<double>& v) {
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double ss = 0.0;
    for (double x : v) ss += (x - mean) * (x - mean);
    return std::sqrt(ss / static_cast<double>(v.size() - 1));
}

double sampleMean(const std::vector<double>& v) {
    double mean = 0.0;
    for (double x : v) mean += x;
    return mean / static_cast<double>(v.size());
}

} // namespace

// ---------------------------------------------------------------------------
// The gate that protects every number already committed to the README: with no
// imperfection configured, the model must be an exact pass-through.
// ---------------------------------------------------------------------------

TEST(SensorModelPerfect, ZeroConfigIsBitForBitIdentity) {
    SensorModel m(kDt, 12345);
    const int h = m.addChannel("x", ChannelConfig{});

    // Values chosen to catch any arithmetic at all: a multiply by 1.0 or an add
    // of 0.0 would survive these, but a reordering or a fused operation would
    // not, and neither would an accidental quantisation.
    const double probes[] = {0.0, -0.0, 1.0, -1.0, 0.1, -0.5435345234,
                             1e-300, 1e300, 123456.789012345,
                             M_PI, -M_E, 0.30000000000000004};
    for (double p : probes) {
        const double out = m.apply(h, p);
        EXPECT_EQ(out, p) << "input " << p << " came back as " << out;
        // Bit-for-bit, not just ==, so that -0.0 cannot silently become +0.0.
        EXPECT_EQ(std::signbit(out), std::signbit(p));
    }
    EXPECT_TRUE(m.perfect());
    EXPECT_TRUE(m.report().empty());
}

TEST(SensorModelPerfect, NonFiniteTruthPassesThroughUntouched) {
    // The sim should never hand us these, but if it does, a perfect channel
    // must not be the thing that hides it -- simulation_node has its own
    // non-finite guards and they need to see what MuJoCo actually produced.
    SensorModel m(kDt, 1);
    const int h = m.addChannel("x", ChannelConfig{});
    EXPECT_TRUE(std::isnan(m.apply(h, std::nan(""))));
    EXPECT_EQ(m.apply(h, INFINITY), INFINITY);
    EXPECT_EQ(m.apply(h, -INFINITY), -INFINITY);
}

TEST(SensorModelPerfect, PerfectChannelNeverTouchesItsRng) {
    // Two models with the same seed, one of which has run a perfect channel a
    // great many times first. If the perfect path drew from the RNG, the noisy
    // channel's stream would differ.
    ChannelConfig noisy;
    noisy.noise_std = 0.01;

    SensorModel a(kDt, 777);
    const int ha = a.addChannel("gyro", noisy);

    SensorModel b(kDt, 777);
    const int perfect = b.addChannel("accel", ChannelConfig{});
    const int hb      = b.addChannel("gyro", noisy);
    for (int i = 0; i < 10000; ++i) b.apply(perfect, 1.0);

    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(a.apply(ha, 0.0), b.apply(hb, 0.0)) << "at i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Each imperfection delivers what was asked for.
// ---------------------------------------------------------------------------

TEST(SensorModelNoise, RealisedSigmaMatchesTheRequest) {
    ChannelConfig cfg;
    cfg.noise_std = 0.03;
    SensorModel m(kDt, 42);
    const int h = m.addChannel("gyro", cfg);

    std::vector<double> out;
    out.reserve(200000);
    for (int i = 0; i < 200000; ++i) out.push_back(m.apply(h, 0.0));

    // At n=2e5 the sample std of a Gaussian is within ~0.16% of sigma, so 3%
    // is far outside flake range while still catching a factor-of-anything bug.
    EXPECT_NEAR(sampleStd(out), cfg.noise_std, 0.03 * cfg.noise_std);
    EXPECT_NEAR(sampleMean(out), 0.0, 5e-4);
}

TEST(SensorModelBias, ConstantBiasIsAnExactOffset) {
    ChannelConfig cfg;
    cfg.bias = -0.25;
    SensorModel m(kDt, 42);
    const int h = m.addChannel("accel", cfg);
    EXPECT_DOUBLE_EQ(m.apply(h, 1.0), 0.75);
    EXPECT_DOUBLE_EQ(m.apply(h, 0.0), -0.25);
}

TEST(SensorModelBias, RandomWalkGrowsAsSqrtTime) {
    // One realisation says nothing, so run an ensemble of independently-seeded
    // channels and check the spread across them. sqrt(t) scaling is the claim
    // the units (per sqrt(second)) make, so it is the thing to verify.
    constexpr double kWalk = 0.01;      // units/sqrt(s)
    constexpr int kEnsemble = 400;
    constexpr int kSteps   = 5000;      // 10 s at 500 Hz
    constexpr int kQuarter = 1250;      // 2.5 s, a 4x span to check sqrt(t) on

    ChannelConfig cfg;
    cfg.bias_walk_std = kWalk;

    std::vector<SensorChannel> ens;
    ens.reserve(kEnsemble);
    for (int i = 0; i < kEnsemble; ++i) {
        ens.emplace_back("walk" + std::to_string(i), cfg, kDt, 9001);
    }

    // Spread of the bias across the ensemble at this instant.
    auto spread = [&] {
        std::vector<double> b;
        b.reserve(kEnsemble);
        for (auto& c : ens) b.push_back(c.bias());
        return sampleStd(b);
    };

    for (int s = 0; s < kQuarter; ++s) {
        for (auto& c : ens) c.apply(0.0);
    }
    const double sd_quarter = spread();
    for (int s = kQuarter; s < kSteps; ++s) {
        for (auto& c : ens) c.apply(0.0);
    }
    const double sd_full = spread();

    // Expected sigma after t seconds is kWalk*sqrt(t).
    const double t_quarter = kDt * static_cast<double>(kQuarter);
    const double t_full    = kDt * static_cast<double>(kSteps);
    EXPECT_NEAR(sd_quarter, kWalk * std::sqrt(t_quarter), 0.15 * kWalk * std::sqrt(t_quarter));
    EXPECT_NEAR(sd_full,    kWalk * std::sqrt(t_full),    0.15 * kWalk * std::sqrt(t_full));
    // And the ratio over a 4x span must be the 2x that sqrt(t) demands.
    EXPECT_NEAR(sd_full / sd_quarter, 2.0, 0.3);
}

TEST(SensorModelScale, ScaleErrorIsMultiplicative) {
    ChannelConfig cfg;
    cfg.scale_error = 0.02;             // +2% sensitivity error
    SensorModel m(kDt, 42);
    const int h = m.addChannel("accel", cfg);
    EXPECT_DOUBLE_EQ(m.apply(h, 10.0), 10.2);
    // Zero truth stays zero: a scale error alone cannot invent a reading.
    EXPECT_DOUBLE_EQ(m.apply(h, 0.0), 0.0);
}

TEST(SensorModelQuantisation, SnapsToTheLsb) {
    ChannelConfig cfg;
    cfg.quantisation = 0.01;
    SensorModel m(kDt, 42);
    const int h = m.addChannel("enc", cfg);
    EXPECT_NEAR(m.apply(h, 0.0123), 0.01, 1e-12);
    EXPECT_NEAR(m.apply(h, 0.0167), 0.02, 1e-12);
    EXPECT_NEAR(m.apply(h, -0.0123), -0.01, 1e-12);
    // Exactly representable values must not move.
    EXPECT_NEAR(m.apply(h, 0.05), 0.05, 1e-12);
}

TEST(SensorModelDelay, DelaysByWholeStepsAndPrimesWithTheFirstReading) {
    ChannelConfig cfg;
    cfg.delay_s = 0.01;                 // 10 ms = 5 steps at 500 Hz
    SensorModel m(kDt, 42);
    const int h = m.addChannel("star", cfg);
    EXPECT_EQ(m.channel(h).delaySteps(), 5);

    // The first reading comes straight out rather than five steps of zero --
    // an unprimed ring would look like a step disturbance at startup.
    EXPECT_DOUBLE_EQ(m.apply(h, 7.0), 7.0);

    // Now a ramp: each output should be the input from five steps earlier.
    std::vector<double> in, out;
    for (int i = 1; i <= 50; ++i) {
        in.push_back(static_cast<double>(i));
        out.push_back(m.apply(h, static_cast<double>(i)));
    }
    for (std::size_t i = 5; i < in.size(); ++i) {
        EXPECT_DOUBLE_EQ(out[i], in[i - 5]) << "at i=" << i;
    }
}

TEST(SensorModelDelay, SubStepDelayRoundsToNoDelay) {
    // At 500 Hz the granularity is 2 ms. A 0.4 ms delay is not representable
    // and must round to none rather than silently becoming one whole step.
    ChannelConfig cfg;
    cfg.delay_s = 0.0004;
    SensorModel m(kDt, 42);
    const int h = m.addChannel("fast", cfg);
    EXPECT_EQ(m.channel(h).delaySteps(), 0);
    EXPECT_DOUBLE_EQ(m.apply(h, 3.0), 3.0);
}

TEST(SensorModelRate, SlowSensorHoldsBetweenItsOwnUpdates) {
    ChannelConfig cfg;
    cfg.rate_hz = 5.0;                  // a star tracker: every 100th step
    SensorModel m(kDt, 42);
    const int h = m.addChannel("star", cfg);

    // Feed a ramp and record which steps produced a new value.
    std::vector<int> refreshed;
    double prev = m.apply(h, 0.0);
    for (int i = 1; i < 500; ++i) {
        const double v = m.apply(h, static_cast<double>(i));
        if (v != prev) refreshed.push_back(i);
        prev = v;
    }
    // Steps 100, 200, 300, 400 -- exactly, with no drift.
    ASSERT_EQ(refreshed.size(), 4u);
    EXPECT_EQ(refreshed[0], 100);
    EXPECT_EQ(refreshed[1], 200);
    EXPECT_EQ(refreshed[2], 300);
    EXPECT_EQ(refreshed[3], 400);
}

TEST(SensorModelPeriod, TimeSettingsMeanSecondsWhateverTheCallRate) {
    // Two channels with the same configuration, one advanced at the 500 Hz
    // physics rate and one at a 100 Hz publish rate. Delay, sample rate and
    // bias walk are specified in seconds, so both must delay by 0.1 s, sample
    // every 0.2 s and walk at the same density -- in their own call counts,
    // 50 vs 10 calls, 100 vs 20 calls.
    ChannelConfig cfg;
    cfg.delay_s = 0.1;
    cfg.rate_hz = 5.0;
    SensorModel m(kDt, 42);
    const int fast = m.addChannel("fast", cfg);              // model period, 2 ms
    const int slow = m.addChannel("slow", cfg, 0.01);        // its own, 10 ms
    EXPECT_EQ(m.channel(fast).delaySteps(), 50);
    EXPECT_EQ(m.channel(slow).delaySteps(), 10);

    // Samples land every 0.2 s of simulated time on both.
    std::vector<int> fresh_slow;
    double prev = m.apply(slow, 0.0);
    for (int i = 1; i < 100; ++i) {               // 1 s at 100 Hz
        const double v = m.apply(slow, static_cast<double>(i));
        if (v != prev) fresh_slow.push_back(i);
        prev = v;
    }
    ASSERT_GE(fresh_slow.size(), 2u);
    EXPECT_EQ(fresh_slow[1] - fresh_slow[0], 20) << "0.2 s at 100 Hz";

    // Bias walk: the same density per sqrt(second) on both periods.
    ChannelConfig walk;
    walk.bias_walk_std = 0.3;
    std::vector<double> end_fast, end_slow;
    for (int r = 0; r < 400; ++r) {
        SensorModel w(kDt, 1000 + static_cast<std::uint64_t>(r));
        const int a = w.addChannel("a", walk);
        const int b = w.addChannel("b", walk, 0.01);
        double va = 0.0, vb = 0.0;
        for (int i = 0; i < 500; ++i) va = w.apply(a, 0.0);   // 1 s at 500 Hz
        for (int i = 0; i < 100; ++i) vb = w.apply(b, 0.0);   // 1 s at 100 Hz
        end_fast.push_back(va);
        end_slow.push_back(vb);
    }
    // After 1 s a walk of density 0.3/sqrt(s) has sigma 0.3 at either rate.
    EXPECT_NEAR(sampleStd(end_fast), 0.3, 0.03);
    EXPECT_NEAR(sampleStd(end_slow), 0.3, 0.03);
}

TEST(SensorModelPeriod, RejectsANonPositivePeriod) {
    SensorModel m(kDt, 42);
    EXPECT_THROW(m.addChannel("x", ChannelConfig{}, 0.0), std::invalid_argument);
    EXPECT_THROW(m.addChannel("y", ChannelConfig{}, -0.01), std::invalid_argument);
}

TEST(SensorModelRate, RateFasterThanTheSimUpdatesEveryStep) {
    ChannelConfig cfg;
    cfg.rate_hz = 5000.0;               // 10x the sim rate
    SensorModel m(kDt, 42);
    const int h = m.addChannel("fast", cfg);
    for (int i = 0; i < 20; ++i) {
        EXPECT_DOUBLE_EQ(m.apply(h, static_cast<double>(i)), static_cast<double>(i));
    }
}

TEST(SensorModelDropout, HoldsAtRoughlyTheRequestedRate) {
    ChannelConfig cfg;
    cfg.dropout_prob = 0.25;
    SensorModel m(kDt, 42);
    const int h = m.addChannel("rel", cfg);

    // A strictly increasing truth, so a held sample is detectable as an output
    // that failed to change.
    constexpr int kN = 100000;
    int held = 0;
    double prev = m.apply(h, 0.0);
    for (int i = 1; i < kN; ++i) {
        const double v = m.apply(h, static_cast<double>(i));
        if (v == prev) ++held;
        prev = v;
    }
    const double realised = static_cast<double>(held) / (kN - 1);
    EXPECT_NEAR(realised, cfg.dropout_prob, 0.01);
}

// ---------------------------------------------------------------------------
// Reproducibility and channel independence -- what makes an A/B trial mean
// something.
// ---------------------------------------------------------------------------

TEST(SensorModelSeed, SameSeedReproducesTheRunExactly) {
    ChannelConfig cfg;
    cfg.noise_std     = 0.01;
    cfg.bias_walk_std = 0.001;
    cfg.dropout_prob  = 0.1;

    auto run = [&](std::uint64_t seed) {
        SensorModel m(kDt, seed);
        const int h = m.addChannel("gyro", cfg);
        std::vector<double> v;
        for (int i = 0; i < 2000; ++i) v.push_back(m.apply(h, std::sin(i * 0.01)));
        return v;
    };

    const auto a = run(2024);
    const auto b = run(2024);
    const auto c = run(2025);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "same seed diverged at i=" << i;
    }
    EXPECT_NE(a, c) << "a different seed produced an identical run";
}

TEST(SensorModelSeed, ChannelsAreIndependentOfEachOther) {
    // The claim the name-derived seed makes: a channel's realised sequence
    // depends only on its own name and the master seed. So enabling noise on
    // the accelerometer must leave the gyro's sequence untouched, and the two
    // must not be the same sequence either.
    ChannelConfig noisy;
    noisy.noise_std = 0.01;

    SensorModel gyro_only(kDt, 55);
    const int g1 = gyro_only.addChannel("gyro_z", noisy);

    SensorModel both(kDt, 55);
    const int a2 = both.addChannel("accel_x", noisy);
    const int g2 = both.addChannel("gyro_z", noisy);

    std::vector<double> gyro_alone, gyro_with_accel, accel;
    for (int i = 0; i < 500; ++i) {
        gyro_alone.push_back(gyro_only.apply(g1, 0.0));
        accel.push_back(both.apply(a2, 0.0));
        gyro_with_accel.push_back(both.apply(g2, 0.0));
    }
    for (std::size_t i = 0; i < gyro_alone.size(); ++i) {
        EXPECT_EQ(gyro_alone[i], gyro_with_accel[i])
            << "enabling another channel moved the gyro's stream at i=" << i;
    }
    EXPECT_NE(accel, gyro_with_accel) << "two channels share one stream";
}

TEST(SensorModelSeed, RegistrationOrderDoesNotChangeAnyStream) {
    // Same reason: a channel added later must not shift an existing one.
    ChannelConfig noisy;
    noisy.noise_std = 0.02;

    SensorModel a(kDt, 8);
    const int ax = a.addChannel("x", noisy);
    a.addChannel("y", noisy);

    SensorModel b(kDt, 8);
    b.addChannel("y", noisy);
    const int bx = b.addChannel("x", noisy);

    for (int i = 0; i < 200; ++i) {
        EXPECT_EQ(a.apply(ax, 0.0), b.apply(bx, 0.0)) << "at i=" << i;
    }
}

TEST(SensorModelReset, ResetReplaysTheSameStream) {
    ChannelConfig cfg;
    cfg.noise_std     = 0.01;
    cfg.bias          = 0.5;
    cfg.bias_walk_std = 0.002;
    cfg.delay_s       = 0.006;

    SensorModel m(kDt, 99);
    const int h = m.addChannel("gyro", cfg);

    std::vector<double> first;
    for (int i = 0; i < 300; ++i) first.push_back(m.apply(h, 1.0));

    // reset() restores the channel's state but deliberately does not reseed,
    // so the second pass is a different draw -- what it must restore is the
    // bias, the hold and the delay ring.
    m.reset();
    EXPECT_DOUBLE_EQ(m.channel(h).bias(), cfg.bias);

    std::vector<double> second;
    for (int i = 0; i < 300; ++i) second.push_back(m.apply(h, 1.0));
    EXPECT_NE(first, second);
    // Both passes should sit around truth+bias, i.e. the reset did not leave a
    // stale walk behind.
    EXPECT_NEAR(sampleMean(second), 1.0 + cfg.bias, 0.05);
}

// ---------------------------------------------------------------------------
// Configuration errors are refused loudly, at startup, not silently absorbed.
// ---------------------------------------------------------------------------

TEST(SensorModelValidation, RejectsBadConfiguration) {
    SensorModel m(kDt, 1);
    m.addChannel("x", ChannelConfig{});
    EXPECT_THROW(m.addChannel("x", ChannelConfig{}), std::invalid_argument);

    ChannelConfig neg;
    neg.noise_std = -1.0;
    EXPECT_THROW(m.addChannel("neg_sigma", neg), std::invalid_argument);

    ChannelConfig neg_delay;
    neg_delay.delay_s = -0.01;
    EXPECT_THROW(m.addChannel("neg_delay", neg_delay), std::invalid_argument);

    ChannelConfig bad_p;
    bad_p.dropout_prob = 1.5;
    EXPECT_THROW(m.addChannel("bad_p", bad_p), std::invalid_argument);
}

TEST(SensorModelValidation, UnknownChannelHasNoHandle) {
    SensorModel m(kDt, 1);
    m.addChannel("gyro_z", ChannelConfig{});
    EXPECT_EQ(m.handle("gyro_z"), 0);
    EXPECT_EQ(m.handle("nope"), -1);
}

TEST(SensorModelReport, NamesOnlyTheImperfectChannels) {
    ChannelConfig noisy;
    noisy.noise_std = 0.01;
    SensorModel m(kDt, 1);
    m.addChannel("clean", ChannelConfig{});
    m.addChannel("dirty", noisy);

    const std::string r = m.report();
    EXPECT_FALSE(m.perfect());
    EXPECT_NE(r.find("dirty"), std::string::npos);
    EXPECT_EQ(r.find("clean"), std::string::npos);
}

TEST(SensorModelVector, AppliesConsecutiveHandlesInOrder) {
    ChannelConfig bias_x, bias_y, bias_z;
    bias_x.bias = 1.0;
    bias_y.bias = 2.0;
    bias_z.bias = 3.0;

    SensorModel m(kDt, 1);
    const int h = m.addChannel("accel_x", bias_x);
    m.addChannel("accel_y", bias_y);
    m.addChannel("accel_z", bias_z);

    const double truth[3] = {10.0, 20.0, 30.0};
    double out[3] = {0, 0, 0};
    m.applyVector(h, truth, out, 3);
    EXPECT_DOUBLE_EQ(out[0], 11.0);
    EXPECT_DOUBLE_EQ(out[1], 22.0);
    EXPECT_DOUBLE_EQ(out[2], 33.0);

    // In-place is allowed and must give the same answer.
    double io[3] = {10.0, 20.0, 30.0};
    SensorModel m2(kDt, 1);
    const int h2 = m2.addChannel("accel_x", bias_x);
    m2.addChannel("accel_y", bias_y);
    m2.addChannel("accel_z", bias_z);
    m2.applyVector(h2, io, io, 3);
    EXPECT_DOUBLE_EQ(io[0], 11.0);
    EXPECT_DOUBLE_EQ(io[1], 22.0);
    EXPECT_DOUBLE_EQ(io[2], 33.0);
}
