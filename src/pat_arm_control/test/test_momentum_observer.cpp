#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <Eigen/Dense>

#include "pat_arm_control/observers/momentum_observer.h"

using pat_arm_control::MomentumObserver;
using pat_arm_control::MomentumObserverParams;

namespace {

constexpr int kNv = 9;        // 6 base DOFs + a planar 3R arm
constexpr int kNJoints = 3;
constexpr double kDt = 0.01;  // 100 Hz, the control rate

MomentumObserverParams baseParams(double gain = 20.0) {
    MomentumObserverParams p;
    p.nv = kNv;
    p.n_joints = kNJoints;
    p.dt = kDt;
    p.gain = Eigen::VectorXd::Constant(1, gain);
    return p;
}

/*! A synthetic floating-base plant obeying M qddot + C qdot = tau + d exactly.
 *
 *  The point of simulating rather than replaying a trajectory is that `d` is
 *  known by construction, so "the observer recovered it" is checkable. The mass
 *  matrix is made configuration dependent -- otherwise Mdot is identically zero
 *  and the term the estimator is most likely to get wrong never gets exercised.
 */
class Plant {
public:
    Plant() {
        q_.setZero(kNv);
        v_.setZero(kNv);
        d_.setZero(kNv);
    }

    /*! Symmetric positive definite, and a function of configuration so that
     *  Mdot is non-trivial. */
    Eigen::MatrixXd M() const {
        Eigen::MatrixXd m = Eigen::MatrixXd::Identity(kNv, kNv);
        for (int i = 0; i < kNv; ++i) m(i, i) = 2.0 + 0.5 * std::cos(q_(i));
        // Off-diagonal coupling that varies with configuration, kept small
        // enough to stay well conditioned.
        for (int i = 0; i + 1 < kNv; ++i) {
            const double c = 0.3 * std::sin(q_(i) + q_(i + 1));
            m(i, i + 1) = c;
            m(i + 1, i) = c;
        }
        return m;
    }

    /*! C qdot, which is what MuJoCo's qfrc_bias gives with gravity disabled.
     *
     *  Derived from M rather than invented, so the identity Mdot = C + C^T that
     *  the observer relies on actually holds for this plant. With
     *  C_ij = sum_k Gamma_ijk qdot_k built from the Christoffel symbols of the
     *  first kind, Mdot = C + C^T is satisfied exactly. */
    Eigen::VectorXd biasForce() const {
        const Eigen::MatrixXd dMdq[kNv] = {dM(0), dM(1), dM(2), dM(3), dM(4),
                                           dM(5), dM(6), dM(7), dM(8)};
        Eigen::VectorXd Cv = Eigen::VectorXd::Zero(kNv);
        for (int i = 0; i < kNv; ++i) {
            for (int j = 0; j < kNv; ++j) {
                for (int k = 0; k < kNv; ++k) {
                    const double gamma = 0.5 * (dMdq[k](i, j) + dMdq[j](i, k) -
                                                dMdq[i](j, k));
                    Cv(i) += gamma * v_(j) * v_(k);
                }
            }
        }
        return Cv;
    }

    /*! Advance one step under tau and the current disturbance. Semi-implicit
     *  Euler, matching how the observer's divided difference is taken. */
    void step(const Eigen::VectorXd& tau, double dt) {
        const Eigen::VectorXd qddot = M().ldlt().solve(tau + d_ - biasForce());
        v_ += dt * qddot;
        q_ += dt * v_;
    }

    void setDisturbance(const Eigen::VectorXd& d) { d_ = d; }
    const Eigen::VectorXd& q() const { return q_; }
    const Eigen::VectorXd& v() const { return v_; }
    const Eigen::VectorXd& d() const { return d_; }
    void setV(const Eigen::VectorXd& v) { v_ = v; }
    void setQ(const Eigen::VectorXd& q) { q_ = q; }

private:
    /*! dM/dq_k, analytic, matching M() above. */
    Eigen::MatrixXd dM(int k) const {
        Eigen::MatrixXd g = Eigen::MatrixXd::Zero(kNv, kNv);
        g(k, k) = -0.5 * std::sin(q_(k));
        for (int i = 0; i + 1 < kNv; ++i) {
            if (i == k || i + 1 == k) {
                const double c = 0.3 * std::cos(q_(i) + q_(i + 1));
                g(i, i + 1) += c;
                g(i + 1, i) += c;
            }
        }
        return g;
    }

    Eigen::VectorXd q_, v_, d_;
};

/*! Run the plant and the observer together for `steps`, returning the final
 *  estimate. tau is zero in the base rows throughout, as an unactuated
 *  floating base requires. */
Eigen::VectorXd converge(MomentumObserver& obs, Plant& plant,
                         const Eigen::VectorXd& tau, int steps) {
    for (int i = 0; i < steps; ++i) {
        obs.update(plant.M(), plant.biasForce(), plant.v(), tau);
        plant.step(tau, kDt);
    }
    return obs.estimate();
}

} // namespace

// ---------------------------------------------------------------------------
// The plant itself has to satisfy the identity the observer rests on, or the
// tests below prove nothing about the observer.
// ---------------------------------------------------------------------------

TEST(MomentumObserverPlant, SatisfiesTheMdotIdentity) {
    // Mdot = C + C^T is the identity the whole derivation turns on. Check the
    // synthetic plant honours it: a divided difference of M against C qdot
    // built from the Christoffel symbols.
    Plant p;
    Eigen::VectorXd q0(kNv), v0(kNv);
    for (int i = 0; i < kNv; ++i) {
        q0(i) = 0.2 * (i + 1);
        v0(i) = 0.1 * ((i % 3) - 1) + 0.05 * (i + 1);
    }
    p.setQ(q0);
    p.setV(v0);

    const Eigen::MatrixXd M0 = p.M();
    const Eigen::VectorXd Cv = p.biasForce();

    // Advance configuration along qdot only, so dM/dt = sum_k dM/dq_k qdot_k.
    constexpr double h = 1e-7;
    Plant p2 = p;
    p2.setQ(q0 + h * v0);
    const Eigen::MatrixXd Mdot = (p2.M() - M0) / h;

    // Mdot qdot should equal (C + C^T) qdot = C qdot + C^T qdot, so
    // Mdot qdot - C qdot is C^T qdot. The identity's testable consequence is
    // that qdot^T Mdot qdot = 2 qdot^T C qdot.
    const double lhs = v0.dot(Mdot * v0);
    const double rhs = 2.0 * v0.dot(Cv);
    EXPECT_NEAR(lhs, rhs, 1e-4 * std::max(1.0, std::fabs(rhs)));
}

// ---------------------------------------------------------------------------
// The headline behaviour.
// ---------------------------------------------------------------------------

TEST(MomentumObserver, ZeroDisturbanceGivesZeroEstimate) {
    // The approximation most likely to leak a false disturbance is the Mdot
    // divided difference, so this runs with the arm genuinely moving and the
    // mass matrix genuinely changing, not from rest.
    MomentumObserver obs(baseParams(25.0));
    Plant plant;
    Eigen::VectorXd v0 = Eigen::VectorXd::Zero(kNv);
    v0(6) = 0.4;
    v0(7) = -0.3;
    v0(8) = 0.2;
    plant.setV(v0);

    Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);
    tau(6) = 0.5;
    tau(7) = -0.2;

    const Eigen::VectorXd r = converge(obs, plant, tau, 400);
    // Only the discretisation error of a divided difference should survive.
    EXPECT_LT(r.cwiseAbs().maxCoeff(), 2e-2)
        << "estimate with no disturbance applied:\n" << r.transpose();
}

TEST(MomentumObserver, RecoversAConstantDisturbance) {
    MomentumObserver obs(baseParams(25.0));
    Plant plant;

    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 0.5;     // a base force, what drag or contact reaction produces
    d(5) = -0.08;   // a base yaw torque
    d(7) = 0.25;    // and an unmodelled joint torque
    plant.setDisturbance(d);

    const Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);
    const Eigen::VectorXd r = converge(obs, plant, tau, 600);

    for (int i = 0; i < kNv; ++i) {
        EXPECT_NEAR(r(i), d(i), 0.02 + 0.05 * std::fabs(d(i)))
            << "DOF " << i << " estimated " << r(i) << " vs true " << d(i);
    }
}

TEST(MomentumObserver, TimeConstantMatchesOneOverGain) {
    // The claim rdot = K(d - r) makes: a step disturbance is tracked as a
    // first-order lag, reaching 1 - 1/e of its value after exactly 1/K
    // seconds. This is the property that makes the gain tunable by reasoning
    // rather than by trial.
    for (double K : {10.0, 40.0}) {
        MomentumObserver obs(baseParams(K));
        Plant plant;
        Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
        d(0) = 1.0;
        plant.setDisturbance(d);

        const Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);
        const int steps_to_tau = static_cast<int>(std::lround((1.0 / K) / kDt));
        ASSERT_GT(steps_to_tau, 2) << "pick a gain resolvable at this dt";

        const Eigen::VectorXd r = converge(obs, plant, tau, steps_to_tau);
        constexpr double k1me = 0.6321205588;    // 1 - 1/e
        EXPECT_NEAR(r(0), k1me * d(0), 0.08)
            << "K=" << K << " reached " << r(0) << " after one time constant";
    }
}

TEST(MomentumObserver, TracksAStepOnAndBackOff) {
    // Contact is transient: it has to come back to zero as well as rise.
    MomentumObserver obs(baseParams(30.0));
    Plant plant;
    const Eigen::VectorXd tau = Eigen::VectorXd::Zero(kNv);

    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(1) = -0.4;
    plant.setDisturbance(d);
    converge(obs, plant, tau, 400);
    EXPECT_NEAR(obs.estimate()(1), d(1), 0.03);

    plant.setDisturbance(Eigen::VectorXd::Zero(kNv));
    converge(obs, plant, tau, 400);
    EXPECT_NEAR(obs.estimate()(1), 0.0, 0.03)
        << "the estimate did not decay after the disturbance stopped";
}

TEST(MomentumObserver, FirstUpdateReportsExactlyZero) {
    // The integrator is seeded with the momentum rather than with zero. Seeded
    // with zero, the first residual would be K times the vehicle's own
    // momentum -- a large disturbance reported the instant the observer starts.
    MomentumObserver obs(baseParams(20.0));
    Plant plant;
    Eigen::VectorXd v0 = Eigen::VectorXd::Constant(kNv, 0.7);
    plant.setV(v0);

    obs.update(plant.M(), plant.biasForce(), plant.v(),
               Eigen::VectorXd::Zero(kNv));
    EXPECT_EQ(obs.estimate().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_FALSE(obs.ready()) << "one update cannot yet support a difference";

    plant.step(Eigen::VectorXd::Zero(kNv), kDt);
    obs.update(plant.M(), plant.biasForce(), plant.v(),
               Eigen::VectorXd::Zero(kNv));
    EXPECT_TRUE(obs.ready());
}

TEST(MomentumObserver, AppliedTorqueIsWhatItEstimatesAgainst) {
    // Feeding the planned torque instead of the applied one makes the clamped
    // difference look like a disturbance. Demonstrated here because it is the
    // easiest mistake to make when wiring this into a controller that
    // saturates per joint before publishing.
    MomentumObserver obs(baseParams(25.0));
    Plant plant;
    const Eigen::VectorXd applied = [] {
        Eigen::VectorXd t = Eigen::VectorXd::Zero(kNv);
        t(6) = 1.0;
        return t;
    }();
    Eigen::VectorXd planned = applied;
    planned(6) = 2.5;        // what the solver wanted before clamping

    // The plant is driven by the applied torque either way; only what the
    // observer is told differs.
    for (int i = 0; i < 500; ++i) {
        obs.update(plant.M(), plant.biasForce(), plant.v(), planned);
        plant.step(applied, kDt);
    }
    // It reports the shortfall as a disturbance on that joint, with the sign
    // of the unrealised torque.
    EXPECT_LT(obs.estimate()(6), -0.5)
        << "the 1.5 N*m that was clamped away should appear as -1.5-ish";
}

// ---------------------------------------------------------------------------
// The base-elimination reduction.
// ---------------------------------------------------------------------------

TEST(MomentumObserverReduce, AppliesTheSchurReduction) {
    MomentumObserver obs(baseParams(20.0));
    Plant plant;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 0.6;
    d(7) = 0.3;
    plant.setDisturbance(d);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 600);

    Eigen::MatrixXd H_bb_inv_H_bm(6, kNJoints);
    H_bb_inv_H_bm.setZero();
    H_bb_inv_H_bm(0, 0) = 0.5;      // couples base x to joint 0
    H_bb_inv_H_bm(5, 2) = -0.25;    // and base yaw to joint 2

    Eigen::VectorXd out(kNJoints);
    obs.reduceToJoints(H_bb_inv_H_bm, out);

    const Eigen::VectorXd r = obs.estimate();
    const Eigen::VectorXd expect =
        r.tail(kNJoints) - H_bb_inv_H_bm.transpose() * r.head(6);
    for (int i = 0; i < kNJoints; ++i) EXPECT_DOUBLE_EQ(out(i), expect(i));

    // The base rows genuinely contribute: a reduction that ignored them would
    // just be the tail.
    EXPECT_NE(out(0), r(6));
}

TEST(MomentumObserverReduce, RejectsAMisshapedReduction) {
    MomentumObserver obs(baseParams());
    Eigen::VectorXd out(kNJoints);
    EXPECT_THROW(obs.reduceToJoints(Eigen::MatrixXd::Zero(5, kNJoints), out),
                 std::invalid_argument);
    EXPECT_THROW(obs.reduceToJoints(Eigen::MatrixXd::Zero(6, kNJoints + 1), out),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Guards, clamps and lifecycle.
// ---------------------------------------------------------------------------

TEST(MomentumObserver, ResetClearsEverything) {
    MomentumObserver obs(baseParams(25.0));
    Plant plant;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 0.5;
    plant.setDisturbance(d);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 300);
    ASSERT_GT(obs.estimate().cwiseAbs().maxCoeff(), 0.1);

    obs.reset();
    EXPECT_EQ(obs.estimate().cwiseAbs().maxCoeff(), 0.0);
    EXPECT_FALSE(obs.ready());
    EXPECT_EQ(obs.updates(), 0);
    EXPECT_EQ(obs.maxMdot(), 0.0);
}

TEST(MomentumObserver, EstimateClampBoundsTheOutput) {
    MomentumObserverParams p = baseParams(25.0);
    p.estimate_clamp = 0.1;
    MomentumObserver obs(p);

    Plant plant;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 2.0;        // far beyond the clamp
    plant.setDisturbance(d);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 400);

    EXPECT_LE(obs.estimate().cwiseAbs().maxCoeff(), p.estimate_clamp + 1e-12);
    EXPECT_GT(obs.estimateClamped(), 0);
}

TEST(MomentumObserver, ClampingDoesNotWindTheIntegratorUp) {
    // A clamped estimator that keeps integrating while its output is pinned
    // takes as long to recover as it spent saturated. The honest test is
    // against the same run WITHOUT the clamp, not against zero: this plant has
    // no damping, so four seconds of a 2 N disturbance leaves the base coasting
    // at several m/s, and at that speed the Mdot divided difference carries a
    // real discretisation error of its own. Comparing the two isolates the
    // clamp from that error instead of blaming the clamp for it.
    auto run = [](double clamp) {
        MomentumObserverParams p = baseParams(25.0);
        p.estimate_clamp = clamp;
        MomentumObserver obs(p);
        Plant plant;
        Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
        d(0) = 2.0;
        plant.setDisturbance(d);
        converge(obs, plant, Eigen::VectorXd::Zero(kNv), 400);
        plant.setDisturbance(Eigen::VectorXd::Zero(kNv));
        converge(obs, plant, Eigen::VectorXd::Zero(kNv), 100);
        return obs.estimate()(0);
    };

    const double unclamped = run(0.0);
    const double clamped   = run(0.1);

    // Whatever the unclamped estimator settles to after the disturbance stops
    // is the discretisation floor; the clamped one must land in the same place
    // rather than carrying a wound-up integral.
    EXPECT_NEAR(clamped, unclamped, 0.02)
        << "clamped " << clamped << " vs unclamped " << unclamped;
}

TEST(MomentumObserverPlant, TimeDifferenceMdotAgainstConfigurationSpaceFd) {
    // Caveat 1 of the design, measured rather than assumed. The observer gets
    // Mdot by differencing consecutive mass matrices, which is free because M
    // is computed every period anyway. The alternative -- one extra mj_fullM at
    // q + h qdot, about 4 us -- can use an arbitrarily small h and is therefore
    // the more accurate reference. This measures the gap, so the decision to
    // keep the free version rests on a number.
    Plant plant;
    Eigen::VectorXd q0(kNv), v0(kNv);
    for (int i = 0; i < kNv; ++i) {
        q0(i) = 0.15 * (i + 1);
        v0(i) = 0.3 * std::sin(0.7 * (i + 1));
    }
    plant.setQ(q0);
    plant.setV(v0);

    const Eigen::MatrixXd M0 = plant.M();

    // The reference: configuration-space forward difference with a tiny step.
    Plant ref = plant;
    constexpr double h_fd = 1e-8;
    ref.setQ(q0 + h_fd * v0);
    const Eigen::MatrixXd Mdot_fd = (ref.M() - M0) / h_fd;

    // What the observer actually uses: one control period of plant motion.
    Plant adv = plant;
    adv.step(Eigen::VectorXd::Zero(kNv), kDt);
    const Eigen::MatrixXd Mdot_td = (adv.M() - M0) / kDt;

    const double scale = std::max(1e-9, Mdot_fd.cwiseAbs().maxCoeff());
    const double abs_err = (Mdot_td - Mdot_fd).cwiseAbs().maxCoeff();
    std::printf("[observer] Mdot peak %.4f, time-difference error %.4f (%.1f%%) "
                "at dt=%g, |v|=%.2f\n",
                scale, abs_err, 100.0 * abs_err / scale, kDt, v0.norm());

    // A one-sided difference over a whole control period is first order in dt,
    // so a few percent at these speeds is expected. The bound is what matters:
    // if it ever exceeds this, the configuration-space fallback in the header's
    // caveat is the mitigation, at about 4 us a step.
    EXPECT_LT(abs_err / scale, 0.10)
        << "time-difference Mdot is now more than 10% off the reference";
}

TEST(MomentumObserver, HighVelocitySetsTheDiscretisationFloor) {
    // Bounding caveat 1 of the design: Mdot is a divided difference, so its
    // error grows with how fast the configuration is moving. Measured here so
    // the floor is a number rather than an assumption, and so that a change
    // which makes it worse is visible.
    auto floor_at = [](double v0) {
        MomentumObserver obs(baseParams(25.0));
        Plant plant;
        plant.setV(Eigen::VectorXd::Constant(kNv, v0));
        // No disturbance at all: anything the observer reports is error.
        converge(obs, plant, Eigen::VectorXd::Zero(kNv), 300);
        return obs.estimate().cwiseAbs().maxCoeff();
    };

    const double slow = floor_at(0.2);
    const double fast = floor_at(3.0);
    std::printf("[observer] zero-disturbance error floor: %.5f at 0.2 rad/s, "
                "%.5f at 3.0 rad/s\n", slow, fast);

    // Slow motion is where the controller actually operates, and there the
    // divided difference is good to a few mN.
    EXPECT_LT(slow, 0.01) << "error at 0.2 rad/s: " << slow;
    // It degrades with speed, which is the caveat; recorded, not asserted
    // tightly, because the exact value is plant-specific.
    EXPECT_GT(fast, slow);
    EXPECT_LT(fast, 1.0) << "error at 3.0 rad/s: " << fast;
}

TEST(MomentumObserver, MdotClampIsCountedAndBounded) {
    MomentumObserverParams p = baseParams(20.0);
    p.mdot_clamp = 0.01;        // deliberately tiny, so it engages
    MomentumObserver obs(p);

    Plant plant;
    Eigen::VectorXd v0 = Eigen::VectorXd::Constant(kNv, 1.5);
    plant.setV(v0);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 200);

    EXPECT_GT(obs.mdotClamped(), 0);
    EXPECT_GT(obs.maxMdot(), p.mdot_clamp);
    // Still finite and bounded, which is the point of the clamp.
    EXPECT_TRUE(obs.estimate().allFinite());
}

TEST(MomentumObserver, GainOfZeroFreezesTheEstimateAtZero) {
    // A legitimate configuration: it disables the observer on that DOF.
    MomentumObserver obs(baseParams(0.0));
    Plant plant;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 1.0;
    plant.setDisturbance(d);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 200);
    EXPECT_EQ(obs.estimate().cwiseAbs().maxCoeff(), 0.0);
}

TEST(MomentumObserver, PerDofGainsAreHonoured) {
    MomentumObserverParams p = baseParams();
    p.gain = Eigen::VectorXd::Zero(kNv);
    p.gain(0) = 30.0;        // base x enabled
    MomentumObserver obs(p);

    Plant plant;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(kNv);
    d(0) = 0.5;
    d(1) = 0.5;
    plant.setDisturbance(d);
    converge(obs, plant, Eigen::VectorXd::Zero(kNv), 500);

    EXPECT_NEAR(obs.estimate()(0), 0.5, 0.03);
    EXPECT_EQ(obs.estimate()(1), 0.0) << "a zero-gain DOF must stay silent";
}

TEST(MomentumObserver, RejectsBadConfiguration) {
    MomentumObserverParams p = baseParams();

    // The extra parens matter: MomentumObserver(q) alone parses as a
    // declaration of a variable named q, not a constructor call.
    auto bad = [](const MomentumObserverParams& q) {
        EXPECT_THROW((MomentumObserver(q)), std::invalid_argument);
    };

    { auto q = p; q.nv = 6;           bad(q); }   // no arm joints
    { auto q = p; q.n_joints = 4;     bad(q); }   // inconsistent with nv
    { auto q = p; q.dt = 0.0;         bad(q); }
    { auto q = p; q.dt = -0.01;       bad(q); }
    { auto q = p; q.mdot_clamp = -1;  bad(q); }
    { auto q = p; q.gain = Eigen::VectorXd::Constant(2, 1.0); bad(q); }
    { auto q = p; q.gain = Eigen::VectorXd::Constant(1, -1.0); bad(q); }
}

TEST(MomentumObserver, RejectsMisSizedUpdate) {
    MomentumObserver obs(baseParams());
    const Eigen::MatrixXd M = Eigen::MatrixXd::Identity(kNv, kNv);
    const Eigen::VectorXd ok = Eigen::VectorXd::Zero(kNv);
    const Eigen::VectorXd bad = Eigen::VectorXd::Zero(kNv - 1);

    EXPECT_THROW(obs.update(Eigen::MatrixXd::Identity(3, 3), ok, ok, ok),
                 std::invalid_argument);
    EXPECT_THROW(obs.update(M, bad, ok, ok), std::invalid_argument);
    EXPECT_THROW(obs.update(M, ok, bad, ok), std::invalid_argument);
    EXPECT_THROW(obs.update(M, ok, ok, bad), std::invalid_argument);
}

TEST(MomentumObserver, SingleGainIsBroadcastToEveryDof) {
    MomentumObserver obs(baseParams(17.0));
    ASSERT_EQ(obs.params().gain.size(), kNv);
    for (int i = 0; i < kNv; ++i) EXPECT_DOUBLE_EQ(obs.params().gain(i), 17.0);
}
