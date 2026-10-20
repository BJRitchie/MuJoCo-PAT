/*! Tests for the benchmark scorers in contact_scenario.h.
 *
 *  These are the functions that turn a recorded run into the numbers quoted in
 *  the README, so a mistake in them does not crash anything -- it produces a
 *  confident wrong figure. Two of the choices they encode were arrived at by
 *  being wrong first, and both are pinned here:
 *
 *    - which WINDOW a run is scored over. Averaging a mostly-quiet run measures
 *      the estimator's error floor rather than the event, which is how a weak
 *      estimate can be made to look excellent.
 *    - which RATIO to quote. The signed-peak ratio is right for a transient and
 *      wrong for a sustained oscillation, where it can report a large negative
 *      number for an estimate that is tracking almost perfectly.
 *
 *  No MuJoCo and no scenario here: these take a vector of rows and return
 *  statistics, so they are tested directly on constructed rows.
 */
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "contact_scenario.h"

using pat_arm_control::bench::ContactRow;
using pat_arm_control::bench::JointScore;
using pat_arm_control::bench::scoreJoint;
using pat_arm_control::bench::scoreJointResidual;

namespace {

//! One row carrying a single joint's truth, residual and estimate.
ContactRow row(double t, int ncon, double truth, double resid, double raw) {
    ContactRow r;
    r.t = t;
    r.ncon = ncon;
    r.truth = {truth};
    r.resid = {resid};
    r.raw = {raw};
    r.reduced = {raw};
    return r;
}

}  // namespace

// ===================== the contact window is the event ==================== //

TEST(ScoreJoint, QuietTicksAreExcludedFromTheContactWindow) {
    // A long quiet run with a brief, perfectly estimated contact. Including the
    // quiet ticks would drag both means towards zero and make the contact look
    // like a small fraction of the record.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 90; ++i) rows.push_back(row(0.01 * i, 0, 0.0, 0.0, 0.0));
    for (int i = 90; i < 100; ++i) rows.push_back(row(0.01 * i, 2, 5.0, 5.0, 5.0));

    const JointScore s = scoreJoint(rows, 0);
    EXPECT_EQ(s.n_contact, 10) << "only the ncon > 0 ticks are the contact";
    EXPECT_DOUBLE_EQ(s.truth_mean, 5.0);
    EXPECT_DOUBLE_EQ(s.est_mean, 5.0);
    EXPECT_DOUBLE_EQ(s.rms_err, 0.0);
}

TEST(ScoreJoint, NoContactLeavesEveryStatisticAtZero) {
    std::vector<ContactRow> rows;
    for (int i = 0; i < 50; ++i) rows.push_back(row(0.01 * i, 0, 0.0, 0.0, 1.0));

    const JointScore s = scoreJoint(rows, 0);
    EXPECT_EQ(s.n_contact, 0);
    EXPECT_DOUBLE_EQ(s.truth_peak, 0.0);
    EXPECT_DOUBLE_EQ(s.est_peak, 0.0);
}

TEST(ScoreJoint, PeakKeepsTheSignOfTheLargestExcursion) {
    // peak is by magnitude but reported signed, so a compressive contact reads
    // negative rather than silently becoming positive.
    std::vector<ContactRow> rows{row(0.0, 1, 2.0, 2.0, 2.0),
                                 row(0.01, 1, -7.0, -7.0, -6.0),
                                 row(0.02, 1, 3.0, 3.0, 3.0)};
    const JointScore s = scoreJoint(rows, 0);
    EXPECT_DOUBLE_EQ(s.truth_peak, -7.0);
    EXPECT_DOUBLE_EQ(s.est_peak, -6.0);
    EXPECT_NEAR(s.ratio(), 6.0 / 7.0, 1e-12);
}

// ================ the residual window starts where told ================== //

TEST(ScoreJointResidual, StartIndexExcludesTheStartupTransient) {
    // The first ten ticks are the filter arriving from zero; the rest is a
    // perfectly tracked signal. Scored from 0 the transient is error; scored
    // from 10 it is not.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 10; ++i) rows.push_back(row(0.01 * i, 0, 1.0, 1.0, 0.0));
    for (int i = 10; i < 50; ++i) rows.push_back(row(0.01 * i, 0, 1.0, 1.0, 1.0));

    const JointScore all = scoreJointResidual(rows, 0, 0);
    const JointScore settled = scoreJointResidual(rows, 0, 10);
    EXPECT_EQ(all.n_contact, 50);
    EXPECT_EQ(settled.n_contact, 40);
    EXPECT_GT(all.rms_err, 0.4) << "the transient counts as error when included";
    EXPECT_DOUBLE_EQ(settled.rms_err, 0.0);
}

TEST(ScoreJointResidual, IgnoresNconBecauseModelErrorTouchesNothing) {
    // The residual is scored on a run where nothing is in contact at all, so a
    // scorer that filtered on ncon the way scoreJoint does would return an
    // empty window and a silent zero.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 20; ++i) rows.push_back(row(0.01 * i, 0, 0.0, 2.0, 2.0));

    const JointScore s = scoreJointResidual(rows, 0, 0);
    EXPECT_EQ(s.n_contact, 20);
    EXPECT_DOUBLE_EQ(s.truth_rms, 2.0);
}

// =================== amplitude ratio against peak ratio =================== //

TEST(AmplitudeRatio, SurvivesAnOscillationWhosePeaksLandOnOppositeHalfCycles) {
    // Truth and estimate are the same sine to within 2%, but sampled so that
    // the truth's largest-magnitude sample is positive and the estimate's is
    // negative. ratio() then reports a large negative number for an estimate
    // that is tracking; amplitudeRatio() reports the 0.98 that is true.
    std::vector<ContactRow> rows;
    const double seq[6] = {0.9, 1.0, 0.2, -0.95, -0.8, 0.1};
    for (int i = 0; i < 6; ++i) {
        // The estimate leans slightly on the negative half, enough to move
        // which sample is the largest by magnitude.
        const double est = seq[i] * (seq[i] < 0.0 ? 1.06 : 0.90);
        rows.push_back(row(0.01 * i, 0, 0.0, seq[i], est));
    }
    const JointScore s = scoreJointResidual(rows, 0, 0);

    EXPECT_GT(s.truth_peak, 0.0);
    EXPECT_LT(s.est_peak, 0.0);
    EXPECT_LT(s.ratio(), 0.0) << "the signed-peak ratio inverts here";
    EXPECT_NEAR(s.amplitudeRatio(), 0.98, 0.03)
        << "the amplitude ratio reports what the estimate actually did";
    EXPECT_GT(s.corr, 0.99);
}

TEST(AmplitudeRatio, IsRmsAboutZeroNotAboutTheMean) {
    // A constant disturbance has a magnitude, and the amplitude ratio has to
    // see it; an RMS about the mean would be zero for both series and report
    // nothing at all.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 10; ++i) rows.push_back(row(0.01 * i, 0, 0.0, 4.0, 2.0));

    const JointScore s = scoreJointResidual(rows, 0, 0);
    EXPECT_DOUBLE_EQ(s.truth_rms, 4.0);
    EXPECT_DOUBLE_EQ(s.est_rms, 2.0);
    EXPECT_DOUBLE_EQ(s.amplitudeRatio(), 0.5);
}

TEST(SignalFloor, DustTruthGivesNotANumberRatherThanAHugeRatio) {
    // The baseline run -- correct model, nothing touching -- leaves a truth of
    // order 1e-14. A bare "!= 0" guard divided by it and printed ratios of
    // 5e13, which reads as catastrophic failure instead of as no signal.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 10; ++i) {
        rows.push_back(row(0.01 * i, 0, 1e-15, 1e-14, 2e-13));
    }
    const JointScore s = scoreJointResidual(rows, 0, 0);
    EXPECT_TRUE(std::isnan(s.amplitudeRatio()));
    EXPECT_TRUE(std::isnan(s.ratio()));
}

TEST(SignalFloor, DoesNotSuppressASignalWorthEstimating) {
    // The floor is 1e-6, far below the observer's own measured error floor of
    // 8e-5, so a real disturbance is never reported as absent.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 10; ++i) rows.push_back(row(0.01 * i, 1, 1e-4, 1e-4, 1e-4));

    const JointScore contact = scoreJoint(rows, 0);
    const JointScore resid = scoreJointResidual(rows, 0, 0);
    EXPECT_NEAR(contact.ratio(), 1.0, 1e-12);
    EXPECT_NEAR(resid.amplitudeRatio(), 1.0, 1e-12);
}

// ======================= correlation and rms error ======================== //

TEST(Statistics, CorrelationIsScaleInvariantButRmsErrorIsNot) {
    // An estimate that is exactly half the truth tracks it perfectly in shape.
    // Correlation says so; rms error says the magnitude is wrong. Both columns
    // are reported because neither alone distinguishes a lagging estimate from
    // an under-reading one.
    std::vector<ContactRow> rows;
    const double seq[5] = {1.0, 3.0, -2.0, 5.0, -4.0};
    for (int i = 0; i < 5; ++i) {
        rows.push_back(row(0.01 * i, 1, seq[i], seq[i], 0.5 * seq[i]));
    }
    const JointScore s = scoreJoint(rows, 0);
    EXPECT_NEAR(s.corr, 1.0, 1e-12);
    EXPECT_GT(s.rms_err, 1.0);
    EXPECT_NEAR(s.ratio(), 0.5, 1e-12);
}

TEST(Statistics, ZeroVarianceTruthLeavesCorrelationUndefined) {
    // A constant truth has no variation to correlate against, so the column has
    // to be undefined rather than an arbitrary 0 or 1.
    std::vector<ContactRow> rows;
    for (int i = 0; i < 5; ++i) rows.push_back(row(0.01 * i, 1, 2.0, 2.0, 2.0));

    const JointScore s = scoreJoint(rows, 0);
    EXPECT_TRUE(std::isnan(s.corr));
    EXPECT_DOUBLE_EQ(s.rms_err, 0.0);
}

TEST(Statistics, OutOfRangeJointIsSkippedRatherThanReadingPastTheRow) {
    std::vector<ContactRow> rows{row(0.0, 1, 1.0, 1.0, 1.0)};
    const JointScore s = scoreJoint(rows, 7);
    EXPECT_EQ(s.n_contact, 0);
}
