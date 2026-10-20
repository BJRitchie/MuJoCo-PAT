#pragma once
/*! Shared fixtures for the NMPC tests: a probe subclass that re-exports the
 *  protected seams, the parameter sets the tests run on, a reference state, and
 *  a closed-loop harness.
 *
 *  These drive the controller directly, through a subclass that re-exports the
 *  protected seams, and compare it against MuJoCo's own forward dynamics
 *  evaluated on the same state. Nothing here touches ROS.
 *
 *  PAT_NMPC_TEST_MJCF is the build-tree pat_platform_planar.xml (see
 *  CMakeLists.txt); the model has a free-floating base plus two planar 3R
 *  arms, so nq = 13, nv = 12, n_joints_ = 6.
 */
#include <map>
#include <string>

#include <Eigen/Dense>

#include "pat_arm_control/control/nmpc/arm_nmpc.h"
#include "pat_rt/testing/alloc_counter.h"

namespace pat_arm_control {
namespace test_support {

using pat_rt::testing::AllocScope;

// Re-exports the protected model-construction seams so a test can build A/B at
// a known state rather than inferring them from closed-loop behaviour.
class ModelProbe : public ArmNMPC {
public:
    ModelProbe(const std::string& mjcf, const NMPCParams& p) : ArmNMPC(mjcf, p) {}

    using ArmNMPC::NodeTaskSpace;
    using ArmNMPC::buildAugmentedModel;
    using ArmNMPC::ee_site_id_;
    using ArmNMPC::enterDamping;
    using ArmNMPC::kd_;
    using ArmNMPC::torqueLimsByIndex;
    using ArmNMPC::vlimsByIndex;
    using ArmNMPC::evalTaskSpace;
    using ArmNMPC::n_owned;
    using ArmNMPC::ownedJointInds;
    using ArmNMPC::qpsolve;
    using ArmNMPC::lastQpStatus;
    using ArmNMPC::mode;
    using ArmNMPC::params_;
    using ArmNMPC::planStageForAge;
    using ArmNMPC::qlimsByIndex;
    using ArmNMPC::qp_;
    using ArmNMPC::reachableWrenchBound;
    using ArmNMPC::resetProfile;
    // Timing members, so a test can check the profile covers the tick.
    using ArmNMPC::build_model;
    using ArmNMPC::defect;
    using ArmNMPC::finalize;
    using ArmNMPC::pinv;
    using ArmNMPC::qp_iters;
    using ArmNMPC::qp_retries;
    using ArmNMPC::qp_time_ms;
    using ArmNMPC::rollout_dynamics;
    using ArmNMPC::rollout_integrate;
    using ArmNMPC::tick;
    using ArmNMPC::trust_region;
    using ArmNMPC::buildJointBoxBounds;
    using ArmNMPC::buildTorqueGeneralConstraint;
    // The live-state path, so FullDynamics.AgreesWithTheLiveStatePath can
    // compare the public Workspace query against the one a control tick uses.
    using ArmNMPC::getDynamics;
    using ArmNMPC::loadLiveState;
    using ArmNMPC::computeInertiaProducts;
    using ArmNMPC::finalizeJointTorques;
    using FloatingBaseDynamics::BaseElim;
    using FloatingBaseDynamics::computeBaseElim;
    using FloatingBaseDynamics::computeSiteJacobian;
    using FloatingBaseDynamics::dynamicsAt;
    using FloatingBaseDynamics::getDynamics;
    using FloatingBaseDynamics::gjm6;
    using FloatingBaseDynamics::integrateStep;
    using FloatingBaseDynamics::jdotQdot6;
    using FloatingBaseDynamics::modelJointRanges;
    using FloatingBaseDynamics::n_joints_;
    using FloatingBaseDynamics::lastMjError;
    using FloatingBaseDynamics::lastMjWarning;
    using FloatingBaseDynamics::mjErrorCount;
    using FloatingBaseDynamics::mjWarningCount;
    using FloatingBaseDynamics::pinvDLS;
    using FloatingBaseDynamics::solveSpd;
};

inline constexpr const char* kMjcf = PAT_NMPC_TEST_MJCF;

/*! Left arm, matching nmpc.yaml's per-instance shape: this controller owns one
 *  arm's three joints out of the model's six. */
NMPCParams leftArmParams(double Ts = 0.01);

/*! The shape nmpc.yaml actually runs: SQP path, N=30, Ts=0.033, 100 Hz,
 *  qp.max_iter 60. Keep these in step with config/nmpc.yaml -- every harness
 *  measurement is read as representative of the deployed controller, so a drift
 *  here silently invalidates the comparison. */
NMPCParams liveParams();

struct State {
    Eigen::VectorXd q, v;
};

/*! A deliberately generic state: base at the origin with identity orientation,
 *  both arms away from any singular (straight or folded) configuration, and
 *  every joint moving so the Coriolis terms are actually exercised.
 *
 *  Every angle sits strictly inside the MJCF joint ranges (joint2 [0, 3.14],
 *  joint3 [-2.967, 0], joint5 [-1.22, 1.22]). This matters: MuJoCo enforces
 *  those ranges with constraint forces, and a state outside them accelerates
 *  violently under forces the prediction model has no term for, which would
 *  make every comparison meaningless. */
State referenceState(int n_joints);

/*! Everything a soft-vs-hard comparison needs from one closed-loop run. The
 *  headline pair is availability and max_consecutive_damping: "failed at every
 *  loop" is a max-consecutive claim, and a mean alone cannot tell 2% scattered
 *  dropouts from one long outage. */
struct RunMetrics {
    double availability = 0.0;          // fraction of ticks that produced NMPC torque
    int    max_consecutive_damping = 0;
    // CAUTION: max_pos_violation is a running max over a run that BEGINS at the
    // violation, so it is pinned to the initial condition and reads ~the starting
    // offset for any controller, including a perfect one. Use initial/final/
    // recovered to ask whether the joint actually came back.
    double max_pos_violation = 0.0;     // [rad]   worst excursion past the config limits
    double pos_violation_integral = 0.0;// [rad s] distinguishes a transient from parking outside
    double initial_pos_violation = 0.0; // [rad]   at the first tick
    double final_pos_violation = 0.0;   // [rad]   at the last tick
    double recovered = 0.0;             // [rad]   initial - final; >0 means it came back
    double max_vel_violation = 0.0;     // [rad/s]
    double mean_ee_err = 0.0;           // [m]
    double max_box_slack = 0.0;
    double guard_relax_frac = 0.0;      // fraction of ticks the guard widened anything
    double guard_relax_max = 0.0;
    double mean_tick_us = 0.0;
    double max_tick_us = 0.0;
    double frac_over_budget = 0.0;      // ticks exceeding one control period
    double qp_iter_mean = 0.0;
    double qp_iter_max = 0.0;
    long   qp_retries = 0;              // solves that needed the cold-start retry
    std::map<int, int> status_hist;     // HPIPM status -> count
};

struct Scenario {
    double setpoint_offset = 0.06;      // [m] per axis, from the measured start pose
    double q_min_override  = 0.0;       // >0 raises joint_lims[0].q_min, parking it outside
    // Optional mid-run setpoint jump, reproducing what ee_target_publisher does
    // live: the arm settles, then the target moves. That transition is where the
    // warm-start iterate stops being a good guess, so a study of warm-start
    // behaviour has to contain one -- a single fixed setpoint never sees it.
    int    step_at_tick  = 0;           // 0 disables
    double step_offset   = 0.0;         // [m] per axis, added to the original target
};

/*! Runs the real controller closed-loop against MuJoCo for `ticks` at 200 Hz
 *  and reports what a constraint comparison needs, so a test can distinguish
 *  "the soft constraints are idle" from "they are absorbing a violation". */
RunMetrics runClosedLoop(NMPCParams params, const Scenario& sc, int ticks);

/*! The scenario that separates warm from cold starting: settle, then jump the
 *  target. A single fixed setpoint never invalidates the warm-start iterate, so
 *  it cannot show the behaviour. */
Scenario warmStartScenario();

}  // namespace test_support
}  // namespace pat_arm_control
