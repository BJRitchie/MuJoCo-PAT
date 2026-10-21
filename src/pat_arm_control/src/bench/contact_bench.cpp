/*! contact_bench - scores the disturbance observer against contact truth,
 *  deterministically.
 *
 *  ============================ WHAT IT DOES ============================
 *
 *  Drives one arm into the target and asks: when MuJoCo says a contact force of
 *  X acted on a joint, what did the observer estimate? MuJoCo's own
 *  qfrc_constraint is the truth -- no algorithm could know it, which is exactly
 *  what makes it a reference.
 *
 *  ========================= WHY IT IS OFFLINE ==========================
 *
 *  The same measurement against the live ROS stack cannot support a comparison.
 *  Two runs of an identical configuration gave peak-force ratios of
 *  0.37/0.41/0.42 and 0.52/0.24/0.29, with the TRUE peak itself varying 50%:
 *  the impact's peak depends on the arm's state when the setpoint arrives, and
 *  node startup plus DDS timing shifts that by tens of milliseconds. A gain
 *  sweep across such runs measures the startup jitter, not the gain.
 *
 *  So this is one process with no ROS, no DDS, no wall clock and no RNG. Same
 *  reasoning as nmpc_bench and the constraint ladder, both offline for exactly
 *  this reason. Two runs of one configuration produce identical output, which
 *  is what makes --sweep-gain meaningful.
 *
 *  ========================== HOW IT IS WIRED ===========================
 *
 *  Three pieces, each stepped by hand:
 *
 *    ContactScenario   the PLANT: the scene MJCF (arms, bus, target, contact).
 *                      Lives in contact_scenario.h. Owns mjModel/mjData.
 *    ArmProbe x2       the CONTROLLERS: one ArmNMPC per arm, each carrying its
 *                      own internal model, which has NO contact in it -- that
 *                      mismatch is the disturbance being estimated.
 *    MomentumObserver  the thing under test.
 *
 *  Per control tick, in this order, and the order matters:
 *
 *    1. read the plant's state and convert it to the controllers' coordinates
 *    2. both controllers solve, producing joint torques
 *    3. assemble the APPLIED torque per model joint
 *    4. evaluate the controller model's dynamics at that state, and step the
 *       observer with (M, bias, v, tau_applied)
 *    5. record the estimate NEXT TO the plant's qfrc_constraint for this tick
 *    6. apply the torque to the plant and advance it one control period
 *
 *  Step 5 before step 6 is the point: the estimate and the truth it is scored
 *  against describe the same instant. Recording after stepping would compare
 *  an estimate to a force from one period later.
 *
 *  ========================= READING THE OUTPUT =========================
 *
 *    contact on N of M ticks   how much of the run was actually in contact;
 *                              scoring happens over those ticks only
 *    first contact             which geoms touched, so an arm-vs-target contact
 *                              is distinguishable from the two arms colliding
 *    detection latency         ticks from contact starting to the estimate
 *                              exceeding the quiet-run error floor
 *    truth pk / est pk         peak generalised force, and what was estimated
 *    ratio                     est/truth at the peak; < 1 means under-reading
 *    corr                      did the estimate move WITH the truth, regardless
 *                              of magnitude
 *    rms err                   average disagreement over the contact window
 *
 *  =============================== MODES ================================
 *
 *  Three disturbances, one scenario harness. Only the first involves contact;
 *  the other two park the target out of reach so the signal is one thing.
 *
 *    contact        the default. Drive the arm into the target and score the
 *                   estimate against qfrc_constraint.
 *    --drag D       score the base rows against the -D*velocity the table
 *                   applies. The truth is analytic, so it is exact.
 *    --mass-scale S score the joint rows against the torque the controller's
 *                   model gets wrong when the real links are S times heavier.
 *
 *  The last two need the plant MOVING -- drag is proportional to velocity and
 *  an inertia error produces torque only under acceleration -- which is why
 *  --mass-scale oscillates the setpoint rather than holding it.
 *
 *    --trajectory circle|rectangle
 *                   not an observer measurement: the NMPC following a reference
 *                   trajectory, fed three ways that differ only in what the
 *                   controller is told -- the window with preview, the window
 *                   without it, and the current position streamed as a plain
 *                   setpoint (--ab-preview, --sweep-speed).
 *
 *  ========================== TRUTH CHANNELS ============================
 *
 *  Two, recorded side by side every tick:
 *
 *    qfrc_constraint       MuJoCo's own constraint force. Exact, and no
 *                          algorithm could know it, but it is zero whenever
 *                          nothing is touching -- so it cannot score model
 *                          error at all.
 *    M*qacc + bias - tau   the controller's model evaluated at the plant's real
 *                          acceleration. This IS `d` by definition, so it
 *                          covers model error too.
 *
 *  They share no arithmetic, so where both apply they check each other, and the
 *  contact report prints how far apart they got.
 *
 *  ================================ USE =================================
 *
 *    contact_bench
 *    contact_bench --gain 60 --out /tmp/contact.csv
 *    contact_bench --sweep-gain 20,60,150
 *    contact_bench --drag 0.5
 *    contact_bench --mass-scale 2.0 --ab-ff 1.0
 *    contact_bench --trajectory rectangle --ab-preview
 *    contact_bench --trajectory circle --sweep-speed 1,2,3,4
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pat_arm_control/control/nmpc/arm_nmpc.h"
#include "pat_arm_control/observers/momentum_observer.h"
#include "pat_arm_control/reference/reference_trajectory.h"
#include "bench_yaml.h"
#include "contact_scenario.h"

namespace {

using pat_arm_control::ArmNMPC;
using pat_arm_control::MomentumObserver;
using pat_arm_control::MomentumObserverParams;
using pat_arm_control::NMPCParams;
using pat_arm_control::bench::ContactRow;
using pat_arm_control::bench::ContactScenario;
using pat_arm_control::bench::JointScore;

using pat_arm_control::IArmController;
using pat_arm_control::TaskSpaceController;

//! Reads the protected members the bench needs off a control law of any type.
template <class Law>
class Exposed : public Law {
public:
    using Law::Law;
    using Law::ff_max_vec_;
    using Law::n_joints_;
    using Law::ownedJointInds;
};

/*! One arm's controller, whichever law runs it.
 *
 *  Holds the law behind its common base and forwards what the bench calls, so
 *  the trial reads the same for every law. Three members the public interface
 *  does not offer are copied out at construction:
 *
 *    n_joints_, ownedJointInds   the bench works out which arm commands which
 *                                joint (protected in the control law because
 *                                nothing else needs it)
 *    ff_max_vec_                 the per-joint feedforward cap, ff_max_frac *
 *                                the joint's torque limit. Reported alongside
 *                                the disturbance because one larger than the
 *                                cap cannot be fed forward however well it is
 *                                estimated: a fact about the actuators, not the
 *                                observer.
 */
class ArmProbe {
public:
    using EePose = IArmController::EePose;
    using ProfileSnapshot = IArmController::ProfileSnapshot;

    explicit ArmProbe(const pat_arm_control::bench::Loaded& cfg) {
        const auto adopt = [this](auto probe) {
            n_joints_       = probe->n_joints_;
            ownedJointInds  = probe->ownedJointInds;
            ff_max_vec_     = probe->ff_max_vec_;
            law_            = std::move(probe);
        };
        if (cfg.law == "ompc") {
            adopt(std::make_unique<Exposed<pat_arm_control::ArmOMPC>>(cfg.mjcf, cfg.ompc));
        } else if (cfg.law == "smc") {
            adopt(std::make_unique<Exposed<pat_arm_control::ArmSMC>>(cfg.mjcf, cfg.smc));
        } else {
            adopt(std::make_unique<Exposed<ArmNMPC>>(cfg.mjcf, cfg.params));
        }
    }

    int n_joints_ = 0;
    std::vector<int> ownedJointInds;
    Eigen::VectorXd ff_max_vec_;

    EePose currentEePose(const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
        return law_->currentEePose(q, v); }
    void setDesiredPos(double x, double y, double z) { law_->setDesiredPos(x, y, z); }
    void setDesiredOrient(double w, double x, double y, double z) {
        law_->setDesiredOrient(w, x, y, z); }
    void setReferenceWindow(const pat_arm_control::ReferenceWindow& w) {
        law_->setReferenceWindow(w); }
    void setReferenceTime(double t) { law_->setReferenceTime(t); }
    Eigen::VectorXd computeControl(const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
        return law_->computeControl(q, v); }
    void fullDynamics(const Eigen::VectorXd& q, const Eigen::VectorXd& v,
                      pat_arm_control::FloatingBaseDynamics::Workspace& ws,
                      Eigen::MatrixXd& M, Eigen::VectorXd& bias, Eigen::MatrixXd& R) const {
        law_->fullDynamics(q, v, ws, M, bias, R); }
    void setDisturbanceFeedforward(const Eigen::VectorXd& d, double dt) {
        law_->setDisturbanceFeedforward(d, dt); }
    long ffSaturated() const { return law_->ffSaturated(); }
    long ffSlewLimited() const { return law_->ffSlewLimited(); }
    long tier1Entries() const { return law_->tier1Entries(); }
    long safeModeTicks() const { return law_->safeModeTicks(); }
    long referenceSamplesOutside() const { return law_->referenceSamplesOutside(); }
    void snapshotProfile(ProfileSnapshot& out) const { law_->snapshotProfile(out); }
    bool hasSolver() const { return law_->hasSolver(); }

private:
    std::unique_ptr<TaskSpaceController> law_;
};

// ======================== Options and parsing ============================= //

/*! Where the scene MJCF is, without linking against pat_simulation.
 *
 *  colcon installs each package under its own prefix, so the compiled-in
 *  CMAKE_INSTALL_PREFIX of THIS package does not contain pat_simulation's
 *  share directory. AMENT_PREFIX_PATH lists every prefix in the overlay, so
 *  scanning it finds the scene wherever colcon put it -- merged or isolated
 *  install alike. Reading one environment variable is not a ROS dependency.
 *
 *  Falls back to the compiled-in guess, so the failure is a clear "cannot open
 *  <path>" rather than an empty string. --scene overrides either way.
 */
std::string findScene() {
    const char* rel = "/share/pat_simulation/models/environment/air_bearing_table.xml";
    if (const char* paths = std::getenv("AMENT_PREFIX_PATH")) {
        std::stringstream ss(paths);
        std::string prefix;
        while (std::getline(ss, prefix, ':')) {
            if (prefix.empty()) continue;
            const std::string candidate = prefix + rel;
            if (std::ifstream(candidate).good()) return candidate;
        }
    }
    return PAT_CONTACT_BENCH_SCENE;
}

struct Options {
    std::string scene  = findScene();              //!< the plant's MJCF
    std::string law = "nmpc";                      //!< nmpc, ompc or smc
    //! The law's own YAML. Empty until parsing picks the installed file for
    //! `law`, so an explicit --params wins whichever law is chosen.
    std::string params;
    std::string common = PAT_NMPC_BENCH_COMMON;    //!< arm_common.yaml, loaded first
    std::string model;                             //!< controller MJCF override
    std::string out_csv;                           //!< per-tick series, optional
    std::string arm = "left";                      //!< which arm is scored
    double seconds = 6.0;
    double hz = 100.0;                             //!< control rate
    //! Negative means "keep whatever nmpc.yaml says", so an unset flag never
    //! silently overrides the deployed configuration.
    double gain = -1.0;
    double ff_gain = -1.0;
    //! The scored arm's EE target, world frame. The default is 8 cm past the
    //! target's near face: a sustained shove, which is what produces a contact
    //! worth measuring. ee_targets.yaml's waypoint loop does NOT -- its offsets
    //! only command the EE out to x ~ 0.127, short of the target.
    double sp_x = 0.26, sp_y = -0.42;
    //! initial_target_qpos x/y, an offset from the target body's MJCF anchor.
    double tgt_x = -0.0175, tgt_y = -0.42;
    bool free_target = false;
    std::vector<double> sweep_gain;                //!< one trial per entry

    /*! Drag mode: measure the observer against air-bearing table drag instead
     *  of against contact.
     *
     *  > 0 replaces the base damping with this value and parks the target well
     *  out of reach, so nothing touches and the ONLY unmodelled force is the
     *  drag -- which the scene applies as -damping * velocity and the
     *  controller's model has no term for at all.
     *
     *  This is the measurement the plan calls the headline validation, and it
     *  could not be made in the live stack: it needs the base MOVING (drag is
     *  proportional to velocity, so a pinned base has none to measure) while
     *  also not drifting far enough to make the setpoint unreachable. A
     *  deterministic short run with moderate damping is both. 500 is the
     *  pinning value and leaves nothing to see; the shipped table value is
     *  0.05. */
    double drag_damping = -1.0;

    /*! Model-error mode: scale every arm link's mass and inertia in the SCENE
     *  only, so the controller's model is genuinely wrong.
     *
     *  1.0 is no perturbation. Like drag mode this parks the target out of
     *  reach, because the question is what the observer does about a wrong
     *  model, not about contact.
     *
     *  Mass error is invisible unless the arm is ACCELERATING -- the mismatch
     *  torque is (M_scene - M_model) * qacc, which is identically zero at rest
     *  however wrong the masses are. A static arm with 3x mass on two links
     *  measured an estimate of 0.000. So this mode drives the setpoint in a
     *  sine (see osc_amp) rather than to a fixed point. */
    double mass_scale = 1.0;

    /*! Oscillate the scored arm's EE setpoint about its start pose instead of
     *  holding a fixed point: amplitude [m] along x, and frequency [Hz].
     *
     *  Peak commanded acceleration is amp * (2*pi*hz)^2, which is what makes an
     *  inertia error observable. Deterministic -- a sine of SIMULATED time, not
     *  of the wall clock. Zero amplitude keeps the fixed-setpoint behaviour. */
    double osc_amp = 0.0;
    double osc_hz  = 0.75;

    /*! Run the whole trial twice, with the feedforward off and then on at this
     *  gain, and report EE tracking for both. Negative disables.
     *
     *  The A/B is the only way to answer whether acting on the estimate HELPS:
     *  scoring d_hat against truth says the observer is correct, not that the
     *  feedforward improves anything. Valid for the same reason the gain sweep
     *  is -- the two runs are identical but for the feedforward. */
    double ab_ff = -1.0;

    /*! Base damping override, any mode. Negative keeps the mode's own default.
     *
     *  This matters more than it looks. The feedforward acts on the
     *  BASE-ELIMINATED estimate, and the Schur reduction
     *  d_m - (H_bb^-1 H_bm)^T d_b mixes the base rows into every joint row. The
     *  pinning value of 500 therefore injects the rig's own drag straight into
     *  the feedforward, where it is an artefact of the test and not something
     *  the flight controller would ever see. So the oscillating modes default
     *  to the shipped table value instead of to pinning. */
    double base_damping = -1.0;

    /*! Trajectory mode: the scored arm follows a reference trajectory --
     *  "circle" or "rectangle" -- placed from its start pose by the same
     *  library code ee_target_publisher uses, sampled into a fresh window every
     *  tick exactly as the live planner sends one. Empty disables.
     *
     *  This is the measurement of what the NMPC's preview buys: the circle is
     *  feasible, so it shows how closely the arm follows a reference it CAN
     *  follow; the rectangle's sharp corners are not, so it shows what the
     *  controller gives up and where. */
    std::string trajectory;
    //! Speeds the shape up: the circle's period is divided by it and the
    //! rectangle's speed multiplied. 1 is the planner's default.
    double traj_speed = 1.0;
    //! Score every node against the reference NOW (reference_preview: false):
    //! the reactive baseline. The horizon is then told the target stands still
    //! at today's position while moving at today's velocity -- a contradiction
    //! the solver resolves by lagging, which is what "no future knowledge"
    //! means for this formulation.
    bool no_preview = false;
    /*! The other baseline: no window at all, the CURRENT reference position
     *  streamed as an ordinary setpoint, velocity and acceleration zero. That
     *  is what today's ee_setpoint interface does with a moving target, so it
     *  is the comparison for "what does the trajectory interface buy". */
    bool stream_setpoint = false;
    //! Run preview on and off and compare them side by side.
    bool ab_preview = false;
    //! One preview A/B per speed factor, in one table.
    std::vector<double> sweep_speed;
    //! Whether --seconds was given; trajectory mode otherwise sizes the run to
    //! the approach plus two laps.
    bool seconds_set = false;
};

//! The air-bearing table's own damping, from simulation.yaml. The realistic
//! value, as against the 500 used to pin the base for contact work.
constexpr double kTableDamping = 0.05;

//! True when --drag asked for the drag measurement rather than contact.
bool dragMode(const Options& o) { return o.drag_damping > 0.0; }

/*! True when the setpoint sweeps: the tracking measurement.
 *
 *  --mass-scale selects it implicitly because model error is only observable
 *  under acceleration, but it is a mode in its own right, and running it with
 *  no mass error at all is the BASELINE the model-error runs are compared
 *  against. Without that baseline a tracking number means nothing: an error of
 *  12 cm could be the model error or could be what this arm does anyway. */
bool trackingMode(const Options& o) { return o.osc_amp > 0.0; }

//! Whether a model error is actually in force within a tracking run.
bool hasModelError(const Options& o) { return o.mass_scale != 1.0; }

//! True when --trajectory asked for trajectory following.
bool trajMode(const Options& o) { return !o.trajectory.empty(); }

//! The alternative modes park the target out of reach, so none is scored
//! against a signal contaminated by contact.
bool quietMode(const Options& o) {
    return dragMode(o) || trackingMode(o) || trajMode(o);
}

/*! The planner's shape for this run: ee_target_publisher's defaults (the
 *  library's), with the speed factor applied. */
pat_arm_control::TrajectorySpec trajSpec(const Options& o) {
    pat_arm_control::TrajectorySpec s;
    s.pattern = o.trajectory;
    s.circle.period /= o.traj_speed;
    s.rectangle.speed *= o.traj_speed;
    return s;
}

void usage() {
    std::cout <<
R"(contact_bench - deterministic scoring of the disturbance observer against
                MuJoCo's own contact force. No ROS, no DDS, no wall clock.

  --scene PATH       scene MJCF (default: the installed air_bearing_table.xml)
  --law nmpc|ompc|smc  which control law drives both arms (default nmpc). The
                     observer scoring and the trajectory run work for any law;
                     --no-preview, --ab-preview and --sweep-speed compare a
                     horizon's preview and so need nmpc
  --params PATH      the law's YAML: nmpc.yaml, ompc.yaml or smc.yaml (default:
                     the installed one for --law)
  --common PATH      arm_common.yaml, loaded before it (default: the installed one)
  --model PATH       the controller's internal MJCF (default: installed)
  --arm left|right   which arm to score (default left)
  --seconds S        simulated seconds (default 6)
  --hz H             control rate (default 100)
  --gain G           observer.gain override
  --ff-gain F        observer.ff_gain override; 0 measures without acting
  --sweep-gain A,B,C run once per gain and print one table. Valid BECAUSE the
                     scenario is deterministic: the contact event is identical
                     across runs, so a difference is the gain and nothing else
  --setpoint X,Y     the scored arm's EE target, world frame (default
                     0.26,-0.42 -- 8 cm past the target's near face, a shove)
  --target X,Y       initial_target_qpos x,y (default -0.0175,-0.42)
  --free-target      leave the target free; the default pins it so contact is
                     sustained and scorable instead of a transient that knocks
                     a 20 kg body away
  --drag D           measure against table DRAG instead of contact: set the
                     base damping to D, park the target out of reach, and score
                     the base-row estimate against the -D*velocity the scene is
                     actually applying. Needs the base moving, so D must be
                     moderate -- 0.05 is the shipped table value, 500 pins it
                     and leaves no drag to see
  --mass-scale S     measure against MODEL ERROR instead of contact: scale every
                     arm link's mass and inertia by S in the scene only, park
                     the target out of reach, and score the estimate against the
                     mismatch torque. Implies --osc, because an inertia error
                     produces no torque at all unless the arm is accelerating
  --osc A[,HZ]       oscillate the scored arm's EE setpoint about its start pose
                     by A metres along x at HZ (default 0.75), instead of
                     driving it to a fixed setpoint. Tracked in the BASE's frame,
                     so base drift cannot put it out of reach
  --base-damping D   override the base damping. Contact mode pins at 500; the
                     oscillating modes use the table's own 0.05, because pinning
                     puts the rig's drag into the base-eliminated feedforward
  --ab-ff G          run the trial twice, feedforward off then on at gain G, and
                     report EE tracking for both. The only test of whether
                     acting on the estimate actually improves anything
  --trajectory circle|rectangle
                     follow a reference trajectory from the start pose, built by
                     the planner's own library code and sent as a fresh window
                     every tick. The circle is feasible; the rectangle's sharp
                     corners deliberately are not
  --traj-speed F     speed the shape up by F (default 1, the planner's default)
  --no-preview       score every horizon node against the reference now: the
                     reactive baseline (reference_preview: false)
  --stream-setpoint  no window: stream the current reference position as a plain
                     setpoint, as today's ee_setpoint interface would
  --ab-preview       run preview on, preview off and streamed setpoint, compared
  --sweep-speed A,B  that three-way comparison per speed factor, in one table
  --out CSV          write the per-tick series
)";
}

/*! Fills `o` from argv.
 *
 *  Returns false when the program should exit -- either because --help was
 *  asked for (code 0) or an argument was bad (code 2).
 */
bool parseArgs(int argc, char** argv, Options& o, int& code) {
    code = 0;
    // "X,Y" -> two doubles, for --setpoint and --target.
    const auto xy = [](const std::string& v, double& x, double& y) {
        const auto comma = v.find(',');
        if (comma == std::string::npos) throw std::runtime_error("wanted X,Y");
        x = std::stod(v.substr(0, comma));
        y = std::stod(v.substr(comma + 1));
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        try {
            // Consumes the next argv entry as this flag's value.
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("needs a value");
                return argv[++i];
            };
            if      (a == "--scene")   o.scene   = next();
            else if (a == "--law")     o.law     = next();
            else if (a == "--params")  o.params  = next();
            else if (a == "--common")  o.common  = next();
            else if (a == "--model")   o.model   = next();
            else if (a == "--arm")     o.arm     = next();
            else if (a == "--out")     o.out_csv = next();
            else if (a == "--seconds") { o.seconds = std::stod(next()); o.seconds_set = true; }
            else if (a == "--trajectory")  o.trajectory = next();
            else if (a == "--traj-speed")  o.traj_speed = std::stod(next());
            else if (a == "--no-preview")  o.no_preview = true;
            else if (a == "--stream-setpoint") o.stream_setpoint = true;
            else if (a == "--ab-preview")  o.ab_preview = true;
            else if (a == "--sweep-speed") {
                std::stringstream ss(next());
                std::string tok;
                while (std::getline(ss, tok, ',')) o.sweep_speed.push_back(std::stod(tok));
            }
            else if (a == "--hz")      o.hz      = std::stod(next());
            else if (a == "--gain")    o.gain    = std::stod(next());
            else if (a == "--ff-gain") o.ff_gain = std::stod(next());
            else if (a == "--free-target") o.free_target = true;
            else if (a == "--drag")    o.drag_damping = std::stod(next());
            else if (a == "--mass-scale") o.mass_scale = std::stod(next());
            else if (a == "--base-damping") o.base_damping = std::stod(next());
            else if (a == "--ab-ff")   o.ab_ff   = std::stod(next());
            else if (a == "--osc") {
                // "A" or "A,HZ" -- the frequency keeps its default when omitted.
                const std::string v = next();
                const auto comma = v.find(',');
                o.osc_amp = std::stod(v.substr(0, comma));
                if (comma != std::string::npos) o.osc_hz = std::stod(v.substr(comma + 1));
            }
            else if (a == "--setpoint") xy(next(), o.sp_x, o.sp_y);
            else if (a == "--target")   xy(next(), o.tgt_x, o.tgt_y);
            else if (a == "--sweep-gain") {
                std::stringstream ss(next());
                std::string tok;
                while (std::getline(ss, tok, ',')) o.sweep_gain.push_back(std::stod(tok));
            } else if (a == "-h" || a == "--help") {
                usage();
                return false;
            } else {
                std::cerr << "unknown argument: " << a << "\n";
                usage();
                code = 2;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "contact_bench: " << a << ": " << e.what() << "\n";
            code = 2;
            return false;
        }
    }
    if (o.arm != "left" && o.arm != "right") {
        std::cerr << "contact_bench: --arm must be left or right\n";
        code = 2;
        return false;
    }
    if (o.law != "nmpc" && o.law != "ompc" && o.law != "smc") {
        std::cerr << "contact_bench: --law must be nmpc, ompc or smc\n";
        code = 2;
        return false;
    }
    if (o.params.empty()) {
        o.params = o.law == "ompc" ? PAT_OMPC_BENCH_PARAMS
                 : o.law == "smc"  ? PAT_SMC_BENCH_PARAMS
                                   : PAT_NMPC_BENCH_PARAMS;
    }
    // Preview is a property of a horizon: a one-step law reads the reference at
    // the current time only, so preview on and off are the same run, and a
    // table claiming to compare them would compare a run with itself.
    if (o.law != "nmpc" && (o.no_preview || o.ab_preview || !o.sweep_speed.empty())) {
        std::cerr << "contact_bench: --no-preview, --ab-preview and --sweep-speed "
                     "compare reference preview, which only the nmpc law has; "
                     "use --law nmpc\n";
        code = 2;
        return false;
    }
    if (o.seconds <= 0.0 || o.hz <= 0.0) {
        std::cerr << "contact_bench: --seconds and --hz must be positive\n";
        code = 2;
        return false;
    }
    if (!(o.mass_scale > 0.0)) {
        std::cerr << "contact_bench: --mass-scale must be positive\n";
        code = 2;
        return false;
    }
    if (o.osc_amp < 0.0 || o.osc_hz <= 0.0) {
        std::cerr << "contact_bench: --osc needs a non-negative amplitude and a "
                     "positive frequency\n";
        code = 2;
        return false;
    }
    // Model error without motion measures nothing: the mismatch torque is
    // (M_scene - M_model) * qacc. Rather than silently reporting zeros, supply
    // the motion the mode needs. An explicit --osc still wins.
    if (hasModelError(o) && o.osc_amp == 0.0) o.osc_amp = 0.08;
    if (dragMode(o) && trackingMode(o)) {
        std::cerr << "contact_bench: --drag and --osc/--mass-scale score "
                     "different rows against different truths; run them "
                     "separately\n";
        code = 2;
        return false;
    }
    if (trajMode(o)) {
        if (dragMode(o) || trackingMode(o)) {
            std::cerr << "contact_bench: --trajectory replaces the setpoint that "
                         "--osc/--mass-scale/--drag drive; run them separately\n";
            code = 2;
            return false;
        }
        if (o.trajectory != "circle" && o.trajectory != "rectangle") {
            std::cerr << "contact_bench: --trajectory must be circle or rectangle\n";
            code = 2;
            return false;
        }
        if (!(o.traj_speed > 0.0)) {
            std::cerr << "contact_bench: --traj-speed must be positive\n";
            code = 2;
            return false;
        }
        for (const double f : o.sweep_speed) {
            if (!(f > 0.0)) {
                std::cerr << "contact_bench: --sweep-speed factors must be positive\n";
                code = 2;
                return false;
            }
        }
    } else if (o.no_preview || o.stream_setpoint || o.ab_preview || !o.sweep_speed.empty()) {
        std::cerr << "contact_bench: --no-preview, --stream-setpoint, --ab-preview and "
                     "--sweep-speed need --trajectory\n";
        code = 2;
        return false;
    }
    return true;
}

// =========================== Configuration =============================== //

/*! nmpc.yaml keys this bench knowingly ignores.
 *
 *  Anything in the file that is neither mapped by bench_yaml nor listed here is
 *  a hard error, so adding a knob to nmpc.yaml without teaching it to the bench
 *  says so rather than quietly measuring a different configuration. These are
 *  the node-only ones: topics, the ROS clock, the realtime policy (this is a
 *  plain process; pin it with chrt/taskset from outside).
 */
const std::set<std::string> kNotOurs = {
    "control_hz", "arm_side",
    "rt_priority", "lock_memory", "rt_cpus",
    "rt_priority_observer", "rt_cpus_observer",
    "trace_file", "trace_capacity", "pub_rate_hz",
};

/*! Build the plant's configuration: where the target sits, how the arm starts,
 *  and what is pinned. */
ContactScenario::Config sceneConfig(const Options& o,
                                    const std::vector<std::string>& model_joints) {
    ContactScenario::Config c;
    c.scene_mjcf   = o.scene;
    c.model_joints = model_joints;

    // simulation.yaml's initial_arm_qpos, mirrored across the two arms (the
    // model joint list is left arm then right arm, so %6 repeats the triple).
    // The MJCF's own all-zero fold pose self-collides at the real arm mount, so
    // starting from the model default would begin the run already in contact.
    const double home[6] = {0.6108652382, -0.2617993878, 0.0,
                            0.6108652382, -0.2617993878, 0.0};
    c.arm_home.resize(model_joints.size());
    for (std::size_t i = 0; i < model_joints.size(); ++i) c.arm_home[i] = home[i % 6];

    c.target_x = o.tgt_x;
    c.target_y = o.tgt_y;
    // Zero damping leaves the target at whatever the MJCF ships, i.e. free.
    if (o.free_target) c.target_damping = {{0.0, 0.0, 0.0}};

    if (dragMode(o)) {
        // Moderate damping on all three planar DOFs, so the base moves (drag
        // is proportional to velocity) without drifting far.
        c.base_damping = {{o.drag_damping, o.drag_damping, o.drag_damping}};
    } else if (trackingMode(o) || trajMode(o)) {
        // NOT the pinning value: see Options::base_damping. Pinning would put
        // the rig's own drag into the feedforward through the base elimination,
        // and a trajectory run should see the free-floating base the arm has.
        c.base_damping = {{kTableDamping, kTableDamping, kTableDamping}};
    }
    if (o.base_damping >= 0.0) {
        c.base_damping = {{o.base_damping, o.base_damping, o.base_damping}};
    }
    c.arm_mass_scale = o.mass_scale;
    if (quietMode(o)) {
        // The target well out of the way, so contact cannot contaminate a
        // measurement that is supposed to be one disturbance alone.
        c.target_x = 2.0;
        c.target_y = 2.0;
    }
    return c;
}

/*! Build the observer's parameters from the config, with an optional gain
 *  override for a sweep. `gain_override` < 0 keeps the config's value. */
MomentumObserverParams observerConfig(const pat_arm_control::bench::Loaded& cfg,
                                      int n_joints, double dt,
                                      double gain_override) {
    MomentumObserverParams op;
    op.nv       = 6 + n_joints;      // 6 floating-base DOFs, then the joints
    op.n_joints = n_joints;
    op.dt       = dt;                // must match the call rate; the filter integrates
    const double g = gain_override >= 0.0 ? gain_override : cfg.observer.gain;
    op.gain = Eigen::VectorXd::Constant(op.nv, g);
    // The base rows can take their own, slower gain: they are a different
    // physical quantity (force, not torque) on a noisier channel.
    if (cfg.observer.gain_base >= 0.0) op.gain.head(6).setConstant(cfg.observer.gain_base);
    op.mdot_clamp     = cfg.observer.mdot_clamp;
    op.estimate_clamp = cfg.observer.estimate_clamp;
    return op;
}

// ============================== The trial ================================= //

/*! Everything one run produced. */
struct Trial {
    std::vector<ContactRow> rows;          //!< one per control tick
    std::vector<int> scored_joints;        //!< model indices the scored arm owns
    std::vector<std::string> joint_names;  //!< all model joints, for labelling
    int contact_ticks = 0;
    int max_ncon = 0;
    int first_contact_tick = -1;           //!< -1 if contact never happened
    int first_detect_tick = -1;            //!< -1 if the observer never saw it
    std::string first_contact_pairs;       //!< which geoms touched, at onset
    double applied_gain = 0.0;             //!< what the joint rows actually used
    double applied_ff = 0.0;               //!< the ff_gain the command used
    long   obs_steps = 0;
    double obs_max_mdot = 0.0;
    std::array<double, 3> base_peak{{0.0, 0.0, 0.0}};   //!< fx, fy, tz

    //! Total arm-link mass the PLANT has. Stated in the report so the size of
    //! the model error is read in kilograms, not inferred from a flag.
    double scene_arm_mass = 0.0;

    //! Base damping in force, and how far the base actually travelled. Reported
    //! because a free base drifting far is the standing confound of this rig,
    //! so a run has to show how much of it happened.
    double base_damping = 0.0;
    double base_drift = 0.0;

    //! How often the feedforward hit its saturation bound or its slew limit.
    //! Nonzero means the guards were load-bearing in this run, so the A/B
    //! compares a limited feedforward and the report has to say so.
    long ff_saturated = 0;
    long ff_slew_limited = 0;

    //! Per model joint: the feedforward cap, and the joint's torque limit.
    std::vector<double> ff_cap;
    std::vector<double> tau_max;

    /*! Per model joint: the largest torque actually commanded, and how many
     *  ticks it sat within 1% of the limit.
     *
     *  Reported because a torque-starved arm and a badly tuned one look the
     *  same from the tracking error alone, and the difference decides whether a
     *  feedforward could have helped at all. */
    std::vector<double> tau_peak;
    std::vector<long>   tau_railed;

    /*! Drag mode only: the analytic drag the scene applied, paired with what
     *  the observer's base rows estimated, per tick. Three entries each:
     *  x force, y force, yaw torque. */
    std::vector<std::array<double, 3>> drag_truth;
    std::vector<std::array<double, 3>> drag_est;

    /*! Trajectory mode only: per tick, the reference the arm was asked to be
     *  at (position and velocity) and where its EE actually was, both world
     *  frame, measured before the solve. */
    struct TrajRow {
        double t = 0.0;
        double rx = 0.0, ry = 0.0, rvx = 0.0, rvy = 0.0;
        double ex = 0.0, ey = 0.0;
    };
    std::vector<TrajRow> traj;
    //! The trajectory the run followed, for the report's geometry.
    std::shared_ptr<pat_arm_control::ReferenceTrajectory> traj_shape;
    //! "preview", "no preview" or "setpoint" (streamed): how the run was fed.
    std::string ref_mode = "preview";
    long   tier1_entries = 0;
    std::string law = "nmpc";   //!< which control law ran
    bool   has_solver = true;   //!< false for a closed-form law: no QP to report
    long   safe_ticks = 0;
    long   ref_outside = 0;
    double qp_iter_mean = 0.0, qp_iter_max = 0.0;
};

/*! The estimate must exceed this to count as having detected contact.
 *
 *  Set above the quiet-run error floor, which the unit tests put at well under
 *  0.01 N*m at operating speeds -- otherwise the floor itself would register as
 *  a detection on the first tick.
 */
constexpr double kDetectThreshold = 0.02;

/*! Run one trial and return everything it measured.
 *
 *  `gain_override` < 0 keeps nmpc.yaml's observer.gain; a sweep passes each
 *  value in turn. Everything else comes from the Options.
 */
Trial runTrial(const Options& o, double gain_override) {
    // --- 1. Configuration ------------------------------------------------- //
    // Both arms' parameter blocks: the scored arm, and the other one, which has
    // to be controlled too (see the hold-pose comment below).
    auto cfg = pat_arm_control::bench::loadConfig(
        {o.common, o.params}, o.arm, o.model, PAT_NMPC_BENCH_MODEL, kNotOurs,
        "contact_bench", o.law);
    const std::string other = (o.arm == "left") ? "right" : "left";
    auto cfg_other = pat_arm_control::bench::loadConfig(
        {o.common, o.params}, other, o.model, PAT_NMPC_BENCH_MODEL, kNotOurs,
        "contact_bench", o.law);
    if (o.ff_gain >= 0.0) cfg.common().ff_gain = o.ff_gain;
    if (trajMode(o) && o.no_preview) cfg.common().reference_preview = false;

    const double dt = 1.0 / o.hz;
    // control_period drives the warm-start shift and the deadline accounting;
    // the node sets it from control_hz, so --hz has to do the same here.
    cfg.common().control_period       = dt;
    cfg_other.common().control_period = dt;

    // --- 2. The plant and the controllers --------------------------------- //
    ContactScenario scene(sceneConfig(o, cfg.joints.model));
    const int n_joints = scene.numJoints();

    ArmProbe scored(cfg);
    ArmProbe held(cfg_other);
    // The two models must agree on how many arm joints exist, or every index
    // below means something different in each.
    if (scored.n_joints_ != n_joints) {
        throw std::runtime_error(
            "the controller model has " + std::to_string(scored.n_joints_) +
            " arm joints but model_joint_names lists " + std::to_string(n_joints));
    }

    const auto op = observerConfig(cfg, n_joints, dt, gain_override);
    MomentumObserver obs(op);
    // The observer's dynamics workspace. One per thread; here there is only one
    // thread, but the API takes it explicitly so the requirement is visible.
    pat_arm_control::FloatingBaseDynamics::Workspace ws;

    // Reused across ticks so the loop does not reallocate. q is 7+n (the
    // freejoint carries a quaternion), v and tau are 6+n.
    Eigen::VectorXd q, v, tau_full(6 + n_joints), d_reduced(n_joints), truth;
    Eigen::MatrixXd M, R;
    Eigen::VectorXd bias;
    // The acceleration-derived truth channel: acc from the plant, resid from the
    // controller's own model evaluated at the same state.
    Eigen::VectorXd acc, resid(6 + n_joints);

    // --- 3. Setpoints ----------------------------------------------------- //
    scene.readControllerState(cfg.anchor.x, cfg.anchor.y, q, v);

    // The other arm holds exactly where it started, which is what the node does
    // when no setpoint has ever arrived for it. This is not cosmetic: left
    // limp, that arm drifts into a joint range limit, and MuJoCo puts
    // joint-limit forces in the SAME qfrc_constraint vector as contact forces.
    // The result is a convincing false contact in the truth signal.
    const ArmProbe::EePose hold = held.currentEePose(q, v);
    held.setDesiredPos(hold.pos.x(), hold.pos.y(), hold.pos.z());
    held.setDesiredOrient(hold.quat(0), hold.quat(1), hold.quat(2), hold.quat(3));

    // The scored arm is driven to a fixed point past the target's face. Its
    // orientation setpoint is simply where it started, so the run is a straight
    // push rather than a push plus a twist.
    const ArmProbe::EePose start = scored.currentEePose(q, v);
    scored.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    // The start pose expressed in the base's frame, so an oscillating setpoint
    // can be put back into the world each tick wherever the base has got to.
    const auto base0 = scene.basePose();
    const double off_x = start.pos.x() - (base0[0] + cfg.anchor.x);
    const double off_y = start.pos.y() - (base0[1] + cfg.anchor.y);

    /*! Where the scored arm is told to put its EE at simulated time `now`.
     *
     *  Fixed and world-frame by default, which is what the contact scenario
     *  needs: it drives the EE to a point inside a target that does not move.
     *
     *  With osc_amp set it is instead a sine about the START pose expressed in
     *  the BASE's frame. Two reasons. It oscillates about the start rather than
     *  about --setpoint because those modes measure what happens while the arm
     *  accelerates, and sweeping about a point 8 cm inside the target would be
     *  a run of repeated collisions. And it follows the base because those
     *  modes do not pin the base: a world-frame setpoint then drifts out of
     *  reach, which is the confound that makes every long free-base run
     *  uninterpretable. Tracking in the base frame removes it, and the base
     *  frame is the right one for arm control regardless. */
    const auto setpointAt = [&](double now, double& x, double& y) {
        if (o.osc_amp == 0.0) {
            x = o.sp_x;
            y = o.sp_y;
            return;
        }
        const auto b = scene.basePose();
        const double s = o.osc_amp * std::sin(2.0 * M_PI * o.osc_hz * now);
        // Rotate the stored offset by the base's current yaw, so a base that
        // has turned does not drag the setpoint sideways.
        const double c = std::cos(b[2] - base0[2]);
        const double sn = std::sin(b[2] - base0[2]);
        x = b[0] + cfg.anchor.x + c * (off_x + s) - sn * off_y;
        y = b[1] + cfg.anchor.y + sn * (off_x + s) + c * off_y;
    };

    Trial t;

    // Trajectory mode: the shape, placed from the start pose by the planner's
    // own code, and the window it is sent in. The window's reference
    // orientation is the start orientation, so holding yaw at 0 holds it.
    // ~20 KB, so it lives for the run rather than being rebuilt from scratch.
    pat_arm_control::ReferenceWindow window;
    if (trajMode(o)) {
        t.traj_shape = pat_arm_control::makeTrajectory(
            trajSpec(o), pat_arm_control::Anchor{start.pos.x(), start.pos.y(), 0.0});
        for (int i = 0; i < 4; ++i) window.quat0[i] = start.quat(i);
        window.yaw0 = 0.0;
        // A one-step law reads the window at the current time only, so it is
        // neither previewing nor not: "window".
        t.ref_mode = o.stream_setpoint ? "setpoint"
                   : o.law != "nmpc" ? "window"
                   : cfg.common().reference_preview ? "preview" : "no preview";
    }
    // The live planner's window: 2 s ahead at 10 ms, the horizon being ~1 s.
    constexpr double kWindowS = 2.0, kSampleDt = 0.01;

    t.applied_gain   = op.gain(6);     // a joint row, past any base override
    t.applied_ff     = cfg.common().ff_gain;
    t.law            = o.law;
    t.has_solver     = scored.hasSolver();
    t.joint_names    = cfg.joints.model;
    t.scored_joints  = scored.ownedJointInds;
    t.scene_arm_mass = scene.armMass();
    t.base_damping   = scene.baseDamping()[0];
    t.ff_cap.assign(static_cast<std::size_t>(n_joints), 0.0);
    t.tau_max.assign(static_cast<std::size_t>(n_joints), 0.0);
    t.tau_peak.assign(static_cast<std::size_t>(n_joints), 0.0);
    t.tau_railed.assign(static_cast<std::size_t>(n_joints), 0);
    for (int i = 0; i < n_joints; ++i) {
        const auto u = static_cast<std::size_t>(i);
        t.ff_cap[u]  = scored.ff_max_vec_(i);
        // ff_max_vec_ is ff_max_frac * the limit, so the limit is recovered
        // from it rather than read from a second place that could disagree.
        t.tau_max[u] = cfg.common().ff_max_frac > 0.0
                           ? scored.ff_max_vec_(i) / cfg.common().ff_max_frac : 0.0;
    }

    // Physics runs faster than control (500 Hz against 100 Hz as shipped), so
    // each control tick advances the plant several physics steps.
    const int steps_per_tick =
        std::max(1, static_cast<int>(std::lround(dt / scene.timestep())));
    const int ticks = static_cast<int>(std::lround(o.seconds * o.hz));
    t.rows.reserve(static_cast<std::size_t>(ticks));

    // --- 4. The loop ------------------------------------------------------ //
    for (int k = 0; k < ticks; ++k) {
        // (a) Plant state -> controller coordinates. Both controllers see the
        //     whole model state; each acts only on the joints it owns.
        scene.readControllerState(cfg.anchor.x, cfg.anchor.y, q, v);

        // (a2) Command this tick's setpoint, and measure how far the EE is from
        //      it right now. Measured BEFORE the solve, against the setpoint
        //      this tick was asked to reach, so the number is a tracking error
        //      and not a comparison with where the arm is about to be sent.
        double sx = 0.0, sy = 0.0;
        const double t_now = scene.time();
        if (trajMode(o)) {
            // A fresh window every tick, as the planner sends one, and the
            // state's own time to evaluate it at -- or, streaming, just the
            // reference position as a setpoint. The error is against the
            // reference for NOW, whatever the horizon was shown.
            const pat_arm_control::RefPoint r = t.traj_shape->at(t_now);
            if (o.stream_setpoint) {
                scored.setDesiredPos(r.x, r.y, start.pos.z());
            } else {
                pat_arm_control::sampleWindow(*t.traj_shape, 0.0, t_now, kWindowS,
                                              kSampleDt, window);
                scored.setReferenceWindow(window);
                scored.setReferenceTime(t_now);
            }
            sx = r.x;
            sy = r.y;
            Trial::TrajRow tr;
            tr.t = t_now;
            tr.rx = r.x;   tr.ry = r.y;
            tr.rvx = r.vx; tr.rvy = r.vy;
            t.traj.push_back(tr);
        } else {
            setpointAt(t_now, sx, sy);
            scored.setDesiredPos(sx, sy, start.pos.z());
        }
        const ArmProbe::EePose now_ee = scored.currentEePose(q, v);
        const double ee_err = std::hypot(now_ee.pos.x() - sx, now_ee.pos.y() - sy);
        if (trajMode(o)) {
            t.traj.back().ex = now_ee.pos.x();
            t.traj.back().ey = now_ee.pos.y();
        }

        // (b) Solve. Each returns a vector over ALL model joints with only its
        //     own entries meaningful.
        const Eigen::VectorXd tau_s = scored.computeControl(q, v);
        const Eigen::VectorXd tau_h = held.computeControl(q, v);

        // (c) The APPLIED torque, assembled per model joint from whichever arm
        //     owns it -- after each controller's own per-joint clamp. The
        //     observer must be told the applied value, not the planned one:
        //     feeding the pre-clamp torque makes the clamped difference read as
        //     a disturbance. Base rows stay zero, the base being unactuated.
        tau_full.setZero();
        for (const int idx : scored.ownedJointInds) tau_full(6 + idx) = tau_s(idx);
        for (const int idx : held.ownedJointInds)   tau_full(6 + idx) = tau_h(idx);

        // (c1) How hard the actuators are working, before anything else looks
        //      at the torque.
        for (int i = 0; i < n_joints; ++i) {
            const auto u = static_cast<std::size_t>(i);
            const double mag = std::fabs(tau_full(6 + i));
            t.tau_peak[u] = std::max(t.tau_peak[u], mag);
            if (t.tau_max[u] > 0.0 && mag >= 0.99 * t.tau_max[u]) ++t.tau_railed[u];
        }

        // (c2) Put that torque on the plant and refresh its derived quantities,
        //      so qacc and qfrc_constraint describe THIS instant under THIS
        //      torque. Without the refresh they would still describe the last
        //      substep of the previous tick, and the residual below would pair
        //      one tick's torque with another tick's acceleration. The plant is
        //      not advanced here -- that happens at (g).
        scene.applyArmTorque(tau_full.segment(6, n_joints));
        scene.refreshDerived();

        // (d) Step the observer. fullDynamics evaluates the CONTROLLER's model
        //     at this state -- the model with no contact in it, which is why
        //     the contact shows up as a disturbance at all.
        scored.fullDynamics(q, v, ws, M, bias, R);
        obs.update(M, bias, v, tau_full);
        obs.reduceToJoints(R, d_reduced);

        // (d2) Hand the estimate to the control law, which will act on it NEXT
        //      tick. One tick of lag is not a simplification: the node's
        //      observer runs on its own thread and the control thread reads
        //      whatever the mailbox last held, so the bench would overstate the
        //      feedforward if it applied the estimate to the solve it came
        //      from. With ff_gain at 0 this records and changes nothing.
        scored.setDisturbanceFeedforward(d_reduced, dt);

        // (d3) The independent truth channel. The observer integrates to avoid
        //      ever touching an acceleration; this evaluates the same residual
        //      directly from the plant's qacc, so the two share no arithmetic
        //      and an agreement between them means something.
        scene.readControllerAccel(acc);
        resid.noalias() = M * acc;
        resid += bias;
        resid -= tau_full;

        // (e) Record estimate and truth together, BEFORE advancing the plant,
        //     so both describe the same instant.
        scene.armConstraintForce(truth);
        ContactRow row;
        row.t    = scene.time();
        row.ncon = scene.contactCount();
        row.ee_err = ee_err;
        row.truth.resize(static_cast<std::size_t>(n_joints));
        row.raw.resize(static_cast<std::size_t>(n_joints));
        row.reduced.resize(static_cast<std::size_t>(n_joints));
        row.resid.resize(static_cast<std::size_t>(n_joints));
        for (int i = 0; i < n_joints; ++i) {
            const auto u = static_cast<std::size_t>(i);
            row.truth[u]   = truth(i);
            row.resid[u]   = resid(6 + i);
            // raw: the full-system estimate's joint rows, which is what
            // qfrc_constraint is directly comparable to. reduced: what the
            // controller would act on, which differs by the base coupling.
            row.raw[u]     = obs.estimate()(6 + i);
            row.reduced[u] = d_reduced(i);
        }
        row.base = {{obs.estimate()(0), obs.estimate()(1), obs.estimate()(5)}};

        // (f) Running tallies: when contact began, what touched, and when the
        //     observer first registered it.
        if (row.ncon > 0) {
            ++t.contact_ticks;
            if (t.first_contact_tick < 0) {
                t.first_contact_tick  = k;
                t.first_contact_pairs = scene.contactPairs();
            }
        }
        t.max_ncon = std::max(t.max_ncon, row.ncon);
        if (t.first_contact_tick >= 0 && t.first_detect_tick < 0) {
            for (const int idx : scored.ownedJointInds) {
                if (std::fabs(row.raw[static_cast<std::size_t>(idx)]) > kDetectThreshold) {
                    t.first_detect_tick = k;
                    break;
                }
            }
        }
        for (std::size_t i = 0; i < 3; ++i) {
            if (std::fabs(row.base[i]) > std::fabs(t.base_peak[i])) {
                t.base_peak[i] = row.base[i];
            }
        }
        // (f2) Drag mode: the truth here is analytic rather than read back.
        //      MuJoCo applies -damping * velocity on each planar DOF, and the
        //      controller's model has no damping at all, so that product IS the
        //      unmodelled force the observer should be reporting on its base
        //      rows. Recorded before the step, like everything else, so the
        //      velocity used is the one the estimate was computed from.
        if (dragMode(o)) {
            const auto vb = scene.baseVelocity();
            const auto db = scene.baseDamping();
            t.drag_truth.push_back({{-db[0] * vb[0], -db[1] * vb[1], -db[2] * vb[2]}});
            t.drag_est.push_back(row.base);
        }
        t.rows.push_back(std::move(row));

        // (g) Advance. The torque was applied at (c2), before the truth was
        //     read, so this only runs the physics forward one control period.
        scene.step(steps_per_tick);
    }

    const auto base_end = scene.basePose();
    t.base_drift      = std::hypot(base_end[0] - base0[0], base_end[1] - base0[1]);
    t.obs_steps       = obs.updates();
    t.obs_max_mdot    = obs.maxMdot();
    t.ff_saturated    = scored.ffSaturated();
    t.ff_slew_limited = scored.ffSlewLimited();
    t.tier1_entries   = scored.tier1Entries();
    t.safe_ticks      = scored.safeModeTicks();
    t.ref_outside     = scored.referenceSamplesOutside();
    ArmProbe::ProfileSnapshot prof;
    scored.snapshotProfile(prof);
    t.qp_iter_mean = prof.qp_iters.mean();
    t.qp_iter_max  = prof.qp_iters.max;
    return t;
}

// ============================== Reporting ================================= //

/*! Drag mode's report: estimate against the analytic -damping*velocity.
 *
 *  Scored over the second half of the run, so the filter's initial transient is
 *  not counted as error -- the estimate starts at exactly zero by construction
 *  and needs a few time constants to catch up. */
void reportDrag(const Trial& t) {
    const std::size_t n = t.drag_truth.size();
    if (n < 4) {
        std::printf("  too few ticks to score drag\n");
        return;
    }
    const std::size_t from = n / 2;
    const char* axis[3] = {"base fx", "base fy", "base tz"};

    std::printf("\n  drag recovery, scored over the second half of the run "
                "(the filter starts at zero\n  and needs a few time constants "
                "to catch up):\n");
    std::printf("    %-9s %12s %12s %8s %10s\n",
                "axis", "truth mean", "est mean", "ratio", "rms err");
    for (int a = 0; a < 3; ++a) {
        double mt = 0.0, me = 0.0, sq = 0.0;
        for (std::size_t i = from; i < n; ++i) {
            mt += t.drag_truth[i][static_cast<std::size_t>(a)];
            me += t.drag_est[i][static_cast<std::size_t>(a)];
        }
        const double cnt = static_cast<double>(n - from);
        mt /= cnt;
        me /= cnt;
        for (std::size_t i = from; i < n; ++i) {
            const double e = t.drag_truth[i][static_cast<std::size_t>(a)] -
                             t.drag_est[i][static_cast<std::size_t>(a)];
            sq += e * e;
        }
        std::printf("    %-9s %12.5f %12.5f %8.2f %10.5f\n", axis[a], mt, me,
                    mt != 0.0 ? me / mt : std::nan(""), std::sqrt(sq / cnt));
    }
    std::printf("  observer: %ld steps, max|Mdot| %.2f\n",
                t.obs_steps, t.obs_max_mdot);
}

/*! The window the steady-state scorers use: everything past the first quarter
 *  of the run.
 *
 *  The filter starts at exactly zero and needs a few time constants to arrive,
 *  and the arm starts from rest and needs a cycle of the setpoint sine before it
 *  is accelerating the way the rest of the run does. Counting either as error
 *  would understate a correct estimate. */
std::size_t steadyFrom(const Trial& t) { return t.rows.size() / 4; }

/*! EE tracking error over the steady window: mean, peak and RMS. */
struct Tracking {
    double mean = 0.0;
    double peak = 0.0;
    double rms  = 0.0;
};

Tracking trackingOf(const Trial& t) {
    Tracking k;
    const std::size_t from = steadyFrom(t);
    if (from >= t.rows.size()) return k;
    double sq = 0.0;
    for (std::size_t i = from; i < t.rows.size(); ++i) {
        const double e = t.rows[i].ee_err;
        k.mean += e;
        sq     += e * e;
        k.peak = std::max(k.peak, e);
    }
    const double n = static_cast<double>(t.rows.size() - from);
    k.mean /= n;
    k.rms = std::sqrt(sq / n);
    return k;
}

/*! Model-error mode's report: the estimate against the mismatch torque.
 *
 *  The truth here is the acceleration residual rather than qfrc_constraint,
 *  because nothing is touching: the disturbance is the controller's own model
 *  being wrong about how much the arm weighs. */
void reportTracking(const Options& o, const Trial& t) {
    if (hasModelError(o)) {
        std::printf("\n  scene arm mass %.3f kg at %.2fx, so the controller's "
                    "model is short by %.3f kg\n", t.scene_arm_mass,
                    o.mass_scale,
                    t.scene_arm_mass - t.scene_arm_mass / o.mass_scale);
    } else {
        std::printf("\n  arm mass %.3f kg, UNPERTURBED: this is the baseline "
                    "run, so the tracking error\n  below is what the arm does "
                    "with a correct model and nothing disturbing it. The\n"
                    "  disturbance table is numerical dust by construction and "
                    "its ratios read n/a.\n", t.scene_arm_mass);
    }
    std::printf("  EE setpoint oscillating %.3f m at %.2f Hz, peak commanded "
                "accel %.2f m/s^2\n", o.osc_amp, o.osc_hz,
                o.osc_amp * std::pow(2.0 * M_PI * o.osc_hz, 2.0));
    std::printf("  base damping %.3f (the table's own value, not the pinning "
                "500), drifted %.3f m\n", t.base_damping, t.base_drift);

    // Nothing can touch in this mode -- the target is parked metres away -- so
    // a constraint force here is the arm against its own JOINT RANGE LIMIT,
    // which MuJoCo puts in the same qfrc_constraint vector. It is impulsive, it
    // dwarfs the smooth mismatch torque, and an observer lags it exactly as it
    // lags contact. Left unsaid it would quietly become the measurement, so it
    // is reported as contamination and the amplitude it needs is named.
    int limit_ticks = 0;
    for (const auto& r : t.rows) {
        if (r.ncon > 0) ++limit_ticks;
    }
    if (limit_ticks > 0) {
        std::printf("\n  WARNING: %d ticks hit a JOINT RANGE LIMIT. Nothing can "
                    "touch in this mode, so that\n  is the arm against its own "
                    "stop -- an impulsive force in the same qfrc_constraint\n"
                    "  vector, large enough to dominate the smooth mismatch "
                    "torque being measured.\n  Reduce --osc below the arm's "
                    "reachable travel.\n", limit_ticks);
    }

    const std::size_t from = steadyFrom(t);
    std::printf("\n  estimate against M*qacc + bias - tau, the mismatch torque, "
                "from %.2f s on\n  (the arm starts at rest and the filter starts "
                "at zero; neither is error).\n  Scored by AMPLITUDE, not by "
                "signed peak: this disturbance oscillates with the\n  setpoint, "
                "and two signed oscillations can peak on opposite half-cycles:\n",
                t.rows[from].t);
    std::printf("    %-10s %10s %10s %7s %7s %8s %9s %8s\n",
                "joint", "truth rms", "est rms", "ratio", "corr", "rms err",
                "ff cap", "tau max");
    bool over_cap = false;
    for (const int idx : t.scored_joints) {
        const JointScore s =
            pat_arm_control::bench::scoreJointResidual(t.rows, idx, from);
        const auto u = static_cast<std::size_t>(idx);
        std::printf("    %-10s %10.4f %10.4f %7.2f %7.2f %8.4f %9.4f %8.4f\n",
                    t.joint_names[u].c_str(), s.truth_rms, s.est_rms,
                    s.amplitudeRatio(), s.corr, s.rms_err, t.ff_cap[u],
                    t.tau_max[u]);
        if (s.truth_rms > t.ff_cap[u]) over_cap = true;
    }
    if (over_cap) {
        std::printf("\n  The disturbance is larger than the feedforward cap "
                    "(ff_max_frac * tau max), so the\n  command cannot cancel "
                    "it however well it is estimated. That is a statement about\n"
                    "  the actuators, not the observer: a %.2f N*m joint limit "
                    "leaves %.3f N*m of\n  feedforward authority. Reduce "
                    "--mass-scale to measure the feedforward itself.\n",
                    t.tau_max[static_cast<std::size_t>(t.scored_joints.front())],
                    t.ff_cap[static_cast<std::size_t>(t.scored_joints.front())]);
    }
    const Tracking k = trackingOf(t);
    std::printf("\n  EE tracking over the same window: mean %.5f  peak %.5f  "
                "rms %.5f m\n", k.mean, k.peak, k.rms);
    std::printf("  commanded torque:");
    for (const int idx : t.scored_joints) {
        const auto u = static_cast<std::size_t>(idx);
        std::printf("  %s peak %.4f (%ld ticks at the rail)",
                    t.joint_names[u].c_str(), t.tau_peak[u], t.tau_railed[u]);
    }
    std::printf("\n");
    std::printf("  observer: %ld steps, max|Mdot| %.2f;  feedforward gain %.2f, "
                "saturated %ld, slew-limited %ld\n",
                t.obs_steps, t.obs_max_mdot, t.applied_ff, t.ff_saturated,
                t.ff_slew_limited);
}

void reportTrajectory(const Options& o, const Trial& t);   // below, with its scoring

void reportTrial(const Options& o, const Trial& t) {
    if (trajMode(o)) {
        reportTrajectory(o, t);
        return;
    }
    if (dragMode(o)) {
        reportDrag(t);
        return;
    }
    if (trackingMode(o)) {
        reportTracking(o, t);
        return;
    }
    std::printf("  contact on %d of %zu ticks, peak ncon %d\n",
                t.contact_ticks, t.rows.size(), t.max_ncon);
    if (t.contact_ticks == 0) {
        std::printf("  NO CONTACT -- nothing to score. The arm never reached the "
                    "target: check --setpoint against --target.\n");
        return;
    }
    // Naming the geoms matters: the loaded DOFs alone cannot tell an
    // arm-vs-target contact from the two arms touching each other.
    std::printf("  first contact: %s\n", t.first_contact_pairs.c_str());
    if (t.first_detect_tick >= 0) {
        const int lat = t.first_detect_tick - t.first_contact_tick;
        std::printf("  detection latency %d ticks (%.0f ms) above %.3f N*m\n",
                    lat, 1000.0 * lat / o.hz, kDetectThreshold);
    } else {
        std::printf("  the estimate never exceeded %.3f N*m on an owned joint: "
                    "contact was NOT detected\n", kDetectThreshold);
    }

    std::printf("\n  scored over the contact window only -- a run is mostly quiet,\n"
                "  and averaging the quiet part measures the error floor, not the "
                "contact:\n");
    std::printf("    %-10s %10s %10s %7s %10s %9s %7s %8s\n",
                "joint", "truth pk", "est pk", "ratio", "truth avg", "est avg",
                "corr", "rms err");
    for (const int idx : t.scored_joints) {
        const JointScore s = pat_arm_control::bench::scoreJoint(t.rows, idx);
        std::printf("    %-10s %10.4f %10.4f %7.2f %10.4f %9.4f %7.2f %8.4f\n",
                    t.joint_names[static_cast<std::size_t>(idx)].c_str(),
                    s.truth_peak, s.est_peak, s.ratio(), s.truth_mean,
                    s.est_mean, s.corr, s.rms_err);
    }
    // Both truth channels apply during contact, and they are computed from
    // nothing in common: one is MuJoCo's own constraint force, the other is the
    // controller's model evaluated at the plant's acceleration. Agreement means
    // the acceleration residual that the model-error mode relies on -- where
    // qfrc_constraint is identically zero and cannot check anything -- is
    // sound. Disagreement on a contact run would mean it is not.
    double worst = 0.0, scale = 0.0;
    for (const auto& r : t.rows) {
        if (r.ncon <= 0) continue;
        for (const int idx : t.scored_joints) {
            const auto u = static_cast<std::size_t>(idx);
            worst = std::max(worst, std::fabs(r.truth[u] - r.resid[u]));
            scale = std::max(scale, std::fabs(r.truth[u]));
        }
    }
    std::printf("\n  truth channels cross-checked over the contact window: "
                "qfrc_constraint against\n  M*qacc + bias - tau differ by at "
                "most %.4f N*m on a %.4f N*m signal (%.2f%%)\n",
                worst, scale, scale != 0.0 ? 100.0 * worst / scale : 0.0);

    std::printf("\n  base wrench estimate peaks: fx %.3f  fy %.3f  tz %.3f\n"
                "  (the pinning drag is in these: the scene damps the base and "
                "the controller's model does\n   not, so they are NOT a contact "
                "measurement)\n",
                t.base_peak[0], t.base_peak[1], t.base_peak[2]);
    std::printf("  observer: %ld steps, max|Mdot| %.2f\n",
                t.obs_steps, t.obs_max_mdot);
}

void writeCsv(const std::string& path, const Trial& t) {
    std::ofstream os(path);
    if (!os) throw std::runtime_error("cannot write " + path);
    // One row per control tick; three columns per scored joint plus the base.
    os << "t_s,ncon,ee_err";
    for (const int idx : t.scored_joints) {
        const auto& n = t.joint_names[static_cast<std::size_t>(idx)];
        os << ",truth_" << n << ",resid_" << n << ",raw_" << n
           << ",reduced_" << n;
    }
    os << ",base_fx,base_fy,base_tz\n";
    for (const auto& r : t.rows) {
        os << r.t << ',' << r.ncon << ',' << r.ee_err;
        for (const int idx : t.scored_joints) {
            const auto u = static_cast<std::size_t>(idx);
            os << ',' << r.truth[u] << ',' << r.resid[u] << ',' << r.raw[u]
               << ',' << r.reduced[u];
        }
        os << ',' << r.base[0] << ',' << r.base[1] << ',' << r.base[2] << '\n';
    }
    std::printf("\n  wrote %zu rows to %s\n", t.rows.size(), path.c_str());
}

/*! The A/B: does acting on the estimate actually improve tracking?
 *
 *  Two runs of one scenario, differing only in ff_gain. That is a fair
 *  comparison for the same reason the gain sweep is -- the scenario is
 *  deterministic, so the arm, the setpoint and the disturbance are identical in
 *  both runs and the difference is the feedforward alone.
 */
void reportAb(const Options& o, const Trial& off, const Trial& on) {
    const Tracking a = trackingOf(off);
    const Tracking b = trackingOf(on);
    std::printf("\nfeedforward A/B -- identical runs but for ff_gain, scored over "
                "the steady window:\n");
    std::printf("  %-14s %12s %12s %12s\n", "ff_gain", "mean err", "peak err",
                "rms err");
    std::printf("  %-14.2f %12.5f %12.5f %12.5f\n", off.applied_ff, a.mean,
                a.peak, a.rms);
    std::printf("  %-14.2f %12.5f %12.5f %12.5f\n", on.applied_ff, b.mean,
                b.peak, b.rms);
    const auto pct = [](double before, double after) {
        return before != 0.0 ? 100.0 * (before - after) / before : 0.0;
    };
    std::printf("  %-14s %11.1f%% %11.1f%% %11.1f%%   (positive = the "
                "feedforward helped)\n", "improvement",
                pct(a.mean, b.mean), pct(a.peak, b.peak), pct(a.rms, b.rms));
    if (on.ff_saturated > 0 || on.ff_slew_limited > 0) {
        std::printf("  the feedforward was limited on the second run: saturated "
                    "%ld, slew-limited %ld\n  (so this understates what an "
                    "unguarded estimate would have done, which is the point of "
                    "the guards)\n", on.ff_saturated, on.ff_slew_limited);
    }
    if (!hasModelError(o) && !dragMode(o)) {
        std::printf("  note: with no model error and nothing touching, there is "
                    "almost no disturbance to\n  feed forward. This row is the "
                    "BASELINE -- the tracking error the arm has anyway -- and a\n"
                    "  result near zero here is the correct answer, not a weak "
                    "one\n");
    }
}

// ========================= Trajectory scoring ============================= //

/*! How a trajectory run went, over its settled part.
 *
 *  Plain tracking error does not separate the two things the preview changes,
 *  so it is broken down by shape:
 *
 *    along / cross   the error resolved along the reference's direction of
 *                    travel and across it. A negative mean along-track error
 *                    is the EE trailing the reference: lag, which a reactive
 *                    controller always has and a previewing one need not.
 *    radial          circle only: distance from the centre minus the radius,
 *                    negative inside the circle.
 *    corner / edge   rectangle only: the error within kCornerWindow of a
 *                    corner, and away from every corner.
 *    overshoot       rectangle only: furthest the EE got OUTSIDE the
 *                    rectangle. Turning late carries the EE past a corner.
 *    cut             rectangle only: deepest the EE got INSIDE the rectangle
 *                    near a corner. Turning early cuts the corner -- the
 *                    trade a controller that sees the corner coming makes.
 */
struct TrajScore {
    int    n = 0;
    double from = 0.0;                              //!< scoring starts here [s]
    double err_mean = 0.0, err_rms = 0.0, err_peak = 0.0;
    double along_mean = 0.0, cross_rms = 0.0;
    double radial_mean = 0.0, radial_rms = 0.0;     //!< circle
    double corner_rms = 0.0, edge_rms = 0.0;        //!< rectangle
    double overshoot = 0.0, cut = 0.0;              //!< rectangle
    double inside_frac = 0.0;                       //!< rectangle, near corners
};

//! The half-width of the window around each corner passage [s].
constexpr double kCornerWindow = 0.25;

TrajScore scoreTrajectory(const Options& o, const Trial& t) {
    using pat_arm_control::CircleTrajectory;
    using pat_arm_control::RectangleTrajectory;
    TrajScore s;
    const auto& shape = *t.traj_shape;
    // After the approach, the circle's ramp, and a second to settle: the
    // transient onto the shape is not what is being compared.
    s.from = shape.approachDuration() + 2.0;
    const pat_arm_control::TrajectorySpec spec = trajSpec(o);

    // Rectangle geometry: bounds, and the times of corner passages.
    const auto* rect = dynamic_cast<const RectangleTrajectory*>(&shape);
    double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
    std::vector<double> corner_t;   // within one lap, measured from the shape start
    if (rect) {
        double acc = 0.0;
        const double speed = spec.rectangle.speed;
        for (int k = 0; k < 4; ++k) {
            double x0, y0, x1, y1;
            rect->corner(k, x0, y0);
            rect->corner(k + 1, x1, y1);
            xmin = std::min(xmin, x0); xmax = std::max(xmax, x0);
            ymin = std::min(ymin, y0); ymax = std::max(ymax, y0);
            corner_t.push_back(acc);
            acc += std::hypot(x1 - x0, y1 - y0) / speed;
        }
    }
    const double cx = shape.anchor().x + spec.circle.center_dx;
    const double cy = shape.anchor().y + spec.circle.center_dy;

    double sq = 0.0, along = 0.0, cross_sq = 0.0, rad = 0.0, rad_sq = 0.0;
    double corner_sq = 0.0, edge_sq = 0.0;
    int n_corner = 0, n_edge = 0, n_inside = 0;
    for (const auto& r : t.traj) {
        if (r.t < s.from) continue;
        const double ex = r.ex - r.rx, ey = r.ey - r.ry;
        const double e = std::hypot(ex, ey);
        ++s.n;
        s.err_mean += e;
        sq += e * e;
        s.err_peak = std::max(s.err_peak, e);
        const double sp = std::hypot(r.rvx, r.rvy);
        if (sp > 1e-9) {
            const double ux = r.rvx / sp, uy = r.rvy / sp;
            along += ex * ux + ey * uy;
            const double c = -ex * uy + ey * ux;
            cross_sq += c * c;
        }
        if (!rect) {
            const double radial = std::hypot(r.ex - cx, r.ey - cy) - spec.circle.radius;
            rad += radial;
            rad_sq += radial * radial;
            continue;
        }
        // Signed distance to the rectangle's boundary: positive outside.
        const double dx = std::max(xmin - r.ex, r.ex - xmax);
        const double dy = std::max(ymin - r.ey, r.ey - ymax);
        const double signed_d = (dx > 0.0 || dy > 0.0)
                                    ? std::hypot(std::max(dx, 0.0), std::max(dy, 0.0))
                                    : std::max(dx, dy);
        s.overshoot = std::max(s.overshoot, signed_d);
        // Time to the nearest corner passage, lap-wrapped.
        const double lap = shape.loopPeriod();
        const double phase = std::fmod(r.t - shape.approachDuration(), lap);
        double near = 1e300;
        for (const double tc : corner_t) {
            const double d = std::fabs(phase - tc);
            near = std::min({near, d, lap - d});
        }
        if (near < kCornerWindow) {
            corner_sq += e * e;
            ++n_corner;
            if (signed_d < 0.0) {
                ++n_inside;
                s.cut = std::max(s.cut, -signed_d);
            }
        } else {
            edge_sq += e * e;
            ++n_edge;
        }
    }
    if (s.n == 0) return s;
    const double n = static_cast<double>(s.n);
    s.err_mean /= n;
    s.err_rms = std::sqrt(sq / n);
    s.along_mean = along / n;
    s.cross_rms = std::sqrt(cross_sq / n);
    s.radial_mean = rad / n;
    s.radial_rms = std::sqrt(rad_sq / n);
    s.corner_rms = n_corner ? std::sqrt(corner_sq / n_corner) : 0.0;
    s.edge_rms = n_edge ? std::sqrt(edge_sq / n_edge) : 0.0;
    s.inside_frac = n_corner ? static_cast<double>(n_inside) / n_corner : 0.0;
    return s;
}

//! The scored arm's peak torque and ticks at the rail, summed over its joints.
void torqueUse(const Trial& t, double& peak, long& railed) {
    peak = 0.0;
    railed = 0;
    for (const int idx : t.scored_joints) {
        const auto u = static_cast<std::size_t>(idx);
        peak = std::max(peak, t.tau_peak[u]);
        railed += t.tau_railed[u];
    }
}

void reportTrajectory(const Options& o, const Trial& t) {
    const TrajScore s = scoreTrajectory(o, t);
    const bool rect = o.trajectory == "rectangle";
    const auto spec = trajSpec(o);
    char shape[96];
    if (rect) {
        std::snprintf(shape, sizeof(shape), "%.0f x %.0f cm at %.1f cm/s",
                      spec.rectangle.width * 100, spec.rectangle.height * 100,
                      spec.rectangle.speed * 100);
    } else {
        std::snprintf(shape, sizeof(shape), "radius %.0f cm, period %.2f s",
                      spec.circle.radius * 100, spec.circle.period);
    }
    std::printf("\n  %s, %s, scored from %.1f s (%d ticks); reference fed as: %s\n",
                o.trajectory.c_str(), shape, s.from, s.n, t.ref_mode.c_str());
    std::printf("    tracking error      mean %6.2f   rms %6.2f   peak %6.2f mm\n",
                s.err_mean * 1e3, s.err_rms * 1e3, s.err_peak * 1e3);
    std::printf("    along-track mean %+7.2f mm (negative = trailing)   cross-track rms "
                "%6.2f mm\n", s.along_mean * 1e3, s.cross_rms * 1e3);
    if (rect) {
        std::printf("    near corners rms %6.2f   on edges rms %6.2f mm\n",
                    s.corner_rms * 1e3, s.edge_rms * 1e3);
        std::printf("    overshoot outside %6.2f mm   cut inside %6.2f mm   (%.0f%% of "
                    "near-corner ticks inside)\n",
                    s.overshoot * 1e3, s.cut * 1e3, 100.0 * s.inside_frac);
    } else {
        std::printf("    radial mean %+6.2f mm (negative = inside the circle)   rms "
                    "%6.2f mm\n", s.radial_mean * 1e3, s.radial_rms * 1e3);
    }
    double peak = 0.0;
    long railed = 0;
    torqueUse(t, peak, railed);
    if (t.has_solver) {
        std::printf("    torque peak %.3f N*m, %ld joint-ticks at the rail | QP iters mean "
                    "%.1f max %.0f | Tier 1 entries %ld, safe-mode ticks %ld | refs held "
                    "past the window %ld\n",
                    peak, railed, t.qp_iter_mean, t.qp_iter_max, t.tier1_entries,
                    t.safe_ticks, t.ref_outside);
    } else {
        std::printf("    [%s] torque peak %.3f N*m, %ld joint-ticks at the rail | "
                    "safe-mode ticks %ld | refs held past the window %ld\n",
                    t.law.c_str(), peak, railed, t.safe_ticks, t.ref_outside);
    }
}

/*! The three ways of feeding the same trajectory, one row each: preview, the
 *  window without preview, and the current position streamed as a setpoint. */
void reportPreviewAb(const Options& o, const std::vector<Trial>& runs) {
    const bool rect = o.trajectory == "rectangle";
    std::printf("\nidentical runs but for how the reference reaches the controller:\n");
    std::printf("  %-11s %9s %9s %9s %10s", "fed as", "err mean", "err rms", "err peak",
                "along");
    if (rect) std::printf(" %10s %9s %9s", "corner rms", "overshoot", "cut");
    else      std::printf(" %10s", "radial");
    std::printf(" %7s %6s\n", "rail", "tier1");
    for (const Trial& t : runs) {
        const TrajScore s = scoreTrajectory(o, t);
        double peak = 0.0;
        long railed = 0;
        torqueUse(t, peak, railed);
        std::printf("  %-11s %9.2f %9.2f %9.2f %+10.2f", t.ref_mode.c_str(),
                    s.err_mean * 1e3, s.err_rms * 1e3, s.err_peak * 1e3,
                    s.along_mean * 1e3);
        if (rect) std::printf(" %10.2f %9.2f %9.2f", s.corner_rms * 1e3,
                              s.overshoot * 1e3, s.cut * 1e3);
        else      std::printf(" %+10.2f", s.radial_mean * 1e3);
        std::printf(" %7ld %6ld\n", railed, t.tier1_entries);
    }
    std::printf("  (mm; along negative = trailing; rail = joint-ticks at the torque "
                "limit)\n");
}

/*! Run length for a trajectory run: the approach, two laps of the shape and a
 *  second spare -- unless --seconds was given. Per speed, since a faster shape
 *  has a shorter lap. */
Options withTrajSeconds(Options o) {
    if (o.seconds_set) return o;
    const auto shape = pat_arm_control::makeTrajectory(trajSpec(o),
                                                       pat_arm_control::Anchor{});
    o.seconds = shape->approachDuration() + 2.0 * shape->loopPeriod() + 1.0;
    return o;
}

/*! Preview on and off at each speed factor, one table. The accuracy-for-
 *  feasibility trade read down the columns: as the shape speeds up past what
 *  the torque limits allow, the error grows -- the question is how, and
 *  whether the solver ever fails. */
void reportSpeedSweep(const Options& o, const std::vector<double>& speeds,
                      const std::vector<std::vector<Trial>>& runs) {
    const bool rect = o.trajectory == "rectangle";
    std::printf("\n%s speed sweep (mm; rail = joint-ticks at the torque limit):\n",
                o.trajectory.c_str());
    std::printf("  %6s  %-11s %8s %9s %9s %9s %7s %6s\n", "speed", "fed as", "err rms",
                "along", rect ? "crnr rms" : "radial", rect ? "overshoot" : "",
                "rail", "tier1");
    for (std::size_t i = 0; i < speeds.size(); ++i) {
        Options oi = o;
        oi.traj_speed = speeds[i];
        for (const Trial& t : runs[i]) {
            const TrajScore s = scoreTrajectory(oi, t);
            double pk = 0.0;
            long railed = 0;
            torqueUse(t, pk, railed);
            std::printf("  %5.2fx  %-11s %8.2f %+9.2f %+9.2f ", speeds[i],
                        t.ref_mode.c_str(), s.err_rms * 1e3, s.along_mean * 1e3,
                        (rect ? s.corner_rms : s.radial_mean) * 1e3);
            if (rect) std::printf("%9.2f", s.overshoot * 1e3);
            else      std::printf("%9s", "");
            std::printf(" %7ld %6ld\n", railed, t.tier1_entries);
        }
    }
}

void writeTrajCsv(const std::string& path, const Trial& t) {
    std::ofstream os(path);
    if (!os) throw std::runtime_error("cannot write " + path);
    os << "t_s,ref_x,ref_y,ref_vx,ref_vy,ee_x,ee_y,err\n";
    for (const auto& r : t.traj) {
        os << r.t << ',' << r.rx << ',' << r.ry << ',' << r.rvx << ',' << r.rvy << ','
           << r.ex << ',' << r.ey << ',' << std::hypot(r.ex - r.rx, r.ey - r.ry) << '\n';
    }
    std::printf("\n  wrote %zu rows to %s\n", t.traj.size(), path.c_str());
}

/*! One table across the sweep, so the gain's effect is read down a column. */
void reportSweep(const Options& o, const std::vector<Trial>& trials) {
    std::printf("\ngain sweep -- valid because the scenario is deterministic, so "
                "the contact event is\nidentical across rows and any difference "
                "is the gain and nothing else:\n");
    std::printf("  %8s %10s  %-10s %10s %10s %7s %7s\n",
                "gain", "latency", "joint", "truth pk", "est pk", "ratio", "corr");
    for (const auto& t : trials) {
        if (t.contact_ticks == 0) {
            std::printf("  %8.1f  (no contact)\n", t.applied_gain);
            continue;
        }
        const int lat = t.first_detect_tick >= 0
                            ? t.first_detect_tick - t.first_contact_tick : -1;
        bool first = true;   // gain and latency print once per trial, not per joint
        for (const int idx : t.scored_joints) {
            const JointScore s = pat_arm_control::bench::scoreJoint(t.rows, idx);
            if (first) {
                std::printf("  %8.1f %7d ms  ", t.applied_gain,
                            lat >= 0 ? static_cast<int>(1000.0 * lat / o.hz) : -1);
            } else {
                std::printf("  %8s %10s  ", "", "");
            }
            std::printf("%-10s %10.4f %10.4f %7.2f %7.2f\n",
                        t.joint_names[static_cast<std::size_t>(idx)].c_str(),
                        s.truth_peak, s.est_peak, s.ratio(), s.corr);
            first = false;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    int code = 0;
    if (!parseArgs(argc, argv, o, code)) return code;

    try {
        if (trajMode(o)) {
            // The law is named only when it is not the default, so the NMPC's
            // output is exactly what it was before there was a choice.
            std::printf("contact_bench: mode=trajectory (%s)%s, %.0f Hz, arm=%s; target "
                        "parked out of reach\n  scene %s\n",
                        o.trajectory.c_str(),
                        o.law == "nmpc" ? "" : (", law=" + o.law).c_str(),
                        o.hz, o.arm.c_str(), o.scene.c_str());
            // Each set below differs ONLY in how the reference reaches the
            // controller: the scenario is deterministic, so the differences are
            // that and nothing else.
            const auto three = [](Options base) {
                base = withTrajSeconds(base);
                std::vector<Trial> runs;
                base.no_preview = false;  base.stream_setpoint = false;
                runs.push_back(runTrial(base, base.gain));
                base.no_preview = true;
                runs.push_back(runTrial(base, base.gain));
                base.no_preview = false;  base.stream_setpoint = true;
                runs.push_back(runTrial(base, base.gain));
                return runs;
            };
            if (!o.sweep_speed.empty()) {
                std::vector<std::vector<Trial>> runs;
                for (const double f : o.sweep_speed) {
                    Options oi = o;
                    oi.traj_speed = f;
                    runs.push_back(three(oi));
                }
                reportSpeedSweep(o, o.sweep_speed, runs);
                return 0;
            }
            if (o.ab_preview) {
                const std::vector<Trial> runs = three(o);
                const Options ot = withTrajSeconds(o);
                for (const Trial& t : runs) reportTrial(ot, t);
                reportPreviewAb(ot, runs);
                if (!o.out_csv.empty()) writeTrajCsv(o.out_csv, runs.front());
                return 0;
            }
            const Options ot = withTrajSeconds(o);
            const Trial t = runTrial(ot, ot.gain);
            reportTrial(ot, t);
            if (!o.out_csv.empty()) writeTrajCsv(o.out_csv, t);
            return 0;
        }

        // The scenario, echoed up front: these numbers decide whether contact
        // happens at all, so a run is not interpretable without them.
        const char* mode = dragMode(o) ? "drag"
                         : trackingMode(o)
                               ? (hasModelError(o) ? "tracking, model error"
                                                   : "tracking, baseline")
                               : "contact";
        std::printf("contact_bench: mode=%s, %.1f s at %.0f Hz, arm=%s\n",
                    mode, o.seconds, o.hz, o.arm.c_str());
        if (quietMode(o)) {
            // The target is parked out of reach in these modes, so echoing a
            // --target would describe something the run does not use.
            std::printf("  target parked out of reach; setpoint %s\n",
                        o.osc_amp > 0.0 ? "oscillating about the start pose"
                                        : "held at the start pose");
        } else {
            std::printf("  target=(%.4f, %.4f), setpoint=(%.3f, %.3f), "
                        "target_%s\n", o.tgt_x, o.tgt_y, o.sp_x, o.sp_y,
                        o.free_target ? "free" : "pinned");
        }
        std::printf("  scene %s\n", o.scene.c_str());

        if (o.ab_ff >= 0.0) {
            // The feedforward gain is the ONLY difference between the two.
            Options off = o;
            off.ff_gain = 0.0;
            Options on = o;
            on.ff_gain = o.ab_ff;
            const Trial t_off = runTrial(off, o.gain);
            const Trial t_on  = runTrial(on, o.gain);
            // The estimate's own scoring once, from the run that acted on it.
            reportTrial(on, t_on);
            reportAb(o, t_off, t_on);
            if (!o.out_csv.empty()) writeCsv(o.out_csv, t_on);
            return 0;
        }

        if (!o.sweep_gain.empty()) {
            std::vector<Trial> trials;
            for (const double g : o.sweep_gain) trials.push_back(runTrial(o, g));
            reportSweep(o, trials);
            // Nonzero exit when the scenario produced nothing to measure, so a
            // script cannot mistake an empty run for a passing one.
            return trials.front().contact_ticks == 0 ? 1 : 0;
        }

        const Trial t = runTrial(o, o.gain);
        std::printf("  observer.gain=%.1f ff_gain=%.2f\n", t.applied_gain,
                    o.ff_gain >= 0.0 ? o.ff_gain : 0.0);
        reportTrial(o, t);
        if (!o.out_csv.empty()) writeCsv(o.out_csv, t);
        // The drag and model-error modes have no contact by design, so "no
        // contact" is not a failure there -- it is the point.
        if (quietMode(o)) return 0;
        return t.contact_ticks == 0 ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "contact_bench: " << e.what() << "\n";
        return 1;
    }
}
