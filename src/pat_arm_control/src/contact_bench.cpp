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
 *  ================================ USE =================================
 *
 *    contact_bench
 *    contact_bench --gain 60 --out /tmp/contact.csv
 *    contact_bench --sweep-gain 20,60,150
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pat_arm_control/control/arm_nmpc.h"
#include "pat_arm_control/observers/momentum_observer.h"
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

/*! Exposes which joints a controller owns.
 *
 *  ownedJointInds is protected because nothing outside the control law needs
 *  it; this bench does, to work out which arm commands which joint. Same
 *  pattern and the same reason as nmpc_bench's BenchProbe.
 */
class ArmProbe : public ArmNMPC {
public:
    ArmProbe(const std::string& mjcf, const NMPCParams& p) : ArmNMPC(mjcf, p) {}
    using ArmNMPC::n_joints_;
    using ArmNMPC::ownedJointInds;
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
    std::string params = PAT_NMPC_BENCH_PARAMS;    //!< nmpc.yaml
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
};

//! True when --drag asked for the drag measurement rather than contact.
bool dragMode(const Options& o) { return o.drag_damping > 0.0; }

void usage() {
    std::cout <<
R"(contact_bench - deterministic scoring of the disturbance observer against
                MuJoCo's own contact force. No ROS, no DDS, no wall clock.

  --scene PATH       scene MJCF (default: the installed air_bearing_table.xml)
  --params PATH      nmpc.yaml (default: the installed one)
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
            else if (a == "--params")  o.params  = next();
            else if (a == "--model")   o.model   = next();
            else if (a == "--arm")     o.arm     = next();
            else if (a == "--out")     o.out_csv = next();
            else if (a == "--seconds") o.seconds = std::stod(next());
            else if (a == "--hz")      o.hz      = std::stod(next());
            else if (a == "--gain")    o.gain    = std::stod(next());
            else if (a == "--ff-gain") o.ff_gain = std::stod(next());
            else if (a == "--free-target") o.free_target = true;
            else if (a == "--drag")    o.drag_damping = std::stod(next());
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
    if (o.seconds <= 0.0 || o.hz <= 0.0) {
        std::cerr << "contact_bench: --seconds and --hz must be positive\n";
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
    "control_hz", "arm_side", "joint_names",
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
        // And the target well out of the way, so contact cannot contaminate a
        // measurement that is supposed to be drag alone.
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
    const double g = gain_override >= 0.0 ? gain_override : cfg.observer_gain;
    op.gain = Eigen::VectorXd::Constant(op.nv, g);
    // The base rows can take their own, slower gain: they are a different
    // physical quantity (force, not torque) on a noisier channel.
    if (cfg.observer_gain_base >= 0.0) op.gain.head(6).setConstant(cfg.observer_gain_base);
    op.mdot_clamp     = cfg.observer_mdot_clamp;
    op.estimate_clamp = cfg.observer_estimate_clamp;
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
    long   obs_steps = 0;
    double obs_max_mdot = 0.0;
    std::array<double, 3> base_peak{{0.0, 0.0, 0.0}};   //!< fx, fy, tz

    /*! Drag mode only: the analytic drag the scene applied, paired with what
     *  the observer's base rows estimated, per tick. Three entries each:
     *  x force, y force, yaw torque. */
    std::vector<std::array<double, 3>> drag_truth;
    std::vector<std::array<double, 3>> drag_est;
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
        o.params, o.arm, o.model, PAT_NMPC_BENCH_MODEL, kNotOurs, "contact_bench");
    const std::string other = (o.arm == "left") ? "right" : "left";
    auto cfg_other = pat_arm_control::bench::loadConfig(
        o.params, other, o.model, PAT_NMPC_BENCH_MODEL, kNotOurs, "contact_bench");
    if (o.ff_gain >= 0.0) cfg.params.ff_gain = o.ff_gain;

    const double dt = 1.0 / o.hz;
    // control_period drives the warm-start shift and the deadline accounting;
    // the node sets it from control_hz, so --hz has to do the same here.
    cfg.params.control_period       = dt;
    cfg_other.params.control_period = dt;

    // --- 2. The plant and the controllers --------------------------------- //
    ContactScenario scene(sceneConfig(o, cfg.model_joints));
    const int n_joints = scene.numJoints();

    ArmProbe scored(cfg.mjcf, cfg.params);
    ArmProbe held(cfg_other.mjcf, cfg_other.params);
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

    // --- 3. Setpoints ----------------------------------------------------- //
    scene.readControllerState(cfg.base_anchor_x, cfg.base_anchor_y, q, v);

    // The other arm holds exactly where it started, which is what the node does
    // when no setpoint has ever arrived for it. This is not cosmetic: left
    // limp, that arm drifts into a joint range limit, and MuJoCo puts
    // joint-limit forces in the SAME qfrc_constraint vector as contact forces.
    // The result is a convincing false contact in the truth signal.
    const ArmNMPC::EePose hold = held.currentEePose(q, v);
    held.setDesiredPos(hold.pos.x(), hold.pos.y(), hold.pos.z());
    held.setDesiredOrient(hold.quat(0), hold.quat(1), hold.quat(2), hold.quat(3));

    // The scored arm is driven to a fixed point past the target's face. Its
    // orientation setpoint is simply where it started, so the run is a straight
    // push rather than a push plus a twist.
    const ArmNMPC::EePose start = scored.currentEePose(q, v);
    scored.setDesiredPos(o.sp_x, o.sp_y, start.pos.z());
    scored.setDesiredOrient(start.quat(0), start.quat(1), start.quat(2), start.quat(3));

    Trial t;
    t.applied_gain  = op.gain(6);     // a joint row, past any base override
    t.joint_names   = cfg.model_joints;
    t.scored_joints = scored.ownedJointInds;

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
        scene.readControllerState(cfg.base_anchor_x, cfg.base_anchor_y, q, v);

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

        // (d) Step the observer. fullDynamics evaluates the CONTROLLER's model
        //     at this state -- the model with no contact in it, which is why
        //     the contact shows up as a disturbance at all.
        scored.fullDynamics(q, v, ws, M, bias, R);
        obs.update(M, bias, v, tau_full);
        obs.reduceToJoints(R, d_reduced);

        // (e) Record estimate and truth together, BEFORE advancing the plant,
        //     so both describe the same instant.
        scene.armConstraintForce(truth);
        ContactRow row;
        row.t    = scene.time();
        row.ncon = scene.contactCount();
        row.truth.resize(static_cast<std::size_t>(n_joints));
        row.raw.resize(static_cast<std::size_t>(n_joints));
        row.reduced.resize(static_cast<std::size_t>(n_joints));
        for (int i = 0; i < n_joints; ++i) {
            const auto u = static_cast<std::size_t>(i);
            row.truth[u]   = truth(i);
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

        // (g) Apply and advance. segment(6, n) is the joint block of the
        //     full-system torque vector.
        scene.applyArmTorque(tau_full.segment(6, n_joints));
        scene.step(steps_per_tick);
    }

    t.obs_steps    = obs.updates();
    t.obs_max_mdot = obs.maxMdot();
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

void reportTrial(const Options& o, const Trial& t) {
    if (dragMode(o)) {
        reportDrag(t);
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
    os << "t_s,ncon";
    for (const int idx : t.scored_joints) {
        const auto& n = t.joint_names[static_cast<std::size_t>(idx)];
        os << ",truth_" << n << ",raw_" << n << ",reduced_" << n;
    }
    os << ",base_fx,base_fy,base_tz\n";
    for (const auto& r : t.rows) {
        os << r.t << ',' << r.ncon;
        for (const int idx : t.scored_joints) {
            const auto u = static_cast<std::size_t>(idx);
            os << ',' << r.truth[u] << ',' << r.raw[u] << ',' << r.reduced[u];
        }
        os << ',' << r.base[0] << ',' << r.base[1] << ',' << r.base[2] << '\n';
    }
    std::printf("\n  wrote %zu rows to %s\n", t.rows.size(), path.c_str());
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
        // The scenario, echoed up front: these numbers decide whether contact
        // happens at all, so a run is not interpretable without them.
        std::printf("contact_bench: %.1f s at %.0f Hz, arm=%s, target=(%.4f, %.4f), "
                    "setpoint=(%.3f, %.3f), target_%s\n  scene %s\n",
                    o.seconds, o.hz, o.arm.c_str(), o.tgt_x, o.tgt_y,
                    o.sp_x, o.sp_y, o.free_target ? "free" : "pinned",
                    o.scene.c_str());

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
        // Drag mode has no contact by design, so "no contact" is not a failure
        // there -- it is the point.
        if (dragMode(o)) return 0;
        return t.contact_ticks == 0 ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "contact_bench: " << e.what() << "\n";
        return 1;
    }
}
