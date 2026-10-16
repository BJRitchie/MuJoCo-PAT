#pragma once
/*! A deterministic contact scenario: the scene MJCF as a plant, plus the
 *  bookkeeping needed to drive the controller's own model from it.
 *
 *  Separated from contact_bench's driver so the scenario is reusable -- a gtest
 *  can construct one and assert on it without going near argument parsing or
 *  report formatting.
 *
 *  Deterministic by construction: no ROS, no DDS, no wall clock, no RNG. Every
 *  run from the same Config produces the same trajectory, which is the whole
 *  point. The same measurement against the live stack varies ~2x run to run
 *  because the impact's peak depends on the arm's state when the setpoint
 *  arrives, and node startup shifts that by tens of milliseconds.
 */
#include <array>
#include <string>
#include <vector>

#include <Eigen/Dense>

struct mjModel_;
struct mjData_;
typedef struct mjModel_ mjModel;
typedef struct mjData_ mjData;

namespace pat_arm_control {
namespace bench {

/*! The scene as a plant.
 *
 *  Owns the scene's mjModel/mjData and resolves, once, every index the
 *  controller mapping needs. Indices come from jnt_dofadr/jnt_qposadr rather
 *  than being assumed: the scene's DOF order is chaser_x/y/yaw, then the arm
 *  joints, then target_x/y/yaw, and hardcoding "the arm block starts at row 3"
 *  is exactly the kind of assumption that silently produces a confident, wrong
 *  number.
 */
class ContactScenario {
public:
    struct Config {
        std::string scene_mjcf;
        //! ALL model joints, MJCF body-tree order (nmpc.yaml's model_joint_names).
        std::vector<std::string> model_joints;
        //! Starting arm pose, one per model joint. simulation.yaml's
        //! initial_arm_qpos: the MJCF's own all-zero fold pose self-collides at
        //! the real arm mount, so the model default would begin in contact.
        std::vector<double> arm_home;
        //! initial_target_qpos x/y, an offset from the target body's anchor.
        double target_x = -0.0175;
        double target_y = -0.42;
        /*! Pin the base through joint damping, as the collision study did.
         *
         *  A free base drifts under arm reaction and makes the world-frame
         *  setpoint unreachable within ~15 s, which confounds everything after
         *  that. Pinning is artificial -- the real table is free-floating -- so
         *  results describe contact and never flight behaviour. It also makes
         *  the base rows of any estimate meaningless as a contact measurement,
         *  because the scene damps the base and the controller's model does
         *  not, so the drag appears as a disturbance. */
        std::array<double, 3> base_damping{{500.0, 500.0, 200.0}};
        //! Pin the target too, so contact is sustained and scorable instead of
        //! a transient that knocks a 20 kg body away. Zeros leave it free.
        std::array<double, 3> target_damping{{500.0, 500.0, 200.0}};
    };

    explicit ContactScenario(const Config& cfg);
    ~ContactScenario();
    ContactScenario(const ContactScenario&) = delete;
    ContactScenario& operator=(const ContactScenario&) = delete;

    //! Back to the configured starting state.
    void reset();

    //! Set the arm actuators from a per-model-joint torque vector (n_joints).
    void applyArmTorque(const Eigen::VectorXd& tau_joints);

    void step(int n_steps);

    double time() const;
    double timestep() const;
    int    numJoints() const { return static_cast<int>(joints_.size()); }

    //! Contacts the solver found this step.
    int contactCount() const;
    //! "geomA--geomB, ..." for the active contacts, or "" for none. Needed
    //! because the loaded DOFs alone cannot tell an arm-vs-target contact from
    //! the two arms touching each other.
    std::string contactPairs() const;

    /*! qfrc_constraint on the arm DOFs, n_joints long.
     *
     *  Carries contact AND joint-limit forces -- MuJoCo puts both in this
     *  vector -- so it is only a contact measurement while contactCount() > 0.
     *  An uncontrolled arm drifting into its range limit produces a convincing
     *  false contact signature in exactly these entries. */
    void armConstraintForce(Eigen::VectorXd& out) const;

    /*! Scene state -> the controller's free-floating-base q (7+n) and v (6+n).
     *
     *  The two models index their DOFs differently and the base blocks do not
     *  correspond: the scene has three planar joints where the controller has a
     *  6-DOF freejoint. `anchor_x`/`anchor_y` are the node's base_anchor_x/y,
     *  applied so a world-frame setpoint means the same thing here as it does
     *  to the running stack. */
    void readControllerState(double anchor_x, double anchor_y,
                             Eigen::VectorXd& q, Eigen::VectorXd& v) const;

    /*! The base's planar velocity (vx, vy, omega) in scene coordinates.
     *
     *  Exposed for the drag measurement: the air-bearing table is modelled as
     *  joint damping, so the drag force on each planar DOF is exactly
     *  -damping * velocity. That makes it a disturbance whose true value is
     *  known analytically at every instant -- unlike contact, which has to be
     *  read back from the solver. */
    std::array<double, 3> baseVelocity() const;

    //! The damping actually in force on the three planar base DOFs, so a
    //! caller can compute the drag it should be seeing.
    std::array<double, 3> baseDamping() const { return cfg_.base_damping; }

private:
    struct JointRef {
        int qpos = 0;    //!< scene qpos address
        int dof  = 0;    //!< scene dof address
        int act  = 0;    //!< the paired <motor> actuator
    };

    Config cfg_;
    mjModel* m_ = nullptr;
    mjData*  d_ = nullptr;
    std::vector<JointRef> joints_;
    int cx_ = 0, cy_ = 0, cw_ = 0;          //!< chaser qpos addresses
    int cxd_ = 0, cyd_ = 0, cwd_ = 0;       //!< chaser dof addresses
    int tx_ = 0, ty_ = 0;                   //!< target qpos addresses
};

/*! One recorded control tick. */
struct ContactRow {
    double t = 0.0;
    int    ncon = 0;
    std::vector<double> truth;    //!< qfrc_constraint on the arm DOFs
    std::vector<double> raw;      //!< observer, full-system joint rows
    std::vector<double> reduced;  //!< observer, base-eliminated
    std::array<double, 3> base{{0.0, 0.0, 0.0}};   //!< fx, fy, tz
};

/*! How well the estimate tracked the truth on one joint.
 *
 *  Scored over the CONTACT WINDOW only. A run is mostly quiet, and a mean or a
 *  correlation taken over the quiet part measures the estimator's error floor
 *  rather than the contact -- which is how a weak estimate can be made to look
 *  excellent. Peaks are reported alongside because for a transient the peak is
 *  the quantity that matters.
 */
struct JointScore {
    int    n_contact = 0;
    double truth_peak = 0.0;
    double est_peak = 0.0;
    double truth_mean = 0.0;
    double est_mean = 0.0;
    double corr = 0.0;
    double rms_err = 0.0;

    double ratio() const;
};

JointScore scoreJoint(const std::vector<ContactRow>& rows, int joint);

}  // namespace bench
}  // namespace pat_arm_control
