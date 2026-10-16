#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>

#include "pat_simulation/mujoco_vis.hpp"
#include "pat_simulation/header.h"

namespace pat_simulation {

struct PlanarState {
    double x{0}, y{0}, theta{0};
    double xdot{0}, ydot{0}, thetadot{0};
};

struct NamedJointState {
    std::vector<std::string> name;
    std::vector<double> pos;
    std::vector<double> vel;
};

/// Thin RAII wrapper around mjModel + mjData. Not thread-safe.
class MuJoCoSim {
public:
    explicit MuJoCoSim(const std::string& model_path, bool visualise = true);
    ~MuJoCoSim();
    MuJoCoSim(const MuJoCoSim&)            = delete;
    MuJoCoSim& operator=(const MuJoCoSim&) = delete;

    void reset();
    void step(const std::vector<double>& ctrl);

    /// Overwrite specific joints' qpos in place (e.g. a randomized initial arm
    /// configuration for offline dataset generation), then re-run mj_forward()
    /// so every derived quantity (site/body world poses, sensor outputs) is
    /// consistent with the new state before the first control step or
    /// publish. Velocities are left at mj_resetData's zero. Intended to be
    /// called ONCE, immediately after construction — not a general-purpose
    /// reset facility (see reset(), which restores qpos0, not an arbitrary
    /// pose). Throws if joint_names.size() != positions.size(), or if any
    /// name isn't found (via jointId()).
    void setJointPositions(const std::vector<std::string>& joint_names,
                            const std::vector<double>& positions);

    /// Overwrite viscous damping and dry (Coulomb) friction on the named
    /// 1-DOF joints, in place (the model's <joint damping/frictionloss> are
    /// otherwise fixed at xacro build time). Intended to be called once after
    /// construction. Throws on size mismatch, unknown joint, or negative value.
    void setJointDrag(const std::vector<std::string>& joint_names,
                      const std::vector<double>& damping,
                      const std::vector<double>& frictionloss);

    double time() const noexcept;
    double dt()   const noexcept;

    PlanarState getChaserState() const;
    PlanarState getTargetState() const;

    /// Generic accessor for any additional actuated joints declared in the
    /// MJCF beyond the fixed chaser/target ones (e.g. a manipulator arm).
    /// Purely data-driven — callers pass whichever joint names they care
    /// about; add joints via the MJCF + config, no source changes needed.
    NamedJointState getJointStates(const std::vector<std::string>& joint_names) const;

    /// Resolve a MuJoCo actuator's index (for indexing into the ctrl vector
    /// passed to step()). Throws if not found, matching jointId().
    int actuatorId(const std::string& name) const;

    /// Resolve a `<sensor>`'s index by name. Throws if not found, matching
    /// jointId(): a caller asking for a sensor the MJCF does not declare has a
    /// configuration error, and silently publishing zeros would be worse.
    int sensorId(const std::string& name) const;

    /// How many scalars the given sensor produces (3 for an accelerometer, 4
    /// for a framequat, and so on).
    int sensorDim(int sensor_id) const;

    /// Pointer to the sensor's values inside mjData::sensordata -- no copy, and
    /// valid only until the next step(). MuJoCo computes these exactly; any
    /// imperfection is applied afterwards by SensorModel, since MuJoCo 3.7 has
    /// no way to switch its own sensor_noise field on.
    ///
    /// Call refreshSensors() first if a step() has happened since the last one.
    const double* sensorData(int sensor_id) const;

    /// Recompute every derived quantity, sensors included, at the CURRENT
    /// qpos/qvel.
    ///
    /// Needed because mj_step evaluates the dynamics at the start of the step
    /// and integrates afterwards, so on return mjData's derived quantities
    /// describe the state the step began from, not the one it ended at. With
    /// this model's RK4 integrator it is worse than a one-step lag: the last
    /// internal evaluation is an intermediate RK stage, so sensordata matches
    /// no particular instant. Measured on the scene model, reading the gyro
    /// straight after step() disagrees with the yaw joint's own velocity by
    /// about 6%.
    ///
    /// That matters here specifically because odom is published from qpos/qvel
    /// while the IMU comes from sensordata; left unsynchronised the two
    /// describe slightly different vehicles, and an observer would read the
    /// inconsistency as a real disturbance. Any sensor lag worth modelling
    /// should be asked for explicitly through SensorModel's delay_s, where it
    /// is a stated number rather than an integrator artefact.
    ///
    /// Costs one extra forward dynamics evaluation, so call it once per publish
    /// rather than once per physics step.
    void refreshSensors();

    /// Resolve a body's index by name. Throws if not found, matching jointId().
    int bodyId(const std::string& name) const;

    /// Apply a planar external wrench to a body, in WORLD coordinates, acting
    /// at the body's centre of mass. Writes mjData::xfrc_applied, which MuJoCo
    /// keeps applying until it is changed again, so this must be called every
    /// step a time-varying disturbance is wanted.
    ///
    /// This is the one disturbance whose value is known exactly, which is what
    /// makes it the instrument for scoring a disturbance observer.
    void setExternalWrench(int body_id, double fx, double fy, double tz);

    /// Clear every body's applied wrench.
    void clearExternalWrenches();

    /// MuJoCo's constraint force in generalised coordinates (nv long) -- the
    /// exact contact force the solver found, which no algorithm could know.
    ///
    /// Ground truth for scoring a contact estimate, and therefore strictly
    /// simulation-only: it may be published on a diagnostic topic but must
    /// never be consumed by pat_gnc, pat_robotics or pat_arm_control, which
    /// have to work on real hardware where nothing like it exists.
    const double* constraintForce() const noexcept;

    /// How many contacts the solver found this step (mjData::ncon).
    ///
    /// Needed alongside constraintForce() because qfrc_constraint carries
    /// contact forces AND joint-limit forces in the same vector, and the two
    /// mean entirely different things to anything scoring it: a nonzero row
    /// with ncon == 0 is a joint against its range, not a collision.
    int contactCount() const noexcept;

    /// "geomA--geomB, geomC--geomD" for the contacts active right now, or ""
    /// when there are none.
    ///
    /// qfrc_constraint says a constraint acted and on which DOFs; it does not
    /// say what touched what, and the DOFs alone are ambiguous -- an arm-vs-arm
    /// contact loads both arms and leaves the base rows near zero, which is
    /// hard to tell from two joints simultaneously hitting their limits.
    /// Scenario authoring needs the names.
    std::string contactPairs() const;

    int nv() const noexcept;

    /// Scale the mass and rotational inertia of the named bodies.
    ///
    /// A uniform density change, so mass and inertia scale together. Intended
    /// to be called once after construction, from config, to make the
    /// simulated vehicle genuinely different from the model the controller
    /// carries -- which is the realistic ISAM case (an unknown grappled
    /// payload, a link mass taken from a CAD model that was never weighed).
    /// Perturbing the scene this way and leaving pat_arm_control's own MJCF
    /// alone is what makes the controller's model wrong rather than merely
    /// coarse.
    ///
    /// Throws on size mismatch, unknown body, or a non-positive factor.
    void setBodyMassScale(const std::vector<std::string>& bodies,
                          const std::vector<double>& factors);

    /// Resolve a mocap body's index (into mocap_pos/mocap_quat, for
    /// setMocapPose() below) by name. Returns -1 if no such body exists, or
    /// it exists but isn't a mocap body — deliberately lenient, unlike
    /// jointId()/actuatorId(): a caller driving an optional cosmetic marker
    /// should degrade gracefully on a model that doesn't have it, not crash.
    int mocapId(const std::string& name) const;

    /// Move a mocap body (a purely kinematic, non-colliding visual marker —
    /// no joint, no mass, no dynamics) to the given world-frame pose.
    /// `mocap_id` must be >= 0, from a prior mocapId() call.
    void setMocapPose(int mocap_id, double x, double y, double z,
                       double qw, double qx, double qy, double qz);

    mjModel* model() noexcept { return model_; }
    mjData*  data()  noexcept { return data_; }

private:
    // Mujoco model and data 
    mjModel* model_{nullptr};
    mjData*  data_{nullptr};

    // Visualiser
    std::unique_ptr<mujoco_vis::MuJoCoVisualiser> vis_ = nullptr;
    // step() is called at the ~500 Hz physics rate; only render every
    // Nth step (~30 Hz) so rendering doesn't throttle the physics/publish loop.
    static constexpr int kRenderEveryNSteps = 17;
    int render_counter_{0};

    int    jointId(const char* name) const;
    double jointPos(int id) const noexcept;
    double jointVel(int id) const noexcept;

    int jid_cx_{-1}, jid_cy_{-1}, jid_cyaw_{-1};
    int jid_tx_{-1}, jid_ty_{-1}, jid_tyaw_{-1};
};

} // namespace pat_simulation
