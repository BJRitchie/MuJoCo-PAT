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
