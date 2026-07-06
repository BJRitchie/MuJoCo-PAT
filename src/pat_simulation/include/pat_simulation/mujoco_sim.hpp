#pragma once
#include <array>
#include <memory>
#include <string>
#include <mujoco/mujoco.h>

#include "pat_simulation/mujoco_vis.hpp"
#include "pat_simulation/header.h"

namespace pat_simulation {

struct PlanarState {
    double x{0}, y{0}, theta{0};
    double xdot{0}, ydot{0}, thetadot{0};
};

/// Thin RAII wrapper around mjModel + mjData. Not thread-safe.
class MuJoCoSim {
public:
    explicit MuJoCoSim(const std::string& model_path, bool visualise = true);
    ~MuJoCoSim();
    MuJoCoSim(const MuJoCoSim&)            = delete;
    MuJoCoSim& operator=(const MuJoCoSim&) = delete;

    void reset();
    void step(const std::array<double, 4>& ctrl);

    double time() const noexcept;
    double dt()   const noexcept;

    PlanarState getChaserState() const;
    PlanarState getTargetState() const;

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
