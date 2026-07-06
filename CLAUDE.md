# CLAUDE.md — Agent context for MuJoCo-PAT

Read this before making any changes.

## What this repo is

MuJoCo-PAT is a ROS 2 workspace (C++17 core, Python launch only) simulating a
planar air-bearing table for ISAM proximity operations research.
Physics: MuJoCo 3. Control: PID (MVP) → LQR → MPC. Navigation: direct (MVP) → EKF.

## MVP scope

| Package | Status | Notes |
|---|---|---|
| `pat_msgs` | ✅ MVP | ThrusterCommand, ControlError |
| `pat_simulation` | ✅ MVP | MuJoCo simulation node |
| `pat_gnc` | ✅ MVP | IController + INavigator, PID, DirectNav, EKF lib |
| `pat_robotics` | ✅ Scaffold | IManipulator interface only — no node |
| `pat_vision` | ❌ Phase 3 | Not created |

## Sim-to-real: the constraint that governs everything

`pat_simulation` is a software stand-in for physical hardware. It publishes
sensor data and subscribes to actuator commands on the same ROS 2 topics that
real hardware drivers would use. `pat_gnc` has no knowledge of whether the
data comes from MuJoCo or real sensors.

At deployment, `pat_simulation` is removed and replaced by physical driver
nodes that expose the same topic interface. The algorithm packages (`pat_gnc`,
`pat_robotics`, `pat_vision`) are deployed unchanged.

**Consequence for development:** Never add simulation-specific topics,
parameters, or logic to algorithm packages. Never add ROS headers to algorithm
library headers (`include/pat_gnc/...`). If you find yourself adding a
conditional "if in simulation" to an algorithm node, that is a design error.

## Environment setup

Always source `setup.sh` before building or running anything:

```bash
source setup.sh   # sets MUJOCO_DIR=~/.mujoco, sources ROS 2 Humble
```

If MuJoCo is not at `~/.mujoco`:
```bash
MUJOCO_DIR=/your/path source setup.sh
```

Do NOT add MUJOCO_DIR, CMAKE_PREFIX_PATH, or LD_LIBRARY_PATH to `~/.bashrc`.

## Build and test

```bash
source setup.sh
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select pat_gnc
colcon test-result --verbose
```

## ROS 2 topic contracts

These topics are the boundary between the hardware interface and the algorithm
nodes. `pat_simulation` plays the role of the hardware interface in simulation;
physical driver nodes play it on real hardware. The algorithm nodes are
identical in both cases.

| Topic | Publisher in sim | Publisher on real hw | Subscriber |
|---|---|---|---|
| `/chaser/odom` | `pat_simulation` | motion capture driver | `pat_gnc` |
| `/chaser/imu` | `pat_simulation` | IMU driver | `pat_gnc` |
| `/target/odom` | `pat_simulation` | motion capture driver | `pat_gnc` |
| `/chaser/thruster_command` | `pat_gnc` | `pat_gnc` | `pat_simulation` / thruster driver |
| `/chaser/gnc/control_error` | `pat_gnc` | `pat_gnc` | logging / diagnostics |

Do not rename a topic without updating both `pat_simulation` and the matching
hardware driver, and recording the change in this file.

## Extension patterns

### Adding a new controller (e.g. LQR for Phase 2)

1. Create `include/pat_gnc/control/lqr_controller.hpp` — inherit `IController`
2. Create `src/lqr_controller.cpp`
3. Add `src/lqr_controller.cpp` to `pat_gnc_lib` in `CMakeLists.txt`
4. Add `else if (ctrl_type == "lqr")` in the `gnc_node.cpp` factory
5. Set `controller: "lqr"` in `config/pid.yaml`

No other files change.

### Adding a new navigator (e.g. UKF)

Same pattern: inherit `INavigator`, add `else if` in factory, update YAML.

### Adding a new scenario (e.g. docking)

1. Add `config/scenarios/docking.yaml` with initial conditions and goal
2. Pass as launch argument: `ros2 launch full_stack.launch.py scenario:=docking`
3. No node code changes required

## Package architecture pattern

Every C++ package follows this exactly — do not deviate:

1. `{pkg}_lib` — static library, Eigen3 only, zero ROS dependency
2. `{pkg}_node.cpp` — thin ROS 2 wrapper: subscribe → call lib → publish
3. `test/` — GTest tests that link only against `{pkg}_lib`

Algorithm logic in the lib. ROS I/O in the node.

## MuJoCo XML conventions

- Scene: `src/pat_simulation/models/environment/air_bearing_table.xml`
- Single self-contained file — never split via `<include>`
- `<actuator>` must be a direct child of `<mujoco>`, not inside `<worldbody>`
- Joint naming: `{vehicle}_{axis}` e.g. `chaser_x`, `target_yaw`
- Actuator index must match `ThrusterCommand.force[4]`: [fwd=0, aft=1, port=2, stbd=3]
- Gravity disabled: `gravity="0 0 0"` in `<option>` — do not re-enable
- Physics dt = 2 ms (500 Hz). Control = 10 Hz. Publishing = 100 Hz.

## State conventions

State vector: `[px, py, theta, vx, vy, omega]`
- Metres, radians, m/s, rad/s
- Positive theta = CCW from above (right-hand rule)
- Angle wrapping: always `std::remainder(a, 2.0 * M_PI)` — never `fmod`
- Thruster forces enter `PlanarDynamics::f()` in the **body frame**; rotation
  to world frame happens inside `f()`

## Things not to do

- Do not modify `~/.bashrc` for MuJoCo paths — use `setup.sh`
- Do not add ROS headers to `pat_gnc/include/` or `pat_robotics/include/`
- Do not split the MuJoCo XML using `<include>` elements
- Do not use `fmod` for angle arithmetic — use `std::remainder`
- Do not add simulation-specific logic to `pat_gnc` or `pat_robotics`
- Do not hardcode machine paths — use `FindPackageShare` in launch files
- Do not create `pat_vision` — it is Phase 3

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `find_package(mujoco)` fails | `CMAKE_PREFIX_PATH` not set | `source setup.sh` |
| `mj_loadXML` error at runtime | Wrong `model_path` param | Rebuild; check launch file |
| Topic `/chaser/odom` silent | Overlay not sourced or node crashed | `source install/setup.bash`; check `ros2 node list` |
| Eigen link error | Raw `-I` flag instead of imported target | Use `target_link_libraries(... Eigen3::Eigen)` |
| Tests not found | `BUILD_TESTING` not ON | Pass `-DBUILD_TESTING=ON` to CMake |
| Angle divergence past ±π | Missing `wrapAngle` | Apply `std::remainder(a, 2*M_PI)` after every angle update |
