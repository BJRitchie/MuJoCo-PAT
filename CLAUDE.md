# CLAUDE.md — Agent context for MuJoCo-PAT

Read this before making any changes.

## What this repo is

MuJoCo-PAT is a ROS 2 workspace (C++17 core, Python launch only) simulating a
planar air-bearing table for ISAM proximity operations research.
Physics: MuJoCo 3. Control: PID (MVP) → LQR → MPC. Navigation: direct (MVP) → EKF.

## Package scope

| Package | Status | Notes |
|---|---|---|
| `pat_msgs` | ✅ | ThrusterCommand, ControlError |
| `pat_rt` | ✅ | Header-only, zero-ROS realtime primitives: `SeqlockMailbox` / `SpscRing` (`rt_mailbox.h`) and `applyAffinity` / `applyRealtimePolicy` / `lockMemory` / `PeriodicWaker` (`rt_thread.h`). Every periodic-loop thread (arm control, arm observer, base navigator) uses these rather than its own copy; `PeriodicWaker`'s resync must stay before the sleep, or the loop runs at half rate (`PeriodicWaker.CheckingAfterTheWorkWouldHalveTheRate`). |
| `pat_platform_description` | ✅ | shared xacro macros: `pat_platform_bus` (real PAT bus) + `piper_planar_arm` (planar 3R). Consumed by `pat_simulation` and `pat_arm_control`; expanded to MJCF at build time. |
| `pat_simulation` | ✅ | MuJoCo simulation node (the hardware stand-in) |
| `pat_gnc` | ✅ | IController + INavigator; PID, DirectNav, EKF. **PID gains / `planar_dynamics.hpp` still tuned for a 15 kg chaser — the bus is now 49 kg; retune pending.** |
| `pat_robotics` | ✅ | IJointController + JointPID + ArmController, arm_control_node. IManipulator (FK/IK/Jacobian) still a stub |
| `pat_arm_control` | 🟡 | Dual-arm task-space NMPC (port of VORTEX `fswAlgorithms/jointControl`). Both planar (x,y,θz) paths are functional against the acados/HPIPM QP — iterative linearization (`nmpc.yaml`'s `full_nonlinear: false`) and multiple-shooting SQP (`full_nonlinear: true`); tuning is in progress, so the active `nmpc.yaml` block moves. Weights `Q_pos`/`Q_vel`/`R` are `[x, y, θz]` and **all three components apply**; `Q_ori`/`Q_angvel`/`R_ori` are unused. A failed or non-finite solve enters a sticky joint-damping safe mode (`safe_mode.*`) rather than substituting an approximate command; the joint and torque constraint groups are independently soft or hard (`qp.soft_joint_limits`/`soft_torque_limits`). The default is a **soft box with hard torque rows**: the box's slack is the only thing that recovers a joint parked outside its limit, while soft torque rows are HPIPM's fragile group near infeasibility and collapse on aarch64 (21.7% availability against 96.7%) — see the README's "Constraint configuration", and do not change either without re-running `DISABLED_ConstraintStudy.Ladder` on both architectures. `relax_box_to_contain_x0` applies to a hard BOX only — never make the box hard without it, and leave it off with a soft box, where it weakens the slack that does the restoring. The rollout's position clamp goes to the MJCF range, NOT the configured margin: clamping to the margin makes the joint box vacuous (see the README's "Settled by measurement"). The control path is allocation-free apart from `computeControl`'s returned vector (one per tick), which matters for the realtime Jetson deployment; keep it that way — `RealtimeProfile.ControlTickAllocationCountIsBounded` and `SteadyStateTicksDoNotKeepAllocating` fail if a per-node allocation comes back, and `DISABLED_AllocStudy.Attribution` locates it. Still open: solve time can exceed the control period at high `control_hz`. `ee_target_publisher` is a stand-in mission node (fixed reachable-waypoint loop); a real planner is still future work. A momentum disturbance observer (`observers/momentum_observer.h`) runs on its own RT thread (`rt_cpus_observer`) and publishes `/chaser/arm/<side>/disturbance`; it is selected by `observer: "momentum"` and defaults to `"none"`. Its estimate can be fed into the command (`observer.ff_gain`, capped by `ff_max_frac`, rate-limited by `ff_slew`, zeroed in safe mode), but **`ff_gain` ships at 0 and must stay there until velocity sensing is characterised**: at σ = 0.01 rad/s the estimate noise is ~0.52 N·m against a feedforward cap of 0.075 N·m (0.3 × the 0.25 N·m torque limit), so the command would saturate on noise. The 92% model-error tracking recovery in the README was measured on exact measurements and is an upper bound. `observer.gain: 20` is settled by measurement (see the README's "What the gain costs"). Observer scoring is offline in `contact_bench` (deterministic; modes: contact, `--drag`, `--mass-scale`/`--osc`, `--ab-ff`), not against the live stack, whose run-to-run variance swamps the comparison. |
| `pat_vision` | ❌ Phase 3 | Not created |
| `pat_telemetry` | ✅ | Merges `joint_states`/`torque_command` by name and publishes each arm's EE distance from setpoint (`/chaser/telemetry/*`) for live plotting (PlotJuggler, dev image only — see `docker/Dockerfile`'s `dev` stage). Builds for `runtime` too; touches only existing sim/hardware-boundary topics, so it works unchanged against real hardware. |

## Sim-to-real: the constraint that governs everything

`pat_simulation` is a software stand-in for physical hardware. It publishes
sensor data and subscribes to actuator commands on the same ROS 2 topics that
real hardware drivers would use. `pat_gnc` has no knowledge of whether the
data comes from MuJoCo or real sensors.

At deployment, `pat_simulation` is removed and replaced by physical driver
nodes that expose the same topic interface. The algorithm packages (`pat_gnc`,
`pat_robotics`, `pat_arm_control`, `pat_vision`) are deployed unchanged — the
`runtime` container image is literally `colcon build --packages-skip pat_simulation`.

**Consequence for development:** Never add simulation-specific topics,
parameters, or logic to algorithm packages. Never add ROS headers to algorithm
library headers (`include/pat_gnc/...`). If you find yourself adding a
conditional "if in simulation" to an algorithm node, that is a design error.

## Environment setup

**Preferred: the container.** Multi-stage `docker/Dockerfile`:

```
base → acados (v0.4.2, HPIPM only) / mujoco (arch tarball) → ws-base
     → dev     (x86; toolchain + desktop + GLFW; NO baked workspace build)
     → runtime (headless; bakes colcon build --packages-skip pat_simulation — Jetson image)
```

`docker-compose.yml` is the single source of truth (VORTEX-style): `service:
dev`, `user: "${UID}:${GID}"` (from `.env`), `./:/ws` bind (colcon builds into
`./build` + `./install` on the host, like VORTEX builds into `/workspace` — no
named volumes), `network_mode/ipc/pid: host` (ROS 2 DDS SHM discovery needs all
three), X11 socket. The `dev` image has a `dev` user at UID/GID 1000 and
auto-sources ROS + overlay for every shell.

```bash
xhost +local:docker
# one-time if you've built on the host: rm -rf build install log
docker compose up -d dev             # builds the image on first run, starts it in the background
docker compose exec dev bash         # attach a shell — use this for EVERY terminal, first included
colcon build                         # ~10-15 min first time, serial
```

Use `up -d` + `exec`, not `docker compose run --rm dev bash` — `run --rm` tears the
container down the moment that one shell exits, killing anything else running
inside it (a background `ros2 launch`, a second `exec`'d shell) and forcing a
full re-attach next time. `up -d` starts the one persistent, named container
(`mujoco_pat_dev`) that every subsequent `docker compose exec dev bash` (or
`docker exec mujoco_pat_dev bash`) reliably attaches to — open as many shells
into it as you want. `docker compose down` stops it when you're done (or just
leave it running; `docker compose up -d dev` on an already-running container is
a no-op).

`.devcontainer/devcontainer.json` just references the compose `dev` service
(+ VS Code extensions + a first-build `postCreateCommand`). CI
(`.github/workflows/ci.yml`) builds `dev`/`runtime` via `docker buildx`.
`BUILD_JOBS` / `MAKEFLAGS=-j2` / `--parallel-workers 1` keep compiles
near-serial — the reference dev box is RAM-tight (see Common failures).

**Bare host:** `source setup.sh` — sets `MUJOCO_DIR` (default `~/.mujoco`,
descends into `mujoco-3.x.x/`) and `ACADOS_SOURCE_DIR` (default `~/acados`),
sources ROS 2 Humble + the install overlay. acados must be built from source
first (see the `acados` stage in `docker/Dockerfile` for the recipe).

```bash
MUJOCO_DIR=/your/path ACADOS_SOURCE_DIR=/your/acados source setup.sh
```

Do NOT add MUJOCO_DIR / ACADOS_SOURCE_DIR / CMAKE_PREFIX_PATH / LD_LIBRARY_PATH
to `~/.bashrc`.

## Build and test

```bash
source setup.sh                 # or work inside the container
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select pat_gnc pat_robotics pat_simulation pat_arm_control
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
| `/chaser/arm/joint_states` | `pat_simulation` | arm encoder driver | `pat_robotics`, `pat_arm_control` |
| `/chaser/arm/joint_setpoint` | mission/teleop node (not built) | mission/teleop node | `pat_robotics` (joint-space PID) |
| `/chaser/arm/<side>/ee_setpoint` | `ee_target_publisher` (`pat_arm_control`) or a mission/teleop node | mission/teleop node | `pat_arm_control` (task-space, `geometry_msgs/PoseStamped`, planar — x, y, yaw) |
| `/chaser/arm/<side>/ee_pose` | `pat_arm_control` (measured EE pose, `geometry_msgs/PoseStamped`, world) | `pat_arm_control` | mission/teleop node, diagnostics |
| `/chaser/arm/torque_command` | `pat_robotics` **or** `pat_arm_control` | same | `pat_simulation` / arm motor driver |
| `/clock` | `pat_simulation` (MuJoCo's simulated time, once per publish tick) | **none** — run with `use_sim_time:=false` | every node with `use_sim_time:=true` (the launch default) |

`pat_simulation` stamps every message it publishes with simulated time and
publishes `/clock` from the same value, so stamps and every node's ROS clock
agree with the physics. Stamps are what `pat_gnc`'s IMU filter integrates
between, so they must be real: on hardware the drivers stamp with acquisition
time and `use_sim_time` is false. Periodic loops never run on the ROS clock —
every timer here is a wall timer or a `pat_rt::PeriodicWaker`, and log
throttles use a steady clock — so the sim clock changes stamps, not cadence.

The three `JointState` arm topics are matched **by name** (not array position)
— `joint_states` populates `name`+`position`+`velocity`; `joint_setpoint`
populates `name`+`position`; `torque_command` populates `name`+`effort`.
`pat_robotics` and `pat_arm_control` are alternatives on `torque_command` (run one
or the other for a given set of joints); the name-matching lets two
`pat_arm_control` node instances (left/right arm) coexist on the one topic. This,
plus `pat_simulation`'s `arm_joint_names`/`arm_actuator_names` params and each
controller's `joint_names` param, is what lets joints — or a whole second arm —
be added via model + YAML config only: see "Adding an arm joint" below.

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

### Adding an arm joint (or a whole second arm)

Everything arm-related is name-driven (MuJoCo joint/actuator names, matched
by name in `sensor_msgs/msg/JointState` messages) — no source changes:

1. Add the `<body>`/`<joint>`/`<motor>` in the xacro — for a new arm segment,
   `src/pat_platform_description/xacro/piper_planar_arm.xacro`; for a whole new
   limb, add a `pat_platform_bus`-style instantiation. Both `air_bearing_table`
   and `pat_platform_planar` regenerate.
2. Append the joint/actuator names to `pat_simulation`'s `arm_joint_names`/
   `arm_actuator_names` in `config/simulation.yaml`
3. Update the controller config:
   - PID: `pat_robotics/config/pid.yaml` `joint_names`/`pid.*`
   - NMPC: `pat_arm_control/config/nmpc.yaml` — the per-node `joint_names` /
     `owned_joints` / `limits.*` (this arm's joints) and the shared
     `model_joint_names` (ALL model joints, MuJoCo body-tree order)

Multiple controller instances can share `/chaser/arm/*` since messages are
name-matched and each instance only touches its own joints (that is how
`pat_arm_nmpc_left` / `_right` coexist). `IJointController`/`JointPID` follow
the same `IController`/`PIDController` extension pattern for any future
per-joint control law.

## Package architecture pattern

Every C++ package follows this exactly — do not deviate:

1. `{pkg}_lib` — static library, **zero ROS dependency**. Eigen3 for `pat_gnc`
   / `pat_robotics`; `pat_arm_control_lib` additionally links MuJoCo (rigid-body
   dynamics) and acados/HPIPM (the QP) — numerical libs, still no ROS.
2. `{pkg}_node.cpp` — thin ROS 2 wrapper: subscribe → call lib → publish
3. `test/` — GTest tests that link only against `{pkg}_lib`

Algorithm logic in the lib. ROS I/O in the node. `pat_arm_control` exposes the
lib entrypoints `ArmNMPC::computeControl(q,v)` and `currentEePose(q,v)`; the
node assembles the free-base MuJoCo `q`/`v` from `/chaser/odom` + all joints of
`/chaser/arm/joint_states` and publishes only its own joints' effort.

`pat_arm_control` groups its headers by role, since the package is meant to hold
more than one control law: `dynamics/` (`FloatingBaseDynamics` — the mass matrix,
generalised Jacobian, base elimination and `integrateStep`, i.e. the plant, not a
controller), `control/` (`ArmNMPC`), `solvers/` (acados/HPIPM, NMPC-specific),
`observers/` (`MomentumObserver`), and the two node headers at the top level.
The lock-free mailbox/ring and the realtime thread helpers live in `pat_rt`,
shared with any other node that runs a periodic loop on its own thread.

**The package is `pat_arm_control`, but the ROS node instances are still
`pat_arm_nmpc_left` / `_right` and the log tag is still `pat_arm_nmpc[side]`.**
That is deliberate, not a leftover: those names identify the control *law*, they
key the per-node blocks in `nmpc.yaml`, and `run_trial.sh`,
`summarise_timing.py`, `plot_trace.py` and `tail_source.py` all parse the log
tag. Renaming them means changing the four parsers and every archived log in the
same commit. Do it when the controller becomes a YAML choice, not before. The
executable, by contrast, *is* `arm_control_node` — note `pat_robotics` has an
executable of the same name (they are alternatives on `torque_command`), so
always qualify by package in launch files.

## MuJoCo XML conventions

- Models are **xacro**, expanded to a single self-contained MJCF at build time
  (`add_custom_command` → `xacro -o *.xml`, installed to the package share;
  the node `mj_loadXML`s the generated `.xml`):
  - `src/pat_simulation/models/environment/air_bearing_table.xacro` → the sim
    scene (chaser on `chaser_x/y/yaw` + thrusters + target + walls)
  - `src/pat_arm_control/models/pat_platform_planar.xacro` → the NMPC's *internal*
    dynamics model (same bus + arms on a `<freejoint>`, no scene furniture)
  - shared macros in `src/pat_platform_description/xacro/`: `pat_platform_bus`
    (real PAT bus: 0.6×0.5×0.305 m, 49.22 kg, real inertia) and
    `piper_planar_arm` (planar 3R; joints 1/4/6 welded, `mount_xyz` param).
    Edit the arm/bus geometry here — both consumers pick it up.
- The generated MJCF must still be one flat file: never use MuJoCo's own
  runtime `<include>`. `<xacro:include>` (build-time text expansion) is fine.
- `<actuator>` must be a direct child of `<mujoco>`, not inside `<worldbody>`
- Joint naming: `{vehicle}_{axis}` e.g. `chaser_x`, `target_yaw`. Arm joints are
  the Piper numbering `joint{2,3,5}_{L,R}` (1/4/6 are welded), each paired with
  a `motor{2,3,5}_{L,R}` `<motor>` actuator.
- Actuator index order: thrusters `[fwd=0, aft=1, port=2, stbd=3]` (must match
  `ThrusterCommand.force[4]`), then the 6 arm motors (4..9). `nu == 10`.
- Arm joints always use `<motor>` (raw torque), never `<position>`/`<general>`
  with an implicit PD gain — the controller owns the loop; a servo would fight it
- Gravity disabled: `gravity="0 0 0"` in `<option>` — do not re-enable
- Physics dt = 2 ms (500 Hz). GNC control = 10 Hz, arm NMPC = 100 Hz (`control_hz`,
  matched to publishing so it never re-solves on a state it already used).
  Publishing = 100 Hz.

## State conventions

State vector: `[px, py, theta, vx, vy, omega]`
- Metres, radians, m/s, rad/s
- Positive theta = CCW from above (right-hand rule)
- Angle wrapping: always `std::remainder(a, 2.0 * M_PI)` — never `fmod`
- Thruster forces enter `PlanarDynamics::f()` in the **body frame**; rotation
  to world frame happens inside `f()`

## Things not to do

- Do not modify `~/.bashrc` for MuJoCo / acados paths — use `setup.sh` or the container
- Do not add ROS headers to `pat_gnc/include/`, `pat_robotics/include/`, or
  `pat_arm_control/include/` (MuJoCo + acados/Eigen are allowed there — they are
  numerical libraries, not the simulator)
- Do not use MuJoCo's runtime `<include>` in the generated MJCF
- Do not use `fmod` for angle arithmetic — use `std::remainder`
- Do not add simulation-specific logic to `pat_gnc` / `pat_robotics` / `pat_arm_control`
- Do not hardcode machine paths — use `FindPackageShare` in launch files
- Do not run GNC station-keeping alongside `pat_arm_control` (it models a
  free-floating base; thrusting to hold the chaser double-counts arm reaction)
- Do not bump the acados pin off `v0.4.2` without checking `d_ocp_qp_dim_set_nsbx/_nsg`
  still exist — `v0.5.x`'s HPIPM dropped them
- Do not create `pat_vision` — it is Phase 3

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `find_package(mujoco)` / `find_package(acados)` fails | `CMAKE_PREFIX_PATH` / `ACADOS_SOURCE_DIR` not set | `source setup.sh`, or build in the container |
| `acados_c/ocp_qp_interface.h: No such file` | acados not installed on the host | build acados (see `docker/Dockerfile` `acados` stage) or use the container |
| `d_ocp_qp_dim_set_nsbx was not declared` | acados newer than `v0.4.2` | pin `ACADOS_REF=v0.4.2` |
| `mj_loadXML` error at runtime | Wrong `model_path` / stale generated MJCF | Rebuild; check the launch file |
| Topic `/chaser/odom` silent | Overlay not sourced or node crashed | `source install/setup.bash`; check `ros2 node list` |
| Tests not found | `BUILD_TESTING` not ON | Pass `-DBUILD_TESTING=ON` |
| Angle divergence past ±π | Missing `wrapAngle` | `std::remainder(a, 2*M_PI)` after every angle update |
| VS Code dies during `colcon`/`docker build` | `systemd-oomd` killing the session cgroup under memory pressure | build in a separate terminal; `--build-arg BUILD_JOBS=1`; raise `ManagedOOMMemoryPressureLimit`, add swap (dev box is 15 GiB) |
| `docker build` OOM / host thrash | `-j$(nproc)` compiles | Dockerfile already forces `BUILD_JOBS=1` / `--parallel-workers 1`; raise only with RAM headroom |
