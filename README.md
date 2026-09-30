# MuJoCo-PAT

Planar Air-bearing Table simulator for ISAM GNC and manipulation research.
**Stack:** C++17 · MuJoCo 3 · ROS 2 Humble · Eigen3 · acados/HPIPM

## What works in this release

- Chaser + stationary target on a 2 m × 2 m air-bearing table (MuJoCo)
- Rendezvous: chaser drives to a configurable hold point ahead of the target
- **GNC** — PID controller (`IController`) + direct or EKF navigation (`INavigator`)
- **Arm control**
  - `pat_robotics` — per-joint PID (`IJointController`)
  - `pat_arm_nmpc` — dual-arm task-space NMPC: planar (x, y, θz) receding-horizon
    control with joint position/velocity/torque limits solved by an acados/HPIPM
    QP, soft (penalised) or hard per constraint group. On a failed or non-finite
    solve it enters a sticky safe mode commanding pure joint damping, so the arm
    slows to rest rather than acting on an untrustworthy plan
- Arm-torque watchdog in `pat_simulation`: the simulated actuators zero on command
  silence, as a real motor driver's watchdog would, so a dead controller cannot
  leave a torque latched on
- Chaser bus + two planar Agilex-Piper 3R arms, shared between the sim plant and
  the NMPC's internal dynamics model via a `pat_platform_description` xacro
- Real CAD meshes (from the PAT digital twin) rendered over simple collision
  primitives; the primitives stay the sole authority for physics and collision
- Configurable air-bearing table drag (viscous damping + dry friction per planar
  DOF) in `pat_simulation/config/simulation.yaml`
- Live telemetry plotting (`pat_telemetry` + PlotJuggler)

## Packages

| Package | Role |
|---|---|
| `pat_msgs` | `ThrusterCommand`, `ControlError` messages |
| `pat_platform_description` | shared xacro: bus + planar arm, expanded to MJCF at build time |
| `pat_simulation` | MuJoCo sim node + viewer — the hardware stand-in |
| `pat_gnc` | PID controller, direct/EKF navigation |
| `pat_robotics` | per-joint arm PID |
| `pat_arm_nmpc` | dual-arm task-space NMPC |
| `pat_telemetry` | merges joint state/torque, publishes EE-to-setpoint distance |

`tools/pinn_datagen/` (offline training-data generator, not a ROS package) and
`src/PAT_digital_twin-main/` (RViz digital-twin viewer and CAD source) live
alongside these; see their own READMEs. Architecture rules and conventions for
contributors are in [`CLAUDE.md`](CLAUDE.md).

## Docker (recommended)

The container pins every from-source / arch-specific dependency (ROS Humble,
MuJoCo, acados + HPIPM + BLASFEO) so it builds the same on a dev box, in CI, and
on a Jetson. `docker-compose.yml` is the single source of truth; the CLI and VS
Code use it identically. See [`docker/Dockerfile`](docker/Dockerfile) for the
stages (`base → acados / mujoco → ws-base → dev / runtime`).

**One-time:** if `id -u` isn't `1000`, edit `UID`/`GID` in [`.env`](.env).
If you've built on the host before, `rm -rf build install log` (host paths are
baked into those and won't work in the container). Then:

```bash
git clone <this repo> && cd MuJoCo-PAT
xhost +local:docker                    # let the container open the MuJoCo viewer
docker compose up -d dev               # builds the image on first run, starts it in the background
docker compose exec dev bash           # attach a shell
```

Inside you land in `/ws` as your own user, ROS + the overlay auto-sourced. The
whole repo is bind-mounted — `colcon build` writes to `./build` + `./install`
on the host (gitignored). Build and run:

```bash
colcon build                                    # first time (~10-15 min serial)
ros2 launch launch/full_stack_nmpc.launch.py     # sim + dual-arm NMPC (undriven chaser)
# another terminal into the SAME container:  docker compose exec dev bash
```

Use `up -d` + `exec`, not `docker compose run --rm dev bash` — `run --rm`
destroys the container the moment that one shell exits, taking down anything
else running inside it (the launch above included) and any other shell you'd
attached with `exec`. `up -d` starts one persistent, named container
(`mujoco_pat_dev`) that every `exec` reliably attaches to, so you can open as
many terminals into it as you like. `docker compose down` stops it when you're
done; leaving it running is also fine.

Rebuild one package: `colcon build --packages-select pat_arm_nmpc` (`MAKEFLAGS=-j2`
is preset so it won't OOM). Rebuild after editing anything under `src/`
(including YAML configs and xacro models — installed files are copies, and
xacro is expanded to MJCF at build time). Only the top-level `launch/*.py`
files run straight from the source tree and pick up edits immediately.

**VS Code / Cursor:** *"Dev Containers: Reopen in Container"* uses the same
`dev` compose service; [`.devcontainer/`](.devcontainer/devcontainer.json) only
adds the extension set and runs the first `colcon build`.

**Graphics acceleration.** `docker-compose.yml` passes through `/dev/dri` (and
reserves an NVIDIA GPU) so the MuJoCo viewer is hardware-accelerated — with the
CAD meshes, software rendering (Mesa `llvmpipe`) is very slow. Two host-specific
details to check on a new machine:

- `group_add` uses the maintainer's `video`/`render` GIDs (`44`, `110`); check
  yours with `getent group video render` and edit to match.
- If the container won't start because `/dev/dri` or the NVIDIA runtime doesn't
  exist on your host, delete those entries (or override them).

Verify inside the container with `sudo apt-get install -y mesa-utils && glxinfo -B`
— you want `Accelerated: yes`, not `llvmpipe`. On a hybrid-graphics laptop the
`/dev/dri` passthrough is what matters, since the X server usually runs on the
integrated GPU rather than the NVIDIA card.

Other local-machine tweaks (extra mounts, ports) go in a gitignored
`docker-compose.override.yml`.

Other targets: `--target runtime` is the headless Jetson deploy image (algorithm
packages only, no `pat_simulation`); `--target acados` / `--target mujoco` are
just the dependency layers.

## Native build (host)

Needs Ubuntu 22.04 + ROS 2 Humble, `libeigen3-dev`, a MuJoCo 3.x release under
`~/.mujoco`, and acados built from source (see the `acados` stage in
[`docker/Dockerfile`](docker/Dockerfile) for the exact recipe) with
`ACADOS_SOURCE_DIR` pointing at the install prefix.

```bash
cd MuJoCo-PAT
source setup.sh                              # project-scoped env; no ~/.bashrc changes
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
source install/setup.bash
```

`setup.sh` only affects the current shell. Override paths before sourcing:
```bash
MUJOCO_DIR=/your/path ACADOS_SOURCE_DIR=/your/acados source setup.sh
```

## Run

All launch files take `use_sim_time:=true` by default.

| Launch | Brings up |
|---|---|
| `full_stack.launch.py` | sim + GNC (PID) + arm PID (`pat_robotics`) |
| `full_stack_nmpc.launch.py` | sim + dual-arm NMPC (`pat_arm_nmpc`), chaser **undriven** — `with_gnc:=true` also runs GNC; `with_targets:=true` also runs `ee_target_publisher`; `telemetry:=true` adds `pat_telemetry`; `plotjuggler:=true` also opens PlotJuggler with the saved layout |
| `simulation.launch.py` | MuJoCo sim only |
| `gnc.launch.py` | GNC only (against the sim or real hardware) — choose the EKF with `navigator: "ekf"` in `src/pat_gnc/config/pid.yaml` |
| `arm_control.launch.py` | per-joint arm PID (`pat_robotics`) only |

### Command the arms (task-space NMPC)

With `full_stack_nmpc.launch.py` running, publish a planar end-effector target
(world frame; x, y, and yaw are used). Use `-r 2` (not `--once` — a single
message races DDS discovery and is usually dropped); Ctrl-C once the arm moves:

```bash
ros2 topic pub -r 2 /chaser/arm/left/ee_setpoint geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 0.2, y: 0.4, z: 0.0}, orientation: {w: 1.0}}}'
```

Same for `/chaser/arm/right/ee_setpoint`. With no setpoint each arm holds its
start pose. Swap tuning: `ros2 launch launch/full_stack_nmpc.launch.py params_file:=/path/to/nmpc.yaml`.

Each arm also publishes its measured EE pose on `/chaser/arm/<side>/ee_pose`
(world frame) — useful for eyeballing tracking (`ros2 topic echo`) or seeding
external planners.

`pat_simulation` also mirrors each arm's live `ee_setpoint` onto a cosmetic
marker in the MuJoCo viewer (a magenta sphere for left, cyan for right) — a
purely visual, non-colliding `mocap` body, so you can watch how far the
controller actually is from where it's been told to go. Needs a real X11
display (`xhost +local:docker`); no effect on physics, collisions, or
recorded data either way.

#### Fixed waypoint loop

To exercise both arms without publishing setpoints by hand, run
`ee_target_publisher` — it walks each arm through a loop of known-reachable
poses. By default (`mode: relative`) the waypoints are small planar offsets
from each arm's start pose (read off `/chaser/arm/<side>/ee_pose`), so they are
reachable by construction:

```bash
ros2 launch launch/full_stack_nmpc.launch.py with_targets:=true
# or, against an already-running stack:
ros2 launch pat_arm_nmpc ee_targets.launch.py
```

Edit `src/pat_arm_nmpc/config/ee_targets.yaml` (or pass
`ee_targets_params_file:=...` — NOT `params_file`, which belongs to
`arm_nmpc.launch.py`'s `nmpc.yaml` and would silently take over if the two
collided in `full_stack_nmpc.launch.py`'s shared launch-argument namespace)
for a different set — `mode: absolute` takes world-frame `[x, y, yaw]`
triples directly.

If nothing moves: check `ros2 node list` shows `/pat_arm_nmpc_left` and
`/pat_arm_nmpc_right`, and `ros2 topic echo /chaser/arm/torque_command` is
non-zero. An empty `ros2 node list` or a `ros2 topic hz` segfault means the DDS
transport is broken — the container needs `--ipc=host --pid=host` (above).

### The viewer

| Input | Action |
|---|---|
| Left-drag | rotate camera |
| Right-drag / scroll | zoom |
| `C` | toggle collision geometry (semi-transparent magenta) and the EE-site markers, both hidden by default |
| `Space` | pause / resume rendering (physics keeps running) |
| `Esc` / `Q` | close the window |

For a headless run, add `visualise: false` under `pat_simulation.ros__parameters`
in `src/pat_simulation/config/simulation.yaml` (then rebuild).

### Simulation parameters

`src/pat_simulation/config/simulation.yaml` also holds the initial arm pose
(`initial_arm_qpos`) and the air-bearing table drag on the chaser and target,
per planar DOF `[x, y, yaw]`:

```yaml
chaser_damping:      [0.05, 0.05, 0.02]   # viscous
chaser_frictionloss: [0.0, 0.0, 0.0]      # dry / Coulomb
target_damping:      [0.01, 0.01, 0.005]
target_frictionloss: [0.0, 0.0, 0.0]
```

The platforms never contact the table geometry (gravity is off and they sit in a
different collision group), so these joint terms are the only drag they feel.
`pat_arm_nmpc`'s internal model has no such drag, so raising them appears to the
controller as unmodelled disturbance.

### Live telemetry

```bash
ros2 launch launch/full_stack_nmpc.launch.py with_targets:=true telemetry:=true plotjuggler:=true
```

`pat_telemetry` publishes `/chaser/telemetry/joint_states` (position, velocity
and torque merged by joint name) and `/chaser/telemetry/ee_distance/{left,right}`
(planar distance from each end effector to its setpoint). PlotJuggler opens with
the layout in `src/pat_telemetry/config/telemetry.xml` — joint torque / position /
velocity, base odometry (the free-floating base's velocity is the disturbance
from arm reaction), and EE tracking distance. It still asks you to confirm the
streaming plugin and the topic list on each start; the delay before it opens is
the `TimerAction` period in `launch/full_stack_nmpc.launch.py`. To run PlotJuggler
yourself: `ros2 run plotjuggler plotjuggler --layout <path to telemetry.xml>`.
PlotJuggler is installed in the `dev` image only.

## Test

```bash
colcon test --packages-select pat_gnc pat_robotics pat_simulation pat_arm_nmpc pat_telemetry
colcon test-result --verbose
```

`pat_arm_nmpc`'s suite is the one worth knowing about, since it asserts things
that are otherwise invisible until the arm misbehaves. It links the algorithm lib
only (no ROS, no launch) and checks the prediction model against MuJoCo's own
forward dynamics at a known state: that the generalised inertia reduction is
symmetric positive-definite and its damped inverse is not over-damped, that the
rollout integrator advances position by `Ts·v + Ts²/2·a`, that both the task-space
and joint rows of the augmented `A`/`B` reproduce the plant's one-step response
and respond to control, that every model joint gets its MJCF range, that the
derived trust region matches the reachable wrench, that the QP's hard bounds hold
exactly while its stages stay dynamically consistent, and that the soft
constraints are idle inside the limits but absorb a violation outside them.

It also covers safe mode (the damping command is dissipative, finite even when the
measured velocity is not, within the torque limits, stable at the control rate, and
sticky until the solver recovers) and the soft/hard constraint switch (the soft-row
layout matches what the solver was built with for every flag combination, a hard box
with an out-of-box `x0` fails while a soft one tolerates it, and the relax-to-contain
guard makes it solvable again without moving the pin).

Two `DISABLED_` tests are experiments rather than assertions, excluded from CI:

```bash
./build/pat_arm_nmpc/test_pat_arm_nmpc --gtest_also_run_disabled_tests \
    --gtest_filter='DISABLED_ConstraintStudy.*' 2>&1 | grep -v 'ArmNMPC profile'
```

`Ladder` prints a soft-vs-hard comparison across scenarios (availability, longest
safe-mode run, realised violation, guard-relax fraction, per-tick timing, HPIPM
status histogram); `BoxIsActuallyEnforced` prints what the QP plans for a joint
parked outside its limit, which is how the rollout-clamp problem was found. The
profile line is filtered out because it interleaves with the table.

## Sim-to-real

`pat_simulation` (MuJoCo) publishes `/chaser/odom`, `/chaser/imu`,
`/target/odom`, `/chaser/arm/joint_states` and subscribes to
`/chaser/thruster_command`, `/chaser/arm/torque_command` — exactly the topics a
real hardware driver stack would use. The algorithm packages (`pat_gnc`,
`pat_robotics`, `pat_arm_nmpc`) only ever see those topic names.

To deploy: bring up hardware driver nodes on the same topics and drop
`pat_simulation` (the `runtime` container image does exactly this —
`--packages-skip pat_simulation`). The algorithm packages deploy unchanged.

| Topic | In simulation | On real hardware |
|---|---|---|
| `/chaser/odom` | `pat_simulation` | motion capture driver |
| `/chaser/imu` | `pat_simulation` | IMU driver |
| `/target/odom` | `pat_simulation` | motion capture driver |
| `/chaser/arm/joint_states` | `pat_simulation` | arm encoder driver |
| `/chaser/thruster_command` | `pat_simulation` | thruster driver |
| `/chaser/arm/torque_command` | `pat_simulation` | arm motor driver |

## Known limitations and future work

Standing debt in the current implementation — open items a contributor would
otherwise have to rediscover. None are regressions.

### NMPC formulation (`pat_arm_nmpc`)

- **The arm's real torque limit is unsettled, and the two models disagree.**
  `nmpc.yaml`'s `model_tau_max` is 0.25 N·m per joint; `air_bearing_table.xacro`'s
  `<motor ctrlrange>` is ±10 N·m (joint2/joint3) and ±5 N·m (joint5), so the
  simulated actuators deliver up to 40× what the controller will command. One of
  the two is wrong about the hardware. Everything downstream scales off whichever
  it is — the derived SQP trust region and the torque constraint both come from
  `model_tau_max`.
- **The soft-constraint penalties are expensive once the box is actually active.**
  With `joint_slack_*` at 1e3/1e4 and the arm parked outside a software limit, the
  solve runs ~8 ms against the 5 ms period at `control_hz: 200`, spends 55% of
  ticks over budget, and produces a handful of NaN solves (~9 in 300). In nominal
  operation the slacks are inert (~1e-16), so this only bites off-nominal — which
  is when the deadline matters most. `peak slack` on the profile line is the
  signal to watch. That cost is buying something real — full recovery of the joint,
  see below — so the retune question is how cheaply the same recovery can be had,
  not whether to abandon it. Note the earlier measurement suggesting 100× smaller
  weights were nearly free was taken while the rollout clamp was still masking the
  constraint, so it needs redoing.
- **The 200 Hz control rate is buying nothing.** State arrives at
  `pub_rate_hz: 100` while `control_hz` is 200, so every other tick re-solves on a
  measurement it has already used. Dropping to 100 Hz would double the per-tick
  budget at no information cost. Relatedly, `Ts` is 25 ms against a 5 ms period, so
  the first horizon step models 5× the interval the command is applied for.
- **Joint velocity limits have no model source.** MJCF has no velocity range, so
  `qd_max` comes from the per-node `limits.qd_max` (2.0 rad/s) for owned joints and
  a hardcoded 10.0 rad/s default for the rest — and the rollout clamp is
  `rollout_v_clamp_mult` × that, so non-owned joints are clamped 5× looser than
  owned ones. Position and torque limits are both single-sourced now; this one is
  not. The velocity rows also carry a milder version of the clamp problem fixed
  for positions: the clamp sits at 2× the limit, so the velocity box can only be
  masked once the rollout exceeds twice its bound.
- **`nmpc.yaml` still carries dead weights.** `Q_ori`, `Q_angvel` and `R_ori` are
  unused by the planar law (the θz channel reads `Q_pos`/`Q_vel`/`R`'s third
  entry). The commented-out reference block's `full_nonlinear` comment also still
  describes the SQP path as untuned and QP-failing.

### Settled by measurement — don't redo these

- **Hard `lb == ub` bounds are deliberately NOT declared as equalities.** The
  stage-0 state pin and the terminal velocity constraint reach HPIPM as coincident
  inequality bounds rather than through its equality path (`nbxe` / `idxbxe`). The
  equality path was implemented and measured, and lost on every count:
  enforcement is identical either way (`lb == ub` pins the row on its own), the
  solution moves only at ~1e-8, and under `PARTIAL_CONDENSING_HPIPM` the solve is
  consistently about **2× slower** (~70 µs vs ~35 µs on the double-integrator case
  in `test/`). The `idxbxe` index convention is also undocumented in the installed
  HPIPM headers and cannot be distinguished by any observable behaviour, since a
  mis-declared row is not unpinned — it silently mis-declares a different row
  instead. Revisit only with HPIPM sources to hand.
- **`qp.warm_start: 2` (primal+dual) earns its place.** Over a 400-tick closed
  loop: 443 µs/solve at `2`, against 1176 µs cold and 1221 µs primal-only, with
  identical tracking and torque in all three. The dual warm start is doing the
  work, not the primal one.
- **The rollout clamps joint position to the MECHANICAL range, never to the
  software margin.** This looks like an arbitrary choice and is not. The joint box
  is written on `Δq = q − q0`, so clamping the rollout into the configured margin
  puts the nominal trajectory inside the box by construction, and the
  multiple-shooting defect `b_k` then hands the QP that teleport as achievable
  dynamics. Measured with the clamp on the software margin: the QP planned
  `Δq₁ = 0.70` where bounded torque reaches ~0.017 in one step, soft slack read
  ~1e-13 with the arm 0.7 rad outside its limit, and hard constraints never
  failed. Clamping to the mechanical range keeps the singularity protection the
  clamp exists for — that range *is* the reachable configuration space — while
  leaving the margin for the QP to enforce.
- **Soft constraints are the default because they are the only variant that
  recovers a joint, and that is measured.** With a joint parked 0.70 rad outside
  its software limit for 300 ticks:

  | config | recovers | time outside (rad·s) | availability | worst outage | NaN |
  |---|---|---|---|---|---|
  | soft box, soft torque | **0.699 → 0.000** | 0.33 | 70% | 10 ticks | 9 |
  | soft box, hard torque | 0.699 → 0.000 | 0.37 | 55% | **68 ticks** | 15 |
  | hard box + guard | 0.699 → 0.547 | 0.91 | 100% | 0 | 0 |
  | hard box, no guard | 0.699 → 0.593 | 0.96 | 0% | 300 ticks | 300 |

  The soft box's slack **is** the restoring mechanism: it brings the joint all the
  way back. Hard + guard looks best on availability and speed but recovers only
  22% of the excursion and spends 2.8× as long outside, because the guard relaxes
  on 100% of ticks and the box stops constraining anything. Raw hard is a total
  outage — the historical "fails every loop", reproduced.

  **Hard torque is worse, not better.** It changes only the QP's plan (the applied
  torque is clamped before publishing either way), and paired with a soft box it
  forbids the large corrective torque the straining box is asking for: the worst
  safe-mode outage grows from 10 ticks to 68, NaN solves from 9 to 15, and box
  slack from 1.04 to 1.73. The "the plan should match what gets applied" argument
  does not survive the measurement.

  Inside the margin all four are indistinguishable in tracking (`ee_err` 0.0359),
  and hard is ~25% faster — so the choice is entirely about off-nominal behaviour.
  `DISABLED_ConstraintStudy.Ladder` in `test/` regenerates the table.
- **A dedicated restoring cost term is not needed.** A box forbids being outside a
  limit and a cost term with an inward gradient looked like the missing piece, but
  the soft box already supplies one: `buildJointBoxBounds` sets
  `lbx = q_min − q0`, an absolute persistent target rather than a per-tick relative
  one, and the table above shows it recovering completely. Note also that the owned
  arm has **no redundancy** — three owned joints against a three-dimensional task,
  so the owned `J_task` block is 3×3 and invertible — meaning any joint recovery is
  necessarily paid for in EE error, roughly 0.2 m per 0.7 rad. That cost is visible
  in the table as soft's `ee_err` of 0.170 against hard+guard's 0.036.
- **Stage 0's soft-constraint mapping indexes from `nx`, not 0.** Stage 0's `nbx`
  is the full state (the measurement pin), so its general/torque rows begin at
  `nx`; every other stage's begin at `nbx_k`. Starting from 0 there softens the
  first `ng` rows of the pin itself — the task error and its rate — with the
  *torque* penalties, and leaves the stage-0 torque rows hard, the exact opposite
  of what the dims declare. It was latent (nominal slack is ~1e-16 either way) and
  invisible, because nothing reads stage-0 slack.

### Realtime readiness (Jetson)

`docs/realtime_deployment_research.md` holds the full research brief.

- **The control loop allocates on the heap every tick.** Each SQP iteration
  builds six horizon-length `std::vector<Eigen::MatrixXd/VectorXd>` plus per-node
  temporaries — estimated at several hundred `malloc`/`free` per tick, order 1e5/s
  at `control_hz: 200`. Invisible on a general-purpose kernel; an unbounded
  latency violation under `PREEMPT_RT`. The fix is a preallocated workspace with
  `Eigen::Map` views over it.
- **`std::cerr` on the QP-failure path**, which fires every tick precisely when
  the solver is already struggling. Not realtime-safe.
- **Off-nominal solves blow the 5 ms period even though nominal ones are
  comfortable.** At `N: 40` a nominal tick costs ~0.4–0.9 ms of QP solve. A
  setpoint well outside the workspace pushes it to ~4 ms, and a strained soft
  joint box to ~8.3 ms (see the slack item above). The budget is exceeded exactly
  in the cases a deadline guarantee is for.
- **Profiling reports means only.** `ScopedTimer` accumulates a sum and a count;
  the profile line now also carries peak soft-constraint slack, but there is still
  no max, percentile, wakeup jitter or deadline-miss counter — the numbers that
  decide whether a realtime deadline holds.
- **No realtime scheduling anywhere.** Default `rclcpp` executor, default-priority
  DDS threads, no `SCHED_FIFO`, no `mlockall`, no core pinning. The dev container
  also lacks `CAP_SYS_NICE`, so it cannot currently request an RT policy at all.

### Elsewhere

- **`IManipulator` (FK/IK/Jacobian) is an unimplemented interface.**
  `pat_robotics/include/pat_robotics/i_manipulator.hpp` declares it and nothing
  implements it. The NMPC takes its kinematics from MuJoCo directly, so nothing
  depends on it yet.
- **`ee_target_publisher` is a stand-in**, cycling fixed reachable waypoints. A
  planner or teleop node publishing `/chaser/arm/<side>/ee_setpoint` is the
  intended replacement; the topic contract already accommodates it.
- **PID gains and `planar_dynamics.hpp` are still tuned for a 15 kg chaser**,
  where the modelled bus is 49.22 kg.

## Phase roadmap

| Phase | Features |
|---|---|
| MVP | Rendezvous · PID GNC · direct navigation · scaffold interfaces |
| Phase 2 (in progress) | EKF navigation · dual-arm task-space NMPC (acados QP) · GNC retune for the real bus mass · additional scenarios · target tumbling |
| Phase 3 | SQP NMPC tuning and realtime hardening · mission/teleop node for setpoints · `pat_vision` CV pipeline · Jetson deployment |
