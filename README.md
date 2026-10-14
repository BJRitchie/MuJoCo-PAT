# MuJoCo-PAT

Planar Air-bearing Table simulator for ISAM GNC and manipulation research.
**Stack:** C++17 · MuJoCo 3 · ROS 2 Humble · Eigen3 · acados/HPIPM

## What works in this release

- Chaser + stationary target on a 2 m × 2 m air-bearing table (MuJoCo)
- Rendezvous: chaser drives to a configurable hold point ahead of the target
- **GNC** — PID controller (`IController`) + direct or EKF navigation (`INavigator`)
- **Arm control**
  - `pat_robotics` — per-joint PID (`IJointController`)
  - `pat_arm_control` — dual-arm task-space NMPC: planar (x, y, θz) receding-horizon
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
| `pat_arm_control` | dual-arm task-space NMPC |
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

Rebuild one package: `colcon build --packages-select pat_arm_control` (`MAKEFLAGS=-j2`
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
| `full_stack_nmpc.launch.py` | sim + dual-arm NMPC (`pat_arm_control`), chaser **undriven** — `with_gnc:=true` also runs GNC; `with_targets:=true` also runs `ee_target_publisher`; `telemetry:=true` adds `pat_telemetry`; `plotjuggler:=true` also opens PlotJuggler with the saved layout |
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
ros2 launch pat_arm_control ee_targets.launch.py
```

Edit `src/pat_arm_control/config/ee_targets.yaml` (or pass
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
`pat_arm_control`'s internal model has no such drag, so raising them appears to the
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
colcon test --packages-select pat_gnc pat_robotics pat_simulation pat_arm_control pat_telemetry
colcon test-result --verbose
```

`pat_arm_control`'s suite is the one worth knowing about, since it asserts things
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
./build/pat_arm_control/test_pat_arm_control --gtest_also_run_disabled_tests \
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
`pat_robotics`, `pat_arm_control`) only ever see those topic names.

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

### NMPC formulation (`pat_arm_control`)

- **The arm's real torque limit is unsettled, and the two models disagree.**
  `nmpc.yaml`'s `model_tau_max` is 0.25 N·m per joint; `air_bearing_table.xacro`'s
  `<motor ctrlrange>` is ±10 N·m (joint2/joint3) and ±5 N·m (joint5), so the
  simulated actuators deliver up to 40× what the controller will command. One of
  the two is wrong about the hardware. Everything downstream scales off whichever
  it is — the derived SQP trust region and the torque constraint both come from
  `model_tau_max`.
- **The soft-constraint penalties are expensive once the box is actually active.**
  With `joint_slack_*` at 1e3/1e4 and the arm parked outside a software limit, the
  solve runs ~8 ms and produces a handful of NaN solves (~9 in 300). The
  accompanying "55% of ticks over budget" was measured at `control_hz: 200` with
  `warm_start: 2`; both have since changed, so the over-budget share needs
  redoing against the 10 ms period. In nominal
  operation the slacks are inert (~1e-16), so this only bites off-nominal — which
  is when the deadline matters most. `peak slack` on the profile line is the
  signal to watch. That cost is buying something real — full recovery of the joint,
  see below — so the retune question is how cheaply the same recovery can be had,
  not whether to abandon it. Note the earlier measurement suggesting 100× smaller
  weights were nearly free was taken while the rollout clamp was still masking the
  constraint, so it needs redoing.
- **Constraints are only imposed on the 25 ms horizon grid, not between its
  nodes.** `Ts: 0.025` against a 10 ms control period is the ordinary
  receding-horizon arrangement — a 1 s horizon at the control rate would need
  N=100 — and re-solving faster than the grid is extra feedback, not a defect.
  `Ts` being a non-integer multiple of the period costs almost nothing in itself;
  the one place it shows is Tier 1, where a stage is then held for 3,2,3,2 ticks
  instead of uniformly, which is a few ms of irregularity inside a ≤50 ms replay.
  What *was* wrong is separate: the warm-start shift advanced `u_bar_` by a whole
  stage per tick, i.e. 25 ms of plan per 10 ms of real time, so every tick
  linearised around a trajectory 15 ms of plan ahead of the arm. It now advances
  by `control_period / Ts` with linear interpolation. Measured, paired: `it_mean`
  7.41 → 7.42, `it_max` 14 unchanged, tick time indistinguishable, mean EE error
  0.09585 → 0.09574. Correct, and worth ~0.1%. It does **not** rescue
  `warm_start: 2`, which still reaches 106 iterations with the shift corrected —
  the dual iterate is the problem, not the shift rate.
  Two consequences are worth knowing: nothing bounds the joint states between grid
  nodes (each tick re-imposes the box from the fresh measurement, which mitigates
  but does not guarantee), and the solver sizes `u_0` as if it acted for 25 ms when
  it is applied for 10, so the weights are tuned around an authority the arm does
  not quite have. A non-uniform grid — first step at the control period, coarser
  after — is the fix if either turns out to matter; neither has been measured.
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
- **Constraint configuration: a SOFT box with HARD torque rows.** The box is soft
  because its slack is the only mechanism that recovers a joint; the torque rows
  are hard because soft ones buy nothing on x86-64 and collapse on aarch64. With a
  joint parked 0.70 rad outside its software limit for 300 ticks, on current
  tuning:

  | config | recovers | time outside (rad·s) | availability | worst outage | tick mean | over | NaN |
  |---|---|---|---|---|---|---|---|
  | soft box, **hard torque** | **0.699 → 0.000** | **0.353** | 96.7% | 10 ticks | **2123** | **5.3%** | **5** |
  | soft box, soft torque | 0.699 → 0.000 | 0.372 | 96.7% | 10 ticks | 5686 | 39.7% | 60 |
  | hard box + guard | 0.699 → 0.547 | 0.91 | 100% | 0 | — | — | 0 |
  | hard box, no guard | 0.699 → 0.593 | 0.96 | 0% | 300 ticks | — | — | 300 |

  The soft box's slack **is** the restoring mechanism: it brings the joint all the
  way back. Hard + guard looks best on availability and speed but recovers only
  22% of the excursion and spends 2.8× as long outside, because the guard relaxes
  on 100% of ticks and the box stops constraining anything. Raw hard is a total
  outage — the historical "fails every loop", reproduced.

  **Hard torque rows are better or equal on every column**, which inverts what this
  section used to say. Same availability, same worst outage and the same full
  recovery, plus less time outside the limit, a 2.7× faster mean tick in this
  scenario, 7.5× fewer deadline overruns and 12× fewer NaN solves. Nominal and
  aggressive-setpoint runs are indistinguishable between the two, and the live
  stack is slightly faster with hard rows (mean 859/841 µs against 966/948, QP
  iterations 7.58 against 8.17 — fewer soft rows is a smaller QP). Hardening
  changes only the QP's *plan*; the applied torque is clamped per joint before
  publishing either way.

  **The decisive evidence came from a different architecture.** Soft general rows
  are HPIPM's fragile group near infeasibility, and how fragile depends on the BLAS
  kernels underneath. Same commit, same scenario, soft box + soft torque:

  | | availability | recovery | QP status |
  |---|---|---|---|
  | x86-64 (`X64_AUTOMATIC`) | 96.7% | 0.699 → **0.000** | `0:231 2:9 3:60` |
  | aarch64 (`ARMV8A_ARM_CORTEX_A57`) | **21.7%** | 0.699 → **0.075** | `0:60 3:240` |

  Four fifths of solves returned NaN on aarch64 and the joint never came back.
  Hardening the torque rows gives 96.7% and full recovery on aarch64, matching
  x86-64 exactly; with `terminal_velocity_constraint` off the two agree to the
  printed digit on every column, so the model and the QP are portable and only
  this configuration was not. Regularisation does **not** rescue soft/soft: a
  `reg_prim` × torque-slack sweep on aarch64 peaked at 58% and was non-monotonic in
  both axes, which is scatter rather than a tunable optimum
  (`DISABLED_RegularisationStudy.SoftSoftNearInfeasibility`, and run it on the
  failing architecture — x86 passes every row, so it cannot rank them).

  Inside the margin all variants are indistinguishable in tracking
  (`ee_err` 0.0355). `DISABLED_ConstraintStudy.Ladder` regenerates the table.

  **Worth knowing how this was missed.** The previous conclusion — "hard torque is
  worse, not better", 55% availability and a 68-tick outage — was measured at
  `N: 40`, before the `nx` reduction from 19 to 13, the Cholesky `H_g` inverse and
  the warm-start retune. All of those changed the QP's conditioning, and none of
  them prompted a re-run of the study that produced the default, even though the
  study still existed and still took seconds. A measured decision outlived its
  measurement, and it took a second architecture to expose it.
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
**`tools/nmpc_timing/README.md` is how to run these measurements yourself** — a
trial runner, a log summariser, how to read the report line, and the ways a trial
gets silently invalidated.

**Where this stands.** The loop is hardened and instrumented: allocation-free on
the hot path bar `computeControl`'s returned vector, no mutex anywhere in the node,
the control loop on its own thread woken against an absolute deadline, `SCHED_FIFO`
/ `mlockall` / core pinning available and off by default, and `nmpc_bench` for
measuring a new machine without ROS. What the measurements say, repeatedly, is that
**this loop is compute-bound and its tail is set by clocks and power, not by
scheduling** — `cpu ~= wall` through every spike, and the governor, a battery cap
and an underpowered charger each cost more than any scheduling change.
`SCHED_FIFO`'s one demonstrated benefit is under contention, where it halves p95.

Three gates remain, and only the middle one needs hardware:

| gate | needs | status |
|---|---|---|
| aarch64 builds and computes correctly | any arm64 box | **CLEARED** — 71/71 on a GB10 (Cortex-X925/A725), natively |
| the Orin is fast enough per QP iteration | an Orin + MAXN/`jetson_clocks` | blocked on hardware; `nmpc_bench` makes it one command |
| a realtime kernel is necessary at all | evidence of preemption or wakeup latency | **no evidence yet**; revisit only if step 4 leaves a scheduling-shaped tail |

Clearing the first gate was not a formality: it found a real defect. The soft/soft
constraint configuration collapsed on aarch64 where x86-64 passed, which is what
led to hard torque rows becoming the default (see "Constraint configuration").
Everything else — the model fidelity tests against MuJoCo's own dynamics, the
lock-free mailbox and ring, the allocation bounds — passed on the first try.

Each node logs a timing report once a second (`reportDiagnostics`), covering
period jitter, callback wall *and* thread-CPU time, deadline misses, per-phase
durations with maxima and a histogram, QP iterations and HPIPM's own solve time.
Everything is per report window, not cumulative, so a spike does not get averaged
away. Measured on the dev box over 290 s runs of `full_stack_nmpc
with_targets:=true` at `N: 40`, both arms. The current defaults are the last row:

| `warm_start` | `control_hz` | tick mean | tick max | p95 | over deadline | `qp_iter` max | retries | periods skipped | worst miss run |
|---|---|---|---|---|---|---|---|---|---|
| 2 primal+dual | 200 | 2.4 ms | **141 ms** | 79 ms | 4.57% | **125** | 223 | 3758 | 47 |
| 0 cold | 200 | 3.5 ms | 23 ms | 13 ms | 12.99% | 26 | 0 | 328 | 97 |
| 1 primal only | 200 | 3.1 ms | 34 ms | 10 ms | 4.12% | 53 | 0 | 51 | 10 |
| **1 primal only** | **100** | 3.2 ms | **23 ms** | **10 ms** | **0.06%** | 48 | 0 | **1** | **5** |

Tracking error is identical across all four, so both settings are free on control
quality. `WarmStart.*` and `DISABLED_WarmStartStudy.Ladder` reproduce the
comparison on a deterministic closed loop.

- **Carrying the dual iterate across a setpoint step is the common worst case.**
  It makes the warm start a bad guess, HPIPM runs to `qp_max_iter` at ~0.9 ms per
  iteration, and the cold-start retry then solves the same QP again — 141 ms
  against a 5 ms period, 1–2x per setpoint change. Primal-only warm starting makes
  that **rare rather than routine**: 223 retries per 290 s becomes 0–1.
- **A failed solve no longer abandons the task.** The response is now graded
  (`safe_mode.*`): Tier 1 replays the last accepted plan, time-shifted; Tier 2 is
  the joint damping that already existed; Tier 3 latches damping so a solver that
  never recovers cannot flap between damping and tracking. Tier 1 exists because
  the solver computes N stages and applies only the first, so one bad solve leaves
  39 usable controls on the table — and on a free-floating base replaying the plan
  beats holding a torque, which keeps pushing the shared bus with nothing scheduled
  to undo it. Only the stale task *wrench* is reused; the Jacobian and the `Cv`
  feedforward are re-derived at the current measurement.

  Measured with `qp.max_iter` deliberately crippled to 14 (nominal needs ~8, hard
  ticks ~26) to provoke failures, per arm over 50 s: **104 Tier 1 entries, 209
  replay ticks, 35 damped ticks, 0 latches**, and the typical episode is a single
  replay tick. The previous policy entered damping on every failure and held it for
  `recovery_ticks`, so the same 104 episodes would have cost **at least 1040**
  damped ticks. Tier 1 recovers with no hysteresis — one good solve ends it —
  because replay was still tracking, so the handover is between two task-space
  commands rather than up from damping.
- **It does not make it impossible, and the tail reaches total solver failure.**
  The `qp_iter` maxima above (48, 53) are single samples; further runs of the same
  waypoint loop at the default settings have reached 85, and one reached **200 —
  both the warm solve and its cold retry burning the whole 100-iteration cap**,
  for a 93 ms tick, HPIPM status 2 (minimum step), and the only safe-mode entry
  observed in any run (11 ticks damped, then clean recovery). Roughly one event
  per 500 report windows, so a single run can easily miss it. Consequences:
  `qp_max_iter` cannot be tightened into a worst-case bound while the solver still
  reaches it, and the cold retry is not a reliable rescue. The tiered response above
  is what now covers it: that event would be absorbed by Tier 1 rather than costing
  11 damped ticks. What remains unbounded is the 93 ms itself -- the tiers make a
  failure cheap, they do not make the solve finish on time.
- **It is compute, not scheduling.** Thread CPU time tracks wall time to within
  0.1% even through the spikes, and HPIPM's own reported time accounts for all but
  ~60 us of the solve. A realtime kernel cannot help here; a bounded solver can.
- **Nominal is comfortable; the tail is not.** 2 iterations and ~0.85 ms of QP in
  a 3.2 ms tick, 20–40 us of period jitter. The remaining 23 ms outlier is the
  only thing between this and a 10 ms deadline guarantee.
- **The control loop allocates once per tick**, down from 1369 at the same
  `N: 30` tuning, and that one is `computeControl`'s returned vector — the
  signature returns by value, so the caller's result is necessarily a heap
  allocation. Nothing else on the tick allocates. Heap traffic on the control path
  is the classic unbounded-latency source under `PREEMPT_RT` — glibc's arena lock,
  and the page faults behind a fresh mapping — so it was removed whether or not it
  showed in a tail here. `RealtimeProfile.ControlTickAllocationCountIsBounded`
  bounds the average, `SteadyStateTicksDoNotKeepAllocating` asserts the first tick
  is where sizing happens (327, then 1 steady), and the two
  `DISABLED_AllocStudy` tests attribute whatever appears.

  | | allocations/tick |
  |---|---|
  | baseline at `N: 30` | 1369 |
  | per-node temporaries held (`NodeScratch`), `dynamicsAt` out-parameter | 843 |
  | `computeInertiaProducts` workspace + caller-owned Cholesky | 753 |
  | `evalTaskSpace` out-parameter | 629 |
  | `buildAugmentedModel` writes blocks directly | 569 |
  | per-stage QP containers held across ticks | 351 |
  | the rollout's remaining per-node destinations | 80 |
  | per-tick cost/bounds, and the solver's constant blocks | 66 |
  | torque-limit mirror, in-place defect, out-parameter returns | **1** |

  Every primitive — `dynamicsAt`, `evalTaskSpace`, `computeInertiaProducts`,
  `buildAugmentedModel`, `integrateStep`, `buildJointBoxBounds`,
  `buildTorqueGeneralConstraint`, `finalizeJointTorques` — is at **zero**, and so
  are acados, HPIPM, BLASFEO and MuJoCo: not one of the allocations was ever
  theirs. Going from 66 to 1 was two sites at exactly N each (a torque-limit
  vector rebuilt per node from values fixed at construction, and the
  multiple-shooting defect, where Eigen evaluates a matrix-vector product nested
  in a larger expression into a temporary) plus four once-per-tick locals.

  Verified numerically identical at every step (`ee_err` 0.09486, `it_mean` 7.30,
  `it_max` 15, availability 1.000 throughout). Through the 66 state it was also
  **−106 us a tick, 8 of 8 paired runs** — removing allocator traffic was a speed
  win as well as an RT one; the last 65 were not separately A/B'd.

  Four things worth knowing from doing it. Hoisting into members is **not**
  automatically faster — the compiler loses some aliasing freedom — so each step
  needs its own paired measurement. Sharing `H_bb^-1 H_bm` between `getDynamics`
  and `gjm6` (they each solved it) read as a regression over four paired runs and a
  ~8% win over the next six, so four samples settles nothing on this box. An
  allocation study has to call the code the way production calls it: measuring the
  by-value overloads after production moved to out-parameters made the study
  over-report until its residual went negative, which is what exposed it. And a
  count is not an attribution — "the rest is inside acados" was asserted from the
  fact that the remainder sat outside the measured primitives, and it was wrong.
  `DISABLED_AllocStudy.Attribution` settles it instead by resolving each
  allocation's call site with `dladdr`, grouping by originating shared object, and
  printing offsets that `addr2line` turns into file and line.
- **The QP solve is the tick**, and the first thing that actually moved it was
  shrinking the state rather than speeding up the arithmetic. `nx` was
  `7 + 2*n_model_joints` = 19, carrying Δq/qdot rows for every joint in the model;
  the six belonging to the *other* arm appeared in neither the cost (`Q` is zero on
  all joint rows) nor the constraints (the box covers `ownedJointInds` only) and no
  other row's dynamics read them, so they were free variables carried at full
  Riccati cost. `nx` is now `7 + 2*n_owned` = 13. Measured, paired and interleaved:
  **solve −26%, tick −16 to −20%**, with `ee_err`, `it_mean` and `it_max` all
  identical — the reduction is exact, not an approximation. The other arm's motion
  is still predicted, by the nonlinear rollout, which uses the whole model.
  - Worth noting the gain is far short of the `(13/19)^3 = 0.32` that pure
    Riccati scaling predicts. The realised ratio is ~0.74, i.e. an effective
    `nx^0.8`. That is NOT fixed overhead -- bucketing a 24k-tick trace by iteration
    count shows almost no per-solve constant, the cost is essentially all
    per-iteration. It is that the per-iteration work itself scales far more weakly
    than `nx^3`, because at these dimensions BLASFEO's kernels are latency-bound
    rather than flop-bound. Same conclusion the `HPIPM_TARGET` null result pointed
    at, and it caps what further size or target tuning can return.

#### What a solve actually costs

From one 24k-tick trace, bucketed by iteration count (`hpipm_us` tracks `qpsolve`
to within ~35 us throughout, so this is all inside `ocp_qp_solve`):

| `qp_iters` | solve | per iteration | ticks |
|---|---|---|---|
| 7 | 2.0 ms | 285 us | 9885 |
| 8 | 2.3 ms | 282 us | 10408 |
| 10 | 3.3 ms | 326 us | 3211 |
| 20 | 9.0 ms | 448 us | 8 |
| 50 | 18.4 ms | 368 us | 2 |

So the cost is **iterations x ~280-450 us with a near-zero intercept**: a hard
problem pays both more iterations and somewhat dearer ones. Two consequences:

- `qp_max_iter` IS a genuine time bound, provided it is sized with the *worst*
  per-iteration cost (~450 us) and not the nominal (~280 us). A cap of 20 bounds the
  solve near 9 ms, not 5.6 ms.
- It also explains the 93 ms worst case arithmetically: 200 iterations (the warm
  solve and its cold retry each burning a cap of 100) at ~465 us is ~93 ms.
- **Build targets are not the lever, measured.** Both halves came out null, so the
  remaining cost is algorithmic rather than a missed compiler flag:
  - `-march=native -mtune=native` on this package (`PAT_ARCH_FLAGS`, empty by
    default) moves nothing at tick level — five paired runs scattered either side
    of zero. Only `buildmodel` improved (0.5 → 0.35 us/node), which is ~2% of the
    tick. MuJoCo, acados, HPIPM and BLASFEO are all precompiled `.so`s, so these
    flags reach only the ~13% of the tick that is this package's own code.
  - `HPIPM_TARGET=X64_INTEL_HASWELL` in place of `GENERIC`, compared as the same
    binary against two ABI-identical acados prefixes with byte-identical BLASFEO
    targets: ~3% of the QP solve, ~2% of the tick, inside the noise. BLASFEO
    already carries the AVX2/FMA kernels; HPIPM's target only tunes its glue. The
    `HPIPM_TARGET` build arg exists for a Jetson retest but defaults to `GENERIC`.
  - What is left for the solve is therefore `N: 40`, the RTI split, or a smaller
    QP -- not the toolchain.
- **Realtime scheduling is available but off by default.** The control loop runs on
  its own thread woken against an absolute deadline, and `rt_priority` /
  `lock_memory` / `rt_cpus` in `nmpc.yaml` turn on `SCHED_FIFO`, `mlockall` and core
  pinning. They default to off so enabling them stays a deliberate, measurable
  change, and a refused policy logs once and keeps running rather than failing to
  start. See "Realtime scheduling, measured" below.

Cleared, and worth knowing about because each was invisible rather than noisy:

- **The control loop's inputs arrive without a shared lock.** The tick used to hold
  one mutex across its whole body, including both publishes. That costs nothing
  today — `rclcpp::spin()` is single-threaded, so the lock is uncontended and it is
  the executor that serialises the callbacks — but a tick moved to `SCHED_FIFO`
  blocking on a lock a normal-priority subscription callback holds is priority
  inversion, and the control thread would then wait on the scheduler. The three
  inputs now arrive through `SeqlockMailbox` (`rt_mailbox.h`): each callback
  publishes one whole fixed-size POD snapshot, the tick copies whichever is current
  without waiting, and a read that loses its race leaves the loop on its previous
  value rather than blocking for a fresh one. Measured at 10x the production rate:
  zero torn reads, zero give-ways, zero allocations.

  The tick also no longer writes back into the setpoint it reads. Seeding the
  hold-pose used to have the control loop assign `ee_x_`/`ee_y_`, making it a second
  producer on state the subscription owned; the desired pose is now the loop's own,
  taken from the mailbox when a setpoint exists and seeded from the measured pose
  when none ever arrives.

- **The tick no longer publishes, logs, or allocates.** Both publishes went through
  `rclcpp::Publisher::publish`, which depending on the middleware can allocate, take
  a lock or touch a socket, and the tick carried six `RCLCPP_*` sites, which format
  and write. Output now goes into `SpscRing` queues (`rt_mailbox.h`) that a drain
  timer publishes from, the log sites became atomic counters and a queue of mode
  events that the 1 Hz report turns back into the same log lines, and the state
  vectors are members rather than per-tick `VectorXd` locals.

  Queues rather than a latest-value mailbox for the output, because a dropped
  torque leaves the actuator on the previous command for an extra period; a refused
  push is counted and reported as an overflow rather than passed over. Mode
  transitions are queued for the same reason the report cannot just sample `mode()`
  at 1 Hz: a transition that came and went inside a window would be invisible.

  **The torque command does NOT go through the queue — it is published from the
  tick, because the queue was the wrong answer for it.** Publishing asynchronously
  costs actuation latency: a command computed at *t* waits for the next drain. That
  is inherent, and it is worse in kind than compute, because the NMPC models
  neither but has ten times the headroom for one and none for the other.

  So `publish()` was measured instead of assumed. Over **61,711 calls on mains**
  it costs **14.6 us on average and never exceeded 252 us** (`ee_pose`: 18.2 us,
  worst 308 us); timed from the control thread after the change, 18-23 us mean and
  88 us worst. Against that, queueing it cost 0-1000 us of delay on every single
  command. Publishing inline is roughly fifteen times cheaper than the latency it
  removes, so the rings, the fast poll and the planned semaphore-signalled
  publisher thread were all solving a problem that did not exist.

  `ee_pose` keeps its queue, drained at the control rate, because that one is
  telemetry and a period of delay costs nothing. The output also became more
  regular: inter-message spacing on the shared `torque_command` topic went from
  `std dev 4.4 ms` (both arms bursting when a shared poll fired) to **0.24 ms**,
  since each arm now publishes at its own tick instant.

  Two things make the inline publish safe to do from the control thread. The
  message is a reused member: the obvious version assigns `name`, which copies a
  vector of `std::string` every tick. And the stamp comes from `clock_gettime`
  rather than `get_clock()->now()`, because with `use_sim_time` the rclcpp clock
  takes a mutex the `/clock` subscription also takes — the exact inversion this
  work removes. That stamp is now real wall-clock time where it used to be frozen
  at zero (nothing publishes `/clock`); no consumer reads it, since
  `pat_simulation` and `pat_telemetry` both match on name.

  **A stalled subscriber does not stall the loop, and that is measured.** The
  hazard is specific: a reliable writer blocks up to `max_blocking_time` (100 ms by
  default, so ten missed deadlines) once its history fills and the reader stops
  acking. With `simulation_node` SIGSTOPped for 12 s mid-run:

  | | tick mean | publish mean | publish worst | deadline misses | skipped |
  |---|---|---|---|---|---|
  | before | ~840 µs | 17 µs | 76 µs | 0 | 0 |
  | **subscriber frozen** | ~1200 µs | 60 µs | **418 µs** | **0** | **0** |
  | after | ~860 µs | 18 µs | 67 µs | 0 | 0 |

  The loop held 99 ticks every window throughout, and `deadline misses 0 (longest
  run 0) | skipped periods 0` was the only value any window reported. Publish cost
  roughly triples and the tick mean rises ~40%, both bounded and both well inside
  the period.

  **`KEEP_LAST` is what makes it safe**, so the torque publisher now states its QoS
  explicitly instead of inheriting it: `KEEP_LAST` overwrites the oldest unacked
  sample and never waits, where `KEEP_ALL` would block. The elevated cost under a
  stall is reliability bookkeeping — heartbeats and retransmission tracking — not a
  stall. Changing that publisher to `KEEP_ALL` would put a 100 ms block on the
  control path.

  The caveat that remains: these are Fast DDS, two small messages, and a subscriber
  on the same host reaching it over shared memory. The `pub_torque_`/`pub_eepose_`
  statistics stay on the report line so this is a number to read on the Orin,
  against a real driver node, rather than one to assume carries over.

  The statistics went the same way and **the node now holds no mutex at all**. The
  tick owns the `PhaseStat`s, the deadline counters and the controller's profile,
  publishes a copy of all of it each time round, and clears them when it sees the
  report ask through an atomic flag — so every write stays on one thread and the
  report can never print a window a reset half-cleared. `ArmNMPC::snapshotProfile`
  and `formatProfile` split the old `profileReport()` into the copy (cheap, on the
  control thread) and the formatting (allocates, off it).
- **Realtime scheduling, measured: `SCHED_FIFO` is worth nothing on an idle box and
  halves the tail on a contended one.** Which is the deployment case, so it earns
  its place — but only once something else is competing for the core.

  The experiment: each arm's control thread pinned to ONE logical CPU (`rt_cpus:
  [4]` / `[6]`) with a `SCHED_OTHER` spinner pinned to that same CPU, so the two
  must timeshare. The SMT sibling is left idle deliberately — a spinner there would
  add SMT resource contention, which no scheduling policy can fix, and muddy an
  experiment about scheduling. Governor `performance`, 130 W adapter, interleaved,
  policy verified from each run's own log before reading its timings:

  | | p95 L/R | max L/R |
  |---|---|---|
  | no load, normal priority | **2229 / 1685** | 3027 / 2072 |
  | no load, FIFO 80 | 2334 / 2159 | 3778 / 2441 |
  | load, normal priority (round 1) | **5120 / 6584** | 6735 / 9960 |
  | load, FIFO 80 (round 1) | **3042 / 3374** | 3137 / 4084 |
  | load, normal priority (round 2) | **7062 / 6376** | 8167 / 6855 |
  | load, FIFO 80 (round 2) | **3127 / 3454** | 3549 / 4498 |

  Contention costs **3.2x on p95** at normal priority (2229 -> 7062) and **1.3x**
  under FIFO (2334 -> 3127). FIFO halves p95 and max in 4 of 4 pairs, both arms,
  both rounds, and brings the loaded tail back to within ~35% of its unloaded
  value. It also replicates on battery and on a 15 W supply, so it is not an
  artefact of one power state.

  It is not free: with nothing competing, FIFO is ~6% worse on the mean and 5-28%
  worse on p95, presumably by giving up CFS's placement heuristics. So this is a
  trade that pays off only under load — which is why `rt_priority` defaults to 0
  and the deployment config, not the dev default, is where it should be set to 80.

  **Core pinning matters for a different reason, and only on an asymmetric host.**
  Linux's realtime scheduler places by priority and by which CPU is free, with no
  notion of the core-capability asymmetry CFS handles through Thread Director, so an
  unpinned FIFO thread lands wherever and stays. On battery that was CPUs 16-19,
  all E-cores, for both arms, and the mean tick doubled (1080 -> 2118 us); pinning
  recovered all of it. On mains the placement is just as arbitrary (one arm on an
  E-core, one on a P-core) and costs nothing, because an E-core at its full 3.7 GHz
  handles this tick about as well as a P-core. The 2x was the battery cap biting
  the E-cores, not an E-core penalty on its own. The Orin's cores are homogeneous,
  so that asymmetry does not transfer, but pinning and `isolcpus` there are the
  same control, and `rt_cpus` is cheap insurance on any power- or thermally-limited
  machine.
- **The dev container can now request a realtime policy.** `docker-compose.yml`
  grants `ulimits: {rtprio: 99, memlock: 268435456}`. Deliberately **not**
  `cap_add: [SYS_NICE]`: `user:` runs the container as a non-root user, whose
  effective capability set is empty, and a capability in the bounding set is not
  granted to a non-root process without ambient capabilities — `cap_add` would look
  like it worked and do nothing. `RLIMIT_RTPRIO` is what works unprivileged, and
  `chrt -f 99 true` now succeeds with `CapEff: 0000000000000000`, which is the proof.
  `CAP_SYS_NICE` is only needed to exceed the rlimit, renice negatively, or change
  another process's policy. `CONFIG_RT_GROUP_SCHED` is unset on the host, so there
  is no cgroup RT budget to fight; cgroups are v2, so `--cpu-rt-runtime` is
  unusable regardless.

  `memlock` is 1 GiB, which has to clear the node's **virtual** size (~750 MB), not
  its ~33 MB resident set: the kernel admits an `MCL_CURRENT` `mlockall` by testing
  `RLIMIT_MEMLOCK` against `total_vm`, and it does that even with `MCL_ONFAULT`,
  which only changes how much memory ends up pinned afterwards. So the limit is an
  admission threshold rather than a reservation — with `ONFAULT` the two arms
  together pin ~45 MB, per `/proc/meminfo`'s `Mlocked`. Check `Mlocked` and not the
  process's `VmLck`: that counts the VMAs *marked* for locking, which with
  `ONFAULT` is the whole address space and reads as ~700 MB per node.
- **MuJoCo's own logging no longer runs from the tick.** Its default warning hook
  appends to `MUJOCO_LOG.TXT` -- a file write from a path that fires exactly when
  the solver is struggling -- and its default error hook calls `exit()`, killing a
  process mid-motion with a torque latched in the actuators. Both are now ours:
  warnings are counted with a fixed message buffer and surface on the report line,
  and an error throws so the existing guard turns it into safe-mode damping. The
  file is no longer created by a full run. `pat_simulation` keeps MuJoCo's defaults
  and is not affected by this (it is removed at deployment).
- **Throttled log messages were permanently suppressed.** All five
  `RCLCPP_*_THROTTLE` sites were gated on the node clock, which is frozen at zero
  with `use_sim_time:=true` and nothing publishing `/clock`, so each fired once and
  then went silent for the rest of the run -- including "waiting for joint_states"
  and the dropped non-finite-measurement warnings. They now throttle against a
  steady clock; the startup warning repeats every 2 s as intended. `header.stamp`
  on `torque_command` is now real wall-clock time, set with `clock_gettime` by the
  tick that computed it; `ee_pose` is still stamped from the ROS clock and so
  still reads zero, since nothing publishes `/clock`.
- **The trust-region loop allocated a 3-vector per stage**, i.e. N per tick.
  Hoisted to a member: 1850 -> 1809 allocations per tick, and paired runs put the
  mean 81 us lower with `qp_iter` and tracking unchanged.

### Getting this onto the Orin, in this order

Measure before changing anything. Every tail this loop has had turned out to be
clocks or power -- the governor, a battery cap, a 15 W charger -- and none of it
was visible without measuring. The kernel work is last because nothing yet
observed justifies it.

**0. Retire the arm64 risk first, on any arm64 box, before the Orin exists.**
The deployment carries two independent unknowns -- *does this code build and
compute correctly on aarch64*, and *is the Orin fast enough* -- and only the second
one needs an Orin. The first can be settled now on anything arm64 (a DGX Spark, a
Grace box, a Pi 5, a cloud arm64 instance), natively, which is both faster and more
trustworthy than cross-building under QEMU.

```bash
# on the arm64 host. Orin's flags, NOT the host's -- see the warning below.
docker build -f docker/Dockerfile --target runtime \
    --build-arg BUILD_TESTING=ON --build-arg BUILD_JOBS=8 -t pat:arm64-check .
docker run --rm pat:arm64-check bash -lc '/ws/build/pat_arm_control/test_pat_arm_control'
docker run --rm pat:arm64-check bash -lc \
  '/ws/install/pat_arm_control/lib/pat_arm_control/nmpc_bench --ticks 5000 --out /tmp/a.csv'
```

The test suite is the prize, not the benchmark. It is the only thing that checks
the *numerics* on aarch64: the fidelity tests compare the linearised model against
MuJoCo's own dynamics at tight tolerances, so NEON codegen or different FMA
contraction would show up there and nowhere else. It also exercises the arm-only
C++ hazards that are silent on x86 -- `char` is unsigned by default on ARM and
signed on x86-64, and alignment assumptions are stricter. `BUILD_TESTING` is an
ARG precisely so this is possible without editing the Dockerfile; it stays `OFF`
for a deployment image, where a gtest binary has no business.

**What this does NOT tell you is anything about speed.** A DGX Spark's GB10 is
Cortex-X925/A725 (Armv9.2-A, ~140 W); the Orin is Cortex-A78AE (Armv8.2-A,
15-60 W). Measured, so the trap is concrete rather than hypothetical:
`nmpc_bench` reports **753 us mean / 888 us max** on the GB10 against **977 /
1630** on the x86 dev laptop. The modern big core beats the laptop, as expected,
and the A78AE will not — so reading "fast enough" off a run like that would be
exactly backwards. It also says nothing about the thermal and power behaviour that this
project has repeatedly found to dominate the tail, nor about the Jetson-only RT
kernel. Build it, run the tests, note that `nmpc_bench` produced a parseable CSV,
and throw the timings away.

**The trap:** never let `-mcpu=native` or `-march=native` reach a build on a
non-Orin arm64 host, and do not set a host-specific `BLASFEO_TARGET` there. Native
on an Armv9 box emits instructions that **SIGILL on the Orin**. Building *for*
Cortex-A78 and running that binary on a newer core is fine and is the test you
want; the reverse is not. `PAT_ARCH_FLAGS` defaults to `-mcpu=cortex-a78` for
arm64 and is never `native`, which is the whole reason it defaults that way.

**1. Get the number.** `nmpc_bench` exists so this is one command with no ROS
graph, no DDS, no simulator and no X11. It is in the `runtime` image already:

```bash
/ws/install/pat_arm_control/lib/pat_arm_control/nmpc_bench --ticks 20000 --out /tmp/orin.csv
tools/nmpc_timing/plot_trace.py /tmp/orin.csv      # same CSV the node's trace writes
```

Compare against this laptop's figure from the same binary: **mean 971-977 us per
tick, 7.0 QP iterations, ~105 us per iteration**. The research brief extrapolates
an Orin at 1.5-3x a desktop; at 3x the mean becomes ~2.9 ms, which still fits a
10 ms period, and this is the measurement that confirms or kills that. Set
`jetson_clocks` and MAXN FIRST (step 2) or the number is meaningless.

Caveat on what it measures: the plant is the controller's own model, the base is
fixed, the setpoint does not move, and nothing competes. So it is steady-state
compute cost -- ideal for comparing machines, and deliberately not the tail. QP
iterations come out a flat 7.0 where the live node sees a mean of 8.1 with a max
over 20, because the tail comes from setpoint transients and system interference
that this does not reproduce. For that, run the full stack.

**2. Clocks and power, before any timing.** `sudo nvpmodel -m 0` (MAXN) and
`sudo jetson_clocks`, then confirm. This is the Orin's equivalent of the governor
and the charger, and on this laptop those two were worth 1.5-3x on the tail
between them. Re-run step 1 after.

**3. Build the deployment image for the Orin specifically.** Step 0's build was
deliberately conservative so the binary would run on any arm64 host; this is the
one that targets this chip. `PAT_ARCH_FLAGS` already defaults to
`-mcpu=cortex-a78` for arm64 (Orin is Cortex-A78AE), so the only thing to add is
BLASFEO's target. Its arm64 default is deliberately left at
`ARMV8A_ARM_CORTEX_A57`, which is ARMv8.0-A and runs anywhere; A76 is ARMv8.2-A
and would SIGILL on an older arm64 board, so raising it is opt-in and belongs
here rather than in step 0:

```bash
docker buildx build -f docker/Dockerfile --target runtime --platform linux/arm64 \
    --build-arg BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A76 \
    -t mujoco-pat:runtime-arm64 .
```

Expect little: the same class of flags moved nothing measurable on x86 (see
"Settled by measurement"). The point is not leaving the target unstated.

**4. Realtime scheduling, which is a config change and not a kernel one.** Set
`rt_priority: 80` and the per-arm `rt_cpus` in the deployment `nmpc.yaml`, and
grant the limits -- `ulimits: {rtprio: 99, memlock: <clears VmSize>}`. It is
`RLIMIT_RTPRIO` that permits `SCHED_FIFO` for an unprivileged user, NOT
`CAP_SYS_NICE`, and `cap_add` is inert when the container runs as a non-root user.
Confirm `grep RT_GROUP_SCHED /boot/config-$(uname -r)` is unset, as it is on the
dev host, or a cgroup RT budget will fight the policy. Check the node's own first
report line says `SCHED_FIFO priority 80` -- it degrades to normal priority and
says so rather than failing to start.

**5. Only then consider a realtime kernel.** Pin JetPack R36.4.x: R36.5 has no
published RT packages. `sudo apt install nvidia-l4t-rt-kernel` via OTA, keep the
generic kernel as a boot entry, and **validate Ethernet and USB on the RT kernel
immediately** -- that is the known risk, and a headless board that loses its
network is a reflash. `isolcpus` / `irqaffinity` / `kthread_cpus` on the kernel
command line reserve the control cores properly, which `rt_cpus` only approximates
from userspace. Then `cyclictest --mlockall -p 80` pinned to those cores, for
hours, under representative load -- not idle, which measures nothing.

The honest prior: on this laptop `SCHED_FIFO` is worth nothing idle and halves the
p95 under contention, and `cpu ~= wall` through every spike, so there is still no
evidence of preemption or wakeup latency costing anything. A kernel would address a
problem not yet observed. Revisit it only if step 4 leaves a tail that `cpu` versus
`wall` shows is scheduling rather than compute.

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
- Trajectory tracking in NMPC 
- Define trajectory 
- Sliding Mode Control and OMPC ports from VORTEX as baseline comparisons 
- EKF/Observer implementation 
  - Consider a disturbance observer for unmodelled disturbances?

## Phase roadmap

| Phase | Features |
|---|---|
| MVP | Rendezvous · PID GNC · direct navigation · scaffold interfaces |
| Phase 2 (in progress) | EKF navigation · dual-arm task-space NMPC (acados QP) · GNC retune for the real bus mass · additional scenarios · target tumbling |
| Phase 3 | SQP NMPC tuning and realtime hardening · mission/teleop node for setpoints · `pat_vision` CV pipeline · Jetson deployment |
