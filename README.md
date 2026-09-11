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
    QP (unconstrained backward-Riccati fallback if the QP fails to converge)
- Chaser bus + two planar Agilex-Piper 3R arms, shared between the sim plant and
  the NMPC's internal dynamics model via a `pat_platform_description` xacro

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
docker compose run --rm dev bash       # builds the image on first run
```

Inside you land in `/ws` as your own user, ROS + the overlay auto-sourced. The
whole repo is bind-mounted — `colcon build` writes to `./build` + `./install`
on the host (gitignored). Build and run:

```bash
colcon build                                    # first time (~10-15 min serial)
ros2 launch launch/full_stack_nmpc.launch.py     # sim + dual-arm NMPC (undriven chaser)
# other terminal into the same container:  docker compose exec dev bash
```

Rebuild one package: `colcon build --packages-select pat_arm_nmpc` (`MAKEFLAGS=-j2`
is preset so it won't OOM). Config / launch / xacro edits need no rebuild
(`--symlink-install`).

**VS Code / Cursor:** *"Dev Containers: Reopen in Container"* uses the same
`dev` compose service; [`.devcontainer/`](.devcontainer/devcontainer.json) only
adds the extension set and runs the first `colcon build`.

Local-machine tweaks (GPU, extra mounts, ports) go in a gitignored
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
| `full_stack_nmpc.launch.py` | sim + dual-arm NMPC (`pat_arm_nmpc`), chaser **undriven** — `with_gnc:=true` also runs GNC; `with_targets:=true` also runs `ee_target_publisher` |
| `simulation.launch.py` | MuJoCo sim only |
| `gnc.launch.py` | GNC only (against the sim or real hardware) — `navigator:=ekf` for the EKF |

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
`params_file:=...`) for a different set — `mode: absolute` takes world-frame
`[x, y, yaw]` triples directly.

If nothing moves: check `ros2 node list` shows `/pat_arm_nmpc_left` and
`/pat_arm_nmpc_right`, and `ros2 topic echo /chaser/arm/torque_command` is
non-zero. An empty `ros2 node list` or a `ros2 topic hz` segfault means the DDS
transport is broken — the container needs `--ipc=host --pid=host` (above).

## Test

```bash
colcon test --packages-select pat_gnc pat_robotics pat_simulation pat_arm_nmpc
colcon test-result --verbose
```

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

## Phase roadmap

| Phase | Features |
|---|---|
| MVP | Rendezvous · PID GNC · direct navigation · scaffold interfaces |
| Phase 2 (in progress) | EKF navigation · dual-arm task-space NMPC (acados QP) · GNC retune for the real bus mass · additional scenarios · target tumbling |
| Phase 3 | full-nonlinear SQP NMPC path · mission/teleop node for setpoints · `pat_vision` CV pipeline · Jetson deployment |
