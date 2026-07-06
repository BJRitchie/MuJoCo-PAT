# MuJoCo-PAT

Planar Air-bearing Table simulator for ISAM GNC and manipulation research.
**Stack:** C++17 · MuJoCo 3 · ROS 2 Humble · Eigen3

## What works in this release (MVP)

- Chaser + stationary target on a 2 m × 2 m air bearing table
- Rendezvous scenario: chaser drives to a configurable hold point ahead of the target
- PID controller via `IController` interface (LQR/MPC slot in for Phase 2)
- Direct odometry navigation via `INavigator` interface (EKF slots in for Phase 2)
- `pat_robotics` scaffold with `IManipulator` interface ready for Phase 3

## Prerequisites

| Dependency | Install |
|---|---|
| Ubuntu 22.04 + ROS 2 Humble | [docs.ros.org](https://docs.ros.org/en/humble/Installation.html) |
| MuJoCo ≥ 3.1.6 — `~/.mujoco` | [github.com/google-deepmind/mujoco/releases](https://github.com/google-deepmind/mujoco/releases) |
| Eigen3 | `sudo apt install libeigen3-dev` |

## Setup

```bash
cd MuJoCo-PAT
source setup.sh                              # project-scoped env; no ~/.bashrc changes
rosdep install --from-paths src --ignore-src -y
colcon build --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
```

`setup.sh` only affects the current shell session. If MuJoCo is not at
`~/.mujoco`, set `MUJOCO_DIR` before sourcing:

```bash
MUJOCO_DIR=/your/path source setup.sh
```

## Run

```bash
ros2 launch launch/full_stack.launch.py      # simulation + GNC
ros2 launch launch/simulation.launch.py      # simulation only
ros2 launch launch/gnc.launch.py             # GNC only (against sim or real hardware)
```

## Switch to EKF navigator

```bash
ros2 launch launch/gnc.launch.py navigator:=ekf
```

Or set `navigator: "ekf"` in `src/pat_gnc/config/pid.yaml`.

## Test

```bash
colcon test --packages-select pat_gnc
colcon test-result --verbose
```

## Sim-to-real

`pat_simulation` (MuJoCo) publishes `/chaser/odom`, `/chaser/imu`, and
`/target/odom`, and subscribes to `/chaser/thruster_command` — exactly the
same topics that real hardware drivers would use. `pat_gnc` only ever sees
those topic names and does not know whether MuJoCo or physical sensors are on
the other side.

To deploy on a real vehicle: bring up your hardware driver nodes (IMU driver,
motion capture driver, thruster driver) publishing and subscribing to the same
topic names, and remove `pat_simulation`. The algorithm packages deploy
unchanged.

| Topic | In simulation | On real hardware |
|---|---|---|
| `/chaser/odom` | published by `pat_simulation` | published by motion capture driver |
| `/chaser/imu` | published by `pat_simulation` | published by IMU driver |
| `/target/odom` | published by `pat_simulation` | published by motion capture driver |
| `/chaser/thruster_command` | subscribed by `pat_simulation` | subscribed by thruster driver |

## Phase roadmap

| Phase | Features |
|---|---|
| MVP (now) | Rendezvous · PID · direct navigation · scaffold interfaces |
| Phase 2 | EKF navigation · LQR/MPC · additional scenarios · target tumbling |
| Phase 3 | `pat_robotics` arm · `pat_vision` CV pipeline |
