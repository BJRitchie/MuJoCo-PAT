# PINN training-data generator

Generates dual-arm planar point-to-point trajectories on the real
`pat_simulation` plant (3-DOF planar base + 6 arm joints — the actual
hardware-faithful system, not `pat_arm_nmpc`'s internal 6-DOF free-floating
approximation), validates each one (collision-free replay, joint-limit
compliance, convergence), and — only for accepted trajectories — extracts
ground-truth full-system dynamics (mass matrix, Coriolis/bias, qddot) to
HDF5 + CSV.

Loose scripts, not a ROS package — nothing here needs `colcon build`. First
rclpy code in this repo (`pinn_recorder_node.py`); everything else is plain
Python + raw `mujoco` bindings, no ROS import at all outside the one node and
the launch file.

The target satellite (irrelevant to this 9-DOF dataset — only the base+arms
are recorded) is automatically parked out of every sampled config's reach for
each trial (`helpers.PARKED_TARGET_QPOS`, applied via `pat_simulation`'s
`initial_target_qpos` param), so it never causes a spurious collision
rejection.

## Expected acceptance rate

Per-attempt acceptance is around 50%. Rejections are normal and mostly come
from the random sampler drawing a configuration that turns out to
self-collide or overshoot a joint limit during the transient — not a
tracking failure. Tracking itself converges within ~2-6s and holds
(~0.02-0.05m EE error) for the rest of the episode.

## Prerequisites

Run inside the `dev` container (`docker compose up -d dev && docker compose
exec dev bash`) — the raw Python `mujoco` bindings, `numpy`, `pandas`,
`h5py`, and `pyyaml` are installed there (dev-stage only, see
`docker/Dockerfile`), not on a bare host. Source the workspace overlay first:

```bash
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash   # pat_simulation + pat_arm_nmpc must already be built
```

## Quickstart

```bash
cd tools/pinn_datagen

# One trajectory, watched live in your terminal (no MuJoCo viewer either way
# — headless by default, see pinn_datagen.launch.py's `visualise` arg):
python3 generate_pinn_trajectory.py --traj-index 1 --attempt 1 --seed 0 \
    --out-dir /tmp/pinn_smoke_test

# A real batch, written to the default ../../pinn_data/ (gitignored):
python3 generate_pinn_dataset.py --num-trajectories 20 --clean
```

`generate_pinn_dataset.py` is what you actually want for a dataset — it
loops trajectory indices, retrying each with a fresh seed (a fresh `ros2
launch` per attempt; `pat_simulation` has no reset mechanism, so this is
required, not just cautious) until it accepts one or exhausts
`--max-attempts-per-traj`, then prints summary stats (max |torque|, max
|velocity|, average time-to-converge) read back from the saved files.
`generate_pinn_trajectory.py` (one attempt, one process, exit code
0=accepted/1=rejected-after-sim/2=rejected-by-prefilter) is mainly useful on
its own for debugging a single attempt — set `PINN_DEBUG=1` in the
environment for extra diagnostics (final tracking error, subprocess
stdout/stderr on a hard failure).

## Output layout

```
pinn_data/
  traj1/
    data.h5    # /time, /base/{pose,vel}, /arm_{L,R}/{q,qdot,tau,ee_pos},
               # /dynamics/{mass,coriolis,qddot} (9x9/9/9 — see attrs["state_order"]),
               # plus root attrs: targets, seed, tolerances, xml_path, timestamps, ...
    data.csv   # same data, flat wide table — built from the same field dict as data.h5
  traj2/
    ...
```

Rejected attempts write nothing (no partial files) — their per-trial scratch
YAML/`.npz` are cleaned up even on failure.

## Key flags

Sampling (all in the WORKER's argparse, forwarded by the orchestrator):

| Flag | Default | What it does |
|---|---|---|
| `--joint-margin-deg` | 10 | Keeps sampled angles this far inside each joint's `nmpc.yaml` limits |
| `--joint2-max-deg` | 60 | Caps joint2's *sampling* range (not its control limit) — grid survey found inter-arm collision-free fraction drops from 91% at 0° to 0% by 90° (mounts are only ~0.3m apart) |
| `--joint3-min-bend-deg` | 35 | Caps how close joint3 (elbow) can get to full extension — only 9% collision-free at 0° (same-arm self-fold), ~80% by -35 to -40° |
| `--max-swing-deg` / `--min-swing-deg` | 15 / 5 | Per-joint target distance from init |
| `--collision-waypoints` | 9 | Interpolation density for the cheap pre-launch collision check |

Acceptance criteria:

| Flag | Default | What it does |
|---|---|---|
| `--run-time` | 10.0s | Episode duration — comfortably covers convergence + a held settle window |
| `--pos-tol` / `--vel-tol` | 0.05m / 0.05 rad/s | EE position / joint velocity convergence thresholds |
| `--convergence-window` | 1.0s | Trailing window that must stay under tolerance |
| `--joint-limit-tol-deg` | 3° | Allowance for the controller's own *soft* joint-limit constraint — small transient overshoots are by-design, not a real violation |

Orchestrator-only: `--num-trajectories`, `--max-attempts-per-traj` (15),
`--base-seed`, `--start-index`, `--out-root`, `--clean` (wipes `--out-root`
first).

`--visualise` (either script): opens the live MuJoCo viewer for the
episode(s) it runs — headless by default. Needs a real X11 display (the same
`xhost +local:docker` / `DISPLAY` setup the rest of this repo's MuJoCo viewer
already relies on) — plain `docker exec` with no display forwarded will just
log the same MESA/GL fallback errors you'd see anywhere else in this repo
without one. Doesn't change what's recorded/validated, purely for eyeballing
behavior — pair it with `python3 generate_pinn_trajectory.py ...
--visualise` for a single attempt, or `generate_pinn_dataset.py
--num-trajectories 1 --visualise` if you want the orchestrator's retry loop
while still watching.

Run either script with `--help` for the full list with rationale in-line.

## Troubleshooting

- **Every attempt exits 2** ("prefilter exhausted"): the collision-free
  init/target search is failing before any sim even runs — try loosening
  `--joint-margin-deg` slightly, or check `--joint2-max-deg`/
  `--joint3-min-bend-deg` haven't been pushed to 0.
- **Every attempt exits 1 with `collision_ok=False`**: a real self-collision
  during the transient (the cheap pre-filter only checks a straight-line
  interpolation, not the actual NMPC-driven path) — expected occasionally,
  a bug only if it's *every* attempt.
- **Every attempt exits 1 with `converged_L=False`/`converged_R=False`
  and `collision_ok=True`**: try a longer `--run-time` or looser
  `--pos-tol`/`--vel-tol` — occasional non-convergence is expected, a bug
  only if it's *every* attempt.
- **Leftover ROS processes after a crash**: `pinn_datagen.launch.py` wires
  the whole tree to shut down when the recorder exits, but a hard crash mid-
  episode can leave orphans — `pkill -f simulation_node; pkill -f
  arm_nmpc_node; pkill -f ee_target_publisher; pkill -f pinn_recorder` before
  retrying.
