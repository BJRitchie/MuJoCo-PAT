#!/usr/bin/env python3
"""
Orchestrator: repeatedly invokes generate_pinn_trajectory.py as a FRESH
subprocess per attempt, retrying with a new random seed on rejection, until
either N trajectories are accepted or a trajectory exhausts its
per-trajectory attempt budget. A fresh process per attempt isn't just
caution here (as it was for VORTEX's Basilisk-construction concern) -- it's
required: pat_simulation has no reset mechanism, so a different initial
condition genuinely needs a fresh `ros2 launch`.

No ROS/mujoco import here at all -- pure subprocess/argparse/os/sys (plus
h5py/numpy, for the post-run summary stats below), so this can run from a
plain Python environment without the full ROS 2/MuJoCo stack (the stack is
only needed inside each worker subprocess, via its own `ros2 launch` call).

Usage (inside the container):
    python3 generate_pinn_dataset.py --num-trajectories 25 --clean
"""

import argparse
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(CURRENT_FOLDER, "generate_pinn_trajectory.py")
DEFAULT_OUT_ROOT = os.path.normpath(os.path.join(CURRENT_FOLDER, "..", "..", "pinn_data"))


def _settle_time(t, ee_pos, target, tol):
    """Time from which the EE position error stays under `tol` continuously
    through the end of the recording -- the last index where error >= tol,
    plus one step. 0.0 if the error is under tol for the whole trajectory;
    the trajectory's last timestamp if it's somehow never under tol at all
    (shouldn't happen for an ACCEPTED trajectory, since acceptance already
    checked the trailing convergence_window, but this stays well-defined
    rather than crashing if it somehow does)."""
    err = np.linalg.norm(ee_pos - target, axis=1)
    over = np.nonzero(err >= tol)[0]
    if len(over) == 0:
        return 0.0
    last_over = over[-1]
    return t[last_over + 1] if last_over + 1 < len(t) else t[-1]


def compute_summary_stats(out_root, accepted_indices):
    """Scans each accepted trajectory's data.h5 for: the single largest
    |torque| and |velocity| seen (value + which trajectory/joint), and a
    per-trajectory "time to converge" (max of both arms' _settle_time,
    since the trajectory as a whole isn't done until BOTH arms have
    settled) -- skipped per-file if pos_tol wasn't recorded (older files,
    before this field was added) rather than guessing a tolerance. Returns
    (max_tau, max_qdot, settle_times): max_tau/max_qdot are
    (value, traj_index, joint_label) or (0.0, None, None) if nothing was
    readable; settle_times is a plain list, one entry per trajectory that
    had pos_tol recorded.
    """
    max_tau = (0.0, None, None)
    max_qdot = (0.0, None, None)
    settle_times = []
    for i in accepted_indices:
        path = os.path.join(out_root, f"traj{i}", "data.h5")
        try:
            with h5py.File(path, "r") as f:
                t = f["time"][:]
                for side in ("L", "R"):
                    tau = f[f"arm_{side}/tau"][:]
                    qdot = f[f"arm_{side}/qdot"][:]
                    j_tau = np.unravel_index(np.argmax(np.abs(tau)), tau.shape)
                    if abs(tau[j_tau]) > max_tau[0]:
                        max_tau = (abs(tau[j_tau]), i, f"{side}_j{j_tau[1]}")
                    j_qdot = np.unravel_index(np.argmax(np.abs(qdot)), qdot.shape)
                    if abs(qdot[j_qdot]) > max_qdot[0]:
                        max_qdot = (abs(qdot[j_qdot]), i, f"{side}_j{j_qdot[1]}")

                pos_tol = f.attrs.get("pos_tol")
                if pos_tol is not None:
                    settle_L = _settle_time(t, f["arm_L/ee_pos"][:], np.array(f.attrs["target_ee_pos_L"]), pos_tol)
                    settle_R = _settle_time(t, f["arm_R/ee_pos"][:], np.array(f.attrs["target_ee_pos_R"]), pos_tol)
                    settle_times.append(max(settle_L, settle_R))
        except Exception as e:
            print(f"[summary] could not read traj{i}/data.h5: {e}")
    return max_tau, max_qdot, settle_times


def main():
    parser = argparse.ArgumentParser(
        description="Generate a full PINN training dataset: many dual-arm "
                     "constrained-NMPC point-to-point trajectories, each "
                     "verified collision-free and joint-limit-compliant."
    )
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--max-attempts-per-traj", type=int, default=15,
                         help="Empirically the accept rate per attempt is around 50%% "
                              "(rejections are mostly genuine self-collision or joint-limit "
                              "overshoot from the random sampler, not tracking failures) -- "
                              "15 gives comfortable headroom above that.")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--visualise", action="store_true",
                         help="Open the live MuJoCo viewer for every episode -- for "
                              "eyeballing controller behavior. Pair with "
                              "--num-trajectories 1 (and a generous "
                              "--max-attempts-per-traj) unless you actually want to "
                              "watch a whole batch run back-to-back.")
    parser.add_argument("--clean", action="store_true",
                         help="Remove --out-root entirely before generating "
                              "(e.g. to drop old-schema trajectories from a "
                              "previous run before regenerating). Off by "
                              "default -- no surprise deletions.")
    # Forwarded straight through to each worker invocation -- keep these
    # defaults in sync with generate_pinn_trajectory.py's own.
    parser.add_argument("--run-time", type=float, default=10.0,
                         help="See generate_pinn_trajectory.py's own help for this flag.")
    parser.add_argument("--joint-margin-deg", type=float, default=10.0)
    parser.add_argument("--joint2-max-deg", type=float, default=60.0,
                         help="See generate_pinn_trajectory.py's own help for this flag "
                              "-- caps joint2's sampling range, not its control limit.")
    parser.add_argument("--joint3-min-bend-deg", type=float, default=35.0,
                         help="See generate_pinn_trajectory.py's own help for this flag.")
    parser.add_argument("--max-swing-deg", type=float, default=15.0)
    parser.add_argument("--min-swing-deg", type=float, default=5.0)
    parser.add_argument("--collision-waypoints", type=int, default=9)
    parser.add_argument("--joint-limit-tol-deg", type=float, default=3.0,
                         help="See generate_pinn_trajectory.py's own help for this flag.")
    parser.add_argument("--pos-tol", type=float, default=0.05)
    parser.add_argument("--vel-tol", type=float, default=0.05)
    parser.add_argument("--convergence-window", type=float, default=1.0)
    args = parser.parse_args()

    if args.clean and os.path.exists(args.out_root):
        print(f"--clean: removing {args.out_root}")
        shutil.rmtree(args.out_root)

    forwarded = [
        "--run-time", str(args.run_time),
        "--joint-margin-deg", str(args.joint_margin_deg),
        "--joint2-max-deg", str(args.joint2_max_deg),
        "--joint3-min-bend-deg", str(args.joint3_min_bend_deg),
        "--max-swing-deg", str(args.max_swing_deg),
        "--min-swing-deg", str(args.min_swing_deg),
        "--collision-waypoints", str(args.collision_waypoints),
        "--joint-limit-tol-deg", str(args.joint_limit_tol_deg),
    ]
    if args.visualise:
        forwarded.append("--visualise")
    forwarded += [
        "--pos-tol", str(args.pos_tol),
        "--vel-tol", str(args.vel_tol),
        "--convergence-window", str(args.convergence_window),
    ]

    next_seed = args.base_seed
    results = []
    for i in range(args.start_index, args.start_index + args.num_trajectories):
        accepted = False
        attempt = 0
        for attempt in range(1, args.max_attempts_per_traj + 1):
            seed = next_seed
            next_seed += 1
            rc = subprocess.run(
                [sys.executable, WORKER_SCRIPT,
                 "--traj-index", str(i), "--attempt", str(attempt),
                 "--seed", str(seed), "--out-dir", args.out_root] + forwarded,
                check=False,
            ).returncode
            print(f"[traj {i}] attempt {attempt}/{args.max_attempts_per_traj} "
                  f"seed={seed} -> exit {rc}")
            if rc == 0:
                accepted = True
                break
        results.append((i, accepted, attempt))

        if not accepted:
            print(f"[traj {i}] FAILED after {args.max_attempts_per_traj} attempts, skipping")
        else:
            print(f"[traj {i}] ACCEPTED after {attempt} attempt(s)")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Generated {n_ok}/{args.num_trajectories} trajectories.")
    for i, ok, attempt in results:
        print(f"  traj{i}: {'accepted' if ok else 'FAILED'} after {attempt} attempt(s)")

    accepted_indices = [i for i, ok, _ in results if ok]
    if accepted_indices:
        max_tau, max_qdot, settle_times = compute_summary_stats(args.out_root, accepted_indices)
        print(f"\n=== Summary statistics ({len(accepted_indices)} accepted trajectories) ===")
        if max_tau[1] is not None:
            print(f"  Max |torque|:          {max_tau[0]:.4f} N*m   (traj{max_tau[1]}, {max_tau[2]})")
        if max_qdot[1] is not None:
            print(f"  Max |velocity|:        {max_qdot[0]:.4f} rad/s (traj{max_qdot[1]}, {max_qdot[2]})")
        if settle_times:
            print(f"  Avg time-to-converge:  {np.mean(settle_times):.2f}s "
                  f"(min {np.min(settle_times):.2f}s, max {np.max(settle_times):.2f}s, n={len(settle_times)})")
        else:
            print("  Time-to-converge:      n/a (pos_tol not recorded in these files -- "
                  "regenerate to include it)")

    sys.exit(0 if n_ok == args.num_trajectories else 1)


if __name__ == "__main__":
    main()
