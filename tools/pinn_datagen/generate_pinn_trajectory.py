#!/usr/bin/env python3
"""
Generate ONE PINN training-data trajectory attempt (dual-arm planar
constrained NMPC, random point-to-point setpoint per arm).

Adapted from VORTEX's scenarioPinnDataGen.py, but architecturally different:
that script constructs the sim + controller in-process and steps it
directly; this repo's controller/sim are separate ROS 2 node processes, and
pat_simulation has no reset mechanism (MuJoCoSim::reset() is dead code) --
so this script does its own sampling/prefiltering/FK in plain Python (no
ROS), then shells out to `ros2 launch pinn_datagen.launch.py` ONCE and
blocks for exactly one episode, then does all post-hoc validation/dynamics
extraction back in plain Python against the saved .npz.

Exit codes (read by generate_pinn_dataset.py):
    0 -- accepted: collision-free, joint-limit-compliant, converged;
         data.h5/data.csv written to --out-dir/traj<N>/.
    1 -- rejected after a full episode (collision, joint-limit, sim
         crash/timeout, or non-convergence failure); NO files written.
    2 -- rejected before any episode ran (collision pre-filter exhausted
         its retries on this seed); NO files written.
"""

import argparse
import datetime
import os
import subprocess
import sys

import numpy as np
import yaml

import collision_check
import dynamics_extract
import helpers
import pinn_data_io

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCH_FILE = os.path.join(THIS_DIR, "pinn_datagen.launch.py")

PREFILTER_MAX_RETRIES = 200


# ─────────────────────────────────────────────────────────────────────────────
# Random sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_dual_arm_configs(rng: np.random.Generator, joint_ranges: dict,
                             margin_deg: float, max_swing_deg: float,
                             min_swing_deg: float = 0.0,
                             sampling_ranges: dict = None) -> dict:
    """Returns {"init_L": {...}, "target_L": {...}, "init_R": {...}, "target_R": {...}},
    each an inner dict of full joint name -> angle [rad].

    Per joint: `init` uniform in [qmin+margin, qmax-margin]; `target` in that
    same range intersected with [init-max_swing, init+max_swing], rejected
    and resampled until |target-init| >= min_swing (uniform sampling with no
    floor gives an expected swing of only max_swing/2, so most draws barely
    move -- a handful of retries here is enough in practice).

    `sampling_ranges`: optional {base_name: (qmin, qmax)} that NARROWS
    `joint_ranges` for drawing samples only -- `joint_ranges` itself (from
    nmpc.yaml) stays the real physical/control limit used everywhere else
    (validation, torque/velocity bounds). See generate_pinn_trajectory.py's
    --joint2-max-deg: an empirical grid survey of this dual-arm's mirrored
    (joint2, joint3, joint5) configuration space found the collision-free
    fraction drops from 91% at joint2=0deg to 0% by joint2=90deg, almost
    monotonically -- joint2 sweeps the whole arm toward the OTHER arm's
    mount as it increases (the two mounts are only ~0.3m apart), while
    joint3/joint5 show no comparable pattern. Sampling joint2 uniformly
    across its full nmpc.yaml range (up to 180deg) would waste most of the
    prefilter's retries on configs that are essentially always
    self-colliding for this mount spacing.
    """
    margin = np.deg2rad(margin_deg)
    max_swing = np.deg2rad(max_swing_deg)
    min_swing = np.deg2rad(min_swing_deg)
    ranges_for_sampling = dict(joint_ranges)
    if sampling_ranges:
        ranges_for_sampling.update(sampling_ranges)

    def sample_arm(joint_names_with_suffix, base_names):
        init, target = {}, {}
        for full_name, base_name in zip(joint_names_with_suffix, base_names):
            qmin, qmax = ranges_for_sampling[base_name]
            lo, hi = qmin + margin, qmax - margin
            a_init = rng.uniform(lo, hi)
            t_lo, t_hi = max(lo, a_init - max_swing), min(hi, a_init + max_swing)
            a_target = rng.uniform(t_lo, t_hi)
            for _ in range(50):
                if abs(a_target - a_init) >= min_swing:
                    break
                a_target = rng.uniform(t_lo, t_hi)
            init[full_name] = a_init
            target[full_name] = a_target
        return init, target

    init_L, target_L = sample_arm(helpers.JOINT_NAMES_L, helpers.BASE_NAMES)
    init_R, target_R = sample_arm(helpers.JOINT_NAMES_R, helpers.BASE_NAMES)
    return {"init_L": init_L, "target_L": target_L, "init_R": init_R, "target_R": target_R}


def prefilter_configs(rng, joint_ranges, mjcf_path, args):
    """Cheap, raw-mujoco-only search for a random dual-arm init/target pair
    that's collision-free at both endpoints and along a straight-line
    joint-space interpolation between them. Returns a sampled cfgs dict, or
    None if PREFILTER_MAX_RETRIES draws all failed.

    Includes helpers.PARKED_TARGET_QPOS in every checked config so this
    check matches where the live sim will actually have the target satellite
    (see write_trial_params_file's `initial_target_qpos`).
    """
    # joint3 (elbow) near its own qmax=0deg (full extension) is a SEPARATE,
    # single-arm self-fold hazard, not just the inter-arm one joint2_max_deg
    # addresses: a grid survey (one arm swept, the other parked safely)
    # found only 9% collision-free at joint3=0deg (link1 vs link3/link4/link6
    # of the SAME arm), climbing to ~80% by joint3<=-35..-40deg and
    # plateauing there -- capping joint3's sampling ceiling below full
    # extension matters at least as much as joint2's inter-arm cap.
    sampling_ranges = {
        "joint2": (joint_ranges["joint2"][0], np.deg2rad(args.joint2_max_deg)),
        "joint3": (joint_ranges["joint3"][0], np.deg2rad(-args.joint3_min_bend_deg)),
    }
    for _ in range(PREFILTER_MAX_RETRIES):
        candidate = sample_dual_arm_configs(
            rng, joint_ranges, args.joint_margin_deg, args.max_swing_deg, args.min_swing_deg,
            sampling_ranges=sampling_ranges)
        combined_init = {**candidate["init_L"], **candidate["init_R"], **helpers.PARKED_TARGET_QPOS}
        combined_target = {**candidate["target_L"], **candidate["target_R"], **helpers.PARKED_TARGET_QPOS}
        if (collision_check.is_config_collision_free(mjcf_path, combined_init)
                and collision_check.is_config_collision_free(mjcf_path, combined_target)
                and collision_check.interpolated_configs_collision_free(
                    mjcf_path, combined_init, combined_target, args.collision_waypoints)):
            return candidate
    return None


def compute_targets(mjcf_path, target_L, target_R) -> dict:
    """FK the two arms' sampled target joint configs into the [x,y,yaw]
    world-frame EE poses ee_target_publisher_node's `absolute` mode wants.
    """
    ee_L, yaw_L = helpers.target_ee_pose(mjcf_path, "ee_site_L", target_L)
    ee_R, yaw_R = helpers.target_ee_pose(mjcf_path, "ee_site_R", target_R)
    return {"ee_L": ee_L, "yaw_L": yaw_L, "ee_R": ee_R, "yaw_R": yaw_R}


# ─────────────────────────────────────────────────────────────────────────────
# One episode: write the trial overlay, launch it, load what it recorded
# ─────────────────────────────────────────────────────────────────────────────

def write_trial_params_file(path, cfgs, targets):
    """One small YAML carrying BOTH pat_simulation's `initial_arm_qpos`
    (this trial's randomized, prefiltered init config) + `initial_target_qpos`
    (parks the target satellite at helpers.PARKED_TARGET_QPOS, out of every
    sampled config's reach) and ee_target_publisher's
    `left.waypoints`/`right.waypoints` (this trial's FK-computed target pose
    per arm) -- ROS 2 params files support multiple node namespaces in one
    file, so one YAML covers both nodes.
    """
    initial_arm_qpos = [cfgs["init_L"][n] for n in helpers.JOINT_NAMES_L] + \
                        [cfgs["init_R"][n] for n in helpers.JOINT_NAMES_R]
    initial_target_qpos = [helpers.PARKED_TARGET_QPOS[n] for n in helpers.TARGET_JOINT_NAMES]
    trial_params = {
        "pat_simulation": {"ros__parameters": {
            "initial_arm_qpos": [float(v) for v in initial_arm_qpos],
            "initial_target_qpos": [float(v) for v in initial_target_qpos]}},
        "ee_target_publisher": {"ros__parameters": {
            "left.waypoints": [float(targets["ee_L"][0]), float(targets["ee_L"][1]), float(targets["yaw_L"])],
            "right.waypoints": [float(targets["ee_R"][0]), float(targets["ee_R"][1]), float(targets["yaw_R"])],
        }},
    }
    with open(path, "w") as f:
        yaml.safe_dump(trial_params, f)


def run_episode(trial_params_path, npz_path, run_time_s, visualise=False):
    """Runs the whole episode as one `ros2 launch` subprocess call and
    blocks until it self-terminates (pinn_datagen.launch.py wires the
    recorder's own exit to shut the whole tree down). Returns the completed
    subprocess.CompletedProcess, or None if it never self-terminated in
    time (a hard sim failure -- treat like any other rejection, not a
    partial success).

    `visualise`: forwarded to pinn_datagen.launch.py's own `visualise` arg
    (headless by default there) -- opens the live MuJoCo window for this
    episode. Purely for eyeballing behavior; doesn't change what gets
    recorded/validated. When set, stdout/stderr are NOT captured (left
    attached to this process' own, same as any other `ros2 launch` you'd run
    interactively) so viewer/node logs are visible while you watch.
    """
    try:
        cmd = ["ros2", "launch", LAUNCH_FILE,
               f"trial_params_file:={trial_params_path}",
               f"run_time_s:={run_time_s}",
               f"out_path:={npz_path}",
               f"visualise:={'true' if visualise else 'false'}"]
        return subprocess.run(
            cmd, timeout=run_time_s + 30, check=False,
            capture_output=not visualise, text=True)
    except subprocess.TimeoutExpired:
        return None


def load_recorded_history(npz_path: str) -> dict:
    """Loads the recorder's .npz into a plain dict of in-memory arrays
    (rather than returning the lazy npz handle directly) so the file can be
    safely removed right after this call.
    """
    with np.load(npz_path) as rec:
        return {k: rec[k] for k in rec.files}


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc validation
# ─────────────────────────────────────────────────────────────────────────────

def check_joint_limits(arm_q, joint_ranges, tol_rad, traj_index, attempt) -> bool:
    """tol_rad is NOT numerical slop (that's the 1e-6 baked into the
    comparison) -- it's a deliberate allowance for the controller's own
    documented soft joint-limit constraint (nmpc.yaml's
    joint_slack_linear/quadratic): "a joint that's already strayed outside
    its limit can never make the QP outright infeasible; it just pays a
    penalty". Empirically this shows up as small (observed: ~0.2-2deg),
    transient overshoots even on otherwise well-behaved trajectories --
    checking with zero tolerance rejects those as "limit violations" when
    they're really the QP's constraint doing exactly what it was designed
    to do, not a real safety problem. See --joint-limit-tol-deg.
    """
    ok = True
    for i, name in enumerate(helpers.MODEL_JOINT_NAMES):
        base_name = name.rsplit("_", 1)[0]
        qmin, qmax = joint_ranges[base_name]
        col = arm_q[:, i]
        jmin, jmax = float(col.min()), float(col.max())
        if not (jmin >= qmin - tol_rad - 1e-6 and jmax <= qmax + tol_rad + 1e-6):
            ok = False
            print(f"[traj {traj_index} attempt {attempt}] "
                  f"{name} VIOLATED: [{qmin:.3f},{qmax:.3f}] (tol {tol_rad:.3f}) "
                  f"observed [{jmin:.3f},{jmax:.3f}]")
    return ok


def check_convergence(ee_xy, target_xy, qdot_cols, n_window, pos_tol, vel_tol):
    err = np.linalg.norm(ee_xy[-n_window:] - target_xy, axis=1)
    speeds = qdot_cols[-n_window:]
    pos_ok = bool(np.all(err < pos_tol))
    vel_ok = bool(np.all(np.abs(speeds) < vel_tol))
    return pos_ok and vel_ok, float(err[-1])


def validate_trajectory(mjcf_path, joint_ranges, qpos_hist, qvel_hist, history, targets, args):
    """Runs every post-hoc acceptance check and returns a result dict with
    `accepted` plus everything a caller needs afterwards (EE trajectories,
    per-arm final error) without recomputing FK a second time.
    """
    arm_q, arm_qdot = history["q"], history["qdot"]
    t_hist = history["time"]

    violations = collision_check.trajectory_collision_report(mjcf_path, qpos_hist)
    collision_ok = not violations
    if not collision_ok:
        step, contacts = violations[0]
        print(f"[traj {args.traj_index} attempt {args.attempt}] "
              f"COLLISION at step {step}: {contacts} ({len(violations)} steps total in violation)")

    limits_ok = check_joint_limits(
        arm_q, joint_ranges, np.deg2rad(args.joint_limit_tol_deg), args.traj_index, args.attempt)

    n_window = max(1, int(round(args.convergence_window * len(t_hist) / max(t_hist[-1], 1e-9))))
    ee_L = helpers.ee_xy_trajectory_from_qpos(mjcf_path, "ee_site_L", qpos_hist)
    ee_R = helpers.ee_xy_trajectory_from_qpos(mjcf_path, "ee_site_R", qpos_hist)
    converged_L, final_err_L = check_convergence(
        ee_L, targets["ee_L"], arm_qdot[:, 0:3], n_window, args.pos_tol, args.vel_tol)
    converged_R, final_err_R = check_convergence(
        ee_R, targets["ee_R"], arm_qdot[:, 3:6], n_window, args.pos_tol, args.vel_tol)
    if os.environ.get("PINN_DEBUG"):
        print(f"[debug] final_err_L={final_err_L:.5f}m final_err_R={final_err_R:.5f}m pos_tol={args.pos_tol}")

    accepted = collision_ok and limits_ok and converged_L and converged_R
    return {
        "accepted": accepted, "collision_ok": collision_ok, "limits_ok": limits_ok,
        "converged_L": converged_L, "converged_R": converged_R,
        "final_err_L": final_err_L, "final_err_R": final_err_R,
        "ee_L": ee_L, "ee_R": ee_R,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save an accepted trajectory
# ─────────────────────────────────────────────────────────────────────────────

def save_trajectory(args, mjcf_path, cfgs, targets, history, qpos_hist, qvel_hist, result) -> str:
    arm_q, arm_qdot, arm_tau, t_hist = history["q"], history["qdot"], history["tau"], history["time"]
    mass9, bias9, qddot9 = dynamics_extract.compute_full_system_dynamics(
        mjcf_path, qpos_hist, qvel_hist, arm_tau)

    fields = pinn_data_io.build_field_dict(
        t_hist, history["base_pose"], history["base_vel"], arm_q, arm_qdot, arm_tau,
        result["ee_L"], result["ee_R"], mass9, bias9, qddot9)

    meta = {
        "traj_index": args.traj_index, "attempt": args.attempt, "seed": args.seed,
        "joint_names_L": helpers.JOINT_NAMES_L, "joint_names_R": helpers.JOINT_NAMES_R,
        "model_joint_names": helpers.MODEL_JOINT_NAMES,
        "init_q_L": [cfgs["init_L"][n] for n in helpers.JOINT_NAMES_L],
        "target_q_L": [cfgs["target_L"][n] for n in helpers.JOINT_NAMES_L],
        "init_q_R": [cfgs["init_R"][n] for n in helpers.JOINT_NAMES_R],
        "target_q_R": [cfgs["target_R"][n] for n in helpers.JOINT_NAMES_R],
        "target_ee_pos_L": targets["ee_L"], "target_ee_pos_R": targets["ee_R"],
        "target_ee_yaw_L": targets["yaw_L"], "target_ee_yaw_R": targets["yaw_R"],
        "final_ee_error_L_m": result["final_err_L"], "final_ee_error_R_m": result["final_err_R"],
        "run_time_s": args.run_time, "pos_tol": args.pos_tol, "vel_tol": args.vel_tol,
        "convergence_window": args.convergence_window,
        "xml_path": mjcf_path, "nominal_pub_rate_hz": 100.0,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out_dir = os.path.join(args.out_dir, f"traj{args.traj_index}")
    os.makedirs(out_dir, exist_ok=True)
    pinn_data_io.write_h5(os.path.join(out_dir, "data.h5"), fields, meta)
    pinn_data_io.write_csv(os.path.join(out_dir, "data.csv"), fields)
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(args) -> int:
    rng = np.random.default_rng(args.seed)
    mjcf_path = args.mjcf_path or helpers.default_mjcf_path()
    joint_ranges = helpers.load_joint_ranges()

    cfgs = prefilter_configs(rng, joint_ranges, mjcf_path, args)
    if cfgs is None:
        print(f"[traj {args.traj_index} attempt {args.attempt}] "
              f"prefilter exhausted {PREFILTER_MAX_RETRIES} retries, giving up")
        return 2
    targets = compute_targets(mjcf_path, cfgs["target_L"], cfgs["target_R"])

    os.makedirs(args.out_dir, exist_ok=True)
    trial_params_path = os.path.join(args.out_dir, f".trial_{args.seed}.yaml")
    npz_path = os.path.join(args.out_dir, f".trial_{args.seed}.npz")
    write_trial_params_file(trial_params_path, cfgs, targets)

    try:
        proc = run_episode(trial_params_path, npz_path, args.run_time, args.visualise)
        if proc is None:
            print(f"[traj {args.traj_index} attempt {args.attempt}] "
                  f"ros2 launch timed out (episode never self-shut-down)")
            return 1
        if proc.returncode != 0 or not os.path.exists(npz_path):
            print(f"[traj {args.traj_index} attempt {args.attempt}] "
                  f"episode failed (exit {proc.returncode}, npz exists={os.path.exists(npz_path)})")
            if os.environ.get("PINN_DEBUG") and proc.stdout is not None:
                print(proc.stdout[-4000:])
                print(proc.stderr[-4000:])
            return 1

        history = load_recorded_history(npz_path)
        model, _ = helpers._load_model(mjcf_path)
        qpos_hist, qvel_hist = helpers.build_full_qpos_qvel(
            model, history["base_pose"], history["base_vel"], history["q"], history["qdot"])

        result = validate_trajectory(mjcf_path, joint_ranges, qpos_hist, qvel_hist, history, targets, args)
        if not result["accepted"]:
            print(f"[traj {args.traj_index} attempt {args.attempt}] REJECTED "
                  f"(collision_ok={result['collision_ok']} limits_ok={result['limits_ok']} "
                  f"converged_L={result['converged_L']} converged_R={result['converged_R']})")
            return 1

        out_dir = save_trajectory(args, mjcf_path, cfgs, targets, history, qpos_hist, qvel_hist, result)
        print(f"[traj {args.traj_index} attempt {args.attempt}] ACCEPTED "
              f"(final_err_L={result['final_err_L']:.4f}m final_err_R={result['final_err_R']:.4f}m) -> {out_dir}")
        return 0
    finally:
        for p in (trial_params_path, npz_path):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ONE PINN training-data trajectory attempt "
                     "(dual-arm planar constrained NMPC, random point-to-point setpoint)."
    )
    parser.add_argument("--traj-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--visualise", action="store_true",
                         help="Open the live MuJoCo viewer for this episode -- for "
                              "eyeballing controller behavior, not part of the "
                              "validation. Headless by default (see "
                              "pinn_datagen.launch.py's own `visualise` arg).")
    parser.add_argument("--mjcf-path", type=str, default=None)
    parser.add_argument("--run-time", type=float, default=10.0,
                         help="Episode duration. Empirically, tracking error converges "
                              "within ~2-6s and holds (~0.02-0.05m) from there -- 10.0s "
                              "comfortably captures the settled state per arm without "
                              "wasting wall-clock time on every attempt.")
    parser.add_argument("--joint-margin-deg", type=float, default=10.0)
    parser.add_argument("--joint2-max-deg", type=float, default=60.0,
                         help="Caps joint2's SAMPLING range (not its nmpc.yaml control "
                              "limit) at [qmin, this]. An empirical grid survey found "
                              "this dual-arm's collision-free fraction is 91%% at "
                              "joint2=0deg but 0%% by joint2=90deg (mounts ~0.3m apart) "
                              "-- see sample_dual_arm_configs' docstring. Raise this to "
                              "widen coverage at the cost of a lower prefilter hit rate.")
    parser.add_argument("--joint3-min-bend-deg", type=float, default=35.0,
                         help="Caps joint3's SAMPLING range (not its nmpc.yaml control "
                              "limit) at [qmin, -this]. A grid survey found only 9%% "
                              "collision-free at joint3=0deg (the SAME arm's link1 "
                              "self-folds against link3/link4/link6), climbing to ~80%% "
                              "by -35 to -40deg and plateauing -- see "
                              "sample_dual_arm_configs' docstring.")
    parser.add_argument("--max-swing-deg", type=float, default=15.0)
    parser.add_argument("--min-swing-deg", type=float, default=5.0,
                         help="Reject/resample target draws with |target-init| "
                              "below this per joint (no floor means expected "
                              "swing is only max_swing_deg/2).")
    parser.add_argument("--collision-waypoints", type=int, default=9)
    parser.add_argument("--joint-limit-tol-deg", type=float, default=3.0,
                         help="Joint-limit acceptance allowance, NOT numerical slop -- "
                              "the controller's own joint-limit constraint is soft "
                              "(nmpc.yaml joint_slack_linear/quadratic), so small "
                              "transient overshoots (empirically ~0.2-2deg) are "
                              "by-design, not a real violation. See check_joint_limits.")
    parser.add_argument("--pos-tol", type=float, default=0.05)
    parser.add_argument("--vel-tol", type=float, default=0.05)
    parser.add_argument("--convergence-window", type=float, default=1.0)
    args = parser.parse_args()

    sys.exit(run(args))
