"""
Data extraction and dual-format (HDF5 + CSV) saving for one PINN training
trajectory, produced by generate_pinn_trajectory.py.

No ROS/rclpy/mujoco dependency (numpy/h5py/pandas only) -- keeps this
importable from a plain analysis/training environment later, without needing
the full ROS 2/MuJoCo stack.

Single source of truth: `build_field_dict` is called once per trajectory,
and BOTH write_h5/write_csv consume that same dict, so the two files can
never disagree with each other.

State-vector convention: this repo's own documented `[px, py, theta, vx, vy,
omega]` (CLAUDE.md), applied to the chaser base -- NOT VORTEX's
free-floating-servicer 3D pos/quat convention, since this plant's base is
genuinely a planar 3-DOF joint set (see dynamics_extract.py's own module
docstring for why this 9-DOF (3 base + 6 arm) system, not a 12-DOF
free-floating one, is the actual target of this dataset).
"""

import h5py
import numpy as np
import pandas as pd

STATE_ORDER = [
    "base_x", "base_y", "base_yaw",
    "joint2_L", "joint3_L", "joint5_L",
    "joint2_R", "joint3_R", "joint5_R",
]


def build_field_dict(t, base_pose, base_vel, arm_q, arm_qdot, arm_tau,
                      ee_pos_L, ee_pos_R, mass9, bias9, qddot9) -> dict:
    """t: (n,). base_pose/base_vel: (n,3) [x,y,theta]/[vx,vy,omega].
    arm_q/arm_qdot/arm_tau: (n,6), helpers.MODEL_JOINT_NAMES order -- sliced
    here into the two arms' own (n,3) blocks. ee_pos_{L,R}: (n,2) [x,y]
    (helpers.ee_xy_trajectory_from_qpos's output). mass9/bias9/qddot9:
    (n,9,9)/(n,9)/(n,9) (dynamics_extract.compute_full_system_dynamics's
    output).
    """
    return {
        "time":         np.asarray(t),
        "base_pose":    np.asarray(base_pose),
        "base_vel":     np.asarray(base_vel),
        "arm_L_q":      np.asarray(arm_q)[:, 0:3],
        "arm_L_qdot":   np.asarray(arm_qdot)[:, 0:3],
        "arm_L_tau":    np.asarray(arm_tau)[:, 0:3],
        "arm_L_ee_pos": np.asarray(ee_pos_L),
        "arm_R_q":      np.asarray(arm_q)[:, 3:6],
        "arm_R_qdot":   np.asarray(arm_qdot)[:, 3:6],
        "arm_R_tau":    np.asarray(arm_tau)[:, 3:6],
        "arm_R_ee_pos": np.asarray(ee_pos_R),
        "mass":         np.asarray(mass9),    # (n,9,9), see STATE_ORDER
        "coriolis":     np.asarray(bias9),    # (n,9), see STATE_ORDER
        "qddot":        np.asarray(qddot9),   # (n,9), see STATE_ORDER
    }


def write_h5(path: str, fields: dict, meta: dict) -> None:
    """Layout:
        /time
        /base/{pose, vel}                 -- (n,3) each: [x,y,theta] / [vx,vy,omega]
        /arm_L/{q, qdot, tau, ee_pos}      -- (n,3),(n,3),(n,3),(n,2)
        /arm_R/{q, qdot, tau, ee_pos}
        /dynamics/{mass, coriolis, qddot}  -- (n,9,9), (n,9), (n,9) -- see attrs["state_order"]
    Root-level attrs carry everything that's constant for the whole
    trajectory (targets, seed, tolerances, timing, state_order, etc.) -- see
    `meta`'s keys as populated by generate_pinn_trajectory.py.

    Timing caveat (real, not boilerplate -- see also `meta["nominal_pub_rate_hz"]`):
    unlike VORTEX's Basilisk sim (a single deterministic process sampled at
    an exact fixed Ts, giving a perfect uniform time grid), this stack's
    recorder samples on ROS 2 message arrival driven by wall-clock timers
    (no /clock source anywhere in this repo), so `time` has real jitter --
    anyone finite-differencing q/t downstream must use the actual per-step
    dt_i = time[i]-time[i-1], not an assumed constant. The dynamics fields
    (mass/coriolis/qddot) are NOT affected by this jitter -- mj_forward's
    qacc is an instantaneous forward-dynamics solve given that step's
    qpos/qvel/ctrl, not a derivative. Also: recorded tau is "whatever
    torque_command was most recently cached" at the moment joint_states
    triggered a row, not sampled in perfect lockstep with q/qdot
    (torque_command can arrive at up to control_hz while joint_states
    publishes at pub_rate_hz) -- a small, bounded latency, not a bug.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("time", data=fields["time"])
        base = f.create_group("base")
        base.create_dataset("pose", data=fields["base_pose"])
        base.create_dataset("vel", data=fields["base_vel"])
        for side in ("L", "R"):
            g = f.create_group(f"arm_{side}")
            g.create_dataset("q", data=fields[f"arm_{side}_q"])
            g.create_dataset("qdot", data=fields[f"arm_{side}_qdot"])
            g.create_dataset("tau", data=fields[f"arm_{side}_tau"])
            g.create_dataset("ee_pos", data=fields[f"arm_{side}_ee_pos"])
        dyn = f.create_group("dynamics")
        dyn.create_dataset("mass", data=fields["mass"])
        dyn.create_dataset("coriolis", data=fields["coriolis"])
        dyn.create_dataset("qddot", data=fields["qddot"])
        f.attrs["state_order"] = STATE_ORDER
        for k, v in meta.items():
            f.attrs[k] = v


def write_csv(path: str, fields: dict) -> None:
    """Flat wide per-timestep table -- same arrays as write_h5, expanded to
    named scalar columns, built from the SAME `fields` dict (never a second
    independent extraction), so the .h5/.csv contents can never drift apart.
    Metadata is per-trajectory-constant and intentionally NOT duplicated
    into every row here -- it lives only in the .h5 attrs (this includes
    STATE_ORDER, needed to interpret the M_*/C_*/QDD_* columns below --
    consult the matching .h5's `state_order` attr).
    """
    cols = {"time": fields["time"]}

    base_pose, base_vel = fields["base_pose"], fields["base_vel"]
    for i, ax in enumerate(("x", "y", "yaw")):
        cols[f"base_pos_{ax}"] = base_pose[:, i]
    for i, ax in enumerate(("vx", "vy", "omega")):
        cols[f"base_vel_{ax}"] = base_vel[:, i]

    for side in ("L", "R"):
        for field, label in (("q", "q"), ("qdot", "qdot"), ("tau", "tau")):
            arr = fields[f"arm_{side}_{field}"]
            for j in range(arr.shape[1]):
                cols[f"{side}_{label}_j{j}"] = arr[:, j]
        ee = fields[f"arm_{side}_ee_pos"]
        for i, ax in enumerate(("x", "y")):
            cols[f"{side}_ee_{ax}"] = ee[:, i]

    # Zero-padded, underscore-delimited indices (not "M_{i}{j}") -- with a
    # 9x9 matrix, plain concatenation makes e.g. i=0,j=10 hard to eyeball
    # apart from i=1,j=0; "M_00_08" vs "M_01_00" isn't.
    mass = fields["mass"]
    for i in range(mass.shape[1]):
        for j in range(mass.shape[2]):
            cols[f"M_{i:02d}_{j:02d}"] = mass[:, i, j]
    coriolis = fields["coriolis"]
    for j in range(coriolis.shape[1]):
        cols[f"C_{j:02d}"] = coriolis[:, j]
    qddot = fields["qddot"]
    for j in range(qddot.shape[1]):
        cols[f"QDD_{j:02d}"] = qddot[:, j]

    pd.DataFrame(cols).to_csv(path, index=False)
