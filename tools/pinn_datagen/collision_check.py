"""
Raw-mujoco collision-checking for tools/pinn_datagen/, against the SAME model
pat_simulation actually runs (air_bearing_table.xml). Now that arm geoms are
contype=2 (see air_bearing_table.xacro), a nonzero data.ncon here reflects
genuine, would-actually-happen contact -- not a vacuous check.

The base is 3 independent scalar joints (not a <freejoint>), and every
joint's qpos0 default is 0 (see helpers.py's module docstring) -- so
set_joint_config's "data.qpos[:] = 0.0 then write named joints" makes the
prefilter's checked state bit-for-bit identical to what
MuJoCoSim::setJointPositions actually applies at launch, as long as callers
pass the same joint values pat_simulation will actually be given -- e.g.
generate_pinn_trajectory.py's prefilter_configs() always includes
helpers.PARKED_TARGET_QPOS alongside the sampled arm angles. There's no
free-joint base pose to special-case the way VORTEX's own collision_check.py
does.
"""

import mujoco
import numpy as np


def _load_model(mjcf_path: str):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    return model, data


def set_joint_config(model, data, joint_angles: dict) -> None:
    """Write `joint_angles` (full joint name -> angle [rad]) into data.qpos
    in place, with every other joint at 0. Does NOT call mj_forward --
    callers decide when the (cheap, but not free) forward/collision pass
    runs.
    """
    data.qpos[:] = 0.0
    for name, angle in joint_angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"no joint named '{name}' in the model")
        data.qpos[model.jnt_qposadr[jid]] = angle


def _contacts_report(model, data) -> list:
    """Return [(bodyA_name, bodyB_name), ...] for every active contact after
    the most recent mj_forward()/mj_collision() call."""
    report = []
    for i in range(data.ncon):
        con = data.contact[i]
        body1 = model.body(model.geom_bodyid[con.geom1]).name
        body2 = model.body(model.geom_bodyid[con.geom2]).name
        report.append((body1, body2))
    return report


def config_collision_report(mjcf_path: str, joint_angles: dict) -> list:
    """Static single-config check: build a fresh model/data, set the given
    joint config, run mj_forward (which runs collision detection as part of
    mj_fwdPosition), and report any contacts found."""
    model, data = _load_model(mjcf_path)
    set_joint_config(model, data, joint_angles)
    mujoco.mj_forward(model, data)
    return _contacts_report(model, data)


def is_config_collision_free(mjcf_path: str, joint_angles: dict) -> bool:
    return len(config_collision_report(mjcf_path, joint_angles)) == 0


def interpolated_configs_collision_free(mjcf_path: str, start: dict, end: dict,
                                         n_waypoints: int = 9) -> bool:
    """Fast PRE-FILTER: linearly interpolate every joint present in `start`
    (same key set as `end` -- pass the COMBINED dict of all joints across
    both arms, so arm-vs-arm collisions are checked too, not just
    arm-vs-bus) at n_waypoints values of alpha in [0, 1] inclusive,
    rejecting on the first collision found. This is a cheap proxy for the
    transient path, NOT a substitute for trajectory_collision_report below:
    a real NMPC transient can swing wider than a straight-line joint-space
    interpolation, and with two independently-controlled arms the real
    sim's per-arm convergence rates need not match (this interpolation
    moves both arms in lockstep by construction) -- the post-hoc replay
    check is the actual acceptance authority; this is purely to avoid
    wasting sim time on samples that are already doomed at the kinematic
    level.
    """
    model, data = _load_model(mjcf_path)
    names = list(start.keys())
    for alpha in np.linspace(0.0, 1.0, n_waypoints):
        cfg = {n: (1 - alpha) * start[n] + alpha * end[n] for n in names}
        set_joint_config(model, data, cfg)
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            return False
    return True


def trajectory_collision_report(mjcf_path: str, qpos_history) -> list:
    """POST-HOC, rigorous check: replay a RECORDED FULL qpos history
    (helpers.build_full_qpos_qvel's output, shape (n, model.nq)) through
    mj_forward at every step on a fresh raw model/data, and collect any
    contacts found. This is the real acceptance authority for a generated
    trajectory -- the pre-filter functions above only avoid wasting sim time
    on samples doomed before the sim even runs.

    Returns [(step_index, [(bodyA, bodyB), ...]), ...] for every step with
    at least one contact; empty list if the whole trajectory is clean.
    """
    qpos_history = np.asarray(qpos_history)
    model, data = _load_model(mjcf_path)
    violations = []
    for step in range(qpos_history.shape[0]):
        data.qpos[:] = qpos_history[step]
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            violations.append((step, _contacts_report(model, data)))
    return violations


def is_trajectory_collision_free(mjcf_path: str, qpos_history) -> bool:
    return len(trajectory_collision_report(mjcf_path, qpos_history)) == 0
