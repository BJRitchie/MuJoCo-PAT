"""
Raw-mujoco (no ROS, no rclpy) forward-kinematics and joint-range utilities for
tools/pinn_datagen/. Loads a fresh mujoco.MjModel/MjData directly off the
installed air_bearing_table.xml -- this repo's controller and sim run as
separate ROS 2 node processes (unlike VORTEX's Basilisk MJScene, whose SWIG
wrapper %ignores getMujocoModel()/getMujocoData(), hiding mj_fullM/qfrc_bias/
data.ncon from Python), so there's no in-process object to reuse; loading the
XML directly is simply the natural way to do FK/dynamics work here.

Model facts baked into this module (see CLAUDE.md / air_bearing_table.xacro):
  - the base is 3 independent scalar joints (chaser_x, chaser_y slide;
    chaser_yaw hinge) -- NOT a <freejoint> -- so there is no base pos/quat to
    special-case the way VORTEX's target_ee_position() does for its
    free-floating servicer. qpos0 is all-zero for every joint in this model
    (no <keyframe>, confirmed in mujoco_sim.cpp) EXCEPT that
    config/simulation.yaml now sets a nonzero `initial_arm_qpos` default
    (small elbow bend, to keep the arms out of self-collision now that
    air_bearing_table.xacro's arm_contype=2 makes that collision real) --
    using model.qpos0 directly (rather than hardcoding zeros) is correct
    regardless of that default, since qpos0 only reflects the MJCF's own
    joint defaults, which are still all-zero; per-trial scripts always
    explicitly set all 6 arm angles anyway, so this only matters for the
    remaining 3 base dof, which stay at their true qpos0 (zero) -- the
    target's 3 dof are handled separately (PARKED_TARGET_QPOS, applied via
    `initial_target_qpos`), not left at qpos0.
  - gravity is already zeroed in the MJCF's own <option> (unlike VORTEX,
    which has to force it in Basilisk's MJSpec loader) -- no code-side
    override is needed here.
  - every EE (ee_site_L/ee_site_R) rotation is exactly about world Z for any
    arm configuration (piper_planar_arm.xacro's own guarantee -- joints
    1/4/6 are welded) -- so orientation is fully captured by a single yaw
    angle, extracted as 2*atan2(qz, qw) after mju_mat2Quat on site_xmat.
    This is also the only orientation form ee_target_publisher_node's
    `absolute` mode actually accepts ([x, y, yaw] triples), so there's no
    need to port VORTEX's full-quaternion target_ee_orientation().
"""

import os

import mujoco
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

MODEL_JOINT_NAMES = ["joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"]
JOINT_NAMES_L = ["joint2_L", "joint3_L", "joint5_L"]
JOINT_NAMES_R = ["joint2_R", "joint3_R", "joint5_R"]
BASE_JOINT_NAMES = ["chaser_x", "chaser_y", "chaser_yaw"]
TARGET_JOINT_NAMES = ["target_x", "target_y", "target_yaw"]
# Parked well outside any sampled arm's reach (mount-to-EE max reach is
# ~0.63m; this is >1.1m from either mount) so a trial's random configs can
# never touch it -- keeps it inside the table's walls (+-1.05) for a sane
# --visualise view. Applied to the live sim via pat_simulation's
# `initial_target_qpos` param; also used wherever this module/collision_check
# build a joint-config dict, so every check stays consistent with where the
# target actually is at launch.
PARKED_TARGET_QPOS = {"target_x": 0.75, "target_y": 0.75, "target_yaw": 0.0}
# Per-arm joint suffix stripped -- both arms share one range table (see
# load_joint_ranges' own consistency check).
BASE_NAMES = ["joint2", "joint3", "joint5"]


def default_mjcf_path() -> str:
    return os.path.join(get_package_share_directory("pat_simulation"),
                         "models", "environment", "air_bearing_table.xml")


def default_nmpc_yaml_path() -> str:
    return os.path.join(get_package_share_directory("pat_arm_nmpc"),
                         "config", "nmpc.yaml")


def load_joint_ranges(nmpc_yaml_path: str = None) -> dict:
    """Returns {base_name: (qmin, qmax)} for base_name in BASE_NAMES, read
    from nmpc.yaml's own limits.q_min/q_max -- the single source of truth for
    this arm's physical joint limits (matches piper_planar_arm.xacro's own
    <joint range=...> values), rather than a second hardcoded table that
    could silently drift from it.

    Asserts pat_arm_nmpc_left and pat_arm_nmpc_right agree (they do today) --
    raises loudly rather than silently picking one side, since a real
    per-arm difference would mean callers need per-arm ranges, not one
    shared table.
    """
    path = nmpc_yaml_path or default_nmpc_yaml_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def ranges_for(node_name):
        lims = cfg[node_name]["ros__parameters"]["limits"]
        qmin, qmax = lims["q_min"], lims["q_max"]
        return {base: (float(qmin[i]), float(qmax[i])) for i, base in enumerate(BASE_NAMES)}

    left = ranges_for("pat_arm_nmpc_left")
    right = ranges_for("pat_arm_nmpc_right")
    if left != right:
        raise ValueError(
            f"load_joint_ranges: pat_arm_nmpc_left/_right limits differ "
            f"({left} vs {right}) -- this sampler assumes one shared "
            f"per-base-joint range table; update it to be per-arm if that's "
            f"now intentional.")
    return left


def _load_model(mjcf_path: str = None):
    model = mujoco.MjModel.from_xml_path(mjcf_path or default_mjcf_path())
    data = mujoco.MjData(model)
    return model, data


def _site_id(model, site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"no <site> named '{site_name}' in the model")
    return site_id


def _joint_qposadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"no joint named '{name}' in the model")
    return model.jnt_qposadr[jid]


def _joint_dofadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"no joint named '{name}' in the model")
    return model.jnt_dofadr[jid]


def target_ee_pose(mjcf_path: str, site_name: str, joint_angles: dict):
    """joint_angles: {full joint name (e.g. 'joint2_L'): angle [rad]} for
    ONE arm's 3 joints. Returns (pos_xy: np.ndarray(2), yaw: float) -- the
    exact [x, y, yaw] triple ee_target_publisher_node's `absolute` mode
    wants.
    """
    model, data = _load_model(mjcf_path)
    data.qpos[:] = model.qpos0
    for name, angle in joint_angles.items():
        data.qpos[_joint_qposadr(model, name)] = angle
    mujoco.mj_forward(model, data)

    site_id = _site_id(model, site_name)
    pos_xy = data.site_xpos[site_id][:2].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    yaw = 2.0 * float(np.arctan2(quat[3], quat[0]))  # [w,x,y,z] -> 2*atan2(qz,qw)
    return pos_xy, yaw


def ee_xy_trajectory_from_qpos(mjcf_path: str, site_name: str, qpos_history: np.ndarray) -> np.ndarray:
    """Replays a recorded FULL (model.nq-length) qpos history through FK at
    every step. Returns (n, 2) EE world-frame [x, y] -- z is dropped since
    the planar task-space law never uses it (arm_nmpc.cpp's controlLaw:
    `desiredPos[2] (Z) unused`).
    """
    model, data = _load_model(mjcf_path)
    site_id = _site_id(model, site_name)
    qpos_history = np.asarray(qpos_history)
    n = qpos_history.shape[0]
    ee_xy = np.zeros((n, 2))
    for i in range(n):
        data.qpos[:] = qpos_history[i]
        mujoco.mj_forward(model, data)
        ee_xy[i] = data.site_xpos[site_id][:2]
    return ee_xy


def build_full_qpos_qvel(model, base_pose: np.ndarray, base_vel: np.ndarray,
                          arm_q: np.ndarray, arm_qdot: np.ndarray):
    """base_pose/base_vel: (n,3) [x,y,theta]/[vx,vy,omega]. arm_q/arm_qdot:
    (n,6), MODEL_JOINT_NAMES order. Returns (qpos_hist, qvel_hist), each
    (n, model.nq)/(n, model.nv) -- target's qpos slots are held at
    PARKED_TARGET_QPOS for every step; its 3 qvel slots stay exactly 0.

    This is EXACT, not approximate: target has no actuator, gravity is
    zeroed, and it starts at rest, parked outside every sampled config's
    reach (PARKED_TARGET_QPOS), so its true acceleration is identically
    zero at every step -- it provably never moves. This is why the recorder
    doesn't subscribe to /target/odom at all.
    """
    n = base_pose.shape[0]
    qpos = np.zeros((n, model.nq))
    qvel = np.zeros((n, model.nv))

    for i, name in enumerate(BASE_JOINT_NAMES):
        qpos[:, _joint_qposadr(model, name)] = base_pose[:, i]
        qvel[:, _joint_dofadr(model, name)] = base_vel[:, i]
    for name, angle in PARKED_TARGET_QPOS.items():
        qpos[:, _joint_qposadr(model, name)] = angle
    for i, name in enumerate(MODEL_JOINT_NAMES):
        qpos[:, _joint_qposadr(model, name)] = arm_q[:, i]
        qvel[:, _joint_dofadr(model, name)] = arm_qdot[:, i]
    return qpos, qvel
