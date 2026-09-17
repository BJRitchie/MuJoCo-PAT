"""
Post-hoc extraction of the full-system generalized mass matrix, bias vector,
and acceleration (qddot) for a recorded trajectory, via a raw-mujoco replay
of the SAME model pat_simulation runs (air_bearing_table.xml).

Unlike VORTEX's dynamics_extract.py (which has to reach around Basilisk's
MJScene SWIG wrapper, since it %ignores getMujocoModel()/getMujocoData()),
this repo has no such barrier -- pat_simulation's own C++ node already talks
to mjModel/mjData directly. This module just does the same kind of thing
from Python, against a recorded trajectory instead of a live sim.

STATE_ORDER is this model's real nv-order (chaser's own 3 joints, then its
nested arm subtree's 6 joints, then the sibling target body's 3 joints --
see air_bearing_table.xacro's body nesting): 12 dof total.
compute_full_system_dynamics returns only the first N_SYSTEM_DOF=9 rows/cols
(base + both arms) -- target's block is PROVABLY zero for any trajectory
that has already passed the collision-free check (see
helpers.build_full_qpos_qvel's docstring), so it's dropped rather than
shipped as physically-meaningless padding. This 9-dof system (3 planar base
+ 6 arm joints) is the actual real-hardware-faithful dynamics this dataset
targets -- see pat_arm_nmpc's own internal model (pat_platform_planar.xacro,
a 6-dof free-floating base) for the idealized approximation the controller
uses instead, which this dataset is deliberately NOT reproducing.

bias_hist = qfrc_bias - qfrc_passive, NOT qfrc_bias alone: THIS model's base
joints have real nonzero damping (chaser_x/y: 0.05, chaser_yaw: 0.02 --
air_bearing_table.xacro), unlike VORTEX's fully-undamped servicer -- so
qfrc_passive is genuinely nonzero here and this correction is load-bearing,
not defensive boilerplate. The identity tau_full = M @ qacc + bias_full
holds to near machine precision at every step of a trajectory with zero
contact force throughout (qfrc_constraint == 0) -- which is exactly why this
must only ever be called on an ALREADY-validated, collision-free trajectory;
if it weren't, a step with nonzero qfrc_constraint would silently break this
identity (M @ qacc + bias would be missing the constraint-force term).

Gravity is already zeroed in the MJCF's own <option> (unlike VORTEX, which
must force it in code) -- no override needed here.
"""

import mujoco
import numpy as np

STATE_ORDER = [
    "chaser_x", "chaser_y", "chaser_yaw",
    "joint2_L", "joint3_L", "joint5_L",
    "joint2_R", "joint3_R", "joint5_R",
    "target_x", "target_y", "target_yaw",
]
N_SYSTEM_DOF = 9  # base(3) + both arms(6); target's rows are provably zero, dropped


def compute_full_system_dynamics(mjcf_path: str, qpos_history, qvel_history, tau_arm_history):
    """qpos_history/qvel_history: (n, model.nq)/(n, model.nv) --
    helpers.build_full_qpos_qvel's output. tau_arm_history: (n, 6),
    helpers.MODEL_JOINT_NAMES order (the recorded arm torques only --
    thrusters are never commanded in this scenario, so their 4 ctrl slots
    are zero throughout).

    Returns (mass_hist, bias_hist, qddot_hist): (n,9,9), (n,9), (n,9) -- see
    STATE_ORDER[:9] for row/col order.
    """
    qpos_history = np.asarray(qpos_history)
    qvel_history = np.asarray(qvel_history)
    tau_arm_history = np.asarray(tau_arm_history)

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    nv = model.nv
    n = qpos_history.shape[0]
    mass_full = np.empty((n, nv, nv))
    bias_full = np.empty((n, nv))
    qddot_full = np.empty((n, nv))
    full_M = np.empty((nv, nv))

    # data.ctrl order must match the <actuator> block exactly: 4 thrusters
    # (thr_fwd, thr_aft, thr_port, thr_stbd) then the 6 arm motors
    # (motor2_L, motor3_L, motor5_L, motor2_R, motor3_R, motor5_R) -- see
    # air_bearing_table.xacro.
    for step in range(n):
        data.qpos[:] = qpos_history[step]
        data.qvel[:] = qvel_history[step]   # required for correct Coriolis/centrifugal terms
        data.ctrl[:4] = 0.0
        data.ctrl[4:10] = tau_arm_history[step]  # required for qacc to reflect the real applied torque
        mujoco.mj_forward(model, data)

        mujoco.mj_fullM(model, full_M, data.qM)
        mass_full[step] = full_M
        bias_full[step] = data.qfrc_bias - data.qfrc_passive
        qddot_full[step] = data.qacc

    idx = slice(0, N_SYSTEM_DOF)
    mass_hist = mass_full[:, idx, idx]
    bias_hist = bias_full[:, idx]
    qddot_hist = qddot_full[:, idx]
    return mass_hist, bias_hist, qddot_hist
