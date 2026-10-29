#!/usr/bin/env python3
"""Record one arm's run on the live stack in the bench's per-tick series format.

Subscribes to the topics the deployed nodes already publish and, when stopped
(SIGINT/SIGTERM), writes the same CSV columns as `contact_bench --series`, so
`series_summary` scores it with the bench's own metrics. Everything is stamped
with the simulator's clock, so what is scored is what happened in the plant, not
when this process got to run.

  recorder.py --side left --out run.csv [--base-topic /chaser/nav/odom]

One row per joint_states message (the plant's 100 Hz publish, one per control
period) from the first trajectory window on:

  true EE     forward kinematics of the MuJoCo model at the plant's TRUE base
              pose (/sim_only/ground_truth) and the measured joint angles, which
              is what the bench scores (the arm's own ee_pose uses the base state
              it believed, and would hide an estimator's error)
  reference   the arm's trajectory window, interpolated to the row's stamp
  torque      the arm's published torque_command in force at that stamp
  est error   the base state the arm read (--base-topic) minus the truth

tick_us and mode come from the node's own per-tick trace (`trace_file`), which
the driver joins afterwards with `join_trace`. That needs the stack to have exited
first (so the trace is flushed) and this recorder to have outlived it (so its last
torque message is the node's last tick): stop targets, stop the stack, then stop
the recorder.
"""
import argparse
import bisect
import csv
import json
import math
import os
import signal
import sys

import numpy as np

try:
    import mujoco
    import rclpy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import MultiDOFJointTrajectory
except ImportError as e:  # pragma: no cover
    sys.exit(f"needs a sourced ROS 2 environment and python mujoco: {e}")

#: The model's joints, in body-tree order (arm_common.yaml's model_joint_names).
MODEL_JOINTS = ["joint2_L", "joint3_L", "joint5_L", "joint2_R", "joint3_R", "joint5_R"]


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def yaw_of_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_of_wxyz(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Recorder(Node):
    def __init__(self, args):
        super().__init__(f"live_recorder_{args.side}")
        self.args = args
        self.suffix = "_L" if args.side == "left" else "_R"
        qos = QoSProfile(depth=5000, reliability=ReliabilityPolicy.RELIABLE)
        self.joint_states = []        # (ns, {name: (q, v)})
        self.truth = {}               # ns -> (x, y, yaw, vx, vy, wz, ncon)
        self.base = []                # (ns, x, y, yaw, vx, vy, wz): what the arm read
        self.torque = []              # (ns_arrival_sim, {name: effort}), arrival order
        self.windows = []             # (t0_s, times, x, y, vx, vy)
        self.setpoints = []           # (ns, x, y)
        self.t_sim = 0                # latest joint_states stamp, to date torque messages
        side = args.side
        self.create_subscription(JointState, "/chaser/arm/joint_states", self.on_joints, qos)
        self.create_subscription(JointState, "/chaser/arm/torque_command", self.on_torque, qos)
        self.create_subscription(JointState, "/sim_only/ground_truth", self.on_truth, qos)
        self.create_subscription(Odometry, args.base_topic, self.on_base, qos)
        self.create_subscription(MultiDOFJointTrajectory, f"/chaser/arm/{side}/ee_trajectory",
                                 self.on_window, qos)
        self.create_subscription(PoseStamped, f"/chaser/arm/{side}/ee_setpoint",
                                 self.on_setpoint, qos)

    def on_joints(self, m):
        ns = stamp_ns(m)
        self.t_sim = ns
        self.joint_states.append((ns, {n: (p, v) for n, p, v in
                                       zip(m.name, m.position, m.velocity)}))

    def on_torque(self, m):
        # Both arms publish on this topic, each only its own joints.
        if any(n.endswith(self.suffix) for n in m.name):
            self.torque.append((self.t_sim, dict(zip(m.name, m.effort))))

    def on_truth(self, m):
        e = dict(zip(m.name, m.effort))
        if "chaser.x" in e:
            self.truth[stamp_ns(m)] = (e["chaser.x"], e["chaser.y"], e["chaser.yaw"],
                                       e["chaser.vx"], e["chaser.vy"], e["chaser.omega"],
                                       int(e.get("ncon", 0)))

    def on_base(self, m):
        p, v = m.pose.pose, m.twist.twist
        self.base.append((stamp_ns(m), p.position.x, p.position.y, yaw_of_quat(p.orientation),
                          v.linear.x, v.linear.y, v.angular.z))

    def on_window(self, m):
        t0 = stamp_ns(m) * 1e-9
        pts = m.points
        self.windows.append((
            t0,
            np.array([t0 + p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts]),
            np.array([p.transforms[0].translation.x for p in pts]),
            np.array([p.transforms[0].translation.y for p in pts]),
            np.array([p.velocities[0].linear.x for p in pts]),
            np.array([p.velocities[0].linear.y for p in pts])))

    def on_setpoint(self, m):
        self.setpoints.append((stamp_ns(m), m.pose.position.x, m.pose.position.y))

    # ------------------------------------------------------------------ out --

    def write(self):
        a = self.args
        if not self.windows:
            sys.exit("recorder: no trajectory window was ever received, nothing to write")
        owned = [n for n in MODEL_JOINTS if n.endswith("_L" if a.side == "left" else "_R")]
        model = mujoco.MjModel.from_xml_path(a.model)
        data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, a.site)
        if site < 0:
            sys.exit(f"recorder: no site {a.site} in {a.model}")
        qadr = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in MODEL_JOINTS}

        # The task starts when the first window does; its stamp is the publisher's
        # clock less the look-back, so the shape's own time zero is that plus it.
        t_start = self.windows[0][0] + a.lookback
        base_ns = np.array([b[0] for b in self.base], dtype=np.int64)
        win_t0 = np.array([w[0] for w in self.windows])
        tq_ns = np.array([t[0] for t in self.torque], dtype=np.int64)

        rows, tick_of_row = [], {}
        yaw_start = None
        last_tau = {n: 0.0 for n in owned}
        tq_i = 0
        for ns, js in self.joint_states:
            t = ns * 1e-9
            if t < t_start or ns not in self.truth:
                continue
            if a.duration > 0.0 and t - t_start > a.duration:
                break
            x, y, yaw, vx, vy, wz, ncon = self.truth[ns]

            # Forward kinematics at the true base pose and the measured joints.
            data.qpos[:] = 0.0
            data.qpos[0], data.qpos[1] = x + a.anchor_x, y + a.anchor_y
            data.qpos[3], data.qpos[6] = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
            for n in MODEL_JOINTS:
                data.qpos[qadr[n]] = js[n][0]
            mujoco.mj_kinematics(model, data)
            ee = data.site_xpos[site]
            R = data.site_xmat[site].reshape(3, 3)
            ee_yaw = math.atan2(R[1, 0], R[0, 0])
            if yaw_start is None:
                yaw_start = ee_yaw

            # The reference at this stamp: the newest window that started by now,
            # interpolated between its samples.
            wi = max(0, bisect.bisect_right(win_t0, t) - 1)
            _, wt, wx, wy, wvx, wvy = self.windows[wi]
            rx, ry = np.interp(t, wt, wx), np.interp(t, wt, wy)
            rvx, rvy = np.interp(t, wt, wvx), np.interp(t, wt, wvy)

            # The torque in force: every torque_command published since the last
            # row counts as this row's tick; the last one is the value held.
            while tq_i < len(tq_ns) and tq_ns[tq_i] <= ns:
                last_tau = {n: self.torque[tq_i][1].get(n, last_tau[n]) for n in owned}
                tick_of_row[tq_i] = len(rows)
                tq_i += 1

            # The base state the arm read: the latest estimate not after this stamp.
            est = [math.nan] * 6
            if len(base_ns):
                bi = int(np.searchsorted(base_ns, ns, side="right")) - 1
                if bi >= 0:
                    _, bx, by, byaw, bvx, bvy, bwz = self.base[bi]
                    est = [bx - x, by - y, math.remainder(byaw - yaw, 2 * math.pi),
                           bvx - vx, bvy - vy, bwz - wz]
            rows.append([t - t_start, rx, ry, rvx, rvy, ee[0], ee[1],
                         math.hypot(ee[0] - rx, ee[1] - ry),
                         math.remainder(ee_yaw - yaw_start, 2 * math.pi)]
                        + [last_tau[n] for n in owned] + [js[n][1] for n in owned]
                        + [js[n][0] for n in owned]
                        + [ncon, x, y, yaw, wz, 0.0, 0] + est)

        n = len(owned)
        head = (["t_s", "ref_x", "ref_y", "ref_vx", "ref_vy", "ee_x", "ee_y", "err_m", "eth_rad"]
                + [f"tau{j}" for j in range(n)] + [f"qd{j}" for j in range(n)]
                + [f"q{j}" for j in range(n)]
                + ["ncon", "base_x", "base_y", "base_yaw", "base_wz", "tick_us", "mode"]
                + ["est_x", "est_y", "est_yaw", "est_vx", "est_vy", "est_wz"])
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(head)
            for r in rows:
                w.writerow([f"{v:.9g}" if isinstance(v, float) else v for v in r])
        os.replace(tmp, a.out)
        with open(a.out + ".ticks.json", "w") as fh:
            json.dump({"row_of_torque_msg": tick_of_row, "n_torque_msgs": len(self.torque),
                       "n_rows": len(rows), "t_start": t_start}, fh)
        self.get_logger().info(f"wrote {len(rows)} rows to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--side", choices=("left", "right"), default="left")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-topic", default="/chaser/nav/odom")
    ap.add_argument("--model", required=True, help="pat_platform_planar.xml")
    ap.add_argument("--site", default=None, help="default ee_site_L / ee_site_R")
    ap.add_argument("--anchor-x", type=float, default=-0.5)
    ap.add_argument("--anchor-y", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="keep rows only this long after the task starts (0 = all)")
    ap.add_argument("--lookback", type=float, default=0.1,
                    help="ee_target_publisher's window_lookback_s")
    args = ap.parse_args()
    args.site = args.site or ("ee_site_L" if args.side == "left" else "ee_site_R")

    rclpy.init()
    node = Recorder(args)
    stop = {"now": False}

    def on_signal(*_):
        stop["now"] = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    while rclpy.ok() and not stop["now"]:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.write()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
