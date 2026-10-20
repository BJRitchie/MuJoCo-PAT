#!/usr/bin/env python3
"""Score the arms' TRUE end-effector tracking, whatever base state they were given.

Each arm node publishes the EE pose it computes from the base state it reads
(/chaser/odom, or nav_node's estimate on /chaser/nav/odom) and the joint angles.
That pose is in the frame the arm BELIEVES the base is in, so tracking scored
on it hides the estimator's error entirely. This recovers the true EE position
and scores that, alongside the believed one:

  the arm computes   ee_pose = (p_hat + anchor) + R(th_hat) * r(q)
  so its reach is    r       = R(th_hat)^T (ee_pose - p_hat - anchor)
  and the true EE is ee_true = (p + anchor) + R(th) * r

with (p_hat, th_hat) the base state the arm read, (p, th) the simulator's
ground truth, and anchor the node's base_anchor_x/y. r depends only on the
joint angles, so this is exact whenever the joint encoders are -- as they are
in both the default and the flight sensor profiles.

    tools/nav_score/score_tracking.py --seconds 40
    tools/nav_score/score_tracking.py --seconds 40 --base-topic /chaser/nav/odom

Run it alongside a live stack with waypoint targets (with_targets:=true); it
only subscribes. Truth comes from /sim_only/ground_truth when the simulator
publishes it (sensors_flight.yaml turns it on). With no ground truth it falls
back to /chaser/odom as truth, which is correct only when the sensors are
perfect, and says so.

Reported per arm, after --skip seconds:
  true err     |ee_true - setpoint|   what the arm actually achieves
  believed err |ee_pose - setpoint|   what the arm thinks it achieves
  est-induced  |ee_true - ee_pose|    EE error caused by the base estimate alone
"""
import argparse
import bisect
import math
import statistics
import sys

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
except ImportError:
    sys.exit("needs a sourced ROS 2 environment (source install/setup.bash)")


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Series:
    """A time-sorted (t, value) series with nearest-stamp lookup."""

    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        # Messages arrive in stamp order per topic; guard anyway.
        if self.t and t < self.t[-1]:
            i = bisect.bisect(self.t, t)
            self.t.insert(i, t)
            self.v.insert(i, v)
        else:
            self.t.append(t)
            self.v.append(v)

    def nearest(self, t, tol):
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t)):
                best = j
        return self.v[best] if abs(self.t[best] - t) <= tol else None


class Recorder(Node):
    def __init__(self, base_topic):
        super().__init__("tracking_score")
        qos = QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE)
        self.truth = Series()       # (x, y, yaw) from ground truth
        self.odom = Series()        # /chaser/odom, the fallback truth
        self.base = Series()        # what the arms read
        self.ee = {"left": [], "right": []}       # (t, x, y)
        self.sp = {"left": None, "right": None}   # latest setpoint (x, y)
        self.create_subscription(JointState, "/sim_only/ground_truth", self.on_truth, qos)
        self.create_subscription(Odometry, "/chaser/odom", self.on_odom, qos)
        if base_topic != "/chaser/odom":
            self.create_subscription(Odometry, base_topic, self.on_base, qos)
        self.base_is_odom = base_topic == "/chaser/odom"
        for side in ("left", "right"):
            self.create_subscription(PoseStamped, f"/chaser/arm/{side}/ee_pose",
                                     lambda m, s=side: self.on_ee(s, m), qos)
            self.create_subscription(PoseStamped, f"/chaser/arm/{side}/ee_setpoint",
                                     lambda m, s=side: self.on_sp(s, m), qos)

    def on_truth(self, m):
        idx = {n: i for i, n in enumerate(m.name)}
        if "chaser.x" in idx:
            self.truth.add(stamp_s(m), (m.effort[idx["chaser.x"]], m.effort[idx["chaser.y"]],
                                        m.effort[idx["chaser.yaw"]]))

    def on_odom(self, m):
        v = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        self.odom.add(stamp_s(m), v)
        if self.base_is_odom:
            self.base.add(stamp_s(m), v)

    def on_base(self, m):
        self.base.add(stamp_s(m), (m.pose.pose.position.x, m.pose.pose.position.y,
                                   yaw_of(m.pose.pose.orientation)))

    def on_sp(self, side, m):
        self.sp[side] = (m.pose.position.x, m.pose.position.y)

    def on_ee(self, side, m):
        # Paired with the setpoint in force when it arrived: the waypoint loop
        # steps every few seconds, so the latest one is the one being tracked.
        if self.sp[side] is not None:
            self.ee[side].append((stamp_s(m), m.pose.position.x, m.pose.position.y, self.sp[side]))


def summarise(rec, anchor, skip, truth_src):
    truth = rec.truth if truth_src == "ground_truth" else rec.odom
    t0 = None
    for side in ("left", "right"):
        if rec.ee[side]:
            t0 = rec.ee[side][0][0] if t0 is None else min(t0, rec.ee[side][0][0])
    if t0 is None:
        print("no ee_pose received -- is the stack running with targets?")
        return
    ax, ay = anchor
    print(f"truth from {truth_src}; anchor ({ax}, {ay}); skipping the first {skip:.0f} s")
    print(f"  {'arm':6} {'n':>5} {'true err mean':>14} {'p95':>8} {'believed mean':>14} "
          f"{'p95':>8} {'est-induced mean':>17} {'max':>8}   [mm]")
    for side in ("left", "right"):
        true_e, bel_e, ind = [], [], []
        for (t, ex, ey, (sx, sy)) in rec.ee[side]:
            if t - t0 < skip:
                continue
            b = rec.base.nearest(t, 0.02)
            p = truth.nearest(t, 0.02)
            if b is None or p is None:
                continue
            bx, by, bth = b
            px, py, pth = p
            # The arm's body-frame reach, then the same reach on the true base.
            dx, dy = ex - (bx + ax), ey - (by + ay)
            c, s = math.cos(bth), math.sin(bth)
            rx, ry = c * dx + s * dy, -s * dx + c * dy
            c, s = math.cos(pth), math.sin(pth)
            tx, ty = (px + ax) + c * rx - s * ry, (py + ay) + s * rx + c * ry
            true_e.append(math.hypot(tx - sx, ty - sy) * 1e3)
            bel_e.append(math.hypot(ex - sx, ey - sy) * 1e3)
            ind.append(math.hypot(tx - ex, ty - ey) * 1e3)
        if not true_e:
            print(f"  {side:6} no aligned samples")
            continue
        p95 = lambda v: sorted(v)[int(0.95 * (len(v) - 1))]
        print(f"  {side:6} {len(true_e):5d} {statistics.mean(true_e):14.2f} {p95(true_e):8.2f} "
              f"{statistics.mean(bel_e):14.2f} {p95(bel_e):8.2f} {statistics.mean(ind):17.3f} "
              f"{max(ind):8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--skip", type=float, default=8.0, help="startup seconds not scored")
    ap.add_argument("--base-topic", default="/chaser/odom",
                    help="the base state the arms read (their odom_topic)")
    ap.add_argument("--anchor", type=float, nargs=2, default=(-0.5, 0.0),
                    help="arm_common.yaml base_anchor_x base_anchor_y")
    a = ap.parse_args()

    rclpy.init()
    rec = Recorder(a.base_topic)
    end = None
    while rclpy.ok():
        rclpy.spin_once(rec, timeout_sec=0.1)
        now = rec.get_clock().now().nanoseconds * 1e-9
        if end is None:
            end = now + a.seconds
        if now >= end:
            break
    truth_src = "ground_truth" if rec.truth.t else "/chaser/odom (valid only with perfect sensors)"
    summarise(rec, a.anchor, a.skip, "ground_truth" if rec.truth.t else "odom")
    if not rec.truth.t:
        print(f"  note: truth taken from {truth_src}")
    rec.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
