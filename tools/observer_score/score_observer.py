#!/usr/bin/env python3
"""Score the disturbance observer against the simulator's ground truth.

The observer publishes an estimate; `pat_simulation` publishes what actually
happened. This records both, aligns them, and reports how close they are --
which is the difference between "the observer looks plausible" and a number.

    tools/observer_score/score_observer.py --seconds 20 --arm left
    tools/observer_score/score_observer.py --seconds 20 --csv /tmp/run.csv

Needs `publish_ground_truth: true` on pat_simulation and `observer: "momentum"`
on the arm node. Run it alongside a live stack; it only subscribes.

What gets compared, and why it is the raw rows and not the reduced ones:
MuJoCo's qfrc_constraint is a generalised force on the model's own DOFs, so the
comparable quantity is the observer's FULL-system estimate (published as
`raw_<joint>`), not the base-eliminated value the controller acts on. The two
differ by the base-coupling term, and only the raw one has a ground truth.

The two models index their DOFs differently and that is not a detail:

  scene    (air_bearing_table)   chaser_x, chaser_y, chaser_yaw | 6 arm joints | target_x, target_y, target_yaw
  control  (pat_platform_planar) 6 freejoint base DOFs          | 6 arm joints

so the ARM rows line up by joint name while the base rows do not line up by
index at all. This matches the arm rows by name, which is what the JointState
`name` field is there for, and leaves the base comparison alone.
"""
import argparse
import csv
import math
import sys

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
except ImportError:
    sys.exit("needs a sourced ROS 2 environment (source install/setup.bash)")


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Recorder(Node):
    """Collects both streams with their own stamps; alignment happens after."""

    def __init__(self, arm):
        super().__init__("observer_score")
        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
        self.truth = []     # (t, {joint: qfrc})
        self.est = []       # (t, {joint: raw_estimate}, {joint: reduced})
        self.create_subscription(JointState, "/sim_only/ground_truth",
                                 self.on_truth, qos)
        self.create_subscription(JointState,
                                 f"/chaser/arm/{arm}/disturbance",
                                 self.on_est, qos)

    def on_truth(self, msg):
        # qfrc_constraint.<i> entries are indexed by the SCENE model's DOF, which
        # carries no joint names -- the arm joints are whichever rows they are.
        # Kept whole and resolved later against the estimate's own names.
        vals = {n: e for n, e in zip(msg.name, msg.effort)}
        self.truth.append((stamp_s(msg), vals))

    def on_est(self, msg):
        raw, red = {}, {}
        for n, e in zip(msg.name, msg.effort):
            if n.startswith("raw_"):
                raw[n[4:]] = e
            elif n.startswith("base_"):
                raw[n] = e
            else:
                red[n] = e
        self.est.append((stamp_s(msg), raw, red))


def nearest(series, t):
    """The sample closest in time to t, or None if the series is empty.

    Linear scan rather than a bisect: these runs are tens of seconds at 100 Hz
    and clarity is worth more here than the microseconds.
    """
    if not series:
        return None
    best, best_dt = None, float("inf")
    for s in series:
        dt = abs(s[0] - t)
        if dt < best_dt:
            best, best_dt = s, dt
    return best if best_dt < 0.05 else None


def stats(pairs, label):
    """Report how well `est` tracked `truth` over a list of (truth, est)."""
    if not pairs:
        print(f"  {label:<22} no overlapping samples")
        return
    tr = [p[0] for p in pairs]
    es = [p[1] for p in pairs]
    n = len(pairs)
    mean_t = sum(tr) / n
    mean_e = sum(es) / n
    # Peak magnitudes, because a contact is a transient: a mean over a run that
    # is mostly quiet says nothing about whether the event was caught.
    peak_t = max(tr, key=abs)
    peak_e = max(es, key=abs)
    # Correlation, which answers "did it move with the truth" independently of
    # whether it got the magnitude right.
    st = math.sqrt(sum((x - mean_t) ** 2 for x in tr))
    se = math.sqrt(sum((x - mean_e) ** 2 for x in es))
    cov = sum((tr[i] - mean_t) * (es[i] - mean_e) for i in range(n))
    corr = cov / (st * se) if st > 1e-12 and se > 1e-12 else float("nan")
    rms_err = math.sqrt(sum((tr[i] - es[i]) ** 2 for i in range(n)) / n)

    print(f"  {label:<22} n={n:<5} truth peak {peak_t:+8.4f}  "
          f"est peak {peak_e:+8.4f}  ratio {peak_e / peak_t if peak_t else float('nan'):6.2f}  "
          f"corr {corr:+5.2f}  rms err {rms_err:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--csv", help="write the aligned series here")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="seconds to discard at the start, while the observer's "
                         "integrator is still establishing itself (default 2)")
    ap.add_argument("--model-joints",
                    help="comma-separated model_joint_names in MJCF body-tree "
                         "order, e.g. joint2_L,joint3_L,joint5_L,joint2_R,"
                         "joint3_R,joint5_R. Needed to map the estimate's named "
                         "joints onto qfrc_constraint's index-named rows")
    ap.add_argument("--arm-row-offset", type=int, default=3,
                    help="scene DOF index of the first arm joint (default 3: "
                         "chaser_x, chaser_y, chaser_yaw come first)")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.arm)
    t_end = node.get_clock().now().nanoseconds * 1e-9 + args.seconds
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"\ncaptured {len(node.truth)} ground-truth and {len(node.est)} "
          f"estimate messages")
    if not node.truth:
        print("  no ground truth -- is publish_ground_truth true?")
    if not node.est:
        print(f"  no estimate -- is observer 'momentum' on the {args.arm} arm?")
    if not node.truth or not node.est:
        node.destroy_node()
        rclpy.shutdown()
        return 1

    # Which scene DOF rows correspond to the arm joints is not discoverable from
    # the ground-truth message, which names its entries by index. So the arm
    # rows are identified by scanning for the ones that move with contact: every
    # qfrc_constraint row that is ever non-zero is reported, and the operator
    # reads off the arm ones. Explicit rather than guessed, because a wrong
    # mapping would produce a confident and meaningless number.
    t0 = node.est[0][0] + args.settle
    active = {}
    for t, vals in node.truth:
        if t < t0:
            continue
        for k, v in vals.items():
            if k.startswith("qfrc_constraint."):
                active[k] = max(active.get(k, 0.0), abs(v))
    # ncon separates the two things qfrc_constraint mixes together. A row that
    # is nonzero only while ncon is 0 is a joint against its range limit, not a
    # collision -- and scoring a contact estimate against a joint limit would be
    # a confident, meaningless number.
    peak_ncon = max((v.get("ncon", 0.0) for t, v in node.truth if t >= t0),
                    default=0.0)
    nonzero = {k: v for k, v in active.items() if v > 1e-9}
    print(f"\npeak contact count: {peak_ncon:.0f}")
    print(f"qfrc_constraint rows that ever acted (peak |force|):")
    if not nonzero:
        print("  none -- nothing acted at all, so there is nothing to score. "
              "Check the target placement.")
    elif peak_ncon == 0:
        print("  NOTE: ncon stayed 0, so every row below is a JOINT LIMIT, not "
              "contact. qfrc_constraint carries both.")
    for k in sorted(nonzero, key=lambda s: int(s.split('.')[1])):
        print(f"  {k:<24} {nonzero[k]:.4f}")

    joints = sorted(node.est[0][2].keys())
    print(f"\nobserver estimate on {args.arm} arm joints ({', '.join(joints)}):")
    for j in joints:
        series = [(t, raw.get(j, 0.0), red.get(j, 0.0))
                  for t, raw, red in node.est if t >= t0]
        if not series:
            continue
        raws = [s[1] for s in series]
        reds = [s[2] for s in series]
        print(f"  {j:<12} raw peak {max(raws, key=abs):+8.4f}  "
              f"reduced peak {max(reds, key=abs):+8.4f}")

    base = [(t, raw.get("base_fx", 0.0), raw.get("base_fy", 0.0),
             raw.get("base_tz", 0.0)) for t, raw, _ in node.est if t >= t0]
    if base:
        print(f"\nbase wrench estimate: fx peak {max((b[1] for b in base), key=abs):+8.4f}  "
              f"fy peak {max((b[2] for b in base), key=abs):+8.4f}  "
              f"tz peak {max((b[3] for b in base), key=abs):+8.4f}")

    # --- The comparison ---------------------------------------------------- #
    # Mapping the estimate's named joints onto the ground truth's index-named
    # rows. The scene model's DOF order is chaser_x, chaser_y, chaser_yaw, then
    # the arm joints in MJCF body-tree order, then target_x/y/yaw -- so the arm
    # block starts at row 3 and runs in simulation.yaml's arm_joint_names order.
    # Printed rather than assumed, and overridable, because a wrong mapping
    # yields a confident and meaningless number. Cross-check it against the
    # "rows that ever acted" list above.
    model_joints = args.model_joints.split(",") if args.model_joints else []
    if not model_joints:
        print("\n--model-joints not given, so the named-to-indexed mapping "
              "cannot be made; the peaks above are still valid on their own.")
    else:
        print(f"\nscoring with arm block at row {args.arm_row_offset}:")
        aligned = []
        for j in joints:
            if j not in model_joints:
                print(f"  {j}: not in --model-joints, skipped")
                continue
            row = f"qfrc_constraint.{args.arm_row_offset + model_joints.index(j)}"
            pairs = []
            for t, raw, _ in node.est:
                if t < t0:
                    continue
                tr = nearest(node.truth, t)
                if tr is None or row not in tr[1]:
                    continue
                pairs.append((tr[1][row], raw.get(j, 0.0)))
                aligned.append((t, j, tr[1][row], raw.get(j, 0.0)))
            stats(pairs, f"{j} vs {row}")

        if args.csv and aligned:
            with open(args.csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_s", "joint", "qfrc_constraint", "estimate_raw"])
                w.writerows(aligned)
            print(f"\nwrote {len(aligned)} aligned samples to {args.csv}")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
