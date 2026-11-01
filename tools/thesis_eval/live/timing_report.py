#!/usr/bin/env python3
"""Where a live run's slow control ticks come from.

    python3 tools/thesis_eval/live/timing_report.py --out data/thesis_eval/live/main

Reads every finished trial's left-arm trace (trace_left.csv: one row per control
tick, the node's own timing and the QP's iterations and status) and classifies
each tick that overran the control period:

  failed solve   the QP ran to its iteration cap (qp_status != 0) or the
                 controller retried it; the tick then costs up to two full solves
  solved         the QP converged, and the tick was still late: a solve that
                 needed many iterations, or the machine taking the core away
  no solver      OMPC / SMC (closed form)

Prints, per controller: ticks, overruns and their causes, the worst tick of each
kind, the cost of one QP iteration, and the cells with the most overruns. The
trace covers the whole run, warm-up included, so the counts are the node's, not
just the scoring window's. Standard library only: runs in the container.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics


def trace_rows(path):
    with open(path) as fh:
        first = fh.readline()
        header = first if not first.startswith("#") else fh.readline()
        names = header.strip().split(",")
        for line in fh:
            if line.startswith("#"):
                continue
            yield dict(zip(names, line.strip().split(",")))


def num(r, k, default=0.0):
    try:
        return float(r.get(k, default))
    except (TypeError, ValueError):
        return default


def cell_of(spec):
    out = [spec["shape"]]
    if float(spec["speed"]) != 1.0:
        out.append(f"{float(spec['speed']):g}x")
    if spec["state"] != "ekf":
        out.append(spec["state"])
    if float(spec["damping"]):
        out.append(f"damp {float(spec['damping']):g}")
    for k, lab in (("mass_scale", "mass"), ("tau_scale", "tau")):
        if k in spec:
            out.append(f"{lab} {float(spec[k]):g}x")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/thesis_eval/live/main")
    ap.add_argument("--period-us", type=float, default=10000.0)
    ap.add_argument("--top", type=int, default=8, help="cells to list per controller")
    a = ap.parse_args()

    per = collections.defaultdict(lambda: collections.Counter())
    worst = collections.defaultdict(lambda: collections.defaultdict(float))
    per_iter = collections.defaultdict(list)
    by_cell = collections.defaultdict(collections.Counter)
    trials_with = collections.Counter()
    n_trials = collections.Counter()
    for d in sorted(glob.glob(os.path.join(a.out, "trials", "*"))):
        rep_p, tr_p = os.path.join(d, "report.json"), os.path.join(d, "trace_left.csv")
        if d.endswith(".partial") or not (os.path.isfile(rep_p) and os.path.isfile(tr_p)):
            continue
        spec = json.load(open(rep_p))["spec"]
        tag, cell = spec["tag"], cell_of(spec)
        n_trials[tag] += 1
        any_over = False
        for r in trace_rows(tr_p):
            t = num(r, "tick_us")
            if t <= 0:
                continue
            c = per[tag]
            c["ticks"] += 1
            has_qp = spec["law"] == "nmpc"
            failed = has_qp and (num(r, "qp_status") != 0 or num(r, "retried") != 0)
            kind = "no solver" if not has_qp else ("failed solve" if failed else "solved")
            c[kind] += 1
            worst[tag][kind] = max(worst[tag][kind], t)
            if has_qp and not failed and num(r, "qp_iters") > 0 and num(r, "hpipm_us") > 0:
                per_iter[tag].append(num(r, "hpipm_us") / num(r, "qp_iters"))
            if t > a.period_us:
                any_over = True
                c["overrun"] += 1
                c["overrun " + kind] += 1
                by_cell[tag][cell] += 1
        trials_with[tag] += any_over

    if not per:
        raise SystemExit(f"no trials with a trace_left.csv under {a.out}/trials")
    order = [t for t in ("sqp", "iter", "ompc", "smc") if t in per] + \
            [t for t in per if t not in ("sqp", "iter", "ompc", "smc")]
    for tag in order:
        c = per[tag]
        print(f"\n== {tag}: {n_trials[tag]} trials, {c['ticks']} ticks, "
              f"{c['overrun']} over {a.period_us / 1000:g} ms "
              f"({100 * c['overrun'] / c['ticks']:.3f}%) in {trials_with[tag]} trials")
        for kind in ("solved", "failed solve", "no solver"):
            if c[kind]:
                print(f"   {kind:13s} {c[kind]:8d} ticks, {c['overrun ' + kind]:6d} overran, "
                      f"worst {worst[tag][kind]:8.0f} us")
        if per_iter[tag]:
            it = per_iter[tag]
            print(f"   one QP iteration: median {statistics.median(it):.0f} us, "
                  f"p99 {sorted(it)[int(0.99 * (len(it) - 1))]:.0f} us")
        if by_cell[tag]:
            print("   overruns by cell: " + ", ".join(
                f"{k} {v}" for k, v in by_cell[tag].most_common(a.top)))


if __name__ == "__main__":
    main()
