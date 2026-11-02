#!/usr/bin/env python3
"""Gather a live run's trials into one summary.csv, the shape figures.py reads.

    python3 tools/thesis_eval/live/collect.py --out data/thesis_eval/live/main

Each finished trial (trials/<hash>/) carries its spec and health (report.json) and
its metrics (summary.csv, series_summary's row: the same columns as the bench's).
This writes <out>/summary.csv with one row per trial: the knobs that define its
cell (tag, shape, speed, source, damping, mass_scale, tau_scale), the repeat, its
health (deadline misses, skipped periods, nav overruns), its hash, and every
metric. `cell` names it the way studies.live_twin_name names its offline twin, so
the two can be matched. Standard library only: it runs in the container or on the
host.

Trials from a failed or discarded run never reach trials/<hash>/ (the driver only
renames a trial there when all of it worked), so everything found is kept.
"""
import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import studies  # noqa: E402

KNOBS = ("tag", "law", "shape", "speed", "source", "damping", "mass_scale", "tau_scale",
         "repeat", "duration", "nav_warmup")
HEALTH = ("deadline_misses", "skipped_periods", "nav_overruns", "rows", "rows_expected")


def row_of(trial_dir):
    rep = json.load(open(os.path.join(trial_dir, "report.json")))
    met = next(csv.DictReader(open(os.path.join(trial_dir, "summary.csv"))))
    s = rep["spec"]
    v = {"shape": s["shape"], "speed": s["speed"], "source": s["state"],
         "damping": s["damping"], "mass_scale": s.get("mass_scale", 1.0),
         "tau_scale": s.get("tau_scale", 1.0)}
    out = {"cell": studies.live_twin_name(v), "tag": s["tag"], "law": s["law"],
           "shape": s["shape"], "speed": s["speed"], "source": s["state"],
           "damping": s["damping"], "mass_scale": v["mass_scale"],
           "tau_scale": v["tau_scale"], "repeat": s["repeat"], "duration": s["duration"],
           "nav_warmup": s["nav_warmup"]}
    for k in HEALTH:
        out[k] = rep["health"].get(k, "")
    out["hash"] = rep["hash"]
    out["machine"] = rep.get("machine", {}).get("hostname", "")
    out["finished"] = rep.get("finished", "")
    # The metrics, minus the labels series_summary repeats from the spec.
    for k, val in met.items():
        if k not in out and k not in ("scenario", "seed", "state_source", "partner",
                                      "traj_speed", "base_damping"):
            out[k] = val
    return out


def current_args():
    """tuned.json's arguments per controller: a trial run with other settings (an
    earlier tuning, or before the QP cap) is not a result for the current ones."""
    with open(studies.TUNED_JSON) as fh:
        return {k: v["args"] for k, v in json.load(fh).items()}


def is_current(trial_dir, args=None):
    spec = json.load(open(os.path.join(trial_dir, "report.json")))["spec"]
    args = args if args is not None else current_args()
    return spec.get("tuned_args") == args.get(spec["tag"])


def collect(out_dir, include_stale=False):
    rows, stale = [], 0
    args = current_args()
    for d in sorted(glob.glob(os.path.join(out_dir, "trials", "*"))):
        if d.endswith(".partial") or not os.path.isfile(os.path.join(d, "report.json")):
            continue
        if not include_stale and not is_current(d, args):
            stale += 1
            continue
        try:
            rows.append(row_of(d))
        except (OSError, ValueError, KeyError, StopIteration) as e:
            print(f"collect: skipping {d}: {e}", file=sys.stderr)
    if not rows:
        sys.exit(f"collect: no finished trials under {out_dir}/trials")
    head = list(rows[0])
    for r in rows[1:]:
        head += [k for k in r if k not in head]
    if stale:
        print(f"collect: left out {stale} trials run with settings tuned.json no longer has "
              "(--include-stale keeps them)", file=sys.stderr)
    rows.sort(key=lambda r: (r["cell"], r["tag"], int(r["repeat"])))
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join("data", "thesis_eval", "live", "main"))
    ap.add_argument("--include-stale", action="store_true",
                    help="keep trials run with settings other than tuned.json's")
    a = ap.parse_args()
    path, rows = collect(a.out, a.include_stale)
    cells = {(r["cell"], r["tag"]) for r in rows}
    misses = sum(int(r["deadline_misses"] or 0) for r in rows)
    print(f"collect: {len(rows)} trials in {len(cells)} cells -> {path}"
          f" ({misses} deadline misses in all)")


if __name__ == "__main__":
    main()
