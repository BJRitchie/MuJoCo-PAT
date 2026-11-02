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
import math
import os
import shutil
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
    cap = ""
    if s["law"] == "nmpc":
        cap = next((a.split("=", 1)[1] for a in s.get("tuned_args", [])
                    if a.startswith("qp.max_iter=")), "none")
    out["qp_cap"] = cap
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


def write_rows(path, rows):
    head = list(rows[0])
    for r in rows[1:]:
        head += [k for k in r if k not in head]
    rows.sort(key=lambda r: (r["cell"], r["tag"], int(r["repeat"])))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


#: common.py's tick-histogram grid (HIST_*): kept equal by hand, since this file
#: is standard library only and runs in the container.
HIST_PER_DECADE, HIST_LO_EXP, HIST_HI_EXP = 100, -1, 5
#: The cell fig21_live_trace draws, one trial per controller.
TRACE_CELL = "rectangle_x1_m1_bd0_tau1x_ekf"


def tick_hists(out_dir, rows):
    """Per controller, the histogram of every tick of every collected trial, from
    the per-tick records (series.csv), which stay out of git. None if no trial
    has its records on this machine."""
    nb = (HIST_HI_EXP - HIST_LO_EXP) * HIST_PER_DECADE
    hists, found = {}, 0
    for r in rows:
        p = os.path.join(out_dir, "trials", r["hash"], "series.csv")
        if not os.path.isfile(p):
            continue
        found += 1
        h = hists.setdefault(r["tag"], {"n": 0, "max": 0.0, "counts": {}})
        with open(p, newline="") as fh:
            for rec in csv.DictReader(fh):
                try:
                    t = float(rec["tick_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (t > 0) or math.isinf(t):
                    continue
                i = int(math.floor(math.log10(t) * HIST_PER_DECADE)) - HIST_LO_EXP * HIST_PER_DECADE
                i = min(max(i, 0), nb - 1)
                h["counts"][str(i)] = h["counts"].get(str(i), 0) + 1
                h["n"] += 1
                h["max"] = max(h["max"], t)
    return (hists if found else None), found


def plot_series(out_dir, rows):
    """The per-tick records fig21_live_trace draws (TRACE_CELL, the first repeat of
    each controller), copied to plot_series/ so they can go in git."""
    dst = os.path.join(out_dir, "plot_series")
    copied = 0
    for tag in sorted({r["tag"] for r in rows}):
        cand = sorted((int(r["repeat"]), r["hash"]) for r in rows
                      if r["tag"] == tag and r["cell"] == TRACE_CELL)
        for _, h in cand:
            src = os.path.join(out_dir, "trials", h, "series.csv")
            if os.path.isfile(src):
                os.makedirs(dst, exist_ok=True)
                shutil.copyfile(src, os.path.join(dst, f"{tag}_{TRACE_CELL}.csv"))
                copied += 1
                break
    return copied


def collect(out_dir, include_stale=False):
    rows, stale_rows = [], []
    args = current_args()
    for d in sorted(glob.glob(os.path.join(out_dir, "trials", "*"))):
        if d.endswith(".partial") or not os.path.isfile(os.path.join(d, "report.json")):
            continue
        current = is_current(d, args)
        try:
            row = row_of(d)
        except (OSError, ValueError, KeyError, StopIteration) as e:
            print(f"collect: skipping {d}: {e}", file=sys.stderr)
            continue
        if current or include_stale:
            rows.append(row)
        else:
            stale_rows.append(row)
    stale = len(stale_rows)
    # Trials run with settings tuned.json no longer has (the uncapped first run):
    # kept apart, for the before/after comparison, never mixed into the results.
    stale_path = os.path.join(out_dir, "summary_stale.csv")
    if stale_rows:
        write_rows(stale_path, stale_rows)
    elif os.path.isfile(stale_path):
        os.remove(stale_path)
    if not rows:
        sys.exit(f"collect: no finished trials under {out_dir}/trials")
    if stale:
        print(f"collect: {stale} trials run with settings tuned.json no longer has went to "
              f"{stale_path}, not the results (--include-stale mixes them in)",
              file=sys.stderr)
    path = os.path.join(out_dir, "summary.csv")
    write_rows(path, rows)
    hists, found = tick_hists(out_dir, rows)
    if hists is not None:
        with open(os.path.join(out_dir, "tick_hist.json"), "w") as fh:
            json.dump({"per_decade": HIST_PER_DECADE, "lo_exp": HIST_LO_EXP,
                       "hi_exp": HIST_HI_EXP, "trials": found, "tags": hists}, fh)
        n_plot = plot_series(out_dir, rows)
        print(f"collect: tick_hist.json from {found} trials' per-tick records; "
              f"{n_plot} traces in plot_series/")
    else:
        print("collect: no per-tick records on this machine: tick_hist.json and "
              "plot_series/ left as they are", file=sys.stderr)
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
