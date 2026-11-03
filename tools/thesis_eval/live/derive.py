#!/usr/bin/env python3
"""Small, git-tracked tables derived from the live run's and the machine runs'
per-tick records, for the figures that need per-tick data.

    .venv/Scripts/python tools/thesis_eval/live/derive.py            # Windows venv
    ~/.venvs/thesis_eval/bin/python tools/thesis_eval/live/derive.py

The per-tick records (each live trial's series.csv and trace_left.csv, each
machine run's series/) are ~1.5 GB and stay on the machine that ran them
(data/thesis_eval/.gitignore). What a figure draws from them is written here, a
few MB in all, so figures.py renders every figure from a clone:

  data/thesis_eval/live/main/derived/
    track_<shape>.csv     the nominal cell on the EKF, one repeat per controller
                          (the one with the median RMSE of its three), every
                          second tick plus every safe-mode entry: time, reference,
                          EE, error, base yaw, mode
    errdist.csv           per shape and controller, the per-tick EE error in the
                          scoring window pooled over all three repeats, as 501
                          quantiles (0, 0.2%, ..., 100%: the minimum and the
                          maximum are exact), with the pooled RMSE
    estimate.csv          SQP-NMPC on the nominal rectangle, raw sensors and the
                          EKF, one repeat each (median base-position error): the
                          true base pose and the estimate's error, every second tick
    exposed_ticks.csv     every NMPC control tick (left arm's node trace, warm-up
                          included) of the uncapped first run and of the current
                          capped run, SQP-NMPC and LMPC: every failed solve (status
                          != 0 or retried) and every tick over the period, plus a
                          seeded sample of 10 000 converged ticks per run and law
                          (`sampled` = 1). qp_iters is the tick's TOTAL over its
                          solves, the retry included; qp_iters1 the first alone
    exposed_stages.csv    per run, law and kind (converged / failed), over EVERY
                          tick: the count, the mean of each node stage, the tick,
                          HPIPM's own time and QP iterations, and the worst tick

  data/thesis_eval/machines/derived/
    tick_pairs_<a>_<b>.csv    per controller, a 2-D histogram of each tick on
                              machine b against the same tick on machine a
                              (jetson_timing, paired row by row: the runs are
                              bit-identical), 20 bins per decade
    tick_pairs_<a>_<b>_stats.csv  per controller: ticks, ratio b/a (geometric mean,
                              median, p1, p99), each machine's mean, p99 and worst,
                              and whether every cell's trajectory was identical

Idempotent: the same raw data gives byte-identical files. Prints what it wrote.
A source that is missing is skipped with a message (a machine without the raw
data keeps the derived tables in git).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import common as C  # noqa: E402

FLOAT = "%.7g"
NMPC = ("sqp", "iter")
#: The node trace's stage columns [us], in the order the node writes them.
STAGES = ["dynamics", "jdotqdot", "lambda", "qpsolve", "rollout_dynamics",
          "rollout_integrate", "pinv", "build_model", "defect", "trust_region", "finalize"]
CONVERGED_SAMPLE = 10_000
QUANTILES = np.linspace(0.0, 1.0, 501)
#: The machine pair the tick-pair figure draws, and the runs' study.
PAIRS = (("dgx_n24", "orin_nano"),)
PAIR_STUDY = "jetson_timing"


def write(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format=FLOAT, lineterminator="\n")
    print(f"  wrote {os.path.relpath(path, C.ROOT)}  ({len(df)} rows, "
          f"{os.path.getsize(path) / 1024:.0f} kB)")


def nominal_ekf(live: pd.DataFrame) -> pd.DataFrame:
    return C.at_nominal(live[live["source"] == "ekf"])


def median_repeat(rows: pd.DataFrame, col: str) -> pd.Series:
    """The trial whose `col` is the median of its repeats (the lower middle one
    for an even count): a representative run, not the best."""
    r = rows.sort_values([col, "repeat"])
    return r.iloc[(len(r) - 1) // 2]


def decimate(s: pd.DataFrame, every: int = 2) -> pd.DataFrame:
    """Every `every`-th tick, and every tick where safe mode is entered, so the
    figures' safe-mode crosses survive the thinning."""
    m = s["mode"].to_numpy() >= 2 if "mode" in s else np.zeros(len(s), bool)
    entry = m & ~np.r_[False, m[:-1]]
    keep = (np.arange(len(s)) % every == 0) | entry
    return s[keep]


# ------------------------------------------------------------------ live --

def derive_live(run: str, run_dir: str, out_dir: str) -> None:
    live = C.live_summary(run)
    if live is None:
        print(f"  live: no summary.csv under {run_dir}; skipped")
        return
    trials = os.path.join(run_dir, "trials")

    def series(h):
        p = os.path.join(trials, h, "series.csv")
        return pd.read_csv(p) if os.path.isfile(p) else None

    nom = nominal_ekf(live)
    if not os.path.isfile(os.path.join(trials, nom["hash"].iloc[0], "series.csv")):
        print(f"  live: per-tick records not under {trials}; skipped")
        return

    dist = []
    for shape in ("circle", "rectangle"):
        tracks = []
        for c in C.CTRL:
            rows = nom[(nom["shape"] == shape) & (nom["tag"] == c)]
            if rows.empty:
                continue
            pick = median_repeat(rows, "rmse_mm")
            s = series(pick["hash"])
            if s is None:
                continue
            t = decimate(s)[["t_s", "ref_x", "ref_y", "ee_x", "ee_y", "err_m", "base_yaw",
                             "mode"]].copy()
            t.insert(0, "repeat", int(pick["repeat"]))
            t.insert(0, "tag", c)
            t["window_start_s"] = float(pick["window_start_s"])
            tracks.append(t)
            # Every repeat's scoring window, pooled: the distribution panel.
            errs = []
            for _, r in rows.sort_values("repeat").iterrows():
                sr = series(r["hash"])
                if sr is not None:
                    errs.append(sr.loc[sr["t_s"] >= float(r["window_start_s"]), "err_m"]
                                .to_numpy() * 1e3)
            e = np.concatenate(errs)
            dist.append(pd.DataFrame({
                "shape": shape, "tag": c, "q": QUANTILES,
                "err_mm": np.quantile(e, QUANTILES), "n": len(e), "repeats": len(errs),
                "rmse_pooled_mm": float(np.sqrt(np.mean(e ** 2))),
                "rmse_mean_of_repeats_mm": float(rows["rmse_mm"].mean()),
                "safe_ticks": int(rows["safe_ticks"].sum())}))
        if tracks:
            write(pd.concat(tracks, ignore_index=True), os.path.join(out_dir, f"track_{shape}.csv"))
    if dist:
        write(pd.concat(dist, ignore_index=True), os.path.join(out_dir, "errdist.csv"))

    # The base estimate, SQP-NMPC on the rectangle, raw sensors against the EKF.
    est = []
    for src in ("raw", "ekf"):
        rows = C.at_nominal(live[(live["source"] == src) & (live["shape"] == "rectangle") &
                                 (live["tag"] == "sqp")])
        if rows.empty:
            continue
        pick = median_repeat(rows, "est_pos_rms_mm")
        s = series(pick["hash"])
        if s is None:
            continue
        t = decimate(s)[["t_s", "base_x", "base_y", "base_yaw", "est_x", "est_y", "est_yaw",
                         "mode"]].copy()
        t.insert(0, "repeat", int(pick["repeat"]))
        t.insert(0, "source", src)
        t["est_pos_rms_mm"] = float(pick["est_pos_rms_mm"])
        t["window_start_s"] = float(pick["window_start_s"])
        est.append(t)
    if est:
        write(pd.concat(est, ignore_index=True), os.path.join(out_dir, "estimate.csv"))

    derive_exposed(run, run_dir, out_dir, live)


def _runs(run: str, live: pd.DataFrame):
    """(run label, trials) for the NMPCs: the uncapped first run, and now."""
    stale = C.live_summary(run, stale=True)
    out = []
    if stale is not None and "qp_cap" in stale:
        cap = stale["qp_cap"].astype(str).str.strip().str.lower()
        out.append(("uncapped", stale[stale["tag"].isin(NMPC) & cap.isin(("none", "nan", ""))]))
    out.append(("capped", live[live["tag"].isin(NMPC)]))
    return out


def derive_exposed(run_name: str, run_dir: str, out_dir: str, live: pd.DataFrame) -> None:
    trials = os.path.join(run_dir, "trials")
    keep_cols = ["tick_us", "qp_iters", "qp_iters1", "qp_status", "qp_status1", "retried",
                 "mode", "hpipm_us"] + STAGES
    ticks, stages = [], []
    for run, df in _runs(run_name, live):
        for c in NMPC:
            frames = []
            for h in sorted(df[df["tag"] == c]["hash"]):
                p = os.path.join(trials, h, "trace_left.csv")
                if os.path.isfile(p):
                    frames.append(pd.read_csv(p, comment="#", usecols=keep_cols))
            if not frames:
                continue
            t = pd.concat(frames, ignore_index=True)
            t = t[t["qp_iters"] >= 0]                 # a tick that ran a solve
            failed = (t["qp_status"] != 0) | (t["retried"] != 0)
            t = t.assign(failed=failed.astype(int))
            for kind, sub in (("converged", t[~failed]), ("failed", t[failed])):
                row = {"run": run, "tag": c, "kind": kind, "ticks": len(sub),
                       "trials": len(frames),
                       "over_period": int((sub["tick_us"] > C.PERIOD_US).sum()),
                       "tick_us_max": float(sub["tick_us"].max()) if len(sub) else np.nan}
                for k in ["tick_us", "hpipm_us", "qp_iters"] + STAGES:
                    row[k] = float(sub[k].mean()) if len(sub) else np.nan
                stages.append(row)
            rare = failed | (t["tick_us"] > C.PERIOD_US)
            common = t[~rare]
            n = min(CONVERGED_SAMPLE, len(common))
            sample = common.sample(n=n, random_state=12345 + len(ticks)) if n else common
            out = pd.concat([t[rare], sample]).sort_index()
            out = out.assign(run=run, tag=c, sampled=(~out.index.isin(t[rare].index)).astype(int))
            ticks.append(out[["run", "tag", "sampled", "failed", "tick_us", "qp_iters",
                              "qp_iters1", "qp_status", "retried", "hpipm_us"]])
    if ticks:
        t = pd.concat(ticks, ignore_index=True)
        t["tick_us"] = t["tick_us"].round(1)
        t["hpipm_us"] = t["hpipm_us"].round(0).astype(int)
        write(t, os.path.join(out_dir, "exposed_ticks.csv"))
        write(pd.DataFrame(stages), os.path.join(out_dir, "exposed_stages.csv"))
    else:
        print("  exposed: no node traces found; skipped")


# -------------------------------------------------------------- machines --

PAIR_PER_DECADE = 20
PAIR_LO, PAIR_HI = 0, 5          # 1 us .. 100 ms


def derive_pairs(machines_dir: str, out_dir: str) -> None:
    edges = 10.0 ** (np.arange(PAIR_LO * PAIR_PER_DECADE, PAIR_HI * PAIR_PER_DECADE + 1)
                     / PAIR_PER_DECADE)
    for a, b in PAIRS:
        sa = os.path.join(machines_dir, a, PAIR_STUDY)
        sb = os.path.join(machines_dir, b, PAIR_STUDY)
        if not (os.path.isdir(os.path.join(sa, "series")) and
                os.path.isdir(os.path.join(sb, "series"))):
            print(f"  pairs {a}/{b}: per-tick records not here; skipped")
            continue
        summ = pd.read_csv(os.path.join(sa, "summary.csv"))
        hist_rows, stat_rows = [], []
        for c in C.CTRL:
            xa, xb, identical, cells = [], [], True, 0
            for cell in sorted(summ[summ["tag"] == c]["cell"]):
                pa = os.path.join(sa, "series", cell + ".csv")
                pb = os.path.join(sb, "series", cell + ".csv")
                if not (os.path.isfile(pa) and os.path.isfile(pb)):
                    continue
                da, db = pd.read_csv(pa), pd.read_csv(pb)
                n = min(len(da), len(db))
                identical &= (len(da) == len(db) and
                              np.array_equal(da[["ee_x", "ee_y", "tau0"]].to_numpy(),
                                             db[["ee_x", "ee_y", "tau0"]].to_numpy()))
                xa.append(da["tick_us"].to_numpy()[:n])
                xb.append(db["tick_us"].to_numpy()[:n])
                cells += 1
            if not cells:
                continue
            xa, xb = np.concatenate(xa), np.concatenate(xb)
            ok = (xa > 0) & (xb > 0)
            xa, xb = xa[ok], xb[ok]
            H, _, _ = np.histogram2d(np.clip(xa, edges[0], edges[-1] * 0.999),
                                     np.clip(xb, edges[0], edges[-1] * 0.999), bins=[edges, edges])
            ia, ib = np.nonzero(H)
            hist_rows.append(pd.DataFrame({"tag": c, "a_lo_us": edges[ia], "a_hi_us": edges[ia + 1],
                                           "b_lo_us": edges[ib], "b_hi_us": edges[ib + 1],
                                           "count": H[ia, ib].astype(int)}))
            r = xb / xa
            stat_rows.append({"tag": c, "cells": cells, "ticks": len(xa),
                              "identical_trajectories": int(identical),
                              "ratio_gmean": float(np.exp(np.mean(np.log(r)))),
                              "ratio_median": float(np.median(r)),
                              "ratio_p1": float(np.quantile(r, 0.01)),
                              "ratio_p99": float(np.quantile(r, 0.99)),
                              f"{a}_mean_us": float(xa.mean()), f"{a}_p99_us": float(np.quantile(xa, 0.99)),
                              f"{a}_max_us": float(xa.max()),
                              f"{b}_mean_us": float(xb.mean()), f"{b}_p99_us": float(np.quantile(xb, 0.99)),
                              f"{b}_max_us": float(xb.max()),
                              f"{b}_over_period": int((xb > C.PERIOD_US).sum())})
        if hist_rows:
            write(pd.concat(hist_rows, ignore_index=True),
                  os.path.join(out_dir, f"tick_pairs_{a}_{b}.csv"))
            write(pd.DataFrame(stat_rows), os.path.join(out_dir, f"tick_pairs_{a}_{b}_stats.csv"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=C.LIVE_RUN, help="live run under data/thesis_eval/live/")
    a = ap.parse_args()
    C.set_variant("tuned")
    run_dir = os.path.join(C.LIVE_DIR, a.run)
    derive_live(a.run, run_dir, os.path.join(run_dir, "derived"))
    machines = os.path.join(C.DATA_ROOT, "machines")
    derive_pairs(machines, os.path.join(machines, "derived"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
