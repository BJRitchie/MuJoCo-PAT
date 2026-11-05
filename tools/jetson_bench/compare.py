#!/usr/bin/env python3
"""Compare two machines' runs of the Jetson timing studies: the same scenarios,
the same image and build, so what differs is the hardware.

    ~/.venvs/thesis_eval/bin/python tools/jetson_bench/compare.py            # dgx vs orin_nx
    ~/.venvs/thesis_eval/bin/python tools/jetson_bench/compare.py --a dgx --b orin_nx

Reads data/thesis_eval/machines/<name>/{timing,jetson_timing,worst_case}/ (each
with summary.csv, manifest.json and, for the per-tick record, series/). Writes:

  data/thesis_eval/machines/compare_<a>_<b>.txt    the report below, as printed
  docs/assets/thesis/fig22_jetson_timing.{pdf,png} tick distributions, worst case
  docs/assets/thesis/tables/tab_jetson.tex         the numbers, LaTeX
  docs/assets/thesis/tables/tab_jetson_control_<a>_<b>.tex   what the QP settings
                                                   cost the control (section 6)
(fig22 and tab_jetson get a _<a>_vs_<b> suffix unless --a is dgx, so comparing two
runs of one machine, e.g. warm against cold start, keeps the DGX comparison.)

The report, in order:
  1. the two machines as their manifests recorded them, and anything that makes
     the comparison unfair (different image or build, throttled clocks, a timing
     run allowed on a noisy machine);
  2. whether the per-tick records are identical between machines (same binary,
     same inputs: they should be, bit for bit, and every timing comparison is
     then tick for tick on identical inputs);
  3. the control tick per controller on each machine: mean, p99, p99.9, worst,
     ticks over the period;
  4. the forced worst case per NMPC: the worst tick the QP cap allows, the cost
     of one iteration, and the largest cap that keeps the worst tick inside the
     budget on each machine;
  5. the horizon sweep;
  6. what the settings cost the control, cell for cell: RMSE, failed solves,
     safe mode, QP iterations. Two runs of one machine at other caps or with
     cold-start-only QPs (set_caps.sh --cold) are compared this way:
       compare.py --a orin_nano --b orin_nano_cold_cap17_18
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "thesis_eval"))
import figures as F  # noqa: E402  (sets the plot style; brings common as F.C)
import tables as T   # noqa: E402

C = F.C
plt = F.plt
MACHINES = os.path.join(C.DATA_ROOT, "machines")
NMPC = ("sqp", "iter")


class Out:
    """Print and keep, so the report is also written to a file."""

    def __init__(self):
        self.lines: List[str] = []

    def __call__(self, s: str = "") -> None:
        print(s)
        self.lines.append(s)


def load(machine: str, study: str):
    d = os.path.join(MACHINES, machine, study)
    summ = pd.read_csv(os.path.join(d, "summary.csv")) \
        if os.path.isfile(os.path.join(d, "summary.csv")) else None
    man = json.load(open(os.path.join(d, "manifest.json"))) \
        if os.path.isfile(os.path.join(d, "manifest.json")) else None
    return summ, man


def series(machine: str, study: str, cell: str) -> Optional[pd.DataFrame]:
    p = os.path.join(MACHINES, machine, study, "series", cell + ".csv")
    return pd.read_csv(p) if os.path.isfile(p) else None


def caps_of(manifest: Optional[dict]) -> Dict[str, int]:
    """The QP cap each NMPC ran with, from the run's own manifest: a rerun at
    other caps (set_caps.sh) reports its own, not the repo's tuned.json."""
    out: Dict[str, int] = {}
    for c in (manifest or {}).get("cells", []):
        if c.get("tag") in NMPC and c["tag"] not in out:
            for a in c.get("args", []):
                m = re.match(r"qp\.max_iter=(\d+)", a)
                if m:
                    out[c["tag"]] = int(m.group(1))
    return out


def solves_of(manifest: Optional[dict]) -> Dict[str, int]:
    """QP solves a failing tick can cost, per NMPC: 2 (warm start, then the
    cold-start retry), or 1 when the run set qp.warm_start=0 (set_caps.sh --cold:
    every solve cold, never retried)."""
    out: Dict[str, int] = {}
    for c in (manifest or {}).get("cells", []):
        if c.get("tag") in NMPC and c["tag"] not in out:
            out[c["tag"]] = 1 if "qp.warm_start=0" in c.get("args", []) else 2
    return out


def qp_mode(manifest: Optional[dict]) -> str:
    caps, solves = caps_of(manifest), solves_of(manifest)
    return ", ".join(f"{C.LABEL[t]} cap {caps.get(t)} "
                     f"{'cold, no retry' if solves.get(t) == 1 else 'warm + retry'}"
                     for t in NMPC if t in caps) or "no NMPC"


def pair_suffix(a: str, b: str) -> str:
    """The DGX-against-a-Jetson comparison keeps the plain names (fig22, tab_jetson);
    any other pair (two runs of one machine) gets its own, so neither overwrites
    the other."""
    return "" if a == "dgx" else f"_{a}_vs_{b}"


# ---------------------------------------------------------------- 1. machines --

def describe(out: Out, name: str, man: dict) -> List[str]:
    """Print a machine's record; return what makes its timing untrustworthy."""
    m = man["machine"]
    pin = man.get("pinned_to")
    out(f"  {name}: {m.get('cpu_model')} ({m.get('machine')}, {m.get('n_cpus')} CPUs), "
        f"governor {m.get('cpu_governor')}, pinned to {pin}")
    out(f"     build {m.get('build_type')} {m.get('pat_arch_flags') or '(no -march/-mcpu)'}, "
        f"BLASFEO {m.get('blasfeo_target')}, image {str(m.get('image_id'))[:19]}, "
        f"max temp {m.get('max_temp_c')} C, git {man['git']['revision'][:8]}"
        f"{' (dirty)' if man['git']['dirty'] else ''}")
    out(f"     QP: {qp_mode(man)}")
    problems = []
    if man.get("noisy_timing_allowed"):
        problems.append(f"{name}: timing run with --allow-noisy-timing")
    if not pin:
        problems.append(f"{name}: timing cells were not pinned to a core")
    for cpu in (pin or "").replace("-", ",").split(","):
        f = m.get("cpu_freq_khz", {}).get(f"cpu{cpu}", {}) if cpu else {}
        if f and f.get("cur") and f.get("max") and f["cur"] < 0.95 * f["max"]:
            problems.append(f"{name}: cpu{cpu} at {f['cur'] / 1e6:.2f} GHz of "
                            f"{f['max'] / 1e6:.2f} at the start (throttled, or clocks not locked)")
    return problems


def machines_section(out: Out, a: str, b: str, mans: Dict[str, dict]) -> None:
    out("1. MACHINES")
    problems = []
    for name in (a, b):
        if mans.get(name):
            problems += describe(out, name, mans[name])
        else:
            out(f"  {name}: no jetson_timing manifest")
    if mans.get(a) and mans.get(b):
        ma, mb = mans[a]["machine"], mans[b]["machine"]
        for k in ("image_id", "blasfeo_target", "build_type", "pat_arch_flags"):
            # A field one machine did not record (a packaged workspace carries no
            # CMakeCache.txt) is unknown, not different: the binary's sha256,
            # checked below, is what proves the build is the same.
            if ma.get(k) is None or mb.get(k) is None:
                continue
            if ma.get(k) != mb.get(k):
                problems.append(f"{k} differs ({ma.get(k)} vs {mb.get(k)}): not the same "
                                "build, so a timing difference is not the hardware alone")
        if mans[a]["bench"]["sha256"] != mans[b]["bench"]["sha256"]:
            problems.append("contact_bench differs between the machines (sha256): not the "
                            "same binary")
    out("  comparison: " + ("FAIR" if not problems else "CAVEATS"))
    for p in problems:
        out(f"    - {p}")
    out()


# ------------------------------------------------------------ 2. identity --

#: Per-tick columns that depend on the arithmetic, not on the clock.
STATE_COLS = ("ee_x", "ee_y", "err_m", "base_x", "base_y", "base_yaw")


def identity_section(out: Out, a: str, b: str, sa: pd.DataFrame, sb: pd.DataFrame) -> None:
    out("2. SAME INPUTS ON BOTH MACHINES?")
    cells = sorted(set(sa["cell"]) & set(sb["cell"]))
    same, diff, missing = 0, [], 0
    for cell in cells:
        xa, xb = series(a, "jetson_timing", cell), series(b, "jetson_timing", cell)
        if xa is None or xb is None:
            missing += 1
            continue
        cols = [c for c in STATE_COLS if c in xa and c in xb]
        if len(xa) == len(xb) and all(np.array_equal(xa[c].to_numpy(), xb[c].to_numpy())
                                      for c in cols):
            same += 1
        else:
            n = min(len(xa), len(xb))
            dev = max(float(np.nanmax(np.abs(xa[c].to_numpy()[:n] - xb[c].to_numpy()[:n])))
                      for c in cols) if n else float("nan")
            diff.append((cell, len(xa), len(xb), dev))
    if missing == len(cells):
        out("  the per-tick records are not on this machine (they stay where the runs "
            "were made): see the report written there, compare_*.txt")
        out()
        return
    out(f"  {same} of {len(cells) - missing} cells identical tick for tick"
        + (f" ({missing} without per-tick records on one machine)" if missing else ""))
    if diff:
        out("  differing (cell, ticks a/b, largest state difference):")
        for cell, na, nb, dev in diff[:10]:
            out(f"    {cell}: {na}/{nb} ticks, {dev:.3g}")
        out("  Not identical means the arithmetic differs (a different build, or a library "
            "choosing code by CPU). The timing comparison is then of nearly, not exactly, "
            "the same inputs; the RMSE below says how far apart the runs went.")
    m = sa.merge(sb, on="cell", suffixes=("_a", "_b"))
    if len(m):
        d = (m["rmse_mm_b"] - m["rmse_mm_a"]).abs()
        out(f"  RMSE, |{b} - {a}|: max {d.max():.3g} mm, mean {d.mean():.3g} mm "
            f"over {len(m)} cells")
    out()


# --------------------------------------------------------------- 3. ticks --

def ticks_of(machine: str, summ: pd.DataFrame, tag: str) -> np.ndarray:
    out = []
    for cell in summ[summ["tag"] == tag]["cell"]:
        s = series(machine, "jetson_timing", cell)
        if s is not None:
            out.append(s["tick_us"].to_numpy())
    return np.concatenate(out) if out else np.array([])


def hist_path(machine: str) -> str:
    return os.path.join(MACHINES, machine, "jetson_timing", "tick_hist.json")


def machine_hists(machine: str, summ: Optional[pd.DataFrame]) -> Dict[str, dict]:
    """Per controller, the tick histogram: from the per-tick records when they are
    on this machine (and then saved, so git has it), else the saved one."""
    hists = {}
    for tag in C.CTRL:
        t = ticks_of(machine, summ, tag) if summ is not None else np.array([])
        if len(t):
            hists[tag] = C.tick_hist_of(t)
    if hists:
        C.write_tick_hist(hist_path(machine), hists)
        return hists
    return C.read_tick_hist(hist_path(machine)) or {}


def hist_quantile(h: dict, q: float) -> float:
    """A quantile of a binned distribution: the upper edge of the bin it falls in
    (within 2.3% at 100 bins per decade)."""
    c = np.cumsum(h["counts"])
    i = int(np.searchsorted(c, q * c[-1]))
    return float(C.hist_edges()[min(i + 1, len(c))])


def tick_stats(h: Optional[dict]) -> Dict[str, float]:
    if not h or not h["n"]:
        return {}
    e = C.hist_edges()
    mid = np.sqrt(e[:-1] * e[1:])
    return {"n": h["n"], "mean": float((h["counts"] * mid).sum() / h["n"]),
            "p99": hist_quantile(h, 0.99), "p999": hist_quantile(h, 0.999),
            "max": float(h["max"]), "over": int(h["counts"][e[:-1] >= C.PERIOD_US].sum())}


def ticks_section(out: Out, a: str, b: str, sa, sb) -> Dict[Tuple[str, str], dict]:
    out(f"3. CONTROL TICK, jetson_timing (all cells) [us]; period {C.PERIOD_US:.0f} us "
        "(mean and percentiles from the tick histogram, to its 2% bins; worst exact)")
    out(f"  {'':10s} {'machine':10s} {'ticks':>8s} {'mean':>8s} {'p99':>8s} {'p99.9':>8s} "
        f"{'worst':>8s} {'>period':>8s}")
    stats = {}
    hists = {a: machine_hists(a, sa), b: machine_hists(b, sb)}
    for tag in C.CTRL:
        for name, summ in ((a, sa), (b, sb)):
            st = tick_stats(hists[name].get(tag))
            stats[(tag, name)] = st
            if st:
                out(f"  {C.LABEL[tag]:10s} {name:10s} {st['n']:8d} {st['mean']:8.0f} "
                    f"{st['p99']:8.0f} {st['p999']:8.0f} {st['max']:8.0f} {st['over']:8d}")
        sa_, sb_ = stats.get((tag, a)), stats.get((tag, b))
        if sa_ and sb_:
            out(f"  {'':10s} {b + '/' + a:10s} {'':8s} {sb_['mean'] / sa_['mean']:7.2f}x "
                f"{sb_['p99'] / sa_['p99']:7.2f}x {'':8s} {sb_['max'] / sa_['max']:7.2f}x")
    out()
    return stats


# ---------------------------------------------------------- 4. worst case --

def worst_section(out: Out, a: str, b: str, budget_us: float) -> Dict[Tuple[str, str], dict]:
    out(f"4. FORCED WORST CASE (every QP fails: the cap once per solve, two solves a "
        f"tick with the warm start's cold retry, one without); budget "
        f"{budget_us:.0f} us of the {C.PERIOD_US:.0f} us period")
    res = {}
    for name in (a, b):
        summ, man = load(name, "worst_case")
        if summ is None:
            out(f"  {name}: no worst_case study")
            continue
        caps, solves = caps_of(man), solves_of(man)
        for tag in NMPC:
            d = summ[summ["tag"] == tag]
            if d.empty:
                continue
            cap = caps.get(tag)
            ns = solves.get(tag, 2)
            iters = float(d["qp_iter_mean"].mean())
            worst = float(d["tick_us_max"].max())
            # Cost of one iteration, from the WORST tick: conservative, since it
            # also carries the tick's work outside the QP.
            per_iter = worst / iters if iters > 0 else float("nan")
            fit = int(budget_us // (ns * per_iter)) if per_iter > 0 else None
            full = cap and iters >= 0.95 * ns * cap
            res[(tag, name)] = {"cap": cap, "cold": ns == 1, "iters": iters, "worst": worst,
                                "mean": float(d["tick_us_mean"].mean()),
                                "p99": float(d["tick_us_p99"].mean()),
                                "per_iter": per_iter, "fit": fit}
            verdict = "FITS" if worst <= C.PERIOD_US else "OVER THE PERIOD"
            out(f"  {C.LABEL[tag]:9s} {name:9s} cap {cap}{' cold' if ns == 1 else ''}: "
                f"{iters:5.1f} iterations/tick "
                f"{'(full path)' if full else '(NOT the full path: see below)'}, worst "
                f"{worst / 1000:6.2f} ms -> {verdict}; {per_iter:5.0f} us per iteration; "
                f"largest cap inside the budget: {fit}")
            if not full:
                out("     The QP stopped short of the cap (HPIPM can stop on a minimum step "
                    "before the iteration limit), so this underestimates the worst case: "
                    "take the worst tick of jetson_timing's hardest cells instead.")
    out()
    return res


# --------------------------------------------------------------- figure --

def figure(a: str, b: str, sa, sb, worst: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(F.FULL, 2.8), gridspec_kw={"wspace": 0.3,
                                                                       "width_ratios": [1.6, 1]})
    styles = {a: dict(ls="-"), b: dict(ls="--")}
    hists = {a: machine_hists(a, sa), b: machine_hists(b, sb)}
    for tag in C.CTRL:
        for name in (a, b):
            h = hists[name].get(tag)
            if not h or not h["n"]:
                continue
            C.stairs_tick_hist(axes[0], h, color=C.COLOUR[tag], **styles[name])
            axes[0].plot(h["max"], 1.0, marker="v" if name == a else "1", color=C.COLOUR[tag],
                         ms=5, ls="none", transform=axes[0].get_xaxis_transform(),
                         clip_on=False)
    axes[0].set_xlim(0.5, 5e4)
    axes[0].axvline(C.PERIOD_US, color="k", ls=":", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("control tick [µs], same scenarios (markers: longest)")
    axes[0].set_ylabel("density")

    x = np.arange(len(NMPC))
    for i, name in enumerate((a, b)):
        vals = [worst.get((tag, name), {}).get("worst", np.nan) / 1000 for tag in NMPC]
        bars = axes[1].bar(x + (i - 0.5) * 0.38, vals, 0.36,
                           color=[C.COLOUR[t] for t in NMPC], alpha=1.0 if i == 0 else 0.5,
                           edgecolor="k", lw=0.5, hatch=None if i == 0 else "//")
        for rect, tag in zip(bars, NMPC):
            w = worst.get((tag, name), {})
            cap = w.get("cap")
            if cap and np.isfinite(rect.get_height()):
                axes[1].text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                             f"{name}\ncap {cap}{' cold' if w.get('cold') else ''}",
                             ha="center", va="bottom", fontsize=5.5)
    axes[1].axhline(C.PERIOD_US / 1000, color="k", ls=":", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([C.LABEL[t] for t in NMPC])
    axes[1].set_ylabel("forced longest tick [ms]")
    axes[1].set_ylim(0, max(C.PERIOD_US / 1000 * 1.3, axes[1].get_ylim()[1] * 1.15))
    handles = F.ctrl_handles() + [
        plt.Line2D([], [], color="0.3", ls="-", label=a),
        plt.Line2D([], [], color="0.3", ls="--", label=b),
        plt.Line2D([], [], color="k", ls=":", lw=0.8, label="10 ms period")]
    fig.subplots_adjust(bottom=0.32)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    F.save(fig, "fig22_jetson_timing" + pair_suffix(a, b))


def table(a: str, b: str, stats: Dict, worst: Dict) -> None:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{control tick [\textmu s]} & \multicolumn{2}{c}"
             r"{forced worst case} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r" & machine & mean & p99 & longest & $>$10\,ms & QP cap & longest [ms] \\",
             r"\midrule"]
    for tag in C.CTRL:
        for name in (a, b):
            st = stats.get((tag, name)) or {}
            w = worst.get((tag, name)) or {}
            lines.append(
                f"{C.LABEL[tag] if name == a else ''} & {name.replace('_', chr(92) + '_')} & "
                f"{T.fmt(st.get('mean'))} & {T.fmt(st.get('p99'))} & {T.fmt(st.get('max'))} & "
                f"{st.get('over', '--')} & {w.get('cap', '--')} & "
                f"{T.fmt(w['worst'] / 1000) if w else '--'} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    d = os.path.join(C.OUT_ROOT, "tables")
    os.makedirs(d, exist_ok=True)
    name = f"tab_jetson{pair_suffix(a, b)}.tex"
    with open(os.path.join(d, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote tables/{name}")


def timing_sweep(out: Out, a: str, b: str) -> None:
    out("5. HORIZON SWEEP (timing study, 1x circle, exact state) [us]: mean / worst")
    rows = {}
    for name in (a, b):
        summ, _ = load(name, "timing")
        if summ is None:
            continue
        hz = summ[summ["cell"].str.contains("_N")].copy()
        if hz.empty:
            continue
        hz["N"] = hz["cell"].str.extract(r"_N([0-9]+)_")[0].astype(int)
        for (tag, n), d in hz.groupby(["tag", "N"]):
            rows.setdefault((tag, n), {})[name] = (d["tick_us_mean"].mean(), d["tick_us_max"].max())
    for (tag, n), v in sorted(rows.items()):
        cells = "   ".join(f"{name} {v[name][0]:6.0f} / {v[name][1]:6.0f}" for name in (a, b)
                           if name in v)
        out(f"  {C.LABEL[tag]:9s} N {n:2d}: {cells}")
    out()


# ------------------------------------------------------------- 6. accuracy --

def scenario_of(cell: str, tag: str) -> str:
    return cell.rsplit(f"_{tag}_s", 1)[0]


def accuracy_section(out: Out, a: str, b: str, sa, sb) -> Dict[str, dict]:
    """What the QP settings cost the control, cell for cell on the same scenarios
    and seeds: two runs of one machine at other caps or warm-start modes, or two
    machines at the same ones (then this should read "no change")."""
    out(f"6. CONTROL, jetson_timing, {b} against {a}, cell for cell (same scenario and seed)")
    out("  RMSE ratio: geometric mean, and the worst cell. Failed solves: tier1_entries "
        "(a solve that failed, the last plan replayed), summed. Safe mode: ticks, summed.")
    res: Dict[str, dict] = {}
    if sa is None or sb is None:
        out("  needs both runs' jetson_timing summaries")
        out()
        return res
    m = sa.merge(sb, on=["cell", "tag"], suffixes=("_a", "_b"))
    for tag in C.CTRL:
        d = m[m["tag"] == tag]
        if d.empty:
            continue
        ra, rb = d["rmse_mm_a"].to_numpy(float), d["rmse_mm_b"].to_numpy(float)
        ok = (ra > 0) & (rb > 0) & np.isfinite(ra) & np.isfinite(rb)
        ratio = rb[ok] / ra[ok]
        g = float(np.exp(np.mean(np.log(ratio)))) if ok.any() else float("nan")
        i = int(np.argmax(ratio)) if ok.any() else None
        worst_cell = d["cell"].to_numpy()[ok][i] if i is not None else "-"
        r = {"n": len(d), "gmean": g, "worst": float(ratio.max()) if ok.any() else float("nan"),
             "worst_cell": worst_cell,
             "fail_a": int(d["tier1_entries_a"].sum()), "fail_b": int(d["tier1_entries_b"].sum()),
             "safe_a": int(d["safe_ticks_a"].sum()), "safe_b": int(d["safe_ticks_b"].sum()),
             "it_a": float(d["qp_iter_mean_a"].mean()), "it_b": float(d["qp_iter_mean_b"].mean()),
             "p99_a": float(d["tick_us_p99_a"].mean()), "p99_b": float(d["tick_us_p99_b"].mean()),
             "max_a": float(d["tick_us_max_a"].max()), "max_b": float(d["tick_us_max_b"].max())}
        res[tag] = r
        out(f"  {C.LABEL[tag]:9s} {r['n']:3d} cells: RMSE x{g:.3f} (worst x{r['worst']:.2f}, "
            f"{worst_cell}); failed solves {r['fail_a']} -> {r['fail_b']}; safe-mode ticks "
            f"{r['safe_a']} -> {r['safe_b']}")
        if tag in NMPC:
            out(f"  {'':9s} QP iterations/tick {r['it_a']:.1f} -> {r['it_b']:.1f}; tick p99 "
                f"{r['p99_a']:.0f} -> {r['p99_b']:.0f} us, worst {r['max_a']:.0f} -> "
                f"{r['max_b']:.0f} us")
        # The scenarios that moved most, so a cost is located, not just averaged.
        if ok.any() and (r["worst"] > 1.05 or r["fail_a"] != r["fail_b"]):
            d2 = d[ok].copy().assign(ratio=ratio, scen=[scenario_of(c, tag) for c in d["cell"][ok]])
            by = d2.groupby("scen").agg(ratio=("ratio", lambda x: float(np.exp(np.mean(np.log(x))))),
                                        fa=("tier1_entries_a", "sum"),
                                        fb=("tier1_entries_b", "sum"))
            for scen, row in by.sort_values("ratio", ascending=False).head(4).iterrows():
                out(f"  {'':9s}   {scen}: RMSE x{row['ratio']:.2f}, failed solves "
                    f"{int(row['fa'])} -> {int(row['fb'])}")
    out()
    return res


def accuracy_table(a: str, b: str, acc: Dict[str, dict]) -> None:
    if not acc:
        return
    esc = lambda n: n.replace("_", chr(92) + "_")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             rf" & RMSE ratio & \multicolumn{{2}}{{c}}{{failed solves}} & "
             rf"\multicolumn{{2}}{{c}}{{QP iter./tick}} & longest tick [ms] \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             rf" & {esc(b)}/{esc(a)} & {esc(a)} & {esc(b)} & {esc(a)} & {esc(b)} & "
             rf"{esc(a)} / {esc(b)} \\", r"\midrule"]
    for tag in C.CTRL:
        r = acc.get(tag)
        if not r:
            continue
        nm = tag in NMPC
        lines.append(f"{C.LABEL[tag]} & {r['gmean']:.3f} & {r['fail_a'] if nm else '--'} & "
                     f"{r['fail_b'] if nm else '--'} & "
                     f"{T.fmt(r['it_a']) if nm else '--'} & {T.fmt(r['it_b']) if nm else '--'} & "
                     f"{r['max_a'] / 1000:.2f} / {r['max_b'] / 1000:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    d = os.path.join(C.OUT_ROOT, "tables")
    os.makedirs(d, exist_ok=True)
    name = f"tab_jetson_control_{a}_{b}.tex"
    with open(os.path.join(d, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote tables/{name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", default="dgx", help="reference machine (default dgx)")
    ap.add_argument("--b", default="orin_nx", help="machine compared with it (default orin_nx)")
    ap.add_argument("--budget-us", type=float, default=8000.0,
                    help="worst-tick budget the cap must fit, leaving the rest of the "
                         "10 ms period for jitter and the node's own work (default 8000)")
    a = ap.parse_args()
    C.set_variant("tuned")
    out = Out()
    out(f"Jetson comparison: {a.a} vs {a.b}")
    out()
    sa, ma = load(a.a, "jetson_timing")
    sb, mb = load(a.b, "jetson_timing")
    machines_section(out, a.a, a.b, {a.a: ma, a.b: mb})
    if sa is not None and sb is not None:
        identity_section(out, a.a, a.b, sa, sb)
    stats = ticks_section(out, a.a, a.b, sa, sb)
    worst = worst_section(out, a.a, a.b, a.budget_us)
    timing_sweep(out, a.a, a.b)
    acc = accuracy_section(out, a.a, a.b, sa, sb)
    path = os.path.join(MACHINES, f"compare_{a.a}_{a.b}.txt")
    os.makedirs(MACHINES, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(out.lines) + "\n")
    print(f"wrote {path}")
    figure(a.a, a.b, sa, sb, worst)
    table(a.a, a.b, stats, worst)
    accuracy_table(a.a, a.b, acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
