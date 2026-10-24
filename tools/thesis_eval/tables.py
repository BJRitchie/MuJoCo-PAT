#!/usr/bin/env python3
"""The thesis tables, as LaTeX (booktabs), from the study CSVs.

    ~/.venvs/thesis_eval/bin/python tools/thesis_eval/tables.py

Writes docs/assets/thesis/tables/*.tex, each a bare tabular for \\input inside a
table environment, so the caption and label stay in the thesis source. Needs
\\usepackage{booktabs}. The best value of each column within a block is bold.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def fmt(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 0.01:
        return f"{v:.{digits}g}"
    e = int(math.floor(math.log10(a)))
    return f"${v / 10 ** e:.2g}\\times10^{{{e}}}$"


def write(name: str, lines: Sequence[str]) -> None:
    d = os.path.join(C.ensure_out(), "tables")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote tables/{name}.tex")


def block(rows: List[Tuple[str, List[float]]], lower_is_better: Sequence[bool]) -> List[str]:
    """Table rows, with the best value of each column in bold."""
    vals = np.array([v for _, v in rows], dtype=float)
    best = []
    for j, low in enumerate(lower_is_better):
        col = vals[:, j]
        if np.all(np.isnan(col)):
            best.append(None)
        else:
            best.append(np.nanargmin(col) if low else np.nanargmax(col))
    out = []
    for i, (label, v) in enumerate(rows):
        cells = [f"\\textbf{{{fmt(x)}}}" if best[j] == i else fmt(x) for j, x in enumerate(v)]
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


NOMINAL = [("rmse_mm", "RMSE [mm]"), ("err_var_mm2", r"$\sigma^2_e$ [mm$^2$]"),
           ("err_peak_mm", "peak [mm]"), ("tau2_int_win", r"$\int\|\tau\|^2$"),
           ("tau_var", r"$\sigma^2_\tau$"), ("rail_frac", "at limit"),
           ("yaw2_int_total", r"$\int\psi^2$"), ("tick_us_mean", r"tick [\textmu s]")]


def tab_nominal() -> None:
    df = C.summary("nominal")
    if df is None:
        return
    head = " & ".join(l for _, l in NOMINAL)
    lines = [r"\begin{tabular}{l" + "r" * len(NOMINAL) + "}", r"\toprule",
             f" & {head} \\\\"]
    for scen, title in (("step6", "6 cm step"), ("step12", "12 cm step"),
                        ("circle", "circle"), ("rectangle", "rectangle")):
        sub = df[df["cell"].str.startswith(scen + "_")].copy()
        sub["tag"] = sub["cell"].str.split("_").str[-1]
        sub = C.ordered(sub)
        lines += [r"\midrule", rf"\multicolumn{{{len(NOMINAL) + 1}}}{{l}}{{\emph{{{title}}}}} \\"]
        lines += block([(C.LABEL[r["tag"]], [r[k] for k, _ in NOMINAL])
                        for _, r in sub.iterrows()], [True] * len(NOMINAL))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_nominal", lines)


def tab_step() -> None:
    df = C.summary("nominal")
    if df is None:
        return
    cols = [("step_rise_s", "rise [s]"), ("step_overshoot_pct", r"overshoot [\%]"),
            ("step_settle_s", "settle 2\\% [s]"), ("step_sse_mm", "ss error [mm]"),
            ("tau2_int_win", r"$\int\|\tau\|^2$"), ("yaw_peak_mrad", r"peak $|\psi|$ [mrad]")]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(l for _, l in cols) + r" \\"]
    for scen, title in (("step6", "6 cm diagonal step"), ("step12", "12 cm step along x")):
        sub = df[df["cell"].str.startswith(scen + "_")].copy()
        sub["tag"] = sub["cell"].str.split("_").str[-1]
        sub = C.ordered(sub)
        lines += [r"\midrule", rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\emph{{{title}}}}} \\"]
        lines += block([(C.LABEL[r["tag"]], [r[k] for k, _ in cols])
                        for _, r in sub.iterrows()], [True] * len(cols))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_step", lines)


def tab_timing() -> None:
    df = C.summary("timing")
    if df is None:
        return
    nom = df[~df["cell"].str.contains("_N")]
    cols = ["mean", "p50", "p95", "p99", "max", "headroom", "QP iter"]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{5}{c}{control tick [\textmu s]} & & \\",
             r"\cmidrule(lr){2-6}", " & " + " & ".join(cols) + r" \\", r"\midrule"]
    rows = []
    for c in C.CTRL:
        d = nom[nom["tag"] == c]
        if d.empty:
            continue
        p99 = d["tick_us_p99"].mean()
        rows.append((C.LABEL[c], [d["tick_us_mean"].mean(), d["tick_us_p50"].mean(),
                                  d["tick_us_p95"].mean(), p99, d["tick_us_max"].max(),
                                  C.PERIOD_US / p99, d["qp_iter_mean"].mean()]))
    lines += block(rows, [True] * 5 + [False, True])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_timing", lines)


def tab_sensitivity() -> None:
    df = C.summary("sensitivity")
    if df is None:
        return
    idx = C.sensitivity_index()
    meta = pd.DataFrame([dict(cell=k, **v) for k, v in idx.items()])
    df = df.merge(meta, on="cell")
    v = C.VARIANT
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             rf"controller & knob & range tested & RMSE, best & RMSE, worst & {v} \\",
             rf" & & ($\times$ {v}) & [mm] & [mm] & [mm] \\", r"\midrule"]
    for c in C.CTRL:
        sub = df[df["ctrl"] == c]
        for knob, d in sub.groupby("knob", sort=False):
            nominal = d[d["nominal"]]["rmse_mm"].iloc[0]
            rng = f"{d['mult'].min():.2g}--{d['mult'].max():.2g}"
            lines.append(f"{C.LABEL[c]} & \\texttt{{{knob}}} & {rng} & "
                         f"{fmt(d['rmse_mm'].min())} & {fmt(d['rmse_mm'].max())} & "
                         f"{fmt(nominal)} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("tab_sensitivity", lines)


def tab_observer() -> None:
    df = C.summary("observer")
    if df is None:
        return
    parts = df["cell"].map(C.observer_parts)
    df["obs_scen"], df["src"], df["ff"] = parts.str[0], parts.str[1], parts.str[2]
    for scen, name in (("circle-tq", "tab_observer_torque"), ("circle-m2", "tab_observer_mass"),
                       ("circle", "tab_observer_circle")):
        lines = [r"\begin{tabular}{ll" + "r" * len(C.CTRL) + "}", r"\toprule",
                 r"state & ff gain & " + " & ".join(C.LABEL[c] for c in C.CTRL) + r" \\",
                 r"\midrule"]
        d = df[df["obs_scen"] == scen]
        for src, src_label in (("exact", "exact"), ("raw", "raw sensors"),
                               ("ekf-jkf", "EKF + joint KF")):
            for ff in (0.0, 0.5, 1.0):
                cells = []
                for c in C.CTRL:
                    m, h = C.ci95(d[(d["tag"] == c) & (d["src"] == src) & (d["ff"] == ff)]["rmse_mm"])
                    cells.append(fmt(m) if h == 0 else f"{fmt(m)} $\\pm$ {fmt(h, 2)}")
                lines.append(f"{src_label} & {ff:g} & " + " & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        lines[-1] = r"\bottomrule"
        lines.append(r"\end{tabular}")
        write(name, lines)


def tab_tuned() -> None:
    """What the tuning changed: every knob, shipped and tuned, and the objective."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")
    if not os.path.isfile(path):
        return
    import json
    with open(path) as fh:
        tuned = json.load(fh)
    import tune
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"controller & knob & shipped & tuned \\", r"\midrule"]
    for c in C.CTRL:
        if c not in tuned:
            continue
        start = tune.SPACES[c]()[2]
        for knob, v in tuned[c]["state"].items():
            lines.append(f"{C.LABEL[c]} & \\texttt{{{knob.replace('_', chr(92) + '_')}}} & "
                         f"{fmt(start[knob])} & {fmt(v)} \\\\")
        g = tuned[c]["gmean_rmse_mm"]
        lines.append(f" & \\emph{{objective [mm]}} & {fmt(g['shipped'])} & "
                     f"{fmt(g['tuned'])}{'' if tuned[c]['feasible'] else ' (infeasible)'} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    d = os.path.join(C.OUT_ROOT, "tables")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tab_tuned.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote tables/tab_tuned.tex")


TABLES: List[Callable[[], None]] = [tab_nominal, tab_step, tab_timing, tab_sensitivity,
                                    tab_observer, tab_tuned]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=("shipped", "tuned"), default=C.studies.DEFAULT_VARIANT)
    C.set_variant(ap.parse_args().variant)
    print(f"writing to {os.path.join(C.OUT, 'tables')}")
    for t in TABLES:
        t()
