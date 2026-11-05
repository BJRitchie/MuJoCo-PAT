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
    # LF on every OS, so a table rewritten on Windows is not a spurious diff.
    with open(os.path.join(d, name + ".tex"), "w", newline="\n") as fh:
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
    cols = ["mean", "p50", "p95", "p99", "max", "headroom p99", "headroom max", "QP iter"]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{5}{c}{control tick [\textmu s]} & & & \\",
             r"\cmidrule(lr){2-6}", " & " + " & ".join(cols) + r" \\", r"\midrule"]
    rows = []
    for c in C.CTRL:
        d = nom[nom["tag"] == c]
        if d.empty:
            continue
        p99 = d["tick_us_p99"].mean()
        worst = d["tick_us_max"].max()      # the worst tick of every run, not a mean
        rows.append((C.LABEL[c], [d["tick_us_mean"].mean(), d["tick_us_p50"].mean(),
                                  d["tick_us_p95"].mean(), p99, worst,
                                  C.PERIOD_US / p99, C.PERIOD_US / worst,
                                  d["qp_iter_mean"].mean()]))
    lines += block(rows, [True] * 5 + [False, False, True])
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


#: Each tuning knob: the YAML key it sets (and which [x, y, thetaz] components),
#: and what it does. The family is the law (both NMPCs share theirs).
TUNED_KNOBS = {
    "nmpc": {
        "qpos_xy": (r"weights.Q_pos", "x, y", "weight on the EE position error"),
        "qpos_th": (r"weights.Q_pos", r"$\theta_z$", "weight on the EE orientation error"),
        "qvel_xy": (r"weights.Q_vel", "x, y", "weight on the EE velocity error"),
        "qvel_th": (r"weights.Q_vel", r"$\theta_z$", "weight on the EE angular-velocity error"),
        "r": (r"weights.R", "all", "weight on the task-space force (control effort)"),
        "N": (r"nmpc.N", "", "prediction horizon [steps]"),
        "Ts": (r"nmpc.Ts", "", r"horizon step [s]; the horizon spans $N\,T_s$ "
               "(the control period stays 10 ms)"),
        "term": (r"nmpc.terminal_cost_multiplier", "",
                 r"terminal weight, as a multiple of $Q$"),
    },
    "ompc": {
        "Ts": (r"ompc.Ts", "", "how far ahead the error is predicted [s]; sets how hard "
               "the law pushes, not the control period"),
        "q_xy": (r"ompc.Q", "x, y", r"weight on the predicted EE error (against \texttt{ompc.R} = 1)"),
        "q_th": (r"ompc.Q", r"$\theta_z$", "weight on the predicted orientation error"),
    },
    "smc": {
        "lam_xy": (r"smc.lam", "x, y", r"sliding-surface slope $\lambda$ [1/s]: the error's decay "
                   "rate once on the surface"),
        "lam_th": (r"smc.lam", r"$\theta_z$", r"sliding-surface slope $\lambda$ for orientation [1/s]"),
        "alpha": (r"smc.alpha", "all", r"floor of the switching gain $\rho$ [m/s$^2$, rad/s$^2$]"),
        "epsilon": (r"smc.epsilon", "", r"boundary-layer width [units of $\sigma$]: linear "
                    "instead of sign() inside it"),
        "tau_bound": (r"smc.tau_bound", "", r"force bound in $\rho$'s robustness term "
                      r"[N\,m]; 0 = the torque limit (0.25)"),
        "k2": (r"smc.k2", "", r"gain of $\rho$'s robustness term (model-error bound)"),
    },
}


def _exact(v) -> str:
    """A setting as the bench was given it (tune.py writes 6 significant figures),
    so the table reproduces the run rather than rounding it."""
    if isinstance(v, str):
        return r"\texttt{" + v + "}"
    s = f"{v:.6g}"
    if abs(v) >= 1e5 and "e" not in s:          # 100000 as 1e5, like 1.6e6
        s = f"{v:e}"
    if "e" in s:
        m, e = s.split("e")
        m = m.rstrip("0").rstrip(".")
        return f"${m}\\times10^{{{int(e)}}}$"
    return s


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _set_args(args: Sequence[str]) -> dict:
    """{key: value} from a `--set key=value ...` argument list."""
    return dict(a.split("=", 1) for a in args[1::2])


def _yaml_scalar(path: str, section: str, key: str) -> str:
    """One `key:` under `section:` in a law's YAML, without a YAML library."""
    import re
    text = _read(path)
    m = re.search(rf"^\s+{section}:\s*\n((?:\s+.*\n)*?)\s+{key}:\s*([^\s#]+)", text, re.M)
    return m.group(2) if m else "?"


def _broken(tuned: dict) -> str:
    """Which tuning constraint each infeasible tuned setting breaks, and where."""
    import tune
    out = []
    for c in C.CTRL:
        if c not in tuned or tuned[c]["feasible"]:
            continue
        for scen, m in tuned[c]["per_scenario"]["tuned"].items():
            name = scen.replace("2x", r" 2$\times$").replace("step6", "6 cm step")
            if m["rail_frac"] > tune.RAIL_MAX:
                out.append(f"{C.LABEL[c]}: {100 * m['rail_frac']:.1f}\\,\\% of joint-ticks at "
                           f"the torque limit on the {name} (limit "
                           f"{100 * tune.RAIL_MAX:.0f}\\,\\%)")
            if m["safe_ticks"] > 0:
                out.append(f"{C.LABEL[c]}: {m['safe_ticks']:.0f} safe-mode ticks on the {name}")
            if m["tick_us_max"] > tune.TICK_MAX_US:
                out.append(f"{C.LABEL[c]}: a {m['tick_us_max'] / 1000:.1f}\\,ms tick on the {name}")
    return " (" + "; ".join(out) + ")" if out else ""


def tab_tuned() -> None:
    """Every controller setting the studies run with that differs from or is
    chosen against the committed YAML: the tuned knobs, the solver path and the
    QP cap, each with its YAML key, what it does, the shipped and tuned values (as
    run, unrounded), and the tuning objective. Everything else is the committed
    config, stated in the table's last row."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")
    if not os.path.isfile(path):
        return
    import json
    with open(path) as fh:
        tuned = json.load(fh)
    import tune
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                       "pat_arm_control", "config")
    shipped_args = {c: _set_args(a) for c, _, a in C.studies.CONTROLLERS}

    def row(key: str, comp: str, desc: str, shipped, new) -> str:
        k = r"\texttt{" + key.replace("_", r"\_").replace(".", r".\allowbreak ") + "}"
        return f"{k}{' ' + comp if comp else ''} & {desc} & {_exact(shipped)} & {_exact(new)} \\\\"

    lines = [r"\begin{tabular}{@{}p{4.8cm}p{6.0cm}rr@{}}", r"\toprule",
             r"setting & what it does & shipped & tuned \\", r"\midrule"]
    for c in C.CTRL:
        if c not in tuned:
            continue
        law = tuned[c]["law"]
        start = tune.SPACES[c]()[2]
        args = _set_args(tuned[c]["args"])
        lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + C.LABEL[c] + r"}} \\")
        if law == "nmpc":
            fn_ship = shipped_args[c].get("nmpc.full_nonlinear",
                                          _yaml_scalar(os.path.join(cfg, "nmpc.yaml"),
                                                       "nmpc", "full_nonlinear"))
            lines.append(row("nmpc.full_nonlinear", "", "solver: true = multiple-shooting "
                             "SQP, false = one QP linearised about the last plan (LMPC)",
                             fn_ship, args.get("nmpc.full_nonlinear", fn_ship)))
        for knob, v in tuned[c]["state"].items():
            key, comp, desc = TUNED_KNOBS[law][knob]
            lines.append(row(key, comp, desc, start[knob], float(f"{v:.6g}")))
        if "qp.max_iter" in args:
            lines.append(row("qp.max_iter", "", "QP iterations allowed per solve: the "
                             "realtime bound, set after tuning, not tuned",
                             int(_yaml_scalar(os.path.join(cfg, "nmpc.yaml"), "qp", "max_iter")),
                             int(args["qp.max_iter"])))
        g = tuned[c]["gmean_rmse_mm"]
        mark = "" if tuned[c]["feasible"] else r"$^\dagger$"
        lines.append(r"\emph{objective} & geometric-mean EE RMSE over the tuning scenarios "
                     f"[mm] & {fmt(g['shipped'])} & {fmt(g['tuned'])}{mark} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\midrule"
    lines.append(r"\multicolumn{4}{@{}p{\linewidth}@{}}{\footnotesize Every other setting is "
                 r"the committed \texttt{arm\_common.yaml} and the law's own YAML in "
                 r"\texttt{pat\_arm\_control/config/}, among them the 100\,Hz control rate, "
                 r"the 0.25\,N\,m torque limit and the joint limits. A component not listed "
                 r"keeps its shipped value. $^\dagger$ broke a tuning constraint in the tuning run"
                 + _broken(tuned) + r".} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    d = os.path.join(C.OUT_ROOT, "tables")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tab_tuned.tex"), "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote tables/tab_tuned.tex")


def _pm(m: float, h: float) -> str:
    r"""mean $\pm$ 95% half-width, the half-width dropped when there is none."""
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    return fmt(m) if not h else fmt(m) + r"\,$\pm$\," + fmt(h, 2)


def tab_live() -> None:
    """The live stack against its offline twin and the exact-state bench, per
    controller and shape, at the nominal setting on the EKF; then the live
    timing, worst tick first."""
    if C.VARIANT != "tuned":
        return                          # the live stack runs tuned.json only
    live = C.live_summary()
    if live is None:
        return
    twin = C.live_twin()
    exact = C.summary("nominal")
    lv = C.at_nominal(live[live["source"] == "ekf"])
    tw = C.at_nominal(twin[twin["source"] == "ekf"]) if twin is not None else None
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{circle RMSE [mm]} & \multicolumn{3}{c}{rectangle RMSE [mm]}"
             r" & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r" & exact & twin & live & exact & twin & live & live trials \\", r"\midrule"]
    for c in C.CTRL:
        cells = []
        for shape in ("circle", "rectangle"):
            e = exact[(exact["scenario"] == shape) & (exact["tag"] == c)]["rmse_mm"] \
                if exact is not None else []
            cells.append(fmt(float(e.iloc[0])) if len(e) else "--")
            t = tw[(tw["shape"] == shape) & (tw["tag"] == c)]["rmse_mm"] if tw is not None else []
            cells.append(_pm(*C.ci95(t)) if len(t) else "--")
            d = lv[(lv["shape"] == shape) & (lv["tag"] == c)]
            cells.append(_pm(*C.ci95(d["rmse_mm"])) + (r"$^\times$" if
                         len(d) and d["safe_ticks"].max() > 0 else ""))
        n = len(lv[lv["tag"] == c])
        lines.append(f"{C.LABEL[c]} & " + " & ".join(cells) + f" & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_live_accuracy", lines)

    # Timing over EVERY live trial of the controller, all cells: the worst tick
    # is the maximum over all of them, not a mean of per-trial maxima.
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{control tick [\textmu s]} & & & \\",
             r"\cmidrule(lr){2-4}",
             r" & mean & p99 & longest & headroom (longest) & missed deadlines & trials \\",
             r"\midrule"]
    rows = []
    for c in C.CTRL:
        d = live[live["tag"] == c]
        if d.empty:
            continue
        worst = d["tick_us_max"].max()
        rows.append((C.LABEL[c], [d["tick_us_mean"].mean(), d["tick_us_p99"].mean(), worst,
                                  C.PERIOD_US / worst, d["deadline_misses"].fillna(0).sum()],
                     len(d)))
    body = block([(label, v) for label, v, _ in rows], [True, True, True, False, True])
    # The trial count is context, not a result: appended unbolded.
    lines += [b[:-len(r" \\")] + f" & {n} \\\\" for b, (_, _, n) in zip(body, rows)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_live_timing", lines)


SAFE_MARK = r"$^\times$"


def _mm(v: float) -> str:
    """An RMSE as the chapter quotes it: one decimal under 10 mm, whole above."""
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


#: (label, knob filter): the live rectangle conditions of tab:degradation, in its order.
DEGRADATION = [("nominal", {}), (r"2$\times$ speed", {"speed": 2.0}),
               (r"link mass $0.5\times$", {"mass": 0.5}), (r"link mass $2\times$", {"mass": 2.0}),
               (r"link mass $3\times$", {"mass": 3.0}),
               (r"base damping 0.5\,N\,s/m", {"damping": 0.5}),
               (r"base damping 5\,N\,s/m", {"damping": 5.0}),
               (r"torque limit $0.5\times$", {"tau": 0.5}),
               ("raw sensors (no EKF)", {"source": "raw"})]


def tab_live_degradation() -> None:
    """Live rectangle RMSE, mean of three repeats, SQP-NMPC against the best of
    the three baselines under each non-ideal condition, and their ratio. A value
    whose trials entered safe mode carries a cross."""
    if C.VARIANT != "tuned":
        return
    live = C.live_summary()
    if live is None:
        return
    rect = live[live["shape"] == "rectangle"]
    lines = [r"\begin{tabular}{lrlr}", r"\toprule",
             r"condition (live, rectangle) & SQP-NMPC [mm] & best baseline [mm] & ratio \\",
             r"\midrule"]
    for label, knobs in DEGRADATION:
        d = rect[rect["source"] == knobs.get("source", "ekf")]
        knob = next((k for k in knobs if k != "source"), None)
        d = C.at_nominal(d, but=knob)
        if knob:
            d = d[(d[knob].astype(float) - knobs[knob]).abs() < 1e-9]
        g = d.groupby("tag").agg(m=("rmse_mm", "mean"), safe=("safe_ticks", "max"))
        if "sqp" not in g.index or len(g) < 2:
            continue
        base = g.drop("sqp")["m"]
        b = base.idxmin()
        mark = lambda t: SAFE_MARK if g.loc[t, "safe"] > 0 else ""  # noqa: E731
        ratio = base[b] / g.loc["sqp", "m"]
        lines.append(f"{label} & {_mm(g.loc['sqp', 'm'])}{mark('sqp')} & "
                     f"{_mm(base[b])}{mark(b)} ({C.LABEL[b]}) & {_mm(ratio)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_live_degradation", lines)


def _hist_quantile(h: dict, q: float) -> float:
    """A quantile of a tick histogram (C.read_tick_hist), interpolated within its
    bin on the log scale (bins are 2.3% wide)."""
    e = C.hist_edges()
    c = np.asarray(h["counts"], dtype=float)
    cum = np.cumsum(c) / max(h["n"], 1)
    i = int(np.searchsorted(cum, q))
    lo = cum[i - 1] if i else 0.0
    f = (q - lo) / (cum[i] - lo) if cum[i] > lo else 0.5
    return float(e[i] * (e[i + 1] / e[i]) ** f)


def _pooled(summary: pd.DataFrame, hist: dict, tag: str) -> List[float]:
    """[mean, p99, worst, ticks over the period] over every tick of a run set:
    the mean weighted by each run's tick count, p99 and worst from the pooled
    histogram (the worst is its exact recorded maximum)."""
    d = summary[summary["tag"] == tag]
    h = hist.get(tag) if hist else None
    if d.empty or h is None or not h["n"]:
        return [float("nan")] * 4
    w = d["n_ticks"].astype(float)
    mean = float((d["tick_us_mean"] * w).sum() / w.sum())
    over = float(np.asarray(h["counts"])[C.hist_edges()[:-1] >= C.PERIOD_US].sum())
    return [mean, _hist_quantile(h, 0.99), float(h["max"]), over]


def _pct(v: float) -> str:
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def tab_timing_all() -> None:
    """Control tick per controller on every platform, pooled over all its ticks:
    the bench (DGX, pinned, the timing study), the live stack (all trials), and
    the Jetson comparison's 132 cells on the DGX and the Orin Nano (warm start
    with a cold retry at cap 20; cold start, no retry, caps 17/18)."""
    if C.VARIANT != "tuned":
        return
    blocks = []
    t = C.summary("timing")
    if t is not None:
        t = t[~t["cell"].str.contains("_N")]
        hists = {}
        for c in C.CTRL:
            ticks = [s["tick_us"].to_numpy() for s in
                     (C.series("timing", cell) for cell in t[t["tag"] == c]["cell"]) if s is not None]
            if ticks:
                hists[c] = C.tick_hist_of(np.concatenate(ticks))
        if hists:
            blocks.append((r"bench: DGX, pinned, simulated time waits (1$\times$ circle, 5 runs)",
                           t, hists))
    live = C.live_summary()
    lh = C.read_tick_hist(os.path.join(C.LIVE_DIR, C.LIVE_RUN, "tick_hist.json"))
    if live is not None and lh:
        blocks.append((r"live: DGX, \texttt{SCHED\_FIFO} 80, real time (444 trials, 37 cells)",
                       live, lh))
    machines = os.path.join(C.DATA_ROOT, "machines")
    for run, label in (("dgx_n24", r"DGX, Jetson cells: warm start, cold retry, cap 20"),
                       ("orin_nano", r"Orin Nano: warm start, cold retry, cap 20"),
                       ("orin_nano_cold_cap17_18", r"Orin Nano: cold start, no retry, caps 17/18")):
        sp = os.path.join(machines, run, "jetson_timing", "summary.csv")
        h = C.read_tick_hist(os.path.join(machines, run, "jetson_timing", "tick_hist.json"))
        if os.path.isfile(sp) and h:
            blocks.append((label + r" (11 scenarios $\times$ 3 seeds)", pd.read_csv(sp), h))
    if not blocks:
        return
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{control tick [\textmu s]} & longest / & ticks \\",
             r"\cmidrule(lr){2-4}",
             r" & mean & p99 & longest & period & over 10\,ms \\"]
    for label, summ, hist in blocks:
        lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\emph{{{label}}}}} \\"]
        for c in C.CTRL:
            m, p99, worst, over = _pooled(summ, hist, c)
            if math.isnan(m):
                continue
            lines.append(f"{C.LABEL[c]} & {fmt(m)} & {fmt(p99)} & {fmt(worst)} & "
                         f"{_pct(100 * worst / C.PERIOD_US)}\\,\\% & {int(over)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_timing_all", lines)


def _levels(study: str, pattern: str) -> List[str]:
    """The distinct values a study's cells take, read from their names."""
    import re
    vals = []
    for cell in C.studies.build(study):
        m = re.search(pattern, cell.name)
        v = f"{float(m.group(1)):g}" if m else None
        if v is not None and v not in vals:
            vals.append(v)
    return vals


def _seeds(study: str) -> int:
    return len({cell.seed for cell in C.studies.build(study)})


def _live_levels(study: str, key: str) -> List[str]:
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live", "live_studies.py")
    spec = importlib.util.spec_from_file_location("live_studies", path)
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    out = []
    for v in ls.STUDIES[study]():
        if key in v:
            s = f"{float(v[key]):g}"
            if s not in out:
                out.append(s)
    return out


def tab_conditions() -> None:
    """Every condition the chapter tests, its levels, and where it was run: the
    bench (deterministic; seeds where the sensors are noisy) and the live stack
    (three repeats per cell). Levels are read from studies.py and
    live/live_studies.py, so the table follows the studies."""
    if C.VARIANT != "tuned":
        return
    S = C.studies
    join = lambda v, unit="": ", ".join(v) + unit  # noqa: E731
    live = C.live_summary()
    reps = int(live["repeat"].nunique()) if live is not None else 3
    live_note = f"{reps} repeats"
    rows = [
        ("reference", r"6 and 12\,cm steps", "yes", "--"),
        ("", r"circle (6\,cm radius, 8\,s lap); rectangle (12$\times$10\,cm, 5.5\,cm/s)",
         "yes", live_note),
        ("speed", r"circle " + join(_levels("speed", r"circle_x([0-9.]+)")) + r"$\times$; "
         r"rectangle " + join(_levels("speed", r"rectangle_x([0-9.]+)")) + r"$\times$",
         "yes", r"same, " + live_note),
        ("link mass", join(_levels("model_error", r"circle_m([0-9.]+)"), r"$\times$ model"),
         "yes", join(_live_levels("model_error", "mass_scale"), r"$\times$")),
        ("base damping", join(_levels("damping", r"circle_bd([0-9.]+)"), r"\,N\,s/m"), "yes",
         join(_live_levels("damping", "damping"))),
        ("torque limit", join(_levels("torque_limit", r"circle_tau([0-9.]+)x"),
                              r"$\times$ 0.25\,N\,m"), "yes",
         join(_live_levels("torque_limit", "tau_scale"), r"$\times$")),
        ("joint torque", join(_levels("disturbance", r"circle_tq([0-9.]+)"), r"\,N\,m"), "yes",
         "-- (body wrench only)"),
        ("partner arm", "holds; mirrors; same direction", "yes", "holds"),
        ("state source", "exact; raw flight sensors; EKF; EKF + joint KF",
         f"exact; others {_seeds('estimator') - 1} seeds", "raw, EKF, EKF + joint KF"),
        ("sensor noise", join(_levels("noise", r"circle_n([0-9.]+)"), r"$\times$ flight profile"),
         f"{_seeds('noise')} seeds", "flight profile"),
        ("state delay", join(_levels("delay", r"circle_d([0-9]+)"), " ticks"),
         f"{_seeds('delay')} seeds", "real latency"),
        ("estimator warm-up", rf"{S.NAV_WARMUP}\,s (bench), {S.LIVE_NAV_WARMUP}\,s (live, twins)",
         "yes", "yes"),
        ("tick timing", r"DGX pinned; live \texttt{SCHED\_FIFO} 80; Orin Nano",
         f"{S.REPEATS} runs", "all trials"),
    ]
    lines = [r"\begin{tabular}{p{2.3cm}p{6.6cm}ll}", r"\toprule",
             r"condition & levels & testbench & live \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_conditions", lines)


# ------------------------------------------------------- settings tables --
# One table per component: what every study ran with, read from the committed
# config and model files when the table is written, so it cannot drift from
# them. Each row: the key (as it appears in its file), what it does, the value.
# tab_tuned holds the tuned controller knobs; tab_conditions what each study
# varies; these hold everything else.

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _src(*parts: str) -> str:
    return os.path.join(REPO, "src", *parts)


def _flow(s: str, i: int = 0):
    """Parse one YAML flow value ([..], {..} or a scalar) at s[i]: (value, next i)."""
    while s[i] == " ":
        i += 1
    if s[i] in "[{":
        close, out = ("]", []) if s[i] == "[" else ("}", {})
        i += 1
        while True:
            while s[i] in " ,\n":
                i += 1
            if s[i] == close:
                return out, i + 1
            if close == "]":
                v, i = _flow(s, i)
                out.append(v)
            else:
                j = s.index(":", i)
                key = s[i:j].strip()
                v, i = _flow(s, j + 1)
                out[key] = v
    j = i
    while j < len(s) and s[j] not in ",]}":
        j += 1
    return _scalar(s[i:j]), j


def _scalar(t: str):
    t = t.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    if t in ("true", "false"):
        return t == "true"
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def _yaml(path: str) -> dict:
    """The block-mapping YAML subset the repo's config files use (nested maps,
    flow lists and dicts, possibly over several lines), without a YAML library."""
    import re
    lines = []
    for raw in _read(path).splitlines():
        line = re.sub(r"\s+#.*$", "", raw.rstrip("\n"))
        if line.strip() and not line.lstrip().startswith("#"):
            lines.append(line)
    root: dict = {}
    stack = [(-1, root)]
    k = 0
    while k < len(lines):
        line = lines[k]
        ind = len(line) - len(line.lstrip())
        key, _, rest = line.strip().partition(":")
        rest = rest.strip()
        while rest.count("[") + rest.count("{") > rest.count("]") + rest.count("}"):
            k += 1
            rest += " " + lines[k].strip()
        while stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip().strip("\"'")
        if rest:
            parent[key] = _flow(rest)[0]
        else:
            parent[key] = {}
            stack.append((ind, parent[key]))
        k += 1
    return root


def _get(d: dict, *path: str):
    for p in path:
        d = d[p]
    return d


def _val(v, unit: str = "") -> str:
    """A setting's value for the table: numbers as run, lists bracketed."""
    if isinstance(v, bool):
        return r"\texttt{" + str(v).lower() + "}"
    u = rf"\,{unit}" if unit else ""
    if isinstance(v, (list, tuple)):
        if len(v) > 1 and len(set(map(str, v))) == 1:
            return _val(v[0]) + u + " (each)"
        return "[" + ", ".join(_val(x) for x in v) + "]" + u
    if isinstance(v, str):
        return r"\texttt{" + v.replace("_", r"\_") + "}"
    if isinstance(v, int):
        return str(v) + u                    # a count or a seed, never 2.0e7
    return _exact(v) + u


def _settings_table(name: str, sections: Sequence[Tuple[str, Sequence[Tuple[str, str, str]]]],
                    note: str = "") -> None:
    """sections: (heading, [(key, description, value), ...]). A key is set as code
    and breaks after a dot; one starting with @ is a plain-text name (not a key in
    any file). The description wraps."""
    lines = [r"\begin{tabular}{@{}p{4.3cm}p{7.2cm}r@{}}", r"\toprule",
             r"setting & what it does & value \\", r"\midrule"]
    for heading, rows in sections:
        lines.append(r"\multicolumn{3}{@{}l}{\textbf{" + heading + r"}} \\")
        for key, desc, value in rows:
            k = key[1:] if key.startswith("@") else \
                r"\texttt{" + key.replace("_", r"\_").replace(".", r".\allowbreak ") + "}"
            lines.append(f"{k} & {desc} & {value} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule" if not note else r"\midrule"
    if note:
        lines += [r"\multicolumn{3}{@{}p{\linewidth}@{}}{\footnotesize " + note + r"} \\",
                  r"\bottomrule"]
    lines.append(r"\end{tabular}")
    write(name, lines)


def _xacro_attr(path: str, pattern: str) -> List[str]:
    import re
    return re.findall(pattern, _read(path), re.S)


def tab_settings_sim() -> None:
    """The plant: MuJoCo, the scene, the simulator node, and the chaser model. The
    bench steps the same scene MJCF, so these hold for both."""
    sim = _get(_yaml(_src("pat_simulation", "config", "simulation.yaml")),
               "pat_simulation", "ros__parameters")
    flight = _get(_yaml(_src("pat_simulation", "config", "sensors_flight.yaml")),
                  "pat_simulation", "ros__parameters")
    scene = _src("pat_simulation", "models", "environment", "air_bearing_table.xacro")
    arm = _src("pat_platform_description", "xacro", "piper_planar_arm.xacro")
    bus = _src("pat_platform_description", "xacro", "pat_platform_bus.xacro")
    import re
    otag = re.search(r"<option[^>]*>", _read(scene)).group(0)
    opt = dict(re.findall(r'(\w+)="([^"]+)"', otag))
    bus_mass, bus_inertia = _xacro_attr(bus, r'<inertial[^>]*?mass="([^"]+)"\s*'
                                             r'fullinertia="([^"]+)"')[0]
    izz = float(bus_inertia.split()[2])
    links = _xacro_attr(arm, r'<body name="link(\d)_\$\{side\}".*?mass="([^"]+)"')
    moving = sum(float(m) for n, m in links if int(n) >= 2)     # link1 is welded
    pos = dict(_xacro_attr(arm, r'<body name="link(\d)_\$\{side\}" pos="([^"]+)"'))
    seg = [float(pos["3"].split()[0]),
           math.hypot(*map(float, pos["4"].split()[:2])),
           math.hypot(*map(float, pos["6"].split()[:2]))]
    rng = dict(_xacro_attr(arm, r'<joint name="joint(\d)_\$\{side\}"[^>]*?range="([^"]+)"'))
    tgt = _xacro_attr(scene, r'<body name="target"[^>]*>.*?<inertial[^>]*?mass="([^"]+)"')[0]
    ctrl = [dict(_xacro_attr(scene, r'<motor name="motor(\d)_L"[^>]*?ctrlrange="([^"]+)"'))[j]
            for j in "235"]
    common = _get(_yaml(_src("pat_arm_control", "config", "arm_common.yaml")),
                  "/**", "ros__parameters")
    deg = lambda r: f"{math.degrees(r):.0f}"  # noqa: E731
    q0 = sim["initial_arm_qpos"][:3]
    _settings_table("tab_settings_sim", [
        ("Physics (MuJoCo, scene MJCF)", [
            ("option timestep", "physics step, the same in the live simulator and the bench",
             _val(float(opt["timestep"]), "s")),
            ("option integrator", "numerical integrator", _val(opt["integrator"])),
            ("option gravity", "off: the air-bearing table as a planar microgravity analogue",
             _val(opt["gravity"].replace(" ", ", "))),
            ("sim_rate_hz", "physics rate of the live simulator", _val(sim["sim_rate_hz"], "Hz")),
            ("pub_rate_hz", "rate the joint states, the odometry and \\texttt{/clock} are "
             "published", _val(sim["pub_rate_hz"], "Hz")),
            ("imu_rate_hz", "rate the IMU is sampled and published (flight overlay)",
             _val(flight["imu_rate_hz"], "Hz")),
            ("control_hz", "arm control rate (live and bench: 10\\,ms ticks)",
             _val(common["control_hz"], "Hz")),
        ]),
        ("Table and plant", [
            ("chaser_damping", r"table drag on $x$, $y$, yaw [N\,s/m, N\,m\,s/rad]; every study "
             "sets 0 (frictionless) except the damping study", _val(sim["chaser_damping"])),
            ("chaser_frictionloss", "dry friction on the same three axes",
             _val(sim["chaser_frictionloss"])),
            ("arm_torque_timeout", "an arm command older than this is zeroed, as a motor "
             "driver's watchdog would", _val(sim["arm_torque_timeout"], "s")),
            ("initial_arm_qpos", rf"start pose of joints 2, 3, 5, each arm: {deg(q0[0])}, "
             rf"{deg(q0[1])}, {deg(q0[2])}\,deg", _val([round(v, 4) for v in q0], "rad")),
            ("sensor_seed", "noise seed; the live runs and the bench set it per repeat",
             _val(sim["sensor_seed"])),
        ]),
        ("Chaser model (xacro)", [
            ("@bus mass", "the PAT servicer bus", _val(float(bus_mass), "kg")),
            ("@bus $I_{zz}$", "yaw inertia of the bus about its centre",
             _val(izz, r"kg\,m$^2$")),
            ("@arm joints", "two planar arms, three revolute joints each (2, 3, 5 of the "
             "Piper; 1, 4, 6 welded)", "3 per arm"),
            ("@arm segments", "joint 2 to 3, joint 3 to 5, joint 5 to EE",
             _val([round(v, 3) for v in seg], "m")),
            ("@arm moving mass", "mass that joint 2 carries, each arm", _val(round(moving, 3), "kg")),
            ("@joint range", "MJCF limits of joints 2, 3, 5 (the controllers keep 0.05 rad "
             "inside them)", "[" + "; ".join(rng[j].replace(" ", ", ") for j in "235")
             + r"]\,rad"),
            ("@motor ctrlrange", "MJCF actuator limit of joints 2, 3, 5; the controllers "
             "clamp to \\texttt{model\\_tau\\_max} (0.25\\,N\\,m)",
             "[" + ", ".join(_val(float(r.split()[1])) for r in ctrl) + r"]\,N\,m"),
            ("@target mass", "the client mock-up, its centre 0.9\\,m ahead of the chaser's",
             _val(float(tgt), "kg")),
        ]),
    ])


def tab_settings_sensors() -> None:
    """The flight sensors (sensors_flight.yaml + the encoder overlay): every channel
    with an error. The bench reads the same files for its raw and EKF sources."""
    fl = _get(_yaml(_src("pat_simulation", "config", "sensors_flight.yaml")),
              "pat_simulation", "ros__parameters")
    enc = _get(_yaml(_src("pat_simulation", "config", "sensors_flight_encoders.yaml")),
               "pat_simulation", "ros__parameters", "sensors")
    what = {"imu_accel_x": (r"accelerometer $x$", r"m/s$^2$"),
            "imu_accel_y": (r"accelerometer $y$", r"m/s$^2$"),
            "imu_gyro_z": ("gyro yaw rate", "rad/s"),
            "nav_x": ("relative position, along the line of sight", "m"),
            "nav_y": ("relative position, across it", "m"),
            "star_yaw": ("star-tracker yaw", "rad")}
    lines = [r"\begin{tabular}{@{}p{4.6cm}lrrrrr@{}}", r"\toprule",
             r"channel & unit & noise $\sigma$ & bias & bias walk $\sigma$ & rate [Hz] & "
             r"delay [s] \\", r"\midrule"]
    rate = lambda c: _val(c["rate_hz"]) if c["rate_hz"] else str(int(fl["imu_rate_hz"]))  # noqa
    for ch, (desc, unit) in what.items():
        c = fl["sensors"][ch]
        lines.append(f"{desc} \\newline \\texttt{{{ch.replace('_', chr(92) + '_')}}} & {unit} & "
                     f"{_val(c['noise_std'])} & {_val(c['bias'])} & {_val(c['bias_walk_std'])} & "
                     f"{rate(c)} & {_val(c['delay_s'])} \\\\")
    for grp, desc, unit in (("joint_pos", "joint encoder angle", "rad"),
                            ("joint_vel", "joint encoder velocity", "rad/s")):
        c = next(iter(enc[grp].values()))
        lines.append(f"{desc} (every joint) \\newline \\texttt{{{grp.replace('_', chr(92) + '_')}}}"
                     f" & {unit} & {_val(c['noise_std'])} & {_val(c['bias'])} & "
                     f"{_val(c['bias_walk_std'])} & 100 & {_val(c['delay_s'])} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{7}{@{}p{\linewidth}@{}}{\footnotesize Every other channel error "
              r"(scale error, quantisation, dropout) is zero. Noise $\sigma$ is per sample at "
              r"the channel's rate; the bias walk is a density, unit/$\sqrt{\mathrm{s}}$. The "
              r"encoders are read once per control tick. The noise study scales noise, bias "
              r"and bias walk together. File: \texttt{pat\_simulation/config/sensors\_flight.yaml}"
              r" and \texttt{sensors\_flight\_encoders.yaml}.} \\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_settings_sensors", lines)


def tab_settings_estimation() -> None:
    """The base estimator (pat_gnc's ImuEkf, nav_flight.yaml) and the joint encoder
    Kalman filter (arm_common.yaml)."""
    nav = _get(_yaml(_src("pat_gnc", "config", "nav_flight.yaml")), "pat_nav", "ros__parameters")
    common = _get(_yaml(_src("pat_arm_control", "config", "arm_common.yaml")),
                  "/**", "ros__parameters")
    e = lambda k: nav["imu_ekf." + k]  # noqa: E731
    _settings_table("tab_settings_estimation", [
        ("Base estimator: IMU error-state EKF (\\texttt{pat\\_gnc})", [
            ("navigator", "the estimator the flight-sensing runs use", _val(nav["navigator"])),
            ("nav_hz", "filter rate: one IMU propagation per sample", _val(nav["nav_hz"], "Hz")),
            ("publish_hz", "rate the estimate is published to the arm controllers",
             _val(nav["publish_hz"], "Hz")),
            ("imu_ekf.gyro_noise_std", "gyro noise the filter assumes", _val(e("gyro_noise_std"), "rad/s")),
            ("imu_ekf.accel_noise_std", "accelerometer noise it assumes",
             _val(e("accel_noise_std"), r"m/s$^2$")),
            ("imu_ekf.gyro_bias_walk", r"gyro bias random-walk density (the bias is a state) "
             r"[rad/s/$\sqrt{\mathrm{s}}$]", _val(e("gyro_bias_walk"))),
            ("imu_ekf.accel_bias_walk", r"accelerometer bias random-walk density "
             r"[m/s$^2$/$\sqrt{\mathrm{s}}$]", _val(e("accel_bias_walk"))),
            ("imu_ekf.yaw_meas_std", "star-tracker noise it assumes", _val(e("yaw_meas_std"), "rad")),
            ("imu_ekf.pos_meas_std", "relative-position noise it assumes, both axes "
             "(the larger, along-sight figure)", _val(e("pos_meas_std"), "m")),
            ("imu_ekf.use_position", "fuse the relative-position fixes", _val(e("use_position"))),
            ("imu_ekf.use_twist", "fuse a velocity measurement (none in flight)", _val(e("use_twist"))),
            ("imu_ekf.latency_pos", "age of a position fix, placed by state-history replay",
             _val(e("latency_pos"), "s")),
            ("imu_ekf.latency_yaw", "age of a star-tracker reading", _val(e("latency_yaw"), "s")),
            ("imu_ekf.history_len", "IMU samples kept for the replay", _val(e("history_len"))),
            ("imu_ekf.gate_pos", r"$\chi^2$ gate on a position fix (2 dof, 0.1\,\%)", _val(e("gate_pos"))),
            ("imu_ekf.gate_yaw", r"$\chi^2$ gate on a yaw reading (1 dof, 0.1\,\%)", _val(e("gate_yaw"))),
            ("imu_ekf.init_pos_std", "initial position uncertainty", _val(e("init_pos_std"), "m")),
            ("imu_ekf.init_yaw_std", "initial yaw uncertainty", _val(e("init_yaw_std"), "rad")),
            ("imu_ekf.init_vel_std", "initial velocity uncertainty", _val(e("init_vel_std"), "m/s")),
            ("imu_ekf.init_gyro_bias_std", "initial gyro-bias uncertainty",
             _val(e("init_gyro_bias_std"), "rad/s")),
            ("imu_ekf.init_accel_bias_std", "initial accelerometer-bias uncertainty",
             _val(e("init_accel_bias_std"), r"m/s$^2$")),
            ("imu_ekf.max_imu_gap", "a longer IMU gap restarts the propagation",
             _val(e("max_imu_gap"), "s")),
        ]),
        ("Joint encoder Kalman filter (\\texttt{pat\\_arm\\_control})", [
            ("joint_filter", r"off by default; \texttt{kalman} in the EKF + joint KF runs",
             _val(common["joint_filter"])),
            ("joint_kf.accel_psd", r"process noise: white joint acceleration [rad$^2$/s$^3$]",
             _val(common["joint_kf.accel_psd"])),
            ("joint_kf.pos_meas_std", "encoder angle noise it assumes (the bench tells it "
             "the true value)", _val(common["joint_kf.pos_meas_std"], "rad")),
            ("joint_kf.vel_meas_std", "encoder velocity noise it assumes",
             _val(common["joint_kf.vel_meas_std"], "rad/s")),
            ("joint_kf.init_vel_std", "initial velocity uncertainty",
             _val(common["joint_kf.init_vel_std"], "rad/s")),
        ]),
    ], note=r"Warm-up before each arm task: 60\,s on the bench, 15\,s live. Files: "
            r"\texttt{pat\_gnc/config/nav\_flight.yaml}, "
            r"\texttt{pat\_arm\_control/config/arm\_common.yaml}.")


def tab_settings_arm() -> None:
    """What every arm controller shares (arm_common.yaml and each law's per-arm
    block) and the settings of each law that were not tuned (tab_tuned has those)."""
    cfg = lambda f: _yaml(_src("pat_arm_control", "config", f))  # noqa: E731
    common = _get(cfg("arm_common.yaml"), "/**", "ros__parameters")
    nm = cfg("nmpc.yaml")
    n = _get(nm, "/**", "ros__parameters")
    left = _get(nm, "pat_arm_nmpc_left", "ros__parameters", "limits")
    om = _get(cfg("ompc.yaml"), "/**", "ros__parameters", "ompc")
    sm = _get(cfg("smc.yaml"), "/**", "ros__parameters", "smc")
    sf = common["safe_mode"]
    _settings_table("tab_settings_arm", [
        ("Shared by every controller", [
            ("control_hz", "control rate", _val(common["control_hz"], "Hz")),
            ("model_tau_max", "torque limit of every arm joint", _val(common["model_tau_max"], r"N\,m")),
            ("limits.q_min", "lower joint limits (joints 2, 3, 5), 0.05\\,rad inside the MJCF",
             _val(left["q_min"], "rad")),
            ("limits.q_max", "upper joint limits", _val(left["q_max"], "rad")),
            ("limits.qd_max", "joint speed limits", _val(left["qd_max"], "rad/s")),
            ("safe_mode.damping_frac", r"safe mode commands joint damping $-k_d\dot q$, "
             r"$k_d$ = this $\times\,\tau_{\max}/\dot q_{\max}$", _val(sf["damping_frac"])),
            ("safe_mode.recovery_ticks", "good ticks needed to leave safe mode",
             _val(sf["recovery_ticks"])),
            ("safe_mode.tier3_latch_after_ticks", "ticks of unbroken safe mode after which "
             "it latches, with no automatic recovery", _val(sf["tier3_latch_after_ticks"])),
            ("observer", "disturbance observer (studied separately)", _val(common["observer"])),
            ("observer.ff_gain", "observer feedforward into the command", _val(common["observer.ff_gain"])),
        ]),
        ("NMPC, both formulations (not tuned)", [
            ("nmpc.sqp_iters", "SQP iterations per tick (SQP-NMPC)", _val(n["nmpc"]["sqp_iters"])),
            ("nmpc.reference_preview", "score each horizon node against the reference at its "
             "own time, so the trajectory is seen coming", _val(n["nmpc"]["reference_preview"])),
            ("nmpc.terminal_velocity_constraint", "hard constraint: zero EE velocity error at "
             "the horizon's end", _val(n["nmpc"]["terminal_velocity_constraint"])),
            ("nmpc.du_max_frac", "SQP trust region, as a fraction of the reachable wrench",
             _val(n["nmpc"]["du_max_frac"])),
            ("nmpc.rollout_v_clamp_mult", "rollout joint-speed clamp, in multiples of "
             "\\texttt{qd\\_max}", _val(n["nmpc"]["rollout_v_clamp_mult"])),
            ("safe_mode.tier1_max_replay_ticks", "ticks a failed solve may replay the last "
             "plan before safe mode", _val(n["safe_mode"]["tier1_max_replay_ticks"])),
            ("qp.tol", "QP (HPIPM) convergence tolerance", _val(n["qp"]["tol"])),
            ("qp.warm_start", "0 cold, 1 primal, 2 primal and dual; a failed warm solve "
             "retries cold", _val(n["qp"]["warm_start"])),
            ("qp.reg_prim", "primal regularisation", _val(n["qp"]["reg_prim"])),
            ("qp.soft_joint_limits", "joint box as a soft constraint", _val(n["qp"]["soft_joint_limits"])),
            ("qp.soft_torque_limits", "torque rows as a soft constraint", _val(n["qp"]["soft_torque_limits"])),
            ("qp.joint_slack_linear", "linear penalty on joint-box slack",
             _val(n["qp"]["joint_slack_linear"])),
            ("qp.joint_slack_quadratic", "quadratic penalty on joint-box slack",
             _val(n["qp"]["joint_slack_quadratic"])),
            ("qp.relax_box_to_contain_x0", "widen a hard box to the current state "
             "(hard box only)", _val(n["qp"]["relax_box_to_contain_x0"])),
        ]),
        ("OMPC (not tuned)", [
            ("ompc.R", "weight on the task-space force; only Q/R matters", _val(om["R"])),
            ("ompc.damping", r"added to $J H_g^{-1} J^\top$ near a singularity", _val(om["damping"])),
        ]),
        ("SMC (not tuned)", [
            ("smc.pi_bound", r"bound on the model-error ratio in $\rho$", _val(sm["pi_bound"])),
            ("smc.b2", r"second gain of $\rho$'s robustness term", _val(sm["b2"])),
            ("smc.lambda_damping", "damping of the task-space inertia inverse",
             _val(sm["lambda_damping"])),
        ]),
    ], note=r"The tuned settings are in the tuned-settings table. Files: "
            r"\texttt{pat\_arm\_control/config/arm\_common.yaml}, \texttt{nmpc.yaml}, "
            r"\texttt{ompc.yaml}, \texttt{smc.yaml}; the right arm's limits equal the left's.")


def tab_settings_reference() -> None:
    """The reference generator (ee_target_publisher, ee_targets.yaml): the shapes the
    arms follow, live and on the bench (which uses the same library and defaults)."""
    r = _get(_yaml(_src("pat_arm_control", "config", "ee_targets.yaml")),
             "ee_target_publisher", "ros__parameters")
    c, q = r["circle"], r["rectangle"]
    _settings_table("tab_settings_reference", [
        ("Circle", [
            ("circle.center_dx", "centre, ahead of the arm's start EE position",
             _val(c["center_dx"], "m")),
            ("circle.radius", "radius", _val(c["radius"], "m")),
            ("circle.period", r"lap time at 1$\times$ speed (the speed study divides it)",
             _val(c["period"], "s")),
            ("circle.ramp_s", "speed ramp from rest", _val(c["ramp_s"], "s")),
            ("circle.ccw", "direction: counter-clockwise", _val(c["ccw"])),
        ]),
        ("Rectangle", [
            ("rectangle.center_dx", "centre, ahead of the arm's start EE position",
             _val(q["center_dx"], "m")),
            ("rectangle.width", "size along $x$", _val(q["width"], "m")),
            ("rectangle.height", "size along $y$", _val(q["height"], "m")),
            ("rectangle.speed", r"edge speed at 1$\times$ (the speed study multiplies it)",
             _val(q["speed"], "m/s")),
        ]),
        ("Trajectory windows", [
            ("approach_s", "move from the start pose onto the shape", _val(r["approach_s"], "s")),
            ("trajectory_hz", "rate a new window is published", _val(r["trajectory_hz"], "Hz")),
            ("window_s", "length of each window", _val(r["window_s"], "s")),
            ("sample_dt", "spacing of the samples in a window", _val(r["sample_dt"], "s")),
            ("window_lookback_s", "how far each window starts before now",
             _val(r["window_lookback_s"], "s")),
        ]),
    ], note=r"A run is the approach, two laps and 1\,s; scoring starts at 5\,s. File: "
            r"\texttt{pat\_arm\_control/config/ee\_targets.yaml}.")


TABLES: List[Callable[[], None]] = [tab_nominal, tab_step, tab_timing, tab_sensitivity,
                                    tab_observer, tab_tuned, tab_live, tab_live_degradation,
                                    tab_timing_all, tab_conditions, tab_settings_sim,
                                    tab_settings_sensors, tab_settings_estimation,
                                    tab_settings_arm, tab_settings_reference]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=("shipped", "tuned"), default=C.studies.DEFAULT_VARIANT)
    C.set_variant(ap.parse_args().variant)
    print(f"writing to {os.path.join(C.OUT, 'tables')}")
    for t in TABLES:
        t()
