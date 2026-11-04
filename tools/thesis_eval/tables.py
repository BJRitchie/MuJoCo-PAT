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


def _set_args(args: Sequence[str]) -> dict:
    """{key: value} from a `--set key=value ...` argument list."""
    return dict(a.split("=", 1) for a in args[1::2])


def _yaml_scalar(path: str, section: str, key: str) -> str:
    """One `key:` under `section:` in a law's YAML, without a YAML library."""
    import re
    text = open(path, encoding="utf-8").read()
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
             r" & mean & p99 & worst & headroom (worst) & missed deadlines & trials \\",
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
             r" & \multicolumn{3}{c}{control tick [\textmu s]} & worst / & ticks \\",
             r"\cmidrule(lr){2-4}",
             r" & mean & p99 & worst & period & over 10\,ms \\"]
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


TABLES: List[Callable[[], None]] = [tab_nominal, tab_step, tab_timing, tab_sensitivity,
                                    tab_observer, tab_tuned, tab_live, tab_live_degradation,
                                    tab_timing_all, tab_conditions]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=("shipped", "tuned"), default=C.studies.DEFAULT_VARIANT)
    C.set_variant(ap.parse_args().variant)
    print(f"writing to {os.path.join(C.OUT, 'tables')}")
    for t in TABLES:
        t()
