#!/usr/bin/env python3
"""The thesis figures, from the study CSVs that run_study.py writes.

    ~/.venvs/thesis_eval/bin/python tools/thesis_eval/figures.py --all
    ~/.venvs/thesis_eval/bin/python tools/thesis_eval/figures.py step timing

Reads only data/thesis_eval/*/summary.csv and */series/*.csv, so every figure is
regenerated from the recorded runs without running anything. Writes a PDF (vector,
for the thesis) and a PNG (for a quick look) of each to docs/assets/thesis/. A
figure whose data has not been produced yet is skipped with a message.

Sizes are for an A4 thesis: FULL is the text width, HALF a single column.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FULL = 6.3     # inches: text width of an A4 page with 2.5 cm margins
HALF = 3.1

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "figure.dpi": 100,
})


def save(fig, name: str) -> None:
    out = C.ensure_out()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def legend_ctrl(ax, **kw):
    handles = [plt.Line2D([], [], color=C.COLOUR[c], marker=C.MARKER[c], lw=1.1,
                          ms=4, label=C.LABEL[c]) for c in C.CTRL]
    ax.legend(handles=handles, **kw)


NOMINAL_STYLE = dict(color="0.45", ls=(0, (4, 2)), lw=0.8, zorder=0)


def mark_nominal(ax, x: float) -> None:
    """A vertical line where a sweep passes through the controllers' own setting
    (the tuned or shipped configuration every other figure runs)."""
    ax.axvline(x, **NOMINAL_STYLE)


def nominal_handle() -> plt.Line2D:
    return plt.Line2D([], [], label=f"{C.VARIANT} setting", **NOMINAL_STYLE)


def ctrl_handles() -> List[plt.Line2D]:
    return [plt.Line2D([], [], color=C.COLOUR[c], marker=C.MARKER[c], lw=1.1, ms=4,
                       label=C.LABEL[c]) for c in C.CTRL]


def legend_below(fig, handles, ncol: int) -> None:
    """One legend row under the whole figure, clear of every panel's data."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False)


def plain_log_y(ax) -> None:
    """Plain numbers on a log y axis (20, not 2x10^1); between decades only 2, 3
    and 5 are labelled, so the labels do not crowd."""
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "235" else ""))
    ax.tick_params(axis="y", which="minor", labelsize=6)


def skip(name: str, why: str) -> None:
    print(f"  {name}: skipped ({why})")


# ------------------------------------------------------------- safe mode --
# Every figure marks a run that entered safe mode (the NMPC's joint damping after
# a failed solve) with the same black cross, so a number that came from a
# controller that gave up is never read as a tracking result.

SAFE_STYLE = dict(ls="none", marker="x", color="k", ms=5, mew=1.2)
#: IArmController::ControlMode values that are safe mode (Damping, Latched).
SAFE_MODE_MIN = 2


def safe_handle() -> plt.Line2D:
    return plt.Line2D([], [], label="safe mode entered", **SAFE_STYLE)


def failed(rows) -> bool:
    """Did any of these summary rows (one run, or one per seed) enter safe mode?"""
    return bool(len(rows)) and float(rows["safe_ticks"].max()) > 0


def mark_safe_at(ax, xs, ys, flags) -> bool:
    """A cross on each point whose flag is set. Returns whether any was."""
    pts = [(x, y) for x, y, f in zip(xs, ys, flags) if f and np.isfinite(y)]
    if pts:
        ax.plot(*zip(*pts), zorder=6, clip_on=False, **SAFE_STYLE)
    return bool(pts)


def mark_safe_series(ax, s: pd.DataFrame, x, y) -> bool:
    """On a per-tick record: a cross where the controller entered safe mode."""
    if "mode" not in s:
        return False
    m = s["mode"].to_numpy() >= SAFE_MODE_MIN
    idx = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    x, y = np.asarray(x), np.asarray(y)
    if idx.size:
        ax.plot(x[idx], y[idx], zorder=6, clip_on=False, **SAFE_STYLE)
    return bool(idx.size)


# ------------------------------------------------------------------- step --

def fig_step() -> None:
    """Step response: EE error, the largest joint torque, and the base yaw."""
    if C.summary("nominal") is None:
        return skip("step", "no nominal study")
    fig, axes = plt.subplots(3, 2, figsize=(FULL, 4.6), sharex="col")
    safe = False
    for col, (scen, title) in enumerate((("step6", "6 cm diagonal step"),
                                         ("step12", "12 cm step along x"))):
        for c in C.CTRL:
            s = C.series("nominal", f"{scen}_{c}")
            if s is None:
                continue
            t = s["t_s"]
            tau = s[[k for k in s.columns if k.startswith("tau")]].abs().max(axis=1)
            yaw = np.unwrap(s["base_yaw"].to_numpy())
            kw = dict(color=C.COLOUR[c], label=C.LABEL[c])
            for row, y in enumerate((s["err_m"] * 1e3, tau, (yaw - yaw[0]) * 1e3)):
                axes[row, col].plot(t, y, **kw, lw=0.8 if row == 1 else 1.1)
                safe |= mark_safe_series(axes[row, col], s, t, y)
        axes[0, col].set_title(title)
        axes[1, col].axhline(C.TAU_MAX, color="k", ls=":", lw=0.8)
        axes[2, col].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("EE error [mm]")
    axes[1, 0].set_ylabel(r"max$_i\,|\tau_i|$ [N m]")
    axes[2, 0].set_ylabel(r"base yaw $\psi$ [mrad]")
    axes[0, 1].legend(handles=ctrl_handles() + ([safe_handle()] if safe else []),
                      loc="upper right")
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig01_step")


# ------------------------------------------------------------- trajectory --

def _path_deviation(ref: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """[m] distance from each EE point to the nearest point of the reference
    path, wherever along it: how far off the path the EE is, ignoring lag."""
    d2 = ((ee[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


#: [s] one lap of either shape at the default speed (the 6 cm circle's period;
#: the rectangle's 44 cm perimeter at 5.5 cm/s).
LAP_S = 8.0


def _trajectory(scen: str, name: str, title: str, corner_zoom: bool = False) -> None:
    """The last lap's path at true scale, a true-scale zoom on where the paths
    differ most, the error against time, and its distribution.

    The path panels show only where the EE went; an EE exactly on the path but
    behind the reference looks perfect there. That lag is in the error-vs-time
    panel, which is the time-matched error the metrics use."""
    df = C.summary("nominal")
    if df is None:
        return skip(name, "no nominal study")
    fig = plt.figure(figsize=(FULL, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.8], height_ratios=[1.25, 1.0],
                          wspace=0.4, hspace=0.42)
    ax_xy, ax_z, ax_v = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_t = fig.add_subplot(gs[1, :])
    errs, paths = [], []
    ref = None
    safe = False
    for c in C.CTRL:
        s = C.series("nominal", f"{scen}_{c}")
        row = df[df["cell"] == f"{scen}_{c}"]
        if s is None or row.empty:
            continue
        w0 = float(row["window_start_s"].iloc[0])
        win = s[s["t_s"] >= w0]
        lap = s[s["t_s"] >= s["t_s"].iloc[-1] - LAP_S]
        if ref is None:
            ref = lap[["ref_x", "ref_y"]].to_numpy()
        ee = lap[["ee_x", "ee_y"]].to_numpy()
        paths.append((c, ee, _path_deviation(ref, ee)))
        ax_t.plot(s["t_s"], s["err_m"] * 1e3, color=C.COLOUR[c], label=C.LABEL[c])
        safe |= mark_safe_series(ax_t, s, s["t_s"], s["err_m"] * 1e3)
        ax_t.axvline(w0, color="0.6", lw=0.6, ls=":")
        errs.append((c, win["err_m"].to_numpy() * 1e3, failed(row)))
    if ref is None:
        plt.close(fig)
        return skip(name, f"no {scen} series")

    c_w, ee_w, dev_w = max(paths, key=lambda p: p[2].max())
    k = int(dev_w.argmax())
    if corner_zoom:
        # Zoom on the reference corner nearest the worst deviation, sized to how
        # far the paths stray from the reference within 3 cm of that corner (so
        # each controller's corner is in view), between 8 mm and 40 mm.
        lo, hi = ref.min(axis=0), ref.max(axis=0)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
        centre = corners[int(((corners - ee_w[k]) ** 2).sum(axis=1).argmin())]
        dev_near = [dev[np.linalg.norm(ee - centre, axis=1) < 0.03]
                    for _, ee, dev in paths]
        stray = max((d.max() for d in dev_near if d.size), default=0.0)
        half = min(max(1.5 * stray, 8e-3), 0.04)
    else:
        # Zoom on the EE point furthest off the path, over every controller.
        nearest = ref[int(((ref - ee_w[k]) ** 2).sum(axis=1).argmin())]
        centre = (nearest + ee_w[k]) / 2.0
        half = max(1.6 * dev_w[k], 1.5e-3)

    ax_xy.plot(ref[:, 0], ref[:, 1], color="k", ls="--", lw=0.8)
    ax_z.plot((ref[:, 0] - centre[0]) * 1e3, (ref[:, 1] - centre[1]) * 1e3,
              color="k", ls="--", lw=0.9)
    for c, ee, _ in paths:
        ax_xy.plot(ee[:, 0], ee[:, 1], color=C.COLOUR[c], lw=0.8)
        ax_z.plot((ee[:, 0] - centre[0]) * 1e3, (ee[:, 1] - centre[1]) * 1e3,
                  color=C.COLOUR[c], lw=1.0)
    ax_xy.add_patch(plt.Rectangle(centre - half, 2 * half, 2 * half, fill=False,
                                  ec="k", lw=0.7))
    ax_xy.annotate("zoom", (centre[0] - half, centre[1] + half), xytext=(0, 1.5),
                   textcoords="offset points", fontsize=6, ha="left", va="bottom")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("last lap (true scale)")
    ax_z.set_xlim(-half * 1e3, half * 1e3)
    ax_z.set_ylim(-half * 1e3, half * 1e3)
    ax_z.set_aspect("equal")
    ax_z.set_xlabel(r"$x - x_0$ [mm]")
    ax_z.set_ylabel(r"$y - y_0$ [mm]")
    ax_z.set_title("zoom (true scale)")

    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("EE error [mm]")
    ax_t.set_yscale("log")
    ax_t.set_ylim(bottom=0.01)
    ax_t.set_title(f"{title}: error against the reference at the same time", loc="left")
    if errs:
        parts = ax_v.violinplot([e for _, e, _ in errs], showmedians=True, widths=0.8)
        for body, (c, _, _) in zip(parts["bodies"], errs):
            body.set_facecolor(C.COLOUR[c])
            body.set_alpha(0.6)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("0.2")
            parts[key].set_linewidth(0.8)
        safe |= mark_safe_at(ax_v, range(1, len(errs) + 1),
                             [e.max() * 1.08 for _, e, _ in errs], [f for _, _, f in errs])
        ax_v.set_xticks(range(1, len(errs) + 1))
        ax_v.set_xticklabels([C.LABEL[c].split("-")[0] for c, _, _ in errs], rotation=30,
                             ha="right", rotation_mode="anchor")
        ax_v.set_ylabel("EE error in window [mm]")
    handles = [plt.Line2D([], [], color=C.COLOUR[c], lw=1.5, label=C.LABEL[c])
               for c in C.CTRL]
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=0.8, label="reference"))
    if safe:
        handles.append(safe_handle())
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98),
               ncol=len(handles), frameon=False)
    save(fig, name)


def fig_circle() -> None:
    _trajectory("circle", "fig02_circle", "6 cm circle, 8 s lap")


def fig_rectangle() -> None:
    _trajectory("rectangle", "fig03_rectangle", "12 x 10 cm rectangle, 5.5 cm/s",
                corner_zoom=True)


# -------------------------------------------------------------- dashboard --

DASH = [("rmse_mm", "RMSE [mm]"), ("err_var_mm2", "var(e) [mm²]"),
        ("err_peak_mm", "peak error [mm]"), ("tau2_int_win", "∫‖τ‖²dt [N²m²s]"),
        ("tau_var", "var(τ) [N²m²]"), ("chatter_tv", "Σ|Δτ| [N m]"),
        ("rail_frac", "at torque limit [-]"), ("yaw2_int_total", "∫ψ²dt [rad²s]"),
        ("tick_us_mean", "tick [µs]")]


def _fmt(v: float) -> str:
    """Short enough to fit a heatmap cell: at most two significant figures."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 0.01:
        return f"{v:.2g}"
    e = int(math.floor(math.log10(a)))
    return f"{v / 10 ** e:.1f}e{e}"


def fig_dashboard() -> None:
    """Every headline metric, per scenario: colour is how many times worse than
    the best controller on that metric (log scale), the number is the value."""
    df = C.summary("nominal")
    if df is None:
        return skip("dashboard", "no nominal study")
    scen = [("step6", "6 cm step"), ("step12", "12 cm step"),
            ("circle", "circle"), ("rectangle", "rectangle")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.6), sharey=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.06})
    safe = False
    for ax, (s, title) in zip(axes.flat, scen):
        sub = df[df["cell"].str.startswith(s + "_")].copy()
        sub["tag"] = sub["cell"].str.split("_").str[-1]
        sub = C.ordered(sub)
        # metrics down, controllers across
        vals = sub[[k for k, _ in DASH]].to_numpy(dtype=float).T
        # A metric that is exactly 0 for the best controller (no ticks at the
        # limit) still has to colour the others: compare against a floor of 1%
        # of the column's largest value rather than against zero.
        floor = 0.01 * np.nanmax(np.abs(vals), axis=1, keepdims=True)
        best = np.nanmin(vals, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log10((vals + floor) / (best + floor))
        ratio = np.nan_to_num(ratio, nan=0.0)   # a column that is 0 for everyone
        im = ax.imshow(ratio, cmap="YlOrRd", vmin=0, vmax=2.5, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, _fmt(vals[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if ratio[i, j] > 1.6 else "black")
        # A controller that entered safe mode carries a cross on its column.
        bad = [failed(sub[sub["tag"] == t]) for t in sub["tag"]]
        safe |= any(bad)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([C.LABEL[t].replace("-NMPC", "") + (" ×" if b else "")
                            for t, b in zip(sub["tag"], bad)])
        ax.set_yticks(range(len(DASH)))
        ax.set_yticklabels([l for _, l in DASH])
        ax.set_title(title)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("log10(value / best)")
    if safe:
        fig.text(0.02, 0.01, "× safe mode entered", fontsize=7)
    save(fig, "fig04_dashboard")


# ----------------------------------------------------------------- pareto --

def fig_pareto() -> None:
    """Control effort against accuracy, across speeds."""
    df = C.summary("speed")
    if df is None:
        return skip("pareto", "no speed study")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    safe = False
    # The speed is the marker's size (1x smallest), so no labels have to share
    # the crowded low-effort corner.
    size = {1: 3.0, 2: 5.0, 3: 7.0, 4: 9.0}
    for ax, shape in zip(axes, ("circle", "rectangle")):
        sub = df[df["scenario"] == shape]
        for c in C.CTRL:
            d = sub[sub["tag"] == c].sort_values("traj_speed")
            ax.plot(d["tau2_int_win"], d["mse_mm2"], color=C.COLOUR[c], lw=0.8, zorder=2)
            for _, r in d.iterrows():
                ax.plot(r["tau2_int_win"], r["mse_mm2"], color=C.COLOUR[c],
                        marker=C.MARKER[c], ms=size[int(r["traj_speed"])], mec="k",
                        mew=0.3, zorder=3)
            safe |= mark_safe_at(ax, d["tau2_int_win"], d["mse_mm2"], d["safe_ticks"] > 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\int\|\tau\|^2 dt$ in window [N$^2$m$^2$s]")
        ax.set_ylabel("MSE [mm²]")
        ax.set_title(shape)
    speeds = [plt.Line2D([], [], ls="none", marker="o", color="0.6", mec="k", mew=0.3,
                         ms=size[v], label=f"{v}x speed") for v in sorted(size)]
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    fig.legend(handles=ctrl_handles() + ([safe_handle()] if safe else []),
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False)
    fig.legend(handles=speeds, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4,
               frameon=False)
    save(fig, "fig05_pareto")


# ------------------------------------------------------------------- base --

def fig_base() -> None:
    """What the base feels: yaw with the other arm held, mirrored, same-direction."""
    df = C.summary("balanced")
    if df is None:
        return skip("base", "no balanced study")
    fig = plt.figure(figsize=(FULL, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.42)
    titles = {"hold": "partner holds", "mirror": "partner mirrors", "same": "partner same"}
    safe = False
    for j, p in enumerate(("hold", "mirror", "same")):
        ax = fig.add_subplot(gs[0, j])
        tracks = {}
        for c in C.CTRL:
            s = C.series("balanced", f"circle_p-{p}_{c}")
            if s is not None:
                yaw = np.unwrap(s["base_yaw"].to_numpy())
                tracks[c] = (s["t_s"], yaw - yaw[0], s)
        # With a mirrored partner the reactions cancel and what is left is tiny,
        # so a controller whose residual is orders of magnitude larger (the bars
        # below show it) is left out rather than set the scale for the rest.
        hidden = []
        if p == "mirror" and len(tracks) > 2:
            peak = {c: np.abs(y).max() for c, (_, y, _) in tracks.items()}
            for c in tracks:
                rest = np.median([v for k, v in peak.items() if k != c])
                if rest > 0 and peak[c] > 10 * rest:
                    hidden.append(c)
        scale, unit = (1e6, r"$\mu$rad") if p == "mirror" else (1e3, "mrad")
        for c, (t, y, s) in tracks.items():
            if c not in hidden:
                ax.plot(t, y * scale, color=C.COLOUR[c], label=C.LABEL[c])
                safe |= mark_safe_series(ax, s, t, y * scale)
        for c in hidden:
            ax.text(0.03, 0.97, f"{C.LABEL[c]} not shown:\npeak "
                    f"{np.abs(tracks[c][1]).max() * 1e3:.1f} mrad",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6)
        ax.set_title(f"circle, {titles[p]}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(rf"base yaw $\psi$ [{unit}]")
    ax = fig.add_subplot(gs[1, :])
    groups = [(shape, p) for shape in ("circle", "rectangle") for p in ("hold", "mirror", "same")]
    w = 0.2
    for k, c in enumerate(C.CTRL):
        ys, bad = [], []
        for shape, p in groups:
            row = df[df["cell"] == f"{shape}_p-{p}_{c}"]
            ys.append(float(row["yaw2_int_total"].iloc[0]) if not row.empty else np.nan)
            bad.append(failed(row))
        xs = np.arange(len(groups)) + (k - 1.5) * w
        ax.bar(xs, ys, w, color=C.COLOUR[c], label=C.LABEL[c])
        safe |= mark_safe_at(ax, xs, [y * 2.0 for y in ys], bad)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{s}\n{titles[p]}" for s, p in groups])
    ax.set_ylabel(r"$\int\psi^2 dt$ [rad$^2$s]")
    ax.legend(handles=ctrl_handles() + ([safe_handle()] if safe else []),
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    save(fig, "fig06_base_attitude")


# ------------------------------------------------------------------ speed --

def fig_speed() -> None:
    df = C.summary("speed")
    if df is None:
        return skip("speed", "no speed study")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.6), sharex="col")
    for j, shape in enumerate(("circle", "rectangle")):
        sub = df[df["scenario"] == shape]
        for c in C.CTRL:
            d = sub[sub["tag"] == c].sort_values("traj_speed")
            bad = d["safe_ticks"] > 0
            axes[0, j].plot(d["traj_speed"], d["rmse_mm"], color=C.COLOUR[c],
                            marker=C.MARKER[c], ms=4, label=C.LABEL[c])
            mark_safe_at(axes[0, j], d["traj_speed"], d["rmse_mm"], bad)
            axes[1, j].plot(d["traj_speed"], 100 * d["rail_frac"], color=C.COLOUR[c],
                            marker=C.MARKER[c], ms=4)
            mark_safe_at(axes[1, j], d["traj_speed"], 100 * d["rail_frac"], bad)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(shape)
        axes[1, j].set_xlabel("speed factor")
        for ax in axes[:, j]:
            mark_nominal(ax, 1.0)
    axes[0, 0].set_ylabel("RMSE [mm]")
    axes[1, 0].set_ylabel("joint-ticks at limit [%]")
    legend_below(fig, ctrl_handles() + [safe_handle(), nominal_handle()], ncol=6)
    save(fig, "fig07_speed")


# ------------------------------------------------------------- robustness --

def _seeded(ax, df: pd.DataFrame, x: str, y: str = "rmse_mm") -> bool:
    """Mean and 95% band over seeds per controller, a cross where any seed
    entered safe mode. Returns whether any did."""
    safe = False
    for c in C.CTRL:
        d = df[df["tag"] == c]
        pts = sorted(d[x].unique())
        m, h = zip(*(C.ci95(d[d[x] == v][y]) for v in pts)) if pts else ((), ())
        m, h = np.array(m), np.array(h)
        ax.plot(pts, m, color=C.COLOUR[c], marker=C.MARKER[c], ms=3.5, label=C.LABEL[c])
        ax.fill_between(pts, m - h, m + h, color=C.COLOUR[c], alpha=0.2, lw=0)
        safe |= mark_safe_at(ax, pts, m, [failed(d[d[x] == v]) for v in pts])
    return safe


#: The shape the robustness and estimator figures score: the rectangle's corners
#: are where the controllers differ, where the circle's smooth reference
#: flatters them all.
SCORE_SHAPE = "rectangle"


def fig_robustness() -> None:
    def shape_rows(study: str):
        df = C.summary(study)
        if df is None:
            return None
        df = df[df["scenario"] == SCORE_SHAPE]
        return None if df.empty else df

    # (x label, rows, x column, symlog x, the nominal x or None)
    panels = []
    me = shape_rows("model_error")
    if me is not None:
        panels.append(("link mass / model mass", me, "mass_scale", False, 1.0))
    di = shape_rows("disturbance")
    if di is not None:
        panels.append(("unmodelled joint torque [N m]", di, "disturb_torque", False, 0.0))
    tl = shape_rows("torque_limit")
    if tl is not None:
        tl = tl.assign(tau_scale=tl["cell"].str.extract(r"tau([0-9.]+)x")[0].astype(float))
        panels.append(("torque limit / 0.25 N m", tl, "tau_scale", False, 1.0))
    bd = shape_rows("damping")
    if bd is not None:
        panels.append(("base damping [N s/m]", bd, "base_damping", True,
                       float(C.studies.PLANT[1])))
    nz = shape_rows("noise")
    if nz is not None:
        panels.append(("flight sensor error (x profile)", nz, "noise_scale", False, 1.0))
    dl = shape_rows("delay")
    if dl is not None:
        panels.append(("extra state delay [ticks]", dl, "delay_ticks", False, 0.0))
    if not panels:
        return skip("robustness", f"no {SCORE_SHAPE} robustness runs")
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.9))
    safe = False
    for ax, (xl, df, x, symlog, nominal) in zip(axes.flat, panels):
        safe |= _seeded(ax, df, x)
        if nominal is not None:
            mark_nominal(ax, nominal)
        ax.set_yscale("log")
        if symlog:
            # Damping spans 0 to 50: linear near zero, logarithmic above.
            ax.set_xscale("symlog", linthresh=0.05, linscale=0.5)
            ticks = sorted(df[x].unique())
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.set_xlim(left=0)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(xl)
        plain_log_y(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{SCORE_SHAPE} RMSE [mm]")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    handles = ctrl_handles() + [nominal_handle()] + ([safe_handle()] if safe else [])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig08_robustness")


# ------------------------------------------------------------ sensitivity --

KNOB_LABEL = {"weights.Q_pos": "Q_pos", "weights.R": "R", "nmpc.N": "N",
              "nmpc.Ts": "Ts", "qp.max_iter": "QP max iter", "nmpc.sqp_iters": "SQP iters",
              "ompc.Ts": "Ts", "ompc.Q": "Q (vs R)", "smc.epsilon": "ε",
              "smc.lam": "λ", "smc.alpha": "α", "smc.tau_bound": "τ bound"}


def _sensitivity_frame() -> pd.DataFrame:
    df = C.summary("sensitivity")
    if df is None:
        return None
    idx = C.sensitivity_index()
    meta = pd.DataFrame([dict(cell=k, **v) for k, v in idx.items()])
    df = df.merge(meta, on="cell", how="inner")
    nominal = df[df["nominal"]].groupby("ctrl")[["rmse_mm", "tau2_int_win"]].first()
    df["rmse_rel"] = df["rmse_mm"] / df["ctrl"].map(nominal["rmse_mm"])
    df["tau2_rel"] = df["tau2_int_win"] / df["ctrl"].map(nominal["tau2_int_win"])
    return df


def _shades(colour: str, n: int) -> List[tuple]:
    """n shades of one colour, dark to light: the controller's hue is kept, so a
    knob's line still says which controller it belongs to."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    out = []
    span = min(0.5, 0.15 * n)
    for f in np.linspace(-span, span, n) if n > 1 else (0.0,):
        out.append(tuple(rgb * (1 + f) if f < 0 else rgb + (1 - rgb) * f))
    return out


KNOB_MARKERS = ["o", "s", "^", "D", "v", "P"]


def fig_sensitivity() -> None:
    df = _sensitivity_frame()
    if df is None:
        return skip("sensitivity", "no sensitivity study")
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 4.4), sharey="row",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.22})
    # A setting that entered safe mode measures the failure, not the knob, and
    # its error would set the scale for every other curve: it is left out of the
    # curve and marked at the top edge of the panel instead.
    off_scale = []
    for j, c in enumerate(C.CTRL):
        sub = df[df["ctrl"] == c]
        knobs = list(sub.groupby("knob", sort=False))
        shades = _shades(C.COLOUR[c], len(knobs))
        for k, (knob, d) in enumerate(knobs):
            d = d.sort_values("mult")
            bad = d["safe_ticks"] > 0
            off_scale += [(j, m, shades[k]) for m in d["mult"][bad]]
            d = d[~bad]
            kw = dict(color=shades[k], ls="-", lw=0.9, marker=KNOB_MARKERS[k],
                      ms=3.5, mec="k", mew=0.3, label=KNOB_LABEL.get(knob, knob))
            axes[0, j].plot(d["mult"], d["rmse_rel"], **kw)
            axes[1, j].plot(d["mult"], d["tau2_rel"], **kw)
        axes[0, j].set_title(C.LABEL[c])
        axes[1, j].set_xlabel(f"value / {C.VARIANT}")
        axes[1, j].legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                          ncol=2, frameon=False, handlelength=1.3, handletextpad=0.4,
                          columnspacing=0.6, borderaxespad=0.0)
        for i in (0, 1):
            axes[i, j].set_xscale("log")
            axes[i, j].set_yscale("log")
            axes[i, j].axhline(1, color="k", lw=0.6)
            mark_nominal(axes[i, j], 1.0)
            plain_log_y(axes[i, j])
        axes[0, j].tick_params(labelbottom=False)
    axes[0, 0].set_ylabel(f"RMSE / {C.VARIANT}")
    axes[1, 0].set_ylabel(rf"$\int\tau^2$ / {C.VARIANT}")
    for j, m, colour in off_scale:
        for ax in axes[:, j]:
            ax.plot([m], [1], transform=ax.get_xaxis_transform(), ls="none", marker="x",
                    ms=6, mew=1.5, color=colour, clip_on=False, zorder=5)
    handles = [nominal_handle()]
    if off_scale:
        handles.append(plt.Line2D([], [], ls="none", marker="x", ms=6, mew=1.5, color="0.3",
                                  label="safe mode entered (off scale)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.94),
               ncol=len(handles), frameon=False, fontsize=6)
    save(fig, "fig09_sensitivity")


# ----------------------------------------------------------------- timing --

def fig_timing() -> None:
    df = C.summary("timing")
    if df is None:
        return skip("timing", "no timing study")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw={"wspace": 0.3})
    bins = np.logspace(0, 4.3, 120)
    safe = False
    labels = []
    for c in C.CTRL:
        runs = df[(df["tag"] == c) & ~df["cell"].str.contains("_N")]
        ticks = []
        for cell in runs["cell"]:
            s = C.series("timing", cell)
            if s is not None:
                ticks.append(s["tick_us"].to_numpy())
        if not ticks:
            continue
        # A distribution has no point to cross, so a controller whose timing runs
        # entered safe mode carries the cross in its legend entry.
        bad = failed(runs)
        safe |= bad
        labels.append((c, C.LABEL[c] + (" ×" if bad else "")))
        t = np.concatenate(ticks)
        axes[0].hist(t, bins=bins, histtype="step", color=C.COLOUR[c], density=True)
    axes[0].set_xscale("log")
    axes[0].axvline(C.PERIOD_US, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("control tick [µs]")
    axes[0].set_ylabel("density")
    axes[0].set_yscale("log")
    axes[0].legend(handles=[plt.Line2D([], [], color=C.COLOUR[c], marker=C.MARKER[c],
                                       lw=1.1, ms=4, label=l) for c, l in labels],
                   loc="upper center", fontsize=6)

    hz = df[df["cell"].str.contains("_N")].copy()
    hz["N"] = hz["cell"].str.extract(r"_N([0-9]+)_")[0].astype(int)
    for c in ("sqp", "iter"):
        d = hz[hz["tag"] == c]
        pts = sorted(d["N"].unique())
        for col, ls in (("tick_us_mean", "-"), ("tick_us_p99", "--")):
            m, h = zip(*(C.ci95(d[d["N"] == n][col]) for n in pts))
            axes[1].errorbar(pts, m, yerr=h, color=C.COLOUR[c], ls=ls, marker=C.MARKER[c],
                             ms=3.5, capsize=2,
                             label=f"{C.LABEL[c]} {'mean' if ls == '-' else 'p99'}")
            safe |= mark_safe_at(axes[1], pts, m, [failed(d[d["N"] == n]) for n in pts])
    # Each NMPC's own horizon, in its colour (30 as shipped; tuned per controller).
    for label, _, args in C.studies.active_controllers():
        if label in ("sqp", "iter"):
            sets = dict(a.split("=", 1) for a in args[1::2])
            axes[1].axvline(float(sets.get("nmpc.N", 30)),
                            **{**NOMINAL_STYLE, "color": C.COLOUR[label]})
    axes[1].set_xlabel("horizon N")
    axes[1].set_ylabel("tick [µs]")
    handles, _ = axes[1].get_legend_handles_labels()
    own_n = nominal_handle()
    own_n.set_label(f"{C.VARIANT} N")
    axes[1].legend(handles=handles + [own_n] + ([safe_handle()] if safe else []),
                   fontsize=5.5)
    save(fig, "fig10_timing")


# ------------------------------------------------------------- constraint --

def _box_violation(s: pd.DataFrame, q_min, q_max) -> np.ndarray:
    """[rad] how far the worst joint is outside its configured box, per tick."""
    out = np.zeros(len(s))
    for j, (lo, hi) in enumerate(zip(q_min, q_max)):
        q = s[f"q{j}"].to_numpy()
        out = np.maximum(out, np.maximum(lo - q, q - hi))
    return np.maximum(out, 0.0)


def fig_constraint() -> None:
    """The configured joint box under stress: a reach that the box forbids, a
    joint parked outside it, and the reach swept in length."""
    df = C.summary("constraint")
    if df is None:
        return skip("constraint", "no constraint study")
    park_max = list(C.studies.Q_MAX)
    park_max[1] = C.studies.PARK_Q3_MAX
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.9),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.38})
    for row, (prefix, q_max, title) in enumerate((
            ("reach_dy0.1", C.studies.Q_MAX, "reach 10 cm (+y)"),
            ("parked", park_max, "joint3 parked 0.088 rad outside"))):
        for c in C.CTRL:
            s = C.series("constraint", f"{prefix}_{c}")
            if s is None:
                continue
            kw = dict(color=C.COLOUR[c], label=C.LABEL[c])
            viol = _box_violation(s, C.studies.Q_MIN, q_max)
            axes[row, 0].plot(s["t_s"], viol * 1e3, **kw)
            axes[row, 1].plot(s["t_s"], s["err_m"] * 1e3, **kw)
            mark_safe_series(axes[row, 0], s, s["t_s"], viol * 1e3)
            mark_safe_series(axes[row, 1], s, s["t_s"], s["err_m"] * 1e3)
        axes[row, 0].set_ylabel("outside box [mrad]")
        axes[row, 1].set_ylabel("EE error [mm]")
        axes[row, 0].set_title(title, loc="left")
    for ax in axes[:, :2].flat:
        ax.set_xlabel("time [s]")

    reach = df[df["cell"].str.startswith("reach_")].copy()
    reach["dy_cm"] = reach["cell"].str.extract(r"_dy([0-9.]+)_")[0].astype(float) * 100
    for c in C.CTRL:
        d = reach[reach["tag"] == c].sort_values("dy_cm")
        for ax, col, scale in ((axes[0, 2], "step_sse_mm", 1.0),
                               (axes[1, 2], "qlim_peak_rad", 1e3)):
            ax.plot(d["dy_cm"], d[col] * scale, color=C.COLOUR[c], marker=C.MARKER[c],
                    ms=3.5)
            mark_safe_at(ax, d["dy_cm"], d[col] * scale, d["safe_ticks"] > 0)
    axes[0, 2].set_ylabel("final EE error [mm]")
    axes[1, 2].set_ylabel("peak outside box [mrad]")
    axes[0, 2].set_title("reach swept", loc="left")
    for ax in axes[:, 2]:
        ax.set_xlabel("reach [cm]")
    legend_ctrl(axes[0, 0], loc="upper left", fontsize=6)
    axes[1, 2].legend(handles=[safe_handle()], loc="upper left", fontsize=6)
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig11_constraint")


# --------------------------------------------------------------- observer --

def _observer_frame() -> pd.DataFrame:
    df = C.summary("observer")
    if df is None:
        return None
    parts = df["cell"].map(C.observer_parts)
    df["obs_scen"] = parts.str[0]
    df["src"] = parts.str[1]
    df["ff"] = parts.str[2]
    return df


OBS_SCEN = [("circle", "circle"), ("step6", "6 cm step"), ("circle-mirror", "mirror"),
            ("circle-tq", "+0.05 N m"), ("circle-m2", "2x mass")]
#: The observer study's state sources: exact, the raw flight sensors, and the
#: full flight estimation (EKF + joint Kalman filters).
SRC = [("exact", "exact"), ("raw", "raw sensors"), ("ekf-jkf", "EKF + joint KF")]


def fig_observer() -> None:
    """RMSE of each (state source, feedforward gain) relative to exact state
    without feedforward, per controller and scenario. Mean over seeds."""
    df = _observer_frame()
    if df is None:
        return skip("observer", "no observer study")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.4), sharey=True, sharex=True,
                             gridspec_kw={"hspace": 0.25, "wspace": 0.08})
    rows = [(s, ff) for s, _ in SRC for ff in (0.0, 0.5, 1.0)]
    safe = False
    for ax, c in zip(axes.flat, C.CTRL):
        d = df[df["tag"] == c]
        M = np.full((len(rows), len(OBS_SCEN)), np.nan)
        bad = np.zeros(M.shape, dtype=bool)
        for j, (scen, _) in enumerate(OBS_SCEN):
            base = d[(d["obs_scen"] == scen) & (d["src"] == "exact") & (d["ff"] == 0.0)]
            if base.empty:
                continue
            b = base["rmse_mm"].mean()
            for i, (src, ff) in enumerate(rows):
                cell = d[(d["obs_scen"] == scen) & (d["src"] == src) & (d["ff"] == ff)]
                M[i, j] = cell["rmse_mm"].mean() / b
                bad[i, j] = failed(cell)
        im = ax.imshow(np.log2(M), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center", fontsize=6,
                            color="white" if abs(math.log2(M[i, j])) > 2 else "black")
                if bad[i, j]:
                    # A cell where any seed entered safe mode: a cross in its corner.
                    ax.plot(j + 0.36, i - 0.28, marker="x", color="k", ms=3.5, mew=1.0)
        safe |= bool(bad.any())
        ax.set_xticks(range(len(OBS_SCEN)))
        ax.set_xticklabels([l for _, l in OBS_SCEN], rotation=30, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{dict(SRC)[s]}, ff {ff:g}" for s, ff in rows])
        ax.set_title(C.LABEL[c])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("RMSE / RMSE(exact, ff 0)   [log2 colour]")
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cb.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    if safe:
        fig.legend(handles=[safe_handle()], loc="upper center", bbox_to_anchor=(0.5, 0.02),
                   frameon=False)
    save(fig, "fig12_observer")


def fig_ffgain() -> None:
    """Does acting on the disturbance estimate help, with each state source?

    One row per disturbance, one column per state source, so each panel holds only
    the four controllers: RMSE against the feedforward gain, on one scale across
    the row. The last column is how often the feedforward hit its cap at gain 1,
    the price of acting on a noisy estimate."""
    df = _observer_frame()
    if df is None:
        return skip("ffgain", "no observer study")
    scen = [("circle-tq", "unmodelled\n0.05 N m torque"), ("circle-m2", "links\n2x heavier")]
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.9], "wspace": 0.12,
                                          "hspace": 0.18})
    safe = False
    for i, (s, title) in enumerate(scen):
        lo, hi = math.inf, 0.0             # the row's one RMSE scale
        for j, (src, src_label) in enumerate(SRC):
            ax = axes[i, j]
            for c in C.CTRL:
                d = df[(df["obs_scen"] == s) & (df["tag"] == c) & (df["src"] == src)]
                ffs = sorted(d["ff"].unique())
                if not ffs:
                    continue
                m, h = (np.array(v) for v in zip(*(C.ci95(d[d["ff"] == f]["rmse_mm"])
                                                   for f in ffs)))
                ax.errorbar(ffs, m, yerr=h, color=C.COLOUR[c], marker=C.MARKER[c], ms=3.5,
                            capsize=2)
                safe |= mark_safe_at(ax, ffs, m, [failed(d[d["ff"] == f]) for f in ffs])
                lo = min(lo, float(np.min(np.maximum(m - h, m * 0.5))))
                hi = max(hi, float(np.max(m + h)))
            ax.set_yscale("log")
            mark_nominal(ax, 0.0)          # observer.ff_gain ships at 0
            ax.set_xticks([0.0, 0.5, 1.0])
            if i == 0:
                ax.set_title(src_label)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("feedforward gain")
            if j:
                ax.tick_params(labelleft=False)
        for j, ax in enumerate(axes[i, :3]):
            ax.set_ylim(lo * 0.85, hi * 1.15)
            plain_log_y(ax)
            if j:
                ax.tick_params(axis="y", which="both", labelleft=False)
        axes[i, 0].set_ylabel(f"{title}\nRMSE [mm]")

        # How often the feedforward was capped at gain 1, per source.
        ax = axes[i, 3]
        w = 0.2
        for k, c in enumerate(C.CTRL):
            vals = []
            for src, _ in SRC:
                d = df[(df["obs_scen"] == s) & (df["tag"] == c) & (df["src"] == src) &
                       (df["ff"] == 1.0)]
                vals.append(100 * C.ci95(d["ff_saturated"] / d["n_ticks"])[0])
            ax.bar(np.arange(len(SRC)) + (k - 1.5) * w, vals, w, color=C.COLOUR[c])
        ax.set_xticks(range(len(SRC)))
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("feedforward capped\nat gain 1 [% ticks]")
        if i == 0:
            ax.set_title("cap at gain 1")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels([l.replace(" + ", "+\n").replace(" sensors", "")
                                for _, l in SRC], fontsize=6)
    extra = [safe_handle()] if safe else []
    legend_below(fig, ctrl_handles() + [nominal_handle()] + extra, ncol=6)
    save(fig, "fig13_ffgain")


#: The estimator study's sources, in the order the figure shows them.
EST_SRC = [("exact", "exact"), ("raw", "raw"), ("ekf", "EKF"), ("ekf-jkf", "EKF +\njoint KF")]
#: The base sources' own colours: raw light grey, EKF black.
EST_SHADE = {"raw": "0.65", "ekf": "0.1"}


def fig_estimator() -> None:
    """What each state source gives the arm, on the flight sensors.

    The top panel is the base x estimate's error against time for one seed: raw
    jumps with every 2 Hz fix and lags it by 0.5 s, the EKF is smooth and on time
    but drifts slowly. Below: each source's own error, what it costs each
    controller's tracking, and what the joint filter buys, which is a smoother
    command."""
    df = C.summary("estimator")
    if df is None:
        return skip("estimator", "no estimator study")
    df = df.assign(src=df["cell"].str.split("_").str[1])
    d = df[df["scenario"] == SCORE_SHAPE]
    fig = plt.figure(figsize=(FULL, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.42)
    ax_t = fig.add_subplot(gs[0, :])
    ax_e, ax_r, ax_u = (fig.add_subplot(gs[1, i]) for i in range(3))

    safe = False
    labels = {"raw": "raw: last 2 Hz fix (0.5 s old, held)",
              "ekf": "EKF: 500 Hz IMU, corrected at each fix"}
    for src in ("raw", "ekf"):
        s = C.series("estimator", f"{SCORE_SHAPE}_{src}_sqp_s1")
        if s is None:
            continue
        e = s["est_x"].to_numpy() * 1e3
        ax_t.plot(s["t_s"], e, color=EST_SHADE[src], lw=0.7 if src == "raw" else 1.1,
                  label=labels[src])
        if src == "raw":
            # A dot where each new fix arrives: the raw estimate changes only then
            # (between fixes it moves only with the base under the held value).
            new = np.flatnonzero(np.abs(np.diff(e)) > 0.5) + 1
            ax_t.plot(s["t_s"].to_numpy()[new], e[new], ls="none", marker="o", ms=2.2,
                      color=EST_SHADE[src], label="new fix arrives")
        safe |= mark_safe_series(ax_t, s, s["t_s"], e)
    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("base x error [mm]")
    ax_t.set_title(f"error in the base state given to {C.LABEL['sqp']} "
                   f"({SCORE_SHAPE}, one seed)", loc="left")
    ax_t.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=6,
                frameon=False)

    metrics = [("est_pos_rms_mm", "pos.\nmm"), ("est_yaw_rms_mrad", "yaw\nmrad"),
               ("est_vel_rms_mm_s", "vel.\nmm/s"), ("est_wz_rms_mrad_s", "rate\nmrad/s")]
    w = 0.38
    for k, src in enumerate(("raw", "ekf")):
        m, h = zip(*(C.ci95(d[d["src"] == src][mk]) for mk, _ in metrics))
        ax_e.bar(np.arange(len(metrics)) + (k - 0.5) * w, m, w, yerr=h,
                 color=EST_SHADE[src], label=dict(EST_SRC)[src], capsize=1.5)
    ax_e.set_xticks(range(len(metrics)))
    ax_e.set_xticklabels([l for _, l in metrics], fontsize=6)
    ax_e.set_ylabel("base state error, rms")
    ax_e.set_yscale("log")
    plain_log_y(ax_e)
    ax_e.legend(fontsize=5.5)

    # Tracking and command smoothness per controller and source; a cross marks
    # a configuration that entered safe mode in any seed.
    for ax, col, scale, ylab, log in ((ax_r, "rmse_mm", 1.0, f"{SCORE_SHAPE} EE RMSE [mm]",
                                       True),
                                      (ax_u, "tau_step_rms", 1e3, r"rms $\Delta\tau$ per tick "
                                       "[mN m]", False)):
        for k, c in enumerate(C.CTRL):
            x = np.arange(len(EST_SRC)) + (k - 1.5) * 0.2
            stats = [C.ci95(d[(d["tag"] == c) & (d["src"] == s)][col] * scale)
                     for s, _ in EST_SRC]
            m, h = (np.array(v) for v in zip(*stats))
            ax.bar(x, m, 0.2, yerr=h, color=C.COLOUR[c], capsize=1.5)
            bad = [failed(d[(d["tag"] == c) & (d["src"] == s)]) for s, _ in EST_SRC]
            safe |= mark_safe_at(ax, x, m * (1.25 if log else 1.06) + (0 if log else 0.5),
                                 bad)
        ax.set_xticks(range(len(EST_SRC)))
        ax.set_xticklabels([l for _, l in EST_SRC], fontsize=6, rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel(ylab)
        if log:
            ax.set_yscale("log")
            plain_log_y(ax)
    legend_below(fig, ctrl_handles() + ([safe_handle()] if safe else []), ncol=5)
    save(fig, "fig14_estimator")


def fig_warmup() -> None:
    """How long the base estimator must run before the arm task: tracking error
    against the time it had, per controller, raw sensors (nothing to learn, the
    flat reference) against the EKF."""
    df = C.summary("warmup")
    if df is None:
        return skip("warmup", "no warmup study")
    df = df.assign(src=df["cell"].str.split("_").str[1].str.split("-w").str[0])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6),
                             gridspec_kw={"wspace": 0.3})
    ls = {"raw": "--", "ekf": "-"}
    safe = False
    for src in ("raw", "ekf"):
        for c in C.CTRL:
            d = df[(df["tag"] == c) & (df["src"] == src)]
            pts = sorted(d["nav_warmup"].unique())
            m, h = (np.array(v) for v in zip(*(C.ci95(d[d["nav_warmup"] == w]["rmse_mm"])
                                               for w in pts)))
            axes[0].plot(pts, m, color=C.COLOUR[c], ls=ls[src], marker=C.MARKER[c], ms=3)
            axes[0].fill_between(pts, m - h, m + h, color=C.COLOUR[c], alpha=0.12, lw=0)
            safe |= mark_safe_at(axes[0], pts, m,
                                 [failed(d[d["nav_warmup"] == w]) for w in pts])
        d = df[df["src"] == src]
        pts = sorted(d["nav_warmup"].unique())
        m, h = (np.array(v) for v in zip(*(C.ci95(d[d["nav_warmup"] == w]["est_pos_rms_mm"])
                                           for w in pts)))
        axes[1].errorbar(pts, m, yerr=h, color=EST_SHADE[src], ls=ls[src], marker="o", ms=3,
                         capsize=2, label=dict(EST_SRC)[src])
    for ax in axes:
        ax.set_xlabel("estimator running before the task [s]")
        mark_nominal(ax, float(C.studies.NAV_WARMUP))
    axes[0].set_yscale("log")
    plain_log_y(axes[0])
    axes[0].set_ylabel(f"{SCORE_SHAPE} EE RMSE [mm]")
    axes[1].set_ylabel("base position error, rms [mm]")
    axes[1].legend(fontsize=6)
    srcs = [plt.Line2D([], [], color="0.3", ls=ls[s], label=dict(EST_SRC)[s])
            for s in ("raw", "ekf")]
    extra = [safe_handle()] if safe else []
    legend_below(fig, ctrl_handles() + srcs + [nominal_handle()] + extra, ncol=5)
    save(fig, "fig16_warmup")


# --------------------------------------------------------- shipped vs tuned --

# (study, cell prefix or filter, label): the tuning scenarios first, then the test
# set the tuned settings were not fitted to.
TUNING_SET = [("nominal", lambda d: d["cell"].str.startswith("circle_"), "circle"),
              ("nominal", lambda d: d["cell"].str.startswith("rectangle_"), "rectangle"),
              ("nominal", lambda d: d["cell"].str.startswith("step6_"), "6 cm step"),
              ("speed", lambda d: (d["scenario"] == "circle") & (d["traj_speed"] == 2),
               "circle 2x"),
              ("speed", lambda d: (d["scenario"] == "rectangle") & (d["traj_speed"] == 2),
               "rect. 2x")]
TEST_SET = [
    ("nominal", lambda d: d["cell"].str.startswith("step12_"), "12 cm step"),
    ("speed", lambda d: (d["scenario"] == "circle") & (d["traj_speed"] == 3), "circle 3x"),
    ("speed", lambda d: (d["scenario"] == "rectangle") & (d["traj_speed"] == 3), "rect. 3x"),
    ("balanced", lambda d: d["cell"].str.startswith("circle_p-mirror_"), "mirror"),
    ("disturbance", lambda d: (d["scenario"] == "circle") & (d["disturb_torque"] == 0.05),
     "+0.05 N m"),
    ("model_error", lambda d: (d["scenario"] == "circle") & (d["mass_scale"] == 2.0), "2x mass"),
    ("noise", lambda d: (d["scenario"] == "circle") & (d["noise_scale"] == 1),
     "flight sensing"),
]


def fig_tuning() -> None:
    """Shipped against tuned, per controller: the tuning scenarios, then the
    scenarios the tuning never saw. Mean over seeds where there are seeds."""
    rows = []
    for (study, pick, label), kind in ([(t, "tuning") for t in TUNING_SET] +
                                       [(t, "test") for t in TEST_SET]):
        for variant in ("shipped", "tuned"):
            d = C.summary_of(variant, study)
            if d is None:
                continue
            d = d[pick(d)]
            for c in C.CTRL:
                dc = d[d["tag"] == c]
                if dc.empty:
                    continue
                rows.append(dict(scenario=label, kind=kind, variant=variant, ctrl=c,
                                 rmse=dc["rmse_mm"].mean(), tau2=dc["tau2_int_win"].mean(),
                                 rail=dc["rail_frac"].mean(), safe=failed(dc)))
    df = pd.DataFrame(rows)
    if df.empty or (df["variant"] == "tuned").sum() == 0:
        return skip("tuning", "no tuned studies yet")
    scen = list(dict.fromkeys(df["scenario"]))
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 4.4), sharex=True)
    w = 0.2
    safe = False
    for k, c in enumerate(C.CTRL):
        for ax, metric, ylab in ((axes[0], "rmse", "RMSE [mm]"),
                                 (axes[1], "tau2", r"$\int\|\tau\|^2dt$")):
            for variant, alpha, hatch in (("shipped", 0.35, None), ("tuned", 1.0, None)):
                d = df[(df["ctrl"] == c) & (df["variant"] == variant)].set_index("scenario")
                ys = [d[metric].get(s, np.nan) for s in scen]
                bad = [bool(d["safe"].get(s, False)) for s in scen]
                xs = np.arange(len(scen)) + (k - 1.5) * w
                if variant == "shipped":
                    ax.scatter(xs, ys, marker="_", s=60, color=C.COLOUR[c], zorder=3)
                    safe |= mark_safe_at(ax, xs, ys, bad)
                else:
                    ax.bar(xs, ys, w, color=C.COLOUR[c], alpha=alpha,
                           label=C.LABEL[c] if metric == "rmse" else None)
                    safe |= mark_safe_at(ax, xs, [y * 1.35 for y in ys], bad)
            ax.set_ylabel(ylab)
            ax.set_yscale("log")
    n_tune = len([s for s in scen if s in {t[2] for t in TUNING_SET}])
    for ax in axes:
        ax.axvline(n_tune - 0.5, color="k", lw=0.8)
    axes[0].text(n_tune / 2 - 0.5, 1.02, "tuned on", transform=axes[0].get_xaxis_transform(),
                 ha="center", fontsize=7)
    axes[0].text((n_tune + len(scen)) / 2 - 0.5, 1.02, "tested on",
                 transform=axes[0].get_xaxis_transform(), ha="center", fontsize=7)
    axes[1].set_xticks(range(len(scen)))
    axes[1].set_xticklabels(scen, rotation=30, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.COLOUR[c], label=f"{C.LABEL[c]} tuned")
               for c in C.CTRL]
    handles.append(plt.Line2D([], [], ls="none", marker="_", ms=9, mew=1.5, color="0.3",
                              label="shipped"))
    if safe:
        handles.append(safe_handle())
    legend_below(fig, handles, ncol=6 if safe else 5)
    out_dir = C.OUT
    C.OUT = C.OUT_ROOT       # the comparison belongs with the thesis's (tuned) figures
    save(fig, "fig15_shipped_vs_tuned")
    C.OUT = out_dir


FIGURES: Dict[str, Callable[[], None]] = {
    "step": fig_step, "circle": fig_circle, "rectangle": fig_rectangle,
    "dashboard": fig_dashboard, "pareto": fig_pareto, "base": fig_base,
    "speed": fig_speed, "robustness": fig_robustness, "sensitivity": fig_sensitivity,
    "timing": fig_timing, "constraint": fig_constraint,
    "observer": fig_observer, "ffgain": fig_ffgain, "warmup": fig_warmup,
    "estimator": fig_estimator, "tuning": fig_tuning,
}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("figures", nargs="*", help=f"any of {list(FIGURES)}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", choices=("shipped", "tuned"),
                    default=C.studies.DEFAULT_VARIANT,
                    help="which studies to draw: the tuned settings (docs/assets/"
                         "thesis/, the default) or the shipped ones (docs/assets/"
                         "thesis/shipped/)")
    args = ap.parse_args(argv)
    C.set_variant(args.variant)
    names = list(FIGURES) if args.all or not args.figures else args.figures
    bad = [n for n in names if n not in FIGURES]
    if bad:
        sys.exit(f"unknown figures {bad}; choose from {list(FIGURES)}")
    print(f"writing to {C.OUT}")
    for n in names:
        FIGURES[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
