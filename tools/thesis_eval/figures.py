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

# The house style, after the AMME/ACFR formatting guide (formatting_tips-tricks,
# python/format_figures.py): a serif face to sit with the thesis text (STIX, which
# ships with matplotlib, so every machine renders the same glyphs, Unicode
# included), the Okabe-Ito colour cycle, thin axes, a fine light grid on every
# graph, legends kept off the data with a white edge, vector PDF for the thesis.
# Captions carry the description; panels carry (a), (b), ... and only the short
# titles that say which condition a panel is.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                                                "#56B4E9", "#D55E00", "#F0E442", "#000000"]),
    "axes.grid": True, "axes.grid.which": "major", "grid.alpha": 0.5, "grid.linewidth": 0.3,
    "grid.color": "0.7",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.8, "legend.edgecolor": "white", "legend.fancybox": False,
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
    label_panels(axes)
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


def _zoom_box(ref: np.ndarray, paths, corner_zoom: bool):
    """(centre, half-width) [m] of a true-scale zoom on where the paths
    [(tag, ee, deviation)] differ most from the reference."""
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
    return centre, half


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

    centre, half = _zoom_box(ref, paths, corner_zoom)

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
    label_panels([ax_xy, ax_z, ax_v, ax_t])
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
        ("tick_us_mean", "tick [µs]"), ("tick_us_max", "worst tick [µs]")]


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
    label_panels(axes)
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
    label_panels(axes)
    save(fig, "fig05_pareto")


# ------------------------------------------------------------------- base --

def fig_base() -> None:
    """What the base feels: yaw with the other arm held, mirrored, same-direction."""
    df = C.summary("balanced")
    if df is None:
        return skip("base", "no balanced study")
    fig = plt.figure(figsize=(FULL, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.75, wspace=0.42)
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
        panel_label(ax, f"({'abc'[j]})")
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.6)
    ax = fig.add_subplot(bottom[0])
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
    ax.set_xticklabels([p for _, p in groups], rotation=90)
    # A bracket under each shape's three groups, below the rotated tick labels.
    import matplotlib.transforms as mtrans
    under = mtrans.blended_transform_factory(ax.transData, ax.transAxes) + \
        mtrans.ScaledTranslation(0, -33 / 72, fig.dpi_scale_trans)
    for i, shape in enumerate(("circle", "rectangle")):
        x0, x1 = 3 * i - 0.4, 3 * i + 2.4
        ax.plot([x0, x0, x1, x1], [0.035, 0, 0, 0.035], color="0.25", lw=0.7,
                transform=under, clip_on=False)
        ax.annotate(shape, xy=(1 + 3 * i, 0), xycoords=under, xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top")
    ax.axvline(2.5, color="0.6", lw=0.5)
    ax.set_ylabel(r"$\int\psi^2 dt$ [rad$^2$s]")
    panel_label(ax, "(d)")
    fig.legend(handles=ctrl_handles() + ([safe_handle()] if safe else []),
               loc="center", bbox_to_anchor=(0.5, 0.51), ncol=5, frameon=False)
    fig.legend(handles=[plt.Line2D([], [], color=C.COLOUR["sqp"], ls=ls, lw=1.0,
                                   label=f"partner {p}") for p, ls in PARTNER_LS.items()],
               loc="center", bbox_to_anchor=(0.5, 0.475), ncol=3, frameon=False,
               handlelength=2.2)
    _base_reaction(fig, bottom[1], bottom[2])
    save(fig, "fig06_base_attitude")


#: The partner cases in the reaction panels, told apart by line style.
PARTNER_LS = {"hold": "-", "same": "--", "mirror": ":"}


def _base_reaction(fig, spec_tau, spec_mom) -> None:
    """The arm's reaction on the bus, in the units an actuator is rated in: the
    torque it must cancel [N m] and the momentum it must absorb [N m s], for
    SQP-NMPC on the circle. The table is frictionless and nothing else acts on
    the bus, so the torque is the tick-mean d(bus_lz)/dt, exactly (impulse over
    the 10 ms tick); bus_lz is about the chaser's centre of mass, whose total
    momentum is conserved."""
    ax_t, ax_m = fig.add_subplot(spec_tau), fig.add_subplot(spec_mom)
    peaks = {}
    for p, ls in PARTNER_LS.items():
        s = C.series("balanced", f"circle_p-{p}_sqp")
        if s is None or "bus_lz" not in s:
            continue
        t, lz = s["t_s"].to_numpy(), s["bus_lz"].to_numpy()
        tau = np.diff(lz) / np.diff(t)
        tm = 0.5 * (t[1:] + t[:-1])
        kw = dict(color=C.COLOUR["sqp"], ls=ls, lw=1.0 if p != "mirror" else 1.2)
        ax_t.plot(tm, tau, **kw)
        ax_m.plot(t, lz, **kw)
        peaks[p] = (np.abs(tau).max(), np.abs(lz).max())
    if not peaks:
        for ax in (ax_t, ax_m):
            ax.text(0.5, 0.5, "no momentum series", transform=ax.transAxes,
                    ha="center", va="center")
        return
    for ax, k, unit, label in ((ax_t, 0, r"N$\,$m", r"reaction torque $\tau_z$ [N$\,$m]"),
                               (ax_m, 1, r"N$\,$m$\,$s", r"bus momentum $L_z$ [N$\,$m$\,$s]")):
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(label)
        top = max(v[k] for v in peaks.values())
        ax.set_ylim(-1.25 * top, 2.0 * top)
        ax.text(0.03, 0.97, "peak [{}]:  ".format(unit)
                + ",  ".join(f"{p} {v[k]:.2g}" for p, v in peaks.items() if p != "mirror")
                + (f"\nmirror {peaks['mirror'][k]:.1e}" if "mirror" in peaks else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax_t.set_title("circle, SQP-NMPC", fontsize=7)
    ax_m.set_title("circle, SQP-NMPC", fontsize=7)
    panel_label(ax_t, "(e)")
    panel_label(ax_m, "(f)")


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
    label_panels(axes)
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
    label_panels(list(axes.flat)[:len(panels)])
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
    label_panels(axes)
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
        # The worst tick recorded, which is what a realtime deadline is judged by:
        # a histogram's tail is too thin to see at one sample in 10^4.
        axes[0].plot(t.max(), 1.0, marker="v", color=C.COLOUR[c], ms=4, ls="none",
                     transform=axes[0].get_xaxis_transform(), clip_on=False)
    axes[0].set_xscale("log")
    axes[0].axvline(C.PERIOD_US, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("control tick [µs] (▼ worst recorded)")
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
        for col, ls, name in (("tick_us_mean", "-", "mean"), ("tick_us_p99", "--", "p99")):
            m, h = zip(*(C.ci95(d[d["N"] == n][col]) for n in pts))
            axes[1].errorbar(pts, m, yerr=h, color=C.COLOUR[c], ls=ls, marker=C.MARKER[c],
                             ms=3.5, capsize=2, label=f"{C.LABEL[c]} {name}")
            safe |= mark_safe_at(axes[1], pts, m, [failed(d[d["N"] == n]) for n in pts])
        # The worst case is the maximum over every run at that N, not a mean of
        # maxima: it is the single tick a deadline has to survive.
        worst = [d[d["N"] == n]["tick_us_max"].max() for n in pts]
        axes[1].plot(pts, worst, color=C.COLOUR[c], ls=":", marker=C.MARKER[c], ms=3.5,
                     mfc="none", label=f"{C.LABEL[c]} max")
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
    label_panels(axes)
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
    label_panels(axes)
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
    label_panels(axes)
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
    label_panels(axes)
    save(fig, "fig13_ffgain")


#: The estimator study's sources, in the order the figure shows them.
EST_SRC = [("exact", "exact"), ("raw", "raw"), ("ekf", "EKF"), ("ekf-jkf", "EKF +\njoint KF")]
#: The base sources' own colours: raw light grey, EKF black.
EST_SHADE = {"raw": "0.65", "ekf": "0.1"}


def fig_estimator() -> None:
    """What each state source gives the arm, on the flight sensors.

    The top panels are the base state itself against time for one seed (x and y
    on the left, yaw on the right, each measured from where the base started):
    the plant's truth, the raw sensors (the last 5 Hz fix, 0.15 s old, held) and
    the EKF. Below: each source's own error, what it costs each controller's
    tracking, and what the joint filter buys, which is a smoother command."""
    df = C.summary("estimator")
    if df is None:
        return skip("estimator", "no estimator study")
    df = df.assign(src=df["cell"].str.split("_").str[1])
    d = df[df["scenario"] == SCORE_SHAPE]
    fig = plt.figure(figsize=(FULL, 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.15], hspace=0.38)
    top = outer[0].subgridspec(2, 3, hspace=0.3, wspace=0.42)
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)
    ax_x = fig.add_subplot(top[0, :2])
    ax_y = fig.add_subplot(top[1, :2], sharex=ax_x)
    ax_w = fig.add_subplot(top[:, 2], sharex=ax_x)
    ax_e, ax_r, ax_u = (fig.add_subplot(bottom[0, i]) for i in range(3))

    # The estimate each source handed over is the truth plus its recorded error;
    # the truth comes from the EKF run (both runs see the same plant to well
    # under the plot's resolution, and each estimate is drawn against its own).
    safe = False
    styles = {"raw": dict(color=EST_SHADE["raw"], lw=0.8,
                          label="raw: last 5 Hz fix (0.15 s old, held)"),
              "ekf": dict(color=EST_SHADE["ekf"], lw=1.0,
                          label="EKF: 500 Hz IMU, corrected at each fix")}
    truth_drawn = False
    for src in ("raw", "ekf"):
        s = C.series("estimator", f"{SCORE_SHAPE}_{src}_sqp_s1")
        if s is None:
            continue
        t = s["t_s"].to_numpy()
        x0, y0 = s["base_x"].iloc[0], s["base_y"].iloc[0]
        yaw = np.unwrap(s["base_yaw"].to_numpy())
        yaw0 = yaw[0]
        tx, ty, tw = (s["base_x"] - x0) * 1e3, (s["base_y"] - y0) * 1e3, (yaw - yaw0) * 1e3
        ex = tx + s["est_x"] * 1e3
        ey = ty + s["est_y"] * 1e3
        ew = tw + np.remainder(s["est_yaw"].to_numpy() + np.pi, 2 * np.pi) * 1e3 - np.pi * 1e3
        if not truth_drawn:
            for ax, v in ((ax_x, tx), (ax_y, ty), (ax_w, tw)):
                ax.plot(t, v, color="#C0392B", ls="--", lw=1.0, label="truth", zorder=3)
            truth_drawn = True
        for ax, v in ((ax_x, ex), (ax_y, ey), (ax_w, ew)):
            ax.plot(t, v, **styles[src])
            safe |= mark_safe_series(ax, s, t, v)
    ax_x.set_ylabel("base x [mm]")
    ax_y.set_ylabel("base y [mm]")
    ax_w.set_ylabel("base yaw [mrad]")
    ax_x.tick_params(labelbottom=False)
    for ax in (ax_y, ax_w):
        ax.set_xlabel("time [s]")
    ax_w.set_xticks([0, 5, 10, 15, 20])
    ax_x.set_title(f"base state given to {C.LABEL['sqp']}, from its start "
                   f"({SCORE_SHAPE}, one seed)", loc="left")
    handles, _ = ax_x.get_legend_handles_labels()
    ax_x.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.75, 1.18), ncol=3,
                fontsize=6, frameon=False)

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
    label_panels([ax_x, ax_y, ax_w, ax_e, ax_r, ax_u])
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
    label_panels(axes)
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
    label_panels(axes)
    out_dir = C.OUT
    C.OUT = C.OUT_ROOT       # the comparison belongs with the thesis's (tuned) figures
    save(fig, "fig15_shipped_vs_tuned")
    C.OUT = out_dir


# ------------------------------------------------------------------- live --
# The live stack (live/driver.py): the deployed nodes against the simulator in
# real time, a few repeats per cell. Each figure sets the live result against
# its offline twin (study live_twin: the same scenario, sensors and estimator on
# the bench), so what the real stack adds -- threads, DDS, the 100 Hz estimate
# hand-over -- is the gap between the two and nothing else.

LIVE_STYLE = dict(ls="none", capsize=2, ms=4.5, mew=1.0, zorder=4)
TWIN_STYLE = dict(lw=1.0, alpha=0.9, zorder=3)


def _live_frames(need_twin: bool = True):
    live = C.live_summary()
    if live is None:
        return None, None, "no live run collected (live/collect.py)"
    twin = C.live_twin()
    if need_twin and twin is None:
        return live, None, "no live_twin study (run_study.py run live_twin)"
    return live, twin, ""


def _live_failed(rows) -> bool:
    return bool(len(rows)) and float(rows["safe_ticks"].max()) > 0


def live_handles(with_twin: bool = True) -> List[plt.Line2D]:
    out = [plt.Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=4.5,
                      label="live (mean, 95% CI)")]
    if with_twin:
        out.append(plt.Line2D([], [], color="0.3", lw=1.0, label="offline twin"))
    return out


def fig_live_nominal() -> None:
    live, twin, why = _live_frames()
    if live is None:
        return skip("live_nominal", why)
    exact = C.summary("nominal")
    live = C.at_nominal(live[live["source"] == "ekf"])
    twin = C.at_nominal(twin[twin["source"] == "ekf"]) if twin is not None else None
    metrics = (("rmse_mm", "RMSE [mm]", True), ("tau_step_rms", "rms Δτ per tick [N m]", True))
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.8), sharex=True)
    safe = False
    x = np.arange(len(C.CTRL))
    for j, shape in enumerate(("circle", "rectangle")):
        for i, (col, label, log) in enumerate(metrics):
            ax = axes[i, j]
            for k, c in enumerate(C.CTRL):
                d = live[(live["shape"] == shape) & (live["tag"] == c)]
                m, h = C.ci95(d[col])
                ax.errorbar(x[k] + 0.12, m, yerr=h, color=C.COLOUR[c],
                            marker=C.MARKER[c], mfc="none", **LIVE_STYLE)
                ax.plot(np.full(len(d), x[k] + 0.12), d[col], ls="none", marker=".",
                        ms=2.5, color=C.COLOUR[c], alpha=0.5, zorder=3)
                safe |= mark_safe_at(ax, [x[k] + 0.12], [m], [_live_failed(d)])
                if twin is not None:
                    t = twin[(twin["shape"] == shape) & (twin["tag"] == c)]
                    tm, th = C.ci95(t[col])
                    ax.errorbar(x[k] - 0.12, tm, yerr=th, color=C.COLOUR[c],
                                marker=C.MARKER[c], **LIVE_STYLE)
                if exact is not None:
                    e = exact[(exact["scenario"] == shape) & (exact["tag"] == c)]
                    if len(e):
                        ax.plot(x[k] - 0.36, e[col].iloc[0], marker="_", ms=9, mew=1.6,
                                color=C.COLOUR[c], ls="none", zorder=3)
            if log:
                ax.set_yscale("log")
                plain_log_y(ax)
            if j == 0:
                ax.set_ylabel(label)
        axes[0, j].set_title(shape)
        axes[1, j].set_xticks(x)
        axes[1, j].set_xticklabels([C.LABEL[c] for c in C.CTRL])
    handles = [plt.Line2D([], [], color="0.3", marker="_", ms=9, mew=1.6, ls="none",
                          label="offline, exact state"),
               plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=4.5,
                          label="offline twin (EKF)"),
               plt.Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=4.5,
                          label="live (EKF), mean and repeats")]
    legend_below(fig, handles + ([safe_handle()] if safe else []), ncol=4)
    label_panels(axes)
    save(fig, "fig17_live_nominal")


#: (live knob, x label, symlog x)
LIVE_SWEEPS = (("speed", "speed factor", False), ("mass", "link mass / model mass", False),
               ("damping", "base damping [N s/m]", True), ("tau", "τ limit / 0.25 N m", False))


def fig_live_robustness() -> None:
    live, twin, why = _live_frames()
    if live is None:
        return skip("live_robustness", why)
    live = live[live["source"] == "ekf"]
    twin = twin[twin["source"] == "ekf"] if twin is not None else None
    fig = plt.figure(figsize=(FULL, 5.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95], hspace=0.42, wspace=0.36)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)])
    safe = False
    for i, shape in enumerate(("circle", "rectangle")):
        for j, (knob, xl, symlog) in enumerate(LIVE_SWEEPS):
            ax = axes[i, j]
            lv = C.at_nominal(live[live["shape"] == shape], but=knob)
            tw = C.at_nominal(twin[twin["shape"] == shape], but=knob) if twin is not None else None
            for c in C.CTRL:
                if tw is not None:
                    d = tw[tw["tag"] == c]
                    pts = sorted(d[knob].unique())
                    if pts:
                        m, h = map(np.array, zip(*(C.ci95(d[d[knob] == v]["rmse_mm"]) for v in pts)))
                        ax.plot(pts, m, color=C.COLOUR[c], **TWIN_STYLE)
                        ax.fill_between(pts, m - h, m + h, color=C.COLOUR[c], alpha=0.15, lw=0)
                d = lv[lv["tag"] == c]
                pts = sorted(d[knob].unique())
                if pts:
                    m, h = map(np.array, zip(*(C.ci95(d[d[knob] == v]["rmse_mm"]) for v in pts)))
                    ax.errorbar(pts, m, yerr=h, color=C.COLOUR[c], marker=C.MARKER[c],
                                mfc="none", **LIVE_STYLE)
                    safe |= mark_safe_at(ax, pts, m, [_live_failed(d[d[knob] == v]) for v in pts])
            mark_nominal(ax, C.LIVE_KNOBS[knob])
            ax.set_yscale("log")
            plain_log_y(ax)
            if symlog:
                ax.set_xscale("symlog", linthresh=0.05, linscale=0.5)
                ticks = sorted(set(lv[knob].unique()) | (set(tw[knob].unique()) if tw is not None else set()))
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=6)
                ax.set_xlim(left=0)
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if i == 1:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"{shape} RMSE [mm]")
    # Bottom row: how often the speed sweep drove a joint to its torque limit
    # (bench fig07's second row, live), circle and rectangle.
    rail_axes = []
    for j, shape in enumerate(("circle", "rectangle")):
        ax = fig.add_subplot(gs[2, j])
        rail_axes.append(ax)
        lv = C.at_nominal(live[live["shape"] == shape], but="speed")
        tw = C.at_nominal(twin[twin["shape"] == shape], but="speed") if twin is not None else None
        for c in DRAW_ORDER:
            if tw is not None:
                d = tw[tw["tag"] == c]
                pts = sorted(d["speed"].unique())
                if pts:
                    m, h = map(np.array, zip(*(C.ci95(100 * d[d["speed"] == v]["rail_frac"])
                                               for v in pts)))
                    ax.plot(pts, m, color=C.COLOUR[c], **TWIN_STYLE)
                    ax.fill_between(pts, m - h, m + h, color=C.COLOUR[c], alpha=0.15, lw=0)
            d = lv[lv["tag"] == c]
            pts = sorted(d["speed"].unique())
            if pts:
                m, h = map(np.array, zip(*(C.ci95(100 * d[d["speed"] == v]["rail_frac"])
                                           for v in pts)))
                ax.errorbar(pts, m, yerr=h, color=C.COLOUR[c], marker=C.MARKER[c], mfc="none",
                            **LIVE_STYLE)
                safe |= mark_safe_at(ax, pts, m, [_live_failed(d[d["speed"] == v]) for v in pts])
        mark_nominal(ax, 1.0)
        ax.set_xticks(sorted(lv["speed"].unique()))
        ax.set_xlabel("speed factor")
        ax.text(0.04, 0.96, shape, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        if j == 0:
            ax.set_ylabel("joint-ticks at\ntorque limit [%]")
    handles = ctrl_handles() + live_handles() + [nominal_handle()] + \
        ([safe_handle()] if safe else [])
    ax_leg = fig.add_subplot(gs[2, 2:])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=6.5)
    label_panels(list(axes.flat) + rail_axes)
    save(fig, "fig18_live_robustness")


def fig_live_estimator() -> None:
    live, twin, why = _live_frames()
    if live is None:
        return skip("live_estimator", why)
    live = C.at_nominal(live)
    twin = C.at_nominal(twin) if twin is not None else None
    srcs = [s for s in C.LIVE_SOURCES if s in set(live["source"])]
    if len(srcs) < 2:
        return skip("live_estimator", "fewer than two state sources run live")
    metrics = (("rmse_mm", "RMSE [mm]"), ("tau_step_rms", "rms Δτ per tick [N m]"),
               ("est_pos_rms_mm", "base position error [mm]"))
    fig, axes = plt.subplots(len(metrics), 2, figsize=(FULL, 5.0), sharex=True)
    x = np.arange(len(srcs))
    safe = False
    for j, shape in enumerate(("circle", "rectangle")):
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            for k, c in enumerate(C.CTRL):
                off = (k - 1.5) * 0.06
                if twin is not None:
                    t = twin[(twin["shape"] == shape) & (twin["tag"] == c)]
                    tm = [C.ci95(t[t["source"] == s][col])[0] for s in srcs]
                    ax.plot(x + off, tm, color=C.COLOUR[c], ls="--", **TWIN_STYLE)
                d = live[(live["shape"] == shape) & (live["tag"] == c)]
                m, h = map(np.array, zip(*(C.ci95(d[d["source"] == s][col]) for s in srcs)))
                ax.errorbar(x + off, m, yerr=h, color=C.COLOUR[c], marker=C.MARKER[c],
                            mfc="none", **LIVE_STYLE)
                ax.plot(x + off, m, color=C.COLOUR[c], lw=0.8, zorder=3)
                safe |= mark_safe_at(ax, x + off, m,
                                     [_live_failed(d[d["source"] == s]) for s in srcs])
            ax.set_yscale("log")
            plain_log_y(ax)
            if j == 0:
                ax.set_ylabel(label)
        axes[0, j].set_title(shape)
        axes[-1, j].set_xticks(x)
        axes[-1, j].set_xticklabels([C.SOURCE_LABEL[s] for s in srcs])
    handles = ctrl_handles() + [
        plt.Line2D([], [], color="0.3", marker="o", mfc="none", lw=0.8, ms=4.5,
                   label="live (mean, 95% CI)"),
        plt.Line2D([], [], color="0.3", ls="--", lw=1.0, label="offline twin")]
    legend_below(fig, handles + ([safe_handle()] if safe else []), ncol=4)
    label_panels(axes)
    save(fig, "fig19_live_estimator")


def fig_live_timing() -> None:
    """Every control tick of every live trial: the distribution, and the worst
    tick of each trial, which is what a deadline is judged by."""
    live, _, why = _live_frames(need_twin=False)
    if live is None:
        return skip("live_timing", why)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"wspace": 0.3})
    labels, have_series = [], False
    # The per-tick records if this machine has them; otherwise the histogram
    # collect.py kept in git (tick_hist.json), the same numbers binned.
    stored = C.read_tick_hist(os.path.join(C.LIVE_DIR, C.LIVE_RUN, "tick_hist.json"))
    for c in C.CTRL:
        d = live[live["tag"] == c]
        ticks = []
        for h in d["hash"]:
            s = C.live_series(h)
            if s is not None and "tick_us" in s:
                ticks.append(s["tick_us"].to_numpy())
        misses = int(d["deadline_misses"].fillna(0).sum())
        labels.append((c, f"{C.LABEL[c]} ({misses} missed)"))
        hist = C.tick_hist_of(np.concatenate(ticks)) if ticks else \
            (stored or {}).get(c)
        if hist is not None and hist["n"]:
            have_series = True
            C.stairs_tick_hist(axes[0], hist, color=C.COLOUR[c])
            axes[0].plot(hist["max"], 1.0, marker="v", color=C.COLOUR[c], ms=4, ls="none",
                         transform=axes[0].get_xaxis_transform(), clip_on=False)
        # Right: per trial, the worst tick (filled) over its p99 (hollow).
        k = C.CTRL.index(c)
        jit = (np.random.default_rng(k).random(len(d)) - 0.5) * 0.35
        axes[1].plot(k + jit, d["tick_us_max"], ls="none", marker=C.MARKER[c], ms=3,
                     color=C.COLOUR[c], alpha=0.8)
        axes[1].plot(k + jit, d["tick_us_p99"], ls="none", marker=C.MARKER[c], ms=3,
                     mfc="none", color=C.COLOUR[c], alpha=0.5)
    for ax in axes:
        (ax.axvline if ax is axes[0] else ax.axhline)(C.PERIOD_US, color="k", ls=":", lw=0.8)
    if have_series:
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(0.5, 5e4)
        axes[0].set_xlabel("control tick [µs] (▼ worst recorded)")
        axes[0].set_ylabel("density")
    else:
        axes[0].text(0.5, 0.5, "per-tick records not on this machine", ha="center",
                     va="center", transform=axes[0].transAxes, fontsize=7)
        axes[0].set_axis_off()
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(C.CTRL)))
    axes[1].set_xticklabels([C.LABEL[c] for c in C.CTRL])
    axes[1].set_ylabel("tick per trial [µs]")
    handles = [plt.Line2D([], [], color=C.COLOUR[c], marker=C.MARKER[c], lw=1.1, ms=4,
                          label=l) for c, l in labels]
    handles += [plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=3, label="worst"),
                plt.Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=3,
                           label="p99"),
                plt.Line2D([], [], color="k", ls=":", lw=0.8, label="10 ms period")]
    fig.subplots_adjust(bottom=0.33)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    label_panels(axes)
    save(fig, "fig20_live_timing")


def fig_live_trace() -> None:
    """One live trial per controller on the rectangle: the error against time."""
    live, _, why = _live_frames(need_twin=False)
    if live is None:
        return skip("live_trace", why)
    d = C.at_nominal(live[(live["shape"] == "rectangle") & (live["source"] == "ekf")])
    if d.empty:
        return skip("live_trace", "no live rectangle on the EKF")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw={"wspace": 0.28,
                                                                     "width_ratios": [1.6, 1]})
    safe = drawn = False
    ref = None
    for c in C.CTRL:
        r = d[d["tag"] == c].sort_values("repeat")
        if r.empty:
            continue
        s = C.live_series(r["hash"].iloc[0])
        if s is None:
            # The trace collect.py copied into git for this figure.
            p = os.path.join(C.LIVE_DIR, C.LIVE_RUN, "plot_series",
                             f"{c}_rectangle_x1_m1_bd0_tau1x_ekf.csv")
            s = pd.read_csv(p) if os.path.isfile(p) else None
        if s is None:
            continue
        drawn = True
        axes[0].plot(s["t_s"], s["err_m"] * 1e3, color=C.COLOUR[c], lw=0.9, label=C.LABEL[c])
        safe |= mark_safe_series(axes[0], s, s["t_s"], s["err_m"] * 1e3)
        axes[1].plot(s["ee_x"] * 1e3, s["ee_y"] * 1e3, color=C.COLOUR[c], lw=0.8)
        ref = s
    if not drawn:
        return skip("live_trace", "per-tick records not on this machine")
    axes[1].plot(ref["ref_x"] * 1e3, ref["ref_y"] * 1e3, color="k", ls=":", lw=0.8)
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    axes[0].axvline(float(d["window_start_s"].iloc[0]), **NOMINAL_STYLE)
    axes[0].set_yscale("log")
    plain_log_y(axes[0])
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("EE error [mm]")
    handles = ctrl_handles() + [plt.Line2D([], [], color="k", ls=":", lw=0.8, label="reference"),
                                plt.Line2D([], [], label="scoring starts", **NOMINAL_STYLE)]
    legend_below(fig, handles + ([safe_handle()] if safe else []), ncol=6)
    label_panels(axes)
    save(fig, "fig21_live_trace")


def fig_live_cap() -> None:
    """What the QP iteration cap bought and what it cost, live: each NMPC's worst
    tick per trial before (the first run, uncapped: 60 iterations, twice) and
    after, against the period; and each cell's RMSE after against before."""
    now = C.live_summary()
    before = C.live_summary(stale=True)
    if before is not None and "qp_cap" in before:
        # summary_stale.csv holds every superseded setting (LMPC's cap 14 at N 45
        # since its retune): "before" is the uncapped first run only.
        cap = before["qp_cap"].astype(str).str.strip().str.lower()
        before = before[before["law"].ne("nmpc") | cap.isin(("none", "nan", ""))]
        before = before if len(before) else None
    if now is None or before is None:
        return skip("live_cap", "needs both the capped run and the uncapped one "
                                "(live/collect.py writes summary_stale.csv)")
    nmpc = [c for c in ("sqp", "iter") if c in set(now["tag"]) and c in set(before["tag"])]
    if not nmpc:
        return skip("live_cap", "no NMPC trials in both runs")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"wspace": 0.32,
                                                                     "width_ratios": [1.15, 1]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    labels = []
    for k, c in enumerate(nmpc):
        for j, (df, cap_label, mfc) in enumerate(((before, "uncapped", "none"),
                                                   (now, "capped", C.COLOUR[c]))):
            d = df[df["tag"] == c]
            def cap_text(v):
                try:
                    return str(int(float(v)))
                except (TypeError, ValueError):
                    return str(v)
            caps = sorted({cap_text(v) for v in d["qp_cap"].dropna()}) if "qp_cap" in d else []
            cap = caps[0] if len(caps) == 1 and caps[0] not in ("", "none", "nan") else None
            x = 2 * k + j
            w = d["tick_us_max"].to_numpy() / 1000.0
            ax.plot(x + (rng.random(len(w)) - 0.5) * 0.45, w, ls="none", marker=C.MARKER[c],
                    ms=3, color=C.COLOUR[c], mfc=mfc, alpha=0.75)
            over = int((w > C.PERIOD_US / 1000).sum())
            ax.text(x, 1.04, f"{over}/{len(w)}\nover", ha="center", va="bottom", fontsize=6,
                    transform=ax.get_xaxis_transform())
            labels.append(f"{C.LABEL[c]}\n{'cap ' + cap if cap else cap_label}")
    ax.axhline(C.PERIOD_US / 1000, color="k", ls=":", lw=0.8)
    ax.text(len(labels) - 0.5, C.PERIOD_US / 1000 * 1.06, "10 ms period", ha="right",
            va="bottom", fontsize=6)
    ax.set_yscale("log")
    plain_log_y(ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("worst tick per live trial [ms]")

    ax = axes[1]
    lo, hi = np.inf, 0.0
    safe = False
    for c in nmpc:
        b = before[before["tag"] == c].groupby("cell")["rmse_mm"].mean()
        n = now[now["tag"] == c].groupby("cell")
        a = n["rmse_mm"].mean()
        failed_now = n["safe_ticks"].max() > 0 if "safe_ticks" in now else None
        cells = b.index.intersection(a.index)
        if not len(cells):
            continue
        x, y = b[cells].to_numpy(), a[cells].to_numpy()
        ratio = float(np.exp(np.mean(np.log(y / x))))
        ax.plot(x, y, ls="none", marker=C.MARKER[c], ms=3.5, color=C.COLOUR[c],
                label=f"{C.LABEL[c]}: x{ratio:.2f} (geo. mean)")
        if failed_now is not None:
            safe |= mark_safe_at(ax, x, y, list(failed_now[cells]))
        lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
    if hi > 0:
        ax.plot([lo * 0.8, hi * 1.25], [lo * 0.8, hi * 1.25], color="0.5", lw=0.8, ls="--",
                zorder=0, label="no change")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_y(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("RMSE per cell, uncapped [mm]")
    ax.set_ylabel("RMSE per cell, capped [mm]")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + ([safe_handle()] if safe else []), fontsize=6,
              loc="upper left")
    label_panels(axes)
    save(fig, "fig23_live_cap")


# ------------------------------------------------------- live, per tick --
# The figures below that need per-tick data read live/derive.py's small tables
# (data/thesis_eval/live/main/derived/, machines/derived/), which are in git, so
# they render from a clone. The comparison the reader must see first is SQP-NMPC
# against the baselines: it is drawn last, on top, with a heavier line.

#: Drawing order: the baselines first, SQP-NMPC last (on top).
DRAW_ORDER = ["smc", "ompc", "iter", "sqp"]
NO_DERIVED = "no derived table: run live/derive.py where the raw data is"
FAILED_COLOUR = "#D55E00"        # Okabe-Ito vermillion: a failed QP solve
CONVERGED_COLOUR = "0.55"


def ctrl_lw(c: str, base: float = 0.8) -> float:
    return base * 1.6 if c == "sqp" else base


def ctrl_z(c: str) -> int:
    return 4 if c == "sqp" else 3


def derived(name: str, where: str = "live"):
    base = os.path.join(C.LIVE_DIR, C.LIVE_RUN, "derived") if where == "live" else \
        os.path.join(C.DATA_ROOT, "machines", "derived")
    p = os.path.join(base, name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def panel_label(ax, text: str) -> None:
    """(a), (b), ... just above the panel's top-left corner, where the caption
    refers to it."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
                textcoords="offset points", ha="right", va="bottom", fontweight="bold")


def label_panels(axes, start: int = 0) -> None:
    """(a), (b), ... on each panel in the order given (row by row for a grid)."""
    for k, ax in enumerate(np.ravel(axes)):
        panel_label(ax, f"({chr(ord('a') + start + k)})")


def quantile_violin(ax, pos: float, q, v, colour, width: float = 0.75,
                    floor: float = 1e-3, sigma_dec: float = 0.03) -> None:
    """A violin on a log axis, drawn from a distribution's quantiles (q, v): the
    density of log10(v) is the derivative of the quantile function's inverse,
    lightly smoothed, so the shape is the data's and not a kernel guess in
    linear space. A bar from the minimum to the maximum and a median tick."""
    q, v = np.asarray(q, float), np.maximum(np.asarray(v, float), floor)
    z = np.maximum.accumulate(np.log10(v))
    if z[-1] - z[0] < 1e-9:
        return
    zg = np.linspace(z[0], z[-1], 500)
    dens = np.gradient(np.interp(zg, z, q), zg)
    sig = max(sigma_dec / (zg[1] - zg[0]), 1.0)
    k = np.arange(-int(4 * sig), int(4 * sig) + 1)
    kern = np.exp(-0.5 * (k / sig) ** 2)
    dens = np.convolve(np.pad(dens, len(k) // 2, mode="edge"), kern / kern.sum(), "valid")
    dens = dens / dens.max() * width / 2
    y = 10.0 ** zg
    ax.fill_betweenx(y, pos - dens, pos + dens, color=colour, alpha=0.5, lw=0, zorder=2)
    ax.plot(np.r_[pos - dens, (pos + dens)[::-1]], np.r_[y, y[::-1]], color=colour, lw=0.6,
            zorder=2)
    ax.plot([pos, pos], [v[0], v[-1]], color="0.25", lw=0.5, zorder=2)
    med = float(np.interp(0.5, q, v))
    ax.plot([pos - width * 0.22, pos + width * 0.22], [med, med], color="0.15", lw=0.9, zorder=3)


def _live_track(shape: str, name: str, corner_zoom: bool) -> None:
    """Live tracking on one shape, flight sensing through the EKF: the last lap's
    path (one repeat per controller, the one with the median RMSE of three) and a
    true-scale zoom on where the paths differ most; that repeat's error against
    time; and the per-tick error pooled over all three repeats' scoring windows,
    with each controller's RMSE (mean of three) and the floors it is judged by."""
    tr, dist = derived(f"track_{shape}.csv"), derived("errdist.csv")
    if tr is None or dist is None:
        return skip(name, NO_DERIVED)
    dist = dist[dist["shape"] == shape]
    fig = plt.figure(figsize=(FULL, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], wspace=0.32, hspace=0.42)
    ax_xy, ax_z = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_t, ax_v = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    paths, ref, safe, w0 = [], None, False, None
    for c in DRAW_ORDER:
        s = tr[tr["tag"] == c]
        if s.empty:
            continue
        lap = s[s["t_s"] >= s["t_s"].iloc[-1] - LAP_S]
        if ref is None or c == "sqp":
            ref = lap[["ref_x", "ref_y"]].to_numpy()
        ee = lap[["ee_x", "ee_y"]].to_numpy()
        paths.append((c, ee, _path_deviation(ref, ee)))
        e = s["err_m"].to_numpy() * 1e3
        ax_t.plot(s["t_s"], e, color=C.COLOUR[c], lw=ctrl_lw(c, 0.6), zorder=ctrl_z(c))
        safe |= mark_safe_series(ax_t, s, s["t_s"], e)
        w0 = float(s["window_start_s"].iloc[0])
    if not paths:
        plt.close(fig)
        return skip(name, f"no {shape} tracks")
    paths = [(c, ee, _path_deviation(ref, ee)) for c, ee, _ in paths]
    # The lap is cut out of a longer run: keep the zoom off its cut ends.
    trim = [(c, ee[10:-10], dev[10:-10]) for c, ee, dev in paths]
    centre, half = _zoom_box(ref, trim, corner_zoom)
    mm = 1e3
    ax_xy.plot(ref[:, 0] * mm, ref[:, 1] * mm, color="k", ls=":", lw=0.9, zorder=5)
    ax_z.plot((ref[:, 0] - centre[0]) * mm, (ref[:, 1] - centre[1]) * mm, color="k", ls=":",
              lw=1.0, zorder=5)
    for c, ee, _ in paths:
        ax_xy.plot(ee[:, 0] * mm, ee[:, 1] * mm, color=C.COLOUR[c], lw=ctrl_lw(c, 0.7),
                   zorder=ctrl_z(c))
        ax_z.plot((ee[:, 0] - centre[0]) * mm, (ee[:, 1] - centre[1]) * mm, color=C.COLOUR[c],
                  lw=ctrl_lw(c, 0.8), zorder=ctrl_z(c))
    ax_xy.add_patch(plt.Rectangle((centre - half) * mm, 2 * half * mm, 2 * half * mm,
                                  fill=False, ec="k", lw=0.7, zorder=6))
    ax_xy.annotate("(b)", ((centre[0] - half) * mm, (centre[1] + half) * mm), xytext=(0, 1.5),
                   textcoords="offset points", fontsize=6, ha="left", va="bottom")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_z.set_xlim(-half * mm, half * mm)
    ax_z.set_ylim(-half * mm, half * mm)
    ax_z.set_aspect("equal")
    ax_z.set_xlabel(r"$x - x_0$ [mm]")
    ax_z.set_ylabel(r"$y - y_0$ [mm]")

    if w0 is not None:
        ax_t.axvline(w0, **NOMINAL_STYLE)
    ax_t.set_yscale("log")
    plain_log_y(ax_t)
    ax_t.set_xlim(0, float(tr["t_s"].max()))
    ax_t.set_xticks(np.arange(0, float(tr["t_s"].max()) + 0.1, 5.0))
    ax_t.set_xlabel("time [s]")
    ax_t.set_ylabel("EE error [mm]")

    # The distribution: every tick of three repeats' scoring windows.
    exact = C.summary("nominal")
    live = C.live_summary()
    for k, c in enumerate(C.CTRL):
        d = dist[dist["tag"] == c]
        if d.empty:
            continue
        quantile_violin(ax_v, k, d["q"], d["err_mm"], C.COLOUR[c])
        r = float(d["rmse_mean_of_repeats_mm"].iloc[0])
        ax_v.plot(k, r, marker="D", ms=4.5, mfc="white", mec=C.COLOUR[c], mew=1.2, ls="none",
                  zorder=6)
        ax_v.text(k + 0.4, r, f"{r:.1f}" if r < 10 else f"{r:.0f}", va="center", ha="left",
                  fontsize=6.5, color=C.COLOUR[c], fontweight="bold" if c == "sqp" else "normal",
                  zorder=6, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
        if int(d["safe_ticks"].iloc[0]) > 0:
            safe |= mark_safe_at(ax_v, [k], [float(d["err_mm"].max()) * 1.3], [True])
    floors = []
    if exact is not None:
        e = exact[exact["cell"] == f"{shape}_sqp"]
        if len(e):
            ax_v.axhline(float(e["rmse_mm"].iloc[0]), color=C.COLOUR["sqp"], ls="--", lw=0.8,
                         zorder=1)
            floors.append(plt.Line2D([], [], color=C.COLOUR["sqp"], ls="--", lw=0.8,
                                     label=f"{C.LABEL['sqp']} RMSE, bench exact state"))
    if live is not None:
        nav = C.at_nominal(live[(live["source"] == "ekf") & (live["shape"] == shape)])
        if len(nav):
            ax_v.axhline(float(nav["est_pos_rms_mm"].mean()), color="0.2", ls="-.", lw=0.8,
                         zorder=1)
            floors.append(plt.Line2D([], [], color="0.2", ls="-.", lw=0.8,
                                     label="EKF base position error (rms)"))
    ax_v.set_yscale("log")
    plain_log_y(ax_v)
    ax_v.set_xticks(range(len(C.CTRL)))
    ax_v.set_xticklabels([C.LABEL[c] for c in C.CTRL])
    ax_v.set_xlim(-0.6, len(C.CTRL) - 0.25)
    ax_v.set_ylabel("EE error per tick [mm]")
    ax_v.grid(axis="x", visible=False)
    for ax, lab in ((ax_xy, "(a)"), (ax_z, "(b)"), (ax_t, "(c)"), (ax_v, "(d)")):
        panel_label(ax, lab)
    handles = [plt.Line2D([], [], color=C.COLOUR[c], lw=ctrl_lw(c, 1.0), label=C.LABEL[c])
               for c in C.CTRL]
    handles += [plt.Line2D([], [], color="k", ls=":", lw=0.9, label="reference"),
                plt.Line2D([], [], label="scoring starts", **NOMINAL_STYLE),
                plt.Line2D([], [], ls="none", marker="D", ms=4.5, mfc="white", mec="0.2",
                           mew=1.2, label="RMSE, mean of 3 repeats"),
                plt.Line2D([], [], color="0.15", lw=0.9, label="median")] + floors
    if safe:
        handles.append(safe_handle())
    legend_below(fig, handles, ncol=4)
    save(fig, name)


def fig_live_rectangle() -> None:
    _live_track("rectangle", "fig26_live_rectangle", corner_zoom=True)


def fig_live_circle() -> None:
    _live_track("circle", "fig25_live_circle", corner_zoom=False)


#: Node stages, grouped for the "where the time goes" bars: (label, grey).
STAGE_GROUPS = (("QP solve (HPIPM)", "0.15"), ("QP set-up", "0.45"),
                ("rollout", "0.68"), ("model and the rest", "0.88"))


def _stage_split(r) -> List[float]:
    """[ms] of one exposed_stages row, in STAGE_GROUPS order; they sum to the
    mean tick."""
    hp = r["hpipm_us"]
    setup = max(r["qpsolve"] - hp, 0.0)
    roll = r["rollout_dynamics"] + r["rollout_integrate"]
    rest = r["tick_us"] - hp - setup - roll
    return [v / 1000.0 for v in (hp, setup, roll, rest)]


def fig_live_exposed() -> None:
    """What the live runs exposed: every NMPC tick's time against the QP
    iterations it took, before the cap (60 per solve, a failed solve retried
    cold) and after it (20); and where a converged and a failed tick's time goes."""
    t, st = derived("exposed_ticks.csv"), derived("exposed_stages.csv")
    if t is None or st is None:
        return skip("live_exposed", NO_DERIVED)
    runs = (("uncapped", "before: 60 iterations, retried"), ("capped", "after: cap 20"))
    fig = plt.figure(figsize=(FULL, 4.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.5)
    top = outer[0].subgridspec(1, 4, wspace=0.22)
    axes = [fig.add_subplot(top[0, 0])]
    axes += [fig.add_subplot(top[0, i], sharex=axes[0], sharey=axes[0]) for i in (1, 2, 3)]
    period_ms = C.PERIOD_US / 1000
    k = 0
    for c in ("sqp", "iter"):
        for run, title in runs:
            ax = axes[k]
            d = t[(t["run"] == run) & (t["tag"] == c)]
            s = st[(st["run"] == run) & (st["tag"] == c)].set_index("kind")
            for failed, colour, z in ((0, CONVERGED_COLOUR, 2), (1, FAILED_COLOUR, 3)):
                p = d[d["failed"] == failed]
                ax.plot(p["qp_iters"] + (np.random.default_rng(k).random(len(p)) - 0.5) * 0.6,
                        p["tick_us"] / 1000, ls="none", marker=C.MARKER[c], ms=1.6, mew=0,
                        color=colour, alpha=0.35 if failed == 0 else 0.5, zorder=z,
                        rasterized=True)
            worst = float(s["tick_us_max"].max())
            over = int(s["over_period"].sum())
            n = int(s["ticks"].sum())
            nf = int(s.loc["failed", "ticks"]) if "failed" in s.index else 0
            wrow = d.loc[d["tick_us"].idxmax()]
            ax.plot(wrow["qp_iters"], worst / 1000, marker="v", ms=4.5, color="k", ls="none",
                    zorder=5)
            ax.axhline(period_ms, color="k", ls=":", lw=0.8, zorder=4)
            ax.text(0.97, 0.03, f"worst {worst / 1000:.1f} ms\n{over:,} of {n:,} ticks > 10 ms\n"
                    f"{100 * nf / n:.1f}% failed", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.8, linespacing=1.15,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
            when, what = title.split(": ")
            ax.set_title(f"{C.LABEL[c]}, {when}\n{what}", fontsize=7)
            ax.set_xlabel("QP iterations in the tick")
            if k:
                ax.tick_params(axis="y", which="both", labelleft=False)
            k += 1
    axes[0].set_yscale("log")
    plain_log_y(axes[0])
    axes[0].set_ylim(0.2, 40)
    axes[0].set_xlim(-4, 126)
    axes[0].set_xticks([0, 20, 40, 60, 120])
    axes[0].set_ylabel("control tick [ms]")
    label_panels(axes)

    # Where the time goes: the mean tick of each kind, split by stage.
    ax = fig.add_subplot(outer[1])
    rows, labels = [], []
    for run, title in runs:
        for c in ("sqp", "iter"):
            for kind in ("converged", "failed"):
                r = st[(st["run"] == run) & (st["tag"] == c) & (st["kind"] == kind)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(_stage_split(r))
                labels.append(f"{title.split(':')[0]}, {C.LABEL[c]}, {kind} "
                              f"({100 * r['ticks'] / st[(st['run'] == run) & (st['tag'] == c)]['ticks'].sum():.1f}%)")
    y = np.arange(len(rows))[::-1].astype(float)
    y[: len(rows) // 2] += 0.5          # a gap between before and after
    left = np.zeros(len(rows))
    vals = np.array(rows)
    for j, (lab, grey) in enumerate(STAGE_GROUPS):
        ax.barh(y, vals[:, j], left=left, height=0.72, color=grey, ec="0.1", lw=0.3, label=lab)
        left += vals[:, j]
    for yi, tot in zip(y, left):
        ax.text(tot + 0.12, yi, f"{tot:.2f} ms", va="center", fontsize=6)
    ax.axvline(period_ms, color="k", ls=":", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlim(0, period_ms * 1.12)
    ax.set_xlabel("mean control tick [ms], by stage")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "(e)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=CONVERGED_COLOUR,
                          label="QP converged"),
               plt.Line2D([], [], ls="none", marker="o", ms=4, color=FAILED_COLOUR,
                          label="QP failed (cap reached / retried)"),
               plt.Line2D([], [], ls="none", marker="v", ms=4.5, color="k", label="worst tick"),
               plt.Line2D([], [], color="k", ls=":", lw=0.8, label="10 ms period")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=4,
               frameon=False)
    save(fig, "fig24_live_exposed")


#: Condition families of the live cells, in panel order: (key, title).
FAMILIES = (("nominal", "nominal"), ("speed", "speed 2-4×"), ("mass", "link mass 0.5-3×"),
            ("damping", "base damping"), ("tau", "torque limit 0.5-2×"),
            ("sensing", "raw sensors / EKF + joint KF"))


def _family(r) -> str:
    if r["source"] != "ekf":
        return "sensing"
    for k, nominal in C.LIVE_KNOBS.items():
        if abs(float(r[k]) - nominal) > 1e-9:
            return k
    return "nominal"


def _cell_label(r) -> str:
    out = [r["shape"][:4] + "."]
    fam = _family(r)
    if fam == "speed":
        out.append(f"{float(r['speed']):g}×")
    elif fam == "mass":
        out.append(f"m {float(r['mass']):g}×")
    elif fam == "damping":
        out.append(f"b {float(r['damping']):g}")
    elif fam == "tau":
        out.append(f"τ {float(r['tau']):g}×")
    elif fam == "sensing":
        out.append(r["source"])
    return " ".join(out) + f", {C.LABEL[r['tag']].replace('-NMPC', '')}"


KNOB_COLS = ["shape", "speed", "mass", "damping", "tau", "source", "tag"]


def fig_live_twin_agreement() -> None:
    """Every live cell's RMSE (mean of three repeats, bars: min to max) against
    its bench twin's (mean of ten sensor-noise seeds), one panel per condition
    family. On the line, the real-time stack changes nothing."""
    live, twin, why = _live_frames()
    if live is None or twin is None:
        return skip("live_twin_agreement", why or "no live_twin study")
    lv = live.groupby(KNOB_COLS).agg(m=("rmse_mm", "mean"), lo=("rmse_mm", "min"),
                                     hi=("rmse_mm", "max"), safe=("safe_ticks", "max"),
                                     n=("rmse_mm", "size")).reset_index()
    tw = twin.groupby(KNOB_COLS).agg(t=("rmse_mm", "mean"),
                                     tsafe=("safe_ticks", "max")).reset_index()
    d = lv.merge(tw, on=KNOB_COLS, how="inner")
    if d.empty:
        return skip("live_twin_agreement", "no live cell matches a twin")
    d["family"] = d.apply(_family, axis=1)
    d["ratio"] = d["m"] / d["t"]
    lo = min(d["lo"].min(), d["t"].min()) * 0.7
    hi = max(d["hi"].max(), d["t"].max()) * 1.4
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.5), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.28})
    safe = False
    for ax, (fam, title), lab in zip(axes.flat, FAMILIES, "abcdef"):
        f = d[d["family"] == fam]
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=0.8, zorder=1)
        for fac in (0.5, 2.0):
            ax.plot([lo, hi], [lo * fac, hi * fac], color="0.6", lw=0.6, ls="--", zorder=1)
        for c in DRAW_ORDER:
            for shape, filled in (("circle", False), ("rectangle", True)):
                p = f[(f["tag"] == c) & (f["shape"] == shape)]
                if p.empty:
                    continue
                m = p["m"].to_numpy()
                ax.errorbar(p["t"].to_numpy(), m,
                            yerr=np.vstack([m - p["lo"].to_numpy(), p["hi"].to_numpy() - m]),
                            ls="none", marker=C.MARKER[c], ms=3.6 if c == "sqp" else 3.0,
                            mfc=C.COLOUR[c] if filled else "white", mec=C.COLOUR[c],
                            ecolor=C.COLOUR[c], elinewidth=0.6, mew=0.9, capsize=0,
                            zorder=ctrl_z(c))
                safe |= mark_safe_at(ax, p["t"], p["m"], list(p["safe"] > 0))
        # The cells furthest off the line, named (at most three per panel).
        off = f.assign(dev=np.abs(np.log(f["ratio"])))
        off = off[off["dev"] > np.log(1.6)].nlargest(3, "dev")
        # Numbered on the plot and named in a key in the panel's empty corner
        # below the line (the points sit on or above it).
        key = []
        for i, (_, r) in enumerate(off.sort_values("m", ascending=False).iterrows(), 1):
            ax.annotate(str(i), (r["t"], r["m"]), xytext=(-5, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6, fontweight="bold", color="0.1")
            key.append(f"{i}  {_cell_label(r)}")
        if key:
            ax.text(0.97, 0.03, "\n".join(key), transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.5, linespacing=1.25,
                    bbox=dict(fc="white", ec="0.8", lw=0.4, alpha=0.9, pad=1.5))
        within = float((np.abs(np.log(f["ratio"])) <= np.log(1.5)).mean()) if len(f) else 0
        ax.text(0.03, 0.97, f"{len(f)} cells, {100 * within:.0f}% within ×1.5",
                transform=ax.transAxes, ha="left", va="top", fontsize=6,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
        ax.set_title(title, fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        plain_log_y(ax)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        panel_label(ax, f"({lab})")
    for ax in axes[1]:
        ax.set_xlabel("bench twin RMSE [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("live RMSE [mm]")
    within = float((np.abs(np.log(d["ratio"])) <= np.log(1.5)).mean())
    print(f"    live/twin: {len(d)} cells, {100 * within:.0f}% within x1.5, "
          f"median ratio {d['ratio'].median():.2f}")
    handles = [plt.Line2D([], [], ls="none", marker=C.MARKER[c], color=C.COLOUR[c], ms=4,
                          label=C.LABEL[c]) for c in C.CTRL]
    handles += [plt.Line2D([], [], ls="none", marker="o", mfc="white", mec="0.3", ms=4,
                           label="circle"),
                plt.Line2D([], [], ls="none", marker="o", color="0.3", ms=4, label="rectangle"),
                plt.Line2D([], [], color="0.3", lw=0.8, label="live = twin"),
                plt.Line2D([], [], color="0.6", lw=0.6, ls="--", label="×2, ×½")]
    if safe:
        handles.append(safe_handle())
    legend_below(fig, handles, ncol=5)
    save(fig, "fig27_live_twin_agreement")


def fig_live_pareto() -> None:
    """fig05 on the live speed cells: MSE against torque energy in the scoring
    window, mean of three repeats, the speed as the marker's size and label."""
    live, _, why = _live_frames(need_twin=False)
    if live is None:
        return skip("live_pareto", why)
    sp = C.at_nominal(live[live["source"] == "ekf"], but="speed")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw={"wspace": 0.28})
    size = {1: 3.0, 2: 4.5, 3: 6.0, 4: 7.5}
    safe = False
    for ax, shape, lab in zip(axes, ("circle", "rectangle"), "ab"):
        sub = sp[sp["shape"] == shape]
        for c in DRAW_ORDER:
            d = sub[sub["tag"] == c].groupby("speed").agg(
                x=("tau2_int_win", "mean"), y=("mse_mm2", "mean"),
                safe=("safe_ticks", "max")).reset_index().sort_values("speed")
            if d.empty:
                continue
            ax.plot(d["x"], d["y"], color=C.COLOUR[c], lw=ctrl_lw(c, 0.8), zorder=ctrl_z(c))
            for _, r in d.iterrows():
                ax.plot(r["x"], r["y"], color=C.COLOUR[c], marker=C.MARKER[c],
                        ms=size[int(r["speed"])], mec="k", mew=0.3, ls="none",
                        zorder=ctrl_z(c) + 1)
                ax.annotate(f"{r['speed']:g}×", (r["x"], r["y"]), xytext=(4, -1),
                            textcoords="offset points", fontsize=5.5, color=C.COLOUR[c],
                            ha="left", va="top")
            safe |= mark_safe_at(ax, d["x"], d["y"], list(d["safe"] > 0))
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log_y(ax)
        ax.set_xlabel(r"$\int\|\tau\|^2 dt$ in window [N$^2$m$^2$s]")
        ax.set_ylabel("MSE [mm²]")
        ax.set_title(shape, fontsize=7)
        panel_label(ax, f"({lab})")
    speeds = [plt.Line2D([], [], ls="none", marker="o", color="0.6", mec="k", mew=0.3,
                         ms=size[v], label=f"{v}× speed") for v in sorted(size)]
    fig.subplots_adjust(bottom=0.2)
    legend_below(fig, ctrl_handles() + speeds + ([safe_handle()] if safe else []), ncol=5)
    save(fig, "fig28_live_pareto")


def fig_live_estimate() -> None:
    """The base estimate's error against time, raw flight sensors against the
    EKF, SQP-NMPC on the live rectangle (one repeat each, the median one)."""
    est = derived("estimate.csv")
    if est is None:
        return skip("live_estimate", NO_DERIVED)
    fig, axes = plt.subplots(3, 1, figsize=(FULL, 3.8), sharex=True,
                             gridspec_kw={"hspace": 0.3})
    styles = {"raw": dict(color=EST_SHADE["raw"], lw=0.8),
              "ekf": dict(color=EST_SHADE["ekf"], lw=0.9)}
    names = {"raw": "raw sensors (last 5 Hz fix, held)", "ekf": "EKF (500 Hz IMU, fixes fused)"}
    handles = []
    for src in ("raw", "ekf"):
        s = est[est["source"] == src]
        if s.empty:
            continue
        yaw = np.remainder(s["est_yaw"].to_numpy() + np.pi, 2 * np.pi) - np.pi
        for ax, v in zip(axes, (s["est_x"] * 1e3, s["est_y"] * 1e3, yaw * 1e3)):
            ax.plot(s["t_s"], v, zorder=3 if src == "ekf" else 2, **styles[src])
        handles.append(plt.Line2D([], [], **styles[src],
                                  label=f"{names[src]}: position error "
                                        f"{float(s['est_pos_rms_mm'].iloc[0]):.1f} mm rms"))
        w0 = float(s["window_start_s"].iloc[0])
    for ax, lab, p in zip(axes, ("x error [mm]", "y error [mm]", "yaw error [mrad]"), "abc"):
        ax.axhline(0, color="0.4", lw=0.5)
        ax.axvline(w0, **NOMINAL_STYLE)
        ax.set_ylabel(lab)
        panel_label(ax, f"({p})")
    axes[-1].set_xlabel("time [s]")
    axes[-1].set_xlim(0, float(est["t_s"].max()))
    axes[-1].set_xticks(np.arange(0, float(est["t_s"].max()) + 0.1, 5.0))
    fig.align_ylabels(axes)
    legend_below(fig, handles + [plt.Line2D([], [], label="scoring starts", **NOMINAL_STYLE)],
                 ncol=2)
    save(fig, "fig29_live_estimate")


def fig_live_base_yaw() -> None:
    """The base's yaw against time on the live nominal cells (partner held), one
    repeat per controller: what the arm's reaction does to the bus."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    drawn = False
    for ax, shape, lab in zip(axes, ("circle", "rectangle"), "ab"):
        tr = derived(f"track_{shape}.csv")
        if tr is None:
            continue
        for c in DRAW_ORDER:
            s = tr[tr["tag"] == c]
            if s.empty:
                continue
            yaw = np.unwrap(s["base_yaw"].to_numpy())
            ax.plot(s["t_s"], (yaw - yaw[0]) * 1e3, color=C.COLOUR[c], lw=ctrl_lw(c, 0.7),
                    zorder=ctrl_z(c))
            drawn = True
        ax.set_title(shape, fontsize=7)
        ax.set_xlabel("time [s]")
        panel_label(ax, f"({lab})")
    if not drawn:
        plt.close(fig)
        return skip("live_base_yaw", NO_DERIVED)
    axes[0].set_ylabel(r"base yaw $\psi - \psi_0$ [mrad]")
    fig.subplots_adjust(bottom=0.24)
    legend_below(fig, ctrl_handles(), ncol=4)
    save(fig, "fig31_live_base_yaw")


def fig_jetson_tick_pairs() -> None:
    """Each Orin Nano control tick against the same tick on the DGX (the runs are
    bit-identical, so they pair row by row), per controller: a 2-D histogram on
    log axes, the fitted ratio and the 10 ms period."""
    a, b = "dgx_n24", "orin_nano"
    h = derived(f"tick_pairs_{a}_{b}.csv", where="machines")
    st = derived(f"tick_pairs_{a}_{b}_stats.csv", where="machines")
    if h is None or st is None:
        return skip("jetson_tick_pairs", NO_DERIVED)
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.95), gridspec_kw={"wspace": 0.45})
    vmax = h["count"].max()
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    for ax, c, lab in zip(axes, C.CTRL, "abcd"):
        d = h[h["tag"] == c]
        s = st[st["tag"] == c]
        for _, r in d.iterrows():
            ax.add_patch(plt.Rectangle((r["a_lo_us"] / 1000, r["b_lo_us"] / 1000),
                                       (r["a_hi_us"] - r["a_lo_us"]) / 1000,
                                       (r["b_hi_us"] - r["b_lo_us"]) / 1000,
                                       color=plt.cm.Greys(0.25 + 0.75 * norm(r["count"])), lw=0))
        # The NMPCs' ticks are ms, the closed forms' us: each its own range, up
        # to past the period so the deadline is on every panel.
        lo = 0.2 if c in ("sqp", "iter") else 1e-3
        x = np.array([lo, 30.0])
        ax.plot(x, x, color="0.5", lw=0.6, ls="--")
        if len(s):
            k = float(s["ratio_gmean"].iloc[0])
            ax.plot(x, k * x, color=C.COLOUR[c], lw=1.0)
            mx_a, mx_b = float(s[f"{a}_max_us"].iloc[0]), float(s[f"{b}_max_us"].iloc[0])
            ax.text(0.97, 0.12, f"Nano/DGX ×{k:.2f}\nNano worst {mx_b / 1000:.1f} ms",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                    color=C.COLOUR[c])
            ax.plot(mx_a / 1000, 0.0, marker="^", ms=4, color=C.COLOUR[c], ls="none",
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
            ax.plot(0.0, mx_b / 1000, marker=">", ms=4, color=C.COLOUR[c], ls="none",
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
        ax.axhline(C.PERIOD_US / 1000, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, 30)
        ax.set_ylim(lo, 30)
        ax.set_aspect("equal")
        ax.set_title(C.LABEL[c], fontsize=7)
        ax.set_xlabel("DGX tick [ms]")
        panel_label(ax, f"({lab})")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes:
        ax.set_ylabel("Orin Nano tick [ms]", labelpad=1)
    fig.subplots_adjust(bottom=0.34, top=0.87)
    handles = [plt.Line2D([], [], color="0.3", lw=1.0, label="geometric-mean ratio"),
               plt.Line2D([], [], color="0.5", lw=0.6, ls="--", label="same speed"),
               plt.Line2D([], [], color="k", ls=":", lw=0.8, label="10 ms period"),
               plt.Line2D([], [], ls="none", marker="^", color="0.3", ms=4,
                          label="worst tick (on each axis)"),
               plt.Rectangle((0, 0), 1, 1, color=plt.cm.Greys(0.7), label="ticks (log count)")]
    # Square panels shrink to fit the width: hang the legend from where they end.
    fig.canvas.draw()
    y0 = min(ax.get_position().y0 for ax in axes)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, y0 - 0.42 / fig.get_figheight()))
    save(fig, "fig30_jetson_tick_pairs")


def fig_robustness_bench() -> None:
    """fig08's panels that have no live equivalent: an unmodelled joint torque,
    the sensor noise and the state delay (bench, rectangle, flight sensing for
    the last two), so the live fig18 and this cover the conditions between them."""
    panels = []
    for study, xl, col, nominal in (("disturbance", "unmodelled joint torque [N m]",
                                     "disturb_torque", 0.0),
                                    ("noise", "flight sensor error (× profile)", "noise_scale",
                                     1.0),
                                    ("delay", "extra state delay [ticks]", "delay_ticks", 0.0)):
        df = C.summary(study)
        if df is not None:
            df = df[df["scenario"] == SCORE_SHAPE]
            if not df.empty:
                panels.append((xl, df, col, nominal))
    if not panels:
        return skip("robustness_bench", "no disturbance, noise or delay study")
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL, 2.3), squeeze=False,
                             gridspec_kw={"wspace": 0.32})
    safe = False
    for ax, (xl, df, x, nominal), lab in zip(axes[0], panels, "abc"):
        safe |= _seeded(ax, df, x)
        mark_nominal(ax, nominal)
        ax.set_yscale("log")
        plain_log_y(ax)
        ax.set_xlabel(xl)
        panel_label(ax, f"({lab})")
    axes[0, 0].set_ylabel(f"{SCORE_SHAPE} RMSE [mm]")
    handles = ctrl_handles() + [nominal_handle()] + ([safe_handle()] if safe else [])
    fig.subplots_adjust(bottom=0.26)
    legend_below(fig, handles, ncol=len(handles))
    save(fig, "fig08_robustness_bench")


FIGURES: Dict[str, Callable[[], None]] = {
    "step": fig_step, "circle": fig_circle, "rectangle": fig_rectangle,
    "dashboard": fig_dashboard, "pareto": fig_pareto, "base": fig_base,
    "speed": fig_speed, "robustness": fig_robustness, "sensitivity": fig_sensitivity,
    "timing": fig_timing, "constraint": fig_constraint,
    "observer": fig_observer, "ffgain": fig_ffgain, "warmup": fig_warmup,
    "estimator": fig_estimator, "tuning": fig_tuning,
    "live_nominal": fig_live_nominal, "live_robustness": fig_live_robustness,
    "live_estimator": fig_live_estimator, "live_timing": fig_live_timing,
    "live_trace": fig_live_trace, "live_cap": fig_live_cap,
    "live_exposed": fig_live_exposed, "live_circle": fig_live_circle,
    "live_rectangle": fig_live_rectangle, "live_twin_agreement": fig_live_twin_agreement,
    "live_pareto": fig_live_pareto, "live_estimate": fig_live_estimate,
    "jetson_tick_pairs": fig_jetson_tick_pairs, "live_base_yaw": fig_live_base_yaw,
    "robustness_bench": fig_robustness_bench,
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
