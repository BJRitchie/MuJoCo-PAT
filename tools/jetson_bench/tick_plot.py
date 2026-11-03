#!/usr/bin/env python3
"""The control-tick distribution of one or more machine runs, drawn the way
fig10_timing's left panel draws the DGX's: per controller, the tick histogram as a
density on log axes, the worst tick recorded (a marker on the top edge: a
histogram's tail is too thin to see at one tick in 10^4), and the 10 ms period.
One panel per run, on shared axes, so runs can be read against each other.

    ~/.venvs/thesis_eval/bin/python tools/jetson_bench/tick_plot.py orin_nano orin_nano_cold_cap17_18
    ~/.venvs/thesis_eval/bin/python tools/jetson_bench/tick_plot.py orin_nano_cold_cap17_18 --study timing

--study jetson_timing (default): all 11 scenarios x 3 seeds, the hard ones
included, from the run's tick_hist.json (in git, so this works on any machine
with a clone) or its per-tick records when they are here.
--study timing: fig10's own cells, the 1x circle at the tuned settings; needs
the per-tick records (series/), which stay on the machine that ran it and
whatever it was copied to (the DGX's Jetson clone).

Writes docs/assets/thesis/fig_ticks_<study>_<run>[_vs_<run>...].{pdf,png}.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "thesis_eval"))
import compare as K  # noqa: E402  (brings figures as K.F, common as K.C)

F, C, plt = K.F, K.C, K.plt


def hists_of(run: str, study: str) -> dict:
    """{tag: histogram} for one run: from its per-tick records when here, else
    (jetson_timing only) its saved tick_hist.json."""
    summ, _ = K.load(run, study)
    out = {}
    if summ is not None:
        for tag in C.CTRL:
            cells = summ[summ["tag"] == tag]["cell"]
            if study == "timing":
                cells = cells[~cells.str.contains("_N")]   # fig10's cells, not the sweep
            ticks = [s["tick_us"].to_numpy() for s in
                     (K.series(run, study, c) for c in cells) if s is not None]
            if ticks:
                out[tag] = C.tick_hist_of(np.concatenate(ticks))
    if not out and study == "jetson_timing":
        out = C.read_tick_hist(K.hist_path(run)) or {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="machine runs (data/thesis_eval/machines/<run>)")
    ap.add_argument("--study", choices=("jetson_timing", "timing"), default="jetson_timing")
    a = ap.parse_args()
    C.set_variant("tuned")

    data = {r: hists_of(r, a.study) for r in a.runs}
    missing = [r for r, h in data.items() if not h]
    if missing:
        sys.exit(f"no tick data for {', '.join(missing)} ({a.study}): "
                 + ("the per-tick records are not on this machine"
                    if a.study == "timing" else "no tick_hist.json or per-tick records"))

    n = len(a.runs)
    fig, axes = plt.subplots(1, n, figsize=(F.FULL if n > 1 else F.FULL / 2, 2.8),
                             sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    axes = axes[0]
    for ax, run in zip(axes, a.runs):
        _, man = K.load(run, a.study)
        for tag in C.CTRL:
            h = data[run].get(tag)
            if not h or not h["n"]:
                continue
            C.stairs_tick_hist(ax, h, color=C.COLOUR[tag])
            ax.plot(h["max"], 1.0, marker="v", color=C.COLOUR[tag], ms=4, ls="none",
                    transform=ax.get_xaxis_transform(), clip_on=False)
            over = int(h["counts"][C.hist_edges()[:-1] >= C.PERIOD_US].sum())
            if over:
                ax.text(0.98, 0.95 - 0.08 * C.CTRL.index(tag),
                        f"{C.LABEL[tag]}: {over} over", color=C.COLOUR[tag],
                        ha="right", va="top", fontsize=5.5, transform=ax.transAxes)
        ax.axvline(C.PERIOD_US, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5, 3e4)
        ax.set_xlabel("control tick [µs] (▼ worst recorded)")
        ax.set_title(f"{run.replace('_', ' ')}\n{K.qp_mode(man)}", fontsize=6)
    axes[0].set_ylabel("density")
    fig.subplots_adjust(bottom=0.24)
    F.legend_below(fig, [plt.Line2D([], [], color=C.COLOUR[c], marker=C.MARKER[c], lw=1.1,
                                    ms=4, label=C.LABEL[c]) for c in C.CTRL] +
                   [plt.Line2D([], [], color="k", ls=":", lw=0.8, label="10 ms period")],
                   ncol=5)
    F.save(fig, f"fig_ticks_{a.study}_" + "_vs_".join(a.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
