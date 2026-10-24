#!/usr/bin/env python3
"""Tune each controller for the comparison, by the same rule for all four.

Run inside the dev container, like run_study.py:

    python3 tools/thesis_eval/tune.py sqp iter ompc smc --jobs 3
    python3 tools/thesis_eval/tune.py smc --passes 4

The rule (decided for the thesis, the same for every controller):

    minimise   the GEOMETRIC mean of the EE RMSE over the circle and the
               rectangle at 1x and 2x speed and the 6 cm step, exact state, the
               other arm held (the TUNING scenarios), with the NMPC horizon held
               at N >= 20. Geometric, so each scenario counts by its relative
               change: the step's RMSE is ~30 mm, most of it the unavoidable
               initial 85 mm error, against ~1 mm on the circle, and an
               arithmetic mean would tune for the step alone.
    subject to in every tuning scenario: under 5% of joint-ticks at the torque
               limit, no safe-mode entry, and no control tick over the 10 ms
               period

A setting that breaks a limit scores 1000 + how far it breaks them, so the search
first walks to feasibility and only then minimises error. Every other study
(3x and 4x speed, noise, estimator, disturbance, model error, mirror, timing) is
NOT used here: it is the test set the tuned settings are judged on afterwards.

The search is coordinate descent from the shipped values: one knob at a time,
each tried at a ladder of multiples of its current value, the best move kept,
repeated over all knobs until a pass changes nothing (or --passes runs out).
Every evaluation is three cached contact_bench runs (run_study.py's cache), so a
rerun or an extra pass re-uses everything already measured.

Writes data/thesis_eval/tuning/<controller>.json (every evaluation) and updates
tools/thesis_eval/tuned.json, the tuned `--set` overrides the studies run as the
`tuned` variant.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_study as R  # noqa: E402
import studies  # noqa: E402

#: The shapes at the planner's default speed AND at twice it: tuning on the
#: default speed alone found short horizons (N = 12) that tracked it superbly and
#: then entered safe mode at 2x, where the shipped settings had not.
#: Each on the studies' frictionless plant, so the settings are tuned for the
#: conditions they are tested in.
TUNING_SCENARIOS = tuple(
    (name, tuple(args) + studies.PLANT) for name, args in (
        ("circle", studies.CIRCLE), ("rectangle", studies.RECTANGLE),
        ("step6", studies.STEP_SMALL),
        ("circle2x", tuple(studies.CIRCLE) + ("--traj-speed", "2")),
        ("rectangle2x", tuple(studies.RECTANGLE) + ("--traj-speed", "2"))))
RAIL_MAX = 0.05
TICK_MAX_US = 10_000.0
MULTS = (0.25, 0.5, 0.7, 1.4, 2.0, 4.0)
#: While the best setting so far breaks a limit, the steps are orders of magnitude:
#: a gain that is 1000x too large (the SMC's robustness term, shipped) cannot be
#: walked back by factors of 4 when every step in between still breaks the limit.
MULTS_WIDE = (1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 100.0)
#: The NMPC's look-ahead, N * Ts, may not fall below this [s]: bounding N alone
#: would let the search shorten the horizon through Ts instead. 20 steps at the
#: shipped 33 ms.
MIN_HORIZON_S = 0.66
#: Knobs for which exactly zero is a meaningful setting (it switches a term off),
#: and is offered as a candidate alongside the multiples.
ZERO_OK = {"k2"}
#: Optional ceiling on the integrated squared torque, per tuning scenario
#: (--torque-cap-from-others): exceeding it is a violation like the others, so
#: the search finds the most accurate setting that is no harder on the actuators
#: than the other controllers are on average.
TORQUE_CAP: Dict[str, float] = {}
#: Optional ceiling on the RMSE, per tuning scenario (--beat-others): the best
#: of the other controllers' tuned RMSE there, so a feasible setting is at least
#: as accurate as every other controller in every tuning scenario.
RMSE_CAP: Dict[str, float] = {}
TUNED_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


def vec(x: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.6g}" for v in x) + "]"


# ---------------------------------------------------------- the knobs --
# Each controller: its law, any fixed arguments, the knobs with their shipped
# values and bounds, and how a knob setting becomes --set overrides. Vectors are
# [x, y, thetaz]; x and y are tuned together (the task is symmetric in them),
# thetaz on its own.

def nmpc_space(full_nonlinear: bool):
    fixed = () if full_nonlinear else ("--set", "nmpc.full_nonlinear=false")
    start = {"qpos_xy": 10.0, "qpos_th": 1.0, "qvel_xy": 1.0, "qvel_th": 3.0,
             "r": 0.1, "N": 30, "Ts": 0.033, "term": 100.0}
    bounds = {"qpos_xy": (0.1, 1e4), "qpos_th": (0.01, 1e3), "qvel_xy": (0.01, 1e3),
              # N at least 20: with Ts fixed near the shipped 33 ms that is a
              # horizon of at least ~0.66 s, the look-ahead the faster shapes need.
              "qvel_th": (0.01, 1e3), "r": (1e-4, 10.0), "N": (20, 60), "Ts": (0.01, 0.2),
              "term": (1.0, 1e4)}

    def overrides(s: Dict[str, float]) -> List[str]:
        return ["--set", f"weights.Q_pos={vec([s['qpos_xy'], s['qpos_xy'], s['qpos_th']])}",
                "--set", f"weights.Q_vel={vec([s['qvel_xy'], s['qvel_xy'], s['qvel_th']])}",
                "--set", f"weights.R={vec([s['r']] * 3)}",
                "--set", f"nmpc.N={int(s['N'])}",
                "--set", f"nmpc.Ts={s['Ts']:.6g}",
                "--set", f"nmpc.terminal_cost_multiplier={s['term']:.6g}"]
    return "nmpc", fixed, start, bounds, overrides


def ompc_space():
    start = {"Ts": 1.0, "q_xy": 1e5, "q_th": 1e5}
    bounds = {"Ts": (0.05, 10.0), "q_xy": (1.0, 1e9), "q_th": (1.0, 1e9)}

    def overrides(s):
        return ["--set", f"ompc.Ts={s['Ts']:.6g}",
                "--set", f"ompc.Q={vec([s['q_xy'], s['q_xy'], s['q_th']])}"]
    return "ompc", (), start, bounds, overrides


def smc_space():
    start = {"lam_xy": 3.0, "lam_th": 3.0, "alpha": 0.5, "epsilon": 0.05,
             "tau_bound": 0.25, "k2": 1.3}
    bounds = {"lam_xy": (0.1, 50.0), "lam_th": (0.1, 50.0), "alpha": (1e-3, 10.0),
              "epsilon": (1e-3, 5.0), "tau_bound": (1e-6, 1.0), "k2": (0.0, 10.0)}

    def overrides(s):
        return ["--set", f"smc.lam={vec([s['lam_xy'], s['lam_xy'], s['lam_th']])}",
                "--set", f"smc.alpha={vec([s['alpha']] * 3)}",
                "--set", f"smc.epsilon={s['epsilon']:.6g}",
                "--set", f"smc.tau_bound={s['tau_bound']:.6g}",
                "--set", f"smc.k2={s['k2']:.6g}"]
    return "smc", (), start, bounds, overrides


SPACES = {"sqp": lambda: nmpc_space(True), "iter": lambda: nmpc_space(False),
          "ompc": ompc_space, "smc": smc_space}


# ---------------------------------------------------------- evaluation --

class Evaluator:
    def __init__(self, ctrl: str, jobs: int):
        self.ctrl = ctrl
        self.law, self.fixed, self.start, self.bounds, self.overrides = SPACES[ctrl]()
        self.ctx = R.Context(R.find_bench())
        self.pool = cf.ThreadPoolExecutor(max_workers=max(1, jobs))
        self.log: List[dict] = []

    def cells(self, s: Dict[str, float]) -> List[studies.Cell]:
        args = tuple(self.fixed) + tuple(self.overrides(s))
        h = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:10]
        return [studies.Cell("tuning", f"{self.ctrl}_{h}_{scen}", self.law, self.ctrl,
                             tuple(sargs) + args) for scen, sargs in TUNING_SCENARIOS]

    def evaluate(self, s: Dict[str, float]) -> Tuple[float, dict]:
        cells = self.cells(s)
        recs = list(self.pool.map(lambda c: R.run_cell(c, self.ctx, False, None), cells))
        per = {}
        for (scen, _), c, rec in zip(TUNING_SCENARIOS, cells, recs):
            if rec["status"] == "failed":
                per[scen] = None
                continue
            with open(R.cell_paths(c, self.ctx)[1], newline="") as fh:
                row = next(csv.DictReader(fh))
            per[scen] = {k: float(row[k]) for k in
                         ("rmse_mm", "rail_frac", "safe_ticks", "tick_us_max",
                          "tau2_int_win", "err_var_mm2")}
        violation = 0.0
        for scen, m in per.items():
            if m is None:
                violation += 10.0
                continue
            violation += max(0.0, m["rail_frac"] - RAIL_MAX) * 10.0
            violation += min(m["safe_ticks"], 1000.0) / 100.0
            violation += max(0.0, m["tick_us_max"] - TICK_MAX_US) / TICK_MAX_US
            if scen in TORQUE_CAP:
                violation += max(0.0, m["tau2_int_win"] / TORQUE_CAP[scen] - 1.0) * 10.0
            if scen in RMSE_CAP:
                violation += max(0.0, m["rmse_mm"] / RMSE_CAP[scen] - 1.0) * 10.0
        rmse = [m["rmse_mm"] for m in per.values() if m is not None]
        mean = sum(rmse) / len(rmse) if rmse else math.inf
        gmean = math.exp(sum(math.log(max(r, 1e-9)) for r in rmse) / len(rmse)) \
            if len(rmse) == len(per) else math.inf
        score = gmean if violation == 0.0 else 1000.0 + violation
        entry = {"state": dict(s), "score": score, "gmean_rmse_mm": gmean,
                 "mean_rmse_mm": mean, "feasible": violation == 0.0,
                 "violation": violation, "per": per}
        self.log.append(entry)
        return score, entry

    def candidates(self, s: Dict[str, float], knob: str,
                   wide: bool = False) -> List[Dict[str, float]]:
        lo, hi = self.bounds[knob]
        out = []
        if knob == "N":
            values = sorted({max(lo, min(hi, int(round(s[knob] * m)))) for m in
                             (0.5, 0.67, 0.8, 1.25, 1.5, 2.0)} - {int(s[knob])})
        elif s[knob] == 0.0:
            # Switched off: try switching it back on at a few scales of its bound.
            values = [hi * f for f in (1e-3, 1e-2, 0.1)]
        else:
            values = [min(hi, max(lo, s[knob] * m)) for m in (MULTS_WIDE if wide else MULTS)]
            if knob in ZERO_OK:
                values.append(0.0)
            values = sorted({v for v in values if not math.isclose(v, s[knob])})
        for v in values:
            c = dict(s)
            c[knob] = v
            if "N" in c and "Ts" in c and c["N"] * c["Ts"] < MIN_HORIZON_S - 1e-9:
                continue
            out.append(c)
        return out


def clamp_start(ev: "Evaluator", s: Dict[str, float]) -> Dict[str, float]:
    """A start point inside the current bounds, including the horizon length."""
    out = {}
    for k, v in s.items():
        lo, hi = ev.bounds[k]
        out[k] = min(hi, max(lo, v))
    if "N" in out and "Ts" in out and out["N"] * out["Ts"] < MIN_HORIZON_S:
        out["N"] = math.ceil(MIN_HORIZON_S / out["Ts"])
    return out


def tune(ctrl: str, passes: int, jobs: int, start: Dict[str, float] = None) -> dict:
    """Coordinate descent from `start` (the shipped values by default). The
    shipped point is always evaluated too, as the reference."""
    ev = Evaluator(ctrl, jobs)
    _, shipped = ev.evaluate(dict(ev.start))
    state = clamp_start(ev, start) if start else dict(ev.start)
    best_score, best = ev.evaluate(state)
    if start:
        print(f"{ctrl}: start from {state}", flush=True)
    print(f"{ctrl}: shipped  score {shipped['score']:.4g}  gmean RMSE "
          f"{shipped['gmean_rmse_mm']:.3f} mm feasible={shipped['feasible']}; start score "
          f"{best_score:.4g}", flush=True)
    for p in range(1, passes + 1):
        moved = False
        for knob in ev.start:
            cand = ev.candidates(state, knob, wide=not best["feasible"])
            results = [ev.evaluate(c) for c in cand]
            score, entry = min(results, key=lambda r: r[0]) if results else (math.inf, None)
            if entry is not None and score < best_score * (1.0 - 1e-3):
                state, best_score, best, moved = entry["state"], score, entry, True
                print(f"  pass {p} {knob:9s} -> {state[knob]:.4g}   score {score:.4g}  "
                      f"gmean RMSE {entry['gmean_rmse_mm']:.3f}  feasible={entry['feasible']}",
                      flush=True)
        if not moved:
            print(f"  pass {p}: no knob improved; converged", flush=True)
            break
    out = {"controller": ctrl, "law": ev.law, "start": state if start else "shipped",
           "rule": {
               "scenarios": [s for s, _ in TUNING_SCENARIOS], "rail_max": RAIL_MAX,
               "tick_max_us": TICK_MAX_US, "tau2_cap": dict(TORQUE_CAP),
               "rmse_cap": dict(RMSE_CAP),
               "objective": "geometric mean of the scenarios' RMSE [mm]"},
           "shipped": shipped, "tuned": best,
           "args": list(ev.fixed) + ev.overrides(best["state"]),
           "evaluations": ev.log}
    d = os.path.join(R.DATA, "tuning")
    os.makedirs(d, exist_ok=True)
    tag = "" if not start else "_restart"
    with open(os.path.join(d, f"{ctrl}{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("controllers", nargs="+", choices=list(SPACES))
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--start-from", default=None,
                    help="a tuned.json whose state for each controller is a SECOND "
                         "start point (clamped into the current bounds). The result "
                         "replaces the one in tuned.json only if it scores better.")
    ap.add_argument("--torque-cap-from-others", choices=("mean", "max"), default=None,
                    help="cap each tuning scenario's integrated squared torque at the "
                         "mean (or the largest) of the OTHER controllers' tuned values in "
                         "tuned.json. The objective changes, so the result always "
                         "replaces the controller's entry.")
    ap.add_argument("--beat-others", action="store_true",
                    help="require each tuning scenario's RMSE to be no more than the "
                         "best of the OTHER controllers' tuned RMSE in tuned.json")
    args = ap.parse_args(argv)
    tuned = {}
    if os.path.isfile(TUNED_JSON):
        with open(TUNED_JSON) as fh:
            tuned = json.load(fh)
    starts = {}
    if args.start_from:
        with open(args.start_from) as fh:
            starts = {c: v["state"] for c, v in json.load(fh).items()}
    for c in args.controllers:
        TORQUE_CAP.clear()
        RMSE_CAP.clear()
        others = [v["per_scenario"]["tuned"] for k, v in tuned.items() if k != c]
        if args.torque_cap_from_others:
            for scen, _ in TUNING_SCENARIOS:
                vals = [o[scen]["tau2_int_win"] for o in others]
                TORQUE_CAP[scen] = (max(vals) if args.torque_cap_from_others == "max"
                                    else sum(vals) / len(vals))
            print(f"{c}: torque cap {({k: round(v, 4) for k, v in TORQUE_CAP.items()})}",
                  flush=True)
        if args.beat_others:
            for scen, _ in TUNING_SCENARIOS:
                RMSE_CAP[scen] = min(o[scen]["rmse_mm"] for o in others)
            print(f"{c}: RMSE cap {({k: round(v, 3) for k, v in RMSE_CAP.items()})}",
                  flush=True)
        out = tune(c, args.passes, args.jobs, starts.get(c))
        prev = tuned.get(c)
        changed = args.torque_cap_from_others or args.beat_others
        if args.start_from and prev is not None and not changed:
            prev_score = prev["gmean_rmse_mm"]["tuned"] if prev["feasible"] else math.inf
            new_score = out["tuned"]["score"] if out["tuned"]["feasible"] else math.inf
            if new_score >= prev_score:
                print(f"{c}: restart reached {new_score:.4g}, no better than the "
                      f"{prev_score:.4g} already in tuned.json; kept that")
                continue
        tuned[c] = {"law": out["law"], "args": out["args"], "state": out["tuned"]["state"],
                    "feasible": out["tuned"]["feasible"],
                    "gmean_rmse_mm": {"shipped": out["shipped"]["gmean_rmse_mm"],
                                      "tuned": out["tuned"]["gmean_rmse_mm"]},
                    "per_scenario": {"shipped": out["shipped"]["per"],
                                     "tuned": out["tuned"]["per"]},
                    "evaluations": len(out["evaluations"])}
        with open(TUNED_JSON, "w") as fh:
            json.dump(tuned, fh, indent=1)
        print(f"{c}: tuned gmean RMSE {out['tuned']['gmean_rmse_mm']:.3f} mm "
              f"(shipped {out['shipped']['gmean_rmse_mm']:.3f}), "
              f"feasible={out['tuned']['feasible']}, {len(out['evaluations'])} evaluations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
