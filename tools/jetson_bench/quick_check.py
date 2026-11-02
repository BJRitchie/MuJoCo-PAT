#!/usr/bin/env python3
"""The verdict, on the spot: does every NMPC's worst tick fit the period on this
machine, and if not, which QP cap would? Standard library only, so it runs in the
image on the Jetson, right after the suite (compare.py, with the plots, needs the
host's plotting venv):

    tools/jetson_bench/enter.sh python3 tools/jetson_bench/quick_check.py orin_nx
"""
import csv
import json
import os
import re
import sys

PERIOD_US = 10_000.0
BUDGET_US = 8_000.0          # the cap must fit this; the rest is margin
LABEL = {"sqp": "SQP-NMPC", "iter": "LMPC", "ompc": "OMPC", "smc": "SMC"}


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.isfile(path) else []


def cap_of(manifest, tag):
    for c in manifest.get("cells", []):
        if c.get("tag") == tag:
            for a in c.get("args", []):
                m = re.match(r"qp\.max_iter=(\d+)", a)
                if m:
                    return int(m.group(1))
    return None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "orin_nx"
    root = os.path.join("data", "thesis_eval", "machines", name)
    jt = rows(os.path.join(root, "jetson_timing", "summary.csv"))
    wc = rows(os.path.join(root, "worst_case", "summary.csv"))
    man_p = os.path.join(root, "worst_case", "manifest.json")
    man = json.load(open(man_p)) if os.path.isfile(man_p) else {}
    if not jt and not wc:
        sys.exit(f"no results under {root}: run run_suite.sh {name} first")
    print(f"{name}: the worst control tick, against the {PERIOD_US / 1000:g} ms period\n")
    print("  the scenarios (jetson_timing):")
    for tag in LABEL:
        d = [r for r in jt if r["tag"] == tag]
        if d:
            worst = max(float(r["tick_us_max"]) for r in d)
            p99 = sum(float(r["tick_us_p99"]) for r in d) / len(d)
            print(f"    {LABEL[tag]:9s} worst {worst / 1000:6.2f} ms, mean p99 {p99 / 1000:5.2f} ms"
                  f"  {'OK' if worst <= PERIOD_US else 'OVER'}")
    print("\n  every QP forced to its cap, twice per tick (worst_case):")
    advice = []
    for tag in ("sqp", "iter"):
        d = [r for r in wc if r["tag"] == tag]
        if not d:
            continue
        cap = cap_of(man, tag)
        iters = sum(float(r["qp_iter_mean"]) for r in d) / len(d)
        worst = max(float(r["tick_us_max"]) for r in d)
        per_iter = worst / iters if iters else float("nan")
        fit = int(BUDGET_US // (2 * per_iter)) if per_iter > 0 else None
        full = cap and iters >= 0.95 * 2 * cap
        ok = worst <= PERIOD_US
        print(f"    {LABEL[tag]:9s} cap {cap}: worst {worst / 1000:6.2f} ms  "
              f"{'OK' if ok else 'OVER'}; {per_iter:4.0f} us per iteration, "
              f"{iters:4.1f} iterations per tick{'' if full else ' (NOT the full path)'}; "
              f"the largest cap inside {BUDGET_US / 1000:g} ms: {fit}")
        if not ok or (fit is not None and cap and fit < cap):
            advice.append((tag, fit))
    if advice:
        caps = dict(advice)
        print("\n  The current caps do not fit this machine. To rerun with ones that do:")
        print(f"    tools/jetson_bench/set_caps.sh {caps.get('sqp', 'keep')} "
              f"{caps.get('iter', 'keep')}")
        print(f"    tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh "
              f"{name}_cap{caps.get('sqp', 'k')}_{caps.get('iter', 'k')}")
    else:
        print("\n  Every NMPC's worst case fits the period with the current caps.")


if __name__ == "__main__":
    main()
