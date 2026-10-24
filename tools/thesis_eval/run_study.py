#!/usr/bin/env python3
"""Run the thesis comparison: one `contact_bench` process per cell.

Run this INSIDE the dev container, from the workspace root, with the overlay
sourced (the bench finds the scene and the estimator's files on
AMENT_PREFIX_PATH):

    docker compose exec dev bash
    source install/setup.bash
    python3 tools/thesis_eval/run_study.py list
    python3 tools/thesis_eval/run_study.py run nominal balanced --jobs 2
    python3 tools/thesis_eval/run_study.py run all --quick --dry-run
    python3 tools/thesis_eval/run_study.py run timing --pin 4,5
    python3 tools/thesis_eval/run_study.py status

Standard library only, on purpose: the dev image's matplotlib is broken, and the
figures are made on the host from the CSVs this writes (figures.py).

What it writes, under data/thesis_eval/<study>/:

    cells/<hash>.csv    one summary row per run: the cache. A cell is re-run only
                        if its command line, the bench binary, or any config file
                        the bench reads has changed (or with --force).
    summary.csv         every cell of the study in the study's own order, one row
                        each (tracked in git; small)
    series/<name>.csv   the per-tick record, for cells that ask for it (large,
                        not tracked)
    manifest.json       what produced the numbers: git revision and whether the
                        tree was dirty, the binary's hash, the machine's CPU
                        governor, power source and clock, the exact command of
                        every cell, how long each took, and any that failed

Timing cells are the one measurement that is not deterministic, so they run
serially, optionally pinned to cores with --pin, and are refused unless the
machine is on mains with the `performance` governor (--allow-noisy-timing
overrides, and the manifest then says so).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studies  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "thesis_eval")


# ------------------------------------------------------------------ files --

def find_bench() -> str:
    """The installed contact_bench, found the way the bench finds its own files."""
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        cand = os.path.join(prefix, "lib", "pat_arm_control", "contact_bench")
        if prefix and os.path.isfile(cand):
            return cand
    sys.exit("contact_bench not found on AMENT_PREFIX_PATH: source install/setup.bash "
             "(inside the dev container) after building pat_arm_control")


def config_files() -> List[str]:
    """Every installed file whose contents change what a run measures."""
    out: List[str] = []
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        for pkg, pattern in (("pat_arm_control", "config/*.yaml"),
                             ("pat_simulation", "config/*.yaml"),
                             ("pat_gnc", "config/nav*.yaml")):
            out += glob.glob(os.path.join(prefix, "share", pkg, pattern))
        out += glob.glob(os.path.join(prefix, "share", "pat_simulation", "models",
                                      "environment", "air_bearing_table.xml"))
        out += glob.glob(os.path.join(prefix, "share", "pat_arm_control", "models",
                                      "pat_platform_planar.xml"))
    return sorted(set(out))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Context:
    """What is the same for every cell of a run, and so goes into every cache key."""

    def __init__(self, bench: str):
        self.bench = bench
        self.bench_sha = sha256(bench)
        self.config_sha = hashlib.sha256(
            b"".join(open(p, "rb").read() for p in config_files())).hexdigest()

    def cell_hash(self, cell: studies.Cell) -> str:
        h = hashlib.sha256()
        for part in (self.bench_sha, self.config_sha, cell.law, cell.tag,
                     str(cell.seed), "\0".join(cell.args)):
            h.update(part.encode())
            h.update(b"\x01")
        return h.hexdigest()[:16]


# ---------------------------------------------------------------- machine --

def read_first(pattern: str) -> Optional[str]:
    for p in sorted(glob.glob(pattern)):
        try:
            return open(p).read().strip()
        except OSError:
            continue
    return None


def machine_state() -> Dict[str, object]:
    governors = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        try:
            governors.add(open(p).read().strip())
        except OSError:
            pass
    ac = None
    for p in glob.glob("/sys/class/power_supply/A[CD]*/online"):
        try:
            ac = open(p).read().strip() == "1"
        except OSError:
            pass
    watts = None
    for d in glob.glob("/sys/class/power_supply/ucsi-source-psy-*/"):
        try:
            if open(d + "online").read().strip() == "1":
                v = int(open(d + "voltage_max").read())
                i = int(open(d + "current_max").read())
                if v > 0 and i > 0:
                    watts = (v // 1_000_000) * (i // 1_000_000)
                    break
        except (OSError, ValueError):
            pass
    return {
        "hostname": platform.node(),
        "cpu_governor": ",".join(sorted(governors)) or "unknown",
        "on_mains": ac,
        "usb_pd_watts": watts,
        "python": platform.python_version(),
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def git_state() -> Dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                              text=True).stdout.strip()
    return {"revision": run("rev-parse", "HEAD"), "branch": run("rev-parse",
            "--abbrev-ref", "HEAD"), "dirty": bool(run("status", "--porcelain",
            "--untracked-files=no"))}


def timing_allowed(state: Dict[str, object], allow_noisy: bool) -> Optional[str]:
    """None if timing cells may run, else why they may not."""
    problems = []
    if state["cpu_governor"] != "performance":
        problems.append(f"CPU governor is {state['cpu_governor']!r}, not performance")
    if state["on_mains"] is False:
        problems.append("the machine is on battery")
    if state["usb_pd_watts"] is not None and state["usb_pd_watts"] < 60:
        problems.append(f"the USB-C supply negotiated only {state['usb_pd_watts']} W")
    if problems and not allow_noisy:
        return "; ".join(problems)
    return None


# -------------------------------------------------------------------- run --

def study_dir(study: str) -> str:
    """data/thesis_eval/<study> for the shipped settings, .../tuned/<study> for
    the tuned ones, so the two never share a summary."""
    if studies._variant == "tuned" and study != "tuning":
        return os.path.join(DATA, "tuned", study)
    return os.path.join(DATA, study)


def cell_paths(cell: studies.Cell, ctx: Context):
    h = ctx.cell_hash(cell)
    d = study_dir(cell.study)
    return (h, os.path.join(d, "cells", h + ".csv"),
            os.path.join(d, "series", cell.name + ".csv"))


def command(cell: studies.Cell, ctx: Context, out_csv: str, series_csv: str,
            pin: Optional[str]) -> List[str]:
    cmd = [ctx.bench, "--law", cell.law, *cell.args, "--tag", cell.tag,
           "--seed", str(cell.seed), "--summary-csv", out_csv]
    if cell.series:
        cmd += ["--series", series_csv]
    if cell.timing and pin:
        cmd = ["taskset", "-c", pin, *cmd]
    return cmd


def run_cell(cell: studies.Cell, ctx: Context, force: bool, pin: Optional[str]) -> dict:
    h, out_csv, series_csv = cell_paths(cell, ctx)
    rec = {"name": cell.name, "study": cell.study, "hash": h, "tag": cell.tag,
           "seed": cell.seed, "law": cell.law, "args": list(cell.args)}
    if os.path.isfile(out_csv) and not force and (
            not cell.series or os.path.isfile(series_csv)):
        rec.update(status="cached")
        return rec
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if cell.series:
        os.makedirs(os.path.dirname(series_csv), exist_ok=True)
    tmp = out_csv + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    cmd = command(cell, ctx, tmp, series_csv, pin)
    rec["command"] = cmd
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec["seconds"] = round(time.monotonic() - t0, 2)
    if proc.returncode != 0 or not os.path.isfile(tmp):
        rec.update(status="failed", returncode=proc.returncode,
                   stderr=proc.stderr.strip().splitlines()[-5:],
                   stdout=proc.stdout.strip().splitlines()[-3:])
        if os.path.exists(tmp):
            os.remove(tmp)
        return rec
    os.replace(tmp, out_csv)
    rec.update(status="ran")
    return rec


def assemble(study: str, cells: Sequence[studies.Cell], ctx: Context) -> int:
    """summary.csv: every cached cell of the study, in the study's order."""
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    for c in cells:
        _, out_csv, _ = cell_paths(c, ctx)
        if not os.path.isfile(out_csv):
            continue
        with open(out_csv, newline="") as fh:
            r = list(csv.reader(fh))
        if len(r) != 2:
            continue
        if header is None:
            header = r[0]
        elif r[0] != header:
            sys.exit(f"{study}: cell {c.name} has different columns from the rest; "
                     "re-run with --force after rebuilding")
        rows.append([c.name] + r[1])
    if header is None:
        return 0
    path = os.path.join(study_dir(study), "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell"] + header)
        w.writerows(rows)
    return len(rows)


def run_studies(names: Sequence[str], args: argparse.Namespace) -> int:
    ctx = Context(find_bench())
    state = machine_state()
    failures = 0
    for study in names:
        cells = studies.build(study, quick=args.quick)
        if args.select:
            cells = [c for c in cells if args.select in c.name]
        if not cells:
            print(f"{study}: nothing to run for the {studies._variant} variant")
            continue
        timing = any(c.timing for c in cells)
        if timing and not args.dry_run:
            why = timing_allowed(state, args.allow_noisy_timing)
            if why:
                print(f"{study}: REFUSED, timing needs a quiet machine: {why}. "
                      "Fix it, or pass --allow-noisy-timing (the manifest records it).",
                      file=sys.stderr)
                failures += 1
                continue
        todo = [c for c in cells if args.force or not os.path.isfile(cell_paths(c, ctx)[1])
                or (c.series and not os.path.isfile(cell_paths(c, ctx)[2]))]
        print(f"{study}: {len(cells)} cells, {len(cells) - len(todo)} cached, "
              f"{len(todo)} to run")
        if args.dry_run:
            for c in todo[:args.show]:
                print("   ", " ".join(command(c, ctx, "<out.csv>", "<series.csv>", args.pin)))
            if len(todo) > args.show:
                print(f"    ... and {len(todo) - args.show} more")
            continue

        records: List[dict] = []
        t0 = time.monotonic()
        jobs = 1 if timing else max(1, args.jobs)
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_cell, c, ctx, args.force, args.pin): c for c in cells}
            done = 0
            for fut in cf.as_completed(futures):
                rec = fut.result()
                records.append(rec)
                done += 1
                if rec["status"] == "failed":
                    failures += 1
                    print(f"  FAILED {rec['name']}: {' | '.join(rec.get('stderr', []))}",
                          file=sys.stderr)
                if done % 25 == 0 or done == len(cells):
                    print(f"  {done}/{len(cells)} ({time.monotonic() - t0:.0f} s)",
                          flush=True)
        n = assemble(study, cells, ctx)
        order = {c.name: i for i, c in enumerate(cells)}
        records.sort(key=lambda r: order[r["name"]])
        manifest = {
            "study": study,
            "variant": studies._variant,
            "quick": args.quick,
            "git": git_state(),
            "machine": state,
            "timing_study": timing,
            "noisy_timing_allowed": bool(timing and args.allow_noisy_timing),
            "pinned_to": args.pin if timing else None,
            "bench": {"path": ctx.bench, "sha256": ctx.bench_sha,
                      "config_sha256": ctx.config_sha},
            "cells": records,
        }
        with open(os.path.join(study_dir(study), "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        ran = sum(r["status"] == "ran" for r in records)
        print(f"{study}: ran {ran}, summary.csv has {n} rows")
    return 1 if failures else 0


def status(names: Sequence[str], quick: bool) -> None:
    ctx = Context(find_bench())
    print(f"{'study':14s} {'cells':>6s} {'cached':>7s}")
    for study in names:
        cells = studies.build(study, quick=quick)
        cached = sum(os.path.isfile(cell_paths(c, ctx)[1]) for c in cells)
        print(f"{study:14s} {len(cells):6d} {cached:7d}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the studies and their sizes")
    st = sub.add_parser("status", help="how many cells of each study are cached")
    st.add_argument("studies", nargs="*")
    st.add_argument("--quick", action="store_true")
    rn = sub.add_parser("run", help="run studies (all cached cells are skipped)")
    rn.add_argument("studies", nargs="+", help="study names, or 'all'")
    rn.add_argument("--jobs", type=int, default=2, help="parallel processes (default 2; "
                    "the dev box is RAM-tight, and timing studies are always serial)")
    rn.add_argument("--quick", action="store_true", help="few seeds and repeats, to check "
                    "the pipeline end to end")
    rn.add_argument("--force", action="store_true", help="re-run cached cells")
    rn.add_argument("--dry-run", action="store_true", help="print what would run")
    rn.add_argument("--show", type=int, default=3, help="commands to show in a dry run")
    rn.add_argument("--select", default="", help="only cells whose name contains this")
    rn.add_argument("--pin", default=None, help="taskset core list for timing cells, "
                    "e.g. 4,5")
    rn.add_argument("--allow-noisy-timing", action="store_true",
                    help="run timing cells on battery / without the performance "
                         "governor (recorded in the manifest)")
    for p in (st, rn):
        p.add_argument("--variant", choices=studies.VARIANTS,
                       default=studies.DEFAULT_VARIANT,
                       help="controller settings: tune.py's result in tuned.json "
                            "(tuned, the default) or the committed YAML (shipped)")
    args = ap.parse_args(argv)
    studies.set_variant(getattr(args, "variant", studies.DEFAULT_VARIANT))

    if args.cmd == "list":
        for name in studies.STUDIES:
            print(f"{name:14s} {len(studies.build(name)):5d} cells "
                  f"({len(studies.build(name, quick=True))} quick)")
        return 0
    names = list(studies.STUDIES) if not getattr(args, "studies", None) or \
        "all" in args.studies else args.studies
    unknown = [n for n in names if n not in studies.STUDIES]
    if unknown:
        sys.exit(f"unknown studies: {unknown}; choose from {list(studies.STUDIES)}")
    if args.cmd == "status":
        status(names, args.quick)
        return 0
    return run_studies(names, args)


if __name__ == "__main__":
    sys.exit(main())
