#!/usr/bin/env python3
"""Plot a per-tick trace written by pat_arm_control's `trace_file` parameter.

summarise_timing.py reads the node's once-a-second report, so its histogram is the
eight fixed bins the C++ carries and its percentiles are over per-window maxima.
This reads the trace instead -- one row per control tick -- so the distribution is
the real one, at whatever resolution you ask for.

    tools/nmpc_timing/plot_trace.py data/nmpc_timing/trace_left.csv
    tools/nmpc_timing/plot_trace.py --bins 30 --column qpsolve trace_*.csv
    tools/nmpc_timing/plot_trace.py --phases trace_left.csv

ASCII on purpose: it works over any shell into the container, and the dev image's
matplotlib is broken (numpy 1.x/2.x ABI clash). The CSV is plain, so pandas or
gnuplot can take it from here if you want publication output.
"""
import argparse
import csv
import math
import sys

PHASES = ['dynamics', 'jdotqdot', 'lambda', 'qpsolve', 'rollout_dynamics',
          'rollout_integrate', 'pinv', 'build_model', 'defect', 'trust_region',
          'finalize']


def read(path):
    """-> (header comment dict, list of row dicts with floats/ints)."""
    meta, rows = {}, []
    with open(path, newline='') as fh:
        lines = [ln for ln in fh]
    for ln in lines:
        if ln.startswith('#'):
            for tok in ln.lstrip('#').split():
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    meta[k] = v
    body = [ln for ln in lines if not ln.startswith('#')]
    for r in csv.DictReader(body):
        try:
            rows.append({k: float(v) for k, v in r.items() if v != ''})
        except (TypeError, ValueError):
            continue          # a torn final line, e.g. if the process was killed
    return meta, rows


def histogram(values, bins, period=None, label='us'):
    lo, hi = min(values), max(values)
    if hi <= lo:
        print(f"    all {len(values)} samples at {lo:.0f} {label}")
        return
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / width), bins - 1)] += 1
    widest = max(counts) or 1
    total = len(values)
    # CDF (Cumulative Distribution Function) is the running total of the share column: 
    # the proportion of all samples at or below that bin's upper edge. It is the 
    # column a deadline is read off -- 100 minus it at the budget is the share 
    # that missed. 
    print(f"    {'range [' + label + ']':<17}{'':<44} {'count':>9} {'share':>7} {'CDF':>9}")
    running = 0
    for i, n in enumerate(counts):
        edge_lo, edge_hi = lo + i * width, lo + (i + 1) * width
        running += n
        cells = int(round(44.0 * n / widest))
        bar = '#' * cells if cells else ('.' if n else '')
        mark = ''
        if period is not None and edge_lo <= period < edge_hi:
            mark = '  <-- deadline'
        print(f"    {edge_lo:8.0f}-{edge_hi:<8.0f} {bar:<44} {n:>7} "
              f"{100.0*n/total:7.3f}% {100.0*running/total:8.3f}%{mark}")


def percentiles(values):
    s = sorted(values)

    def p(q):
        return s[min(int(len(s) * q), len(s) - 1)]
    return (f"min {s[0]:.0f}  p50 {p(0.50):.0f}  p90 {p(0.90):.0f}  "
            f"p99 {p(0.99):.0f}  p99.9 {p(0.999):.0f}  max {s[-1]:.0f} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--column', default='tick_us',
                    help='which column to plot (default tick_us). Any of the '
                         'phase names works too, e.g. qpsolve.')
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--phases', action='store_true',
                    help='also break the mean tick down by phase')
    ap.add_argument('--top', type=int, default=0, metavar='N',
                    help='also list the N slowest ticks, with their phase split')
    args = ap.parse_args()

    status = 0
    for path in args.traces:
        try:
            meta, rows = read(path)
        except OSError as exc:
            print(f'{path}: {exc.strerror}', file=sys.stderr)
            status = 1
            continue
        if not rows or args.column not in rows[0]:
            if not rows:
                # The two tools read different files and the mistake is easy to
                # make, so name the other one rather than just failing.
                why = (' -- this is a launch log, not a trace; use '
                       'summarise_timing.py (or re-run with trace_file:=)')
            else:
                why = (f" -- no column {args.column!r}; have: "
                       + ', '.join(sorted(rows[0])))
            print(f'{path}: nothing to plot{why}', file=sys.stderr)
            status = 1
            continue

        period = float(meta['period_us']) if 'period_us' in meta else None
        vals = [r[args.column] for r in rows]
        span_s = (rows[-1]['t_us'] - rows[0]['t_us']) / 1e6
        print(f"\n{path}")
        print("  " + "  ".join(f"{k}={v}" for k, v in meta.items()))
        print(f"  {len(rows)} ticks over {span_s:.1f} s")

        print(f"  {args.column} distribution, every tick")
        histogram(vals, args.bins, period if args.column == 'tick_us' else None)
        print(f"  {args.column}: {percentiles(vals)}")
        if period:
            over = sum(1 for v in vals if v > period)
            print(f"  over the {period:.0f} us period: {over} "
                  f"({100.0*over/len(vals):.2f}%)")

        damped = sum(1 for r in rows if r.get('mode', 0) >= 1)
        retried = sum(1 for r in rows if r.get('retried', 0) >= 1)
        if 'qp_iters' in rows[0]:
            it = [r['qp_iters'] for r in rows]
            print(f"  qp_iters: mean {sum(it)/len(it):.2f}  max {max(it):.0f}"
                  f"   retries {retried}   damped ticks {damped}")

        if args.phases:
            print("  mean tick by phase")
            tick_mean = sum(r['tick_us'] for r in rows) / len(rows)
            attributed = 0.0
            for ph in PHASES:
                if ph not in rows[0]:
                    continue
                m = sum(r[ph] for r in rows) / len(rows)
                if m <= 0.0:
                    continue        # not on the active path
                attributed += m
                cells = int(round(44.0 * m / tick_mean))
                print(f"    {ph:>18} {'#' * cells:<44} {m:8.1f} us "
                      f"{100.0*m/tick_mean:5.1f}%")
            print(f"    {'unattributed':>18} {'':<44} "
                  f"{tick_mean-attributed:8.1f} us "
                  f"{100.0*(tick_mean-attributed)/tick_mean:5.1f}%")
            print(f"    {'TOTAL':>18} {'':<44} {tick_mean:8.1f} us")

        if args.top:
            print(f"  {args.top} slowest ticks")
            worst = sorted(rows, key=lambda r: -r['tick_us'])[:args.top]
            for r in worst:
                parts = ' '.join(
                    f"{ph.split('_')[0]}={r[ph]:.0f}" for ph in PHASES
                    if ph in r and r[ph] > 0.05 * r['tick_us'])
                # iters/status are the tick total and the LAST solve; the "1"
                # columns are the first solve alone, which is what says whether a
                # retry rescued the tick or just doubled a hopeless solve.
                split = ''
                if 'qp_iters1' in r:
                    split = (f"first=({r['qp_iters1']:.0f}it,st{r.get('qp_status1', 0):.0f}) ")
                print(f"    t={r['t_us']/1e6:7.2f}s  tick={r['tick_us']:8.0f} us  "
                      f"iters={r.get('qp_iters', -1):.0f} "
                      f"status={r.get('qp_status', 0):.0f} "
                      f"retried={r.get('retried', 0):.0f}  {split}{parts}")
    return status


if __name__ == '__main__':
    sys.exit(main())
