#!/usr/bin/env python3
"""Attribute a trace's slow ticks: this controller, or the machine under it?

plot_trace.py shows the distribution. This asks where its tail comes from, which
is the question that decides whether a tail is worth tuning. Three tests, all on
traces you already have -- no new run:

  1. The 1 Hz trace flush. It writes on the same executor as the control
     callback, so it is the obvious suspect. Compares the tick right after each
     flush against all the others, and shows where in the second the slow ticks
     fall.
  2. Cross-arm co-occurrence. The two arms are separate processes solving
     independent QPs. Outliers that coincide in wall-clock time have a cause
     outside both controllers -- a CPU frequency dip, or something else on the
     box. Outliers that do not coincide belong to the arm they happened in.
  3. The slowest ticks themselves, column by column, so a tick that is slow
     because HPIPM took more iterations is not confused with one that ran the
     same iterations more slowly.

    tools/nmpc_timing/tail_source.py data/nmpc_timing/trace_{left,right}.csv
    tools/nmpc_timing/tail_source.py --pct 99.9 trace_left.csv

Stdlib only and ASCII out, for the same reasons as plot_trace.py.
"""
import argparse
import sys

from plot_trace import read

# The flush timer's period, as arm_nmpc_node.cpp creates it.
FLUSH_PERIOD_US = 1_000_000.0


def pctl(values, q):
    """Nearest-rank percentile, so the threshold is always an observed value."""
    if not values:
        return float('nan')
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def mean(values):
    return sum(values) / len(values) if values else float('nan')


def flush_test(rows, pct, bins=20):
    """Is the tick that follows a flush slower than the rest?"""
    ticks = [r['tick_us'] for r in rows]
    thr = pctl(ticks, pct)
    period = _tick_period(rows)

    # "First tick of the second" = the one that ran right after the flush. One
    # control period of tolerance, since the two timers are independent.
    first = [r['tick_us'] for r in rows
             if r['t_us'] % FLUSH_PERIOD_US < period]
    rest = [r['tick_us'] for r in rows
            if r['t_us'] % FLUSH_PERIOD_US >= period]
    print("  1. the 1 Hz trace flush")
    for name, vals in (('tick after each flush', first), ('every other tick', rest)):
        if not vals:
            continue
        print(f"     {name:<22} n={len(vals):<6} mean {mean(vals):6.0f}  "
              f"p99 {pctl(vals, 99):6.0f}  max {max(vals):7.0f} us")
    if first and rest:
        verdict = ("flush-adjacent ticks are NOT slower"
                   if mean(first) <= mean(rest) else
                   "flush-adjacent ticks ARE slower -- investigate")
        print(f"     -> {verdict}")

    # Where in the second the slow ticks fall. A flush cause concentrates them
    # in the first bin; anything else does not.
    slow = [r['t_us'] % FLUSH_PERIOD_US for r in rows if r['tick_us'] >= thr]
    if slow:
        width = FLUSH_PERIOD_US / bins
        hist = [0] * bins
        for ph in slow:
            hist[min(bins - 1, int(ph / width))] += 1
        top = max(hist)
        exp = len(slow) / bins
        print(f"     phase within the second of the {len(slow)} ticks "
              f">= p{pct} ({thr:.0f} us), {exp:.1f} expected per bin:")
        for i, c in enumerate(hist):
            bar = '#' * int(round(30 * c / top)) if top else ''
            lo = i * width / 1000.0
            print(f"       {lo:4.0f}-{lo + width / 1000.0:4.0f} ms {c:5d}  {bar}")


def _tick_period(rows):
    """Median inter-tick spacing, as the trace itself reports it."""
    if len(rows) < 3:
        return 10_000.0
    d = sorted(rows[i + 1]['t_us'] - rows[i]['t_us'] for i in range(len(rows) - 1))
    return d[len(d) // 2]


def cooccurrence(a_rows, b_rows, a_name, b_name, pct, win_us=20_000.0):
    """Do the two arms' outliers land at the same wall-clock moment?"""
    ta = [r['t_us'] for r in a_rows
          if r['tick_us'] >= pctl([x['tick_us'] for x in a_rows], pct)]
    tb = [r['t_us'] for r in b_rows
          if r['tick_us'] >= pctl([x['tick_us'] for x in b_rows], pct)]
    if not ta or not tb:
        return

    # Each node's t_us starts at its own bring-up, so the two clocks have an
    # unknown offset. Scan for the one that lines the outliers up best; a
    # coincidence would not beat its neighbours by much.
    span = max(r['t_us'] for r in a_rows) - min(r['t_us'] for r in a_rows)
    tb_sorted = sorted(tb)
    best, best_off = -1, 0
    for off_ms in range(-1000, 1001):
        off = off_ms * 1000.0
        hits = sum(1 for t in ta if _near(tb_sorted, t - off, win_us))
        if hits > best:
            best, best_off = hits, off_ms

    # If the two sets were independent, each of a's outliers would land within
    # the window of one of b's purely by chance at this rate.
    expected = len(ta) * len(tb) * 2 * win_us / span if span else float('nan')
    print(f"  2. cross-arm co-occurrence at p{pct} "
          f"({len(ta)} {a_name} / {len(tb)} {b_name} outliers)")
    print(f"     best clock alignment {best_off:+d} ms")
    print(f"     {best} of {len(ta)} {a_name} outliers have a {b_name} outlier "
          f"within +-{win_us / 1000:.0f} ms "
          f"({100.0 * best / len(ta):.0f}%)")
    print(f"     expected if independent: {expected:.1f} ({100.0 * expected / len(ta):.0f}%)")
    if expected > 0 and best > 3 * expected:
        print(f"     -> {best / expected:.0f}x enrichment: the tail has a cause "
              f"OUTSIDE both controllers")
    else:
        print("     -> consistent with independent tails, i.e. per-arm causes")


def _near(sorted_ts, t, win):
    """Is any element of sorted_ts within win of t? Bisect, so the scan stays cheap."""
    import bisect
    i = bisect.bisect_left(sorted_ts, t - win)
    return i < len(sorted_ts) and sorted_ts[i] <= t + win


def worst(rows, n, period):
    """The slowest ticks, split into iteration count vs cost per iteration."""
    ranked = sorted(rows, key=lambda r: -r['tick_us'])[:n]
    med_iter_cost = _median_iter_cost(rows)
    print(f"  3. the {len(ranked)} slowest ticks "
          f"(median cost per QP iteration {med_iter_cost:.0f} us)")
    print("       t_s     tick_us  hpipm_us  share  iters  us/iter  vs median")
    for r in ranked:
        it = r.get('qp_iters', 0) or 0
        hp = r.get('hpipm_us', 0.0)
        per = hp / it if it else float('nan')
        ratio = per / med_iter_cost if med_iter_cost else float('nan')
        flag = '  <-- same work, slower' if ratio > 3 else ''
        print(f"     {r['t_us'] / 1e6:7.2f} {r['tick_us']:9.0f} {hp:9.0f} "
              f"{100.0 * hp / r['tick_us']:5.0f}% {it:6.0f} {per:8.0f} "
              f"{ratio:8.1f}x{flag}")
    over = [r for r in rows if r['tick_us'] > period]
    print(f"     over the {period:.0f} us period: {len(over)} of {len(rows)} "
          f"({100.0 * len(over) / len(rows):.2f}%)")


def _median_iter_cost(rows):
    costs = [r['hpipm_us'] / r['qp_iters'] for r in rows
             if r.get('qp_iters') and r.get('hpipm_us')]
    return pctl(costs, 50)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traces', nargs='+',
                    help='one trace, or two (left and right) for the cross-arm test')
    ap.add_argument('--pct', type=float, default=99.0,
                    help='percentile that defines an outlier (default 99)')
    ap.add_argument('--top', type=int, default=8,
                    help='how many of the slowest ticks to list (default 8)')
    args = ap.parse_args()

    loaded = []
    for path in args.traces:
        meta, rows = read(path)
        if not rows:
            print(f"{path}: no rows", file=sys.stderr)
            continue
        loaded.append((path, meta, rows))

    for path, meta, rows in loaded:
        period = float(meta.get('period_us', 10000))
        print(f"\n{path}")
        print(f"  arm={meta.get('arm', '?')}  {len(rows)} ticks over "
              f"{(rows[-1]['t_us'] - rows[0]['t_us']) / 1e6:.1f} s")
        flush_test(rows, args.pct)
        worst(rows, args.top, period)

    if len(loaded) == 2:
        (pa, ma, ra), (pb, mb, rb) = loaded
        print()
        cooccurrence(ra, rb, ma.get('arm', 'A'), mb.get('arm', 'B'), args.pct)
    elif len(loaded) > 2:
        print("\n(cross-arm test needs exactly two traces)", file=sys.stderr)


if __name__ == '__main__':
    main()
