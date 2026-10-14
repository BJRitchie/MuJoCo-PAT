#!/usr/bin/env python3
"""Aggregate pat_arm_control's per-second timing reports into one comparable row.

The node already prints everything; what it cannot do is summarise across a run.
A 300 s trial is ~300 windows per arm, and reading them by eye hides exactly the
thing that matters -- the tail. This collapses each run to the statistics a
deadline is judged on, and prints several runs together so a tuning change can be
read off directly.

    tools/nmpc_timing/summarise_timing.py /tmp/base.log /tmp/try.log
    tools/nmpc_timing/summarise_timing.py --per-arm --warmup 20 /tmp/*.log
    tools/nmpc_timing/summarise_timing.py --plot --per-arm /tmp/base.log

Columns, all microseconds unless noted:
  mean       mean tick, tick-weighted across windows
  max        worst single tick in the run -- the number a hard deadline must fit
  p50 p95    median and 95th percentile OF THE PER-WINDOW MAXIMA, so they describe
             the tail without being pinned to one outlier the way max is
  over       share of ticks that exceeded the control period
  it         HPIPM iterations per tick. A max at or above the cap means a solve
             failed and the cold-start retry ran; solve time is ~linear in this
  retries    cold-start retries, i.e. solves that failed and were redone
  skipped    control periods with no tick at all (the loop ran open-loop through
             them), summed over the run
  miss_run   longest unbroken run of over-deadline ticks seen in any window
  >Nms       windows containing at least one tick over --spike-ms
"""
import argparse
import re
import sys
from collections import defaultdict

# One window per arm. The node line names the side; the profile lines that follow
# carry no side of their own, so they are attached by the launch prefix
# ([arm_control_node-2] etc.) that both share. Either executable name is
# accepted, so logs recorded before the package rename still group per arm
# rather than collapsing into one window.
#
# The log tag is pat_arm_nmpc[side], not the package name: the package is
# pat_arm_control, but the node's logger tag identifies the control law and is
# held fixed so this parser, plot_trace.py and tail_source.py keep reading
# traces recorded before the package was renamed.
PREFIX = r'\[arm_(?:control|nmpc)_node-\d+\]'
RE_NODE = re.compile(
    r'(' + PREFIX + r')?.*?pat_arm_nmpc\[(?P<side>\w+)\]\s+'
    r'(?P<ticks>\d+) ticks @ (?P<period>[\d.]+) us'
    r'.*?jitter mean (?P<jit_mean>[\d.]+) max (?P<jit_max>[\d.]+)'
    r'.*?misses (?P<misses>\d+) \(longest run (?P<miss_run>\d+)\)'
    r'.*?skipped periods (?P<skipped>\d+)')
RE_TICK = re.compile(r'TICK mean=(?P<mean>[\d.]+) max=(?P<max>[\d.]+)'
                     r'(?: over_period=(?P<over>\d+)/(?P<of>\d+))?')
RE_QP = re.compile(r'qp_iter/tick mean=(?P<it_mean>[\d.]+) max=(?P<it_max>[\d.]+)'
                   r'.*?retries=(?P<retries>\d+)')
RE_HIST = re.compile(r'hist\(tick\)\s+(.*)')
RE_BIN = re.compile(r'(<=|>)([\d.]+):(\d+)')
RE_PREFIX = re.compile(r'^(' + PREFIX + r')')


def prefix_of(line):
    """The launch prefix that groups a window's lines, or None outside a launch."""
    m = RE_PREFIX.match(line)
    return m.group(1) if m else None


def parse(path):
    """-> {side: [window dicts]}, or None if the file cannot be read.

    Windows missing a TICK line are dropped.
    """
    by_side = defaultdict(list)
    open_window = {}          # launch prefix -> the window still being filled
    try:
        fh = open(path, errors='replace')
    except OSError as exc:
        print(f'{path}: {exc.strerror}', file=sys.stderr)
        return None
    with fh:
        for line in fh:
            m = RE_NODE.search(line)
            if m:
                pfx = prefix_of(line)
                w = dict(side=m['side'], ticks=int(m['ticks']),
                         period=float(m['period']), jit_mean=float(m['jit_mean']),
                         jit_max=float(m['jit_max']), misses=int(m['misses']),
                         miss_run=int(m['miss_run']), skipped=int(m['skipped']))
                open_window[pfx] = w
                continue
            w = open_window.get(prefix_of(line))
            if w is None:
                continue
            m = RE_TICK.search(line)
            if m:
                w['mean'] = float(m['mean'])
                w['max'] = float(m['max'])
                w['over'] = int(m['over']) if m['over'] else 0
                continue
            m = RE_HIST.search(line)
            if m:
                # [(op, upper_edge_us, count)], in the order the node prints them.
                # The last bin is open-ended, printed as ">5000".
                w['hist'] = [(op, float(edge), int(n))
                             for op, edge, n in RE_BIN.findall(m.group(1))]
                continue
            m = RE_QP.search(line)
            if m:
                w.update(it_mean=float(m['it_mean']), it_max=float(m['it_max']),
                         retries=int(m['retries']))
                # qp_iter is the last line of the block, so the window is complete.
                if 'max' in w:
                    by_side[w['side']].append(w)
                open_window[prefix_of(line)] = None
    return by_side


def summarise(windows, spike_us):
    ticks = sum(w['ticks'] for w in windows)
    maxima = sorted(w['max'] for w in windows)

    def pct(p):
        return maxima[min(int(len(maxima) * p), len(maxima) - 1)]

    return dict(
        windows=len(windows), ticks=ticks,
        period=windows[0]['period'],
        mean=sum(w['mean'] * w['ticks'] for w in windows) / ticks,
        max=maxima[-1], p50=pct(0.50), p95=pct(0.95),
        over=100.0 * sum(w['over'] for w in windows) / ticks,
        jit_mean=sum(w['jit_mean'] * w['ticks'] for w in windows) / ticks,
        jit_max=max(w['jit_max'] for w in windows),
        it_mean=sum(w.get('it_mean', 0.0) * w['ticks'] for w in windows) / ticks,
        it_max=max(w.get('it_max', 0.0) for w in windows),
        retries=sum(w.get('retries', 0) for w in windows),
        skipped=sum(w['skipped'] for w in windows),
        miss_run=max(w['miss_run'] for w in windows),
        spikes=sum(1 for w in windows if w['max'] > spike_us))


def plot(name, windows, period_us):
    """Two ASCII views of one arm's timing, from what the log actually carries.

    The histogram is over EVERY tick -- the node bins each one as it happens, so
    summing the per-window bins is the true distribution across the run. The
    timeline and percentiles below it are over per-window MAXIMA, because a worst
    case is all the log keeps per window. The two are not interchangeable: 61% of
    ticks landing under 2.5 ms says nothing about how bad the worst one was.
    """
    ticks = sum(w['ticks'] for w in windows)
    print(f"\n{name} — {len(windows)} windows, {ticks} ticks @ {period_us:.0f} us period")

    binned = [w for w in windows if 'hist' in w]
    if not binned:
        print("  (no hist(tick) lines in this log — it predates the histogram)")
    else:
        labels = [f'{op}{edge:.0f} us' for op, edge, _ in binned[0]['hist']]
        edges = [edge for _, edge, _ in binned[0]['hist']]
        totals = [0] * len(labels)
        for w in binned:
            for i, (_, _, n) in enumerate(w['hist']):
                totals[i] += n
        wide = max(totals) or 1
        total_n = sum(totals) or 1
        print("  tick time, all ticks")
        for i, (lab, edge, n) in enumerate(zip(labels, edges, totals)):
            cells = int(round(48.0 * n / wide))
            bar = '#' * cells if cells else ('.' if n else '')
            lower = edges[i - 1] if i else 0.0
            flag = '  <-- deadline' if period_us and lower < period_us <= edge else ''
            print(f"    {lab:>12}  {bar:<48} {n:>8} {100.0*n/total_n:5.1f}%{flag}")
        # The node's bin edges were chosen around a 5 ms period. If almost
        # everything lands in two bins the histogram has no resolution where the
        # data is, and PhaseStat::kEdgeUs wants rescaling to the current period.
        busiest = sorted(totals)[-2:]
        if total_n and sum(busiest) > 0.95 * total_n:
            print(f"    NOTE: {100.0*sum(busiest)/total_n:.0f}% of ticks fall in just "
                  "two bins -- PhaseStat::kEdgeUs is scaled for a 5 ms period and is "
                  "too coarse here")

    maxima = [w['max'] for w in windows]
    srt = sorted(maxima)

    def pct(p):
        return srt[min(int(len(srt) * p), len(srt) - 1)]

    print("  per-window max: "
          f"p50 {pct(0.50):.0f}  p90 {pct(0.90):.0f}  p99 {pct(0.99):.0f}  "
          f"max {srt[-1]:.0f} us")

    # One cell per window, in order, so clustering is visible -- spikes bunch
    # after a setpoint change rather than scattering uniformly. Log-scaled
    # against the run's own worst window.
    ramp = ' .:-=+*#'
    lo = min(maxima)
    hi = max(maxima)
    print(f"  per-window max over the run, '{ramp[0]}'={lo:.0f} '{ramp[-1]}'={hi:.0f} us"
          + (" (log scale)" if hi > 4 * lo else ""))
    if hi <= lo:
        cells = ramp[0] * len(maxima)
    else:
        import math
        span = math.log10(hi) - math.log10(lo) if hi > 4 * lo else hi - lo
        def level(v):
            x = (math.log10(v) - math.log10(lo)) if hi > 4 * lo else (v - lo)
            return ramp[min(int(len(ramp) * x / span), len(ramp) - 1)]
        cells = ''.join(level(v) for v in maxima)
    for i in range(0, len(cells), 100):
        print(f"    {cells[i:i+100]}")


HEAD = ("{:<28} {:>6} {:>7} {:>7} {:>8} {:>7} {:>7} {:>6} {:>7} {:>6} {:>5} "
        "{:>8} {:>8} {:>8}")
ROW = ("{:<28} {:>6} {:>7} {:>7.0f} {:>8.0f} {:>7.0f} {:>7.0f} {:>5.2f}% {:>7.2f} "
       "{:>6.0f} {:>5} {:>8} {:>8} {:>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--warmup', type=int, default=10, metavar='N',
                    help='drop the first N windows per arm (default 10). Startup '
                         'loads the model, settles DDS and pays the first solves, '
                         'and those windows are always the worst in the run.')
    ap.add_argument('--per-arm', action='store_true',
                    help='one row per arm instead of pooling left and right')
    ap.add_argument('--spike-ms', type=float, default=25.0, metavar='MS',
                    help='threshold for the spike-window count (default 25)')
    ap.add_argument('--plot', action='store_true',
                    help='also draw the tick-time distribution and a per-window '
                         'max timeline for each run (and each arm, with --per-arm)')
    args = ap.parse_args()
    spike_us = args.spike_ms * 1000.0

    print(HEAD.format('run', 'wins', 'ticks', 'period', 'mean', 'max', 'p95',
                      'over', 'it_mean', 'it_max', 'retr', 'skipped', 'miss_run',
                      f'>{args.spike_ms:g}ms'))
    # Collected while the table prints, drawn after it, so the comparison table
    # stays contiguous instead of being broken up by plots.
    plots = []
    missing = False
    for path in args.logs:
        by_side = parse(path)
        if by_side is None:           # unreadable; parse() already said why
            missing = True
            continue
        if not by_side:
            hint = ''
            # The two tools read different files and the mistake is easy to make,
            # so say which one this is rather than just "nothing found".
            try:
                with open(path, errors='replace') as fh:
                    if any('t_us,tick_us' in ln for ln in (fh.readline()
                                                           for _ in range(5))):
                        hint = ' -- this is a per-tick trace; use plot_trace.py'
            except OSError:
                pass
            print(f'{path}: no complete report windows found{hint}', file=sys.stderr)
            missing = True
            continue
        trimmed = {s: w[args.warmup:] for s, w in by_side.items()}
        groups = (sorted(trimmed.items()) if args.per_arm
                  else [('', [w for ws in trimmed.values() for w in ws])])
        for side, windows in groups:
            if not windows:
                print(f'{path}: every window dropped by --warmup', file=sys.stderr)
                continue
            s = summarise(windows, spike_us)
            name = path.rsplit('/', 1)[-1]
            if side:
                name = f'{name}:{side}'
            print(ROW.format(name[:28], s['windows'], s['ticks'], s['period'],
                             s['mean'], s['max'], s['p95'], s['over'],
                             s['it_mean'], s['it_max'], s['retries'],
                             s['skipped'], s['miss_run'], s['spikes']))
            if args.plot:
                plots.append((name, windows, s['period']))
    for name, windows, period in plots:
        plot(name, windows, period)

    # Non-zero when any log was unusable, so a mistyped filename fails a script
    # that chains this after a trial rather than looking like an empty result.
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
