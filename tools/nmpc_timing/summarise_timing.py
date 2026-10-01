#!/usr/bin/env python3
"""Aggregate pat_arm_nmpc's per-second timing reports into one comparable row.

The node already prints everything; what it cannot do is summarise across a run.
A 300 s trial is ~300 windows per arm, and reading them by eye hides exactly the
thing that matters -- the tail. This collapses each run to the statistics a
deadline is judged on, and prints several runs together so a tuning change can be
read off directly.

    tools/nmpc_timing/summarise_timing.py /tmp/base.log /tmp/try.log
    tools/nmpc_timing/summarise_timing.py --per-arm --warmup 20 /tmp/*.log

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
# ([arm_nmpc_node-2] etc.) that both share.
RE_NODE = re.compile(
    r'(\[arm_nmpc_node-\d+\])?.*?pat_arm_nmpc\[(?P<side>\w+)\]\s+'
    r'(?P<ticks>\d+) ticks @ (?P<period>[\d.]+) us'
    r'.*?jitter mean (?P<jit_mean>[\d.]+) max (?P<jit_max>[\d.]+)'
    r'.*?misses (?P<misses>\d+) \(longest run (?P<miss_run>\d+)\)'
    r'.*?skipped periods (?P<skipped>\d+)')
RE_TICK = re.compile(r'TICK mean=(?P<mean>[\d.]+) max=(?P<max>[\d.]+)'
                     r'(?: over_period=(?P<over>\d+)/(?P<of>\d+))?')
RE_QP = re.compile(r'qp_iter/tick mean=(?P<it_mean>[\d.]+) max=(?P<it_max>[\d.]+)'
                   r'.*?retries=(?P<retries>\d+)')
RE_PREFIX = re.compile(r'^(\[arm_nmpc_node-\d+\])')


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
    args = ap.parse_args()
    spike_us = args.spike_ms * 1000.0

    print(HEAD.format('run', 'wins', 'ticks', 'period', 'mean', 'max', 'p95',
                      'over', 'it_mean', 'it_max', 'retr', 'skipped', 'miss_run',
                      f'>{args.spike_ms:g}ms'))
    missing = False
    for path in args.logs:
        by_side = parse(path)
        if by_side is None:           # unreadable; parse() already said why
            missing = True
            continue
        if not by_side:
            print(f'{path}: no complete report windows found', file=sys.stderr)
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
    # Non-zero when any log was unusable, so a mistyped filename fails a script
    # that chains this after a trial rather than looking like an empty result.
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
