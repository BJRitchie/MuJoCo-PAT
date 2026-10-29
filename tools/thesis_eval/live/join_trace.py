#!/usr/bin/env python3
"""Fill a recorded series' tick_us and mode from the arm node's own per-tick trace.

The recorder writes one row per plant publish and one `.ticks.json` mapping each
torque_command message (one per completed control tick, in order) to the row it
fell in. The node's `trace_file` has one line per tick in the same order, so the
Nth trace line is the Nth torque message. Rows with no tick of their own (a tick
skipped or two landing in one publish) keep the previous tick's values.

  join_trace.py run.csv trace_left.csv

Refuses when the counts disagree by more than a tick or two, since the join would
then be misaligned; the caller records that trial as unhealthy.
"""
import csv
import json
import sys

TICK_US, MODE = 1, 18        # columns of the node's trace (arm_control_report.cpp)


def join(series_csv, trace_csv, tolerance=2):
    with open(series_csv + ".ticks.json") as fh:
        meta = json.load(fh)
    with open(trace_csv) as fh:
        trace = [r for r in csv.reader(fh) if r and not r[0].startswith(("#", "t_us"))]
    n_msg = meta["n_torque_msgs"]
    row_of = {int(k): v for k, v in meta["row_of_torque_msg"].items()}
    # The node ticked for a while before the recorder started, so the trace has
    # more lines than the recorder saw messages. Both end with the node's last
    # tick (recorder outlives the stack), so align from the end; the trace must
    # be at least as long as what the recorder saw.
    if len(trace) < n_msg - tolerance:
        raise SystemExit(f"join_trace: {len(trace)} trace lines but {n_msg} torque "
                         f"messages recorded; ticks are missing from the trace")
    offset = len(trace) - n_msg
    with open(series_csv) as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    ci, cm = head.index("tick_us"), head.index("mode")
    last = (0.0, 0)
    filled = [None] * len(body)
    for k in sorted(row_of):
        t = k + offset
        if 0 <= t < len(trace):
            filled[row_of[k]] = (float(trace[t][TICK_US]), int(float(trace[t][MODE])))
    for i, r in enumerate(body):
        last = filled[i] or last
        r[ci], r[cm] = f"{last[0]:.9g}", str(last[1])
    tmp = series_csv + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head)
        w.writerows(body)
    import os
    os.replace(tmp, series_csv)
    return len(trace), n_msg


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print("joined: %d trace lines, %d torque messages" % join(sys.argv[1], sys.argv[2]))
