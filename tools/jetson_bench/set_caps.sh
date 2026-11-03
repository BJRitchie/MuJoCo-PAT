#!/usr/bin/env bash
# Set the NMPCs' QP iteration caps in THIS checkout's tuned.json, for a rerun of
# the suite at caps that fit this machine (quick_check.py prints the command).
#   tools/jetson_bench/set_caps.sh SQP_CAP LMPC_CAP [--cold|--warm]
#       e.g. 12 6, or "keep" to leave one as it is
#   --cold  every QP starts cold and is never retried (qp.warm_start=0): a tick
#           costs at most ONE cap of iterations, not two (warm start + the
#           cold-start retry on failure), so about twice the cap fits.
#   --warm  back to the tuned warm start (removes the qp.warm_start override).
# A rerun with new caps is new cells (the arguments are part of every cell's
# key), so give it its own results name; the old results stay as they were.
set -euo pipefail
SQP=${1:?usage: set_caps.sh SQP_CAP LMPC_CAP [--cold|--warm]}
LMPC=${2:?usage: set_caps.sh SQP_CAP LMPC_CAP [--cold|--warm]}
MODE=${3:-}
case "$MODE" in ""|--cold|--warm) ;; *) echo "unknown option '$MODE' (--cold or --warm)" >&2; exit 2;; esac
cd "$(dirname "$0")/../.."
python3 - "$SQP" "$LMPC" "$MODE" <<'PY'
import json, sys
p = "tools/thesis_eval/tuned.json"
d = json.load(open(p))
mode = sys.argv[3]
for tag, cap in (("sqp", sys.argv[1]), ("iter", sys.argv[2])):
    a = d[tag]["args"]
    if cap != "keep":
        i = next(k for k, x in enumerate(a) if x.startswith("qp.max_iter="))
        a[i] = f"qp.max_iter={int(cap)}"
    if mode:
        # args alternate "--set", "key=value": drop any warm_start pair first.
        pairs = [(a[k], a[k + 1]) for k in range(0, len(a), 2)]
        pairs = [pr for pr in pairs if not pr[1].startswith("qp.warm_start=")]
        if mode == "--cold":
            pairs.append(("--set", "qp.warm_start=0"))
        a[:] = [x for pr in pairs for x in pr]
    cur = next(x for x in a if x.startswith("qp.max_iter="))
    cold = "qp.warm_start=0" in a
    print(f"{tag}: {cur}, {'cold start, no retry' if cold else 'warm start + cold retry'}")
json.dump(d, open(p, "w"), indent=1)
open(p, "a").write("\n")
PY
