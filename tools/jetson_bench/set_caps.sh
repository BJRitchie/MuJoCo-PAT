#!/usr/bin/env bash
# Set the NMPCs' QP iteration caps in THIS checkout's tuned.json, for a rerun of
# the suite at caps that fit this machine (quick_check.py prints the command).
#   tools/jetson_bench/set_caps.sh SQP_CAP LMPC_CAP       e.g. 12 6, or "keep"
# A rerun with new caps is new cells (the arguments are part of every cell's
# key), so give it its own results name; the old results stay as they were.
set -euo pipefail
SQP=${1:?usage: set_caps.sh SQP_CAP LMPC_CAP}
LMPC=${2:?usage: set_caps.sh SQP_CAP LMPC_CAP}
cd "$(dirname "$0")/../.."
python3 - "$SQP" "$LMPC" <<'PY'
import json, sys
p = "tools/thesis_eval/tuned.json"
d = json.load(open(p))
for tag, cap in (("sqp", sys.argv[1]), ("iter", sys.argv[2])):
    if cap == "keep":
        continue
    a = d[tag]["args"]
    i = next(k for k, x in enumerate(a) if x.startswith("qp.max_iter="))
    a[i] = f"qp.max_iter={int(cap)}"
    print(f"{tag}: qp.max_iter={int(cap)}")
json.dump(d, open(p, "w"), indent=1)
open(p, "a").write("\n")
PY
