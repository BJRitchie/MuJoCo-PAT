#!/usr/bin/env bash
# The whole timing suite on this machine, inside the image (run through enter.sh):
#   tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh orin_nx
#   tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh dgx 6
#
# NAME names the machine's results, data/thesis_eval/machines/NAME/. PIN is the
# CPU every timing cell runs on (default the last one; CPU 0 takes interrupts).
# Each study is cached cell by cell, so re-running after an interruption only
# runs what is missing.
set -euo pipefail
NAME=${1:?usage: run_suite.sh NAME [PIN]}
PIN=${2:-$(( $(nproc) - 1 ))}
cd /ws
source install/setup.bash
RS=(python3 tools/thesis_eval/run_study.py)
OUT=data/thesis_eval/machines/$NAME
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }
T0=$(date +%s)

log "machine '$NAME': $(nproc) CPUs, timing cells pinned to CPU $PIN"
gov=$(cat /sys/devices/system/cpu/cpu"$PIN"/cpufreq/scaling_governor 2>/dev/null || echo unknown)
if [ "$gov" != performance ]; then
    echo "CPU $PIN's governor is '$gov', not 'performance': the timing studies will refuse" \
         "to run. Set it on the HOST (see QUICKSTART.md, 'Prepare the machine') and rerun." >&2
    exit 1
fi

log "1/5 unit tests: realtime allocation bounds, joint filters"
TESTBIN=build/pat_arm_control/test_pat_arm_control
"$TESTBIN" --gtest_filter='RealtimeProfile*:JointReadingFilter*:JointKalmanFilter*' \
    --gtest_brief=1 2>&1 | tail -3

log "2/5 smoke test: one cell, to time this machine"
SMOKE=rectangle_x1_m1_bd0_tau1x_ekf_sqp_s1
t=$(date +%s)
"${RS[@]}" run jetson_timing --select "$SMOKE" --pin "$PIN" --results "$NAME"
dt=$(( $(date +%s) - t ))
[ "$dt" -lt 1 ] && dt=1
# The suite is about 185 smoke cells of work (the timing study ~30, jetson_timing
# ~100, worst_case ~55: its every tick runs the QP cap twice, in both arms).
log "smoke cell took ${dt} s: the rest should take about $(( dt * 185 / 60 )) min"

log "3/5 timing study: horizon sweep, 1x circle, exact state (50 cells)"
"${RS[@]}" run timing --pin "$PIN" --results "$NAME"

log "4/5 jetson_timing: 11 scenarios x 4 controllers x 3 seeds (132 cells)"
"${RS[@]}" run jetson_timing --pin "$PIN" --results "$NAME"

log "5/5 worst_case: every QP forced to its cap, twice per tick (12 cells)"
"${RS[@]}" run worst_case --pin "$PIN" --results "$NAME"

log "done in $(( ($(date +%s) - T0) / 60 )) min: results in $OUT"
