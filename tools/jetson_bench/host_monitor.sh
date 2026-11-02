#!/usr/bin/env bash
# Run on the HOST (not in the container), in a second terminal, for as long as
# the suite runs; Ctrl-C when it finishes. Records the machine's power mode and
# clocks once, then its temperatures and clocks every second, into
# data/thesis_eval/machines/NAME/, so throttling during a run is on record.
#   tools/jetson_bench/host_monitor.sh orin_nx
set -uo pipefail
NAME=${1:?usage: host_monitor.sh NAME}
cd "$(dirname "$0")/../.."
OUT=data/thesis_eval/machines/$NAME
mkdir -p "$OUT"
{
    echo "# host_info, $(date -Is)"
    cat /proc/device-tree/model 2>/dev/null | tr -d '\0'; echo
    uname -a
    echo "## governors"; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
    if command -v nvpmodel >/dev/null; then
        echo "## nvpmodel -q"; sudo nvpmodel -q 2>&1
        echo "## jetson_clocks --show"; sudo jetson_clocks --show 2>&1
    fi
    echo "## lscpu"; lscpu
} > "$OUT/host_info.txt"
echo "wrote $OUT/host_info.txt"
if command -v tegrastats >/dev/null; then
    echo "tegrastats -> $OUT/tegrastats.log (Ctrl-C to stop)"
    exec sudo tegrastats --interval 1000 --logfile "$OUT/tegrastats.log"
fi
echo "no tegrastats: logging CPU clocks and temperatures -> $OUT/monitor.log (Ctrl-C to stop)"
while true; do
    echo "$(date +%T) temps $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | tr '\n' ' ')" \
         "freqs $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq | tr '\n' ' ')"
    sleep 1
done >> "$OUT/monitor.log"
