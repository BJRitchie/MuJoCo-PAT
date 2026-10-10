#!/usr/bin/env bash
# Run one timed NMPC trial and capture its diagnostics to a log.
#
# Run this INSIDE the dev container (docker compose exec dev bash). The timeout
# has to be container-side: a host-side `timeout` around `docker compose exec`
# kills only the client and leaves the launch running, and a second trial then
# competes with the first on the same topics, which silently invalidates both.
#
#   ./tools/nmpc_timing/run_trial.sh -d 300 -o /tmp/warm.log
#   ./tools/nmpc_timing/run_trial.sh -d 300 -o /tmp/cold.log /tmp/my_nmpc.yaml
#
# Then: tools/nmpc_timing/summarise_timing.py /tmp/warm.log /tmp/cold.log
usage() {
    cat <<USAGE
usage: $(basename "$0") [-d SECONDS] [-o LOGFILE] [-t] [PARAMS_FILE]

  -d SECONDS   how long to run after startup (default 300). The report window is
               1 s, so this is roughly how many windows you get per arm.
  -o LOGFILE   where to write the captured output (default ./nmpc_trial.log)
  -T FILE      record every tick's timing to FILE, one CSV per arm (FILE gets
               _left / _right spliced in). Plot with plot_trace.py. Adds a 1 Hz
               file write on the executor thread, so leave it off when the
               absolute worst-case tick is the measurement.
  -g           refuse to run unless the CPU governor is 'performance'. The
               governor is reported either way and recorded in the log; it cannot
               be SET from in here (/sys is read-only in the container), so pin it
               from a host shell first: sudo cpupower frequency-set -g performance
  -e           also bring up pat_telemetry, for EE-tracking plots
  PARAMS_FILE  an nmpc.yaml to use instead of the installed default. Copy
               src/pat_arm_nmpc/config/nmpc.yaml, change one knob, pass it here --
               no rebuild needed, since the launch forwards it as params_file.
USAGE
}

DURATION=300
LOGFILE="nmpc_trial.log"
TELEMETRY=false
TRACE=""
REQUIRE_PERF=false
while getopts ":d:o:T:egh" opt; do
    case $opt in
        d) DURATION=$OPTARG ;;
        o) LOGFILE=$OPTARG ;;
        T) TRACE=$OPTARG ;;
        e) TELEMETRY=true ;;
        g) REQUIRE_PERF=true ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
PARAMS=${1:-}

# The dev image sources ROS from ~/.bashrc, which a non-interactive shell never
# reads, so do it here instead of requiring the caller to. Harmless when the
# caller already has it.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if ! command -v ros2 >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
fi
# shellcheck disable=SC1091
[ -f "$ROOT/install/setup.bash" ] && source "$ROOT/install/setup.bash"
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 not found -- run this inside the dev container, or source setup.sh" >&2
    exit 1
fi
cd "$ROOT" || exit 1

if [ -n "$PARAMS" ] && [ ! -f "$PARAMS" ]; then
    echo "no such params file: $PARAMS" >&2
    exit 1
fi

# A trial measures microseconds, so anything else running on the box lands in the
# numbers. Refuse rather than quietly produce a contaminated log.
if pgrep -x arm_nmpc_node >/dev/null 2>&1 || pgrep -x simulation_node >/dev/null 2>&1; then
    echo "a sim or NMPC node is already running -- stop it first, or this trial" >&2
    echo "will share the CPU (and the topics) with it" >&2
    exit 1
fi

# Which nmpc.yaml will actually be read. With no PARAMS the launch resolves
# FindPackageShare, i.e. the INSTALLED copy -- which is a build-time copy of the
# source one, so editing src/ without rebuilding silently measures the old
# config. Resolve it here, report it, and refuse if the two have diverged.
SRC_PARAMS="$ROOT/src/pat_arm_nmpc/config/nmpc.yaml"
if [ -n "$PARAMS" ]; then
    EFFECTIVE=$(readlink -f "$PARAMS")
else
    PREFIX=$(ros2 pkg prefix pat_arm_nmpc 2>/dev/null)
    EFFECTIVE="$PREFIX/share/pat_arm_nmpc/config/nmpc.yaml"
    if [ -f "$SRC_PARAMS" ] && [ -f "$EFFECTIVE" ] \
       && ! diff -q "$SRC_PARAMS" "$EFFECTIVE" >/dev/null 2>&1; then
        echo "REFUSING: the installed nmpc.yaml differs from src/." >&2
        echo "  installed (what this run would use): $EFFECTIVE" >&2
        echo "  source   (what you have been editing): $SRC_PARAMS" >&2
        diff "$SRC_PARAMS" "$EFFECTIVE" | sed 's/^/    /' >&2
        echo "  Either rebuild:  colcon build --packages-select pat_arm_nmpc" >&2
        echo "  or pass it:      $(basename "$0") ... $SRC_PARAMS" >&2
        exit 1
    fi
fi

# The CPU governor, which this box's measurements are extremely sensitive to:
# cores have been observed spread over 464-2113 MHz on powersave, and the worst
# ticks in a trial are the same arithmetic executed more slowly rather than more
# work. /sys is readable here but mounted read-only, so this reports and refuses;
# it cannot set it. Pin it from a HOST shell:
#   sudo cpupower frequency-set -g performance
GOV_FILES=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
GOVERNORS=""
if [ -r "${GOV_FILES[0]}" ]; then
    GOVERNORS=$(cat "${GOV_FILES[@]}" 2>/dev/null | sort -u | paste -sd, -)
fi
if [ -z "$GOVERNORS" ]; then
    GOVERNORS="unknown (no cpufreq sysfs)"
fi
if [ "$REQUIRE_PERF" = true ] && [ "$GOVERNORS" != "performance" ]; then
    echo "REFUSING: -g given but the CPU governor is '$GOVERNORS', not performance." >&2
    echo "  cpupower is not in the dev image and /sys is mounted read-only here," >&2
    echo "  so set it from a HOST shell, then re-run this:" >&2
    echo "    sudo cpupower frequency-set -g performance" >&2
    exit 1
fi

ARGS=(launch/full_stack_nmpc.launch.py with_targets:=true)
[ -n "$PARAMS" ]        && ARGS+=("params_file:=$PARAMS")
[ -n "$TRACE" ]         && ARGS+=("trace_file:=$TRACE")
[ "$TELEMETRY" = true ] && ARGS+=(telemetry:=true)

echo "running ${DURATION}s trial -> $LOGFILE" >&2
echo "  params: $EFFECTIVE" >&2
[ -n "$TRACE" ] && echo "  trace:  ${TRACE} (_left / _right)" >&2
if [ "$GOVERNORS" = "performance" ]; then
    echo "  governor: performance" >&2
else
    echo "  governor: $GOVERNORS  <-- NOT performance; the tail of this run is" >&2
    echo "            not comparable with a pinned one (see tools/nmpc_timing/README.md)" >&2
fi
# Record what produced this log, and the settings most likely to be the subject of
# whatever comparison it gets used in. A log that cannot say which config it came
# from is worth very little a day later.
{
    echo "# pat_arm_nmpc trial: $(date -Is), ${DURATION}s"
    echo "# params_file: $EFFECTIVE"
    echo "# cpu_governor: $GOVERNORS"
    grep -E '^\s+(control_hz|warm_start|max_iter|N|Ts|full_nonlinear|sqp_iters):' \
        "$EFFECTIVE" 2>/dev/null | sed 's/^/#   /'
} > "$LOGFILE"

# Startup costs ~8 s (model load, DDS discovery, the first few solves) and those
# windows are always the worst ones; summarise_timing.py drops them by default.
timeout -s INT "$((DURATION + 10))" ros2 launch "${ARGS[@]}" >> "$LOGFILE" 2>&1

sleep 3
if pgrep -x arm_nmpc_node >/dev/null 2>&1; then
    echo "WARNING: nodes survived teardown; kill them before the next trial" >&2
    pgrep -ax arm_nmpc_node >&2
fi

WINDOWS=$(grep -c 'ticks @' "$LOGFILE" 2>/dev/null || echo 0)
echo "done: $WINDOWS report windows captured" >&2
