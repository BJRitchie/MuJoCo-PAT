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
  -t           also bring up pat_telemetry, for EE-tracking plots
  PARAMS_FILE  an nmpc.yaml to use instead of the installed default. Copy
               src/pat_arm_nmpc/config/nmpc.yaml, change one knob, pass it here --
               no rebuild needed, since the launch forwards it as params_file.
USAGE
}

DURATION=300
LOGFILE="nmpc_trial.log"
TELEMETRY=false
while getopts ":d:o:th" opt; do
    case $opt in
        d) DURATION=$OPTARG ;;
        o) LOGFILE=$OPTARG ;;
        t) TELEMETRY=true ;;
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

ARGS=(launch/full_stack_nmpc.launch.py with_targets:=true)
[ -n "$PARAMS" ]        && ARGS+=("params_file:=$PARAMS")
[ "$TELEMETRY" = true ] && ARGS+=(telemetry:=true)

echo "running ${DURATION}s trial -> $LOGFILE" >&2
echo "  params: ${PARAMS:-<installed default>}" >&2
# Startup costs ~8 s (model load, DDS discovery, the first few solves) and those
# windows are always the worst ones; summarise_timing.py drops them by default.
timeout -s INT "$((DURATION + 10))" ros2 launch "${ARGS[@]}" > "$LOGFILE" 2>&1

sleep 3
if pgrep -x arm_nmpc_node >/dev/null 2>&1; then
    echo "WARNING: nodes survived teardown; kill them before the next trial" >&2
    pgrep -ax arm_nmpc_node >&2
fi

WINDOWS=$(grep -c 'ticks @' "$LOGFILE" 2>/dev/null || echo 0)
echo "done: $WINDOWS report windows captured" >&2
