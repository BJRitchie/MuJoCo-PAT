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
usage: $(basename "$0") [-d SECONDS] [-o LOGFILE] [-T FILE] [-a ARG]... [-e] [-g] [PARAMS_FILE]

  -d SECONDS   how long to run after startup (default 300). The report window is
               1 s, so this is roughly how many windows you get per arm.
  -o LOGFILE   where to write the captured output (default ./nmpc_trial.log)
  -T FILE      record every tick's timing to FILE, one CSV per arm (FILE gets
               _left / _right spliced in). Plot with plot_trace.py. Adds a 1 Hz
               file write on the executor thread, so leave it off when the
               absolute worst-case tick is the measurement.
  -g           refuse to run unless the machine is in a state worth measuring:
               CPU governor 'performance' AND on mains power. Both are reported
               either way and recorded in the log. Neither can be SET from in here
               (/sys is read-only in the container), so from a host shell:
                 sudo cpupower frequency-set -g performance
               and plug the laptop in - on battery, platform firmware caps the
               clock well below what the governor requests.
  -a ARG       pass one launch argument through to full_stack_nmpc.launch.py,
               repeatable, and record it in the log header. e.g.
                 -a visualise:=false            headless: no viewer to bump
                 -a with_nav:=true              arms on nav_node's estimate
                 -a sensor_profile:=<yaml>      flight sensing
  -e           also bring up pat_telemetry, for EE-tracking plots
  PARAMS_FILE  an nmpc.yaml to use instead of the installed default. Copy
               src/pat_arm_control/config/nmpc.yaml, change one knob, pass it here --
               no rebuild needed, since the launch forwards it as params_file.
USAGE
}

DURATION=300
LOGFILE="nmpc_trial.log"
TELEMETRY=false
TRACE=""
REQUIRE_PERF=false
EXTRA=()
while getopts ":d:o:T:a:egh" opt; do
    case $opt in
        d) DURATION=$OPTARG ;;
        o) LOGFILE=$OPTARG ;;
        T) TRACE=$OPTARG ;;
        a) EXTRA+=("$OPTARG") ;;
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
#
# Matched on the command line, not with `pgrep -x`: the kernel truncates comm to
# 15 characters and "arm_control_node" is 16, so it appears as
# "arm_control_nod" and an exact-name match silently never fires. The predecessor
# name, arm_nmpc_node, was 13 characters and fit, so the rename broke this guard
# without any visible symptom -- which is the worst kind, because a guard that
# never fires looks exactly like a box that is always clean. The bracket keeps
# pgrep from matching its own command line.
if pgrep -f '[a]rm_control_node' >/dev/null 2>&1 || pgrep -x simulation_node >/dev/null 2>&1; then
    echo "a sim or NMPC node is already running -- stop it first, or this trial" >&2
    echo "will share the CPU (and the topics) with it" >&2
    exit 1
fi

# Which nmpc.yaml will actually be read. With no PARAMS the launch resolves
# FindPackageShare, i.e. the INSTALLED copy -- which is a build-time copy of the
# source one, so editing src/ without rebuilding silently measures the old
# config. Resolve it here, report it, and refuse if the two have diverged.
SRC_PARAMS="$ROOT/src/pat_arm_control/config/nmpc.yaml"
if [ -n "$PARAMS" ]; then
    EFFECTIVE=$(readlink -f "$PARAMS")
else
    PREFIX=$(ros2 pkg prefix pat_arm_control 2>/dev/null)
    EFFECTIVE="$PREFIX/share/pat_arm_control/config/nmpc.yaml"
    if [ -f "$SRC_PARAMS" ] && [ -f "$EFFECTIVE" ] \
       && ! diff -q "$SRC_PARAMS" "$EFFECTIVE" >/dev/null 2>&1; then
        echo "REFUSING: the installed nmpc.yaml differs from src/." >&2
        echo "  installed (what this run would use): $EFFECTIVE" >&2
        echo "  source   (what you have been editing): $SRC_PARAMS" >&2
        diff "$SRC_PARAMS" "$EFFECTIVE" | sed 's/^/    /' >&2
        echo "  Either rebuild:  colcon build --packages-select pat_arm_control" >&2
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

# The governor is not the whole story on intel_pstate: the energy/performance
# preference is a separate hardware hint, and power-profiles-daemon sets THAT
# without touching the governor. So two runs can log the same governor and still
# have behaved differently. Recorded alongside it rather than gated on, since the
# governor is the knob that decides whether the clock falls between ticks.
EPP=""
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference ]; then
    EPP=$(cat /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference \
          2>/dev/null | sort -u | paste -sd, -)
fi

# Mains power, which on a laptop matters as much as the governor: the reference
# box pins `performance` and still only reaches ~3.1 GHz of its 5.0 GHz on
# battery, because the limit is applied by platform firmware rather than through
# anything in cpufreq sysfs. A run on battery is not comparable with one on mains.
# Read from the AC adapter rather than the battery, so a machine with no battery
# at all reads as powered rather than as unknown.
ON_AC="unknown"
for _ac in /sys/class/power_supply/A{C,DP}*/online; do
    [ -r "$_ac" ] || continue
    ON_AC=$([ "$(cat "$_ac")" = "1" ] && echo yes || echo no)
    break
done

# "On mains" is necessary but NOT sufficient: a weak USB-C charger reports
# online=1, keeps the battery charging, and still leaves the package power-limited.
# Measured on the reference box, a 15 W PD contract cost 1.5x on the mean tick and
# 3x on the tail against the proper adapter, while on_ac_power read "yes" and -g
# passed. So report the negotiated wattage when a USB-C source is supplying, and
# measure the achieved clock during the run (below), which catches a weak charger,
# thermal throttling and a dying fan alike rather than guessing at the cause.
PD_WATTS=""
for _pd in /sys/class/power_supply/ucsi-source-psy-*/; do
    [ -r "$_pd/online" ] && [ "$(cat "$_pd/online")" = "1" ] || continue
    [ -r "$_pd/voltage_max" ] && [ -r "$_pd/current_max" ] || continue
    # Both are microvolts and microamps, so the product needs 1e6 on each.
    _v=$(cat "$_pd/voltage_max"); _i=$(cat "$_pd/current_max")
    [ "$_v" -gt 0 ] && [ "$_i" -gt 0 ] || continue
    PD_WATTS=$(( _v / 1000000 * _i / 1000000 ))
    break
done

# The hardware's own ceiling, for context on the clock samples below.
CPU_MAX_MHZ=0
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]; then
    for _m in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do
        _mhz=$(( $(cat "$_m") / 1000 ))
        [ "$_mhz" -gt "$CPU_MAX_MHZ" ] && CPU_MAX_MHZ=$_mhz
    done
fi

if [ "$REQUIRE_PERF" = true ]; then
    if [ "$GOVERNORS" != "performance" ]; then
        echo "REFUSING: -g given but the CPU governor is '$GOVERNORS', not performance." >&2
        [ -n "$EPP" ] && \
        echo "  (energy_performance_preference is '$EPP' - on intel_pstate that is a" >&2 && \
        echo "   SEPARATE knob, and power-profiles-daemon sets it without touching the" >&2 && \
        echo "   governor, so a 'performance' power profile is not enough on its own.)" >&2
        echo "  cpupower is not in the dev image and /sys is mounted read-only here," >&2
        echo "  so set it from a HOST shell, then re-run this:" >&2
        echo "    sudo cpupower frequency-set -g performance" >&2
        exit 1
    fi
    if [ "$ON_AC" = "no" ]; then
        echo "REFUSING: -g given but this machine is on battery." >&2
        echo "  Platform firmware caps the clock well below what the governor asks" >&2
        echo "  for, so the tail of this run would not be comparable with a run on" >&2
        echo "  mains. Plug it in, or drop -g to take the run anyway." >&2
        exit 1
    fi
    # On mains is not the same as adequately powered. A 15 W USB-C contract on the
    # reference box left on_ac_power reading "yes", the battery reading "Charging",
    # and the mean tick 1.5x worse with 3x the tail.
    if [ -n "$PD_WATTS" ] && [ "$PD_WATTS" -lt 60 ]; then
        echo "REFUSING: -g given but the USB-C supply negotiated only ${PD_WATTS} W." >&2
        echo "  That is far short of what this machine draws under load, so the" >&2
        echo "  package stays power-limited even though it reports being on mains." >&2
        echo "  Measured cost of exactly this: 1.5x the mean tick and 3x the tail." >&2
        echo "  Use the full-power adapter, or drop -g to take the run anyway." >&2
        exit 1
    fi
fi

ARGS=(launch/full_stack_nmpc.launch.py with_targets:=true)
[ -n "$PARAMS" ]        && ARGS+=("params_file:=$PARAMS")
[ -n "$TRACE" ]         && ARGS+=("trace_file:=$TRACE")
[ "$TELEMETRY" = true ] && ARGS+=(telemetry:=true)
ARGS+=("${EXTRA[@]}")

echo "running ${DURATION}s trial -> $LOGFILE" >&2
echo "  params: $EFFECTIVE" >&2
[ -n "$TRACE" ] && echo "  trace:  ${TRACE} (_left / _right)" >&2
if [ -n "$PD_WATTS" ] && [ "$PD_WATTS" -lt 60 ]; then
    echo "  mains: $ON_AC, but the USB-C supply negotiated only ${PD_WATTS} W" >&2
    echo "         <-- too little to run this box at full clock. 'on mains' is not" >&2
    echo "         the same as 'enough power': measured on the reference box, a" >&2
    echo "         supply this small cost 1.5x the mean tick and 3x the tail" >&2
elif [ "$GOVERNORS" = "performance" ] && [ "$ON_AC" != "no" ]; then
    echo "  governor: performance   mains: $ON_AC" >&2
else
    echo "  governor: $GOVERNORS   mains: $ON_AC" >&2
    echo "            <-- NOT measurement-grade; the tail of this run is not" >&2
    echo "            comparable with a pinned, mains-powered one (see" >&2
    echo "            tools/nmpc_timing/README.md)" >&2
fi
# Record what produced this log, and the settings most likely to be the subject of
# whatever comparison it gets used in. A log that cannot say which config it came
# from is worth very little a day later.
{
    echo "# pat_arm_control trial: $(date -Is), ${DURATION}s"
    echo "# params_file: $EFFECTIVE"
    [ "${#EXTRA[@]}" -gt 0 ] && echo "# launch_args: ${EXTRA[*]}"
    echo "# cpu_governor: $GOVERNORS"
    [ -n "$EPP" ] && echo "# energy_perf_pref: $EPP"
    echo "# on_ac_power: $ON_AC"
    [ -n "$PD_WATTS" ] && echo "# usb_pd_watts: $PD_WATTS"
    grep -E '^\s+(control_hz|warm_start|max_iter|N|Ts|full_nonlinear|sqp_iters):' \
        "$EFFECTIVE" 2>/dev/null | sed 's/^/#   /'
} > "$LOGFILE"

# Sample the clock through the run, as provenance rather than as a gate. The PEAK
# does not discriminate: measured on the reference box, an underpowered 15 W supply
# still let the package burst to 4800 of its 5000 MHz momentarily, because turbo
# can draw on the battery for a moment even when sustained power is capped. The
# SUSTAINED figure -- the mean of the fastest core at each sample -- is the one
# that moves, so both are recorded and neither is trusted alone.
FREQ_FILE=$(mktemp)
echo "0 0 0" > "$FREQ_FILE"
if [ "$CPU_MAX_MHZ" -gt 0 ]; then
    (
        peak=0; sum=0; n=0
        while :; do
            top=0
            for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
                mhz=$(( $(cat "$f" 2>/dev/null || echo 0) / 1000 ))
                [ "$mhz" -gt "$top" ] && top=$mhz
            done
            [ "$top" -gt "$peak" ] && peak=$top
            sum=$(( sum + top )); n=$(( n + 1 ))
            echo "$peak $sum $n" > "$FREQ_FILE"
            sleep 1
        done
    ) &
    SAMPLER_PID=$!
fi

# Startup costs ~8 s (model load, DDS discovery, the first few solves) and those
# windows are always the worst ones; summarise_timing.py drops them by default.
timeout -s INT "$((DURATION + 10))" ros2 launch "${ARGS[@]}" >> "$LOGFILE" 2>&1

if [ -n "${SAMPLER_PID:-}" ]; then
    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
fi
read -r FREQ_PEAK FREQ_SUM FREQ_N < "$FREQ_FILE" 2>/dev/null || { FREQ_PEAK=0; FREQ_N=0; }
FREQ_MEAN=0
[ "${FREQ_N:-0}" -gt 0 ] && FREQ_MEAN=$(( FREQ_SUM / FREQ_N ))
rm -f "$FREQ_FILE"

sleep 3
if pgrep -f '[a]rm_control_node' >/dev/null 2>&1; then
    echo "WARNING: nodes survived teardown; kill them before the next trial" >&2
    pgrep -af '[a]rm_control_node' >&2
fi

WINDOWS=$(grep -c 'ticks @' "$LOGFILE" 2>/dev/null || echo 0)
echo "done: $WINDOWS report windows captured" >&2

# Recorded for provenance, so two runs can be told apart after the fact. Not a
# gate: see the sampler's comment on why neither number cleanly detects a capped
# box for a loop this light. The PD wattage above is the reliable signal.
if [ "$CPU_MAX_MHZ" -gt 0 ] && [ "${FREQ_PEAK:-0}" -gt 0 ]; then
    printf '# cpu_mhz_peak: %d of %d rated\n' "$FREQ_PEAK" "$CPU_MAX_MHZ" >> "$LOGFILE"
    printf '# cpu_mhz_sustained: %d (mean of the fastest core per sample)\n' \
        "$FREQ_MEAN" >> "$LOGFILE"
    echo "  CPU: peak ${FREQ_PEAK} MHz, sustained ${FREQ_MEAN} MHz, rated ${CPU_MAX_MHZ}" >&2
fi
