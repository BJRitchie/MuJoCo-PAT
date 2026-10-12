# nmpc_timing — measuring the NMPC control loop

Two ways to get timing numbers out of `pat_arm_nmpc`, for choosing `control_hz`,
`qp.warm_start`, `qp.max_iter`, `N`, `Ts` and the slack weights from measurement
rather than from feel.

- **A live trial** against the simulator: realistic, includes ROS I/O and both
  arms, but noisy and ~5 minutes per data point.
- **The GTest studies**: one process, one arm, no ROS, seconds per data point and
  repeatable. Use these to narrow the field, then confirm with a live trial.

Both read the same instrumentation, so the numbers mean the same thing.

## Live trials

Two things about the machine have to be right before a run is worth quoting, and
both are set **on the host**, before attaching to the container.

**Pin the CPU governor.** This box has been seen spread over 464–2113 MHz on
`powersave`, and that shows up directly in the tail:

```bash
sudo cpupower frequency-set -g performance                   # host shell
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor     # confirm
```

It cannot be done from inside the container: `/sys` is mounted read-only there, so
the write fails whatever is installed and whoever runs it — installing `cpupower`
in the image would not help.

**Plug the laptop into the FULL-POWER adapter.** Two separate traps here.

On battery the reference box pins `performance` and still only reaches ~3.1 GHz of
its 5.0 GHz under load. Nothing in cpufreq sysfs shows it: `scaling_max_freq` still
reads 5.0 GHz, `energy_performance_preference` still reads `performance`, and RAPL's
long-term limit still reads 200 W — the cap is applied by platform firmware below
all of that.

And **"on mains" is not the same as "enough power"**. A 15 W USB-C charger reports
`AC online=1`, keeps the battery reading `Charging`, passes every check above — and
still cost **1.5x the mean tick and 3x the tail** (p95 5410 vs 2166 us, which is
over half the 10 ms period against about a fifth). Measured, same build and config,
90 s each.

| | mean L/R | p95 | max |
|---|---|---|---|
| full-power adapter | 966 / 948 | 2166 / 1638 | 2707 / 2094 |
| 15 W USB-C supply | 1440 / 1464 | 5410 / 5420 | 7530 / 6472 |

`run_trial.sh` records `# cpu_governor:`, `# on_ac_power:`, `# usb_pd_watts:`,
`# cpu_mhz_peak:` and `# cpu_mhz_sustained:` in every log, and with `-g` refuses to
start unless the governor is `performance`, the machine is on mains, **and** any
USB-C supply has negotiated at least 60 W.

Do not gate on the peak clock: it does not discriminate. The 15 W supply still let
the package burst to 4800 of its 5000 MHz in one run (and only 3857 MHz in the
next), because turbo can draw on the battery momentarily even when sustained power
is capped. The sustained figure moves in the right direction and both are logged,
but the negotiated wattage is the signal worth refusing on.

Then, inside the dev container (`docker compose exec dev bash`):

```bash
# baseline, using the installed config
./tools/nmpc_timing/run_trial.sh -g -d 300 -o /tmp/base.log

# a variant: copy the config, change ONE knob, pass it -- no rebuild
cp src/pat_arm_nmpc/config/nmpc.yaml /tmp/try.yaml
#   ... edit /tmp/try.yaml ...
./tools/nmpc_timing/run_trial.sh -d 300 -o /tmp/try.log /tmp/try.yaml

./tools/nmpc_timing/summarise_timing.py /tmp/base.log /tmp/try.log
```

```
run                           wins   ticks  period     mean     max     p95   over it_mean it_max  retr  skipped miss_run    >25ms
base.log                       566  112961    5000     3084   34008   10089  4.18%    8.07     53     0       50       10        1
try.log                        566   56597   10000     3187   22768    8715  0.04%    8.07     48     0        1        5        0
```

`--per-arm` splits left from right (they do not always behave alike), `--warmup N`
changes how many startup windows are dropped, `--spike-ms` sets the last column's
threshold.

### Seeing the distribution

`--plot` adds, per run and per arm, the tick-time histogram and a timeline of
per-window maxima. ASCII on purpose: it works over any shell into the container,
and the image's matplotlib is broken (numpy 1.x/2.x ABI clash).

```bash
tools/nmpc_timing/summarise_timing.py --per-arm --plot data/nmpc_timing/base.log
```

```
base.log:left — 289 windows, 28897 ticks @ 10000 us period
  tick time, all ticks
       <=2500 us  ################################################    14339  49.6%
       <=5000 us  ###############################################     14147  49.0%
        >5000 us  #                                                     411   1.4%
  per-window max: p50 4635  p90 6442  p99 9608  max 40229 us
  per-window max over the run, ' '=3575 '#'=40229 us (log scale)
        .   ..  -. .... :     . ..  .#  .  .. .... :.. .:.  :.          .:  ..
```

**The two views are not interchangeable.** The histogram covers *every* tick — the
node bins each one as it happens, so summing per-window bins is the true
distribution. The timeline and percentiles are over per-window *maxima*, because a
worst case is all the log keeps per window. "49.6% of ticks under 2.5 ms" says
nothing about how bad the worst one was.

The timeline is one cell per window in run order, so clustering is visible —
spikes bunch after a setpoint change rather than scattering uniformly.

Note the histogram's resolution: `PhaseStat::kEdgeUs` in
`arm_task_space_controller.h` was chosen around a 5 ms period, so at
`control_hz: 100` nearly everything lands in two bins. `--plot` says so when it
happens. For the real distribution, record a trace instead.

## Per-tick traces

`trace_file:=` records **every** control tick and streams it to CSV, one file per
arm, so the distribution is not limited to the report's eight fixed bins.

**`plot_trace.py` needs a trace, and a trial only writes one when asked.** Run this
first — the `-T` is what produces the file, and it is a capital T:

```bash
# 1. produce the trace (-T), with the governor pinned (-g, see Live trials above)
./tools/nmpc_timing/run_trial.sh -g -d 300 -o /tmp/base.log -T data/nmpc_timing/trace.csv

# 2. then plot it. NOTE the _left / _right: one file per arm, spliced into the
#    name you passed, so `-T .../trace.csv` writes trace_left.csv + trace_right.csv
tools/nmpc_timing/plot_trace.py --bins 14 --phases --top 3 data/nmpc_timing/trace_left.csv
```

Without `-T` the run only produces the 1 Hz report log, which is what
`summarise_timing.py` reads; `plot_trace.py` on that log, or on a trace file that
was never written, fails with `FileNotFoundError`. Launching by hand works too:

```bash
ros2 launch launch/full_stack_nmpc.launch.py with_targets:=true \
    trace_file:=data/nmpc_timing/trace.csv
```

```
  5920 ticks over 59.2 s
        1731-2426     ############################################    2054  34.7% (cum  34.7%)
        2426-3121     ######################################          1773  29.9% (cum  64.6%)
        ...
        9379-10074    .                                                  3   0.1% (cum  99.9%)  <-- deadline
  tick_us: min 1731  p50 2741  p90 3960  p99 6000  p99.9 9475  max 11465 us
  over the 10000 us period: 3 (0.05%)
  mean tick by phase
               qpsolve ##################################             2287.2 us  77.6%
      rollout_dynamics ####                                            300.8 us  10.2%
```

`--column qpsolve` plots any single column, `--phases` breaks the mean tick down,
`--top N` lists the slowest ticks with their phase split and QP status — which is
how you find out what a spike actually was.

The CSV columns are `t_us`, `tick_us`, the eleven phase durations, `hpipm_us`,
`qp_iters`, `qp_status`, `qp_iters1`, `qp_status1`, `mode`, `retried` — the `1`
pair is the FIRST solve of the tick on its own, where the unsuffixed pair is the
tick as a whole, so a warm solve that burned the cap and a cold retry that
converged are distinguishable. A `#` provenance line above the header records the
arm and the configuration. It is ordinary CSV, so pandas or gnuplot can take over
for publication plots; note the `#` line if you parse it yourself (`numpy`'s
`genfromtxt` wants `skip_header=1`, not `comments='#'`, or it reads the comment
as the header).

### What tracing costs the control loop

The tick only stores one preallocated row: no file I/O, no allocation, no syscall.
Rows are drained and appended **from the 1 Hz diagnostics callback**, not the tick.
That matters for two reasons:

- Writing only at exit does not work. `ros2 launch` escalates its shutdown to
  SIGKILL on a grace period, and it killed the process partway through a one-shot
  write — the first attempt at this produced 824 of 4342 ticks in one file and 411
  in the other. Draining each second means a killed run keeps everything up to the
  last drain.
- `trace_capacity` (default 60 s of ticks) only has to cover one drain interval,
  so memory stays flat however long the run is. If it ever fills, recording stops
  rather than wrapping or growing, and the report line shows `trace N/CAP FULL`.

File I/O does land on the executor thread once a second, alongside the report line
that was already being logged there. Tracing is a diagnostic mode; leave it off for
a run whose absolute worst-case tick is the measurement.

It does **not** produce expensive ticks, measured rather than assumed — see the
flush test below. Across four 309 s traces the worst post-flush tick is 2.7–2.9 ms
while the worst tick elsewhere is 6.2–11.1 ms, and the post-flush share of each
run's p99 outliers sits inside Poisson noise every time.

### Where a tail comes from

`plot_trace.py` shows the distribution; `tail_source.py` asks what produced its
tail, which is what decides whether the tail is worth tuning:

```bash
tools/nmpc_timing/tail_source.py data/nmpc_timing/trace_{left,right}.csv
tools/nmpc_timing/tail_source.py --pct 99.9 --top 15 data/nmpc_timing/trace_left.csv
```

Three tests, all on traces you already have:

- **The 1 Hz flush**, the obvious suspect since it writes on the control
  callback's executor. Judged on the *outliers* — what share of the run's p99
  ticks were post-flush, against what that group's size alone would give, with
  Poisson error bars, plus the worst post-flush tick against the worst tick
  elsewhere. It also histograms where in the second the slow ticks fall; a flush
  cause concentrates them in the first bin.

  It deliberately does **not** rule on the difference in means. Across two 309 s
  runs of one unchanged config the post-flush mean difference came out −37/−48 µs
  and then +33/+57 µs, each time with a bootstrap interval excluding zero. A
  systematic subsample can be significant within a run and reverse in the next,
  because the machine's state drifts across a run and the 1 s-boundary ticks
  sample that drift differently. The mean difference is printed, unruled, as a
  reminder.
- **Cross-arm co-occurrence.** The two arms are separate processes solving
  independent QPs, so outliers that coincide in wall-clock time have a cause
  outside both controllers. It scans ±1 s for the clock offset that aligns them
  best (each node's `t_us` starts at its own bring-up) and compares the hit rate
  against what independence would give.
- **The slowest ticks, split into iteration count versus cost per iteration.** A
  tick that is slow because HPIPM needed more iterations is a solver problem; one
  that ran the *same* iterations at several times the median cost per iteration
  is the machine.

Across two 309 s runs of the committed config (~30,700 ticks per arm each) all
three tests agree, and none of them point at this package:

- the worst post-flush tick is **2.7–2.9 ms** against **6.2–11.1 ms** elsewhere,
  in all four traces, and the post-flush share of each run's p99 outliers is
  inside Poisson noise every time;
- the two arms' p99 outliers co-occur within ±20 ms at **43% and 52% against 4%
  expected** — 11× and 13× enrichment, replicated;
- the worst ticks run **7–13 iterations at 5–16× the median ~91 µs per
  iteration**.

Same arithmetic, executed more slowly, in two independent processes at once. That
is the machine, not a QP that got harder. Run-to-run variation on the unchanged
config is itself large — worst tick 10.9/11.1 ms in one run and 8.9/6.2 ms in the
next, with 3 deadline misses and then 0 — so pin the governor to `performance`
before reading anything into a tail, and compare paired and interleaved.

Pin it **on the host**, and plug the laptop in: `cpupower` is not installed in the
dev image, `/sys/devices/system/cpu/.../scaling_governor` is not writable from
inside the container, and on battery the clock is capped by platform firmware
whatever the governor says. `run_trial.sh -g` refuses to start unless both are
right (see Live trials).

```bash
sudo cpupower frequency-set -g performance    # host shell, before the trial
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # confirm: performance
cat /sys/class/power_supply/AC*/online                      # confirm: 1
```

### Reading the raw report

`summarise_timing.py` aggregates, but each window is also readable directly — the
node logs one per second per arm:

```
pat_arm_nmpc[left] 100 ticks @ 10000 us | period jitter mean 44 max 224 us
  | callback wall mean 3143 max 4477 / cpu mean 3141 max 4478 us
  | deadline misses 0 (longest run 0) | skipped periods 0
profile over 100 ticks (us)  TICK mean=3050.1 max=4388.2 over_period=0/100
  hist(tick) <=50:0 <=100:0 <=250:0 <=500:0 <=1000:0 <=2500:12 <=5000:88 >5000:0
  phases roll_dyn(n=4100 mean=11.0 max=88.2) ... qpsolve(n=100 mean=840.1 max=1893.7) ...
  unattributed=102.9 us/tick
  qp_iter/tick mean=2.0 max=2.0 (cap 100/solve)  hpipm_us mean=816.5 max=1538.0  retries=0  | peak slack: box=0.0 torque=0.0
```

What each part is for:

| Field | Reads on |
|---|---|
| `period jitter` | the scheduler. Tens of µs is healthy; large values mean the loop is not being woken on time, which is a different problem from being slow. |
| `callback wall` vs `cpu` | whether a slow tick was *work* or *preemption*. Close together means the work really took that long and no kernel tuning will help. |
| `deadline misses (longest run)` | how often the period was blown, and whether the misses were scattered or one sustained outage. |
| `skipped periods` | periods with no tick at all — the loop ran open-loop through them. |
| `TICK max` + `hist(tick)` | the tail. `max` is what a hard deadline must accommodate; the histogram shows whether it was one freak or a fat distribution. |
| `phases` | where the time goes. `qpsolve` is usually dominant; `roll_dyn`/`roll_intg`/`pinv` together are the rollout. |
| `unattributed` | tick time no phase claimed. If it grows, the profile has a blind spot. |
| `qp_iter/tick` | why a solve was slow. Solve time is ~linear in iterations, so a 20x tick is ~20x the iterations. **At or above the cap means a solve failed** and `retries` shows the cold restart. |
| `peak slack` | whether the soft constraints are actually carrying a violation (nominally ~1e-16). Reads 0 when both groups are hard. |

### Trials are easy to invalidate

- **Run the timeout inside the container.** `run_trial.sh` does. A host-side
  `timeout` around `docker compose exec` kills only the client and leaves the
  launch running; the next trial then shares CPU *and topics* with it. The script
  refuses to start if a sim or NMPC node is already up.
- **Do nothing else on the box.** A `colcon build` or `colcon test` running
  alongside a trial inflates its maxima severalfold.
- **Drop the startup windows.** Model load, DDS discovery and the first solves make
  the first few windows the worst in any run; `--warmup 10` is the default.
- **Don't trust a single `max`.** It is one tick out of ~100,000. `p95` and the
  spike-window count are the stable tail statistics; use `max` for the bound you
  must design to, not for comparing two configs.
- **`pkill -f` is dangerous here.** `docker-compose.yml` sets `pid: host`, so
  pattern-matching kills inside the container can match processes on the host. Kill
  by explicit PID.

## Deterministic studies

Faster, repeatable, and they take the knob directly in C++ rather than through
YAML — one process, one arm, no ROS, no simulator.

```bash
colcon build --packages-select pat_arm_nmpc --cmake-args -DBUILD_TESTING=ON
./build/pat_arm_nmpc/test_pat_arm_nmpc \
    --gtest_also_run_disabled_tests --gtest_filter='DISABLED_WarmStartStudy.*'
```

```
mode  tick_mean  tick_max  over_bud  it_mean   it_max retries    avail    ee_err
2          1627     44676     0.030     4.80      113       2    1.000   0.09585
1          1811      4192     0.000     7.41       14       0    1.000   0.09585
0          1838      3855     0.000     7.41       14       0    1.000   0.09585
```

The studies live in `src/pat_arm_nmpc/test/test_pat_arm_nmpc.cpp`:

- `DISABLED_WarmStartStudy.Ladder` — `qp.warm_start` across a setpoint step.
- `DISABLED_ConstraintStudy.Ladder` / `.BoxIsActuallyEnforced` — soft vs hard
  constraints and the slack weights.
- `RealtimeProfile.ControlTickAllocationCountIsBounded` — heap allocations per
  tick (not disabled; it prints the count and asserts a ceiling).

To study a knob the ladders do not cover, copy `DISABLED_WarmStartStudy.Ladder`
and loop over it. `runClosedLoop(params, scenario, ticks)` returns `RunMetrics`
with tick timing, QP iterations, availability, EE error and realised constraint
violations. Use `Scenario::step_at_tick`/`step_offset` for a mid-run target jump —
a single fixed setpoint never stresses the solver, so a study without one will
report that every setting is equally fine.

**Mind the units of comparison.** Harness ticks run faster than live ones: one
arm, no ROS I/O, no simulator sharing the CPU. Compare harness against harness and
live against live, never across.
