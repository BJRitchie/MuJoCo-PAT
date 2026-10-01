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

Inside the dev container (`docker compose exec dev bash`):

```bash
# baseline, using the installed config
./tools/nmpc_timing/run_trial.sh -d 300 -o /tmp/base.log

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
