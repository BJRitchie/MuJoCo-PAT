# thesis_eval: the controller comparison, end to end

Runs the four arm controllers (SQP NMPC, iterative-linearisation NMPC, OMPC, SMC)
through every scenario of the thesis comparison in `contact_bench`, offline and
deterministically, then turns the recorded runs into figures and LaTeX tables.

```
studies.py      the matrix as data: which runs, with which arguments
run_study.py    runs it (in the dev container): one contact_bench per cell, cached
tune.py         tunes all four controllers by one rule (in the dev container)
tuned.json      tune.py's result: the `tuned` variant's --set overrides
common.py       controller names/colours, data loading, decoding cell names
figures.py      the figures (on the host, in a venv)
tables.py       the LaTeX tables (on the host, in a venv)
```

## Controllers and variants

| Label | Run as | |
|---|---|---|
| SQP-NMPC | `contact_bench --law nmpc` | `nmpc.full_nonlinear: true`, multiple-shooting SQP |
| LMPC | `--law nmpc --set nmpc.full_nonlinear=false` | iterative linearisation, otherwise the same file (`iter` in the data) |
| OMPC | `--law ompc` | one-step MPC |
| SMC | `--law smc` | sliding mode |

Every study runs in two **variants**. `tuned`, the default and the one the
thesis reports, adds `tuned.json`'s `--set` overrides; its data goes in
`data/thesis_eval/tuned/<study>/` and its figures in `docs/assets/thesis/`.
`shipped` uses the committed YAML as it is; its data goes in
`data/thesis_eval/<study>/` and its figures in `docs/assets/thesis/shipped/`.
Select it with `--variant shipped` on `run_study.py`, `figures.py` and
`tables.py`. `fig15_shipped_vs_tuned` and `tab_tuned` compare the two and are
always written to `docs/assets/thesis/`. The `sensitivity` study sweeps each
knob around the variant's own value (tuned values, as multiples, for `tuned`).

Where a figure sweeps a setting, a dashed grey line marks the controllers' own
value: speed 1x, mass 1, torque 0, torque limit 1, base damping 0, flight sensor
error 1x, extra delay 0, feedforward gain 0, the 60 s estimator warm-up, and each
knob's tuned value in the sensitivity figure. In the timing figure, dashed lines
in each NMPC's colour mark its own horizon N.

## Studies

All runs are offline in `contact_bench`: the left arm is scored and the right arm
holds its start pose unless `--partner` says otherwise. The base floats on a
**frictionless** table, as the bus would in orbit (`studies.PLANT`,
`--base-damping 0`); only the `damping` study varies it. Tracking is scored
against plant truth.

**State sources.** What the controllers are told about the plant
(`studies.SOURCES`):

| Label | What the controller gets |
|---|---|
| `exact` | the plant's own state |
| `raw` | the flight sensors as they arrive: the last 2 Hz relative-nav fix (0.5 s old), the last star-tracker reading, the raw gyro, velocity differenced between fixes; raw encoders. What the estimation is compared with |
| `ekf` | `pat_gnc`'s `ImuEkf` on the flight sensors (IMU at 500 Hz, star tracker, relative nav); raw encoders |
| `ekf-jkf` | the EKF and a per-joint Kalman filter on the encoders (`estimation/joint_kalman_filter.h`): the full flight estimation |

The flight sensors are `sensors_flight.yaml`'s (a STIM300-class IMU, a 0.1 mrad
star tracker at 5 Hz, a 15 mm relative-nav fix at 2 Hz, 0.5 s late); the encoders
read joint position to 0.5 mrad (an assumption: the encoder is not characterised)
and velocity to 0.01 rad/s. Each channel draws from its own seeded stream, and
the flight sources share their streams, so for one seed raw and EKF see identical
sensor noise. Every flight source runs for 60 s before the task with
the arms at rest (`--nav-warmup`), as a navigation filter would be running long
before an arm task in flight.

| Study | What varies | Figure |
|---|---|---|
| `nominal` | 6 cm diagonal and 12 cm steps, the circle and the rectangle, exact state | 1-4 |
| `balanced` | partner arm `hold` / `mirror` (equal and opposite, so the base reactions cancel) / `same` | 6 |
| `speed` | trajectory speed 1-4x (circle), 1-3x (rectangle) | 5, 7 |

The six robustness studies (`disturbance`, `model_error`, `damping`,
`torque_limit`, `noise`, `delay`) each run on both the circle and the rectangle.
Figure 8 plots the rectangle, whose corners separate the controllers.

| Study | What varies | Figure |
|---|---|---|
| `disturbance` | constant unmodelled torque on the scored arm's first joint, 0-0.1 N·m | 8 |
| `model_error` | link mass in the plant 0.5-3x the model's | 8 |
| `damping` | base damping 0-50 (everything else runs at 0) | 8 |
| `torque_limit` | the 0.25 N·m clamp scaled 0.5-2x (in the controllers' own config too) | 8 |
| `constraint` | a +y reach of 4-13 cm that straightens the elbow into its −0.05 rad limit, and joint3 started 0.088 rad outside its box | 11 |
| `noise` | flight sensing (`ekf-jkf`) with every sensor's error scaled 0.5-8x together, the filters told the scaled figures, 20 seeds | 8 |
| `delay` | flight sensing with the whole state handed over 0-5 control ticks later still, 5 seeds | 8 |
| `observer` | state source (exact / raw sensors / EKF + joint KF) × disturbance-observer feedforward (0, 0.5, 1.0) on five scenarios (circle, 6 cm step, mirrored circle, +0.05 N·m, 2x mass), 10 seeds | 12, 13 |
| `estimator` | every state source on the circle and the rectangle, 10 seeds | 14 |
| `warmup` | the time the base estimator runs before the task, 0-60 s, EKF against raw sensors, rectangle, 10 seeds | 16 |
| `sensitivity` | each controller's knobs, one at a time, around the variant's values | 9 |
| `timing` | wall time per tick, 5 pinned processes per controller, and the NMPC against N | 10 |

## Tuning

`tune.py` tunes every controller by the same rule: minimise the geometric mean of
the EE RMSE over five **tuning scenarios** (circle and rectangle at 1x and 2x, and
the 6 cm step). Exact state, partner held, frictionless table. The constraints are:

- under 5% of joint-ticks at the torque limit in every tuning scenario;
- no safe-mode entry;
- no tick over the 10 ms period;
- for the NMPCs, a look-ahead `N·Ts ≥ 0.66 s` (N ≥ 20 at the shipped `Ts`).

The mean is geometric so that each scenario counts by its relative change; an
arithmetic mean is dominated by the step's ~30 mm. An infeasible setting scores
`1000 + violation`, so the search walks to feasibility first. The search is
coordinate descent over ladders of multiples (orders of magnitude while
infeasible), until a pass changes nothing. Every other study is the test set.

The current `tuned.json` came from two runs. The first started at the shipped
values, on a table with 0.05 N·s/m of drag. The table was then made frictionless
for every study, and the controllers were retuned from that first result
(`--start-from`). SQP's earlier settings broke a limit without the drag, and it
moved to a new optimum; LMPC moved; OMPC's and SMC's earlier settings were still
the best the search could find.

SQP was then retuned once more, alone, so that it is more accurate than every
other controller without being harder on the actuators: `--beat-others` requires
its RMSE in each tuning scenario to be no more than the best of the other three,
and `--torque-cap-from-others max` its `∫‖τ‖²dt` no more than the largest of
theirs. No setting met every requirement. The one chosen (`tuned.json` marks it
`feasible: false`) is the most accurate on the circle, rectangle and both 2x
shapes, with torque below the others' on the circles and up to 1.6x the largest
on the rectangle. It misses three rules: the 6 cm step (35.9 mm against OMPC's
29.3), the rectangle's torque, and 8.6% of joint-ticks at the limit on the 2x
rectangle (the rule is 5%). The earlier, unconstrained SQP and a version capped
at the others' mean torque are kept as `data/thesis_eval/tuning/sqp_unconstrained.json`
and `sqp_meancap.json`.

```bash
python3 tools/thesis_eval/tune.py sqp --jobs 4 --passes 8 \
    --torque-cap-from-others max --beat-others --start-from PREVIOUS_TUNED.json
```

```bash
python3 tools/thesis_eval/tune.py sqp iter ompc smc --jobs 3      # in the container
python3 tools/thesis_eval/tune.py sqp iter ompc smc --jobs 3 --start-from OLD_TUNED.json
python3 tools/thesis_eval/run_study.py run all --jobs 3          # tuned is the default
```

Every evaluation is logged in `data/thesis_eval/tuning/<controller>[_restart].json`.
Tuning made OMPC, SMC and LMPC *worse* by the objective, because their shipped
settings break a limit: SMC rails the torque limit on ~100% of joint-ticks, OMPC
rails it on the 2x rectangle, and LMPC enters safe mode there. The tuned settings
are the most accurate settings that respect the limits. `fig15_shipped_vs_tuned`
and `tab_tuned.tex` show both.

## 1. Run the studies (inside the dev container)

```bash
docker compose exec dev bash
source install/setup.bash
python3 tools/thesis_eval/run_study.py list                    # studies and sizes
python3 tools/thesis_eval/run_study.py run all --jobs 3        # everything but timing is quick
python3 tools/thesis_eval/run_study.py run all --jobs 3 --variant shipped
python3 tools/thesis_eval/run_study.py status
```

Each cell is cached under `data/thesis_eval/<study>/cells/` by a hash of its
command line, the bench binary and every config file the bench reads, so a rerun
only runs what changed. `--force` reruns, `--select TEXT` runs a subset,
`--dry-run` prints the commands, `--quick` uses few seeds.

**Timing** is the one non-deterministic measurement. The `timing` study runs
serially and is refused unless the machine is on mains with the `performance`
governor (set it from a host shell: `sudo cpupower frequency-set -g performance`).
Pin it to one physical core: `run timing --pin 4,5` (`lscpu -e=CPU,CORE,MAXMHZ`
shows which CPUs share a core).

Each study's `summary.csv` (one row per cell, ~60 metric columns) and
`manifest.json` (git revision, binary hash, machine state, every command) are
kept in git. The per-tick series and the cache are not.

## 2. Make the figures and tables (on the host)

```bash
python3 -m venv ~/.venvs/thesis_eval
~/.venvs/thesis_eval/bin/pip install -r tools/thesis_eval/requirements.txt
~/.venvs/thesis_eval/bin/python tools/thesis_eval/figures.py --all
~/.venvs/thesis_eval/bin/python tools/thesis_eval/tables.py
~/.venvs/thesis_eval/bin/python tools/thesis_eval/figures.py --all --variant shipped
~/.venvs/thesis_eval/bin/python tools/thesis_eval/tables.py --variant shipped
```

Output goes to `docs/assets/thesis/` (tuned; PDF for the thesis, PNG to look
at) and `docs/assets/thesis/tables/` (bare `tabular`s; needs `booktabs`), and
the shipped set to `docs/assets/thesis/shipped/`. They read only
the CSVs, so they can be regenerated without rerunning anything.

## Findings

EE RMSE in mm, tuned settings, frictionless table. Exact state unless stated
otherwise, no feedforward; seeded runs are means over 10-20 seeds. The data is in
`data/thesis_eval/tuned/` (and `data/thesis_eval/` for shipped).

Every figure marks a run that entered safe mode with a black cross.

**Accuracy and torque** (exact state). RMSE, then `∫‖τ‖²dt` in the scoring
window:

| | SQP | LMPC | OMPC | SMC |
|---|---|---|---|---|
| circle 1x | **0.32** / 0.015 | 0.49 / 0.015 | 0.53 / 0.015 | 0.68 / 0.015 |
| rectangle 1x | **0.48** / 0.087 | 19.6 / 0.053 | 33.1 / 0.045 | 26.2 / 0.047 |
| 6 cm step | 42.6 / 0.008 | 42.7 / 0.002 | **35.2** / 0.007 | 39.9 / 0.004 |
| circle 2x | **0.55** / 0.102 | 0.78 / 0.110 | 0.89 / 0.109 | 1.11 / 0.109 |
| rectangle 2x | **3.7** / 0.222 | 46.3 / 0.223 | 61.6 / 0.155 | 49.5 / 0.175 |
| circle 3x (test) | **13.1** (safe mode) | 144 (safe mode) | 16.4 | 18.0 |
| rectangle 3x (test) | **15.7** | 164 (safe mode) | 66.2 | 62.5 |
| circle 4x (test) | 123 (safe mode) | 146 (safe mode) | 43.7 | 45.5 |

SQP is the most accurate on every trajectory up to 3x, using the same torque as
the others on the circles and 1.6-1.9x their torque on the 1x rectangle, which
buys a 40-70x smaller error at its corners; at 2x its torque matches the most
demanding of the others. It is not the most accurate on the 6 cm step, where
OMPC moves smoothly and slowly. At 3x on the circle and at 4x SQP's hard torque
rows go infeasible and it enters safe mode; the closed-form laws degrade but
never fail. Shipped, LMPC is the most accurate on the circle (0.31 against SQP's
2.31) and SMC rails the 0.25 N·m limit on 99.7-99.9% of joint-ticks.

**Base attitude.** On the held-partner circle every law gives `∫ψ²dt` ≈ 0.36
rad²·s: the reaction is set by the motion, not the law. A mirrored partner
cancels it by 7 orders of magnitude on the circle and by 6-8x on the rectangle,
whose corners the two arms do not take identically.

**Disturbance and model error** (rectangle, test). A constant 0.05 N·m on joint2
costs SQP 9.5 against 48-59 for the others; a 2x link mass, 7.8 against 32-48.

**Flight sensing.** The relative-nav fix is an AVGS-type cooperative video
guidance sensor (5 Hz, 0.15 s late, 4 mm along the line of sight and 0.2 mm
across it at the scene's 0.9 m; see the main README's estimator section). With
the full estimation (EKF + joint Kalman filters) the base position the arm is
handed is 2.4 mm rms, and the controllers keep their order:

| flight sensing (exact state) | SQP | LMPC | OMPC | SMC |
|---|---|---|---|---|
| circle | 2.4 (0.32) | 2.1 (0.49) | 2.2 (0.53) | 1.9 (0.68) |
| rectangle | **2.4** (0.48) | 19.4 (19.6) | 32.6 (33.1) | 25.7 (26.2) |

On the circle every controller is within the estimate's own error, so all four
land at 2 mm. On the rectangle the others' own tracking error dominates and SQP
keeps an 8-13x lead. Scaling every sensor's error up (test) costs SQP in
proportion (8x: 18.8 mm) and leaves OMPC and SMC almost unchanged; LMPC enters
safe mode from 4x. Delaying the whole state by up to 5 more control ticks costs
SQP 2.6 to 5.1 mm and OMPC and SMC nothing, and makes LMPC unstable from 2 ticks
(19.2 to 170 mm).

**What the estimation buys** (figure 14). Against the raw flight sensors (the
last 5 Hz fix, 0.15 s old; velocity differenced between fixes) the estimation
improves the base state the controller is handed in every component: position
2.4 against 3.9 mm, yaw 0.05 against 6.8 mrad, velocity 2.3 against 27 mm/s. For
SQP, that is better tracking and a gentler command at once. On the rectangle its
RMSE falls from 3.1 to 2.4 mm, its `∫‖τ‖²dt` from 0.172 to 0.108 and its
tick-to-tick torque change from 55 to 20 mN·m; on the circle `∫‖τ‖²dt` falls from
0.107 to 0.028 (4x). LMPC, OMPC and SMC track within 0.1-1 mm of raw either way
(on the circle the estimation saves them about 1 mm). The joint Kalman filter
does most of the smoothing: it alone takes SQP's torque change from 53 to 20
mN·m and LMPC's from 23 to 9.5, with no loss of tracking.

**Estimator start-up** (figure 16). Raw sensors have nothing to learn. With the
5 Hz, 4 mm fix the EKF is converged almost at once: SQP tracks to 2.7 mm with no
warm-up and 2.5 mm after 60 s.

**Disturbance-observer feedforward** (figures 12-13). With the constant 0.05 N·m
torque on exact state, `ff_gain` 1.0 recovers the undisturbed accuracy (SQP 9.6
to 0.32; the others 39-44 to 0.3-0.5). On flight sensing it removes nearly all
of it (EKF + joint KF: the others 38-44 to 3.3-3.4, SQP 9.5 to 2.5), against
9.4-11.4 for the others on raw sensors, where the feedforward is capped on twice
as many ticks. These runs are offline; the live, velocity-sensing caveat on
`ff_gain` (CLAUDE.md, main README) is unchanged by them.

**Constraints.** Only the NMPC enforces the joint box. When joint3 starts 0.088
rad outside it, SQP brings it inside without entering safe mode (RMSE 15.3 while
it does); LMPC also recovers but enters safe mode; OMPC and SMC leave it outside
the whole run. On a reach the box forbids (10-13 cm), SQP never leaves the box and
stops short (final error 30-40 mm), while OMPC and SMC reach further by leaving the
box by 30-53 mrad and pressing joint5 into its hard stop.

**Timing** (5 pinned processes, mains, `performance` governor). Mean / p99 / max
per tick, in µs: tuned SQP (N 24) 717 / 835 / 1106, LMPC (N 45) 990 / 1232 /
1456; OMPC and SMC about 5 / 10 / 37. Every law fits the 10 ms period with at
least 6x headroom.

The shipped-variant studies and `fig15_shipped_vs_tuned` were last run on the
previous relative-nav profile (15 mm, 2 Hz, 0.5 s) and have not been rerun.

## Metrics

Defined once, in `src/pat_arm_control/src/bench/metrics.{h,cpp}` and
`run_summary.cpp`, and unit-tested against signals with closed-form answers
(`test_bench_metrics.cpp`, `test_run_summary.cpp`). `e` is the planar EE position
error against plant truth, `τ` the applied torque of the scored arm's joints, `ψ`
the base yaw relative to where it started.

| column | meaning |
|---|---|
| `rmse_mm`, `mse_mm2`, `err_var_mm2`, `err_peak_mm` | RMSE, MSE, population variance and peak of `e` in the scoring window |
| `along_mean_mm`, `cross_rms_mm` | signed along-track (negative = trailing) and cross-track error against the reference's direction of travel |
| `step_rise_s`, `step_overshoot_pct`, `step_settle_s`, `step_sse_mm` | 10-90% rise, overshoot, 2% settling time and steady-state error of a step, from the EE's progress along it |
| `tau2_int_win`, `tau2_int_total` | `∫‖τ‖²dt` in the window / over the run |
| `tau_var`, `tau_peak`, `rail_frac` | pooled torque variance, peak, fraction of joint-ticks within 1% of the limit |
| `chatter_tv`, `tau_step_rms` | total variation `Σ|Δτ|` and rms tick-to-tick change: smooth versus switching commands |
| `work_int` | mechanical work `∫|τ·q̇|dt` |
| `qlim_peak_rad`, `qlim_int_rad_s`, `qlim_outside_frac`, `qlim_final_rad` | how far the worst joint is outside its configured box: peak, `∫dt`, fraction of ticks outside, at the end |
| `stop_frac` | fraction of ticks with a plant constraint active (a hard joint stop or a contact) |
| `yaw2_int_total`, `yaw_peak_mrad`, `wz2_int_total` | base attitude disturbance: `∫ψ²dt`, peak `|ψ|`, `∫ω²dt` |
| `est_*_rms` | error of the base state the controller was given (noisy or EKF), against truth |
| `tick_us_*` | wall time of `computeControl`: mean and percentiles |

The scoring window starts after a trajectory's approach and ramp (the same as the
bench's own report), at the step for a step, and after the first quarter otherwise.
