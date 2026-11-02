# Jetson timing: quickstart

Measures how long each arm controller's control tick takes on the Jetson Orin NX,
against the DGX, on exactly the same scenarios, and says whether every NMPC's
worst tick fits the 10 ms control period there.

Anyone with the DGX and the Jetson can follow this. Part A is done once on the DGX
beforehand, Part B on the Jetson, Part C wherever the plotting environment is.

## What it measures, and why the comparison is fair

- **Same binary on both machines.** One Docker image and one workspace build,
  made on the DGX with the Orin's instruction set (`-mcpu=cortex-a78`) and
  BLASFEO's kernels for it, are run unchanged on both. A difference between the
  machines is the hardware, nothing else.
- **Same inputs, tick for tick.** The scenarios are `contact_bench` runs: fixed
  plant, start pose, reference and seeded sensor noise, closed loop. With the
  same binary each tick sees identical inputs on both machines; `compare.py`
  checks that the per-tick records match, so the timing comparison is exact.
- **Three studies**, each run one process at a time on one pinned core:

  | Study | Cells | What |
  |---|---|---|
  | `timing` | 50 | 1x circle, exact state: all four controllers, and the NMPCs' tick against horizon N |
  | `jetson_timing` | 132 | 11 scenarios x 4 controllers x 3 seeds: nominal circle and rectangle, past the speed limit (circle 3x and 4x, rectangle 3x), mass error, halved torque limit, raw / EKF / EKF + joint KF sensing |
  | `worst_case` | 12 | the NMPCs with every QP forced to fail, so every tick runs the QP iteration cap twice (solve, then cold retry): the worst tick the cap allows, measured on every tick |

- **The verdict** is the worst tick against the period. The NMPCs' worst case is
  `2 x cap x (cost of one QP iteration)`; the caps (SQP 20, LMPC 14) were sized
  on the DGX, and the Orin's slower cores may need smaller ones.
  `quick_check.py` says which, on the spot.

The bench times the controller's computation (`computeControl`). It does not
simulate the effect of a late command, and it leaves out the ROS node's own work
per tick (small: the DGX live runs put it at a few hundred microseconds), which is
why the caps are sized to an 8 ms budget, not 10.

## Time needed

| Part | Step | Time | Machine must be quiet? |
|---|---|---|---|
| A (DGX) | clone, build image | 35-50 min | no |
| | build workspace | ~10 min | no |
| | rehearsal: the suite on the DGX | 15-20 min | **yes** |
| | package, copy to the Jetson | 15-25 min | no |
| B (Jetson) | check Docker and disk, load the image | 15-30 min | |
| | prepare: power mode, clocks, governor | 10 min (+5 if it reboots) | |
| | the suite | 45-70 min (the smoke test prints the real figure) | **yes** |
| | verdict, copy results back | 5-10 min | |
| | *if the caps do not fit:* rerun at caps that do | *+45-70 min* | **yes** |
| C | compare and plot | 2 min | |

Budget **about 2 hours on the Jetson, 3 if the caps need changing.**

The suite's length is about 185 times the smoke cell's: the timing study is ~30
smoke cells of work, `jetson_timing` ~100, `worst_case` ~55 (each of its ticks
runs the QP cap twice, in both arms). The Orin NX's cores run at about half the
DGX's clock and do less per clock, so expect each cell to take 2-3x as long.

---

## Part A: on the DGX (the day before)

All commands on the DGX host, in a **separate clone**, so the thesis checkout
and its build are untouched.

**A1. Clone the branch** (2 min)

```bash
git clone -b jetson_bench git@github.com:BJRitchie/MuJoCo-PAT.git ~/MuJoCo-PAT-jetson
cd ~/MuJoCo-PAT-jetson
```

**A2. Build the image** (30-45 min, can share the machine)

```bash
tools/jetson_bench/build_image.sh
```

Ends with `built mujoco-pat:jetson sha256:...`.

**A3. Build the workspace in it** (~10 min, can share the machine)

```bash
JOBS=16 WORKERS=4 tools/jetson_bench/enter.sh tools/jetson_bench/build_ws.sh
```

Ends with `build_ws.sh: done`. `JOBS`/`WORKERS` only set build parallelism;
leave them out on a small machine.

**A4. Rehearse: run the suite on the DGX** (15-20 min, **quiet machine**). This
is also the DGX half of the comparison, with exactly the binary the Jetson will
run. Check nobody else is running heavy work (`top`), then in two terminals:

```bash
# terminal 1, the host: power, clocks and temperatures on record (Ctrl-C when the suite ends)
tools/jetson_bench/host_monitor.sh dgx
```
```bash
# terminal 2: the suite, timing pinned to CPU 6 (a fast X925 core)
tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh dgx 6
```

What good looks like: the unit tests end `[  PASSED  ]`; the smoke step prints
`the rest should take about N min`; each study ends `ran N, summary.csv has N
rows` with no `FAILED` lines. Then:

```bash
tools/jetson_bench/enter.sh python3 tools/jetson_bench/quick_check.py dgx
```

Every NMPC's forced worst case should say `OK` on the DGX (about 6 and 8 ms).

**A5. Keep the DGX results**

```bash
git add data/thesis_eval/machines/dgx
git commit -m "Jetson comparison: the DGX half"
git push
```

**A6. Package for the Jetson** (~10 min)

```bash
tools/jetson_bench/package.sh          # writes ../jetson_bench_image.tar.gz and ../jetson_bench_ws.tar.gz
scp ../jetson_bench_image.tar.gz ../jetson_bench_ws.tar.gz <user>@<jetson>:~/
```

Keep the DGX clone: the Jetson's results come back to it in Part C.

---

## Part B: on the Jetson Orin NX

Over SSH, run everything inside `tmux` (`sudo apt install -y tmux`, then
`tmux new -s bench`; detach with Ctrl-b, release, `d`; back with
`tmux attach -t bench`), so a dropped connection does not stop the run.

**B1. Check the machine** (5-20 min)

```bash
cat /etc/nv_tegra_release        # JetPack 6.2 is "R36 (release), REVISION: 4.x"
df -h ~                          # about 20 GB free needed
nproc                            # 8 on the 16 GB Orin NX, 6 on the 8 GB
docker --version
```

If `docker` is missing:

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker
docker run --rm hello-world      # should print "Hello from Docker!"
```

**B2. Unpack and load** (5-15 min)

```bash
mkdir -p ~/jetson_bench && tar -xzf ~/jetson_bench_ws.tar.gz -C ~/jetson_bench
gunzip -c ~/jetson_bench_image.tar.gz | docker load        # ends "Loaded image: mujoco-pat:jetson"
cd ~/jetson_bench
```

**B3. Prepare the machine** (10 min). Timing needs the CPUs at a fixed, maximum
clock.

```bash
sudo nvpmodel -q                                   # the current power mode
grep "POWER_MODEL ID" /etc/nvpmodel.conf           # the modes this board has
sudo nvpmodel -m <ID>                              # the one named MAXN_SUPER if listed, else MAXN
```

If `nvpmodel` asks to reboot, reboot and come back to this step. Then:

```bash
sudo jetson_clocks                                  # lock every clock at its maximum (and the fan, on the dev kit)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo jetson_clocks --show | head -20                # every CPU: MinFreq == MaxFreq == CurrentFreq
```

Close anything else running (a browser, a desktop session's heavy apps). For the
cleanest numbers you can drop to the console with
`sudo systemctl isolate multi-user.target` (it stops the desktop; `sudo systemctl
isolate graphical.target` brings it back).

**B4. Run the suite** (45-70 min, **leave the Jetson alone**)

```bash
# terminal 1 (host): record power mode, clocks and tegrastats; Ctrl-C when the suite ends
tools/jetson_bench/host_monitor.sh orin_nx
```
```bash
# terminal 2: the suite; timing pinned to the last CPU
tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh orin_nx
```

After about a minute the smoke step prints `the rest should take about N min`:
that is the figure to plan around.

**B5. The verdict** (1 min)

```bash
tools/jetson_bench/enter.sh python3 tools/jetson_bench/quick_check.py orin_nx
```

It prints every controller's worst tick against 10 ms, and for each NMPC the
forced worst case, the cost of one QP iteration and the largest cap that fits an
8 ms budget. If every line says `OK`, go to B7.

**B6. Only if a cap does not fit: rerun at caps that do** (+45-70 min)

`quick_check.py` prints the two commands, for example:

```bash
tools/jetson_bench/set_caps.sh 9 5
tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh orin_nx_cap9_5
tools/jetson_bench/enter.sh python3 tools/jetson_bench/quick_check.py orin_nx_cap9_5
```

The rerun has its own results name, so the first run is kept. Smaller caps
mean more solves stop at the cap and fall back to replaying the last plan: the
rerun's RMSE (Part C) shows what that costs in tracking.

**B7. Copy the results back** (5 min)

```bash
tar -czf ~/orin_nx_results.tar.gz data/thesis_eval/machines/orin_nx*
scp ~/orin_nx_results.tar.gz <user>@<dgx>:~/MuJoCo-PAT-jetson/
```

Then put the machine back: `sudo nvpmodel -m <the original ID>`, and reboot to
undo `jetson_clocks` and the governor.

---

## Part C: compare (on the DGX)

```bash
cd ~/MuJoCo-PAT-jetson
tar -xzf orin_nx_results.tar.gz
~/.venvs/thesis_eval/bin/python tools/jetson_bench/compare.py --a dgx --b orin_nx
# and for a rerun at other caps:
~/.venvs/thesis_eval/bin/python tools/jetson_bench/compare.py --a dgx --b orin_nx_cap9_5
```

Writes `data/thesis_eval/machines/compare_dgx_orin_nx.txt`,
`docs/assets/thesis/fig22_jetson_timing.{pdf,png}` and
`docs/assets/thesis/tables/tab_jetson.tex`. Reading the report:

1. **Machines**: `FAIR`, or the caveats (a different image or build, a core
   below its maximum clock, timing allowed on a noisy machine). Caveats mean
   the numbers need explaining before they go in the thesis.
2. **Same inputs**: should be every cell identical tick for tick. If not, it
   shows how far apart the runs went, and the RMSE difference.
3. **Control tick**: per controller and machine, mean / p99 / p99.9 / worst,
   ticks over 10 ms, and the Orin/DGX ratio.
4. **Forced worst case**: the worst tick per NMPC per machine, `(full path)`
   confirming every tick ran the cap twice, the cost of one iteration, and the
   largest cap inside the budget.
5. **Horizon sweep**: mean and worst tick against N on both machines.

Keep the results:

```bash
git add data/thesis_eval/machines docs/assets/thesis/fig22_jetson_timing.* docs/assets/thesis/tables/tab_jetson.tex
git commit -m "Jetson comparison: Orin NX results"
git push
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `REFUSED, timing needs a quiet machine: CPU governor is 'schedutil'` (or `run_suite.sh` stops at the governor check) | the governor is not `performance` | B3's `tee` line, on the host |
| `permission denied ... docker.sock` | not in the `docker` group | `sudo usermod -aG docker $USER && newgrp docker` |
| `exec format error` | the image was built on an x86 machine | rebuild it on the DGX (A2) |
| `contact_bench not found on AMENT_PREFIX_PATH` | a command run outside `enter.sh`, or `install/` missing | run through `tools/jetson_bench/enter.sh`; on the DGX, A3 |
| `No such file ... test_pat_arm_control` | the package was made before A3 finished | A3, then A6 again |
| the report says a core was below its maximum clock | throttling, or clocks not locked | B3 again; check the fan; look at `tegrastats.log` for when it happened |
| `FAILED <cell>` lines | that cell's bench run failed (its stderr is printed) | rerun the same command: finished cells are cached, only the failures run |
| `(NOT the full path)` in the worst case | the QP stopped before the cap (HPIPM's minimum-step exit) | the forced worst case underestimates there; use the worst tick of `jetson_timing`'s hardest cells (circle 4x) instead |
| `nvpmodel` asks to reboot | changing power mode needs it on some boards | reboot, then B3 again |
| disk full during `docker load` | the image needs ~10-15 GB unpacked | free space, or load from a USB drive |

Everything a run measured is in `data/thesis_eval/machines/<name>/<study>/`:
`summary.csv` (one row per cell), `manifest.json` (machine, clocks, temperature,
build, image, git revision, every command), and `series/` (the per-tick record).
