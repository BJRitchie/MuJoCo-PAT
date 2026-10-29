#!/usr/bin/env python3
"""Run one thesis scenario on the live ROS stack and keep the result, or nothing.

The unit of work is a trial: the deployed nodes (simulator, `nav_node`, both arm
controllers) run against each other in real time, one scenario, one repeat. A trial
either finishes and lands under `trials/<hash>/` whole, or leaves nothing behind;
a crash, a power cut or a Ctrl-C costs the trial in flight and no other.

Run it INSIDE the dev container (`docker exec mujoco_pat_dev ...`), as the
`timeout` has to be container-side too.

    live/driver.py run --law ompc --shape circle --repeat 1 --out data/thesis_eval/live/pilot
    live/driver.py status --out data/thesis_eval/live/pilot

What a trial does:
  1. refuses unless the machine is fit (governor `performance`, mains) and no stale
     ROS process or other driver is running;
  2. writes this trial's parameter files (the tuned settings, the realtime
     configuration, the plant, the targets) into its `.partial` directory;
  3. launches the stack headless on its own ROS domain, pinned to its own cores;
  4. waits `nav_warmup` seconds with the arms at rest (the estimator converges),
     then starts the recorder and the target publisher and runs for `duration`;
  5. stops targets, then the stack (its exit flushes the node's trace), then the
     recorder; joins the trace; scores the series with `series_summary`;
  6. renames `<hash>.partial` to `<hash>` only when all of it worked.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import types

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import join_trace  # noqa: E402

# ------------------------------------------------------------------ machine --
# CPU map for the dev laptop (i7-13700H): P-cores 0-11 as SMT pairs, E-cores 12-19.
# Every realtime thread gets a pair of its own; everything else stays off them.
# Another machine needs its own map (the DGX, the Orin).
CORES = {
    "nav": [2, 3],
    "left": [4, 5], "right": [6, 7],
    "left_observer": [8, 9], "right_observer": [10, 11],
    "rest": "0,1,12-19",        # simulator, launch, recorder, driver
}
RT_PRIORITY = 80
STRAY = r"simulation_node|nav_node|nmpc_node|ompc_node|smc_node|ee_target_publisher|recorder\.py"
SETTLE_S = 4.0                  # stack start-up before the warm-up clock starts
STOP_S = 20.0                   # how long the stack gets to exit on SIGINT

# Limits that stop a trial before the machine does it for us (a hung laptop leaves
# no log). Crossing one mid-trial kills the whole stack and discards the trial; the
# queue then waits, and will not start another until the START figures are met.
# The 16 GB dev laptop runs the desktop, an editor and a browser alongside.
MEM_ABORT_KB = 1_500_000        # MemAvailable below this: abort
MEM_START_KB = 4_000_000        # ... and no trial starts below this
SWAP_ABORT_KB = 3_000_000       # swap in use above this: abort (thrashing)
RSS_ABORT_KB = 6_000_000        # all the trial's nodes together above this: abort
TEMP_ABORT_C = 92.0             # CPU package temperature: abort
TEMP_START_C = 80.0             # ... and no trial starts above this

# The ROS-side nodes the trial exercises; their binaries are part of its identity.
NODE_BINARIES = [
    ("pat_simulation", "simulation_node"), ("pat_gnc", "nav_node"),
    ("pat_arm_control", "nmpc_node"), ("pat_arm_control", "ompc_node"),
    ("pat_arm_control", "smc_node"), ("pat_arm_control", "ee_target_publisher_node"),
    ("pat_arm_control", "series_summary"),
]


class Fail(Exception):
    """A trial that cannot be kept; the message says why."""


class Unfit(Fail):
    """The MACHINE was not fit (unplugged, throttled, a stray node), not the trial.
    A queue waits and retries these without counting them as failures."""


def parse_cpus(spec):
    """'0,1,12-19' as a set of CPU numbers."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def sh(*cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout.strip()


def prefix(pkg):
    return sh("ros2", "pkg", "prefix", pkg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# -------------------------------------------------------------------- spec --

def make_spec(a):
    """The scenario, as the plain dict that is also its identity."""
    tuned = json.load(open(os.path.join(ROOT, "tools", "thesis_eval", "tuned.json")))
    if a.tag not in tuned:
        raise SystemExit(f"{a.tag} is not in tuned.json (choose from {list(tuned)})")
    t = tuned[a.tag]
    return {
        "tag": a.tag, "law": t["law"], "tuned_args": t["args"],
        "shape": a.shape, "speed": a.speed, "repeat": a.repeat,
        "duration": a.duration, "nav_warmup": a.nav_warmup, "damping": 0.0,
        "partner": "hold", "state": "ekf",
    }


def trial_hash(spec):
    """Spec, every installed config the stack reads, the nodes' binaries, and this tool."""
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for pkg, pat in (("pat_arm_control", "config"), ("pat_simulation", "config"),
                     ("pat_gnc", "config")):
        d = os.path.join(prefix(pkg), "share", pkg, pat)
        for f in sorted(os.listdir(d)):
            h.update((pkg + "/" + f).encode())
            h.update(sha(os.path.join(d, f)).encode())
    for pkg, exe in NODE_BINARIES:
        h.update(sha(os.path.join(prefix(pkg), "lib", pkg, exe)).encode())
    for f in ("driver.py", "recorder.py", "join_trace.py"):
        h.update(sha(os.path.join(HERE, f)).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------ parameter files --

class Loader(yaml.SafeLoader):
    """PyYAML follows YAML 1.1, where `1.0e5` and `1e5` are strings or integers;
    ROS (and the C++ bench) read them as the floats they are written to be."""


Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
               r"|[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)"
               r"|\.[0-9_]+(?:[eE][-+][0-9]+)?)$"), list("-+0123456789."))


def load_yaml(path):
    with open(path) as fh:
        return yaml.load(fh, Loader)


def coerce(new, old):
    """`new` as the type ROS declared for `old`: a double stays a double (and a list
    of doubles a list of doubles, which ROS insists on), an integer an integer."""
    if isinstance(old, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        return float(new)
    if isinstance(old, list) and old and isinstance(old[0], float) and isinstance(new, list):
        return [float(x) for x in new]
    return new


def set_nested(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = coerce(value, d.get(keys[-1]))


def tuned_overrides(args):
    """`--set key=value` pairs from tuned.json as {dotted key: parsed value}."""
    out, it = {}, iter(args)
    for flag in it:
        if flag != "--set":
            raise Fail(f"tuned.json holds an argument this driver does not translate: {flag}")
        key, _, val = next(it).partition("=")
        out[key] = yaml.load(val, Loader)
    return out


def died(stack_log):
    """The first node the launch reports dead, or None."""
    for line in open(stack_log, errors="replace"):
        if "process has died" in line:
            return line.strip().split("]: ", 1)[-1][:160]
        if "what():" in line:
            return line.strip().split("what():", 1)[-1].strip()[:200]
    return None


def write_params(spec, work):
    """The parameter files this trial's launch is given. Returns their paths."""
    law = spec["law"]
    arm = os.path.join(prefix("pat_arm_control"), "share", "pat_arm_control", "config")

    # The law's file with the tuned values and the realtime configuration applied.
    cfg = load_yaml(os.path.join(arm, f"{law}.yaml"))
    base = cfg["/**"]["ros__parameters"]
    for k, v in tuned_overrides(spec["tuned_args"]).items():
        set_nested(base, k, v)
    for side in ("left", "right"):
        node = cfg[f"pat_arm_{law}_{side}"]["ros__parameters"]
        node["rt_cpus"] = CORES[side]
        node["rt_cpus_observer"] = CORES[side + "_observer"]
    base["rt_priority"] = RT_PRIORITY
    base["lock_memory"] = True
    law_yaml = os.path.join(work, f"{law}.yaml")
    yaml.safe_dump(cfg, open(law_yaml, "w"), sort_keys=False)

    # The shared file: only the observer thread's priority differs.
    common = load_yaml(os.path.join(arm, "arm_common.yaml"))
    common["/**"]["ros__parameters"]["rt_priority_observer"] = RT_PRIORITY
    common_yaml = os.path.join(work, "arm_common.yaml")
    yaml.safe_dump(common, open(common_yaml, "w"), sort_keys=False)

    # The navigator, on its own realtime thread.
    nav = load_yaml(os.path.join(prefix("pat_gnc"), "share", "pat_gnc", "config",
                                 "nav_flight.yaml"))
    p = nav["pat_nav"]["ros__parameters"]
    p["rt_priority_nav"] = RT_PRIORITY
    p["lock_memory"] = True
    p["rt_cpus_nav"] = CORES["nav"]
    nav_yaml = os.path.join(work, "nav_flight.yaml")
    yaml.safe_dump(nav, open(nav_yaml, "w"), sort_keys=False)

    # The plant: the table frictionless, as every thesis study runs it.
    # Each repeat gets its own sensor noise (the repeat number is the seed, as in the
    # bench, where one seed number gives every controller the same noise).
    plant = {"pat_simulation": {"ros__parameters": {
        "chaser_damping": [spec["damping"]] * 3, "sensor_seed": spec["repeat"]}}}
    plant_yaml = os.path.join(work, "plant.yaml")
    yaml.safe_dump(plant, open(plant_yaml, "w"))

    # The targets: the scored (left) arm follows the shape, the other holds.
    tg = load_yaml(os.path.join(arm, "ee_targets.yaml"))
    tp = tg["ee_target_publisher"]["ros__parameters"]
    tp["sides"] = ["left"]
    if spec["shape"] == "circle":
        tp["circle"]["period"] = tp["circle"]["period"] / spec["speed"]
    else:
        tp["rectangle"]["speed"] = tp["rectangle"]["speed"] * spec["speed"]
    tg_yaml = os.path.join(work, "ee_targets.yaml")
    yaml.safe_dump(tg, open(tg_yaml, "w"), sort_keys=False)

    sim = os.path.join(prefix("pat_simulation"), "share", "pat_simulation", "config")
    sensors = ",".join([os.path.join(sim, "sensors_flight.yaml"),
                        os.path.join(sim, "sensors_flight_encoders.yaml"), plant_yaml])
    return {"law": law_yaml, "common": common_yaml, "nav": nav_yaml, "targets": tg_yaml,
            "sensors": sensors, "tau_limit": common["/**"]["ros__parameters"]["model_tau_max"][0]}


# ---------------------------------------------------------------- processes --

class Group:
    """A child process in its own process group, stopped as a whole."""

    def __init__(self, cmd, log, env):
        self.log = open(log, "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)

    def alive(self):
        return self.p.poll() is None

    def stop(self, wait=STOP_S):
        if self.alive():
            try:
                os.killpg(self.p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.p.wait(wait)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
                self.p.wait()
        self.log.close()


def stray_processes():
    r = subprocess.run(["pgrep", "-af", STRAY], capture_output=True, text=True)
    me = str(os.getpid())
    return [l for l in r.stdout.splitlines()
            if "pgrep" not in l and not l.startswith(me + " ") and "driver.py" not in l]


def kill_stale():
    """Stop the nodes a killed driver left behind. Only called once this driver
    holds the directory's lock, so nothing it finds belongs to a live trial."""
    for sig, wait in ((signal.SIGINT, 6.0), (signal.SIGKILL, 1.0)):
        groups = set()
        for line in stray_processes():
            try:
                groups.add(os.getpgid(int(line.split()[0])))
            except (ValueError, ProcessLookupError):
                pass
        for g in groups:
            try:
                os.killpg(g, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t = time.monotonic() + wait
        while stray_processes() and time.monotonic() < t:
            time.sleep(0.3)
        if not stray_processes():
            return


def read_meminfo():
    """MemAvailable and swap in use, in kB."""
    v = {}
    for line in open("/proc/meminfo"):
        k, _, rest = line.partition(":")
        v[k] = int(rest.split()[0])
    return {"avail": v["MemAvailable"], "swap_used": v["SwapTotal"] - v["SwapFree"]}


def read_cpu_temp_c():
    """The CPU package temperature, or None where the machine does not report one."""
    for z in sorted(os.listdir("/sys/class/thermal")):
        try:
            if open(f"/sys/class/thermal/{z}/type").read().strip() == "x86_pkg_temp":
                return int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def read_trial_rss_kb():
    """Resident memory of every node the trial has started, in kB."""
    total = 0
    for line in stray_processes():
        try:
            for l in open(f"/proc/{line.split()[0]}/status"):
                if l.startswith("VmRSS:"):
                    total += int(l.split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return total


class Guard:
    """Watches memory, swap, temperature and the trial's own footprint, and raises
    Unfit the moment one crosses its limit. Remembers the extremes it saw, which go
    into the trial's report. The readers are arguments so the limits can be tested."""

    def __init__(self, mem=read_meminfo, temp=read_cpu_temp_c, rss=read_trial_rss_kb):
        self.mem, self.temp, self.rss = mem, temp, rss
        self.n, self.last_rss = 0, 0
        self.peak = {"mem_avail_min_kb": None, "swap_used_max_kb": 0,
                     "temp_max_c": None, "rss_max_kb": 0}

    def check(self, starting=False):
        m, t = self.mem(), self.temp()
        # Counting the nodes' memory forks a pgrep: every fourth look (2 s) is enough.
        if not starting and self.n % 4 == 0:
            self.last_rss = self.rss()
        self.n += 1
        r = 0 if starting else self.last_rss
        p = self.peak
        p["mem_avail_min_kb"] = m["avail"] if p["mem_avail_min_kb"] is None \
            else min(p["mem_avail_min_kb"], m["avail"])
        p["swap_used_max_kb"] = max(p["swap_used_max_kb"], m["swap_used"])
        p["rss_max_kb"] = max(p["rss_max_kb"], r)
        if t is not None:
            p["temp_max_c"] = t if p["temp_max_c"] is None else max(p["temp_max_c"], t)
        if m["avail"] < (MEM_START_KB if starting else MEM_ABORT_KB):
            raise Unfit(f"free memory is down to {m['avail'] // 1024} MiB")
        if m["swap_used"] > SWAP_ABORT_KB:
            raise Unfit(f"the machine is swapping ({m['swap_used'] // 1024} MiB in use)")
        if r > RSS_ABORT_KB:
            raise Unfit(f"the trial's nodes hold {r // 1024} MiB")
        if t is not None and t >= (TEMP_START_C if starting else TEMP_ABORT_C):
            raise Unfit(f"the CPU is at {t:.0f} C")


def fit_check():
    """Raise Unfit if the machine is not in the state a live trial needs."""
    from run_study import machine_state, timing_allowed
    why = timing_allowed(machine_state(), False)
    if why:
        raise Unfit(f"machine not fit for a live trial: {why}")


def preflight(allow_stray=False, clear_stale=False):
    fit_check()
    Guard().check(starting=True)
    if clear_stale and stray_processes():
        kill_stale()
    if not allow_stray and stray_processes():
        raise Unfit("a ROS node from another run is still alive (--kill-stale stops them):\n  " +
                    "\n  ".join(stray_processes()))


def rt_threads(law):
    """Every thread of the trial's nodes that runs a realtime policy: its priority and
    the CPUs it may use, read from /proc while the stack is up."""
    found = {}
    for label, pattern in (("nav", "__node:=pat_nav"), ("left", f"__node:=pat_arm_{law}_left"),
                           ("right", f"__node:=pat_arm_{law}_right")):
        pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        for pid in pids:
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    stat = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
                    policy, prio = int(stat[38]), int(stat[37])
                    if policy in (1, 2):                      # SCHED_FIFO, SCHED_RR
                        cpus = [l.split(":")[1].strip() for l in
                                open(f"/proc/{pid}/task/{tid}/status") if l.startswith("Cpus_allowed_list")]
                        found.setdefault(label, []).append({"policy": "FIFO" if policy == 1 else "RR",
                                                            "priority": prio, "cpus": cpus[0]})
            except (FileNotFoundError, ProcessLookupError, IndexError):
                pass
    return found


def cpu_list(cpus):
    return ",".join(str(c) for c in cpus) if len(cpus) != 2 else f"{cpus[0]}-{cpus[1]}"


def rt_problems(found):
    """What is missing from the realtime configuration the trial asked for."""
    out = []
    for label in ("nav", "left", "right"):
        want = cpu_list(CORES[label])
        ok = any(t["policy"] == "FIFO" and t["priority"] == RT_PRIORITY and t["cpus"] == want
                 for t in found.get(label, []))
        if not ok:
            out.append(f"{label}: no SCHED_FIFO {RT_PRIORITY} thread on CPUs {want} "
                       f"(found {found.get(label, 'none')})")
    return out


# -------------------------------------------------------------------- trial --

def health(stack_log, trace_lines, torque_msgs, n_rows, spec):
    """What the run itself says about whether it can be trusted."""
    misses = skipped = 0
    for m in re.finditer(r"deadline misses (\d+) \(longest run \d+\) \| skipped periods (\d+)",
                         open(stack_log, errors="replace").read()):
        misses += int(m.group(1))
        skipped += int(m.group(2))
    nav_overruns = sum(int(m.group(1)) for m in re.finditer(
        r"overruns (\d+)", open(stack_log, errors="replace").read()))
    expected = int(spec["duration"] * 100)
    return {"deadline_misses": misses, "skipped_periods": skipped, "nav_overruns": nav_overruns,
            "rows": n_rows, "rows_expected": expected,
            "rows_short": max(0, expected - n_rows),
            "trace_lines": trace_lines, "torque_msgs": torque_msgs}


def run_trial(spec, out, allow_stray=False, clear_stale=False):
    h = trial_hash(spec)
    final, partial = os.path.join(out, "trials", h), os.path.join(out, "trials", h + ".partial")
    if os.path.isdir(final):
        return final, "cached"
    os.makedirs(os.path.join(out, "trials"), exist_ok=True)
    lock = open(os.path.join(out, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise Fail("another driver is already running against this directory")
    # Stop the leftovers first: a recorder that outlived its driver writes its file
    # into the old trial's directory when it is signalled, recreating it.
    preflight(allow_stray, clear_stale)
    for stale in os.listdir(os.path.join(out, "trials")):
        if stale.endswith(".partial"):
            shutil.rmtree(os.path.join(out, "trials", stale), ignore_errors=True)
    os.makedirs(partial)

    # Every process of the trial shares one domain and stays on this host, so two
    # drivers (or a stray node) cannot hear each other.
    env = dict(os.environ, ROS_DOMAIN_ID=str(100 + int(h[:4], 16) % 100),
               ROS_LOCALHOST_ONLY="1")
    groups = []

    def kill_all(*_):
        for g in reversed(groups):
            g.stop(3)
        shutil.rmtree(partial, ignore_errors=True)
        sys.exit(130)
    old = {s: signal.signal(s, kill_all) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        files = write_params(spec, partial)
        sim = prefix("pat_simulation")
        model = os.path.join(prefix("pat_arm_control"), "share", "pat_arm_control", "models",
                             "pat_platform_planar.xml")
        trace = os.path.join(partial, "trace.csv")
        stack_log = os.path.join(partial, "stack.log")
        pin = ["taskset", "-c", CORES["rest"]]
        stack = Group(pin + ["ros2", "launch", os.path.join(ROOT, "launch", "full_stack_nmpc.launch.py"),
                             f"law:={spec['law']}", "with_nav:=true", "visualise:=false",
                             f"sensor_profile:={files['sensors']}", f"nav_params:={files['nav']}",
                             f"params_file:={files['law']}", f"common_params_file:={files['common']}",
                             f"trace_file:={trace}"],
                      os.path.join(partial, "stack.log"), env)
        groups.append(stack)
        guard = Guard()
        deadline = time.monotonic() + SETTLE_S + spec["nav_warmup"]
        while time.monotonic() < deadline:
            guard.check()
            if not stack.alive():
                raise Fail("the stack exited during start-up and warm-up")
            if (why := died(stack_log)):
                raise Fail(f"a node died during start-up: {why}")
            time.sleep(0.5)

        csv_path = os.path.join(partial, "series.csv")
        rec = Group(pin + [sys.executable, os.path.join(HERE, "recorder.py"), "--side", "left",
                           "--out", csv_path, "--model", model, "--duration", str(spec["duration"])],
                    os.path.join(partial, "recorder.log"), env)
        groups.append(rec)
        time.sleep(2.0)
        targets = Group(pin + ["ros2", "launch", "pat_arm_control", "ee_targets.launch.py",
                               f"target_pattern:={spec['shape']}",
                               f"ee_targets_params_file:={files['targets']}"],
                        os.path.join(partial, "targets.log"), env)
        groups.append(targets)
        t_task = time.monotonic()
        deadline = t_task + spec["duration"] + 3.0
        rt_found, next_fit = None, t_task + 5.0
        while time.monotonic() < deadline:
            guard.check()
            if not (stack.alive() and targets.alive() and rec.alive()):
                raise Fail("a node died during the task")
            if (why := died(stack_log)):
                raise Fail(f"a node died during the task: {why}")
            now = time.monotonic()
            if now >= next_fit:
                fit_check()                  # unplugged or throttled mid-trial: discard it
                next_fit = now + 5.0
                if rt_found is None:
                    rt_found = rt_threads(spec["law"])
                    if (bad := rt_problems(rt_found)):
                        raise Fail("realtime configuration not in force: " + "; ".join(bad))
            time.sleep(0.5)

        targets.stop()
        stack.stop()                # its exit flushes the node's trace
        time.sleep(1.0)
        rec.stop()
        groups.clear()

        if not os.path.isfile(csv_path):
            raise Fail("the recorder wrote nothing")
        trace_lines, torque_msgs = join_trace.join(csv_path, os.path.join(partial, "trace_left.csv"))
        n_rows = sum(1 for _ in open(csv_path)) - 1
        info = health(os.path.join(partial, "stack.log"), trace_lines, torque_msgs, n_rows, spec)
        info["rt_threads"] = rt_found
        info["resources"] = guard.peak
        if info["rows_short"] > 0.05 * info["rows_expected"]:
            raise Fail(f"only {n_rows} of {info['rows_expected']} ticks were recorded")

        summarise(spec, files["tau_limit"], partial, csv_path)
        from run_study import machine_state
        report = {"spec": spec, "hash": h, "health": info, "machine": machine_state(),
                  "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(os.path.join(partial, "report.json"), "w") as fh:
            json.dump(report, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(partial, final)
        return final, "ran"
    except BaseException as e:
        for g in reversed(groups):
            g.stop(3)
        if isinstance(e, Fail) and not isinstance(e, Unfit):
            # A trial that failed on its own keeps its logs, away from the results,
            # so the cause can be read. A signal or a crash of the driver keeps nothing.
            kept = os.path.join(out, "failed", f"{h}-{int(time.time())}")
            os.makedirs(os.path.dirname(kept), exist_ok=True)
            shutil.move(partial, kept)
        else:
            shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        for s, f in old.items():
            signal.signal(s, f)
        fcntl.flock(lock, fcntl.LOCK_UN)


def summarise(spec, tau_limit, work, csv_path):
    exe = os.path.join(prefix("pat_arm_control"), "lib", "pat_arm_control", "series_summary")
    cmd = [exe, "--series", csv_path, "--summary-csv", os.path.join(work, "summary.csv"),
           "--scenario", spec["shape"], "--law", spec["law"], "--tag", spec["tag"],
           "--seed", str(spec["repeat"]), "--window-start", "5", "--tau-limit", str(tau_limit)]
    if spec["law"] != "nmpc":
        cmd.append("--no-solver")
    for k, v in (("state_source", "live-" + spec["state"]), ("partner", spec["partner"]),
                 ("traj_speed", spec["speed"]), ("base_damping", spec["damping"]),
                 ("nav_warmup", spec["nav_warmup"])):
        cmd += ["--label", f"{k}={v}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise Fail("series_summary failed: " + r.stderr.strip())


# -------------------------------------------------------------------- queue --

def cell_key(spec):
    return json.dumps({k: v for k, v in spec.items() if k != "repeat"}, sort_keys=True)


def completed_rmse(out, valid_hashes):
    """EE RMSE of every finished trial built from the code and configs of today,
    by cell. A trial from an older binary is not a repeat of this one."""
    res = {}
    d = os.path.join(out, "trials")
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if n.endswith(".partial") or n not in valid_hashes:
            continue
        try:
            rep = json.load(open(os.path.join(d, n, "report.json")))
            row = next(csv.DictReader(open(os.path.join(d, n, "summary.csv"))))
            res.setdefault(cell_key(rep["spec"]), []).append(float(row["rmse_mm"]))
        except (OSError, ValueError, StopIteration, KeyError):
            continue
    return res


def converged(values, rel, min_n):
    """True once the 95% interval on the mean is under `rel` of the mean."""
    n = len(values)
    if n < min_n:
        return False
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    return mean > 0 and 1.96 * sd / n ** 0.5 <= rel * mean


def run_queue(a):
    """Every cell, repeat by repeat: repeat 1 of all of them, then repeat 2, so that
    stopping at any point leaves a balanced set. A cell stops early once its mean is
    pinned down; a failure is retried twice and then logged, never allowed to block
    the rest; an unfit machine pauses the queue instead of costing a trial."""
    cells = [make_spec(types.SimpleNamespace(tag=t, shape=s, speed=a.speed, repeat=1,
                                             duration=a.duration, nav_warmup=a.nav_warmup))
             for t in a.tags for s in a.shapes]
    stop_file = os.path.join(a.out, "STOP")
    os.makedirs(a.out, exist_ok=True)
    ran = failed = 0
    for r in range(1, a.max_repeats + 1):
        for cell in cells:
            if os.path.exists(stop_file):
                print("STOP file found: stopping cleanly")
                return 0
            spec = dict(cell, repeat=r)
            valid = {trial_hash(dict(cell, repeat=k)) for k in range(1, a.max_repeats + 1)}
            have = completed_rmse(a.out, valid).get(cell_key(cell), [])
            if not a.fixed and r > a.min_repeats and converged(have, a.rel_ci, a.min_repeats):
                continue
            if trial_hash(spec) in valid and os.path.isdir(
                    os.path.join(a.out, "trials", trial_hash(spec))):
                continue
            attempts = 0
            while attempts <= 2:
                if os.path.exists(stop_file):
                    return 0
                try:
                    path, how = run_trial(spec, a.out, False, True)
                    ran += 1
                    print(f"[{time.strftime('%H:%M:%S')}] {spec['tag']} {spec['shape']} "
                          f"r{r}: {how}", flush=True)
                    break
                except Unfit as e:
                    print(f"[{time.strftime('%H:%M:%S')}] paused: {e}", flush=True)
                    time.sleep(30)
                except Fail as e:
                    attempts += 1
                    print(f"[{time.strftime('%H:%M:%S')}] {spec['tag']} {spec['shape']} r{r} "
                          f"failed ({attempts}/3): {e}", flush=True)
                    if attempts > 2:
                        failed += 1
                        with open(os.path.join(a.out, "failures.jsonl"), "a") as fh:
                            fh.write(json.dumps({"spec": spec, "reason": str(e),
                                                 "time": time.time()}) + "\n")
    print(f"queue finished: {ran} trials run, {failed} given up on")
    return 0


# ---------------------------------------------------------------------- cli --

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", default="sqp", help="a controller in tuned.json (sqp, iter, ompc, smc)")
    r.add_argument("--shape", choices=("circle", "rectangle"), default="circle")
    r.add_argument("--speed", type=float, default=1.0)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--duration", type=float, default=20.0)
    # The bench's warm-up study finds the EKF indistinguishable from 5 s to 60 s
    # (position error 2.6 mm against 2.4 mm, paired difference inside its noise), so
    # 15 s covers convergence with margin and the stack's own start-up. The pilot
    # compares 15 s against 60 s live before the long runs rely on it.
    r.add_argument("--nav-warmup", type=float, default=15.0)
    r.add_argument("--out", required=True)
    r.add_argument("--allow-stray", action="store_true")
    r.add_argument("--kill-stale", action="store_true",
                   help="stop ROS nodes a killed driver left behind (what a queue passes)")
    q = sub.add_parser("queue", help="every cell, repeat by repeat, adaptively")
    q.add_argument("--tags", nargs="+", default=["sqp", "iter", "ompc", "smc"])
    q.add_argument("--shapes", nargs="+", choices=("circle", "rectangle"),
                   default=["circle", "rectangle"])
    q.add_argument("--speed", type=float, default=1.0)
    q.add_argument("--duration", type=float, default=20.0)
    q.add_argument("--nav-warmup", type=float, default=15.0)
    q.add_argument("--max-repeats", type=int, default=10)
    q.add_argument("--min-repeats", type=int, default=3)
    q.add_argument("--rel-ci", type=float, default=0.10,
                   help="a cell stops once its 95%% interval is under this fraction of its mean")
    q.add_argument("--fixed", action="store_true", help="always run --max-repeats")
    q.add_argument("--out", required=True)
    s = sub.add_parser("status")
    s.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "queue":
        try:
            os.sched_setaffinity(0, parse_cpus(CORES["rest"]))
        except (OSError, ValueError):
            pass
        return run_queue(a)
    if a.cmd == "status":
        d = os.path.join(a.out, "trials")
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        done = [n for n in names if not n.endswith(".partial")]
        print(f"{len(done)} complete, {len(names) - len(done)} partial in {d}")
        return 0
    # The driver's own polling stays off the cores the realtime threads own.
    try:
        os.sched_setaffinity(0, parse_cpus(CORES["rest"]))
    except (OSError, ValueError):
        pass
    spec = make_spec(a)
    try:
        path, how = run_trial(spec, a.out, a.allow_stray, a.kill_stale)
    except Fail as e:
        print(f"driver: FAILED: {e}", file=sys.stderr)
        return 1
    print(f"{how}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
