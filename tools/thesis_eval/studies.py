"""The thesis comparison as data: which runs, with which arguments.

Every cell is one `contact_bench` process (see run_study.py). A study is a list of
cells; nothing here runs anything. Keeping the matrix as plain data is what lets
the runner cache, resume, filter and manifest it, and lets a reader check what a
figure is made of without reading the runner.

Controllers (the primary axis):

    sqp    SQP NMPC, the shipped nmpc.yaml (nmpc.full_nonlinear: true)
    iter   iterative-linearisation NMPC: the SAME file with only that key false,
           so a gap between sqp and iter is the formulation and not the tuning
    ompc   one-step MPC, shipped defaults
    smc    sliding mode, shipped defaults

State sources (what the controllers are told), see SOURCES:

    exact     the plant's own state
    raw       the flight sensors as they arrive, no estimator, raw encoders: what
              the estimation is compared with
    ekf       pat_gnc's ImuEkf, raw encoders
    ekf-jkf   ImuEkf and a per-joint Kalman filter on the encoders: the full
              estimation the thesis proposes

Every flight source runs NAV_WARMUP seconds before the task, as the navigation
filter would be running long before an arm task in flight.

The observer study crosses the state source with the disturbance observer's
feedforward gain (0 = observer running but not acting, as shipped; 0.5 and 1.0
act on it). Every study runs on the frictionless plant (PLANT) except the
damping sweep.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Cell:
    study: str
    name: str                    # unique within the study, readable, filesystem-safe
    law: str                     # contact_bench --law
    tag: str                     # configuration label (controller label, usually)
    args: Tuple[str, ...]        # everything else on the command line
    seed: int = 0
    series: bool = False         # also keep the per-tick record (big, not tracked)
    timing: bool = False         # a timing cell: pinned, serial, mains + performance only

    def key(self) -> str:
        return f"{self.law} {' '.join(self.args)} seed={self.seed} tag={self.tag}"


# (label, law, extra args)
CONTROLLERS: Sequence[Tuple[str, str, Tuple[str, ...]]] = (
    ("sqp", "nmpc", ()),
    ("iter", "nmpc", ("--set", "nmpc.full_nonlinear=false")),
    ("ompc", "ompc", ()),
    ("smc", "smc", ()),
)

TUNED_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")
VARIANTS = ("shipped", "tuned")
#: The thesis reports the tuned controllers; shipped is the comparison.
DEFAULT_VARIANT = "tuned"
_variant = DEFAULT_VARIANT


def set_variant(variant: str) -> None:
    """Which settings the controllers run with: "shipped" (the YAML as committed)
    or "tuned" (tune.py's result, from tuned.json, as --set overrides)."""
    global _variant
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    _variant = variant


def active_controllers() -> Sequence[Tuple[str, str, Tuple[str, ...]]]:
    if _variant == "shipped":
        return CONTROLLERS
    if not os.path.isfile(TUNED_JSON):
        raise FileNotFoundError(f"{TUNED_JSON} not found: run tune.py first")
    with open(TUNED_JSON) as fh:
        tuned = json.load(fh)
    missing = [c for c, _, _ in CONTROLLERS if c not in tuned]
    if missing:
        raise KeyError(f"tuned.json has no settings for {missing}: run tune.py for them")
    return tuple((c, tuned[c]["law"], tuple(tuned[c]["args"])) for c, _, _ in CONTROLLERS)

#: The plant every study runs: the air-bearing table frictionless, as the bus would
#: be in orbit. Only the `damping` robustness study varies it.
PLANT = ("--base-damping", "0")

#: Seconds the flight instruments and estimators run before an arm task, as the
#: navigation filter would be running long before an arm task in flight. Long
#: enough for the EKF to converge (~10 s; the `warmup` study measures it).
NAV_WARMUP = "60"
_FLIGHT = ("--nav-warmup", NAV_WARMUP)

#: label -> contact_bench arguments.
SOURCES: Dict[str, Tuple[str, ...]] = {
    "exact": (),
    "raw": ("--state-source", "raw") + _FLIGHT,
    "ekf": ("--state-source", "ekf") + _FLIGHT,
    "ekf-jkf": ("--state-source", "ekf", "--joint-filter", "kalman") + _FLIGHT,
}
#: The estimation the thesis proposes, used wherever a study needs "flight sensing".
FLIGHT_SENSING = "ekf-jkf"

# What the plant's reference is, per scenario. Trajectory runs size themselves
# (approach + two laps + 1 s); a step needs a length.
STEP_SMALL = ("--step", "0.06,0.06", "--seconds", "10")
STEP_LARGE = ("--step", "0.12,0.0", "--seconds", "10")
CIRCLE = ("--trajectory", "circle")
RECTANGLE = ("--trajectory", "rectangle")

SEEDS = 20          # noise studies
SEEDS_OBS = 10      # the observer grid (each cell is a noisy/ekf run)
SEEDS_DELAY = 5
REPEATS = 5         # timing: independent processes per configuration


def _slug(*parts) -> str:
    s = "_".join(str(p) for p in parts if str(p) != "")
    for ch in " /=,[]{}:":
        s = s.replace(ch, "-")
    return s


def _cells(study: str, scenario: str, scenario_args: Sequence[str],
           extra: Sequence[str] = (), extra_name: str = "", seeds: Iterable[int] = (0,),
           controllers: Optional[Sequence[Tuple[str, str, Tuple[str, ...]]]] = None,
           series: bool = False, timing: bool = False) -> List[Cell]:
    out = []
    seeds = list(seeds)
    plant = () if "--base-damping" in extra else PLANT
    for label, law, ctrl_args in (controllers or active_controllers()):
        for seed in seeds:
            name = _slug(scenario, extra_name, label, f"s{seed}" if seed else "")
            out.append(Cell(study, name, law, label,
                            tuple(scenario_args) + tuple(ctrl_args) + plant + tuple(extra),
                            seed=seed, series=series, timing=timing))
    return out


def _n(quick: bool, full: int, small: int = 2) -> int:
    return small if quick else full


# --------------------------------------------------------------------------- #

def nominal(quick: bool) -> List[Cell]:
    """The four controllers on the headline scenarios, exact state, held partner."""
    cells: List[Cell] = []
    for scen, args in (("step6", STEP_SMALL), ("step12", STEP_LARGE),
                       ("circle", CIRCLE), ("rectangle", RECTANGLE)):
        cells += _cells("nominal", scen, args, series=True)
    return cells


def balanced(quick: bool) -> List[Cell]:
    """Mirror, same-direction and held partner: what the base feels."""
    cells: List[Cell] = []
    for shape, args in (("circle", CIRCLE), ("rectangle", RECTANGLE)):
        for partner in ("hold", "mirror", "same"):
            cells += _cells("balanced", shape, args, ("--partner", partner),
                            f"p-{partner}", series=(shape == "circle"))
    return cells


def speed(quick: bool) -> List[Cell]:
    cells: List[Cell] = []
    for f in (1, 2, 3, 4):
        cells += _cells("speed", "circle", CIRCLE, ("--traj-speed", str(f)), f"x{f}")
    for f in (1, 2, 3):
        cells += _cells("speed", "rectangle", RECTANGLE, ("--traj-speed", str(f)), f"x{f}")
    return cells


#: The shapes every robustness study runs: the smooth circle, and the rectangle,
#: whose corners need the look-ahead.
SHAPES = (("circle", CIRCLE), ("rectangle", RECTANGLE))


def disturbance(quick: bool) -> List[Cell]:
    cells: List[Cell] = []
    for shape, args in SHAPES:
        for tq in (0.0, 0.025, 0.05, 0.1):
            cells += _cells("disturbance", shape, args, ("--disturb-torque", str(tq)),
                            f"tq{tq}")
    return cells


def model_error(quick: bool) -> List[Cell]:
    cells: List[Cell] = []
    for shape, args in SHAPES:
        for ms in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            cells += _cells("model_error", shape, args, ("--mass-scale", str(ms)), f"m{ms}")
    for ms in (1.0, 2.0):
        cells += _cells("model_error", "step6", STEP_SMALL, ("--mass-scale", str(ms)), f"m{ms}")
    return cells


def damping(quick: bool) -> List[Cell]:
    cells: List[Cell] = []
    for shape, args in SHAPES:
        for bd in (0.0, 0.05, 0.5, 5.0, 50.0):
            cells += _cells("damping", shape, args, ("--base-damping", str(bd)), f"bd{bd}")
    return cells


def torque_limit(quick: bool) -> List[Cell]:
    """The 0.25 N.m clamp scaled, in the controllers' own configuration."""
    cells: List[Cell] = []
    for shape, args in SHAPES:
        for scale in (0.5, 0.75, 1.0, 1.5, 2.0):
            lim = 0.25 * scale
            arg = "model_tau_max=[" + ",".join([str(lim)] * 6) + "]"
            cells += _cells("torque_limit", shape, args, ("--set", arg), f"tau{scale}x")
    return cells


#: Joint limits of the scored (left) arm as shipped, [joint2, joint3, joint5].
Q_MIN = (0.05, -2.917, -1.17)
Q_MAX = (3.09, -0.05, 1.17)
#: How far the parked case moves joint3's upper limit below its start (-0.262).
PARK_Q3_MAX = -0.35


def constraint(quick: bool) -> List[Cell]:
    """The configured joint box under stress. Only the NMPC carries it as a
    constraint; OMPC and SMC are told the limits but do not act on them.

    reach   a +y step straightens the elbow (joint3) toward its -0.05 rad
            limit; past ~0.06 m the target is reachable only by leaving the box.
    parked  joint3's limit is moved below where it starts, so the arm begins
            0.088 rad outside its box, with the target where it already is.
    """
    cells: List[Cell] = []
    for dy in (0.04, 0.06, 0.08, 0.10, 0.13):
        cells += _cells("constraint", "reach", ("--step", f"0,{dy}", "--seconds", "8"),
                        extra_name=f"dy{dy}", series=(dy == 0.10))
    q_max = list(Q_MAX)
    q_max[1] = PARK_Q3_MAX
    arg = "limits.q_max=[" + ",".join(str(v) for v in q_max) + "]"
    cells += _cells("constraint", "parked", ("--step", "0,0", "--seconds", "8"),
                    ("--set", arg), series=True)
    return cells


def noise(quick: bool) -> List[Cell]:
    """Flight sensing with every sensor's error (IMU, star tracker, relative nav,
    encoders) scaled together; the estimators are told the scaled figures."""
    cells: List[Cell] = []
    seeds = range(1, _n(quick, SEEDS) + 1)
    for shape, args in SHAPES:
        for scale in (0.5, 1, 2, 4, 8):
            cells += _cells("noise", shape, args,
                            SOURCES[FLIGHT_SENSING] + ("--noise", str(scale)),
                            f"n{scale}", seeds=seeds)
    return cells


def delay(quick: bool) -> List[Cell]:
    """Flight sensing with the whole state the controller is handed (joints and
    base estimate) a further N control ticks late."""
    cells: List[Cell] = []
    seeds = range(1, _n(quick, SEEDS_DELAY) + 1)
    for shape, args in SHAPES:
        for d in (0, 1, 2, 3, 5):
            cells += _cells("delay", shape, args,
                            SOURCES[FLIGHT_SENSING] + ("--delay-ticks", str(d)),
                            f"d{d}", seeds=seeds)
    return cells


def estimator(quick: bool) -> List[Cell]:
    """Every state source on the two shapes: what each estimator gives the arm.
    The rectangle's first seed keeps its per-tick record, for the estimate's
    error against time."""
    cells: List[Cell] = []
    seeds = list(range(1, _n(quick, SEEDS_OBS) + 1))
    for shape, args in SHAPES:
        for src, src_args in SOURCES.items():
            cells += _cells("estimator", shape, args, src_args, src,
                            seeds=([0] if src == "exact" else seeds),
                            series=(shape == "rectangle" and src != "exact"))
    return cells


def warmup(quick: bool) -> List[Cell]:
    """How long the base estimator must run before the arm can start: the task
    begins after 0-60 s of the flight instruments running with the arms at
    rest. The EKF's gains start large and shrink as it learns; raw sensors have
    nothing to learn, so they are the flat reference."""
    cells: List[Cell] = []
    seeds = list(range(1, _n(quick, SEEDS_OBS) + 1))
    for src in ("raw", "ekf"):
        for w in (0, 5, 10, 20, 30, 60):
            cells += _cells("warmup", "rectangle", RECTANGLE,
                            ("--state-source", src, "--nav-warmup", str(w)),
                            f"{src}-w{w}", seeds=seeds)
    return cells


def observer(quick: bool) -> List[Cell]:
    """State source x disturbance-observer feedforward, per controller, on the
    scenarios where an observer or an estimator should matter (and one where it
    should not)."""
    scenarios = (
        ("circle", CIRCLE, ()),
        ("step6", STEP_SMALL, ()),
        ("circle-mirror", CIRCLE, ("--partner", "mirror")),
        ("circle-tq", CIRCLE, ("--disturb-torque", "0.05")),
        ("circle-m2", CIRCLE, ("--mass-scale", "2.0")),
    )
    cells: List[Cell] = []
    seeds = list(range(1, _n(quick, SEEDS_OBS) + 1))
    for scen, args, extra in scenarios:
        for src in ("exact", "raw", FLIGHT_SENSING):
            for ff in (0.0, 0.5, 1.0):
                label = f"{src}-ff{ff}"
                src_args = SOURCES[src] + ("--ff-gain", str(ff))
                cells += _cells("observer", scen, args, tuple(extra) + src_args, label,
                                seeds=([0] if src == "exact" else seeds))
    return cells


# One-at-a-time sensitivity: (controller label, law, base args, knob, values).
# `values` are YAML strings for --set, the first one being the active variant's
# own value (shipped or tuned), which every other value is a multiple of.
def _sensitivity_table():
    if _variant == "tuned":
        return _tuned_sensitivity_table()
    sqp = ("sqp", "nmpc", ())
    itr = ("iter", "nmpc", ("--set", "nmpc.full_nonlinear=false"))

    def scaled(base: Sequence[float], mults: Sequence[float]) -> List[str]:
        return ["[" + ",".join(str(b * m) for b in base) + "]" for m in mults]

    mults = (1.0, 0.1, 0.3, 3.0, 10.0)
    rows = []
    for label, law, base in (sqp, itr):
        rows += [
            (label, law, base, "weights.Q_pos", scaled([10.0, 10.0, 1.0], mults)),
            (label, law, base, "weights.R", scaled([0.1, 0.1, 0.1], mults)),
            (label, law, base, "nmpc.N", ["30", "10", "20", "40", "60"]),
            (label, law, base, "nmpc.Ts", ["0.033", "0.02", "0.05", "0.1"]),
            (label, law, base, "qp.max_iter", ["60", "10", "20", "100"]),
        ]
    rows.append(("sqp", "nmpc", (), "nmpc.sqp_iters", ["1", "2", "3"]))
    rows += [
        ("ompc", "ompc", (), "ompc.Ts", ["1.0", "0.25", "0.5", "2.0", "4.0"]),
        ("ompc", "ompc", (), "ompc.Q", scaled([1e5] * 3, mults)),
        ("smc", "smc", (), "smc.epsilon", ["0.05", "0.01", "0.02", "0.1", "0.2", "0.5"]),
        ("smc", "smc", (), "smc.lam", scaled([3.0] * 3, (1.0, 0.33, 0.67, 1.67, 2.67))),
        ("smc", "smc", (), "smc.alpha", scaled([0.5] * 3, (1.0, 0.2, 0.5, 2.0, 4.0))),
        ("smc", "smc", (), "smc.tau_bound", ["0.0", "0.025", "0.1", "1.0", "4.0"]),
    ]
    return rows


#: Multiples of the tuned value each knob is swept over, the tuned value first.
#: A knob tune.py did not touch (qp.max_iter, nmpc.sqp_iters, nmpc.Ts at its
#: shipped value) is swept around the value the tuned controller runs with.
_TUNED_MULTS = {
    "weights.Q_pos": (1.0, 0.1, 0.3, 3.0, 10.0),
    "weights.R": (1.0, 0.1, 0.3, 3.0, 10.0),
    "nmpc.N": (1.0, 0.5, 0.75, 1.5, 2.0),
    "nmpc.Ts": (1.0, 0.6, 1.5, 3.0),
    "ompc.Ts": (1.0, 0.25, 0.5, 2.0, 4.0),
    "ompc.Q": (1.0, 0.1, 0.3, 3.0, 10.0),
    "smc.epsilon": (1.0, 0.2, 0.4, 2.0, 4.0, 10.0),
    "smc.lam": (1.0, 0.33, 0.67, 1.67, 2.67),
    "smc.alpha": (1.0, 0.2, 0.5, 2.0, 4.0),
    "smc.tau_bound": (1.0, 0.1, 0.3, 3.0, 10.0),
}
#: Knobs swept over a fixed list rather than multiples (iteration counts).
_TUNED_FIXED = {"qp.max_iter": ["60", "10", "20", "100"], "nmpc.sqp_iters": ["1", "2", "3"]}
_TUNED_KNOBS = {
    "sqp": ("weights.Q_pos", "weights.R", "nmpc.N", "nmpc.Ts", "qp.max_iter",
            "nmpc.sqp_iters"),
    "iter": ("weights.Q_pos", "weights.R", "nmpc.N", "nmpc.Ts", "qp.max_iter"),
    "ompc": ("ompc.Ts", "ompc.Q"),
    "smc": ("smc.epsilon", "smc.lam", "smc.alpha", "smc.tau_bound"),
}
#: The shipped value of a knob the tuning left alone.
_SHIPPED_VALUE = {"nmpc.Ts": "0.033", "weights.R": "[0.1,0.1,0.1]"}


def _scale(value: str, m: float, integer: bool) -> str:
    nums = [float(x) for x in value.strip("[]").split(",")]
    out = [max(1, round(v * m)) if integer else v * m for v in nums]
    s = ",".join(str(v) if integer else f"{v:.6g}" for v in out)
    return f"[{s}]" if value.startswith("[") else s


def _tuned_sensitivity_table():
    rows = []
    for label, law, args in active_controllers():
        sets = dict(a.split("=", 1) for a in args[1::2])   # args alternate --set, k=v
        for knob in _TUNED_KNOBS[label]:
            if knob in _TUNED_FIXED:
                values = _TUNED_FIXED[knob]
            else:
                base = sets.get(knob, _SHIPPED_VALUE.get(knob))
                values = [_scale(base, m, knob == "nmpc.N") for m in _TUNED_MULTS[knob]]
            rows.append((label, law, tuple(args), knob, values))
    return rows


def sensitivity(quick: bool) -> List[Cell]:
    cells: List[Cell] = []
    for label, law, base, knob, values in _sensitivity_table():
        for v in values:
            name = _slug("circle", label, knob, v)
            cells.append(Cell("sensitivity", name, law, label,
                              tuple(CIRCLE) + tuple(base) + PLANT +
                              ("--set", f"{knob}={v}")))
    return cells


def timing(quick: bool) -> List[Cell]:
    """Per-tick wall time of each law, repeated as independent processes. Timing
    is the one measurement that is not deterministic, so these are pinned, serial,
    and refused unless the machine is on mains with the performance governor."""
    cells: List[Cell] = []
    reps = _n(quick, REPEATS)
    ctrls = active_controllers()
    for r in range(1, reps + 1):
        for label, law, ctrl_args in ctrls:
            cells.append(Cell("timing", _slug("circle", label, f"r{r}"), law, label,
                              tuple(CIRCLE) + tuple(ctrl_args) + PLANT, seed=r,
                              series=True, timing=True))
    for n in (10, 20, 30, 40, 60):
        for label, law, ctrl_args in ctrls[:2]:
            for r in range(1, _n(quick, 3) + 1):
                cells.append(Cell("timing", _slug("circle", label, f"N{n}", f"r{r}"), law,
                                  label, tuple(CIRCLE) + tuple(ctrl_args) + PLANT +
                                  ("--set", f"nmpc.N={n}"), seed=r, timing=True))
    return cells


STUDIES: Dict[str, Callable[[bool], List[Cell]]] = {
    "nominal": nominal,
    "balanced": balanced,
    "speed": speed,
    "disturbance": disturbance,
    "model_error": model_error,
    "damping": damping,
    "torque_limit": torque_limit,
    "constraint": constraint,
    "noise": noise,
    "delay": delay,
    "observer": observer,
    "estimator": estimator,
    "warmup": warmup,
    "sensitivity": sensitivity,
    "timing": timing,
}


def build(study: str, quick: bool = False) -> List[Cell]:
    cells = STUDIES[study](quick)
    names = [c.name for c in cells]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"study {study!r} has duplicate cell names: {sorted(dup)[:5]}")
    return cells
