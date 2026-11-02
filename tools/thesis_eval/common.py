"""Shared by figures.py and tables.py: where the data is, how a controller is
named and coloured, and how a study's cells are decoded back into their settings.

Host-side (needs pandas): run with the venv in requirements.txt.
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

import studies

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(ROOT, "data", "thesis_eval")
OUT_ROOT = os.path.join(ROOT, "docs", "assets", "thesis")
DATA = OUT = VARIANT = ""


def set_variant(variant: str) -> None:
    """Read the tuned or the shipped studies, and write the figures for them to
    docs/assets/thesis/ (tuned) or docs/assets/thesis/shipped/."""
    global DATA, OUT, VARIANT
    VARIANT = variant
    studies.set_variant(variant)     # the sensitivity cells depend on it
    DATA = DATA_ROOT if variant == "shipped" else os.path.join(DATA_ROOT, "tuned")
    OUT = os.path.join(OUT_ROOT, "shipped") if variant == "shipped" else OUT_ROOT


set_variant(studies.DEFAULT_VARIANT)


def summary_of(variant: str, study: str) -> Optional[pd.DataFrame]:
    """A study's summary for a given variant, whatever the current one is."""
    base = DATA_ROOT if variant == "shipped" else os.path.join(DATA_ROOT, "tuned")
    path = os.path.join(base, study, "summary.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None

# Controller order, label and colour: the same in every figure and table.
# Okabe-Ito, which stays distinguishable for the common colour-vision deficiencies
# and in greyscale by lightness.
CTRL = ["sqp", "iter", "ompc", "smc"]
LABEL = {"sqp": "SQP-NMPC", "iter": "LMPC", "ompc": "OMPC", "smc": "SMC"}
COLOUR = {"sqp": "#0072B2", "iter": "#CC79A7", "ompc": "#E69F00", "smc": "#009E73"}
MARKER = {"sqp": "o", "iter": "s", "ompc": "^", "smc": "D"}

PERIOD_US = 10_000.0     # control period at the shipped 100 Hz
TAU_MAX = 0.25           # shipped per-joint torque limit [N*m]


def summary(study: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA, study, "summary.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def series(study: str, cell: str) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA, study, "series", cell + ".csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------- live --
#: Where a live run's trials and collected summary are (live/driver.py --out,
#: live/collect.py). The tuned settings only: the live stack runs tuned.json.
LIVE_RUN = "main"
LIVE_DIR = os.path.join(DATA_ROOT, "live")

#: The knobs a live cell and its offline twin are matched on, and their nominal
#: values (the controllers' own setting in every other figure).
LIVE_KNOBS = {"speed": 1.0, "mass": 1.0, "damping": 0.0, "tau": 1.0}
LIVE_SOURCES = ["raw", "ekf", "ekf-jkf"]
SOURCE_LABEL = {"exact": "exact state", "raw": "raw sensors", "ekf": "EKF",
                "ekf-jkf": "EKF + joint KF"}


def live_summary(run: str = LIVE_RUN, stale: bool = False) -> Optional[pd.DataFrame]:
    """One row per live trial (live/collect.py's summary.csv), with the knobs under
    the same names as live_twin()'s: shape, speed, mass, damping, tau, source.
    stale=True: the trials run with settings tuned.json no longer has
    (summary_stale.csv, the uncapped first run)."""
    path = os.path.join(LIVE_DIR, run, "summary_stale.csv" if stale else "summary.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df.rename(columns={"mass_scale": "mass", "tau_scale": "tau"})


def live_series(trial_hash: str, run: str = LIVE_RUN) -> Optional[pd.DataFrame]:
    path = os.path.join(LIVE_DIR, run, "trials", trial_hash, "series.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def live_twin() -> Optional[pd.DataFrame]:
    """The offline twins (study live_twin), with the same knob columns as
    live_summary(). Always the tuned variant: the live stack runs tuned.json."""
    df = summary_of("tuned", "live_twin")
    if df is None:
        return None
    df = df.copy()
    df["shape"] = df["scenario"]
    df["speed"] = df["traj_speed"].astype(float)
    df["mass"] = df["mass_scale"].astype(float)
    df["damping"] = df["base_damping"].astype(float)
    df["tau"] = df["cell"].str.extract(r"_tau([0-9.]+)x_")[0].astype(float)
    filt = df["joint_filter"] if "joint_filter" in df else "none"
    df["source"] = df["state_source"].where(filt != "kalman", "ekf-jkf")
    return df


def at_nominal(df: pd.DataFrame, but: Optional[str] = None) -> pd.DataFrame:
    """Rows with every live knob at its nominal value except `but`."""
    keep = pd.Series(True, index=df.index)
    for k, v in LIVE_KNOBS.items():
        if k != but:
            keep &= (df[k].astype(float) - v).abs() < 1e-9
    return df[keep]


# Two-sided 95% Student-t quantiles by degrees of freedom; beyond 30, the normal.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060,
        30: 2.042}


def ci95(values) -> Tuple[float, float]:
    """Mean and the half-width of its 95% confidence interval (Student t). The
    half-width is 0 for a single value: a deterministic run has no spread."""
    x = [float(v) for v in values if pd.notna(v)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(x) / n
    if n == 1:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    dof = n - 1
    t = _T95.get(dof) or (_T95[30] if dof > 30 else _T95[max(k for k in _T95 if k <= dof)])
    return m, t * sd / math.sqrt(n)


def observer_parts(cell: str) -> Tuple[str, str, float]:
    """observer cell name -> (scenario, state source, ff gain)."""
    scen, label = cell.split("_")[:2]
    src, ff = label.split("-ff")
    return scen, src, float(ff)


def first_number(text: str) -> float:
    return float(re.findall(r"-?[0-9.]+(?:e[-+]?[0-9]+)?", text)[0])


def sensitivity_index() -> Dict[str, dict]:
    """sensitivity cell name -> controller, knob, value, and the value as a
    multiple of the shipped one (the first in the table)."""
    out = {}
    for label, law, base, knob, values in studies._sensitivity_table():
        nominal = first_number(values[0])
        if knob == "smc.tau_bound" and nominal == 0.0:
            nominal = TAU_MAX            # 0 means the largest model torque limit
        for i, v in enumerate(values):
            x = first_number(v)
            if knob == "smc.tau_bound" and x == 0.0:
                x = TAU_MAX
            name = studies._slug("circle", label, knob, v)
            out[name] = {"ctrl": label, "knob": knob, "value": v,
                         "mult": x / nominal if nominal else float("nan"),
                         "nominal": i == 0}
    return out


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the standard controller order."""
    return df.assign(_o=df["tag"].map({c: i for i, c in enumerate(CTRL)})) \
             .sort_values("_o").drop(columns="_o")


def ensure_out() -> str:
    os.makedirs(OUT, exist_ok=True)
    return OUT
