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
