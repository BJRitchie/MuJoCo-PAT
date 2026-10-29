"""The studies that run on the live stack, as lists of scenario variations.

Each study is the offline study of the same name (studies.py), cell for cell, with
the knob expressed the way the live stack takes it. A variation is a dict of the
fields `driver.make_spec` accepts on top of a controller and a shape:

    speed        trajectory speed factor (circle period divided, rectangle speed multiplied)
    partner      hold | same | mirror: what the other arm does
    source       ekf (the nav_node estimate) | raw (the flight sensors, no estimator)
    mass_scale   factor on every arm link's mass and inertia, as the bench's --mass-scale
    damping      table drag on all three planar DOF (0 = frictionless)
    tau_scale    factor on the controllers' torque limit (the 0.25 N.m clamp)

What has no live equivalent stays offline and is listed in OFFLINE with the reason,
not faked:

    step6, step12    a single setpoint step needs a one-shot target the publisher has no
                     pattern for yet
    disturbance      the bench applies a torque to one JOINT; the simulator's
                     disturbance model applies a wrench to a BODY, a different
                     disturbance
    the sweeps       noise, delay, warm-up, the observer grid, sensitivity, constraint
                     and tuning run offline by design (hundreds to thousands of cells)
"""

SHAPES = ("circle", "rectangle")


def nominal():
    """The four controllers on the headline shapes, partner held."""
    return [{"shape": s} for s in SHAPES]


def balanced():
    return [{"shape": s, "partner": p} for s in SHAPES for p in ("hold", "mirror", "same")]


def speed():
    return ([{"shape": "circle", "speed": f} for f in (1, 2, 3, 4)] +
            [{"shape": "rectangle", "speed": f} for f in (1, 2, 3)])


def model_error():
    return [{"shape": s, "mass_scale": m} for s in SHAPES for m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]


def damping():
    return [{"shape": s, "damping": d} for s in SHAPES for d in (0.0, 0.05, 0.5, 5.0, 50.0)]


def torque_limit():
    return [{"shape": s, "tau_scale": t} for s in SHAPES for t in (0.5, 0.75, 1.0, 1.5, 2.0)]


def estimator():
    """What each estimator gives the arm. `exact` has no live equivalent (the live
    stack always runs on flight sensing); the joint filter needs wiring into the arm node."""
    return [{"shape": s, "source": src} for s in SHAPES for src in ("raw", "ekf")]


STUDIES = {"nominal": nominal, "balanced": balanced, "speed": speed,
           "model_error": model_error, "damping": damping, "torque_limit": torque_limit,
           "estimator": estimator}

OFFLINE = {
    "step6 / step12": "no one-shot step target in the publisher",
    "disturbance": "joint torque in the bench, body wrench in the simulator",
    "noise, delay, warm-up, observer, sensitivity, constraint, tuning": "offline sweeps",
    "estimator source 'exact'": "the live stack is always on flight sensing",
    "estimator source 'ekf-jkf'": "joint filter not wired into the arm node",
}
