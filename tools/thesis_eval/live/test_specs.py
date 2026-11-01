"""The live-study knobs: what a spec holds, and what reaches the parameter files.
Run: python3 -m unittest tools/thesis_eval/live/test_specs.py (no ROS needed)."""
import os
import shutil
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def args(**kw):
    base = dict(tag="sqp", shape="circle", speed=1.0, repeat=1, duration=20.0,
                nav_warmup=15.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


class SpecTest(unittest.TestCase):
    def test_a_nominal_spec_has_no_scale_keys(self):
        s = driver.make_spec(args())
        self.assertNotIn("mass_scale", s)
        self.assertNotIn("tau_scale", s)
        self.assertEqual((s["state"], s["damping"], s["partner"]), ("ekf", 0.0, "hold"))

    def test_knobs_enter_the_spec(self):
        s = driver.make_spec(args(mass_scale=2.0, tau_scale=0.5, damping=5, source="odom"))
        self.assertEqual((s["mass_scale"], s["tau_scale"], s["damping"], s["state"]),
                         (2.0, 0.5, 5.0, "odom"))

    def test_bad_knobs_are_refused(self):
        with self.assertRaises(SystemExit):
            driver.make_spec(args(mass_scale=0.0))
        with self.assertRaises(SystemExit):
            driver.make_spec(args(source="raw"))

    def test_without_nav_the_nav_thread_is_not_required(self):
        cpus = driver.cpu_list
        found = {k: [{"policy": "FIFO", "priority": driver.RT_PRIORITY,
                      "cpus": cpus(driver.CORES[k])}] for k in ("left", "right")}
        self.assertEqual(driver.rt_problems(found, with_nav=False), [])
        self.assertEqual(len(driver.rt_problems(found, with_nav=True)), 1)

    def test_queue_expands_studies_and_skips_partners(self):
        a = types.SimpleNamespace(studies=["nominal", "speed", "balanced"],
                                  shapes=["circle"])
        vs = driver.queue_variations(a)
        self.assertNotIn({"shape": "circle", "partner": "mirror"}, vs)
        self.assertIn({"shape": "circle", "speed": 4}, vs)


class ParamsTest(unittest.TestCase):
    """write_params against the source configs, laid out as installed shares."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for pkg in ("pat_arm_control", "pat_gnc", "pat_simulation"):
            shutil.copytree(os.path.join(SRC, pkg, "config"),
                            os.path.join(self.tmp, pkg, "share", pkg, "config"))
        self.old = driver.prefix
        driver.prefix = lambda pkg: os.path.join(self.tmp, pkg)
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)

    def tearDown(self):
        driver.prefix = self.old
        shutil.rmtree(self.tmp)

    def plant(self):
        return driver.load_yaml(os.path.join(self.work, "plant.yaml"))["pat_simulation"][
            "ros__parameters"]

    def test_nominal_leaves_mass_and_torque_alone(self):
        files = driver.write_params(driver.make_spec(args()), self.work)
        self.assertNotIn("mass_scale_bodies", self.plant())
        self.assertEqual(files["tau_limit"], 0.25)

    def test_mass_scale_reaches_both_arms(self):
        driver.write_params(driver.make_spec(args(mass_scale=2.0)), self.work)
        p = self.plant()
        self.assertEqual(sorted(p["mass_scale_bodies"]),
                         sorted(f"link{n}_{s}" for s in "LR" for n in (2, 3, 5)))
        self.assertEqual(p["mass_scale_factors"], [2.0] * 6)

    def test_tau_scale_scales_the_clamp_and_the_scoring_limit(self):
        files = driver.write_params(driver.make_spec(args(tau_scale=2.0)), self.work)
        common = driver.load_yaml(files["common"])["/**"]["ros__parameters"]
        self.assertEqual(common["model_tau_max"], [0.5] * 6)
        self.assertEqual(files["tau_limit"], 0.5)

    def test_damping_reaches_the_plant(self):
        driver.write_params(driver.make_spec(args(damping=5.0)), self.work)
        self.assertEqual(self.plant()["chaser_damping"], [5.0] * 3)


if __name__ == "__main__":
    unittest.main()
