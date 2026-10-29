"""The resource guard: each limit stops a trial, the stricter start limits keep one
from beginning on a machine already short of memory or hot, and the extremes seen
are kept for the report. Run: python3 -m unittest tools/thesis_eval/live/test_guard.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402

GIB = 1024 * 1024


def guard(avail=8 * GIB, swap=0, temp=50.0, rss=GIB):
    return driver.Guard(mem=lambda: {"avail": avail, "swap_used": swap},
                        temp=lambda: temp, rss=lambda: rss)


class GuardTest(unittest.TestCase):
    def test_a_healthy_machine_passes(self):
        guard().check()
        guard().check(starting=True)

    def test_low_memory_stops_a_trial(self):
        with self.assertRaises(driver.Unfit) as e:
            guard(avail=driver.MEM_ABORT_KB - 1).check()
        self.assertIn("memory", str(e.exception))

    def test_a_trial_does_not_start_where_it_would_soon_stop(self):
        # Between the two limits a running trial is left alone, a new one is not begun.
        mid = (driver.MEM_ABORT_KB + driver.MEM_START_KB) // 2
        guard(avail=mid).check()
        with self.assertRaises(driver.Unfit):
            guard(avail=mid).check(starting=True)

    def test_swapping_stops_a_trial(self):
        with self.assertRaises(driver.Unfit) as e:
            guard(swap=driver.SWAP_ABORT_KB + 1).check()
        self.assertIn("swapping", str(e.exception))

    def test_the_nodes_own_footprint_is_limited(self):
        with self.assertRaises(driver.Unfit) as e:
            guard(rss=driver.RSS_ABORT_KB + 1).check()
        self.assertIn("nodes hold", str(e.exception))

    def test_heat_stops_a_trial_and_blocks_a_start(self):
        with self.assertRaises(driver.Unfit):
            guard(temp=driver.TEMP_ABORT_C).check()
        warm = (driver.TEMP_START_C + driver.TEMP_ABORT_C) / 2
        guard(temp=warm).check()
        with self.assertRaises(driver.Unfit):
            guard(temp=warm).check(starting=True)

    def test_a_machine_with_no_temperature_sensor_is_not_blocked(self):
        g = driver.Guard(mem=lambda: {"avail": 8 * GIB, "swap_used": 0},
                         temp=lambda: None, rss=lambda: 0)
        g.check()
        self.assertIsNone(g.peak["temp_max_c"])

    def test_the_extremes_are_remembered(self):
        temps, mems = iter([50.0, 71.0, 60.0]), iter([8 * GIB, 5 * GIB, 7 * GIB])
        g = driver.Guard(mem=lambda: {"avail": next(mems), "swap_used": 0},
                         temp=lambda: next(temps), rss=lambda: 2 * GIB)
        for _ in range(3):
            g.check()
        self.assertEqual(g.peak["temp_max_c"], 71.0)
        self.assertEqual(g.peak["mem_avail_min_kb"], 5 * GIB)
        self.assertEqual(g.peak["rss_max_kb"], 2 * GIB)

    def test_the_nodes_memory_is_only_read_every_fourth_look(self):
        calls = []
        g = driver.Guard(mem=lambda: {"avail": 8 * GIB, "swap_used": 0},
                         temp=lambda: 50.0, rss=lambda: calls.append(1) or GIB)
        for _ in range(8):
            g.check()
        self.assertEqual(len(calls), 2)

    def test_cpu_lists_parse(self):
        self.assertEqual(driver.parse_cpus("0,1,12-14"), {0, 1, 12, 13, 14})


if __name__ == "__main__":
    unittest.main()
