"""Tiny synthetic controls only; no producer imports or target files."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("own_statistics", Path(__file__).with_name("statistics.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def maps(nrows=2, width=12):
    cells = []
    for j in range(nrows):
        for k in range(width):
            cells.append({"alpha": 0.0 if k < 6 else .03, "beta": 0.,
                          "log10_temperature": -.5 + j * .1, "chaotic": k < 3 + j % 2,
                          "period": 0 if k < 3 + j % 2 else 4,
                          **{f: float(k + j * (k < 3 + j % 2) * 2 + i / 10)
                             for i, f in enumerate(S.FIELDS)}})
    return cells


def crossover():
    selection = {"cells": [[-.2 if i < 8 else .2, i / 100] for i in range(24)],
                 "labels": ["chaotic"] * 16 + ["periodic"] * 8,
                 "map_exponent": [.3] * 16 + [-.3] * 8}
    cells = []
    for i, (t, a) in enumerate(selection["cells"]):
        arms = dict(zip(S.ARMS, (0., 1., .5, 1.2)))
        cells.append({"log10_temperature": t, "alpha": a, "twin_exponent": 1.2,
                      **arms, **{k + "_windows": [v - .03, v - .01, v + .01, v + .03]
                                 for k, v in arms.items()}})
    return {"args": {"record": 1000, "window": 250}, "cells": cells,
            "labels": selection["labels"][:]}, selection


class Controls(unittest.TestCase):
    def test_weighted_temperature_estimate(self):
        cells = maps()
        r = S.row_contrast(cells, S.FIELDS[0], "periodic", np.random.default_rng(7), 11)
        # Row 0: mean(0,1,2)-mean(3..11)=-6, weight3.
        # Row 1: mean(2,3,4,5)-mean(4..11)=-4, weight4.
        self.assertEqual(r["summary"]["estimate"], -34 / 7)
        self.assertEqual(r["summary"]["group_a_cells"], 7)
        self.assertEqual(r["summary"]["group_b_cells"], 17)

    def test_overlap_and_field_specific_exclusion(self):
        cells = maps()
        cells[0][S.FIELDS[0]] = float("nan")
        r = S.row_contrast(cells, S.FIELDS[0], "near_zero", np.random.default_rng(0), 3)
        self.assertFalse(r["rows"][0]["admitted"])
        self.assertEqual(r["rows"][0]["nonfinite_field_indices"], [0])
        self.assertEqual(r["rows"][1]["overlap_indices"], [12, 13, 14, 15])
        self.assertTrue(r["overlap_in_an_included_row"])

    def test_no_eligible_rows_does_not_consume_rng(self):
        rng = np.random.default_rng(0)
        before = copy.deepcopy(rng.bit_generator.state)
        r = S.row_contrast(maps(width=2), S.FIELDS[0], "periodic", rng, 3)
        self.assertIsNone(r["summary"])
        self.assertEqual(before, rng.bit_generator.state)

    def test_single_row_bootstrap_is_constant(self):
        r = S.row_contrast(maps(1), S.FIELDS[0], "periodic", np.random.default_rng(0), 7)
        self.assertEqual(r["summary"]["ci95"], [-6., -6.])
        self.assertEqual(r["bootstrap_statistics"], [-6.] * 7)
        self.assertEqual(r["summary"]["p_permutation"], (r["permutation_exceedances"] + 1) / 8)

    def test_shared_rng_order_and_no_reset(self):
        a = maps()
        xo, selection = crossover()
        combined = S.combined_statistics(a, a, xo, selection, 5)
        rng = np.random.default_rng(0)
        for fields, group in ((S.FIELDS[:2], "near_zero"), (S.FIELDS, "periodic")):
            for f in fields:
                S.row_contrast(a, f, group, rng, 5)
        expected = S.xo_statistics(xo, selection, rng, 5)
        self.assertEqual(combined["details"]["XO"], expected)
        self.assertNotEqual(expected["contrasts"]["readout_effect (10-00)"]["rng_before"],
                            np.random.default_rng(0).bit_generator.state)

    def test_fixed_xo_membership_and_failed_aggregate_validation(self):
        xo, selection = crossover()
        xo["cells"][1]["twin_exponent"] = -4.
        xo["cells"][20]["twin_exponent"] = -12.
        r = S.xo_statistics(xo, selection, np.random.default_rng(0), 5)
        self.assertEqual(r["contrasts"]["readout_effect (10-00)"]["cell_indices"], list(range(2, 16)))
        self.assertEqual(r["summary"]["validation_cells_within_0.02"], 22)
        self.assertFalse(r["gates"]["all_24_validation_pass"])

    def test_signed_average_closeness_is_not_mean_absolute(self):
        xo, selection = crossover()
        for i in range(2, 16):
            xo["cells"][i]["arm11"] = 0. if i % 2 else 2.
        r = S.xo_statistics(xo, selection, np.random.default_rng(0), 5)
        self.assertEqual(r["gates"]["absolute_mean_arm10_minus_arm11"], 0.)
        self.assertEqual(r["gates"]["mean_absolute_difference_descriptive_only"], 1.)
        self.assertTrue(r["gates"]["original_sufficiency_joint_pass"])

    def test_xo_window_coverage_and_coordinates_refuse(self):
        xo, selection = crossover()
        xo["cells"][0]["arm00_windows"] = [0.]
        with self.assertRaisesRegex(ValueError, "XO windows"):
            S.xo_statistics(xo, selection, np.random.default_rng(0), 5)
        xo, selection = crossover()
        xo["cells"][0]["alpha"] += 1e-5
        with self.assertRaisesRegex(ValueError, "XO coordinate"):
            S.xo_statistics(xo, selection, np.random.default_rng(0), 5)

    def test_finite_range_includes_actual_extremes(self):
        r = S.finite_summary([3., -8., float("nan"), float("inf"), float("-inf")])
        self.assertEqual((r["minimum"], r["maximum"], r["finite_count"]), (-8., 3., 2))
        self.assertEqual((r["nan_count"], r["positive_infinity_count"], r["negative_infinity_count"]), (1, 1, 1))

    def test_positive_draw_count_required(self):
        with self.assertRaisesRegex(ValueError, "draw count"):
            S.row_contrast(maps(), S.FIELDS[0], "periodic", np.random.default_rng(0), 0)


if __name__ == "__main__":
    unittest.main()
