"""Synthetic controls only: no original inputs, model imports, or target paths."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import core

spec = importlib.util.spec_from_file_location("audit_driver", Path(__file__).with_name("run.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def fixture():
    d = np.asarray([[0, 0, .01], [1e-4, 0, .01], [1e-5, 0, .01],
                    [.02, 0, .01], [.03, 0, .01], [.04, 0, .01]], dtype="<f4")
    a = np.asarray([[1, 0, 1], [1, 0, 1], [0, 0, 1],
                    [1, 0, 1], [0, 0, 1], [1, 0, 1]], dtype=bool)
    arrays = {c + "_" + k: z.copy() for c in core.COMPARISONS for k, z in (("dist", d), ("agree", a))}
    specification = {"cells": [[0, .1], [.2, .3], [.4, .5]],
                     "source": ["map-A", "map-A", "map-B"],
                     "label": ["chaotic", "periodic", "other"],
                     "map_exponent": [.02, -.1, 0.], "map_period": [0, 2, 0]}
    rows = [
        {"first_step_dist_above_1e-2": 4, "first_disagreement": 3,
         "disagreeing_for_good_from": None, "tail_disagreement_fraction": .5,
         "final_dist": float(np.float32(.04)), "max_dist": float(np.float32(.04)),
         "min_nonzero_dist": float(np.float32(1e-5))},
        {"first_step_dist_above_1e-2": None, "first_disagreement": 1,
         "disagreeing_for_good_from": 0, "tail_disagreement_fraction": 1.,
         "final_dist": 0., "max_dist": 0., "min_nonzero_dist": 0.},
        {"first_step_dist_above_1e-2": None, "first_disagreement": None,
         "disagreeing_for_good_from": None, "tail_disagreement_fraction": 0.,
         "final_dist": float(np.float32(.01)), "max_dist": float(np.float32(.01)),
         "min_nonzero_dist": float(np.float32(.01))}]
    saved = {"cells": copy.deepcopy(specification["cells"]), "labels": specification["label"][:],
             "args": {"steps": 6, "tail": 2, "compare": list(core.COMPARISONS)},
             "comparisons": {c: copy.deepcopy(rows) for c in core.COMPARISONS}}
    return arrays, saved, specification


class Arithmetic(unittest.TestCase):
    def test_full_small_coverage_and_summary_fields(self):
        result = core.calculate(*fixture(), steps=6, tail=2,
                                expected_counts={x: 1 for x in core.LABELS})
        self.assertEqual(result["original_summary_fields_checked"], 84)
        self.assertTrue(result["all_original_summaries_equal"])
        self.assertEqual(len(result["cells"]), 12)
        self.assertEqual(sum(len(g) for g in result["groups"].values()), 12)
        self.assertEqual(result["groups"]["batch64"]["chaotic"]["crossing_count"], 1)
        self.assertEqual(result["seed_definition_differs_cells"], [0])

    def test_threshold_strict_and_1based(self):
        threshold = np.float32(.01)
        d = np.array([threshold, np.nextafter(threshold, np.float32(np.inf))], dtype="<f4")
        r = core.cell_summary(d, np.ones(2, bool), 1)
        self.assertEqual(r["first_step_dist_above_1e-2"], 2)
        self.assertEqual(core.cell_summary(d[::-1], np.ones(2, bool), 1)["first_step_dist_above_1e-2"], 1)

    def test_literal_suffix_zero_vs_one(self):
        d = np.ones(3, dtype="<f4")
        all_false = core.cell_summary(d, np.zeros(3, bool), 2)
        self.assertEqual(all_false["disagreeing_for_good_from"], 0)
        self.assertEqual(all_false["terminal_disagreeing_suffix_start_step_1based"], 1)
        suffix = core.cell_summary(d, np.array([1, 0, 0], bool), 2)
        self.assertEqual(suffix["disagreeing_for_good_from"], 1)
        self.assertEqual(suffix["terminal_disagreeing_suffix_start_step_1based"], 2)

    def test_any_tail_not_final_or_all_tail(self):
        r = core.cell_summary(np.zeros(4, dtype="<f4"), np.array([0, 1, 0, 1], bool), 2)
        self.assertTrue(r["any_tail_disagreement"])
        self.assertFalse(r["final_disagreement"])
        self.assertEqual(r["tail_disagreement_fraction"], .5)
        self.assertIsNone(r["disagreeing_for_good_from"])
        r = core.cell_summary(np.zeros(4, dtype="<f4"), np.array([1, 0, 1, 1], bool), 2)
        self.assertFalse(r["any_tail_disagreement"])

    def test_zero_first_minimum_and_late_clock(self):
        d = np.array([0, 0, 1e-4, 1e-5, .02], dtype="<f4")
        r = core.cell_summary(d, np.ones(5, bool), 2)
        self.assertEqual(r["first_positive_step_1based"], 3)
        self.assertEqual(r["minimum_positive_steps_1based"], [4])
        r.update(cell=0, map_exponent=.1)
        old = core.forecast_row(r, "min_nonzero_dist", 5)
        new = core.forecast_row(r, "first_positive_dist", 5)
        self.assertGreater(old["prediction_duration"], new["prediction_duration"])
        self.assertAlmostEqual(new["prediction_duration"], np.log(.01 / float(d[2])) / .1)
        self.assertEqual(new["first_positive_step_1based"], 3)

    def test_noncrosser_is_not_imputed_at_horizon(self):
        r = core.cell_summary(np.zeros(4, dtype="<f4"), np.ones(4, bool), 2)
        r.update(cell=3, map_exponent=.01)
        old = core.forecast_row(r, "min_nonzero_dist", 4)
        first = core.forecast_row(r, "first_positive_dist", 4)
        self.assertFalse(old["historical_cohort_eligible"])
        self.assertIsNone(old["observed_crossing_step_1based"])
        self.assertTrue(old["prediction_exceeds_horizon"])
        self.assertEqual(first["disposition"], "undefined_seed")

    def test_invalid_prediction_not_silently_removed(self):
        r = {"cell": 0, "map_exponent": 1., "first_positive_dist": .02,
             "first_positive_step_1based": 1, "first_step_dist_above_1e-2": 1}
        f = core.forecast_row(r, "first_positive_dist", 10)
        summary = core.forecast_summary([f])
        self.assertEqual(summary["historical_cohort_n"], 1)
        self.assertEqual(summary["invalid_prediction_cells"], [0])
        self.assertIsNone(summary["median_ratio"])

    def test_empty_and_constant_correlation(self):
        self.assertEqual(core.forecast_summary([])["aggregate_disposition"], "undefined_empty")
        row = {"historical_cohort_eligible": True, "disposition": "defined_positive",
               "cell": 0, "observed_crossing_step_1based": 2, "prediction_duration": 1}
        ans = core.forecast_summary([row, dict(row, cell=1)])
        self.assertEqual(ans["median_ratio"], 2.)
        self.assertIsNone(ans["loglog_r"])
        self.assertIn("constant", ans["aggregate_disposition"])

    def test_known_log_correlation(self):
        rows = [{"historical_cohort_eligible": True, "disposition": "defined_positive",
                 "cell": i, "observed_crossing_step_1based": x*x, "prediction_duration": x}
                for i, x in enumerate([1, 2, 4])]
        self.assertAlmostEqual(core.forecast_summary(rows)["loglog_r"], 1.)

    def test_even_median_quantile_count_and_empty(self):
        q = core.distribution([0., 10.], [4, 5])
        self.assertEqual(q["median"], 5.)
        self.assertEqual(q["q95"], 9.5)
        self.assertEqual(sum(x <= q["q95"] for x in [0, 10]), 1)
        self.assertEqual(q["max_cells"], [5])
        self.assertIsNone(core.distribution([], [])["median"])
        self.assertGreaterEqual(float(core.upward(.02029819399365)), .02029819399365)

    def test_all_discrepancies_are_retained(self):
        arrays, saved, s = fixture()
        saved["comparisons"]["batch64"][0]["first_disagreement"] = 2
        saved["comparisons"]["bf16"][1]["max_dist"] = 1.
        r = core.calculate(arrays, saved, s, steps=6, tail=2)
        self.assertEqual(len(r["original_summary_differences"]), 2)
        self.assertEqual(len(r["cells"]), 12)
        self.assertFalse(r["all_original_summaries_equal"])

    def test_array_refusals(self):
        for kind in ("missing", "shape", "dtype", "negative", "nan", "inf"):
            with self.subTest(kind=kind):
                arrays, saved, s = fixture()
                if kind == "missing":
                    arrays.pop("bf16_agree")
                elif kind == "shape":
                    arrays["batch64_dist"] = arrays["batch64_dist"][:-1]
                elif kind == "dtype":
                    arrays["batch64_dist"] = arrays["batch64_dist"].astype(float)
                else:
                    arrays["batch64_dist"][0, 0] = {"negative": -1, "nan": np.nan, "inf": np.inf}[kind]
                with self.assertRaises(ValueError):
                    core.calculate(arrays, saved, s, steps=6, tail=2)

    def test_metadata_coverage_refusals(self):
        for kind in ("duplicate", "labels", "summary_missing", "summary_extra", "cohort"):
            with self.subTest(kind=kind):
                arrays, saved, s = fixture()
                counts = {x: 1 for x in core.LABELS}
                if kind == "duplicate":
                    s["cells"][1] = s["cells"][0][:]
                elif kind == "labels":
                    saved["labels"][0] = "other"
                elif kind == "summary_missing":
                    saved["comparisons"]["batch32"].pop()
                elif kind == "summary_extra":
                    saved["comparisons"]["batch32"][0]["extra"] = 1
                else:
                    counts["other"] = 2
                with self.assertRaises(ValueError):
                    core.calculate(arrays, saved, s, steps=6, tail=2, expected_counts=counts)

    def test_input_result_not_modified(self):
        arrays, saved, s = fixture()
        original = copy.deepcopy(saved)
        core.calculate(arrays, saved, s, steps=6, tail=2)
        self.assertEqual(saved, original)

    def test_prepared_plan_refuses_before_inputs(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            driver.authenticate({"schema": "task26b-saved-audit-plan-v1", "admission": "prepared"})

    def test_duplicate_and_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    driver.read_json(path)


if __name__ == "__main__":
    unittest.main()
