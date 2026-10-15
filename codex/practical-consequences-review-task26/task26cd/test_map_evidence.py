"""Focused synthetic coverage controls for the saved-map adapter."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("own_map", Path(__file__).with_name("map_evidence.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture(nt=1, width=6, keep=1000):
    args = {"tgrid": [-.5, -.4, nt], "alpha": [-.12, .12, width], "beta": [0., 0., 1],
            "record": 1500, "chunk": width, "eps": [1e-6, 1e-9]}
    positions = M.grid(args)
    arrays = {name: np.full((keep, len(positions)), j + 0.25, dtype=np.float32) for j, name in enumerate(M.NAMES)}
    arrays["cells"] = positions
    cells = []
    for i, (a, b, t) in enumerate(positions):
        means = {"mean_" + name: j + .5 for j, name in enumerate(M.NAMES)}
        cells.append({"alpha": a, "beta": b, "log10_temperature": t, "lyapunov": [.1, .2],
                      "eps": args["eps"][:], "period": 0, "chaotic": True,
                      "boundary_term": -1.9, "frac_gap_below_1": .2, "frac_gap_below_2": .4,
                      "max_rayleigh_change": 7.25, **means})
    progress = []
    for end in range(width, len(cells) + 1, width):
        progress.extend({"recorded": i, "of": 1500, "seconds": 1.} for i in (500, 1000, 1500))
        progress.append({"done": end, "of": len(cells), "chaotic": width, "seconds": 2.})
    return {"args": args, "cells": cells, "bias": None, "seconds": 3.}, arrays, progress


class Controls(unittest.TestCase):
    def test_distinct_prefix_full_means_and_own_max(self):
        d, a, p = fixture()
        ledger = []
        r = M.audit_map(d, a, copy.deepcopy(d["args"]), p, ledger.append)
        self.assertEqual(len(ledger), 6)
        self.assertEqual(ledger[0]["prefix"]["log_g_net"]["prefix_mean_minus_full_record_mean"], -.25)
        self.assertTrue(ledger[0]["max_rayleigh_matches"])
        self.assertEqual(r["full_record_steps_per_cell"], 1500)
        self.assertEqual(r["stored_steps_per_cell"], 1000)

    def test_exact_coordinates_and_dtype_refuse(self):
        d, a, p = fixture()
        a["cells"][0, 0] += 1e-8
        with self.assertRaisesRegex(ValueError, "grid versus NPZ"):
            M.audit_map(d, a, d["args"], p, lambda x: None)
        d, a, p = fixture()
        a["entropy"] = a["entropy"].astype(float)
        with self.assertRaisesRegex(ValueError, "shape/dtype entropy"):
            M.audit_map(d, a, d["args"], p, lambda x: None)

    def test_nonfinite_and_classifier_failure_are_retained(self):
        d, a, p = fixture()
        d["cells"][0]["mean_entropy"] = float("nan")
        d["cells"][0]["lyapunov"][1] = -.1
        a["gap"][0, 0] = float("inf")
        ledger = []
        r = M.audit_map(d, a, d["args"], p, ledger.append)
        self.assertEqual(r["classification_mismatch_indices"], [0])
        self.assertEqual(r["nonfinite_summary_indices"], [0])
        self.assertEqual(r["nonfinite_step_indices"], [0])
        self.assertEqual(ledger[0]["prefix"]["gap"]["positive_infinity_count"], 1)

    def test_completed_progress_and_saved_max_errors(self):
        d, a, p = fixture()
        with self.assertRaisesRegex(ValueError, "progress marker count"):
            M.audit_map(d, a, d["args"], p[:-1], lambda x: None)
        d["cells"][2]["max_rayleigh_change"] = 99.
        r = M.audit_map(d, a, d["args"], p, lambda x: None)
        self.assertEqual(r["max_rayleigh_mismatch_indices"], [2])


if __name__ == "__main__":
    unittest.main()
