"""Focused serialization, comparison and frozen XO adapter controls only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parent


def own(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R, F = own("run"), own("test_statistics")


class Controls(unittest.TestCase):
    def test_R4_descriptive_all_rows_finite_counts_and_undefined(self):
        alpha, bias = F.maps(), F.maps()
        for cell in bias:
            for field in R.S.FIELDS[:2]:
                cell[field] += 4.
        bias[0][R.S.FIELDS[0]] = float("nan")
        for cell in bias[12:]:
            cell["chaotic"] = False
        result = R.matched_bias_factors(alpha, bias)
        self.assertEqual(len(result["rows"]), 2)
        first = result["rows"][0]["fields"][R.S.FIELDS[0]]
        self.assertEqual(first["bias"]["finite_count"], 2)
        self.assertEqual(first["bias"]["nonfinite_indices"], [0])
        self.assertEqual(first["bias_minus_alpha"], 4.5)
        self.assertIsNone(result["rows"][1]["fields"][R.S.FIELDS[0]]["bias_minus_alpha"])

    def test_nonfinite_and_numpy_flags_serialize_without_fill(self):
        result = R.safe({"values": [float("nan"), float("inf"), float("-inf"), np.bool_(False)]})
        self.assertEqual(result["values"], ["NaN", "Infinity", "-Infinity", False])
        json.dumps(result, allow_nan=False)

    def test_every_nonzero_and_undefined_comparison_retained(self):
        rows = R.compare({"a": [1., 2.], "b": None, "c": float("nan")},
                         {"a": [1., 2. + 1e-13], "b": 1., "c": float("nan")})
        self.assertEqual(len(rows), 4)
        self.assertNotEqual(rows[1]["signed_difference"], 0.)
        self.assertTrue(rows[1]["within_1e-12_absolute"])
        self.assertFalse(rows[2]["exact"])
        self.assertEqual(rows[3]["status"], "nonfinite_numeric")

    def test_xo_exact_count_and_completed_progress(self):
        data, selection = F.crossover()
        progress = [{"recorded": t, "of": 1000, "seconds": 1.} for t in (250, 500, 750, 1000)] + [{"done": 24, "of": 24}]
        R.xo_coverage(data, selection, data["args"], progress)
        with self.assertRaisesRegex(ValueError, "progress count"):
            R.xo_coverage(data, selection, data["args"], progress[:-1])
        data["cells"].pop()
        with self.assertRaisesRegex(ValueError, "fixed 24"):
            R.xo_coverage(data, selection, data["args"], progress)

    def test_precreation_failure_never_writes_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "already-there"
            out.mkdir()
            marker = out / "original.txt"
            marker.write_text("immutable")
            with mock.patch("sys.argv", ["run.py", "--card", "missing-card", "--out", str(out)]), \
                 mock.patch.object(R, "read", side_effect=ValueError("precreation refusal")):
                with self.assertRaisesRegex(ValueError, "precreation refusal"):
                    R.main()
            self.assertEqual([p.name for p in out.iterdir()], ["original.txt"])
            self.assertEqual(marker.read_text(), "immutable")


if __name__ == "__main__":
    unittest.main()
