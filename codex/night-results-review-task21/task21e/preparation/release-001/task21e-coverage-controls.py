"""Only the added saved-curve coverage path; manufactured data only."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import time
import unittest

spec = importlib.util.spec_from_file_location("cusp_scalar", "/tmp/task21e-analyze.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def curve(n=4):
    points = []
    for i in range(n):
        sign = 1.0 if i == 0 or i >= 3 else -1.0
        points.append({"index": i, "parameter_tangent": [sign, 0.0],
                       "cusp_candidate": i in (1, 3), "theta": [i / 100, 0.5],
                       "residual": 1e-12, "g": 1e-11, "critical_multiplier": [1.0, 0.0]})
    return {"kind": "fold", "period": 7, "args": {"alpha_scale": 10.0}, "points": points}


class Controls(unittest.TestCase):
    def test_exact_reversals_and_coverage(self):
        c = M.analyze_curve(curve(), "A")
        self.assertFalse(c["candidate_mismatches"])
        self.assertEqual(c["computed_reversals"], [{"left": 0, "right": 1}, {"left": 2, "right": 3}])
        selection = [{"id": "x", "curve_id": "A", "left": 0}, {"id": "y", "curve_id": "A", "left": 2}]
        self.assertTrue(M.reversal_coverage([c], selection)["selection_exactly_covers_declared_curve_reversals"])

    def test_extra_missing_nonfinite_and_wrong_saved_flag_retained(self):
        raw = curve()
        raw["points"][1]["cusp_candidate"] = False
        c = M.analyze_curve(raw, "A")
        self.assertEqual(len(c["candidate_mismatches"]), 1)
        report = M.reversal_coverage([c], [{"id": "x", "curve_id": "A", "left": 1}])
        self.assertEqual(len(report["unselected_computed_reversals"]), 2)
        self.assertEqual(len(report["selected_without_computed_reversal"]), 1)
        raw["points"][2]["parameter_tangent"][0] = float("nan")
        result = M.analyze_curve(raw, "A")
        self.assertTrue(result["rows"][2]["nonfinite_paths"])
        json.dumps(result, allow_nan=False)

    def test_missing_duplicate_and_index_schema_refuse(self):
        c = M.analyze_curve(curve(), "A")
        with self.assertRaises(ValueError):
            M.reversal_coverage([c], [{"id": "x", "curve_id": "missing", "left": 0}])
        with self.assertRaises(ValueError):
            M.reversal_coverage([c], [{"id": "x", "curve_id": "A", "left": 0},
                                      {"id": "y", "curve_id": "A", "left": 0}])
        raw = curve()
        raw["points"][1]["index"] = 0
        with self.assertRaises(ValueError):
            M.analyze_curve(raw, "A")


def main():
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority > 10:
        raise RuntimeError("exact nice10 required")
    os.nice(10 - priority)
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    out = Path("/tmp/task21e-coverage-controls-001")
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    log = io.StringIO()
    outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (out / "controls.log").write_text(log.getvalue())
    tic = time.perf_counter()
    manufactured = [curve(512) for _ in range(7)]
    raw = json.dumps(manufactured, allow_nan=False).encode()
    input_pin = hashlib.sha256(raw).hexdigest()
    parsed = json.loads(raw)
    results = [M.analyze_curve(c, f"curve-{i}") for i, c in enumerate(parsed)]
    payload = json.dumps(results, indent=2, allow_nan=False).encode()
    (out / "manufactured-curves.json").write_bytes(payload)
    report = {"scope": "manufactured-only added coverage path", "tests_run": outcome.testsRun,
              "successful": outcome.wasSuccessful(), "failures": len(outcome.failures), "errors": len(outcome.errors),
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in [__file__, "/tmp/task21e-analyze.py"]},
              "source_curve_count": 7, "manufactured_points": 7 * 512,
              "manufactured_json_input_bytes": len(raw), "manufactured_input_sha256": input_pin,
              "serialized_output_bytes": len(payload),
              "manufacture_hash_parse_arithmetic_serialize_write_seconds": time.perf_counter() - tic,
              "elapsed_before_report_write_seconds": time.perf_counter() - started,
              "peak_rss_kib_observed_before_report_write": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(log.getvalue(), end="")
    print(json.dumps(report, indent=2))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
