"""Manufactured-only controls and small resource measurement for Task21e."""
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import time
import unittest


def load(path):
    spec = importlib.util.spec_from_file_location("own_cusp_scalars", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("/tmp/task21e-analyze.py")


def record(rounds=3, per_round=5):
    points = [{"theta": [0.0, 1.0], "f": -1.0, "round": 0},
              {"theta": [0.1, 1.0], "f": 1.0, "round": 0}]
    brackets = [{"round": 0, "gap": 1.0, "f": [-1.0, 1.0]}]
    for rnd in range(1, rounds + 1):
        group = []
        for j in range(per_round):
            f = -float(per_round - 1 - j) / per_round if j < per_round - 1 else 1.0 / per_round
            group.append({"round": rnd, "theta": [rnd + j / 10, 1.0], "f": f,
                          "arclength": (j + 1) * 0.25, "critical_multiplier": [1.0, 0.0],
                          "other_spectral_radius": 0.2})
        points.extend(group)
        brackets.append({"round": rnd, "gap": 0.5 ** rnd,
                         "f": [group[-2]["f"], group[-1]["f"]],
                         "theta": [group[-2]["theta"], group[-1]["theta"]]})
    return {"test": "cusp", "kind": "fold", "points": points, "brackets": brackets,
            "initial_f_max": 1.0, "btol": 0.01, "located": True,
            "checks": {"b_small": True, "c_nonzero": True, "unfolding_regular": True},
            "args": {"rounds": rounds, "per_round": per_round, "btol": 0.01},
            "final": {"theta": [0.2, 0.7], "log10_temperature": 0.2, "alpha": 0.7 / 10,
                      "f": 0.001, "b": 7.0, "c": 3.0, "c_plateau_relative_spread": 0.01,
                      "J": [[3.0, 0.0], [0.0, 4.0]], "det_J": 12.0,
                      "singular_values": [4.0, 3.0], "condition": 4.0 / 3,
                      "multiplier": [1.0, 0.0], "critical_multiplier": [1.0, 0.0],
                      "second_modulus": 0.2, "secant_step": 0.125}}


class Controls(unittest.TestCase):
    def test_complete_literal_record(self):
        result = M.analyze_record(record(), "manufactured")
        self.assertTrue(result["source_located_reconstructed"])
        self.assertFalse(result["nonexact_comparisons"])
        self.assertFalse(result["cubic_ladder"]["available"])
        self.assertEqual(result["small_step_oriented_f"], 0.001)
        self.assertEqual(result["coarser_unfolding_b"], 7.0)

    def test_relative_tolerance_and_strict_cubic_spread(self):
        r = record()
        r["final"]["f"] = 0.01
        r["final"]["c_plateau_relative_spread"] = 0.1
        result = M.analyze_record(r, "boundary")
        self.assertTrue(result["source_checks_reconstructed"]["b_small"])
        self.assertFalse(result["source_checks_reconstructed"]["c_nonzero"])
        self.assertFalse(result["source_located_reconstructed"])
        self.assertTrue(result["nonexact_comparisons"])

    def test_null_spread_is_literal_source_acceptance(self):
        r = record()
        r["final"]["c_plateau_relative_spread"] = None
        self.assertTrue(M.analyze_record(r, "null")["source_checks_reconstructed"]["c_nonzero"])

    def test_source_spectral_exact_boundary(self):
        self.assertTrue(M.spectral([1, 1e-6], 0.9995, 1e-6, 0.5)["source_accepts"])
        self.assertFalse(M.spectral([1, 0], math.nextafter(0.9995, math.inf), 1e-6, 0.5)["source_accepts"])
        self.assertFalse(M.spectral([1, math.nextafter(1e-6, math.inf)], 0.1, 1e-6, 0.5)["source_accepts"])

    def test_nonfinite_is_preserved_separately_from_literal_gate(self):
        r = record()
        r["final"]["c"] = float("nan")
        r["final"]["c_plateau_relative_spread"] = None
        result = M.analyze_record(r, "nan")
        self.assertIn("/final/c", result["nonfinite_paths"])
        self.assertTrue(result["source_checks_reconstructed"]["c_nonzero"])
        self.assertEqual(result["original"]["final"]["c"], {"nonfinite": "nan"})
        json.dumps(result, allow_nan=False)

    def test_first_absolute_plateau_and_terms(self):
        table = [{"c": c, "c_direct": 0.5, "c_centre_manifold": c - 0.5}
                 for c in [1.0, 3.0, 5.0, 10.0]]
        value = M.cubic_ladder(table)
        self.assertEqual(value["selected_indices"], [0, 1])
        self.assertEqual(value["c"], 2)
        self.assertEqual(value["spread"], 1)
        self.assertTrue(all(x["exact_equal"] for x in value["rows"]))

    def test_matrix_known_values_and_signed_determinant(self):
        self.assertEqual(M.matrix_scalars([[3, 0], [0, -4]])["det_J"], -12)
        self.assertEqual(M.matrix_scalars([[3, 0], [0, -4]])["singular_values"], [4, 3])
        result = M.matrix_scalars([[1, 2], [2, 4]])
        self.assertEqual(result["det_J"], 0)
        self.assertEqual(result["condition"], math.inf)

    def test_first_saved_crossing_omits_corrected_initial_theta(self):
        r = record(rounds=1)
        r["points"] = r["points"][:2] + [{"round": 1, "theta": [0.1, 1.0], "f": 1.0,
                                            "arclength": 0.25}]
        r["brackets"][1].update(f=[-1.0, 1.0], theta=[[0.001, 1.0], [0.1, 1.0]])
        result = M.analyze_record(r, "omitted-corrector")
        self.assertFalse(result["nonexact_comparisons"])
        self.assertIn("unavailable", result["bracket_checks"]["rows"][1]["left_theta_reconstruction"])

    def test_first_crossing_stops_before_extra_points(self):
        r = record(rounds=1)
        r["points"][3]["f"] = 0.1
        result = M.analyze_record(r, "extra")
        paths = [x["path"] for x in result["nonexact_comparisons"]]
        self.assertIn("/round/1/stopped_at_first_crossing", paths)

    def test_missing_or_wrong_test_refuses(self):
        for mutate in (lambda r: r.pop("initial_f_max"), lambda r: r.update(test="crossing")):
            r = record()
            mutate(r)
            with self.assertRaises((KeyError, ValueError)):
                M.analyze_record(r, "invalid")


def main():
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority > 10:
        raise RuntimeError("exact nice10 required")
    if priority < 10:
        os.nice(10 - priority)
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    out = Path("/tmp/task21e-controls-001")
    out.mkdir(exist_ok=False)
    source_pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), Path("/tmp/task21e-analyze.py")]}
    start = time.perf_counter()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (out / "controls.log").write_text(stream.getvalue())
    timings = []
    for label, rounds in [("typical-nine", 3), ("larger-manufactured-nine", 16)]:
        tic = time.perf_counter()
        payload = [M.analyze_record(record(rounds=rounds), f"manufactured-{i}") for i in range(9)]
        data = json.dumps(payload, allow_nan=False, sort_keys=True, indent=2).encode()
        (out / (label + ".json")).write_bytes(data)
        timings.append({"label": label, "records": 9, "rounds_per_record": rounds,
                        "points_per_record": rounds * 5 + 2, "bytes": len(data),
                        "through_serialization_and_write_seconds": time.perf_counter() - tic})
    report = {"scope": "manufactured only; no target inputs", "source_sha256": source_pins,
              "tests_run": result.testsRun, "successful": result.wasSuccessful(),
              "failures": len(result.failures), "errors": len(result.errors), "measurements": timings,
              "elapsed_before_report_write_seconds": time.perf_counter() - start,
              "peak_rss_kib_observed_before_report_write": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "nice": os.getpriority(os.PRIO_PROCESS, 0), "address_space_bytes": 1024 ** 3}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(stream.getvalue(), end="")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
