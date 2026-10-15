"""Focused admission/CLI controls using manufactured scientific JSON only."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


M = module("own_cusp", "/tmp/task21e-analyze.py")
FIX = module("manufactured_cusps", "/tmp/task21e-controls.py")
ROOT = Path("/tmp/task21e-driver-controls-001")


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return pin(path)


def fixture():
    source = ROOT / "analyze.py"
    source.write_bytes(Path("/tmp/task21e-analyze.py").read_bytes())
    scientific = ROOT / "manufactured"
    scientific.mkdir()
    source_root = Path("/data2/neural-shrimp-data/codex/paper-review-task27/source-001/tools")
    source_files = [{"source_commit": "manufactured-source-association", "repository_path": "tools/" + name,
                     **pin(source_root / name)} for name in ["cusp_locate.py", "bifurcation_curves.py"]]
    source_map = {"schema": "task21e-source-map-v1", "source_files": source_files, "claims": []}
    source_pin = write(scientific / "source-map.json", source_map)
    authority = write(scientific / "authority.json", {"schema": "manufactured-only", "actual_target": False})
    curves, records = [], []
    for index in range(7):
        count = 2 if index < 2 else 1
        points = [{"index": i, "parameter_tangent": [(-1.0) ** i if i <= count else (-1.0) ** count, 0.0],
                   "cusp_candidate": 0 < i <= count} for i in range(4)]
        raw = {"kind": "fold", "period": 7, "args": {"alpha_scale": 10.0}, "points": points}
        binding = write(scientific / f"curve-{index}.json", raw)
        curve = {"id": f"curve-{index}", "declared_kind": "fold", "declared_period": 7,
                 "declared_args": raw["args"], "point_count": 4, "selected_left_indices": list(range(count)),
                 "file": binding, "association_limit": "manufactured"}
        curves.append(curve)
        for left in range(count):
            raw = FIX.record()
            raw.update(curve=curve["id"], left=left, period=7)
            raw["args"].update(curve=curve["id"], left=left, test="cusp")
            identity = f"cusp-{index}-{left}"
            binding = write(scientific / (identity + ".json"), raw)
            records.append({"id": identity, "test": "cusp", "curve_id": curve["id"], "left": left,
                            "right": left + 1, "file": binding, "original_path": binding["path"],
                            "attempt": "manufactured", "source_commit": "manufactured-source-association",
                            "declared_args": raw["args"], "declared_period": 7, "declared_kind": "fold",
                            "curve_file": curve["file"], "curve_metadata_scale": 10.0,
                            "schema": {"points": {"length": len(raw["points"])}, "brackets": {"length": 4}}})
    for index in range(6):
        raw = FIX.record()
        raw.update(test="crossing", curve="unselected-other", left=index, period=2)
        raw["args"].update(curve=raw["curve"], left=index, test="crossing")
        identity = f"other-{index}"
        binding = write(scientific / (identity + ".json"), raw)
        records.append({"id": identity, "test": "crossing", "curve_id": raw["curve"], "left": index,
                        "right": index + 1, "file": binding, "original_path": binding["path"],
                        "attempt": "manufactured", "source_commit": "manufactured-source-association",
                        "declared_args": raw["args"], "declared_period": 2, "declared_kind": "fold",
                        "schema": {"points": {"length": len(raw["points"])}, "brackets": {"length": 4}}})
    manifest = {"schema": "task21e-saved-inputs-v1", "selected_cusp_count": 9, "separate_other_count": 6,
                "curve_count": 7, "records": records, "curves": curves, "authority": authority, "source_map": source_pin}
    binding = write(scientific / "inputs.json", manifest)
    return manifest, source_map, binding, source


class Controls(unittest.TestCase):
    def test_manifest_and_record_coverage(self):
        manifest, source_map, _, _ = self.fixture
        selected = M.admit_manifest(manifest, source_map)
        self.assertEqual(len(selected), 9)
        for spec in manifest["records"]:
            M.admit_record(json.loads(Path(spec["file"]["path"]).read_text()), spec)

    def test_missing_duplicate_source_scale_and_args_refusal(self):
        manifest, source_map, _, _ = self.fixture
        for mutate in [lambda x: x["records"].pop(),
                       lambda x: x["records"][1].update(id=x["records"][0]["id"]),
                       lambda x: x["records"][0].update(curve_metadata_scale=1.0),
                       lambda x: x["records"][0].update(source_commit="unknown")]:
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.assertRaises((KeyError, ValueError)):
                M.admit_manifest(changed, source_map)
        spec = manifest["records"][0]
        raw = json.loads(Path(spec["file"]["path"]).read_text())
        raw["args"]["rounds"] = 4
        with self.assertRaises(ValueError):
            M.admit_record(raw, spec)

    def test_prepared_refusal_then_complete_cli_and_existing_output_refusal(self):
        _, _, inputs_pin, source = self.fixture
        output = ROOT / "synthetic-run"
        card = {"schema": M.SCHEMA, "admission": "prepared", "limits": M.LIMITS, "output": str(output),
                "source_sha256": {"analyze.py": pin(source)["sha256"]}, "inputs_manifest": inputs_pin}
        cardpath = ROOT / "synthetic-card.json"
        write(cardpath, card)
        command = [sys.executable, "-I", str(source), "--card", str(cardpath), "--out", str(output)]
        def run(label):
            result = subprocess.run(command, capture_output=True, timeout=10)
            (ROOT / (label + ".stdout")).write_bytes(result.stdout)
            (ROOT / (label + ".stderr")).write_bytes(result.stderr)
            return result
        rejected = run("prepared-refusal")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(output.exists())
        card["admission"] = "reviewed"
        write(cardpath, card)
        accepted = run("manufactured-complete")
        self.assertEqual(accepted.returncode, 0, accepted.stderr.decode())
        result = json.loads((output / "report.json").read_text())
        self.assertEqual((len(result["cusps"]), len(result["curves"]), len(result["other_dispositions"])), (9, 7, 6))
        self.assertTrue(result["computational_completed"])
        self.assertFalse(result["scientific_claims_accepted"])
        self.assertTrue(result["reversal_coverage"]["selection_exactly_covers_declared_curve_reversals"])
        before = pin(output / "report.json")
        rejected = run("existing-output-refusal")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(pin(output / "report.json"), before)


def main():
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority > 10:
        raise RuntimeError("exact nice10 required")
    os.nice(10 - priority)
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    ROOT.mkdir(exist_ok=False)
    started = time.perf_counter()
    Controls.fixture = fixture()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    (ROOT / "controls.log").write_text(stream.getvalue())
    report = {"scope": "manufactured scientific inputs; original source bytes hashed only, never imported",
              "tests_run": result.testsRun, "successful": result.wasSuccessful(), "failures": len(result.failures),
              "errors": len(result.errors), "elapsed_before_final_report_write_seconds": time.perf_counter() - started,
              "observed_parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "observed_child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "source_sha256": {p: pin(Path(p))["sha256"] for p in [__file__, "/tmp/task21e-analyze.py"]}}
    write(ROOT / "report.json", report)
    print(stream.getvalue(), end="")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
