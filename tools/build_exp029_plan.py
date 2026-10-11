"""Build the result-free EXP-029 machine plan (Track B stage B1).

Two autonomous maps of the da Silva and Rech network: a global survey of the
(alpha, gamma) plane and a high-resolution map of the EXP-004 focus rectangle
containing the reproduced shrimp. The plan reads no outcome; rerunning it
before the freeze reproduces the same substantive plan.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments" / "track-b" / "EXP-029-plan.json"
SOURCE_FILES = [
    "docs/EXP-029-protocol.md",
    "pyproject.toml",
    "uv.lock",
    "src/neural_shrimp/__init__.py",
    "src/neural_shrimp/controls.py",
    "src/neural_shrimp/models.py",
    "src/neural_shrimp/autonomous.py",
    "src/neural_shrimp/capacity.py",
    "tools/runner_common.py",
    "tools/build_exp029_plan.py",
    "tools/validate_exp029_plan.py",
    "tools/run_exp029.py",
    "tools/analyze_exp029.py",
    "tools/audit_exp029.py",
    "tests/test_autonomous.py",
    "tests/test_run_exp029.py",
]
STARTS = {"A": [0.1, 0.1, 0.1], "B": [0.2, -0.1, 0.3]}


def file_record(relative):
    data = (ROOT / relative).read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build():
    return {
        "experiment": "EXP-029",
        "status": "frozen_outcome_masked_plan",
        "label": "prospective exploratory descriptive survey; private Git freeze; not preregistered",
        "date": "2026-10-06",
        "program": "Program C, Track B, stage B1",
        "claim_boundary": (
            "Screened autonomous dynamical classes, periods, finite-time exponents and Floquet "
            "multipliers of the stated network on the stated grids, starts, horizons and tolerances. "
            "No capacity, computation, shrimp-organization or chaos-theorem claim."),
        "network": "dx/dt = -x + W tanh(x); W = [[1.5, 2.9, alpha], [-1.5, 1.18, 0], [gamma, -22, 0.47]]",
        "grid_convention": "rows index gamma (increasing), columns alpha (increasing); "
                           "cell = row * columns + col; axes are inclusive linspace",
        "maps": [
            {"map_id": "global", "alpha": [0.0, 1.5], "gamma": [5.0, 16.0],
             "resolution": [256, 256], "starts": STARTS, "start_strides": {"A": 1, "B": 1}},
            {"map_id": "focus", "alpha": [1.056, 1.080], "gamma": [10.14, 10.55],
             "resolution": [768, 768], "starts": STARTS, "start_strides": {"A": 1, "B": 4}},
        ],
        "integration": {"method": "classical RK4, float64, CPU", "dt": 0.01, "settle": 2000.0,
                        "measure": 1000.0, "max_events": 160, "max_period": 32, "repeats": 4,
                        "atol": 2e-6, "rtol": 2e-5, "ftle_align": 100.0, "tangent_seed": 0,
                        "divergence_threshold": 1e6, "equilibrium_tolerance": 1e-8},
        "labels": {"chaos_threshold_per_time": 0.005,
                   "rule": "diverged; equilibrium (no x3 maxima, field norm < tolerance); periodic "
                           "(screened period >= 1 peaks per cycle); chaotic candidate (no recurrence, "
                           "not equilibrium, FTLE > threshold); unresolved"},
        "floquet": {"start": "last recorded x3-maximum event state", "duration": "screened cycle time",
                    "dt": 0.01, "transverse": "largest modulus after removing the multiplier closest to 1"},
        "audit": {"cells_per_label_per_job": 8, "audit_seed": 2907, "solver": "scipy DOP853",
                  "rtol": 1e-10, "atol": 1e-12, "multiplier_tolerance": 1e-4},
        "resources": {"workers": 14, "blas_threads": 1, "chunk_cells": 4096,
                      "wall_ceiling_seconds": 21600, "max_load_average": 16.0,
                      "min_available_gib": 16.0, "gpu": "not used", "paid_compute": "none"},
        "output": {"raw_root": "../neural-shrimp-data/raw/EXP-029", "attempt": "attempt-001"},
        "result_placeholders": [],
        "source_files": [file_record(path) for path in SOURCE_FILES],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if PLAN_PATH.read_text() != text:
            raise SystemExit("saved EXP-029 plan differs from a fresh build")
        print("EXP-029 plan matches a fresh build")
        return
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(text)
    print(f"wrote {PLAN_PATH.relative_to(ROOT)} sha256={hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
