"""Build the result-free EXP-028 machine plan.

The plan binds systems, seeds, data hashes, initial-curvature rate grids,
iteration and screening settings, the frozen zoom, certification and chaos
diagnostic rules, resources, output location and source hashes. It reads no
target outcome. Rerunning before the freeze reproduces the same substantive
plan; source hashes change only when bound sources change.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.trainability import SYSTEMS, array_sha256, edge_rates  # noqa: E402

PLAN_PATH = ROOT / "experiments" / "trainability" / "EXP-028-plan.json"
SOURCE_FILES = [
    "docs/EXP-028-protocol.md",
    "src/neural_shrimp/gdmaps.py",
    "src/neural_shrimp/trainability.py",
    "src/neural_shrimp/diagnostics.py",
    "src/neural_shrimp/models.py",
    "tools/build_exp028_plan.py",
    "tools/validate_exp028_plan.py",
    "tools/run_exp028.py",
    "tools/analyze_exp028.py",
    "tools/audit_exp028.py",
    "tests/test_gdmaps.py",
    "tests/test_trainability.py",
    "tests/test_run_exp028.py",
]
MAPS = [("S1", 28001), ("S1", 28002), ("S2", 28011), ("S2", 28012)]
HALF_WIDTH = 1.5
RESOLUTION = 256


def file_record(relative):
    path = ROOT / relative
    data = path.read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build():
    maps = []
    for index, (system, seed) in enumerate(MAPS):
        net, x, y, theta0 = SYSTEMS[system](seed)
        hidden_rate, output_rate = edge_rates(net, theta0, x, y)
        lam_h, lam_o = net.gauss_newton_layer_max(theta0, x, y)
        maps.append({
            "map_index": index, "map_id": f"{system}-{seed}", "system": system,
            "seed": seed, "n_parameters": net.n_parameters, "n_examples": int(len(x)),
            "network": {"d": net.d, "width": net.width, "activation": net.activation,
                        "in_scale": net.in_scale, "out_scale": net.out_scale,
                        "bias": net.bias, "loss_scale": net.loss_scale, "ridge": net.ridge},
            "data_sha256": {"x": array_sha256(x), "y": array_sha256(y),
                            "theta0": array_sha256(theta0)},
            "gauss_newton_max": {"hidden": lam_h, "output": lam_o},
            "edge_rate": {"hidden": hidden_rate, "output": output_rate},
            "log10_hidden": {"center": math.log10(hidden_rate),
                             "half_width": HALF_WIDTH, "resolution": RESOLUTION},
            "log10_output": {"center": math.log10(output_rate),
                             "half_width": HALF_WIDTH, "resolution": RESOLUTION},
        })
    plan = {
        "experiment": "EXP-028",
        "status": "frozen_outcome_masked_plan",
        "label": "prospective exploratory pilot; private Git freeze; not preregistered",
        "date": "2026-10-06",
        "program": "Program C, Track A, stage A1",
        "claim_boundary": (
            "Existence of verified attracting primitive cycles of full-batch gradient-descent "
            "maps in the stated learning-rate planes, networks, data, horizons and tolerances. "
            "No shrimp organization, training benefit, stochastic-training or large-model claim."),
        "grid_convention": "rows index hidden-rate log10 values, columns output-rate log10 values; "
                           "cell = row * columns + col; rates are 10**log10",
        "maps": maps,
        "iteration": {"steps": 20000, "tail_steps": 384, "max_period": 64, "min_repeats": 6,
                      "atol": 1e-8, "rtol": 1e-6, "ftle_steps": 4000, "ftle_align": 1000,
                      "tangent_seed": 0, "chaos_threshold": 5e-3,
                      "divergence_threshold": 1e8, "sd_horizons": [100, 500, 1000],
                      "precision": "float64", "device": "cpu"},
        "zoom": {"window": 32, "stride": 16, "resolution": 256, "map_index_offset": 4,
                 "interest": "periodic candidate with period >= 2, or chaotic candidate"},
        "certification": {"min_component_size": 4, "max_components": 16,
                          "max_fixed_components": 4, "atol": 1e-9, "rtol": 1e-8,
                          "root_tolerance": 1e-10, "stability_margin": 1e-7,
                          "candidate_tolerance": 1e-3, "max_evaluations": 4000,
                          "jacobian": "exact I - P H"},
        "chaos_diagnostics": {"count": 16, "seed_base": 2810, "steps": 30000,
                              "ftle_steps": 10000, "ftle_align": 1000, "tangent_seed": 1,
                              "tail_steps": 384, "chaos_threshold": 5e-3},
        "gate": {"min_period": 3, "requires_attracting": True, "requires_borders_chaos": True},
        "audit": {"resimulated_cells_per_coarse_map": 32, "audit_seed": 2806},
        "resources": {"workers": 12, "blas_threads": 1, "chunk_cells": 1024,
                      "wall_ceiling_seconds": 21600, "max_load_average": 18.0,
                      "min_available_gib": 16.0, "gpu": "not used", "paid_compute": "none"},
        "output": {"raw_root": "../neural-shrimp-data/raw/EXP-028", "attempt": "attempt-001"},
        "result_placeholders": [],
        "source_files": [file_record(path) for path in SOURCE_FILES],
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the saved plan equals a fresh build instead of writing")
    args = parser.parse_args()
    plan = build()
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.check:
        saved = PLAN_PATH.read_text()
        if saved != text:
            raise SystemExit("saved EXP-028 plan differs from a fresh build")
        print("EXP-028 plan matches a fresh build")
        return
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(text)
    print(f"wrote {PLAN_PATH.relative_to(ROOT)} sha256={hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
