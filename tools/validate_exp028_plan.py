"""Independent validator for the EXP-028 machine plan.

The checks below do not import the plan builder or the batched engine. The S1
data recipe and both forward passes are rewritten here, and the initial
Gauss-Newton maxima are recomputed from finite-difference prediction
Jacobians rather than the analytic routine used by the builder. Expected
constants are typed from the protocol. Nothing reads a target outcome.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.models import TanhRegressor, make_teacher_regression  # noqa: E402

EXPECTED_MAPS = [("S1", 28001), ("S1", 28002), ("S2", 28011), ("S2", 28012)]
EXPECTED = {
    "iteration": {"steps": 20000, "tail_steps": 384, "max_period": 64, "min_repeats": 6,
                  "atol": 1e-8, "rtol": 1e-6, "ftle_steps": 4000, "ftle_align": 1000,
                  "tangent_seed": 0, "chaos_threshold": 5e-3, "divergence_threshold": 1e8,
                  "sd_horizons": [100, 500, 1000], "precision": "float64", "device": "cpu"},
    "zoom": {"window": 32, "stride": 16, "resolution": 256, "map_index_offset": 4},
    "certification": {"min_component_size": 4, "max_components": 16, "max_fixed_components": 4,
                      "atol": 1e-9, "rtol": 1e-8, "root_tolerance": 1e-10,
                      "stability_margin": 1e-7, "candidate_tolerance": 1e-3},
    "chaos_diagnostics": {"count": 16, "seed_base": 2810, "steps": 30000, "ftle_steps": 10000,
                          "ftle_align": 1000, "tangent_seed": 1, "tail_steps": 384,
                          "chaos_threshold": 5e-3},
    "gate": {"min_period": 3, "requires_attracting": True, "requires_borders_chaos": True},
    "audit": {"resimulated_cells_per_coarse_map": 32, "audit_seed": 2806},
    "resources": {"workers": 12, "blas_threads": 1, "chunk_cells": 1024,
                  "wall_ceiling_seconds": 21600, "max_load_average": 18.0,
                  "min_available_gib": 16.0},
    "output": {"raw_root": "../neural-shrimp-data/raw/EXP-028", "attempt": "attempt-001"},
}


def digest(array):
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(repr(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def s1_recipe(seed, width=8):
    streams = np.random.SeedSequence(int(seed)).spawn(3)
    n = width * width + width
    x = np.random.default_rng(streams[0]).normal(size=(n, width))
    y = np.random.default_rng(streams[1]).normal(size=n)
    theta = np.random.default_rng(streams[2]).normal(size=n)
    return x, y, theta


def s1_predict(theta, x, width=8):
    w0 = theta[:width * width].reshape(width, width)
    w1 = theta[width * width:]
    return np.tanh(math.sqrt(2.0 / width) * (x @ w0)) @ w1 / width


def s2_predict(theta, x):
    w0 = theta[:6].reshape(2, 3)
    b0 = theta[6:9]
    w1 = theta[9:12]
    return np.tanh(x @ w0 + b0) @ w1 + theta[12]


def fd_gauss_newton_max(predict, theta, x, hidden_count, loss_scale, ridge, step=1e-6):
    n = len(x)
    jac = np.empty((n, len(theta)))
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        jac[:, i] = (predict(plus, x) - predict(minus, x)) / (2 * step)
    k = 2.0 * loss_scale / n
    values = []
    for block in (slice(0, hidden_count), slice(hidden_count, len(theta))):
        part = jac[:, block]
        values.append(float(np.linalg.eigvalsh(k * part.T @ part)[-1]) + ridge)
    return values


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def validate(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    failures = []
    check(plan.get("experiment") == "EXP-028", "experiment id", failures)
    check(plan.get("status") == "frozen_outcome_masked_plan", "status label", failures)
    check(plan.get("result_placeholders") == [], "result placeholders must be empty", failures)
    for section, expected in EXPECTED.items():
        actual = plan.get(section, {})
        for key, value in expected.items():
            check(actual.get(key) == value, f"{section}.{key}={actual.get(key)!r} != {value!r}",
                  failures)
    it = plan["iteration"]
    check(it["tail_steps"] >= it["max_period"] * it["min_repeats"],
          "tail too short for the largest screened period", failures)
    check(it["steps"] >= it["ftle_steps"] + it["ftle_align"], "FTLE window exceeds steps", failures)
    maps = plan.get("maps", [])
    check([(m["system"], m["seed"]) for m in maps] == EXPECTED_MAPS, "map list", failures)
    total_cells = 0
    for index, spec in enumerate(maps):
        check(spec["map_index"] == index, f"map_index {index}", failures)
        if spec["system"] == "S1":
            x, y, theta = s1_recipe(spec["seed"])
            gn = fd_gauss_newton_max(s1_predict, theta, x, 64, 1.0, 0.0)
            expected_net = {"d": 8, "width": 8, "activation": "tanh", "bias": False,
                            "loss_scale": 1.0, "ridge": 0.0}
            check(abs(spec["network"]["in_scale"] - math.sqrt(0.25)) < 1e-15, "S1 in_scale", failures)
            check(abs(spec["network"]["out_scale"] - 0.125) < 1e-15, "S1 out_scale", failures)
        else:
            data = make_teacher_regression(seed=spec["seed"], d=2, teacher_width=3, n_train=24,
                                           n_validation=48, n_test=96)
            x, y = data.train_x, data.train_y
            theta = TanhRegressor(2, 3).initialize(seed=spec["seed"] + 100000, scale=0.2)
            gn = fd_gauss_newton_max(s2_predict, theta, x, 9, 0.5, 1e-3)
            expected_net = {"d": 2, "width": 3, "activation": "tanh", "bias": True,
                            "loss_scale": 0.5, "ridge": 1e-3}
        for key, value in expected_net.items():
            check(spec["network"][key] == value, f"{spec['map_id']} network {key}", failures)
        check(spec["n_parameters"] == len(theta), f"{spec['map_id']} parameter count", failures)
        check(spec["data_sha256"] == {"x": digest(x), "y": digest(y), "theta0": digest(theta)},
              f"{spec['map_id']} data hashes", failures)
        for name, value in zip(("hidden", "output"), gn):
            saved = spec["gauss_newton_max"][name]
            check(abs(saved - value) <= 1e-6 * abs(value), f"{spec['map_id']} GN {name}", failures)
            center = spec[f"log10_{name}"]["center"]
            check(abs(center - math.log10(2.0 / saved)) < 1e-12, f"{spec['map_id']} center {name}",
                  failures)
            check(spec[f"log10_{name}"]["half_width"] == 1.5 and
                  spec[f"log10_{name}"]["resolution"] == 256, f"{spec['map_id']} axis {name}", failures)
        total_cells += spec["log10_hidden"]["resolution"] * spec["log10_output"]["resolution"]
    check(total_cells == 4 * 256 * 256, "coarse cell count", failures)
    for record in plan.get("source_files", []):
        data = (ROOT / record["path"]).read_bytes()
        check(hashlib.sha256(data).hexdigest() == record["sha256"] and len(data) == record["bytes"],
              f"source hash {record['path']}", failures)
    required = {"src/neural_shrimp/gdmaps.py", "src/neural_shrimp/trainability.py",
                "tools/run_exp028.py", "tools/analyze_exp028.py", "tools/audit_exp028.py",
                "docs/EXP-028-protocol.md"}
    check(required <= {r["path"] for r in plan.get("source_files", [])}, "bound sources", failures)
    return failures, total_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/trainability/EXP-028-plan.json"))
    args = parser.parse_args()
    failures, cells = validate(args.plan)
    plan_bytes = Path(args.plan).read_bytes()
    report = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "coarse_cells": cells,
              "failures": failures, "passed": not failures}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
