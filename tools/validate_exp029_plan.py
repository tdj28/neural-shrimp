"""Independent validator for the EXP-029 machine plan.

Expected values are typed from the protocol; counts are recomputed here
rather than copied from the builder. Nothing reads an outcome.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INTEGRATION = {"dt": 0.01, "settle": 2000.0, "measure": 1000.0, "max_events": 160,
                        "max_period": 32, "repeats": 4, "atol": 2e-6, "rtol": 2e-5,
                        "ftle_align": 100.0, "tangent_seed": 0, "divergence_threshold": 1e6,
                        "equilibrium_tolerance": 1e-8}
EXPECTED_MAPS = [
    ("global", [0.0, 1.5], [5.0, 16.0], [256, 256], {"A": 1, "B": 1}),
    ("focus", [1.056, 1.080], [10.14, 10.55], [768, 768], {"A": 1, "B": 4}),
]
STARTS = {"A": [0.1, 0.1, 0.1], "B": [0.2, -0.1, 0.3]}


def validate(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)

    check(plan.get("experiment") == "EXP-029", "experiment id")
    check(plan.get("status") == "frozen_outcome_masked_plan", "status")
    check(plan.get("result_placeholders") == [], "result placeholders")
    for key, value in EXPECTED_INTEGRATION.items():
        check(plan["integration"].get(key) == value, f"integration.{key}")
    it = plan["integration"]
    check(it["max_events"] >= it["max_period"] * it["repeats"] + 1, "event buffer too small")
    check(it["measure"] > it["ftle_align"], "FTLE window")
    check(plan["labels"]["chaos_threshold_per_time"] == 0.005, "chaos threshold")
    check(plan["audit"] == {"cells_per_label_per_job": 8, "audit_seed": 2907, "solver": "scipy DOP853",
                            "rtol": 1e-10, "atol": 1e-12, "multiplier_tolerance": 1e-4}, "audit block")
    res = plan["resources"]
    check((res["workers"], res["chunk_cells"], res["wall_ceiling_seconds"], res["max_load_average"],
           res["min_available_gib"]) == (14, 4096, 21600, 16.0, 16.0), "resources")
    check(plan["output"] == {"raw_root": "../neural-shrimp-data/raw/EXP-029", "attempt": "attempt-001"},
          "output")
    total = 0
    check(len(plan["maps"]) == len(EXPECTED_MAPS), "map count")
    for spec, (map_id, alpha, gamma, resolution, strides) in zip(plan["maps"], EXPECTED_MAPS):
        check(spec["map_id"] == map_id and spec["alpha"] == alpha and spec["gamma"] == gamma
              and spec["resolution"] == resolution and spec["start_strides"] == strides
              and spec["starts"] == STARTS, f"map {map_id}")
        rows, cols = resolution
        for start, stride in strides.items():
            total += len(range(0, rows, stride)) * len(range(0, cols, stride))
    check(total == 256 * 256 * 2 + 768 * 768 + 192 * 192, "planned cell count")
    for record in plan["source_files"]:
        data = (ROOT / record["path"]).read_bytes()
        check(hashlib.sha256(data).hexdigest() == record["sha256"] and len(data) == record["bytes"],
              f"source hash {record['path']}")
    required = {"src/neural_shrimp/autonomous.py", "src/neural_shrimp/capacity.py",
                "tools/runner_common.py", "tools/run_exp029.py", "tools/analyze_exp029.py",
                "tools/audit_exp029.py", "docs/EXP-029-protocol.md", "uv.lock"}
    check(required <= {r["path"] for r in plan["source_files"]}, "bound sources")
    return failures, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/track-b/EXP-029-plan.json"))
    args = parser.parse_args()
    failures, total = validate(args.plan)
    print(json.dumps({"plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
                      "planned_cells": total, "failures": failures, "passed": not failures}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
