"""Manufactured PT/QF/depth controls, prepared only. No target I/O or producer import."""
import argparse
import hashlib
import importlib.util
import json
import math
import resource
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def sibling(name):
    spec = importlib.util.spec_from_file_location("task25_"+name, HERE/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pt, dp = sibling("plateau"), sibling("depth")


def cell(role="PTa", i=0):
    eps = list(pt.EXPECTED[role][1])
    return {"log10_temperature": float(i), "alpha": 0., "eps": eps,
            "lyapunov": [.02+.001*j for j in range(len(eps))], "period": 0, "lead_period": 0,
            "lyapunov_tangent": .025,
            "fd_probe": {"h": list(pt.FD_STEPS), "central_ratio": [2.]*len(pt.FD_STEPS), "repeat_diff": 0.}}


def check_pt_confirmation():
    a = cell()
    s = {"log10T": a["log10_temperature"], "alpha": 0., "map_label": "chaotic", "map_exponent": .02, "map_period": 0}
    r = pt.cell("PTa", a, s)
    assert r["frozen_offset_confirmation"] and r["selection_coordinates_match"]
    a["period"] = 4
    r = pt.cell("PTa", a, s)
    assert r["frozen_offset_confirmation"] and r["literal_label"] == "periodic" and not r["no_detected_period"]
    a["lyapunov"][0] = .001
    assert not pt.cell("PTa", a)["frozen_offset_confirmation"]
    a["lyapunov"][0] = float("inf")
    r = pt.cell("PTa", a)
    assert r["literal_all_above_1e3"] and r["frozen_offset_confirmation"] is None


def check_boundary_and_negative():
    b = pt.boundary(.1, .1)
    assert b["equal"] and b["inclusive"] and not b["strict"]
    a = cell("PTc")
    a["lyapunov"] = [-.2, -.19, -.18]
    r = pt.cell("PTc", a)
    assert r["negative_on_plateau"]["strict"] and not r["positive_at_all_offsets"]
    a["lyapunov"][0] = 0
    assert not pt.cell("PTc", a)["all_negative"]
    assert pt.boundary(float("nan"), .1)["inclusive"] is None


def check_qf_probe():
    a = cell("QFa")
    r = pt.cell("QFa", a)
    assert r["chaos_qualification"]["strict"] and r["chaos_qualification"]["inclusive"]
    a["fd_probe"]["central_ratio"][a["fd_probe"]["h"].index(1e-7)] = 1.
    r = pt.cell("QFa", a)
    assert not r["chaos_qualification"]["inclusive"]
    a["lyapunov"] = [-.1]*4
    r = pt.cell("QFa", a)
    assert r["positive_tangent_all_negative_twins"] and not r["chaos_qualification"]["inclusive"]
    a["fd_probe"]["central_ratio"][a["fd_probe"]["h"].index(1e-4)] = 0.
    assert pt.cell("QFa", a)["chaos_qualification"]["inclusive"] is None


def check_qf_missing_ladder():
    a = cell("QFb_p2")
    a["fd_probe"]["h"][-1] = 1e-4
    r = pt.cell("QFb_p2", a)
    assert r["probe"]["status"] == "undefined" and r["chaos_qualification"]["strict"] is None
    del a["fd_probe"]
    assert pt.cell("QFb_p2", a)["chaos_qualification"]["strict"] is None


def check_coverage_and_pairing():
    r = pt.reduce_plateau("PTa", {"cells": [cell()]}, provenance={"status": 1})
    assert not r["coverage_matches"] and r["provenance"]["status"] == 1
    assert pt.reduce_plateau("PTa", None)["missing"]
    a, b, c = cell("PTc", 0), cell("B7b_history", 0), cell("B7b_history", 1)
    b["lyapunov"][0] = .04
    r = pt.compare_exact_coordinates([a, cell("PTc", 2)], [b, c], 1e-5)
    assert r["rows"][0]["status"] == "matched" and r["rows"][1]["status"] == "unmatched"
    assert pt.compare_exact_coordinates([a], [b, b], 1e-5)["rows"][0]["status"] == "ambiguous"


def depth_fixture(layer=20):
    n = 32 if layer == 50 else 24
    le = np.full((n, n), -.1)
    p = np.zeros((n, n), dtype=np.int64)
    le[0, 0], le[1, 1], p[1, 1] = .1, .2, 4
    arrays = {"log10_temperature": np.linspace(-.3, .3, n), "alpha": np.linspace(0, .12, n),
              "lyapunov": le, "period": p, "entropy": np.ones((n, n))}
    args = {"chaos_level": .001, "max_period": 32, "layer": layer, "resolution": [n, n], "vectors": f"synthetic-layer-{layer}"}
    return arrays, args


def check_depth_semantics():
    a, args = depth_fixture()
    r = dp.reduce_depth(20, a, args, provenance={"original_tool_status": 1, "failure_stage": "post-save figure"})
    assert r["counts"]["writer_lyapunov_positive"] == 2
    assert r["counts"]["qualified_no_return_positive"] == 1
    assert r["qualified_chaotic_temperature_range"]["minimum"] == -.3
    a["lyapunov"][2, 2] = float("inf")
    r = dp.reduce_depth(20, a, args)
    assert r["counts"]["writer_lyapunov_positive"] == 3 and r["counts"]["qualified_no_return_positive"] == 1
    assert r["counts"]["nonfinite_lyapunov"] == 1
    a["period"] = a["period"].astype(float)
    a["period"][3, 3] = float("nan")
    r = dp.reduce_depth(20, a, args)
    assert r["raw_period_value_histogram"]["nan"] == 1 and r["counts"]["invalid_period"] == 1
    assert sum(r["raw_period_value_histogram"].values()) == 24*24


def check_depth_partial_shape():
    a, args = depth_fixture()
    a["done"] = 100
    assert "source_refusal" in dp.reduce_depth(20, a, args)
    del a["done"]
    a["lyapunov"] = a["lyapunov"].flatten()
    assert "source_refusal" in dp.reduce_depth(20, a, args)


def check_depth_comparison():
    rows = [dp.reduce_depth(layer, *depth_fixture(layer)) for layer in (20, 35, 50, 65)]
    r = dp.compare_layers(rows)
    assert all(x["status"] == "descriptive_comparison" for x in r["comparisons"])
    assert r["comparisons"][0]["grid_shapes"] == [[24, 24], [32, 32]]
    assert dp.compare_layers(rows[:2])["comparisons"][0]["status"] == "unavailable"


def full_shape():
    plateau = [pt.reduce_plateau(role, {"cells": [cell(role, i) for i in range(n)]}, provenance={"synthetic": True}) for role, (n, _) in pt.EXPECTED.items()]
    depths = [dp.reduce_depth(layer, *depth_fixture(layer), provenance={"synthetic": True}) for layer in (20, 35, 50, 65)]
    return {"plateau": plateau, "depth": depths, "depth_comparison": dp.compare_layers(depths)}


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return {"nonfinite": "nan" if math.isnan(v) else "+inf" if v > 0 else "-inf"}
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        tests = [check_pt_confirmation, check_boundary_and_negative, check_qf_probe, check_qf_missing_ladder,
                 check_coverage_and_pairing, check_depth_semantics, check_depth_partial_shape, check_depth_comparison]
        for test in tests:
            test()
        path = args.out/"manufactured-results.json"
        path.write_text(json.dumps(clean(full_shape()), separators=(",", ":"), allow_nan=False)+"\n")
        report = {"schema": "task25-plateau-depth-manufactured-v1", "passed": len(tests), "failed": 0,
                  "controls": [x.__name__ for x in tests], "elapsed_through_science_serialization": time.monotonic()-start,
                  "ru_maxrss_KiB_before_final_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "scientific_bytes": path.stat().st_size, "scientific_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "scope": "Manufactured shapes only, including separate32x32 baseline. Target decode/hash cost and full worker peak are unmeasured."}
        (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    except BaseException as e:
        (args.out/"failure.json").write_text(json.dumps({"type": type(e).__name__, "message": str(e)})+"\n")
        raise


if __name__ == "__main__":
    main()
