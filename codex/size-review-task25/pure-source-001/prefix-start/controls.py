"""Manufactured controls and a fixed full-shape cost proposal. NOT executed yet.

No producer import, target path, random sampling or native model operation.
After separate admission: python -I -B controls.py --out FRESH_DIRECTORY.
"""
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
    spec = importlib.util.spec_from_file_location("task25_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ps = sibling("prefix")
cp = sibling("start")


def cell(t=0.0, entropy=0.0, le=(.1, .2), period=0, lead=0):
    return {"log10_temperature": t, "entropy_mean": entropy, "lyapunov": list(le), "period": period, "lead_period": lead}


def endpoint(c=2, m=.2, fine=1):
    return {"C1": c, "C2": fine, "M": m}


def prefix_fixture(index=0, offset=0.0):
    t = np.linspace(-1.5, 1.5, 601)
    gt = np.linspace(-1, 1, 81)
    h = np.clip(4*(t+1.5)+offset, 0, 12)
    read = -(gt-.025)**2 - .123456789
    net = -.7 + index*.01
    widths = np.gradient(t)
    above1, above10 = h > 1, h > 10
    crossing = float(t[np.argmax(above10)]-t[np.argmax(above1)]) if above1.any() and above10.any() else float("nan")
    one_step = net + read
    return {"prefix_index": index, "prefix_tokens": 20+index,
            "entropy_at_T1": float(h[np.abs(t).argmin()]),
            "breadth_decades": float(np.sum(widths*((h > 1) & (h < 10)))),
            "crossing_width": crossing, "log_g_net_median": net,
            "log_g_net_range": [net-.2, net+.3],
            "one_step_gain_max": float(one_step.max()),
            "one_step_gain_argmax_log10T": float(gt[int(np.argmax(one_step))]),
            "entropy_profile": h.round(5).tolist(), "log_g_read": read.round(5).tolist()}


def cp_document(offset=0):
    return {"args": {"directions": 8, "eps": 1e-5, "synthetic": True},
            "log10T": np.linspace(-1.5, 1.5, 601).tolist(), "gain_log10T": np.linspace(-1, 1, 81).tolist(),
            "prefixes": [prefix_fixture(i, offset+i*.01) for i in range(10)]}


def check_labels():
    assert ps.label(cell(period=4)) == "periodic"
    assert ps.label(cell(lead=7)) == "hidden chaos"
    assert ps.label(cell()) == "visible chaos"
    assert ps.label(cell(le=(.001, .1))).startswith("irregular")
    r = ps.stage([cell(le=(float("inf"), .1)), cell(period=-1)], 2)
    assert r["rows"][0]["literal_chaotic"] and not r["rows"][0]["scientifically_qualified"]
    assert not r["rows"][1]["period_fields_in_0_64"]


def check_selection():
    r = ps.transition([cell(0, 0), cell(.9, 2), cell(1, 3), cell(2, 4)])
    assert r["selected_increment"] == 0 and r["log10T_bounds"] == [0, 1]
    r = ps.transition([cell(0, 0), cell(1, 2), cell(2, 4), cell(3, 5)])
    assert r["equal_max_indices"] == [0, 1] and r["selected_increment"] == 0
    r = ps.transition([cell(0, 0), cell(1, float("nan")), cell(2, 4)])
    assert not r["finite_inputs"]


def check_ps_coverage():
    coarse = [cell(0, 0), cell(.1, 1, le=(.1, -.1)), cell(.2, 3, period=4)]
    fine = [cell(.05, 0, le=(.2, .15), lead=7), cell(.06, 0, le=(.0005, .002))]
    out = ps.endpoint(coarse, fine, {"original_exit": 1})
    e = out["literal_endpoint"]
    assert (e["C1"], e["C2"], e["M"]) == (1, 1, .15)
    assert not out["complete_finite_coverage"] and out["provenance"]["original_exit"] == 1
    assert ps.endpoint(None, fine)["literal_endpoint"] is None
    assert ps.endpoint(coarse, [])["literal_endpoint"] is None


def check_ps_frozen_rules():
    table = {m: {i: endpoint() for i in range(10)} for m in ps.MODELS}
    for i in range(2, 8):
        table["qwen14b"][i] = endpoint(1, .1)
    assert ps.score(table)["PS1"] == "holds"
    table["qwen14b"][9] = None
    s = ps.score(table)
    assert s["PS1"] == "undetermined" and s["PS2"] == "holds"
    table["qwen14b"][9] = endpoint(5, .2)
    assert ps.score(table)["PS1"] == "fails"
    table["qwen14b"][9] = endpoint(4, .2)  # +2 is not the forbidden >+2.
    assert ps.score(table)["PS1"] == "holds"
    table["qwen14b"][0] = endpoint(1000, 1000)
    assert ps.score(table)["PS1"] == "holds"  # Development cannot decide held-out score.


def check_ranks():
    s = cp.spearman([1, 1, 3], [1, 2, 3])
    assert s["rank_x"] == [1.5, 1.5, 3.0] and abs(s["rho"]-math.sqrt(.75)) < 2e-15
    s = cp.spearman([1, 2, 3, 4], [4, 3, 2, 1])
    assert s["rho"] == -1 and s["p"] == 0
    assert cp.spearman([1, 1], [1, 2])["rho"] is None
    assert cp.spearman([1, 2], [2, 1])["p"] is None
    assert cp.spearman([1, float("nan"), 3], [1, 2, 3])["nonfinite_indices"] == [1]


def check_cp_profiles():
    d = cp_document()
    p = cp.profile_compatibility(d["prefixes"][0], d["log10T"], d["gain_log10T"])
    assert p["source_grid_exact"] and p["breadth"]["compatible"] and p["crossing_width"]["compatible"]
    assert p["entropy_at_T1"]["compatible"] and p["one_step_max"]["compatible"]
    assert p["one_step_max"]["argmax_compatible"] and p["network_range"]["ordered_and_contains_median"]
    assert p["breadth"]["ambiguous_indices"]  # Exact threshold bins remain ambiguous after rounding.
    d["prefixes"][0]["one_step_gain_max"] += .01
    assert not cp.profile_compatibility(d["prefixes"][0], d["log10T"], d["gain_log10T"])["one_step_max"]["compatible"]


def check_cp_saturation_and_nonfinite():
    d = cp_document()
    row = d["prefixes"][0]
    row.update(entropy_profile=[.5]*601, entropy_at_T1=.5, breadth_decades=0., crossing_width=float("nan"))
    p = cp.profile_compatibility(row, d["log10T"], d["gain_log10T"])
    assert p["crossing_width"]["undefined_possible"] and p["crossing_width"]["compatible"]
    row["entropy_profile"][12] = float("nan")
    p = cp.profile_compatibility(row, d["log10T"], d["gain_log10T"])
    assert p["status"] == "unavailable" and p["entropy_rounding_cells"][12] is None


def check_cp_membership():
    key = cp.FROZEN[0][0]
    d = cp_document()
    result = cp.reduce_cp({key: d})
    assert result["models_present"] == 1 and len(result["models_missing"]) == 13
    assert result["score"]["CP1"]["scientific_disposition"] == "undefined"
    d["prefixes"][1]["prefix_index"] = 0
    result = cp.reduce_cp({key: d})
    assert result["score"] is not None and not result["rows"][0]["prefix_identity"]["unique_0_through_9"]
    d["prefixes"] = d["prefixes"][:-1]
    result = cp.reduce_cp({key: d})
    assert result["source_refusals"] == [key] and result["score"] is None


def check_frozen_vs_sensitivity():
    documents, current = {}, {}
    for spec in cp.FROZEN:
        d = cp_document()
        # Intentional scalar/profile mismatch exercises separate evidence roles.
        for p in d["prefixes"]:
            p["breadth_decades"] = spec[4]
        documents[spec[0]] = d
        current[(spec[1], spec[2])] = {"missing": False, "chaos_decades": -spec[4]}
    result = cp.reduce_cp(documents, current)
    assert result["score"]["CP1"]["rho"] == 1
    assert result["score"]["sensitivity_current_decades"]["breadth_p0"]["rho"] == -1
    assert result["score"]["CP1"]["holds"]
    assert any(not p["breadth"]["compatible"] for r in result["rows"] for p in r["profiles"])


def full_shape():
    stages = {}
    for model_index, model in enumerate(ps.MODELS):
        stages[model] = {}
        for i in range(10):
            coarse = [cell(float(t), float(4*t+2), le=(.01, .012) if (j+i+model_index)%3 else (-.01, -.012)) for j, t in enumerate(np.linspace(-.4, .9, 48))]
            bounds = ps.transition(coarse)["log10T_bounds"]
            fine = [cell(float(t), float(4*t+2), le=(.02, .023), period=4 if j%4 == 0 else 0) for j, t in enumerate(np.linspace(*bounds, 24))]
            stages[model][i] = {"coarse": coarse, "fine": fine, "provenance": {"synthetic": True, "original_exit": 0}}
    documents = {s[0]: cp_document(i*.02) for i, s in enumerate(cp.FROZEN)}
    return {"ps": ps.reduce_ps(stages), "cp": cp.reduce_cp(documents)}


def clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"nonfinite": "nan" if math.isnan(obj) else ("+inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)  # Ownership acquired before any failure writes.
    started = time.monotonic()
    controls = [check_labels, check_selection, check_ps_coverage, check_ps_frozen_rules, check_ranks,
                check_cp_profiles, check_cp_saturation_and_nonfinite, check_cp_membership, check_frozen_vs_sensitivity]
    try:
        passed = []
        for control in controls:
            control()
            passed.append(control.__name__)
        result = full_shape()
        target = args.out / "manufactured-results.json"
        target.write_text(json.dumps(clean(result), separators=(",", ":"), allow_nan=False)+"\n")
        report = {"schema": "task25-ps-cp-manufactured-v1", "passed": len(passed), "failed": 0, "controls": passed,
                  "shape": {"ps_models": 2, "ps_prefixes_per_model": 10, "ps_coarse_cells": 48, "ps_fine_cells": 24,
                            "cp_models": 14, "cp_prefixes_per_model": 10, "cp_entropy_points": 601, "cp_gain_points": 81},
                  "elapsed_through_scientific_serialization": time.monotonic()-started,
                  "ru_maxrss_KiB_before_final_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "scientific_bytes": target.stat().st_size, "scientific_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "scope": "Manufactured finite scalar/profile cost only; target input I/O/hash/decompression and actual numeric strings are not measured."}
        (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    except BaseException as e:
        (args.out / "failure.json").write_text(json.dumps({"type": type(e).__name__, "message": str(e), "elapsed": time.monotonic()-started})+"\n")
        raise


if __name__ == "__main__":
    main()
