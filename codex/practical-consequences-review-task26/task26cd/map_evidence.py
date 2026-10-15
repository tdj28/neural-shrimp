"""Independent saved-map coverage and finite-field ledger; no model or producer imports."""
import importlib.util
import math
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("own_statistics", Path(__file__).with_name("statistics.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
NAMES = ("log_g_net", "log_g_read", "entropy", "gap", "p_top", "log_r1_rel", "stable_rank", "rayleigh_change")


def number(value):
    return type(value) in (int, float)


def same_number(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def grid(args):
    t = np.linspace(*args["tgrid"][:2], int(args["tgrid"][2]))
    a = np.linspace(*args["alpha"][:2], int(args["alpha"][2]))
    b = np.linspace(*args["beta"][:2], int(args["beta"][2]))
    return np.asarray([(float(x), float(y), float(z)) for z in t for y in b for x in a], dtype="<f8")


def progress_records(records, cells, chunk, record):
    """The producer emits three within-chunk record markers then one completion marker."""
    expected = []
    for start in range(0, len(cells), chunk):
        expected.extend({"recorded": step, "of": record} for step in range(500, record + 1, 500))
        end = min(start + chunk, len(cells))
        expected.append({"done": end, "of": len(cells), "chaotic": sum(c["chaotic"] for c in cells[start:end])})
    S.require(len(records) == len(expected), "MECH progress marker count")
    for i, (actual, wanted) in enumerate(zip(records, expected)):
        S.require(set(actual) == set(wanted) | {"seconds"}, f"MECH progress fields {i}")
        S.require(all(actual[k] == v for k, v in wanted.items()), f"MECH progress completion {i}")
        S.require(number(actual["seconds"]) and math.isfinite(actual["seconds"]) and actual["seconds"] >= 0,
                  f"MECH progress time {i}")
    return {"markers": records, "complete": True,
            "time_scope": "recorded seconds reset within each chunk; done seconds are cumulative"}


def audit_map(data, arrays, expected_args, progress, emit):
    for key, value in expected_args.items():
        S.require(data["args"].get(key) == value, f"MECH argument {key}")
    args, cells = data["args"], data["cells"]
    positions = grid(args)
    S.require(len(cells) == len(positions), "MECH full grid cell count")
    S.require(set(arrays) == set(NAMES) | {"cells"}, "MECH NPZ members")
    S.require(arrays["cells"].dtype == np.dtype("<f8") and arrays["cells"].shape == positions.shape,
              "MECH NPZ coordinate shape/dtype")
    S.require(np.array_equal(arrays["cells"], positions), "MECH original grid versus NPZ coordinates")
    keep = min(args["record"], 1000)
    for name in NAMES:
        S.require(arrays[name].shape == (keep, len(cells)) and arrays[name].dtype == np.dtype("<f4"),
                  f"MECH NPZ shape/dtype {name}")
    diagnostics = {"classification_mismatch_indices": [], "max_rayleigh_mismatch_indices": [],
                   "nonfinite_summary_indices": [], "nonfinite_step_indices": [], "fraction_disposition_indices": []}
    identity_errors, maximum_differences = [], []
    for i, (cell, xyz) in enumerate(zip(cells, positions)):
        S.require([cell["alpha"], cell["beta"], cell["log10_temperature"]] == xyz.tolist(), f"MECH cell grid {i}")
        S.require(type(cell["period"]) is int and 0 <= cell["period"] <= 64 and type(cell["chaotic"]) is bool,
                  f"MECH classification schema {i}")
        S.require(isinstance(cell["lyapunov"], list) and len(cell["lyapunov"]) == len(args["eps"]) == 2 and
                  cell["eps"] == args["eps"], f"MECH twin coverage {i}")
        scalars = {**{f"mean_{name}": cell[f"mean_{name}"] for name in NAMES},
                   **{key: cell[key] for key in ("boundary_term", "frac_gap_below_1", "frac_gap_below_2", "max_rayleigh_change")},
                   **{f"lyapunov_{j}": value for j, value in enumerate(cell["lyapunov"])}}
        S.require(all(number(value) for value in scalars.values()), f"MECH numeric fields {i}")
        classified = cell["period"] == 0 and all(value > 1e-3 for value in cell["lyapunov"])
        classification_matches = classified == cell["chaotic"]
        if not classification_matches:
            diagnostics["classification_mismatch_indices"].append(i)
        nonfinite = [name for name, value in scalars.items() if not math.isfinite(value)]
        if nonfinite:
            diagnostics["nonfinite_summary_indices"].append(i)
        fractions_valid = 0 <= cell["frac_gap_below_1"] <= cell["frac_gap_below_2"] <= 1
        if not fractions_valid:
            diagnostics["fraction_disposition_indices"].append(i)
        prefix = {}
        for name in NAMES:
            v = arrays[name][:, i].astype(np.float64)
            r = S.finite_summary(v)
            with np.errstate(invalid="ignore", over="ignore"):
                r["arithmetic_mean_including_nonfinite"] = float(v.mean())
                r["prefix_mean_minus_full_record_mean"] = r["arithmetic_mean_including_nonfinite"] - cell[f"mean_{name}"]
            prefix[name] = r
        if any(prefix[name]["finite_count"] != keep for name in NAMES):
            diagnostics["nonfinite_step_indices"].append(i)
        mx = float(arrays["rayleigh_change"][:, i].max())
        delta = mx - cell["max_rayleigh_change"]
        maximum_differences.append(delta)
        max_matches = same_number(mx, cell["max_rayleigh_change"])
        if not max_matches:
            diagnostics["max_rayleigh_mismatch_indices"].append(i)
        identity = cell["mean_log_g_net"] + cell["mean_log_g_read"] + cell["boundary_term"] - cell["lyapunov"][0]
        identity_errors.append(identity)
        emit({"index": i, "saved_cell": cell, "recomputed_source_chaotic_flag": classified,
              "classification_matches": classification_matches, "nonfinite_summary_fields": nonfinite,
              "fraction_fields_valid": fractions_valid, "signed_factor_identity_residual": identity,
              "prefix_steps": keep, "full_record_steps": args["record"], "prefix": prefix,
              "prefix_gap_below_1": float(np.mean(arrays["gap"][:, i] < 1.)),
              "prefix_gap_below_2": float(np.mean(arrays["gap"][:, i] < 2.)),
              "max_rayleigh_own_prefix": mx, "max_rayleigh_signed_difference": delta,
              "max_rayleigh_matches": max_matches})
    return {"cells": len(cells), "stored_steps_per_cell": keep, "full_record_steps_per_cell": args["record"],
            "all_args": args, "saved_bias_metadata": data["bias"], "original_seconds": data["seconds"],
            "progress": progress_records(progress, cells, args["chunk"], args["record"]),
            **diagnostics, "factor_identity_residual_summary": S.finite_summary(identity_errors),
            "max_rayleigh_difference_summary": S.finite_summary(maximum_differences),
            "full_record_scalar_ranges": {key: S.finite_summary([c[key] for c in cells])
                                          for key in [f"mean_{name}" for name in NAMES] +
                                          ["boundary_term", "frac_gap_below_1", "frac_gap_below_2", "max_rayleigh_change"]},
            "scope": "Complete saved cells and first 1000 float32 step fields; full 1500-step means are not reconstructed from this shorter prefix. All scientific mismatches remain recorded; groups use declared original saved flags."}
