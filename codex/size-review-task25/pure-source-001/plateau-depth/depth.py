"""Descriptive L70D saved-grid accounting. Pure array reduction, no native map."""
import math
import numpy as np


def extrema(values):
    a = np.asarray(values, float)
    v = a[np.isfinite(a)]
    if not len(v):
        return {"count": 0, "minimum": None, "maximum": None}
    lo, hi = float(v.min()), float(v.max())
    return {"count": len(v), "minimum": lo, "maximum": hi, "minimum_hex": lo.hex(), "maximum_hex": hi.hex(),
            "outward_binary_interval": [math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)]}


def reduce_depth(layer, arrays, args, provenance=None, summary=None):
    if layer not in (20, 35, 50, 65):
        raise ValueError("unassigned depth")
    out = {"layer": layer, "provenance": provenance, "args": args, "missing": arrays is None,
           "source_summary": summary, "expected_resolution": [32, 32] if layer == 50 else [24, 24],
           "scope": "Descriptive finite grid at its actual layer/vector/resolution; no universal size, disappearance or continued-boundary inference."}
    if arrays is None:
        return out
    if "done" in arrays:
        out["source_refusal"] = "partial checkpoint supplied where a separately authenticated final grid is required"
        return out
    t, a = np.asarray(arrays["log10_temperature"]), np.asarray(arrays["alpha"])
    le, p = np.asarray(arrays["lyapunov"]), np.asarray(arrays["period"])
    h = np.asarray(arrays["entropy"])
    expected = tuple(out["expected_resolution"])
    if t.shape != (expected[0],) or a.shape != (expected[1],) or le.shape != expected or p.shape != expected or h.shape != expected:
        out["source_refusal"] = "shape mismatch; no reshaping/interpolation or borrowed cells"
        out["actual_shapes"] = [list(x.shape) for x in (t, a, le, p, h)]
        return out
    threshold = args["chaos_level"]
    if not math.isfinite(threshold) or threshold != .001:
        out["source_refusal"] = "unbound original chaos threshold"
        return out
    axes_ok = bool(np.isfinite(t).all() and np.isfinite(a).all() and np.all(np.diff(t) > 0) and np.all(np.diff(a) > 0))
    period_valid = np.isfinite(p) & (p == np.floor(p)) & (p >= 0) & (p <= args["max_period"])
    valid = np.isfinite(le) & period_valid
    literal_positive = le > threshold  # This is the actual writer's saved count.
    literal_no_return_positive = literal_positive & (p == 0)
    qualified = valid & literal_no_return_positive
    raw_period_values, raw_period_counts = np.unique(p, return_counts=True)
    TT, AA = np.meshgrid(t, a, indexing="ij")
    ledger = []
    for i, j in np.ndindex(expected):
        ledger.append({"i": i, "j": j, "log10T": float(t[i]), "alpha": float(a[j]),
                       "lyapunov": float(le[i,j]), "period": float(p[i,j]), "entropy": float(h[i,j]),
                       "literal_positive": bool(literal_positive[i,j]), "literal_no_return_positive": bool(literal_no_return_positive[i,j]),
                       "finite_valid_lyapunov_period": bool(valid[i,j]), "finite_entropy": bool(np.isfinite(h[i,j])),
                       "qualified_no_return_positive": bool(qualified[i,j])})
    n = le.size
    counts = {"total": n, "writer_lyapunov_positive": int(literal_positive.sum()),
              "literal_no_return_positive": int(literal_no_return_positive.sum()),
              "finite_valid_denominator": int(valid.sum()), "qualified_no_return_positive": int(qualified.sum()),
              "nonfinite_lyapunov": int((~np.isfinite(le)).sum()), "invalid_period": int((~period_valid).sum()),
              "nonfinite_entropy": int((~np.isfinite(h)).sum()), "periodic_valid": int((valid & (p > 0)).sum()),
              "unresolved_nonpositive_valid": int((valid & (p == 0) & ~(le > threshold)).sum())}
    out.update(axes={"log10T": t.tolist(), "alpha": a.tolist(), "finite_strictly_increasing": axes_ok}, rows=ledger, counts=counts,
               shares={"writer_positive_over_all": counts["writer_lyapunov_positive"]/n,
                       "literal_no_return_positive_over_all": counts["literal_no_return_positive"]/n,
                       "qualified_over_all": counts["qualified_no_return_positive"]/n,
                       "qualified_over_finite_valid": counts["qualified_no_return_positive"]/counts["finite_valid_denominator"] if counts["finite_valid_denominator"] else None},
               finite_lyapunov_extrema=extrema(le), finite_entropy_extrema=extrema(h),
               literal_positive_temperature_range=extrema(TT[literal_positive]),
               qualified_chaotic_temperature_range=extrema(TT[qualified]), qualified_chaotic_alpha_range=extrema(AA[qualified]),
               raw_period_value_histogram={str(float(q)): int(count) for q, count in zip(raw_period_values, raw_period_counts)},
               valid_period_histogram={str(int(q)): int((p[period_valid] == q).sum()) for q in np.unique(p[period_valid])})
    if summary is not None:
        out["writer_comparison"] = {
            "cells": {"own": n, "saved": summary.get("cells"), "equal": n == summary.get("cells")},
            "lyapunov_positive": {"own": counts["writer_lyapunov_positive"], "saved": summary.get("lyapunov_positive"), "equal": counts["writer_lyapunov_positive"] == summary.get("lyapunov_positive")},
            "period_counts": {"own": {str(int(q)): int((p == q).sum()) for q in np.unique(p)} if np.all(period_valid) else None, "saved": summary.get("period_counts"), "source_integer_period_domain": bool(np.all(period_valid))},
            "lyapunov_range": {"own": [float(le.min()), float(le.max())], "saved": summary.get("lyapunov_range")}}
    out["historical_70b_stricter_scorer"] = "unavailable; LE>.001/P0 is an explicitly named own descriptive qualification, not the writer's positive count"
    out["per_layer_vector_scope"] = "Vectors were separately constructed for each layer; equal alpha does not establish identical state perturbations."
    return out


def compare_layers(results):
    """No regridding or coordinate matching between24x24 and32x32."""
    by_layer = {r["layer"]: r for r in results}
    if len(by_layer) != len(results):
        raise ValueError("duplicate layer results")
    base = by_layer.get(50)
    comparisons = []
    for layer in (20, 35, 65):
        r = by_layer.get(layer)
        out = {"layer": layer, "baseline": 50, "status": "unavailable"}
        if r and base and "shares" in r and "shares" in base:
            out["status"] = "descriptive_comparison"
            out["share_differences"] = {k: {"layer": r["shares"][k], "baseline": base["shares"][k],
                                            "signed_layer_minus_baseline": r["shares"][k]-base["shares"][k] if r["shares"][k] is not None and base["shares"][k] is not None else None} for k in r["shares"]}
            out["grid_shapes"] = [r["expected_resolution"], base["expected_resolution"]]
        comparisons.append(out)
    return {"comparisons": comparisons, "scope": "All fixed layers retained, separate resolutions/vectors. No uncertainty model, new significance test or geometry classification."}
