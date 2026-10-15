"""Independent saved PT/QF scalars. No file/model/producer operations."""
import math

EXPECTED = {
    "PTa": (24, [1e-4, 1e-5, 1e-6]), "PTb": (6, [1e-4, 1e-5, 1e-6]),
    "PTc": (16, [1e-4, 3e-5, 1e-5]),
    "B7a_history": (64, [1e-6, 1e-9]), "B7b_history": (64, [1e-5, 1e-6, 1e-7]),
    "QFa": (29, [1e-4, 1e-5, 1e-6, 1e-7]),
    **{f"QFb_p{i}": (12, [1e-5, 1e-6, 1e-7]) for i in range(2, 6)}}
FD_STEPS = [1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def literal_label(r):
    if r["period"] > 0:
        return "periodic"
    if all(v > .001 for v in r["lyapunov"]):
        return "hidden chaos" if r["lead_period"] >= 1 else "visible chaos"
    return "irregular, exponent not positive at both offsets"


def boundary(value, threshold):
    return {"value": value, "threshold": threshold,
            "strict": value < threshold if finite(value) else None,
            "inclusive": value <= threshold if finite(value) else None,
            "equal": value == threshold if finite(value) else None,
            "historical_boundary_implementation": "unavailable; both comparisons retained"}


def probe(row):
    p = row.get("fd_probe")
    if not isinstance(p, dict):
        return {"status": "unavailable", "reason": "missing fd_probe", "offsets": []}
    hs, ratios = p.get("h", []), p.get("central_ratio", [])
    refs = [i for i, h in enumerate(hs) if h == 1e-4]
    result = {"raw": p, "status": "computed", "source_ladder_exact": hs == FD_STEPS,
              "shape_matches": len(hs) == len(ratios), "reference_indices": refs, "offsets": []}
    if len(hs) != len(ratios) or len(refs) != 1:
        result.update(status="undefined", reason="probe lengths or unique reference missing")
        return result
    ref = ratios[refs[0]]
    result["reference_ratio"] = ref
    result["all_probe_rows"] = [{"ordinal": i, "h": h, "ratio": r, "finite_nonnegative": finite(r) and r >= 0} for i, (h, r) in enumerate(zip(hs, ratios))]
    for eps in row["eps"]:
        found = [i for i, h in enumerate(hs) if h == eps]
        deviation = None
        if len(found) == 1 and finite(ref) and ref > 0 and finite(ratios[found[0]]) and ratios[found[0]] >= 0:
            deviation = abs(ratios[found[0]] / ref - 1)
        result["offsets"].append({"eps": eps, "matching_probe_indices": found,
                                  "ratio": ratios[found[0]] if len(found) == 1 else None,
                                  "relative_deviation_from_1e_minus_4": deviation,
                                  "alternative_abs_difference_over_reference": abs(ratios[found[0]]-ref)/abs(ref) if deviation is not None else None,
                                  "resolution": boundary(deviation, .1)})
    return result


def cell(role, raw, original_selection=None):
    out = {"raw": raw, "selection": original_selection, "source_refusal": None}
    try:
        expected = EXPECTED[role][1]
        eps, values = raw["eps"], raw["lyapunov"]
        values_ok = len(values) == len(eps) == len(expected) and all(finite(v) for v in values)
        p = raw["period"]
        valid_p = finite(p) and int(p) == p and 0 <= p <= 64
        label = literal_label(raw)
        out.update(offsets_exact=eps == expected, finite_expected_values=values_ok,
                   valid_period=valid_p, literal_label=label,
                   saved_class=raw.get("class"), saved_class_matches=None if "class" not in raw else raw["class"] == label,
                   literal_all_above_1e3=all(v > .001 for v in values),
                   all_above_1e3=all(v > .001 for v in values) if values_ok else None,
                   all_negative=all(v < 0 for v in values) if values_ok else None,
                   spread=max(values)-min(values) if values_ok else None,
                   no_detected_period=p == 0 if valid_p else None)
        if original_selection is not None:
            out["selection_coordinates_match"] = (raw["log10_temperature"] == original_selection["log10T"] and raw["alpha"] == original_selection["alpha"])
            out["original_map_label"] = original_selection.get("map_label")
        if role in ("PTa", "PTb"):
            out["frozen_offset_confirmation"] = out["all_above_1e3"] if eps == expected else None
            out["scope"] = "Original PT confirmation rule is all three rates>.001; period and finite qualification are separately retained. Original map labels are not recomputed."
        elif role == "PTc":
            out["plateau"] = boundary(out["spread"], .1)
            out["negative_on_plateau"] = {mode: (out["plateau"][mode] and out["all_negative"]) if values_ok and eps == expected else None for mode in ("strict", "inclusive")}
            out["positive_at_all_offsets"] = out["all_above_1e3"] if eps == expected else None
        elif role.startswith("QF"):
            pr = probe(raw)
            tangent = raw.get("lyapunov_tangent")
            out.update(probe=pr, tangent=tangent, tangent_finite=finite(tangent),
                       tangent_positive=finite(tangent) and tangent > .001,
                       tangent_minus_twins=[tangent-v if finite(tangent) and finite(v) else None for v in values],
                       positive_tangent_any_negative_twin=finite(tangent) and tangent > .001 and any(finite(v) and v < 0 for v in values),
                       positive_tangent_all_negative_twins=finite(tangent) and tangent > .001 and out["all_negative"] is True)
            out["chaos_qualification"] = {}
            for mode in ("strict", "inclusive"):
                resolved = [r["resolution"][mode] for r in pr["offsets"]]
                ready = values_ok and eps == expected and valid_p and finite(tangent) and len(resolved) == len(eps) and all(x is not None for x in resolved)
                out["chaos_qualification"][mode] = bool(p == 0 and tangent > .001 and all(v > .001 for v in values) and all(resolved)) if ready else None
            out["scope"] = "Saved native tangent and finite twins/probe only. No new derivative precision or asymptotic-chaos proof. Unresolved offsets are never dropped."
        else:
            out["scope"] = "Original B7 history retained separately; no later precision qualification is applied retroactively."
    except (KeyError, ValueError, TypeError) as e:
        out["source_refusal"] = str(e)
    return out


def reduce_plateau(role, document, selection=None, provenance=None):
    if role not in EXPECTED:
        raise ValueError("unassigned plateau role")
    expected_n, expected_eps = EXPECTED[role]
    out = {"role": role, "provenance": provenance, "expected_cells": expected_n,
           "expected_eps": expected_eps, "missing": document is None, "rows": []}
    if document is None:
        return out
    rows = document["cells"]
    out["args"] = document.get("args")
    out["actual_cells"] = len(rows)
    out["coverage_matches"] = len(rows) == expected_n
    if selection is not None:
        out["selection_count_matches"] = len(selection) == len(rows)
    out["rows"] = [{"ordinal": i, **cell(role, row, selection[i] if selection is not None and i < len(selection) else None)} for i, row in enumerate(rows)]
    out["all_finite_offsets"] = all(r.get("finite_expected_values", False) and r.get("offsets_exact", False) for r in out["rows"]) and len(rows) == expected_n
    out["source_refusal_indices"] = [r["ordinal"] for r in out["rows"] if r["source_refusal"]]
    if role == "PTc":
        out["negative_plateau_counts"] = {}
        for mode in ("strict", "inclusive"):
            flags = [r.get("negative_on_plateau", {}).get(mode) for r in out["rows"]]
            count = sum(f is True for f in flags)
            out["negative_plateau_counts"][mode] = {"count": count, "unknown": sum(f is None for f in flags),
                "denominator": expected_n, "strict_majority_of_fixed_temperatures": count > expected_n/2 if len(rows) == expected_n and all(f is not None for f in flags) else None}
    if role.startswith("QF"):
        out["qualified_counts"] = {mode: {"true": sum(r.get("chaos_qualification", {}).get(mode) is True for r in out["rows"]),
                                          "false": sum(r.get("chaos_qualification", {}).get(mode) is False for r in out["rows"]),
                                          "undefined": sum(r.get("chaos_qualification", {}).get(mode) is None for r in out["rows"])} for mode in ("strict", "inclusive")}
    return out


def compare_exact_coordinates(left, right, eps):
    """No tolerance/nearest-row pairing between the16-point and64-point lines."""
    def key(r):
        return r["log10_temperature"], r["alpha"]
    ledger = []
    for i, a in enumerate(left):
        matches = [j for j, b in enumerate(right) if key(a) == key(b)]
        result = {"left_ordinal": i, "coordinate": list(key(a)), "right_ordinals": matches,
                  "status": "unmatched" if not matches else "ambiguous" if len(matches) > 1 else "matched"}
        if len(matches) == 1:
            b = right[matches[0]]
            ai, bi = [k for k, e in enumerate(a["eps"]) if e == eps], [k for k, e in enumerate(b["eps"]) if e == eps]
            result.update(left_offset_indices=ai, right_offset_indices=bi)
            if len(ai) == len(bi) == 1:
                av, bv = a["lyapunov"][ai[0]], b["lyapunov"][bi[0]]
                result.update(left_value=av, right_value=bv, signed_right_minus_left=bv-av if finite(av) and finite(bv) else None)
        ledger.append(result)
    return {"eps": eps, "rows": ledger, "scope": "Exact stored-coordinate/offset comparisons only; unmatched grid points are retained, not interpolated."}
