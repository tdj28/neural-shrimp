"""Independent PS scalar reduction. Pure functions; no producer imports or file reads."""
import math
import numpy as np

MODELS = ("qwen7b", "qwen14b")
DEVELOPMENT = (0, 1)
HELD_OUT = tuple(range(2, 10))


def label(row):
    positive = all(v > 1e-3 for v in row["lyapunov"])
    if row["period"] > 0:
        return "periodic"
    if positive:
        return "hidden chaos" if row["lead_period"] >= 1 else "visible chaos"
    return "irregular, exponent not positive at both offsets"


def finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def transition(rows):
    """Literal first argmax of raw adjacent entropy increments, not slopes."""
    if rows is None or len(rows) < 2:
        return {"status": "unavailable", "reason": "missing or fewer than two rows"}
    try:
        t = np.asarray([r["log10_temperature"] for r in rows], dtype=float)
        h = np.asarray([r["entropy_mean"] for r in rows], dtype=float)
        increments = np.diff(h)
        selected = int(np.argmax(increments))
        lo, hi = max(selected - 1, 0), min(selected + 2, len(t) - 1)
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "source_refusal", "reason": str(e)}
    return {"status": "computed", "increments": increments.tolist(),
            "selected_increment": selected, "equal_max_indices": np.flatnonzero(increments == increments[selected]).tolist(),
            "selected_bounds_indices": [lo, hi], "log10T_bounds": [float(t[lo]), float(t[hi])],
            "finite_inputs": bool(np.isfinite(t).all() and np.isfinite(h).all()),
            "strictly_increasing_axis": bool(np.all(np.diff(t) > 0)),
            "source_argmax_including_nonfinite": True}


def stage(rows, expected):
    if rows is None:
        return {"status": "missing", "expected_cells": expected, "rows": None}
    ledger = []
    for i, row in enumerate(rows):
        try:
            labs = label(row)
            le = row["lyapunov"]
            p, lead = row["period"], row["lead_period"]
            finite = len(le) == 2 and all(finite_number(x) for x in le)
            valid_periods = all(finite_number(x) and int(x) == x and 0 <= x <= 64 for x in (p, lead))
            coords = finite_number(row["log10_temperature"])
            item = {"ordinal": i, "raw": row, "literal_label": labs,
                    "literal_chaotic": labs in ("hidden chaos", "visible chaos"),
                    "finite_two_offset_exponents": finite, "period_fields_in_0_64": valid_periods,
                    "finite_coordinate": coords, "scientifically_qualified": finite and valid_periods and coords,
                    "minimum_exponent_literal": min(le), "source_refusal": None}
        except (KeyError, TypeError, ValueError) as e:
            item = {"ordinal": i, "raw": row, "source_refusal": str(e), "scientifically_qualified": False}
        ledger.append(item)
    return {"status": "present", "expected_cells": expected, "actual_cells": len(rows),
            "expected_coverage": len(rows) == expected, "rows": ledger,
            "qualified_count": sum(r["scientifically_qualified"] for r in ledger),
            "source_refusal_count": sum(r["source_refusal"] is not None for r in ledger)}


def endpoint(coarse, fine, provenance=None):
    a, b = stage(coarse, 48), stage(fine, 24)
    out = {"stages": {"coarse": a, "fine": b}, "provenance": provenance,
           "transition": transition(coarse), "literal_endpoint": None, "source_refusal": None,
           "complete_finite_coverage": False}
    if coarse is None or fine is None:
        out["source_refusal"] = "missing stage; source endpoint is null"
        return out
    if not fine or not coarse or a["source_refusal_count"] or b["source_refusal_count"]:
        out["source_refusal"] = "empty stage or a row cannot be interpreted by the source"
        return out
    both = a["rows"] + b["rows"]
    out["literal_endpoint"] = {
        "C1": sum(r["literal_chaotic"] for r in a["rows"]),
        "C2": sum(r["literal_chaotic"] for r in b["rows"]),
        "M": max(r["minimum_exponent_literal"] for r in both),
        "stage1_cells": len(coarse), "stage2_cells": len(fine),
        "stage2_log10T": [float(fine[0]["log10_temperature"]), float(fine[-1]["log10_temperature"])],
        "chaotic_log10T": [round(float(r["raw"]["log10_temperature"]), 4) for r in both if r["literal_chaotic"]]}
    out["complete_finite_coverage"] = all(s["expected_coverage"] and s["qualified_count"] == s["actual_cells"] for s in (a, b))
    tr = out["transition"]
    if tr["status"] == "computed":
        expected = np.linspace(*tr["log10T_bounds"], 24)
        actual = [r["log10_temperature"] for r in fine]
        out["adaptive_grid_association"] = {
            "expected_from_literal_selection": expected.tolist(), "actual": actual,
            "exact_same_length_and_coordinates": len(actual) == 24 and all(x == y for x, y in zip(actual, expected)),
            "signed_coordinate_differences": [float(x-y) for x, y in zip(actual, expected)],
            "scope": "Fixed native source grid association, no interpolation or tolerance-based substitution."}
    return out


def score(table):
    """Source endpoint table; finite qualification is separate and never changes literal labels."""
    pairs = [(i, table.get(MODELS[0], {}).get(i), table.get(MODELS[1], {}).get(i)) for i in HELD_OUT]
    known = [(i, a, b) for i, a, b in pairs if a is not None and b is not None]
    missing = len(HELD_OUT) - len(known)
    s1 = sum(b["C1"] < a["C1"] for _, a, b in known)
    violations = [i for i, a, b in known if b["C1"] > a["C1"] + 2]
    ps1 = "fails" if violations or s1 + missing < 6 else ("holds" if s1 >= 6 and not missing else "undetermined")
    s2 = sum(b["M"] < a["M"] for _, a, b in known)
    ps2 = "holds" if s2 >= 6 else ("fails" if s2 + missing < 6 else "undetermined")
    return {"PS1": ps1, "PS1_successes": s1, "PS1_violations": violations,
            "PS2": ps2, "PS2_successes": s2, "held_out_missing": missing,
            "missing_pair_indices": [i for i, a, b in pairs if a is None or b is None],
            "PS3_prefixes_with_stage2_chaos": {m: sum(table.get(m, {}).get(i) is not None and table[m][i]["C2"] > 0 for i in HELD_OUT) for m in MODELS}}


def reduce_ps(stages):
    """stages[model][index] -> coarse/fine lists or null, with optional provenance."""
    rows = {}
    for m in MODELS:
        rows[m] = {}
        for i in range(10):
            record = stages.get(m, {}).get(i) or {}
            rows[m][i] = endpoint(record.get("coarse"), record.get("fine"), record.get("provenance"))
    table = {m: {i: r["literal_endpoint"] for i, r in rows[m].items()} for m in MODELS}
    return {"development": list(DEVELOPMENT), "held_out": list(HELD_OUT), "rows": rows,
            "literal_score": score(table),
            "unqualified_held_out": [{"model": m, "prefix": i} for m in MODELS for i in HELD_OUT if not rows[m][i]["complete_finite_coverage"]],
            "scope": "No native labels re-estimated. Literal source scores are retained even when finite/coverage qualification is absent; no new pass gate."}
