"""Pure Task8 result comparison; no data access, producer imports or computation.

Field map is the frozen items_cmd/generation_cmd output schema in analysis
SHA a38586151f9acf6e60c8d355df22d44d8b8ca7aec78270a6f6e73cfc6b838dc2.
All fields except producer `args` are compared or explicitly classified below.
The caller authenticates inputs, frozen invocation and reducer source separately;
this module does not turn matching outputs into evidence of input provenance.

NaN matches NaN only where a statistic can be undefined. None is NOT a NaN
alias. Excluded-spectrum residuals additionally retain exact None/NaN/+inf/-inf
semantics. All other numerical fields must be finite. Every nonzero numerical
delta is retained even when within the fixed absolute tolerance (no relative
term). Exact discrete types prevent True == 1 and 1.0 == 1 aliases.
"""
from __future__ import annotations

from copy import deepcopy
import math

ABS_TOLERANCE = 1e-12
A_ARMS = ("1.0", "1.05", "1.08", "1.1")
B_ARMS = ("1.0", "1.05", "1.15", "1.20")
A_STATUSES = ("missing", "no fixed point", "empty or nonfinite spectrum",
              "no token-id record", "token ids differ from the scored gold row", "ok")
A_DIRECT = ("requested", "cohort", "minimum_cohort", "setup_failed", "dispositions",
            "primary", "answer_changes_descriptive", "share_detected_unstable_by_arm")
B_DIRECT = ("n", "metrics", "B1_decline", "B3_noninferior", "primary_success",
            "B2_descriptive", "cap_rate", "tokens_median", "strict_format_share")
A_EXTRA = ("cohort_indices", "all_cohort_cells", "bootstrap_statistics")
B_EXTRA = ("all_rows_retained",)
_MISSING = object()


def _class(value):
    if value is None:
        return "null"
    if type(value) in (int, float):
        if type(value) is float and math.isnan(value):
            return "nan"
        if value == math.inf:
            return "+infinity"
        if value == -math.inf:
            return "-infinity"
        return "finite"
    return type(value).__name__


def _valid(value, kind):
    if kind == "args":
        return type(value) is dict
    if kind == "count":
        return type(value) is int and value >= 0
    if kind == "bool":
        return type(value) is bool
    if kind == "string":
        return type(value) is str and bool(value)
    if kind == "status":
        return value in A_STATUSES if type(value) is str else False
    if kind == "number":
        return type(value) in (int, float) and _class(value) == "finite"
    if kind == "statistic":
        return type(value) in (int, float) and _class(value) in ("finite", "nan")
    if kind == "residual":
        return value is None or type(value) in (int, float)
    if kind == "arnoldi":
        return (value is None or type(value) is bool or
                (type(value) is dict and all(type(k) is str and type(v) is bool
                                            for k, v in value.items())))
    raise AssertionError(kind)


def _error(errors, side, path, reason, value=_MISSING):
    row = {"side": side, "path": path, "reason": reason}
    if value is not _MISSING:
        row["value"] = deepcopy(value)
    errors.append(row)


def _walk(value, schema, side, path, errors, leaves):
    """Validate complete explicit schema and retain independently valid leaves."""
    if value is _MISSING:
        _error(errors, side, path, "missing field")
    elif type(schema) is dict:
        if type(value) is not dict:
            _error(errors, side, path, "expected object", value)
            return
        for key in value:
            if key not in schema:
                _error(errors, side, path + "/" + str(key), "unexpected field", value[key])
        for key, child in schema.items():
            _walk(value.get(key, _MISSING), child, side, path + "/" + key, errors, leaves)
    elif type(schema) is list:
        if type(value) is not list:
            _error(errors, side, path, "expected array", value)
            return
        if len(value) != len(schema):
            _error(errors, side, path, "array length differs", value)
        for i, child in enumerate(schema):
            _walk(value[i] if i < len(value) else _MISSING, child, side,
                  path + "/" + str(i), errors, leaves)
    elif not _valid(value, schema):
        _error(errors, side, path, "invalid " + schema, value)
    else:
        leaves[path] = (schema, value)


def _exact_tree(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(_exact_tree(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(_exact_tree(x, y) for x, y in zip(a, b))
    return a == b


def _check(path, kind, a, b):
    row = {"path": path, "kind": kind, "producer": deepcopy(a), "independent": deepcopy(b)}
    if kind in ("number", "statistic", "residual"):
        left, right = _class(a), _class(b)
        row.update(producer_class=left, independent_class=right)
        if left == right == "finite":
            delta = b - a
            row["independent_minus_producer"] = delta
            row["absolute_difference"] = abs(delta)
            status = "exact" if delta == 0 else "within_tolerance" if abs(delta) <= ABS_TOLERANCE else "mismatch"
        else:
            status = "undefined_match" if left == right else "mismatch"
    else:
        status = "exact" if _exact_tree(a, b) else "mismatch"
    row["status"] = status
    row["passed"] = status != "mismatch"
    return row


def _base(part, producer, independent, p_schema, i_schema, direct, extra):
    errors, p_leaves, i_leaves = [], {}, {}
    _walk(producer, p_schema, "producer", "", errors, p_leaves)
    _walk(independent, i_schema, "independent", "", errors, i_leaves)
    checks = [_check(path, kind, value, i_leaves[path][1])
              for path, (kind, value) in p_leaves.items()
              if path in i_leaves and path != "/args"]
    return {"schema": "huginn-independent-comparison-v1", "part": part,
            "absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": 0,
            "scope": "Result-field comparison only; source/input authority is external.",
            "field_map": {"direct_top_level": list(direct),
                          "producer_only": {"args": "retained; separately admitted invocation, not a numerical result"},
                          "independent_only": {k: "retained diagnostic absent from frozen producer output" for k in extra}},
            "producer_only": {"args": deepcopy(producer.get("args"))} if type(producer) is dict else {},
            "independent_only": {k: deepcopy(independent[k]) for k in extra if k in independent}
                                if type(independent) is dict else {},
            "checks": checks, "schema_errors": errors, "representation_differences": []}


def _finish(report):
    report["checked_leaf_count"] = len(report["checks"])
    report["differences"] = [deepcopy(r) for r in report["checks"] if r["status"] not in ("exact", "undefined_match")]
    report["differences"].extend(deepcopy(report["representation_differences"]))
    report["comparison_passed"] = not report["schema_errors"] and all(r["passed"] for r in report["checks"])
    report["exact_result_match"] = report["comparison_passed"] and not report["differences"]
    return report


def _count_arg(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")


def compare_a(producer: dict, independent: dict, *, expected_count=64, expected_boot=2000):
    """Compare all source-defined A fields; overrides support tiny synthetic tests.

    Producer stops before appending its first undefined per-arm correlation.
    Only the canonical finite prefix is ordinary output in that case. Every
    omitted independent arm is retained as a representation difference, and
    the first omission must be independently undefined. No other omission is
    exempt. Producer cohort indices are derived solely from all dispositions.
    """
    _count_arg(expected_count, "expected_count")
    _count_arg(expected_boot, "expected_boot")
    disposition = {"status": "status", "residual_F": "residual", "arnoldi_converged": "arnoldi", "eigenvalues": "count"}
    common = {"requested": "count", "cohort": "count", "minimum_cohort": "count", "setup_failed": "bool",
              "dispositions": {str(q): disposition for q in range(expected_count)},
              "primary": {"statistic": "statistic", "per_arm": {a: "statistic" for a in A_ARMS},
                          "ci95": ["statistic", "statistic"], "undefined_draws": "count", "success": "bool"},
              "answer_changes_descriptive": {k: "count" for k in (
                  "detected_unstable_changed", "detected_unstable_total", "no_unstable_mode_detected_changed", "no_unstable_mode_detected_total")},
              "share_detected_unstable_by_arm": {a: "statistic" for a in A_ARMS}}
    p_schema, i_schema = deepcopy(common), deepcopy(common)
    # args remains an opaque object whose invocation is authenticated by caller.
    p_schema["args"] = "args"
    p_schema["primary"]["per_arm"] = {a: "number" for a in A_ARMS}
    p_arms = producer.get("primary", {}).get("per_arm") if type(producer) is dict and type(producer.get("primary")) is dict else None
    keys = tuple(a for a in A_ARMS if type(p_arms) is dict and a in p_arms)
    is_prefix = type(p_arms) is dict and set(p_arms) == set(A_ARMS[:len(p_arms)])
    omitted = A_ARMS[len(p_arms):] if is_prefix else ()
    if omitted:
        p_schema["primary"]["per_arm"] = {a: "number" for a in keys}
    cohort = independent.get("cohort_indices", []) if type(independent) is dict else []
    cohort_len = len(cohort) if type(cohort) is list else 0
    i_schema.update(cohort_indices=["count"] * cohort_len,
        all_cohort_cells=[{"question": "count", "lambda": "number", "rho": "number", "hash": "string",
                           "residual": "number", "changes": "bool"}] * (4 * cohort_len),
        bootstrap_statistics=["statistic"] * (expected_boot if cohort_len else 0))
    report = _base("A", producer, independent, p_schema, i_schema, A_DIRECT, A_EXTRA)
    if omitted:
        p_stat = producer["primary"].get("statistic")
        i_primary = independent.get("primary", {}) if type(independent) is dict else {}
        i_arms = i_primary.get("per_arm", {}) if type(i_primary) is dict else {}
        first = i_arms.get(omitted[0]) if type(i_arms) is dict else None
        if _class(p_stat) != "nan" or _class(first) != "nan":
            _error(report["schema_errors"], "producer", "/primary/per_arm", "short prefix requires undefined statistic and independently undefined first omitted arm", p_arms)
        for arm in omitted:
            report["representation_differences"].append({"path": "/primary/per_arm/" + arm,
                "status": "producer_source_short_circuit", "producer_present": False,
                "independent": deepcopy(i_arms.get(arm)) if type(i_arms) is dict else None,
                "first_undefined_arm": arm == omitted[0], "passed": True})
    if not is_prefix:
        _error(report["schema_errors"], "producer", "/primary/per_arm", "expected canonical arm prefix", p_arms)
    # Nonfinite residuals are a recorded exclusion, never an admitted fixed point.
    for side, result in (("producer", producer), ("independent", independent)):
        dispositions = result.get("dispositions") if type(result) is dict else None
        if type(dispositions) is not dict:
            continue
        for q, entry in dispositions.items():
            if type(entry) is not dict or entry.get("status") not in A_STATUSES:
                continue
            status, residual = entry["status"], entry.get("residual_F", _MISSING)
            if status == "missing":
                valid = (residual is None and entry.get("arnoldi_converged") is None
                         and type(entry.get("eigenvalues")) is int and entry["eigenvalues"] == 0)
            elif status == "no fixed point":
                valid = residual is None or (type(residual) in (int, float)
                    and (_class(residual) != "finite" or residual > 1e-4))
            else:
                valid = _valid(residual, "number") and 0 <= residual <= 1e-4
            if not valid:
                _error(report["schema_errors"], side, "/dispositions/" + str(q),
                       "residual metadata inconsistent with source disposition", entry)
    if type(producer) is dict:
        report["checks"].append(_check("/requested_vs_frozen_count", "count", producer.get("requested"), expected_count))
        dispositions = producer.get("dispositions")
        if type(dispositions) is dict:
            derived = [q for q in range(expected_count) if type(dispositions.get(str(q))) is dict and dispositions[str(q)].get("status") == "ok"]
            report["field_map"]["derived"] = {"cohort_indices": "ordered questions with producer disposition status ok"}
            report["checks"].append(_check("/cohort_indices_from_dispositions", "exact", derived, cohort))
            report["checks"].append(_check("/cohort_vs_dispositions", "count", producer.get("cohort"), len(derived)))
    if type(independent) is dict and type(cohort) is list:
        cells = independent.get("all_cohort_cells")
        expected_cells = [(q, float(a)) for q in cohort for a in A_ARMS]
        actual = [(r.get("question"), r.get("lambda")) if type(r) is dict else None for r in cells] if type(cells) is list else None
        if actual != expected_cells:
            _error(report["schema_errors"], "independent", "/all_cohort_cells", "diagnostic cells not in canonical cohort/arm order", actual)
    return _finish(report)


def compare_b(producer: dict, independent: dict, *, expected_count=200):
    """All flexible/strict, primary/descriptive and formatting fields; no dropping."""
    _count_arg(expected_count, "expected_count")
    pair = {"difference": "number", "ci95": ["number", "number"], "lower_one_sided_95": "number",
            "mcnemar_less_p": "number", "lost": "count", "gained": "count"}
    metric = {"accuracy": {a: "number" for a in B_ARMS}, **{a: pair for a in B_ARMS[1:]}}
    contrast = {"arm": "string", "difference": "number", "lost": "count", "gained": "count", "p": "number"}
    common = {"n": "count", "metrics": {"correct": metric, "correct_strict": metric},
              "B1_decline": {**contrast, "pass": "bool"},
              "B3_noninferior": {"arm": "string", "difference": "number", "lower_one_sided_95": "number", "margin": "number", "pass": "bool"},
              "primary_success": "bool", "B2_descriptive": contrast,
              **{key: {a: "number" for a in B_ARMS} for key in ("cap_rate", "tokens_median", "strict_format_share")}}
    p_schema, i_schema = deepcopy(common), deepcopy(common)
    p_schema["args"] = "args"
    i_schema["all_rows_retained"] = [{"arm": "string", "index": "count", "correct": "bool",
                                    "correct_strict": "bool", "new_tokens": "count"}] * (expected_count * 4)
    report = _base("B", producer, independent, p_schema, i_schema, B_DIRECT, B_EXTRA)
    if type(producer) is dict:
        report["checks"].append(_check("/n_vs_frozen_count", "count", producer.get("n"), expected_count))
    if type(independent) is dict:
        rows = independent.get("all_rows_retained")
        actual = [(r.get("arm"), r.get("index")) if type(r) is dict else None for r in rows] if type(rows) is list else None
        if actual != [(a, q) for a in B_ARMS for q in range(expected_count)]:
            _error(report["schema_errors"], "independent", "/all_rows_retained", "diagnostic rows not in canonical arm/index order", actual)
    return _finish(report)
