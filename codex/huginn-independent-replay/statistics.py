"""Independent saved-raw reducers for the frozen Huginn contract at 9805ed32.

No producer module, model, dataset loader or network call is imported. These
are floating statistical reproductions, not interval certificates. Admission
of immutable source-bound files and producer completion belongs to the driver.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

FROZEN_COMMIT = "9805ed32055e7ef790df0cc67316357c1067aef0"
A_ARMS = (1.0, 1.05, 1.08, 1.10)
B_ARMS = ("1.0", "1.05", "1.15", "1.20")
READOUT_STEPS = tuple(range(24, 33))


def finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(value)


def integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average tie ranks; independent of scipy.stats.spearmanr."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking needs one finite vector")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        raise ValueError("paired rank vectors must have equal lengths")
    if len(x) < 3:
        return float("nan")
    rx, ry = average_ranks(x), average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if denominator == 0:
        return float("nan")
    return float(np.clip((rx @ ry) / denominator, -1.0, 1.0))


def exact_paired_decline(base: np.ndarray, other: np.ndarray) -> tuple[float, int, int]:
    """Exact binomial rational sum evaluated once as binary64, not scipy.sf."""
    base, other = np.asarray(base), np.asarray(other)
    if base.dtype != bool or other.dtype != bool or base.shape != other.shape or base.ndim != 1:
        raise ValueError("paired correctness must be equal boolean vectors")
    lost = int(np.count_nonzero(base & ~other))
    gained = int(np.count_nonzero(~base & other))
    discordant = lost + gained
    tail = sum(math.comb(discordant, j) for j in range(lost, discordant + 1))
    return tail / (1 << discordant), lost, gained


def pair_summary(base: np.ndarray, other: np.ndarray, *, boot: int = 10000) -> dict:
    if type(boot) is not int or boot <= 0 or len(base) == 0:
        raise ValueError("positive bootstrap count and nonempty pairs required")
    p, lost, gained = exact_paired_decline(base, other)
    difference = other.astype(np.int64) - base.astype(np.int64)
    # Frozen producer restarts seed zero for each metric/arm contrast.
    indices = np.random.default_rng(0).integers(len(base), size=(boot, len(base)))
    distribution = difference[indices].sum(axis=1) / len(base)
    return {"difference": float(difference.sum() / len(base)),
            "ci95": np.percentile(distribution, [2.5, 97.5]).tolist(),
            "lower_one_sided_95": float(np.percentile(distribution, 5)),
            "mcnemar_less_p": p, "lost": lost, "gained": gained}


def _a_panel(rows: list[dict], count: int) -> dict:
    table = {}
    for row in rows:
        q = integer(row["index"], "question")
        lam = finite_number(row["lam"], "lambda")
        arm = next((a for a in A_ARMS if abs(a - lam) < 1e-12), None)
        if not 0 <= q < count or arm is None or (q, arm) in table:
            raise ValueError("extra or duplicate A question/arm")
        if row.get("split") != "test" or type(row.get("seed")) is not int or row["seed"] != q:
            raise ValueError("A split/seed differs from frozen contract")
        gold = integer(row["gold"], "gold index")
        trace = row.get("native_residual_rows")
        if not isinstance(trace, list) or len(trace) != 32 or gold < 0:
            raise ValueError("A requires 32 per-choice residual rows")
        endpoint = trace[-1]
        if not isinstance(endpoint, list) or gold >= len(endpoint):
            raise ValueError("A gold coordinate absent")
        residual = finite_number(endpoint[gold], "gold residual")
        if residual < 0:
            raise ValueError("A residual cannot be negative")
        choices = []
        for step in READOUT_STEPS:
            likelihood = row["by_steps"][str(step)]["loglik"]
            if not isinstance(likelihood, list) or len(likelihood) != len(endpoint):
                raise ValueError("choice dimension changes across readouts")
            vector = [finite_number(v, "choice log-likelihood") for v in likelihood]
            # np.argmax chooses the first maximum, including exact ties.
            choices.append(max(range(len(vector)), key=lambda i: vector[i]))
        table[(q, arm)] = {"hash": row.get("gold_ids_sha256"), "residual": residual,
                            "changes": any(x != y for x, y in zip(choices, choices[1:]))}
    if set(table) != {(q, a) for q in range(count) for a in A_ARMS}:
        raise ValueError("incomplete A Cartesian panel")
    return table


def _spectrum(record: dict | None) -> tuple[str, np.ndarray | None]:
    if record is None:
        return "missing", None
    residual = record.get("fixed_point_residual_F")
    if type(residual) not in (int, float) or not math.isfinite(residual) or residual > 1e-4:
        return "no fixed point", None
    if residual < 0:
        raise ValueError("negative saved norm is malformed evidence")
    components = []
    for kind in ("LM", "SR", "LR"):
        for pair in record["spectra_of_F"].get(kind, []):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Ritz pair has wrong shape")
            if any(type(v) not in (int, float) for v in pair):
                raise ValueError("Ritz component is not numeric")
            components.append(complex(*pair))
    # Ten-decimal rounding/deduplication is part of the frozen feature definition.
    modes = np.unique(np.round(np.asarray(components, dtype=np.complex128), 10))
    if len(modes) == 0 or not np.isfinite(modes).all():
        return "empty or nonfinite spectrum", None
    if not record.get("token_ids_sha256"):
        return "no token-id record", None
    return "ok", modes


def analyze_a(spectra: dict[int, dict | None], rows: list[dict], *, count: int = 64,
              boot: int = 2000) -> dict:
    """All four correlations and every original question disposition retained."""
    if type(count) is not int or count <= 0 or type(boot) is not int or boot <= 0:
        raise ValueError("invalid A count/bootstrap count")
    if any(type(q) is not int or q not in range(count) for q in spectra):
        raise ValueError("extra spectrum question")
    table = _a_panel(rows, count)
    dispositions, cohort, features, outcomes, details = {}, [], [], [], []
    for q in range(count):
        record = spectra.get(q)
        status, modes = _spectrum(record)
        hashes = [table[(q, a)]["hash"] for a in A_ARMS]
        if status == "ok" and (not all(isinstance(h, str) and h for h in hashes)
                               or any(h != record["token_ids_sha256"] for h in hashes)):
            status = "token ids differ from the scored gold row"
        dispositions[str(q)] = {"status": status,
            "residual_F": None if record is None else record.get("fixed_point_residual_F"),
            "arnoldi_converged": None if record is None else record.get("arnoldi_converged"),
            "eigenvalues": 0 if modes is None else len(modes)}
        if status != "ok":
            continue
        cohort.append(q)
        features.append([float(max(abs((1 - a) + a * modes))) for a in A_ARMS])
        outcomes.append([table[(q, a)]["residual"] for a in A_ARMS])
        for a, feature in zip(A_ARMS, features[-1]):
            details.append({"question": q, "lambda": a, "rho": feature, **table[(q, a)]})
    x = np.asarray(features, dtype=np.float64).reshape(-1, 4)
    y = np.asarray(outcomes, dtype=np.float64).reshape(-1, 4)
    def correlations(indices):
        return [rank_correlation(x[indices, j], y[indices, j]) for j in range(4)]
    per_arm = correlations(np.arange(len(cohort)))
    statistic = float(np.mean(per_arm)) if all(map(math.isfinite, per_arm)) else float("nan")
    draws = []
    rng = np.random.default_rng(0)
    if cohort:
        for _ in range(boot):
            rs = correlations(rng.integers(len(cohort), size=len(cohort)))
            draws.append(float(np.mean(rs)) if all(map(math.isfinite, rs)) else float("nan"))
    draws = np.asarray(draws)
    valid = draws[np.isfinite(draws)]
    undefined = int(len(draws) - len(valid))
    interval = np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else [float("nan")] * 2
    minimum = math.ceil(0.75 * count)
    setup_failed = len(cohort) < minimum
    success = not setup_failed and math.isfinite(statistic) and undefined <= .01 * boot and interval[0] > .3
    return {"requested": count, "cohort": len(cohort), "cohort_indices": cohort, "minimum_cohort": minimum,
            "setup_failed": setup_failed, "dispositions": dispositions,
            "primary": {"statistic": statistic, "per_arm": dict(zip(map(str, A_ARMS), per_arm)),
                        "ci95": interval, "undefined_draws": undefined, "success": bool(success)},
            "answer_changes_descriptive": {
                "detected_unstable_changed": sum(r["rho"] >= 1 and r["changes"] for r in details),
                "detected_unstable_total": sum(r["rho"] >= 1 for r in details),
                "no_unstable_mode_detected_changed": sum(r["rho"] < 1 and r["changes"] for r in details),
                "no_unstable_mode_detected_total": sum(r["rho"] < 1 for r in details)},
            "share_detected_unstable_by_arm": {
                str(a): float(np.mean(x[:, j] >= 1)) if len(cohort) else float("nan")
                for j, a in enumerate(A_ARMS)},
            "all_cohort_cells": details, "bootstrap_statistics": draws.tolist()}


def analyze_b(rows: list[dict], *, count: int = 200, boot: int = 10000) -> dict:
    if type(count) is not int or count <= 0:
        raise ValueError("positive B count required")
    table = {}
    for row in rows:
        q = integer(row["index"], "problem")
        arm = row["arm"]
        if arm not in B_ARMS or not 0 <= q < count or (arm, q) in table:
            raise ValueError("extra or duplicate B problem/arm")
        if (row.get("split") != "test" or type(row.get("steps")) is not int or row["steps"] != 32
                or row.get("prompt_style") != "cot8" or type(row.get("seed")) is not int or row["seed"] != q
                or abs(finite_number(row["lam"], "lambda") - float(arm)) > 1e-12):
            raise ValueError("B row differs from frozen metadata")
        if any(type(row.get(metric)) is not bool for metric in ("correct", "correct_strict")):
            raise ValueError("correctness must be boolean")
        length = integer(row["new_tokens"], "new tokens")
        if not 0 <= length <= 256:
            raise ValueError("generation length outside frozen cap")
        table[(arm, q)] = row
    if set(table) != {(a, q) for a in B_ARMS for q in range(count)}:
        raise ValueError("incomplete B Cartesian panel; prefixes cannot be analyzed")
    metrics = {}
    for metric in ("correct", "correct_strict"):
        vectors = {a: np.array([table[(a, q)][metric] for q in range(count)], dtype=bool) for a in B_ARMS}
        record = {"accuracy": {a: float(v.sum() / count) for a, v in vectors.items()}}
        for a in B_ARMS[1:]:
            record[a] = pair_summary(vectors["1.0"], vectors[a], boot=boot)
        metrics[metric] = record
    primary = metrics["correct"]
    high, low, middle = primary["1.20"], primary["1.05"], primary["1.15"]
    b1 = {"arm": "1.20", "difference": high["difference"], "lost": high["lost"], "gained": high["gained"],
          "p": high["mcnemar_less_p"], "pass": high["mcnemar_less_p"] < .05}
    b3 = {"arm": "1.05", "difference": low["difference"], "lower_one_sided_95": low["lower_one_sided_95"],
          "margin": -.04, "pass": low["lower_one_sided_95"] > -.04}
    return {"n": count, "metrics": metrics, "B1_decline": b1, "B3_noninferior": b3,
            "primary_success": b1["pass"] and b3["pass"],
            "B2_descriptive": {"arm": "1.15", "difference": middle["difference"], "lost": middle["lost"],
                               "gained": middle["gained"], "p": middle["mcnemar_less_p"]},
            "cap_rate": {a: sum(table[(a, q)]["new_tokens"] >= 256 for q in range(count)) / count for a in B_ARMS},
            "tokens_median": {a: float(np.median([table[(a, q)]["new_tokens"] for q in range(count)])) for a in B_ARMS},
            "strict_format_share": {a: sum(table[(a, q)]["strict_pred"] is not None for q in range(count)) / count
                                    for a in B_ARMS},
            "all_rows_retained": [{"arm": a, "index": q, "correct": table[(a, q)]["correct"],
                                    "correct_strict": table[(a, q)]["correct_strict"],
                                    "new_tokens": table[(a, q)]["new_tokens"]} for a in B_ARMS for q in range(count)]}
