"""Independent REPRO-T saved-field arithmetic. No producer/model imports."""
import collections
import math
import re
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import numpy as np

DTYPES = ("bf16", "float64")
TEMPS = (0.0, 0.2, 0.4)
MODES = (("soft", 0.0), ("soft", 0.2), ("soft", 0.4), ("greedy", None))
LAYOUTS = ("single", "batch4", "rerun")
DELTAS = (1e-12, 1e-8, 1e-4)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(text):
    require(isinstance(text, str), "answer text type")
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def real(x):
    return type(x) in (int, float) and math.isfinite(x)


def outward(pair):
    return [str(Decimal(str(pair[0])).quantize(Decimal(".1"), rounding=ROUND_FLOOR)),
            str(Decimal(str(pair[1])).quantize(Decimal(".1"), rounding=ROUND_CEILING))]


def summarize(values):
    require(all(real(x) for x in values), "nonfinite summary")
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def growth(distance):
    require(isinstance(distance, list) and len(distance) == 32, "32 distance steps required")
    require(all(real(x) and x >= 0 for x in distance), "finite nonnegative distances required")
    if distance[0] == 0 or distance[-1] == 0:
        return {"value": None, "reason": "zero_first_or_last_distance"}
    return {"value": (math.log(distance[-1]) - math.log(distance[0])) / 31, "reason": None}


def paired_stats(half_units, draws=10000):
    """half_units=2*x; exact integer sums decide sign-flip ties."""
    require(half_units and all(type(x) is int and -2 <= x <= 2 for x in half_units), "contrast half units")
    require(type(draws) is int and draws > 0, "positive draws")
    a = np.array(half_units, dtype=np.int64)
    n, total = len(a), int(a.sum())
    rng = np.random.Generator(np.random.PCG64(0))
    boot = a[rng.integers(0, n, size=(draws, n), dtype=np.int64)].sum(axis=1)
    ci = [float(x) for x in np.quantile(boot * (50.0 / n), [.025, .975], method="linear")]
    rng = np.random.Generator(np.random.PCG64(0))
    signs = 2 * rng.integers(0, 2, size=(draws, n), dtype=np.int64) - 1
    null = (signs * a).sum(axis=1)
    b = int(np.count_nonzero(np.abs(null) >= abs(total)))
    return {"n": n, "half_units": half_units, "sum_half_units": total, "delta_pp": total * 50.0 / n,
            "ci95_pp": ci, "ci95_pp_outward_1dp": outward(ci),
            "interval_excludes_zero": ci[0] > 0 or ci[1] < 0, "positive_lower_above_zero": ci[0] > 0,
            "draws": draws, "undefined_draws": 0, "signflip_exceedances": b,
            "signflip_p_raw": b / draws, "signflip_p_plus_one": (b + 1) / (draws + 1),
            "bootstrap_sum_half_units": boot.tolist(), "signflip_sum_half_units": null.tolist()}


def condition(dtype, mode, lt):
    return f"{dtype}:{mode}:{lt}"


def check_spec(spec):
    require(all(isinstance(spec.get(k), list) and len(spec[k]) == n
                for k, n in (("dev", 8), ("heldout", 96), ("distractors", 3))), "problem counts")
    for which in ("dev", "heldout", "distractors"):
        for p in spec[which]:
            require(set(p) == {"a", "b", "answer", "carry"}, "problem fields")
            require(all(type(p[k]) is int and 10 <= p[k] < 100 for k in ("a", "b")), "operands")
            require(type(p["answer"]) is int and p["answer"] == p["a"] + p["b"], "gold sum")
            require(type(p["carry"]) is bool and p["carry"] == (p["a"] % 10 + p["b"] % 10 >= 10), "carry")
    require(sum(p["carry"] for p in spec["heldout"]) == 48 and sum(p["carry"] for p in spec["dev"]) == 4,
            "frozen carry balance")


def cohort(spec):
    result = {}
    for which in ("dev", "heldout", "distractors"):
        ordered, unordered = collections.defaultdict(list), collections.defaultdict(list)
        for i, p in enumerate(spec[which]):
            ordered[(p["a"], p["b"])].append(i)
            unordered[tuple(sorted((p["a"], p["b"])))].append(i)
        result[which] = {"n": len(spec[which]), "carry": sum(p["carry"] for p in spec[which]),
                         "unique_ordered": len(ordered), "unique_unordered": len(unordered),
                         "ordered_duplicates": [{"operands": list(k), "indices": v} for k, v in ordered.items() if len(v) > 1],
                         "unordered_duplicates": [{"operands": list(k), "indices": v} for k, v in unordered.items() if len(v) > 1]}
    result["cross_set_overlaps"] = [
        {"left": left, "left_index": i, "right": right, "right_index": j,
         "ordered_equal": (a["a"], a["b"]) == (b["a"], b["b"])}
        for left, right in (("dev", "heldout"), ("distractors", "heldout"), ("dev", "distractors"))
        for i, a in enumerate(spec[left]) for j, b in enumerate(spec[right])
        if sorted((a["a"], a["b"])) == sorted((b["a"], b["b"]))]
    return result


def distance_summary(rows):
    values = [r["growth"]["value"] for r in rows if r["growth"]["value"] is not None]
    return {"by_step": [summarize([r["distance"][j] for r in rows]) for j in range(32)],
            "growth": summarize(values), "growth_undefined_count": len(rows) - len(values)}


def calculate(records, progress, spec, draws=10000):
    check_spec(spec)
    expected = {(s, i, d, m, t) for s in ("dev", "heldout") for i in range(len(spec[s])) for d in DTYPES for m, t in MODES}
    expected_mech = {(i, t, delta) for i in range(6) for t in TEMPS for delta in DELTAS}
    pipeline, mechanism, differences, seen, seen_mech = [], [], [], set(), set()

    def compare(location, recorded, own):
        if type(recorded) is not type(own) or recorded != own:
            differences.append({"location": location, "recorded": recorded, "calculated": own})

    for line, r in enumerate(records, 1):
        require(isinstance(r, dict) and type(r.get("i")) is int, f"line {line}: identity")
        require(r.get("log10_temperature") is None or real(r.get("log10_temperature")), "temperature type")
        s, i = r.get("set"), r["i"]
        if s == "mech":
            require(real(r.get("delta")), "delta type")
            key = (i, r.get("log10_temperature"), r.get("delta"))
            require(key in expected_mech and key not in seen_mech, f"line {line}: mechanism key")
            require(r.get("dtype") == "float64" and r.get("problem") == spec["heldout"][i], "mechanism identity")
            require(set(r) == {"set", "i", "problem", "dtype", "log10_temperature", "delta", "base_text", "text", "answers_differ", "distance"}, "mechanism schema")
            require(type(r["answers_differ"]) is bool, "mechanism flag type")
            base, value = canonical(r["base_text"]), canonical(r["text"])
            text_d = r["base_text"] != r["text"]
            compare(f"line{line}/answers_differ", r["answers_differ"], text_d)
            mechanism.append({"input_line": line, "i": i, "temperature": key[1], "delta": key[2],
                              "base_canonical": base, "perturbed_canonical": value,
                              "canonical_disagreement": base != value, "text_disagreement": text_d,
                              "base_text": r["base_text"], "text": r["text"],
                              "distance": r["distance"], "growth": growth(r["distance"])})
            seen_mech.add(key)
            continue
        key = (s, i, r.get("dtype"), r.get("mode"), r.get("log10_temperature"))
        require(key in expected and key not in seen, f"line {line}: pipeline key")
        require(r.get("problem") == spec[s][i], f"line {line}: problem")
        extra = "soft_distance" if key[3] == "soft" else "first_token_difference"
        require(set(r) == {"set", "i", "problem", "dtype", "mode", "log10_temperature", *LAYOUTS,
                           "answers_differ", "rerun_identical", extra}, "pipeline schema")
        require(type(r["answers_differ"]) is bool and type(r["rerun_identical"]) is bool, "flag types")
        layouts = {}
        for layout in LAYOUTS:
            item = r[layout]
            require(set(item) == {"text", "parsed", "correct", "margin"}, "layout schema")
            require(item["parsed"] is None or type(item["parsed"]) is int, "parsed type")
            require(type(item["correct"]) is bool and real(item["margin"]) and item["margin"] >= 0, "correct/margin")
            value = canonical(item["text"])
            layouts[layout] = {"canonical": value, "invalid": value is None, "correct": value == spec[s][i]["answer"],
                               "margin": item["margin"]}
            compare(f"line{line}/{layout}/parsed", item["parsed"], value)
            compare(f"line{line}/{layout}/correct", item["correct"], layouts[layout]["correct"])
        text_d = r["single"]["text"] != r["batch4"]["text"]
        rerun_text = r["single"]["text"] == r["rerun"]["text"]
        compare(f"line{line}/answers_differ", r["answers_differ"], text_d)
        if r["rerun_identical"] and not rerun_text:
            differences.append({"location": f"line{line}/rerun_identical", "recorded": True, "calculated": False,
                                "reason": "combined trajectory/text flag implies text equality"})
        row = {"input_line": line, "set": s, "i": i, "dtype": key[2], "mode": key[3], "temperature": key[4],
               "condition": condition(*key[2:]), "layouts": layouts,
               "canonical_disagreement": layouts["single"]["canonical"] != layouts["batch4"]["canonical"],
               "text_disagreement": text_d, "pair_invalid": layouts["single"]["invalid"] and layouts["batch4"]["invalid"],
               "rerun_text_equal": rerun_text, "reported_rerun_identical": r["rerun_identical"]}
        if extra == "soft_distance":
            row.update(distance=r[extra], growth=growth(r[extra]))
        else:
            v = r[extra]
            require(v is None or type(v) is int and 1 <= v <= 32, "first greedy difference")
            row[extra] = v
        pipeline.append(row)
        seen.add(key)
    require(seen == expected and seen_mech == expected_mech, "missing complete key set")
    pkeys = [(d, s, i + 1, len(spec[s])) for d in DTYPES for s in ("dev", "heldout") for i in range(len(spec[s]))]
    require(len(progress) == len(pkeys), "progress count")
    previous = -1.0
    for r, key in zip(progress, pkeys):
        require(set(r) == {"dtype", "set", "done", "of", "seconds"}, "progress schema")
        require((r["dtype"], r["set"], r["done"], r["of"]) == key and type(r["done"]) is int and type(r["of"]) is int, "progress key/order")
        require(real(r["seconds"]) and r["seconds"] >= previous, "progress seconds")
        previous = r["seconds"]

    summaries = {"dev": {}, "heldout": {}}
    for s in summaries:
        for dtype in DTYPES:
            for mode, lt in MODES:
                cond = condition(dtype, mode, lt)
                rows = [r for r in pipeline if r["set"] == s and r["condition"] == cond]
                n = len(rows)
                fields = ("canonical_disagreement", "text_disagreement", "pair_invalid", "rerun_text_equal", "reported_rerun_identical")
                out = {"n": n, **{f"{f}_count": sum(r[f] for r in rows) for f in fields},
                       "text_vs_canonical_disagreement_count": sum(r["text_disagreement"] != r["canonical_disagreement"] for r in rows),
                       "layouts": {l: {"correct_count": sum(r["layouts"][l]["correct"] for r in rows),
                                        "invalid_count": sum(r["layouts"][l]["invalid"] for r in rows),
                                        "margin": summarize([r["layouts"][l]["margin"] for r in rows])} for l in LAYOUTS}}
                out["canonical_disagreement_percent"] = out["canonical_disagreement_count"] * 100.0 / n
                for l in LAYOUTS:
                    out["layouts"][l]["accuracy"] = out["layouts"][l]["correct_count"] / n
                    out["layouts"][l]["invalid_percent"] = out["layouts"][l]["invalid_count"] * 100.0 / n
                out["batch_minus_single_accuracy_pp"] = (out["layouts"]["batch4"]["correct_count"] - out["layouts"]["single"]["correct_count"]) * 100.0 / n
                if mode == "soft":
                    out["distance"] = distance_summary(rows)
                else:
                    out["first_token_difference_counts"] = dict(collections.Counter(str(r["first_token_difference"]) for r in rows))
                summaries[s][cond] = out
    look = {(r["i"], r["dtype"], r["mode"], r["temperature"]): int(r["canonical_disagreement"]) for r in pipeline if r["set"] == "heldout"}
    endpoints = {}
    for dtype in DTYPES:
        base = [look[(i, dtype, "greedy", None)] for i in range(96)]
        endpoints[f"{dtype}:primary_temperatures_mean"] = paired_stats([
            look[(i, dtype, "soft", 0.0)] + look[(i, dtype, "soft", 0.2)] - 2 * base[i] for i in range(96)], draws)
        for t in TEMPS:
            endpoints[f"{dtype}:soft:{t}_minus_greedy"] = paired_stats([2 * (look[(i, dtype, "soft", t)] - base[i]) for i in range(96)], draws)
    ms = {}
    for t in TEMPS:
        ms[str(t)] = {}
        for delta in (None, *DELTAS):
            rows = [r for r in mechanism if r["temperature"] == t and (delta is None or r["delta"] == delta)]
            ms[str(t)]["all_deltas" if delta is None else str(delta)] = {
                "n": len(rows), "canonical_disagreement_count": sum(r["canonical_disagreement"] for r in rows),
                "text_disagreement_count": sum(r["text_disagreement"] for r in rows), "distance": distance_summary(rows)}
    for i in range(6):
        for t in TEMPS:
            if len({r["base_text"] for r in mechanism if r["i"] == i and r["temperature"] == t}) != 1:
                differences.append({"location": f"mechanism/{i}/{t}/base_text", "reason": "different base texts across deltas"})
    return {"computational_status": "complete", "saved_field_consistency_passed": not differences,
            "coverage": {"pipeline_records": len(pipeline), "mechanism_records": len(mechanism), "progress_records": len(progress),
                         "heldout_instances": 96, "development_instances": 8},
            "cohort": cohort(spec), "pipeline": pipeline, "mechanism": mechanism, "conditions": summaries,
            "endpoints": endpoints, "mechanism_summaries": ms, "saved_field_differences": differences,
            "conventions": {"rng": "PCG64(0) independently reset for each endpoint bootstrap/sign-flip",
                            "quantile": "linear", "signflip": "inclusive absolute half-unit integer sums; plus-one and raw p retained",
                            "growth": "(log(d32)-log(d1))/31; zero endpoint undefined",
                            "identity": "trajectory equality is a reported source flag, not independent raw trajectory/token proof",
                            "mechanism_pooling": "18 dependent perturbation observations per temperature; six problem units"}}
