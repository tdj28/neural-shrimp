"""Own original MECH/XO saved-field calculation; --card PATH --out FRESH_DIR."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent


def own(name):
    spec = importlib.util.spec_from_file_location("own_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S, M = own("statistics"), own("map_evidence")


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pin_check(pin):
    p = Path(pin["path"])
    S.require(p.is_file() and p.stat().st_size == pin["bytes"], f"input path/size {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    S.require(h.hexdigest() == pin["sha256"], f"input SHA {p}")


def compare(own_value, original, path=""):
    """Every selected leaf is compared; unavailable/undefined/schema differences remain explicit."""
    if isinstance(own_value, dict):
        return [row for k, v in own_value.items() for row in compare(v, original.get(k) if isinstance(original, dict) else None,
                                                                   path + "/" + k.replace("~", "~0").replace("/", "~1"))]
    if isinstance(own_value, list):
        if not isinstance(original, list) or len(own_value) != len(original):
            return [{"pointer": path, "own": own_value, "producer": original, "status": "list_schema_difference"}]
        return [row for i, v in enumerate(own_value) for row in compare(v, original[i], path + "/" + str(i))]
    if M.number(own_value) and M.number(original):
        finite = math.isfinite(own_value) and math.isfinite(original)
        delta = own_value - original if finite else None
        return [{"pointer": path, "own": own_value, "producer": original, "signed_difference": delta,
                 "status": "finite_numeric" if finite else "nonfinite_numeric",
                 "exact": own_value == original if finite else M.same_number(own_value, original),
                 "within_1e-12_absolute": abs(delta) <= 1e-12 if finite else None,
                 "role": "reproduction comparison only; tolerance never erases a nonzero difference"}]
    return [{"pointer": path, "own": own_value, "producer": original,
             "status": "literal_or_undefined", "exact": type(own_value) is type(original) and own_value == original}]


def xo_coverage(data, selection, expected_args, progress):
    S.require(len(data["cells"]) == len(data["labels"]) == len(selection["cells"]) == 24, "fixed 24 XO cells")
    S.require(selection["labels"] == ["chaotic"] * 16 + ["periodic"] * 8, "original fixed XO labels")
    for key, value in expected_args.items():
        S.require(data["args"].get(key) == value, f"XO argument {key}")
    for i, cell in enumerate(data["cells"]):
        S.require(M.number(cell["twin_exponent"]), f"XO twin numeric {i}")
        for arm in S.ARMS:
            S.require(M.number(cell[arm]) and isinstance(cell[arm + "_windows"], list) and
                      len(cell[arm + "_windows"]) == 4 and all(M.number(v) for v in cell[arm + "_windows"]),
                      f"XO arm/window numeric coverage {i}/{arm}")
    S.require(len(progress) == 5, "XO full progress count")
    for i, record in enumerate(progress[:4]):
        S.require(set(record) == {"recorded", "of", "seconds"} and record["recorded"] == (i + 1) * 250 and
                  record["of"] == 1000 and M.number(record["seconds"]) and math.isfinite(record["seconds"]) and
                  record["seconds"] >= 0, f"XO progress {i}")
    S.require(progress[4] == {"done": 24, "of": 24}, "XO final completion")


def matched_bias_factors(alpha, bias):
    """Frozen R4's descriptive same-temperature comparison; no new pooled estimate or RNG."""
    groups = {"alpha": alpha, "bias": bias}
    temperatures = {name: np.asarray([c["log10_temperature"] for c in cells]) for name, cells in groups.items()}
    rows = []
    for temperature in np.unique(np.round(np.concatenate(list(temperatures.values())), 6)):
        row = {"rounded_temperature": float(temperature), "fields": {}}
        for field in S.FIELDS[:2]:
            values = {}
            for name, cells in groups.items():
                matched = np.flatnonzero(np.isclose(temperatures[name], temperature)).tolist()
                chaotic = [i for i in matched if cells[i]["chaotic"]]
                finite = [i for i in chaotic if math.isfinite(cells[i][field])]
                values[name] = {"row_cell_count": len(matched), "chaotic_count": len(chaotic),
                                "finite_count": len(finite), "finite_indices": finite,
                                "nonfinite_indices": [i for i in chaotic if i not in finite],
                                "actual_row_coordinates": sorted({cells[i]["log10_temperature"] for i in matched}),
                                "mean": float(np.mean([cells[i][field] for i in finite])) if finite else None}
            means = [values[name]["mean"] for name in ("alpha", "bias")]
            row["fields"][field] = {**values, "bias_minus_alpha": means[1] - means[0] if all(v is not None for v in means) else None}
        rows.append(row)
    return {"rows": rows, "scope": "All fixed temperature rows, original saved chaotic labels and field-finite masks; means/counts only, no pooled estimate, resampling or causal identification."}


def execute(card, out, ownership):
    S.require(card["admission"] == "reviewed" and card["draws"] == 10000 and card["seed"] == 0, "reviewed frozen calculation")
    S.require(set(card["source_sha256"]) == {"run.py", "statistics.py", "map_evidence.py"}, "own source inventory")
    for name, sha in card["source_sha256"].items():
        S.require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == sha, f"own source {name}")
    for pin in card["inputs"].values():
        pin_check(pin)
    authority = read(card["inputs"]["authority"]["path"])
    S.require([r["name"] for r in authority["runs"]] == ["amap-a", "amap-b", "bmap-a", "bmap-b", "cells"], "five original attempts")
    for r in authority["runs"]:
        name = r["name"]
        source = "92f1a20eea7794ca6348cf773e3757026568775e" if name == "cells" else "b203fb56d0daa41441a926097baa7c6743d42b8a"
        S.require(r["commit"] == source and r["runner_returncode"] == 0 and r["hash_verified"] is True and
                  r["recorded_termination_verified"] is True and r["local_sha256"] == r["remote_sha256"] == r["archive"]["sha256"],
                  f"original receipt/source disposition {name}")
        S.require(r["status_lines"][-1].endswith("status 0") and
                  any(x["fields"]["event"] == "terminated_verified" for x in r["ledger_records"]), f"original local closure {name}")
        S.require(r["upload"]["jobset_args"] == card["original_commands"][name], f"original command {name}")
        for role in ("receipt", "archive", "stdout"):
            S.require(r[role] == card["inputs"][name + ":" + role], f"original authority pin {name}/{role}")
        for row in r["selected_files"]:
            role = Path(row["file"]["path"]).name
            S.require(row["archive_member_match"] is True and row["file"] == card["inputs"][name + ":" + role],
                      f"original selected archive member {name}/{role}")
    out.mkdir(exist_ok=False)
    ownership["created"] = True
    start = time.perf_counter()
    retained = 0
    def save(name, value):
        nonlocal retained
        text = json.dumps(safe(value), separators=(",", ":"), allow_nan=False) + "\n"
        retained += len(text.encode())
        S.require(retained <= card["resources"]["max_output_bytes"], "retained output ceiling")
        (out / name).write_text(text)
    alpha, bias, maps = [], [], {}
    for name, count in (("amap-a", 2727), ("amap-b", 2626), ("bmap-a", 3483), ("bmap-b", 3354)):
        data = read(card["inputs"][name + ":gain.json"]["path"])
        S.require(len(data["cells"]) == count, f"fixed map count {name}")
        with np.load(card["inputs"][name + ":steps.npz"]["path"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with (out / (name + "-cells.jsonl")).open("x") as stream:
            def emit(record):
                nonlocal retained
                text = json.dumps(safe(record), separators=(",", ":"), allow_nan=False) + "\n"
                retained += len(text.encode())
                S.require(retained <= card["resources"]["max_output_bytes"], "retained output ceiling")
                stream.write(text)
            maps[name] = M.audit_map(data, arrays, card["expected_args"][name],
                                     jsonl(card["inputs"][name + ":progress.jsonl"]["path"]), emit)
        maps[name]["nonfinite_twin_indices_scientifically_unqualified"] = [
            i for i, c in enumerate(data["cells"]) if not all(math.isfinite(v) for v in c["lyapunov"])]
        maps[name]["literal_chaotic_with_nonfinite_twin_indices"] = [
            i for i in maps[name]["nonfinite_twin_indices_scientifically_unqualified"] if data["cells"][i]["chaotic"]]
        (alpha if name.startswith("amap") else bias).extend(data["cells"])
        del arrays, data
    S.require(len(alpha) == 5353 and len(bias) == 6837, "fixed full map sizes")
    S.require(len({(c["alpha"], c["beta"], c["log10_temperature"]) for c in alpha}) == 5353 and
              len({(c["alpha"], c["beta"], c["log10_temperature"]) for c in bias}) == 6837, "full map unique coverage")
    xo = read(card["inputs"]["cells:crossover.json"]["path"])
    selection = read(card["inputs"]["xo_selection"]["path"])
    progress = jsonl(card["inputs"]["cells:progress.jsonl"]["path"])
    xo_coverage(xo, selection, card["expected_args"]["cells"], progress)
    result = S.combined_statistics(alpha, bias, xo, selection, card["draws"])
    result["R4_same_temperature_descriptive_factors"] = matched_bias_factors(alpha, bias)
    original = read(card["inputs"]["producer_statistics"]["path"])
    comparisons = compare(result["literal_producer_shape"], original)
    report = {"computational_status": "complete", "scientific_admission": "pending_saved_evidence_review",
              "coverage": {"alpha_cells": len(alpha), "bias_cells": len(bias), "XO_cells": len(xo["cells"]),
                           "XO_windows_per_cell_per_arm": 4, "map_stored_steps_per_cell": 1000, "map_full_record_steps": 1500},
              "map_summaries": maps, "XO_original_progress": progress, "statistics": result,
              "producer_comparisons": comparisons,
              "comparison_nonzero_count": sum(row.get("signed_difference") is not None and row["signed_difference"] != 0 for row in comparisons),
              "comparison_nonexact_count": sum(not row.get("exact", False) for row in comparisons),
              "input_pins": card["inputs"], "source_sha256": card["source_sha256"], "scope": card["scope"],
              "original_historical_limits": authority["historical_limits"],
              "nonfinite_serialization": "NaN and signed infinities are explicit strings; neither finite fills nor silent omissions",
              "runtime": {"elapsed_before_final_rehash_serialization_seconds": time.perf_counter() - start,
                          "peak_rss_sampled_before_final_rehash_serialization_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "python": platform.python_version(), "numpy": np.__version__,
                          "scope": "External supervisor records full worker wall and closure, not an independently measured full-lifetime peak RSS. The internal elapsed and ru_maxrss observations stop before final rehash/write."}}
    for pin in card["inputs"].values():
        pin_check(pin)
    report["inputs_unchanged_after_calculation"] = True
    save("report.json", report)
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out.iterdir())]
    save("manifest.json", {"files": files})
    return report["coverage"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out, ownership = Path(args.out), {"created": False}
    try:
        print(json.dumps(execute(read(args.card), out, ownership)), flush=True)
    except BaseException:
        if ownership["created"]:
            with (out / "failure.json").open("x") as stream:
                json.dump({"computational_status": "failed", "traceback": traceback.format_exc()}, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
