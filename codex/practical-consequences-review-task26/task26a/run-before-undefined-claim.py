"""Bounded independent REPRO-T saved-data driver; --card PATH --out FRESH_DIR."""
import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("own_repro_reducer", HERE / "reduce.py")
reduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduce)


def load(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON token: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def pin_check(pin):
    path = Path(pin["path"])
    reduce.require(path.is_file() and path.stat().st_size == pin["bytes"], f"size/path: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    reduce.require(h.hexdigest() == pin["sha256"], f"hash: {path}")
    return dict(pin)


def pointer(obj, path):
    for key in path.strip("/").split("/") if path else ():
        key = key.replace("~1", "/").replace("~0", "~")
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def read_jsonl(path):
    rows = []
    def reject(value):
        raise ValueError(f"nonfinite JSON token: {value}")
    for line in Path(path).read_text().splitlines():
        reduce.require(bool(line.strip()), "blank JSONL record")
        rows.append(json.loads(line, parse_constant=reject))
    return rows


def execute(card, out, owned_output=None):
    reduce.require(card["admission"] == "reviewed", "card is not reviewed")
    reduce.require(card["draws"] == 10000 and card["seed"] == 0, "frozen Monte Carlo scope")
    reduce.require(card["original_source_commit"] == "a9c411ff1bf992a20f595039000c55be15c6d12c", "original source")
    reduce.require(set(card["source_sha256"]) == {"run.py", "reduce.py"}, "own source inventory")
    for name, sha in card["source_sha256"].items():
        reduce.require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == sha, f"own source {name}")
    required = {"authority", "runs", "progress", "problems", "design", "receipt", "archive", "stdout",
                "source_manifest", "loader_manifest", "producer_summary"}
    reduce.require(required <= set(card["inputs"]), "input roles")
    checked = {name: pin_check(pin) for name, pin in card["inputs"].items()}
    authority = load(checked["authority"]["path"])
    for assertion in card["original_assertions"]:
        actual = pointer(authority, assertion["pointer"])
        reduce.require(type(actual) is type(assertion["equals"]) and actual == assertion["equals"],
                       f"original authority assertion {assertion['pointer']}")
    reduce.require(authority["source_contract"] == checked["design"], "source design binding")
    for name in ("receipt", "archive", "stdout"):
        reduce.require(authority[name] == checked[name], f"original {name} binding")
    selected = {row["file"]["path"]: row for row in authority["selected_files"]}
    for name in ("runs", "progress"):
        reduce.require(checked[name]["path"] in selected and selected[checked[name]["path"]]["file"] == checked[name]
                       and selected[checked[name]["path"]]["archive_member_match"] is True, f"archive role {name}")
    out.mkdir(parents=False, exist_ok=False)
    if owned_output is not None:
        owned_output["created"] = True
    start = time.perf_counter()
    report = reduce.calculate(read_jsonl(checked["runs"]["path"]), read_jsonl(checked["progress"]["path"]),
                              load(checked["problems"]["path"]), draws=card["draws"])
    comparisons = []
    for claim in card["producer_comparisons"]:
        own = pointer(report, claim["pointer"])
        reduce.require(reduce.real(own) and reduce.real(claim["reported"]), f"numeric claim {claim['name']}")
        comparisons.append({**claim, "calculated": own, "signed_difference": own - claim["reported"],
                            "exactly_equal_to_reported_number": own == claim["reported"],
                            "comparison_role": "disclosed rounded producer claim, never an admission gate"})
    report["producer_comparisons"] = comparisons
    report["producer_comparison_nonzero_count"] = sum(x["signed_difference"] != 0 for x in comparisons)
    report["input_pins"] = checked
    report["original_source_commit"] = card["original_source_commit"]
    report["own_source_sha256"] = card["source_sha256"]
    report["original_receipt_scope"] = "reviewed source and local original retrieval/archive/termination authority; no reconstructed remote model-byte attestation"
    report["scope_limits"] = card["scope_limits"]
    report["runtime"] = {"arithmetic_elapsed_before_final_rehash_serialization_write_seconds": time.perf_counter() - start,
                         "process_peak_rss_observed_before_final_rehash_serialization_write_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "scope": "Neither field is a separately measured full worker lifetime/final peak; supervisor closure supplies external execution measurements.",
                         "python": platform.python_version(), "numpy": reduce.np.__version__}
    for pin in checked.values():
        pin_check(pin)
    report["inputs_unchanged_after_calculation"] = True
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    reduce.require(len(text.encode()) <= 16 * 1024 * 1024, "16 MiB report cap")
    (out / "report.json").write_text(text)
    manifest = {"files": [{"path": "report.json", "bytes": len(text.encode()),
                          "sha256": hashlib.sha256(text.encode()).hexdigest()}]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report["coverage"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    owned_output = {"created": False}
    try:
        coverage = execute(load(args.card), out, owned_output)
        print(json.dumps({"computational_status": "complete", "coverage": coverage}), flush=True)
    except BaseException as exc:
        if owned_output["created"] and out.is_dir() and not (out / "failure.json").exists():
            (out / "failure.json").write_text(json.dumps({"computational_status": "failed", "exception": repr(exc),
                                                        "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
