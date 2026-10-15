"""One fixed, independent saved-array REPRO1 audit. External ownership is coordinator-managed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
OUTPUT = "/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26b/run-001"
PINNED = {
    "arrays": "8f405d85e864a3109c6432c7fff729c68ba4097ed916429c3c5fec3367e8c8d2",
    "original_summary": "ea92d75f137159a0b1ce814a593cbeaa0593f1fb27353398538fe41e13f2590d",
    "progress": "dd4044bf0c702447a5246472cdecac6be0ec7b2baf0e14ba77499612ac55bd6b",
    "specification": "2eaee5280d01ca2a4d37497880efa5b414c0b9d9f7b28da668d5f004b6ba7e53",
    "original_authority": "d5c2172063fcf79846591ba4104303a4d9a287903ed71558228853e52721a813",
    "source_manifest": "ed14d9dd289f7973a4d8aa65e5316df9361295098920a595264a89279b6fddf9",
    "dependency_manifest": "84d91b0a9ddd7393902be55170db21a87d60a5d0e63ba479b3a3f21e1e3b23f3",
    "evaluator_source": "6ac412b31ffb2e26c3835e8276098976020be5dbfb903b975d892d3f4830c74d",
    "plot_source": "7259dadd0fb52c42fb947d2575731a10f6419b1994e0e599bd8fd9c56399dcd2",
}
CAPS = {"cpu_threads": 1, "nice": 10, "address_space_bytes": 1024**3,
        "external_deadline_seconds": 120, "output_bytes": 16*1024**2,
        "startup_available_bytes": 20*1024**3, "runtime_available_bytes": 16*1024**3}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs)


def authenticate(plan):
    if plan.get("schema") != "task26b-saved-audit-plan-v1" or plan.get("admission") != "reviewed":
        raise ValueError("only the reviewed fixed plan may run")
    if plan["resources"] != CAPS or plan["output"] != OUTPUT:
        raise ValueError("fixed output/resource envelope")
    if set(plan["inputs"]) != set(PINNED) or set(plan["sources"]) != {"core.py", "run.py"}:
        raise ValueError("complete fixed input/source set")
    for name, sha in plan["sources"].items():
        if digest(HERE / name) != sha:
            raise ValueError("captured own source mismatch: " + name)
    observed = {}
    for key, pin in PINNED.items():
        item = plan["inputs"][key]
        if item["sha256"] != pin or digest(item["path"]) != pin or Path(item["path"]).stat().st_size != item["bytes"]:
            raise ValueError("input identity: " + key)
        observed[key] = item
    authority = read_json(plan["inputs"]["original_authority"]["path"])
    if not authority["recorded_termination_verified"] or authority["runner_returncode"] != 0:
        raise ValueError("original finite authority")
    if authority["commit"] != "dbd125a70dd1b1a1f54a12c1ff2fde275b4c7a49":
        raise ValueError("original source commit")
    by_name = {Path(x["file"]["path"]).name: x for x in authority["selected_files"]}
    for key, name in (("arrays", "repro.npz"), ("original_summary", "repro.json"), ("progress", "progress.jsonl")):
        row = by_name[name]
        if not row["archive_member_match"] or row["file"] != plan["inputs"][key]:
            raise ValueError("original file/archive authority: " + key)
    return observed


def establish_limits():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("single-thread environment: " + name)
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current > 10:
        raise ValueError("inherited nice exceeds exact 10")
    if current < 10:
        os.nice(10-current)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and hard < CAPS["address_space_bytes"]:
        raise ValueError("insufficient inherited hard AS")
    resource.setrlimit(resource.RLIMIT_AS, (CAPS["address_space_bytes"], CAPS["address_space_bytes"]))


def progress_check(records, arrays, report, np):
    expected = [(c, s) for c in ("batch64", "batch32", "precision", "bf16") for s in (250, 500, 750, 1000, 1250, 1500, "done")]
    actual = [(r.get("compare"), "done" if r.get("done") is True else r.get("step")) for r in records]
    if actual != expected:
        raise ValueError("progress record coverage/order")
    differences = []
    elapsed = {}
    for row in records:
        comp = row["compare"]
        if row.get("done") is True:
            own = sum(r["first_step_dist_above_1e-2"] is not None for r in report["cells"] if r["comparison"] == comp)
            original = row["cells_ever_above_1e-2"]
        else:
            if row["of"] != 1500 or not np.isfinite(row["seconds"]) or row["seconds"] < elapsed.get(comp, 0):
                raise ValueError("progress time/step")
            elapsed[comp] = row["seconds"]
            own = float(np.median(arrays[comp + "_dist"][row["step"]-1]))
            original = row["median_dist"]
        if own != original:
            differences.append({"record": row, "own": own, "original": original, "delta": own-original})
    return {"records": records, "differences": differences, "all_equal": not differences}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    plan = read_json(args.plan)
    if str(Path(args.output).absolute()) != OUTPUT:
        raise ValueError("exclusive fixed output")
    establish_limits()
    before = authenticate(plan)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    written = 0
    def save(name, payload, closing=False):
        nonlocal written
        data = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
        reserve = 0 if closing else 1024**2
        if written + len(data) > CAPS["output_bytes"] - reserve:
            raise ValueError("output budget (closing reserve retained)")
        with (out / name).open("xb") as f:
            f.write(data)
        written += len(data)
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(HERE))
        import numpy as np
        import core
        if np.__version__ != plan["numpy_version"]:
            raise ValueError("NumPy version")
        source = read_json(plan["inputs"]["original_summary"]["path"])
        expected_args = {"model": "qwen0.5b", "layer": 12, "vectors": "llm-phase-map/vectors-l12",
                         "concept": "philosophy", "cells_file": "docs/h2-horizon-cells.json",
                         "steps": 1500, "tail": 500, "threads": 8, "device": "cuda",
                         "compare": list(core.COMPARISONS), "out": "llm-soft-loop/pod64/repro/h2cells"}
        if source["args"] != expected_args:
            raise ValueError("original exact arguments")
        specification = read_json(plan["inputs"]["specification"]["path"])
        with np.load(plan["inputs"]["arrays"]["path"], allow_pickle=False) as z:
            if len(z.files) != len(set(z.files)):
                raise ValueError("duplicate NPZ members")
            arrays = {k: z[k] for k in z.files}
        result = core.calculate(arrays, source, specification, expected_counts={"chaotic": 192, "periodic": 192, "other": 64})
        records = [json.loads(line) for line in Path(plan["inputs"]["progress"]["path"]).read_text().splitlines() if line.strip()]
        result["progress"] = progress_check(records, arrays, result, np)
        result["original_args"] = source["args"]
        result["source_specification_note"] = specification["note"]
        result["runtime"] = {"numpy": np.__version__, "python": sys.version, "nice": os.getpriority(os.PRIO_PROCESS, 0),
                             "address_space_limits": resource.getrlimit(resource.RLIMIT_AS)}
        result["input_authority"] = before
        result["own_sources"] = plan["sources"]
        result["original_source_commit"] = "dbd125a70dd1b1a1f54a12c1ff2fde275b4c7a49"
        result["producer_exposure"] = "Original frozen paper/log summaries read before this unblinded reproduction. Later producer revisions excluded."
        result["input_hashes_unchanged"] = authenticate(plan) == before
        result["calculation_before_serialization_seconds"] = time.perf_counter() - started
        cell_payload = {"schema": "task26b-cell-results-v1", "cells": result.pop("cells"),
                        "figure_summaries": result.pop("figure_summaries")}
        save("cell-results.json", cell_payload)
        result["cell_results"] = {"path": "cell-results.json", "sha256": digest(out / "cell-results.json"),
                                  "bytes": (out / "cell-results.json").stat().st_size}
        save("report.json", result)
        save("status.json", {"computational_completed": True, "scientific_claims_accepted": False,
             "original_summary_consistency_passed": result["all_original_summaries_equal"] and result["progress"]["all_equal"],
             "worker_elapsed_before_status_seconds": time.perf_counter()-started,
             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, closing=True)
        return 0
    except Exception as exc:
        save("failure.json", {"computational_completed": False, "error_type": type(exc).__name__,
             "error": str(exc), "worker_elapsed_before_failure_seconds": time.perf_counter()-started}, closing=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
