"""Offline continuity diagnostics for SHA-pinned, frozen BSC3 audit receipts.

Compare full-state cycles, not just their shared token readout. This never
loads a model, contacts a provider, or certifies a global orbit branch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
SOURCE = "tools/baby_shrimp_memory_orbit_alignment.py"
AUDITOR = "tools/baby_shrimp_memory_finish.py"
METADATA = ("id", "model_revision", "period", "start", "coordinate_scales",
            "anchor_directory", "anchor_hashes", "intervention")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_sha(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def git_bytes(commit, path, root=ROOT):
    require(valid_sha(commit, 40), "exact Git commit required")
    path = Path(path)
    require(not path.is_absolute() and ".." not in path.parts, "invalid Git source path")
    return subprocess.check_output(["git", "show", f"{commit}:{path.as_posix()}"], cwd=root)


def read(path):
    return json.loads(Path(path).read_text())


def verify_manifest(run, manifest):
    """Reject missing, changed, additional, or symlinked raw files."""
    require(isinstance(manifest, list) and manifest, "empty frozen raw manifest")
    entries = {}
    for entry in manifest:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() not in entries, "invalid or duplicate raw manifest path")
        path = run / relative
        require(path.resolve().is_relative_to(run) and path.is_file()
                and not any(p.is_symlink() for p in (path, *path.parents) if p != run.parent),
                "raw manifest path missing, symlinked or outside attempt")
        require(valid_sha(entry["sha256"]) and path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"], "frozen raw artifact hash/size differs: " + str(relative))
        entries[relative.as_posix()] = entry
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    require(actual == set(entries), "unmanifested or missing raw files; attempt is not a frozen snapshot")
    require("result.json" in entries or "failure.json" in entries,
            "raw attempt has no terminal result/failure; do not inspect a live run")
    return entries


def load_receipt(path, expected_sha, *, root=ROOT, data=DATA):
    path, data = Path(path).resolve(), Path(data).resolve()
    require(valid_sha(expected_sha) and path.is_relative_to(data) and digest(path) == expected_sha,
            "receipt location or explicit SHA-256 differs")
    receipt = read(path)
    require(receipt["id"] == "BSC3-memory-offline-audit" and receipt["mode"] == "continue"
            and receipt["status"] in ("all saved-data gates passed", "partial attempt; qualified subset audited"),
            "receipt is not a completed BSC3 saved-data audit")
    require(hashlib.sha256(git_bytes(receipt["analysis_commit"], AUDITOR, root)).hexdigest()
            == receipt["analysis_source_sha256"], "audit source differs from Git")
    if receipt.get("worker") is not None:
        require(receipt["worker"].get("termination_verified") is True, "rented worker termination is unverified")
    run = Path(receipt["raw_directory"]).resolve()
    require(run.is_relative_to(data), "raw attempt outside data store")
    manifest = verify_manifest(run, receipt["manifest"])
    protocol = read(run / "protocol.json")
    commit = protocol["commit"]
    require(commit == receipt["execution_source_commit"], "execution commit differs")
    require(protocol["source_sha256"] and all(valid_sha(v) for v in protocol["source_sha256"].values()),
            "invalid execution source manifest")
    for source, sha in protocol["source_sha256"].items():
        require(hashlib.sha256(git_bytes(commit, source, root)).hexdigest() == sha,
                "execution source differs from Git: " + source)
    plan_blob = git_bytes(commit, protocol["args"]["plan"], root)
    require(hashlib.sha256(plan_blob).hexdigest() == protocol["plan_sha256"] == receipt["plan_sha256"]
            and json.loads(plan_blob) == protocol["plan"], "plan binding differs")
    require(protocol["plan"]["period"] == 18
            and protocol["plan"]["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775",
            "unexpected orbit period or model")
    qualified = {row["index"]: row for row in receipt["qualified_points"]}
    samples, omitted = [], []
    for center in receipt["accepted_centers"]:
        if center.get("kind") != "continued saddle":
            omitted.append({"parameter": center["parameter"], "kind": center.get("kind"),
                            "reason": "No newly audited continued-saddle endpoint in this receipt."})
            continue
        require(center.get("converged") is True, "accepted continued saddle lacks a true convergence label")
        index = center["cyclic_point_index"]
        require(index in qualified and qualified[index]["qualified"], "accepted saddle lacks a qualified endpoint")
        directory = run / f"point-{index:04d}"
        relative = directory.relative_to(run)
        point_path, orbit_path = directory / "point.json", directory / "points.npy"
        require((relative / "point.json").as_posix() in manifest and (relative / "points.npy").as_posix() in manifest,
                "accepted orbit is not manifest-bound")
        point = read(point_path)
        require(point["qualified"] and point["index"] == index
                and point["beta"] == center["parameter"] and point["coordinate"] == center["center"]
                and [p["phase"] for p in point["phases"]] == [0, 6, 12], "accepted saddle/point linkage differs")
        orbit = np.load(orbit_path, allow_pickle=False)
        require(orbit.shape == (18, 896) and orbit.dtype == np.float64 and np.isfinite(orbit).all(),
                "invalid full-state orbit array")
        norms = np.linalg.norm(orbit, axis=1)
        require(np.all(norms > 0), "zero orbit state")
        lower = np.linalg.norm(orbit[1:] - orbit[0], axis=1) / norms[0]
        primitive = point["primitive"]
        require(primitive["closed"] and primitive["no_lower_period"]
                and primitive["all_returns"][-1] < 1e-11 and min(lower) > 1e-8
                and np.allclose(lower, primitive["all_returns"][:-1], rtol=1e-10, atol=1e-13),
                "saved primitive-orbit evidence differs")
        endpoint = directory / "return-state.npy"
        independent_closure = endpoint.exists()
        if independent_closure:
            final = np.load(endpoint, allow_pickle=False)
            require(final.size == 896 and final.dtype == np.float64 and np.isfinite(final).all()
                    and np.linalg.norm(final.reshape(-1) - orbit[0]) / norms[0] < 1e-12,
                    "saved final state does not close")
        tokens = point["tokens"]
        require(len(tokens) == 18, "token cycle length differs")
        summary = {"parameter": center["parameter"], "center": center["center"],
                   "alpha": point["alpha"], "log10_temperature": point["log10_temperature"],
                   "point_index": index, "receipt": str(path.relative_to(data)), "receipt_sha256": expected_sha,
                   "orbit_path": str(orbit_path.relative_to(data)), "orbit_sha256": digest(orbit_path),
                   "point_sha256": digest(point_path), "minimum_lower_return": float(min(lower)),
                   "closure_recomputed_from_saved_endpoint": independent_closure}
        samples.append({"summary": summary, "orbit": orbit, "tokens": tokens})
    require(samples, "no accepted continued-saddle cycles")
    return {"receipt": receipt, "path": path, "sha256": expected_sha, "run": run,
            "samples": samples, "omitted_centers": omitted,
            "scientific_metadata": {key: protocol["plan"][key] for key in METADATA}}


def compare_orbits(left, right):
    """Use every state and every cyclic alignment, with left-state norm scaling."""
    a, b = left["orbit"], right["orbit"]
    denominator = np.linalg.norm(a, axis=1)
    distances = np.array([np.linalg.norm(np.roll(b, shift, axis=0) - a, axis=1) / denominator
                          for shift in range(len(a))])
    costs = distances.max(axis=1)
    order = np.argsort(costs, kind="stable")
    best, second = map(int, order[:2])
    parameter_step = right["summary"]["parameter"] - left["summary"]["parameter"]
    matching_tokens = np.array_equal(np.roll(np.asarray(right["tokens"]), best), left["tokens"])
    return {"left_point": left["summary"], "right_point": right["summary"],
            "comparison_kind": "same-parameter repeat" if parameter_step == 0 else "parameter-sorted neighbors",
            "parameter_step": float(parameter_step), "best_right_roll": best, "second_best_right_roll": second,
            "maximum_relative_orbit_change": float(costs[best]),
            "rms_relative_orbit_change": float(np.sqrt(np.mean(distances[best]**2))),
            "second_best_maximum_relative_change": float(costs[second]),
            "alignment_margin": float(costs[second] - costs[best]),
            "all_alignment_maximum_relative_changes": costs.tolist(),
            "direct_phase_maximum_relative_change": float(costs[0]),
            "relative_change_per_parameter_step": float(costs[best] / abs(parameter_step)) if parameter_step else None,
            "tokens_match_at_best_full_state_alignment": bool(matching_tokens)}


def analyze(receipt_pairs, *, root=ROOT, data=DATA):
    bundles = [load_receipt(path, sha, root=root, data=data) for path, sha in receipt_pairs]
    require(len({str(b["path"]) for b in bundles}) == len(bundles), "duplicate input receipt")
    require(all(b["scientific_metadata"] == bundles[0]["scientific_metadata"] for b in bundles),
            "scientific metadata differs across receipts")
    samples = [sample for bundle in bundles for sample in bundle["samples"]]
    samples.sort(key=lambda s: (s["summary"]["parameter"], s["summary"]["receipt"], s["summary"]["point_index"]))
    require(len(samples) >= 2, "at least two accepted cycles required")
    comparisons = [compare_orbits(a, b) for a, b in zip(samples, samples[1:])]
    # A live attempt or replaced receipt must not become a frozen result merely
    # because its first snapshot matched. No source files are written here.
    for bundle in bundles:
        verify_manifest(bundle["run"], bundle["receipt"]["manifest"])
        require(digest(bundle["path"]) == bundle["sha256"], "receipt changed during comparison")
    return {"id": "BSC3-memory-full-orbit-alignment", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "frozen inputs verified; numerical continuity diagnostics only",
            "scientific_metadata": bundles[0]["scientific_metadata"],
            "inputs": [{"path": str(b["path"].relative_to(Path(data).resolve())), "sha256": b["sha256"],
                        "complete": b["receipt"]["complete"], "status": b["receipt"]["status"],
                        "execution_source_commit": b["receipt"]["execution_source_commit"],
                        "analysis_commit": b["receipt"]["analysis_commit"],
                        "omitted_centers": b["omitted_centers"]} for b in bundles],
            "accepted_cycles": len(samples), "comparisons": comparisons,
            "summary": {"all_best_alignments_preserve_phase": all(p["best_right_roll"] == 0 for p in comparisons),
                        "all_aligned_token_cycles_match": all(p["tokens_match_at_best_full_state_alignment"] for p in comparisons),
                        "maximum_neighbor_relative_orbit_change": max(p["maximum_relative_orbit_change"] for p in comparisons),
                        "minimum_second_alignment_relative_change": min(p["second_best_maximum_relative_change"] for p in comparisons)},
            "metric": "For each of18 cyclic np.roll shifts of the right orbit, maximize ||right_shift[i]-left[i]||/||left[i]|| over all18 full896-dimensional states. Keep all costs; no threshold promotes branch identity.",
            "limits": ["Parameter-sorted neighboring samples are not the execution warm-start graph or a continuous branch certificate.",
                       "Small unambiguous changes support, but do not prove, continuation on the same orbit sheet; an intervening basin switch remains possible.",
                       "Shared tokens alone do not establish orbit identity. No interval, exact superstability, nonlinear reduction or native-coupling claim follows.",
                       "Missing archived return-state arrays retain the parent audit's recorded-closure limitation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists(), "new output file outside Git under data store required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_sha = digest(ROOT / SOURCE)
    require(hashlib.sha256(git_bytes(commit, SOURCE)).hexdigest() == source_sha, "commit diagnostic source before execution")
    for path, _ in args.receipt:
        require(not out.is_relative_to(Path(read(path)["raw_directory"]).resolve()),
                "output must not add a file to an immutable raw attempt")
    result = analyze(args.receipt)
    result.update(analysis_commit=commit, analysis_source_sha256={SOURCE: source_sha}, threads=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "sha256": digest(out), "cycles": result["accepted_cycles"],
                      "comparisons": len(result["comparisons"])}))


if __name__ == "__main__":
    main()
