"""Independently audit a complete BSMZ1 scan and draw matched measured close-ups.

No model evaluation, resource mutation or inferred contour is performed. Every
pixel is a saved fixed-start finite-horizon calculation. The root's separate
orbit audit is bound and cross-checked, not promoted to an exact crossing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
PLAN = "docs/baby-shrimp-zoom/memory-junction-scan-plan.json"
OLD_RECEIPT = "docs/baby-shrimp-zoom/junction-image-receipt.json"
OLD_PLAN = "docs/baby-shrimp-zoom/junction-scan-plan.json"
SOURCES = (
    "tools/baby_shrimp_memory_scan.py", "tools/baby_shrimp_memory.py",
    "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
    "tools/explore_llm_phase_map.py", "tools/explore_latent_loop.py",
)
FIELDS = ("lyapunov", "period", "entropy", "family", "text_period", "text_family")
SETTINGS = ("model", "layer", "threads", "prompt", "opener", "window", "dtype", "weights",
            "vectors", "concept", "transient", "samples", "record", "max_period", "tol",
            "eps", "chaos_level", "resolution", "alpha")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def safe_path(base, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    path = base / relative
    require(path.resolve().is_relative_to(base.resolve()), "artifact escapes storage root")
    require(not path.is_symlink(), "symlinked artifact")
    return path


def verify_artifact(base, record, key="path"):
    path = safe_path(base, record[key])
    require(path.is_file(), "missing artifact: " + str(path))
    require(digest(path) == record["sha256"], "artifact hash differs: " + str(path))
    if "bytes" in record:
        require(path.stat().st_size == record["bytes"], "artifact size differs: " + str(path))
    return path


def artifact(path):
    return {"path": str(path.resolve().relative_to(DATA.resolve())),
            "sha256": digest(path), "bytes": path.stat().st_size}


def git_bytes(commit, path):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "exact source commit required")
    require(not Path(path).is_absolute() and ".." not in Path(path).parts, "unsafe Git path")
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)


def verify_git(commit, path, expected):
    blob = git_bytes(commit, path)
    require(hashlib.sha256(blob).hexdigest() == expected, "Git source differs: " + path)
    return blob


def gate_manifest(folder):
    complete = read(folder / "complete.json")
    require(complete.get("complete") is True and complete.get("mode") == "scan", "complete scan required")
    rows = complete["files"]
    names = [r["name"] for r in rows]
    require(len(names) == len(set(names)), "duplicate manifest entry")
    require(all(Path(n).name == n for n in names), "manifest must name top-level files")
    expected = {p.name for p in folder.iterdir() if p.name != "complete.json"}
    require(set(names) == expected, "manifest omits or invents files")
    required = {"execution.json", "runtime.json", "map-equivalence.json", "scan.npz",
                "scan-partial.npz", "scan.json", "progress.jsonl", "scan.png"}
    require(required <= set(names), "missing required scan artifacts")
    for row in rows:
        require("bytes" in row, "manifest entry lacks byte size")
        verify_artifact(folder, row, key="name")
    return complete


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def counts(values):
    return {str(int(k)): int(n) for k, n in zip(*np.unique(values, return_counts=True))}


def gate_arrays(data, partial, info, plan, beta, execution, output):
    shape = tuple(plan["resolution"])
    require(len(shape) == 2 and min(shape) >= 2, "invalid grid resolution")
    require(set(data) == set(FIELDS) | {"alpha", "log10_temperature"}, "unexpected scan arrays")
    expected_partial = set(data) | {"done", "family_keys", "text_keys"}
    require(set(partial) == expected_partial, "unexpected checkpoint arrays")
    for axis, n in (("alpha", shape[1]), ("log10_temperature", shape[0])):
        expected = np.linspace(*plan[axis], n)
        value = data[axis]
        require(value.shape == (n,) and np.isfinite(value).all()
                and np.all(np.diff(value) > 0)
                and np.array_equal(value, expected), "wrong " + axis + " grid")
        require(np.array_equal(partial[axis], value), "checkpoint axis differs: " + axis)
    for name in FIELDS:
        value = data[name]
        require(value.shape == shape and np.isfinite(value).all(), "invalid field: " + name)
        require(partial[name].shape == (int(np.prod(shape)),)
                and np.array_equal(partial[name].reshape(shape), value), "checkpoint field differs: " + name)
    for name in ("period", "text_period"):
        value = data[name]
        require(np.issubdtype(value.dtype, np.integer)
                and ((value >= 0) & (value <= plan["max_period"])).all(), "invalid recurrence candidates: " + name)
    for name in ("family", "text_family"):
        require(np.issubdtype(data[name].dtype, np.integer) and (data[name] >= -1).all(), "invalid family labels")
    require((data["entropy"] >= 0).all(), "negative entropy")
    total = int(np.prod(shape))
    require(partial["done"].shape == () and np.issubdtype(partial["done"].dtype, np.integer)
            and int(partial["done"]) == total, "checkpoint incomplete")
    require(info["cells"] == total, "wrong cell summary")
    require(info["period_counts"] == counts(data["period"]), "period summary differs")
    require(info["lyapunov_positive"] == int((data["lyapunov"] > plan["chaos_level"]).sum()),
            "positive-exponent summary differs")
    limits = [float(data["lyapunov"].min()), float(data["lyapunov"].max())]
    require(info["lyapunov_range"] == limits, "growth range summary differs")
    require(np.isfinite(info["seconds"]) and info["seconds"] > 0, "invalid scan timing")
    args = info["args"]
    for key in SETTINGS:
        require(args[key] == plan[key], "scan setting differs: " + key)
    for key, value in (("log_temperature", plan["log10_temperature"]), ("command", "scan"),
                       ("device", execution["device"]), ("chunk", execution["chunk"]), ("out", output)):
        require(args[key] == value, "scan setting differs: " + key)
    require(args.get("feedback_memory", 0.) == beta, "scan memory differs")
    for key in ("prefix", "start_vector", "init_states"):
        require(args[key] is None, "not a fixed-opener scan: " + key)
    for key in ("save_states", "projections", "word_period", "return_period"):
        require(not args[key], "unexpected scan diagnostic: " + key)
    for name, label, period_name, checkpoint_key in (
            ("families", "family", "period", "family_keys"),
            ("texts", "text_family", "text_period", "text_keys")):
        rows = info[name]
        known = {row[label]: row for row in rows}
        require(len(known) == len(rows), "duplicate family summary")
        keys = json.loads(str(partial[checkpoint_key]))
        require(set(known) == set(range(len(keys))), "family dictionary indices differ")
        require(set(np.unique(data[label])) - {-1} == set(known), "family labels differ")
        require(np.array_equal(data[label] >= 0, data[period_name] > 0), "period/family assignment differs")
        for index, row in known.items():
            mask = data[label] == index
            require(row["cells"] == int(mask.sum()), "family cell count differs")
            require(row["period"] == len(row["token_ids"]) == len(row["tokens"])
                    and (data[period_name][mask] == row["period"]).all(), "family period differs")
            require(keys[index] == row["token_ids"], "checkpoint token keys differ")
            require(all(isinstance(t, int) and t >= 0 for t in row["token_ids"]), "invalid token identifiers")
    return {"cells": total, "period_counts": counts(data["period"]), "lyapunov_range": limits,
            "positive_exponent_cells": info["lyapunov_positive"], "seconds": info["seconds"]}


def gate_progress(path, plan, execution):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    total = int(np.prod(plan["resolution"]))
    expected = [(start, phase, step) for start in range(0, total, execution["chunk"])
                for phase in ("transient", "samples") for step in range(100, plan[phase] + 1, 100)]
    require([(r["chunk_start"], r["phase"], r["step"]) for r in rows] == expected,
            "progress missing, duplicated or reordered")
    times = np.array([r["seconds"] for r in rows])
    require(np.isfinite(times).all() and (times >= 0).all() and (np.diff(times) >= 0).all(), "invalid progress timing")


def gate_runtime(runtime, equivalence, execution, plan):
    require(execution["mode"] == "scan" and execution["device"] in ("cpu", "cuda"), "not a scan execution")
    require(isinstance(execution["chunk"], int) and 1 <= execution["chunk"] <= plan["maximum_chunk"], "invalid chunk")
    require(runtime["threads"] == plan["threads"] and runtime["rotary_float64"] is True,
            "runtime precision or thread setting differs")
    require(np.isfinite(runtime["norm_sensitivity"]) and 0 < runtime["norm_sensitivity"] < 1e-9,
            "effective float64 norm gate failed")
    require(isinstance(runtime["torch"], str) and runtime["torch"], "missing torch version")
    if execution["device"] == "cuda":
        require(isinstance(runtime["cuda"], str) and runtime["cuda"] and runtime["device"] != "cpu", "CUDA runtime missing")
    else:
        require(runtime["device"] == "cpu", "CPU device differs")
    require(equivalence["passed"] is True and equivalence["period"] == 18, "map-equivalence gate failed")
    for key, bound in (("one_step_relative_error", 1e-12), ("saved_root_relative_closure", 1e-10)):
        require(np.isfinite(equivalence[key]) and 0 <= equivalence[key] <= bound, "map-equivalence residual failed: " + key)


def gate_completion_times(execution, complete):
    start = datetime.fromisoformat(execution["started_utc"])
    end = datetime.fromisoformat(complete["completed_utc"])
    require(start.tzinfo is not None and end.tzinfo is not None and end >= start,
            "invalid execution/completion timestamps")
    require(isinstance(execution["pid"], int) and not isinstance(execution["pid"], bool)
            and execution["pid"] > 0, "invalid worker PID")
    return (end - start).total_seconds()


def gate_root(plan, commit):
    rec_path = verify_artifact(ROOT, plan["root_receipt"])
    verify_git(commit, plan["root_receipt"]["path"], digest(rec_path))
    compact = read(rec_path)
    full_path = verify_artifact(DATA, compact["audit"])
    full = read(full_path)
    require(compact["complete"] is True and full["complete"] is True, "root continuation incomplete")
    require(compact["execution_source_commit"] == full["execution_source_commit"]
            and compact["plan_sha256"] == full["plan_sha256"], "root source/plan differs")
    root = compact["height_root"]
    result = full["height_root"]["result"]["saddle"]
    require(root["complete"] is True and result["converged"] is True, "root not accepted")
    require(root["parameter"] == result["parameter"] == plan["beta"]
            and root["tau"] == result["tau"] and abs(root["tau"]) <= root["height_tolerance"], "root candidate differs")
    science = compact["scientific_metadata"]
    physical = np.array(science["start"]) + np.array(result["center"]) * np.array(science["coordinate_scales"])
    require(np.array_equal(physical, [root["alpha"], root["log10_temperature"]])
            and plan["center"] == physical.tolist(), "root center differs")
    require(science["period"] == 18 and science["model_revision"] == plan["model_revision"], "root model/period differs")
    expected = Path(compact["raw_directory"]) / f"point-{root['cyclic_point_index']:04d}"
    require(plan["root_state"]["path"] == str(expected / "points.npy"), "wrong root-state selection")
    state_path = verify_artifact(DATA, plan["root_state"])
    raw = safe_path(DATA, compact["raw_directory"])
    manifest = {r["path"]: r for r in full["manifest"]}
    point_dir = state_path.parent
    for name in ("points.npy", "point.json", "return-state.npy"):
        relative = str((point_dir / name).relative_to(raw))
        require(relative in manifest, "root point absent from audited manifest")
        verify_artifact(raw, manifest[relative])
    point = read(point_dir / "point.json")
    require(point["qualified"] is True and point["beta"] == plan["beta"]
            and [point["alpha"], point["log10_temperature"]] == plan["center"], "root point metadata differs")
    orbit = np.load(state_path, allow_pickle=False)
    endpoint = np.load(point_dir / "return-state.npy", allow_pickle=False).reshape(-1)
    require(orbit.shape == (18, 896) and np.isfinite(orbit).all()
            and endpoint.shape == (896,) and np.isfinite(endpoint).all(), "root state shape/nonfinite")
    scale = max(1., float(np.linalg.norm(orbit[0])))
    closure = float(np.linalg.norm(endpoint - orbit[0]) / scale)
    lower = float(np.linalg.norm(orbit[1:] - orbit[0], axis=1).min() / scale)
    require(closure <= 1e-10 and lower > 1e-8, "saved root primitive/closure replay failed")
    return {"receipt": {**plan["root_receipt"], "git_commit": commit}, "audit": compact["audit"],
            "state": plan["root_state"], "beta": plan["beta"], "center": plan["center"],
            "trace_residual": root["tau"], "height_tolerance": root["height_tolerance"],
            "recomputed_saved_endpoint_closure": closure, "minimum_saved_lower_lag": lower,
            "replay_normalization": "Euclidean state difference divided by max(1, norm of first saved orbit state)",
            "scope": "Saved root-state replay only; scan pixels do not inherit its orbit or exact-root status."}


def gate_reference(plan, commit):
    old_path = ROOT / OLD_RECEIPT
    verify_git(commit, OLD_RECEIPT, digest(old_path))
    old = read(old_path)
    require(old["id"] == "BSZ2-image-audit" and len(old["bands"]) == 1, "wrong reference audit")
    band = old["bands"][0]
    require(band["gates"]["passed"] is True and band["termination_verified"] is True
            and band["model_revision"] == plan["model_revision"], "reference gates/model differ")
    for path, sha in band["source_sha256"].items():
        verify_git(band["execution_commit"], path, sha)
    old_plan = json.loads(verify_git(band["execution_commit"], OLD_PLAN, old["plan_sha256"]))
    scan_path = verify_artifact(DATA, plan["reference_scan"])
    indexed = {entry["path"]: entry for entry in band["files"]}
    for name in ("scan.npz", "scan.json", "scan-partial.npz", "progress.jsonl"):
        relative = str((scan_path.parent / name).relative_to(DATA))
        require(relative in indexed, "reference file not bound by historical audit")
        verify_artifact(DATA, indexed[relative])
    old_data = arrays(scan_path)
    info = read(scan_path.with_suffix(".json"))
    expected = {**old_plan, "resolution": list(old_data["period"].shape),
                "threads": old_plan.get("threads", 8)}
    old_execution = {"chunk": info["args"]["chunk"], "device": "cuda"}
    total = gate_arrays(old_data, arrays(scan_path.parent / "scan-partial.npz"), info, expected, 0.,
                        old_execution, str(scan_path.parent.relative_to(DATA / "explore")))
    gate_progress(scan_path.parent / "progress.jsonl", expected, old_execution)
    require(total["period_counts"] == old["totals"]["period_counts"], "historical totals differ")
    # Strict physical spans, translated centers; the old center is an anchor, not an exact saddle.
    for key in ("alpha", "log10_temperature"):
        require(np.isclose(np.ptp(old_data[key]), np.ptp(plan[key]), rtol=0, atol=1e-15), "unmatched physical spans")
    require(old_data["period"].shape == tuple(plan["resolution"]), "unmatched grid resolution")
    return old_data, old["plot"]["color_scale"], {
        "receipt": {"path": OLD_RECEIPT, "sha256": digest(old_path), "git_commit": commit},
        "scan": plan["reference_scan"], "execution_commit": band["execution_commit"],
        "source_sha256": band["source_sha256"], "gpu_type": band["gpu_type"],
        "hardware": band["hardware"], "versions": band["versions"], "beta": 0., "totals": total,
        "center": [float(np.mean(old_data[k][[0, -1]])) for k in ("alpha", "log10_temperature")],
        "scope": "Historical BSZ2 audit rebound to saved source and arrays; no new remote lifecycle query."}


def image_data(data, scale, chaos_level):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    lyap, period = data["lyapunov"], data["period"]
    gray = 1 - .9 * np.clip(np.maximum(-lyap, 0) / scale["shared_gray_contraction_cap"], 0, 1)**.5
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
    hot = (lyap > chaos_level) & (period == 0)
    rgba[hot] = plt.get_cmap("plasma")(Normalize(0, scale["shared_positive_vmax"])(lyap[hot]))
    return rgba


def render(old, new, plan, scale, out, old_evidence, runtime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    for key in ("shared_gray_contraction_cap", "shared_positive_vmax"):
        require(np.isfinite(scale[key]) and scale[key] > 0, "invalid shared color scale")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 8.0))
    fig.subplots_adjust(left=.075, right=.89, top=.82, bottom=.25, wspace=.24)
    centers = [old_evidence["center"], plan["center"]]
    for ax, data, center, title in zip(axes, [old, new], centers,
            ["Original close-up · no retained feedback", f"Translated close-up · {100*plan['beta']:.5f}% retained"]):
        a, t = (data[k] for k in ("alpha", "log10_temperature"))
        da, dt = np.diff(a).mean(), np.diff(t).mean()
        extent = 1e6 * (np.array([a[0]-da/2, a[-1]+da/2, t[0]-dt/2, t[-1]+dt/2])
                        - np.array([center[0], center[0], center[1], center[1]]))
        ax.imshow(image_data(data, scale, plan["chaos_level"]), origin="lower", interpolation="nearest",
                  aspect="auto", extent=extent)
        spans = [np.ptp(plan[k]) * 1e6 / 2 for k in ("alpha", "log10_temperature")]
        ax.set(xlim=(-spans[0], spans[0]), ylim=(-spans[1], spans[1]), title=title,
               xlabel="Steering offset Δα (×10⁻⁶)", ylabel="Temperature offset Δlog₁₀T (×10⁻⁶)")
        ax.set_xticks([-1.5, 0, 1.5]); ax.set_yticks([-4.5, -3, -1.5, 0, 1.5, 3, 4.5])
        ax.tick_params(labelsize=10)
    cap = scale["shared_gray_contraction_cap"]
    rates = np.linspace(-cap, 0, 256)
    gray = 1-.9*np.sqrt(np.maximum(-rates, 0)/cap)
    cmap = ListedColormap(np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1))
    cax = fig.add_axes([.92, .30, .016, .45])
    fig.colorbar(ScalarMappable(norm=Normalize(-cap, 0), cmap=cmap), cax=cax,
                 label="Finite-time growth per step (gray cells)")
    fig.suptitle("The child's center, with and without retained feedback", x=.075, ha="left",
                 y=.96, fontsize=21, fontweight="bold")
    fig.text(.075, .88, "Measured network scans · same physical span and gray scale · no fitted crossing overlaid", fontsize=12)
    text = (f"Left center: α = {centers[0][0]:.11f}, log₁₀T = {centers[0][1]:.11f}; β = 0.\n"
            f"Right center: α = {centers[1][0]:.11f}, log₁₀T = {centers[1][1]:.11f}; β = {plan['beta']:.10f}.\n"
            f"Each panel: {plan['resolution'][1]} × {plan['resolution'][0]} fixed-start cells; "
            f"{plan['transient']} settling + {plan['samples']} measurement steps; nearest-cell rendering.\n"
            "Darker gray means stronger contraction. Trace reconnection has separate orbit evidence; these ridges need not connect.\n"
            "Purple–yellow, if present: exponent >0.001 with no detected recurrence ≤32. No exact superstability is inferred.")
    fig.text(.075, .06, text, fontsize=9.2, linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig-memory-junction-pair.{ext}", dpi=190)
    plt.close(fig)
    return {"interpolation": "nearest", "coordinate_display": "offsets from each window center, multiplied by 1e6",
            "centers": centers, "beta_values": [0., plan["beta"]], "color_scale": scale,
            "spans": [float(np.ptp(plan[k])) for k in ("alpha", "log10_temperature")],
            "source_devices": [old_evidence["gpu_type"], runtime["device"]], "caption": text,
            "overlaid_fitted_curves": False, "visual_review": "pending"}


def run(scan, out):
    scan, out = scan.resolve(), out.resolve()
    require(scan.is_relative_to(DATA.resolve()) and out.is_relative_to(DATA.resolve()), "scan and audit must be external data")
    require(not out.is_relative_to(scan) and not scan.is_relative_to(out), "audit and raw directories must be disjoint")
    out.mkdir(parents=True, exist_ok=False)
    try:
        analysis = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        verify_git(analysis, str(Path(__file__).relative_to(ROOT)), digest(__file__))
        initial = digest(scan / "complete.json")
        complete = gate_manifest(scan)
        execution = read(scan / "execution.json")
        elapsed = gate_completion_times(execution, complete)
        require(set(execution["source_sha256"]) == set(SOURCES), "execution source set differs")
        for path, sha in execution["source_sha256"].items():
            verify_git(execution["source_commit"], path, sha)
        plan = json.loads(verify_git(execution["source_commit"], PLAN, execution["plan_sha256"]))
        require(plan["id"] == "BSMZ1" and plan["window"] == 1 and plan["dtype"] == "float64"
                and np.isfinite(plan["beta"]) and 0 < plan["beta"] <= .05, "unsupported memory scan")
        require(np.allclose([np.mean(plan[k]) for k in ("alpha", "log10_temperature")],
                            plan["center"], rtol=0, atol=1e-16), "new window is not centered on candidate")
        root = gate_root(plan, execution["source_commit"])
        vector = verify_artifact(DATA, plan["steering_vectors"])
        require(plan["steering_vectors"]["path"] == "explore/" + plan["vectors"] + "/vectors.npz", "steering path differs")
        old, scale, old_evidence = gate_reference(plan, execution["source_commit"])
        runtime, equivalence = read(scan / "runtime.json"), read(scan / "map-equivalence.json")
        gate_runtime(runtime, equivalence, execution, plan)
        info, new = read(scan / "scan.json"), arrays(scan / "scan.npz")
        totals = gate_arrays(new, arrays(scan / "scan-partial.npz"), info, plan, plan["beta"], execution,
                             str(scan.relative_to(DATA / "explore")))
        gate_progress(scan / "progress.jsonl", plan, execution)
        require(info["prefix_tokens"] == 37 and info["prefix_text"] == read(
            safe_path(DATA, plan["reference_scan"]["path"]).with_suffix(".json"))["prefix_text"], "prefix differs")
        plot = render(old, new, plan, scale, out, old_evidence, runtime)
        gate_manifest(scan)
        require(digest(scan / "complete.json") == initial, "completion manifest changed during audit")
        for bound in (plan["reference_scan"], plan["root_state"], plan["steering_vectors"], root["audit"]):
            verify_artifact(DATA, bound)
        verify_artifact(ROOT, plan["root_receipt"])
        require(digest(ROOT / OLD_RECEIPT) == old_evidence["receipt"]["sha256"],
                "reference audit changed during rendering")
        receipt = {"id": "BSMZ1-memory-scan-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "complete": True, "status": "all saved-array, source, plan, root, reference and runtime gates passed",
                   "analysis_commit": analysis, "analysis_source_sha256": digest(__file__),
                   "execution": execution, "completion_manifest": artifact(scan / "complete.json"),
                   "execution_to_completion_seconds": elapsed,
                   "raw_files": [artifact(scan / r["name"]) for r in complete["files"]],
                   "plan": {"path": PLAN, "sha256": execution["plan_sha256"]}, "root_evidence": root,
                   "steering_vectors": artifact(vector), "reference": old_evidence, "runtime": runtime,
                   "map_equivalence": equivalence, "totals": totals, "plot": plot,
                   "figures": [artifact(p) for p in sorted(out.iterdir()) if p.suffix in (".png", ".pdf")],
                   "visual_review": "pending", "lifecycle": "not checked here; separate controller/owner audit required",
                   "limits": ["Finite-horizon fixed-start measurements; period labels are recurrence candidates, not verified primitive orbits.",
                              "Every pixel is measured. No fitted crossing or interpolated saddle field replaces data.",
                              "The right panel is at a residual-qualified trace-root estimate, not an exact certified crossing.",
                              "Trace-zero contours, exact contraction ridges and finite-time image tracks are distinct.",
                              "The original panel is centered on the saved anchor; the translated panel uses the continued candidate.",
                              "Root-state replay is independent of scan recurrence candidates; no full orbit was retained for every pixel."]}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        return receipt
    except BaseException as error:
        (out / "failed.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)}) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.scan, args.out)
    print(json.dumps({"status": receipt["status"], "totals": receipt["totals"], "visual_review": "pending"}))


if __name__ == "__main__":
    main()
