"""Gate BSZ1's four retrieved bands, stitch them, and render the closer view.

No model evaluation or paid-resource mutation is performed. Inputs are the
original plan, exact worker receipts, raw scans and each immutable pod archive.
Every band must pass before any combined figure is produced. Period labels
remain finite-horizon detections, not verified primitive orbits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
SCIENCE_FILES = (
    "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py",
    "tools/explore_llm_phase_map.py", "tools/explore_latent_loop.py",
    "tools/runpod_soft_runner.sh",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path)
    return {"path": str(path.relative_to(DATA)), "sha256": digest(path), "bytes": path.stat().st_size}


def counts(a):
    return {str(int(v)): int(n) for v, n in zip(*np.unique(a, return_counts=True))}


def gate_arrays(arrays, info, plan, band):
    """Check every cell and the exact band geometry, without interpreting its fate."""
    expected_shape = tuple(band["resolution"])
    rows = slice(*band["rows"])
    expected_t = np.linspace(*plan["log10_temperature"], plan["resolution"][0])[rows]
    expected_a = np.linspace(*plan["alpha"], plan["resolution"][1])
    for name, expected in (("alpha", expected_a), ("log10_temperature", expected_t)):
        value = arrays[name]
        require(value.shape == expected.shape and np.allclose(value, expected, rtol=0, atol=1e-15),
                f"{band['name']}: wrong {name} axis")
    for key in ("lyapunov", "period", "entropy", "family", "text_period", "text_family"):
        require(arrays[key].shape == expected_shape, f"{band['name']}: wrong {key} shape")
        require(np.isfinite(arrays[key]).all(), f"{band['name']}: nonfinite {key}")
    for key in ("period", "text_period"):
        value = arrays[key]
        require(np.issubdtype(value.dtype, np.integer) and ((value >= 0) & (value <= plan["max_period"])).all(),
                f"{band['name']}: invalid {key} labels")
    require((arrays["entropy"] >= 0).all(), f"{band['name']}: negative entropy")
    require(info["cells"] == int(np.prod(expected_shape)), f"{band['name']}: wrong cell count")
    require(info["period_counts"] == counts(arrays["period"]), f"{band['name']}: period summary differs")
    require(info["lyapunov_positive"] == int((arrays["lyapunov"] > plan["chaos_level"]).sum()),
            f"{band['name']}: exponent summary differs")
    require(np.isfinite(info["seconds"]) and info["seconds"] > 0, f"{band['name']}: invalid timing")
    args = info["args"]
    for key in ("model", "layer", "vectors", "concept", "prompt", "opener", "window", "dtype",
                "weights", "transient", "samples", "record", "max_period", "tol", "eps", "chaos_level"):
        require(args[key] == plan[key], f"{band['name']}: setting changed: {key}")
    for key, value in (("command", "scan"), ("device", "cuda"), ("threads", plan.get("threads", 8)), ("chunk", plan.get("chunk", 2048)),
                       ("resolution", list(expected_shape)), ("alpha", plan["alpha"]),
                       ("log_temperature", band["log10_temperature"]),
                       ("out", f"llm-soft-loop/pod64/bands/{band['name']}")):
        require(args[key] == value, f"{band['name']}: setting changed: {key}")
    for key in ("init_states", "start_vector", "prefix"):
        require(not args.get(key), f"{band['name']}: unexpected alternative initialization")
    for key in ("save_states", "projections", "word_period", "return_period"):
        require(not args.get(key), f"{band['name']}: unexpected optional diagnostic")
    for key, index in (("families", "family"), ("texts", "text_family")):
        known = {row[index]: row for row in info[key]}
        require(set(np.unique(arrays[index])) - {-1} == set(known), f"{band['name']}: unmatched {index}")
        for family, row in known.items():
            mask = arrays[index] == family
            require(int(mask.sum()) == row["cells"], f"{band['name']}: wrong family count")
            period_field = "period" if index == "family" else "text_period"
            require((arrays[period_field][mask] == row["period"]).all(), f"{band['name']}: wrong family period")
    return {"passed": True, "cells": info["cells"], "seconds": info["seconds"],
            "period_counts": counts(arrays["period"]),
            "lyapunov_range": [float(arrays["lyapunov"].min()), float(arrays["lyapunov"].max())]}


def gate_band(plan, band, worker):
    result = json.loads((worker / "result.json").read_text())
    require(result.get("hash_verified") is True, f"{band['name']}: transfer not verified")
    require(result.get("runner_returncode") == 0, f"{band['name']}: runner failed")
    require(not result.get("extract_clash"), f"{band['name']}: extraction collision needs review")
    archive = worker / "results.tgz"
    sha = digest(archive)
    require(sha == result["local_sha256"] == result["remote_sha256"], f"{band['name']}: archive hash mismatch")
    require(result["jobset"] == "scanband", f"{band['name']}: unexpected job set")
    stdout = (worker / "runner-stdout.txt").read_text()
    require(re.search(rf"end scanband-{re.escape(band['name'])} status 0(?:\s|$)", stdout),
            f"{band['name']}: numerical command did not finish successfully")
    folder = DATA / "explore/llm-soft-loop/pod64/bands" / band["name"]
    expected = [folder / name for name in ("scan.npz", "scan.json", "scan-partial.npz", "progress.jsonl")]
    with tarfile.open(archive) as tar:
        for path in expected:
            raw = tar.extractfile(str(path.relative_to(DATA))).read()
            require(hashlib.sha256(raw).hexdigest() == digest(path), f"{band['name']}: local/raw mismatch: {path.name}")
        logs = "explore/llm-soft-loop/pod64/logs/"
        hardware = tar.extractfile(logs + "hardware.txt").read()
        versions = tar.extractfile(logs + "versions.txt").read()
        numerical = tar.extractfile(logs + f"scanband-{band['name']}.log").read().decode()
    info = json.loads((folder / "scan.json").read_text())
    with np.load(folder / "scan.npz", allow_pickle=False) as arrays:
        gates = gate_arrays(arrays, info, plan, band)
        with np.load(folder / "scan-partial.npz", allow_pickle=False) as partial:
            require(int(partial["done"]) == info["cells"], f"{band['name']}: checkpoint incomplete")
            for key in ("lyapunov", "period", "entropy", "family", "text_period", "text_family"):
                require(np.array_equal(partial[key].reshape(arrays[key].shape), arrays[key]),
                        f"{band['name']}: final/checkpoint mismatch: {key}")
    progress = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
    chunk=plan.get("chunk",2048)
    expected_progress = [(start, phase, step) for start in range(0, info["cells"], chunk)
                         for phase in ("transient", "samples") for step in range(100,plan[phase]+1,100)]
    require([(r["chunk_start"], r["phase"], r["step"]) for r in progress] == expected_progress,
            f"{band['name']}: missing, duplicate or reordered progress")
    done = [json.loads(line) for line in numerical.splitlines() if line.startswith('{"done"')]
    require([r["done"] for r in done] == [min(start+chunk,info["cells"]) for start in range(0,info["cells"],chunk)],
            f"{band['name']}: not every chunk completed")
    source = {name: hashlib.sha256(subprocess.check_output(["git", "show", result["commit"] + ":" + name],
                                                          cwd=ROOT)).hexdigest() for name in SCIENCE_FILES}
    # Do not copy the private orchestration receipt: it includes rental rates.
    return {"name": band["name"], "rows": band["rows"], "gates": gates,
            "pod_id": result["pod_id"], "execution_commit": result["commit"], "source_sha256": source,
            "gpu_type": result["gpu_type"], "wall_seconds": result["wall_seconds"],
            "runner_returncode": 0, "transfer_hash_verified": True,
            "hardware": hardware.decode().strip(), "versions": versions.decode().strip(),
            "hardware_sha256": hashlib.sha256(hardware).hexdigest(),
            "versions_sha256": hashlib.sha256(versions).hexdigest(),
            "archive": artifact(archive), "files": [artifact(p) for p in expected],
            "runner_stdout": artifact(worker / "runner-stdout.txt"),
            "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "precision": "native float64 with the committed normalization/rotary patches; no per-band derivative qualification"}


def render(plan, combined, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    with np.load(DATA / plan["reference_scan"]["path"], allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    with np.load(combined / "scan.npz", allow_pickle=False) as z:
        new = {k: z[k] for k in z.files}
    require(digest(DATA / plan["reference_scan"]["path"]) == plan["reference_scan"]["sha256"], "reference scan changed")
    ly = np.concatenate([d["lyapunov"].ravel() for d in (old, new)])
    per = np.concatenate([d["period"].ravel() for d in (old, new)])
    hot = (ly > plan["chaos_level"]) & (per == 0)
    vmax = float(ly[hot].max()) if hot.any() else 1.0
    cap = max(float(np.percentile(np.maximum(-ly[~hot], 0), 99)), 1e-12)

    def picture(d):
        l, p = d["lyapunov"], d["period"]
        gray = 1 - .9 * np.clip(np.maximum(-l, 0) / cap, 0, 1) ** .5
        rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
        mask = (l > plan["chaos_level"]) & (p == 0)
        rgba[mask] = plt.get_cmap("plasma")(Normalize(0, vmax)(l[mask]))
        a, t = d["alpha"], d["log10_temperature"]
        da, dt = (a[1]-a[0])/2, (t[1]-t[0])/2
        return rgba, [a[0]-da, a[-1]+da, t[0]-dt, t[-1]+dt]

    window = [*plan["display_window"]["alpha"], *plan["display_window"]["log10_temperature"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    def panel(ax, d, title, bounds):
        rgba, extent = picture(d)
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", aspect="auto")
        ax.set(xlim=bounds[:2], ylim=bounds[2:], title=title, xlabel="Steering strength α", ylabel="log₁₀ temperature")
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=9)

    fig, ax = plt.subplots(figsize=(8.8, 8.8), layout="constrained")
    panel(ax, new, "The small island beneath the main shrimp", window)
    fig.suptitle("A finer look", fontsize=25, fontweight="bold", x=.04, ha="left")
    fig.supxlabel("512 × 512 new evaluations across the padded region · roughly 8× finer spacing\n"
                  "Finite-time sensitivity: purple–yellow marks expansion; darker gray marks contraction", fontsize=10)
    fig.savefig(dest / "baby-shrimp-close.png", dpi=180); fig.savefig(dest / "baby-shrimp-close.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), layout="constrained")
    panel(axes[0], old, "Original scan", window); panel(axes[1], new, "New, finer scan", window)
    fig.suptitle("Same region. More detail.", fontsize=23, fontweight="bold")
    fig.supxlabel("Identical view and color scale · nearest-cell rendering\n"
                  "Purple–yellow: positive exponent above 0.001 and no detected period. Gray: remaining cells; darker means stronger contraction.",
                  fontsize=9)
    fig.savefig(dest / "baby-shrimp-comparison.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 8.8), layout="constrained")
    padded = [*plan["alpha"], *plan["log10_temperature"]]
    panel(ax, new, "Full scan, including the protective margin", padded)
    ax.add_patch(Rectangle((window[0], window[2]), window[1]-window[0], window[3]-window[2],
                           fill=False, edgecolor="#00d9e8", lw=1.4, ls="--"))
    fig.supxlabel("Dashed box: closer view · structure may continue beyond the scanned boundary", fontsize=10)
    fig.savefig(dest / "baby-shrimp-padded.png", dpi=180); plt.close(fig)

    probes_path=ROOT/"docs/baby-shrimp-zoom/probe-receipt.json"
    probes=json.loads(probes_path.read_text())
    verified={r["name"]:r["period"] for r in probes["orbit_rows"] if r["verified_primitive_orbit"]}
    fig,axes=plt.subplots(1,2,figsize=(13.4,7.4),layout="constrained")
    parent_bounds=[float(old["alpha"][0]),float(old["alpha"][-1]),
                   float(old["log10_temperature"][0]),float(old["log10_temperature"][-1])]
    panel(axes[0],old,"Mother and the selected region",parent_bounds)
    panel(axes[1],new,"Small island: probe locations",window)
    axes[0].add_patch(Rectangle((plan["alpha"][0],plan["log10_temperature"][0]),
                               np.ptp(plan["alpha"]),np.ptp(plan["log10_temperature"]),
                               fill=False,edgecolor="#00d9e8",lw=1.8))
    for row in probes["trajectory_rows"]:
        ax=axes[1] if row["name"].startswith("baby-") else axes[0]
        a,t=row["alpha"],row["log10_temperature"]
        ax.scatter(a,t,s=65,facecolors="none",edgecolors="#00d9e8",lw=1.3)
        label=str(verified[row["name"]]) if row["name"] in verified else {
            "baby-chaos-left":"L","baby-chaos-right":"R","baby-chaos-below":"B","mother-chaos":"M"}[row["name"]]
        ax.annotate(label,(a,t),xytext=(6,6),textcoords="offset points",fontsize=10,color="black",
                    bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":1})
    fig.suptitle("Where the comparisons were measured",fontsize=24,fontweight="bold")
    fig.supxlabel("Numbers: verified primitive periods at separate probe points. Letters: neighboring expanding trajectories.\n"
                  "Left background: original HR10 scan. Right: new BSZ1 scan. The cyan rectangle marks the padded region.",fontsize=10)
    fig.savefig(dest/"baby-mother-locations.png",dpi=180);fig.savefig(dest/"baby-mother-locations.pdf");plt.close(fig)
    return {"shared_positive_vmax": vmax, "shared_gray_contraction_cap": cap,
            "scale_reference": "union of original HR10 and complete BSZ1 cells",
            "gray_definition": "all cells outside exponent > 0.001 AND period == 0; not all gray cells are verified periodic",
            "display_window": window, "interpolation": "nearest", "probe_receipt_sha256":digest(probes_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "docs/baby-shrimp-zoom/plan.json")
    parser.add_argument("--workers", nargs=4, type=Path, required=True, help="exact worker attempt directories, c00..c03")
    parser.add_argument("--out", type=Path, required=True, help="new audit directory, never reused")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        committed = subprocess.check_output(["git", "show", "HEAD:tools/baby_shrimp_finish.py"], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == digest(__file__), "commit analysis source before execution")
        rows = [gate_band(plan, band, worker) for band, worker in zip(plan["cloud_option"]["bands"], args.workers)]
        require(all(r["source_sha256"] == rows[0]["source_sha256"] for r in rows), "scientific source differs across bands")
        import runpod_pod as rp
        ledger = [json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
        present = {p["id"] for p in rp.inventory()["pods"]}
        for row in rows:
            pid = row["pod_id"]
            require(any(e.get("event") == "terminated_verified" and e.get("pod_id") == pid for e in ledger),
                    f"pod {pid}: termination not recorded")
            require(pid not in present and rp.pod_status(pid) is None, f"pod {pid}: still present")
            row["termination_verified"] = True
        combined = DATA / plan["output"]
        require(not combined.exists(), "combined output already exists; preserve it and investigate")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        command = [sys.executable, str(ROOT / "tools/explore_llm_soft_loop.py"), "stitch", "--bands",
                   *[f"llm-soft-loop/pod64/bands/{b['name']}" for b in plan["cloud_option"]["bands"]],
                   "--out", str(Path(plan["output"]).relative_to("explore"))]
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=180)
        with np.load(combined / "scan.npz", allow_pickle=False) as z:
            require(z["period"].shape == tuple(plan["resolution"]), "stitched shape mismatch")
            for key, bounds, n in (("alpha", plan["alpha"], plan["resolution"][1]),
                                   ("log10_temperature", plan["log10_temperature"], plan["resolution"][0])):
                require(np.allclose(z[key], np.linspace(*bounds, n), rtol=0, atol=1e-15), "stitched grid mismatch")
            for key in ("lyapunov", "period", "entropy"):
                pieces = [np.load(DATA / "explore/llm-soft-loop/pod64/bands" / b["name"] / "scan.npz")[key]
                          for b in plan["cloud_option"]["bands"]]
                require(np.array_equal(z[key], np.concatenate(pieces)), f"stitch changed {key}")
            totals = {"cells": int(z["period"].size), "period_counts": counts(z["period"]),
                      "lyapunov_range": [float(z["lyapunov"].min()), float(z["lyapunov"].max())]}
        scale = render(plan, combined, args.out)
        receipt = {"id": "BSZ1", "generated_utc": datetime.now(timezone.utc).isoformat(), "status": "all numerical and lifecycle gates passed",
                   "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "analysis_source_sha256": digest(__file__), "plan_sha256": digest(args.plan), "bands": rows,
                   "totals": totals, "combined_files": [artifact(combined / name) for name in ("scan.npz", "scan.json")],
                   "figures": [artifact(p) for p in sorted(args.out.glob("*.png"))] + [artifact(p) for p in args.out.glob("*.pdf")],
                   "plot": scale, "verified_primitive_orbit": False,
                   "limits": ["Fixed-start, finite-horizon descriptive scan only.",
                              "No Newton, primitive-period, Floquet stability or boundary continuation in this scan.",
                              "Any structure reaching the padded edge is truncated by the scan; inspect all edges visually."],
                   "cost": "see private ledger"}
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps({"status": receipt["status"], "out": str(args.out), "totals": totals}))
    except BaseException as error:
        (args.out / "failed.json").write_text(json.dumps({"error": str(error), "type": type(error).__name__})+"\n")
        raise


if __name__ == "__main__":
    main()
