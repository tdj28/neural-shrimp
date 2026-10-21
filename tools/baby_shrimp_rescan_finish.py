"""Audit BSZ2/GBZ1 measured images and render physically aligned close views.

Reuses the original per-band numerical gates. Images are nearest-cell fields,
never interpolation from the saddle model. Optional roots require their own
completed full-state/cyclic-spectral audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
import baby_shrimp_finish as audit


def load_scan(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def picture(data, scale, chaos_level):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    l, p = data["lyapunov"], data["period"]
    gray = 1-.9*np.clip(np.maximum(-l, 0)/scale["shared_gray_contraction_cap"], 0, 1)**.5
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
    hot = (l > chaos_level) & (p == 0)
    rgba[hot] = plt.get_cmap("plasma")(Normalize(0, scale["shared_positive_vmax"])(l[hot]))
    a, t = data["alpha"], data["log10_temperature"]
    da, dt = np.diff(a).mean(), np.diff(t).mean()
    return rgba, [a[0]-da/2, a[-1]+da/2, t[0]-dt/2, t[-1]+dt/2]


def verify_roots(path, bounds):
    receipt = json.loads(path.read_text())
    audit.require(receipt["id"] == "BSC2-direct-branch-audit" and receipt["worker"]["termination_verified"],
                  "completed direct-root audit required")
    for entry in receipt["files"]:
        source = audit.DATA / entry["path"]
        audit.require(audit.digest(source) == entry["sha256"], "root audit artifact changed")
    roots = receipt["summary"]["roots"]
    audit.require(len(roots) == 6 and len(receipt["summary"]["gaps"]) == 3, "incomplete root audit")
    for root in roots:
        p = root["point"]
        audit.require(bounds[0] < p["alpha"] < bounds[1] and bounds[2] < p["log10_temperature"] < bounds[3],
                      "direct root outside displayed scan")
    return roots, {"receipt": audit.artifact(path), "summary": receipt["summary"]}


def render(plan, data, out, curves=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    baseline = json.loads((audit.ROOT / "docs/baby-shrimp-zoom/zoom-receipt.json").read_text())
    scale = baseline["plot"]
    reference = audit.DATA / plan["reference_scan"]["path"]
    audit.require(audit.digest(reference) == plan["reference_scan"]["sha256"], "parent scan hash differs")
    old = load_scan(reference)
    bounds = [*plan["alpha"], *plan["log10_temperature"]]
    junction = plan["id"] == "BSZ2"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    def panel(ax, source, title, view=bounds):
        rgba, extent = picture(source, scale, plan["chaos_level"])
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", aspect="auto")
        ax.set(xlim=view[:2], ylim=view[2:], title=title, xlabel="Steering strength α", ylabel="log₁₀ temperature")
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.7f" if junction else "%.6f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.7f" if junction else "%.6f"))
        ax.tick_params(labelsize=9)
    title = "A measured close-up of the junction" if junction else "A finer look at the next small island"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.7))
    fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.85, wspace=.30)
    panel(axes[0], old, "Previous measurements, enlarged")
    panel(axes[1], data, f"New {len(data['alpha'])} × {len(data['log10_temperature'])} scan")
    fig.suptitle(title, fontsize=24, fontweight="bold", x=.08, ha="left", y=.96)
    fig.text(.08, .065, "Identical physical window, orientation and original color scale; nearest-cell rendering, with no smoothing.\n"
             "Each new pixel is a fresh finite-time network calculation. Dark tracks show strong contraction, not proven multiplier zeros.", fontsize=10)
    stem = "junction-measured-comparison" if junction else "grandchild-measured-comparison"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    fig.subplots_adjust(left=.16, right=.97, bottom=.17, top=.86)
    panel(ax, data, title)
    fig.text(.16, .065, "Fresh fixed-start measurements: 300 settling + 300 measurement steps.\n"
             "Gray: contraction shading; purple–yellow: positive exponent >0.001 with no period ≤32.\n"
             "The image does not by itself verify primitive orbits or complete boundary organization.", fontsize=9)
    stem = "junction-measured" if junction else "grandchild-measured"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180)
    plt.close(fig)
    root_evidence = None
    if curves:
        audit.require(junction, "trace roots only apply to the junction scan")
        roots, root_evidence = verify_roots(curves, bounds)
        fig, ax = plt.subplots(figsize=(10.2, 8.6))
        fig.subplots_adjust(left=.15, right=.84, bottom=.22, top=.86)
        _, extent = picture(data, scale, plan["chaos_level"])
        im = ax.imshow(data["lyapunov"], origin="lower", extent=extent, interpolation="nearest", aspect="auto",
                       cmap="gray", vmin=float(data["lyapunov"].min()), vmax=float(data["lyapunov"].max()))
        for side, color, label in [("negative", "#00b8c7", "Measured branch A roots"), ("positive", "#ff9a32", "Measured branch B roots")]:
            selected = [r["point"] for r in roots if r["side"] == side]
            ax.plot([r["alpha"] for r in selected], [r["log10_temperature"] for r in selected],
                    "o--", color=color, ms=7, lw=1.3, markeredgecolor="black", markeredgewidth=.5, label=label)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="Steering strength α", ylabel="log₁₀ temperature")
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.7f")); ax.yaxis.set_major_formatter(FormatStrFormatter("%.7f"))
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc="upper left", framealpha=.9)
        cax = fig.add_axes([.88, .26, .025, .53])
        fig.colorbar(im, cax=cax, label="Measured finite-time growth rate per step")
        fig.suptitle("Fresh scan with independently measured roots", fontsize=22, fontweight="bold", y=.96)
        fig.text(.15, .085, "Contrast uses this scan's full measured range; darker means stronger contraction.\n"
                 "Dots: six actual trace-zero roots. Dashed segments only connect those dots.\n"
                 "Trace zero balances two nonzero opposite-sign multipliers; it is not exact superstability.", fontsize=9)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"junction-measured-with-roots.{ext}", dpi=180)
        plt.close(fig)
    if not junction:
        # The unchanged two parent arrays establish the location across scales.
        parent_plan = json.loads((audit.ROOT / "docs/baby-shrimp-zoom/plan.json").read_text())
        mother_path = audit.DATA / parent_plan["reference_scan"]["path"]
        audit.require(audit.digest(mother_path) == parent_plan["reference_scan"]["sha256"], "mother source differs")
        mother = load_scan(mother_path)
        mother_bounds = [float(mother["alpha"][0]), float(mother["alpha"][-1]),
                         float(mother["log10_temperature"][0]), float(mother["log10_temperature"][-1])]
        child_bounds = baseline["plot"]["display_window"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
        fig.subplots_adjust(left=.055, right=.985, bottom=.17, top=.83, wspace=.30)
        for ax, source, name, view in zip(axes, [mother, old, data], ["Mother window", "Child window", "Grandchild candidate"],
                                         [mother_bounds, child_bounds, bounds]):
            panel(ax, source, name, view)
        for ax, box in zip(axes[:2], [child_bounds, bounds]):
            ax.add_patch(Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2],
                                   fill=False, edgecolor="#00c0d0", lw=1.8))
        fig.suptitle("The same parameter plane, at three scales", fontsize=24, fontweight="bold", y=.96)
        fig.text(.055, .06, "Cyan boxes show the next panel. Physical coordinates and orientation are preserved.\n"
                 "All backgrounds are measured arrays with a common color scale; family names identify the selected nested views.", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"nested-window-overview.{ext}", dpi=180)
        plt.close(fig)
    return {"reference": plan["reference_scan"], "color_scale": scale, "interpolation": "nearest",
            "display_bounds": bounds, "root_evidence": root_evidence,
            "minimum_growth_rate": float(data["lyapunov"].min()), "maximum_growth_rate": float(data["lyapunov"].max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--workers", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--curves", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        plan = json.loads(args.plan.read_text())
        audit.require(plan["id"] in ("BSZ2", "GBZ1"), "unsupported rescan")
        bands = plan["cloud_option"]["bands"]
        audit.require(len(args.workers) == len(bands), "one exact worker per band required")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=audit.ROOT, text=True).strip()
        for source in ("tools/baby_shrimp_rescan_finish.py", "tools/baby_shrimp_finish.py"):
            blob = subprocess.check_output(["git", "show", commit+":"+source], cwd=audit.ROOT)
            audit.require(hashlib.sha256(blob).hexdigest() == audit.digest(audit.ROOT/source), "commit audit source before execution")
        rows = [audit.gate_band(plan, band, worker) for band, worker in zip(bands, args.workers)]
        for row in rows:
            rel = str(args.plan.resolve().relative_to(audit.ROOT))
            blob = subprocess.check_output(["git", "show", row["execution_commit"]+":"+rel], cwd=audit.ROOT)
            audit.require(hashlib.sha256(blob).hexdigest() == audit.digest(args.plan), "worker plan differs")
        audit.require(all(row["source_sha256"] == rows[0]["source_sha256"] for row in rows), "band scientific sources differ")
        import runpod_pod as rp
        ledger = [json.loads(s) for s in rp.LEDGER.read_text().splitlines()]
        present = {p["id"] for p in rp.inventory()["pods"]}
        for row in rows:
            pid = row["pod_id"]
            audit.require(any(e.get("event") == "terminated_verified" and e.get("pod_id") == pid for e in ledger)
                          and pid not in present and rp.pod_status(pid) is None, "pod termination unconfirmed")
            row["termination_verified"] = True
        combined = audit.DATA / plan["output"]
        if len(bands) > 1:
            audit.require(not combined.exists(), "preserve previous combined scan")
            command = [sys.executable, str(audit.ROOT / "tools/explore_llm_soft_loop.py"), "stitch", "--bands",
                       *["llm-soft-loop/pod64/bands/"+b["name"] for b in bands], "--out", str(Path(plan["output"]).relative_to("explore"))]
            env = {**os.environ, **{k:"1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
            subprocess.run(command, cwd=audit.ROOT, env=env, check=True, timeout=180)
        data = load_scan(combined / "scan.npz")
        audit.require(data["lyapunov"].shape == tuple(plan["resolution"]), "combined shape differs")
        for key, n in (("alpha", plan["resolution"][1]), ("log10_temperature", plan["resolution"][0])):
            audit.require(np.allclose(data[key], np.linspace(*plan[key], n), rtol=0, atol=1e-15), "combined axis differs")
        for key in ("lyapunov", "period", "entropy"):
            chunks = [load_scan(audit.DATA / "explore/llm-soft-loop/pod64/bands" / b["name"] / "scan.npz")[key] for b in bands]
            audit.require(np.array_equal(data[key], np.concatenate(chunks)), "combined field differs: "+key)
        plot = render(plan, data, args.out, args.curves)
        receipt = {"id": plan["id"]+"-image-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "all band, array, plan, source, archive and lifecycle gates passed",
                   "analysis_commit": commit, "analysis_source_sha256": audit.digest(__file__), "plan_sha256": audit.digest(args.plan),
                   "bands": rows, "totals": {"cells": int(data["period"].size), "period_counts": audit.counts(data["period"]),
                                              "lyapunov_range": [float(data["lyapunov"].min()), float(data["lyapunov"].max())]},
                   "combined_files": [audit.artifact(combined / name) for name in ("scan.npz", "scan.json")], "plot": plot,
                   "figures": [audit.artifact(p) for p in sorted(args.out.iterdir()) if p.suffix in (".png", ".pdf")],
                   "limits": "Finite-horizon fixed-start scan; labels are recurrence candidates through32 only. The fresh field is measured, not a saddle fit. Full-state roots have separate evidence. No complete boundary or exact superstable intersection is certified.", "cost": "see private ledger"}
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"status": receipt["status"], "totals": receipt["totals"]}))
    except BaseException as error:
        (args.out / "failed.json").write_text(json.dumps({"type": type(error).__name__, "error": str(error)})+"\n")
        raise


if __name__ == "__main__":
    main()
