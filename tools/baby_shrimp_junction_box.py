"""Locate the proposed junction rescan within the audited BSZ1 image.

Display only: the same saved data, color mapping and orientation as A finer look.
The enlarged panel adds no observations and does not decide crossing/avoidance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    receipt_path = ROOT / "docs/baby-shrimp-zoom/zoom-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    scan = next(v for v in receipt["combined_files"] if v["path"].endswith("scan.npz"))
    source = DATA / scan["path"]
    if digest(source) != scan["sha256"]:
        raise ValueError("audited fine scan changed")
    plan_path = ROOT / "docs/baby-shrimp-zoom/crossing-curve-plan.json"
    plan = json.loads(plan_path.read_text())
    center = np.asarray(plan["start"])
    scales = np.asarray(plan["coordinate_scales"])
    # Axis-aligned physical coordinates, never the rotated Hessian coordinates.
    half = .6 * scales
    box = [float(center[0]-half[0]), float(center[0]+half[0]),
           float(center[1]-half[1]), float(center[1]+half[1])]
    context = [float(center[0]-6*scales[0]), float(center[0]+6*scales[0]),
               float(center[1]-6*scales[1]), float(center[1]+6*scales[1])]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    with np.load(source, allow_pickle=False) as d:
        l, p = d["lyapunov"], d["period"]
        a, t = d["alpha"], d["log10_temperature"]
        cap = receipt["plot"]["shared_gray_contraction_cap"]
        gray = 1 - .9*np.clip(np.maximum(-l, 0)/cap, 0, 1)**.5
        rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
        hot = (l > .001) & (p == 0)
        rgba[hot] = plt.get_cmap("plasma")(Normalize(0, receipt["plot"]["shared_positive_vmax"])(l[hot]))
        da, dt = np.diff(a).mean(), np.diff(t).mean()
        extent = [a[0]-da/2, a[-1]+da/2, t[0]-dt/2, t[-1]+dt/2]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11})
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8.2))
    fig.subplots_adjust(left=.075, right=.975, bottom=.17, top=.82, wspace=.28)
    whole = receipt["plot"]["display_window"]
    for ax, bounds, title in zip(axes, [whole, context],
                                ["The existing ‘A finer look’ window", "Same junction, enlarged 7×"]):
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest", aspect="auto")
        ax.set(xlim=bounds[:2], ylim=bounds[2:], title=title,
               xlabel="Steering strength α  →", ylabel="log₁₀ temperature  →")
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        ax.tick_params(labelsize=9)
        ax.add_patch(Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2],
                               fill=False, edgecolor="#ff8c00", lw=2.3, zorder=5))
    axes[0].add_patch(Rectangle((context[0], context[2]), context[1]-context[0], context[3]-context[2],
                               fill=False, edgecolor="#00b9cf", lw=1.7, ls="--"))
    axes[0].annotate("Cyan box → right panel", (context[0], context[3]),
                     xytext=(-135, 25), textcoords="offset points", fontsize=10,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3),
                     arrowprops=dict(arrowstyle="-", color="#0094a8", lw=1.2))
    axes[1].annotate("Proposed new scan", (box[1], box[3]), xytext=(20, 48),
                     textcoords="offset points", fontsize=11, color="#8a4100",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.95, pad=4),
                     arrowprops=dict(arrowstyle="->", color="#e87500", lw=1.5))
    fig.suptitle("Where the junction zoom sits", x=.075, ha="left", fontsize=25, fontweight="bold", y=.96)
    fig.text(.075, .885, "Orange rectangle = new scan target in both panels.  Same orientation; no rotation or reflection.", fontsize=12)
    fig.text(.075, .065,
             "Both backgrounds use the completed 512 × 512 scan with its original color scale and nearest-cell rendering.\n"
             "The enlarged pixels are existing measurements; this locator does not add detail or decide whether the tracks cross.", fontsize=10)
    fig.savefig(args.out / "junction-zoom-location.png", dpi=180)
    fig.savefig(args.out / "junction-zoom-location.pdf")
    plt.close(fig)
    record = {"source": scan, "source_receipt_sha256": digest(receipt_path),
              "anchor_plan_sha256": digest(plan_path), "center": center.tolist(),
              "target_alpha": box[:2], "target_log10_temperature": box[2:],
              "context_window": context, "original_window": whole,
              "original_spacing": [float(da), float(dt)], "orientation": "alpha right; log10 temperature up",
              "new_measurements": False, "interpolation": "nearest",
              "figures": [{"name": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                          for p in sorted(args.out.glob("junction-zoom-location.*"))]}
    (args.out / "receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
