"""Summarize a completed EXP-028 attempt and draw its maps.

Reads only saved raw evidence after verifying the receipt inventory. Writes a
compact machine summary (counts, period histograms, components, SD cross
tabulation, certificates, chaos diagnostics and the frozen stage-A1 gate)
and one figure per map. Every cell is plotted; nothing is filtered.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import (  # noqa: E402
    STATUS_CHAOTIC, STATUS_DIVERGENT, STATUS_FIXED, STATUS_NAMES, STATUS_PERIODIC,
    STATUS_UNRESOLVED, connected_components,
)

# Validated categorical slots (all-pairs, light surface) plus neutrals.
CLASS_COLORS = {STATUS_FIXED: "#2a78d6", STATUS_PERIODIC: "#eb6834", STATUS_CHAOTIC: "#1baf7a",
                STATUS_DIVERGENT: "#52514e", STATUS_UNRESOLVED: "#e1e0d9"}
CLASS_LABELS = {STATUS_FIXED: "fixed-point candidate", STATUS_PERIODIC: "periodic candidate (p >= 2)",
                STATUS_CHAOTIC: "chaotic candidate", STATUS_DIVERGENT: "divergent",
                STATUS_UNRESOLVED: "unresolved"}
PERIOD_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
               "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c23434"]
SURFACE, INK, MUTED, NEUTRAL = "#fcfcfb", "#0b0b0b", "#898781", "#f0efec"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_receipt(raw):
    receipt = json.loads((raw / "receipt.json").read_text())
    bad = [item["path"] for item in receipt["files"]
           if sha256_file(raw / item["path"]) != item["sha256"]]
    if bad:
        raise SystemExit(f"receipt hash mismatch: {bad[:5]}")
    return receipt


def map_summary(saved, min_size):
    status, period = saved["status"], saved["period"]
    counts = {STATUS_NAMES[k]: int((status == k).sum()) for k in STATUS_NAMES}
    periodic = (status == STATUS_PERIODIC) & (period >= 2)
    values, freq = np.unique(period[periodic], return_counts=True)
    labels = connected_components(period, periodic)
    components = {}
    for component in range(int(labels.max()) + 1):
        members = labels == component
        size = int(members.sum())
        p = int(period[members][0])
        entry = components.setdefault(str(p), {"components": 0, "components_min_size": 0,
                                               "largest": 0})
        entry["components"] += 1
        entry["components_min_size"] += int(size >= min_size)
        entry["largest"] = max(entry["largest"], size)
    ftle = saved["ftle"][status == STATUS_CHAOTIC]
    sd = {}
    for key in saved.files:
        if key.startswith("sd_convergent_"):
            conv = saved[key]
            sd[key.removeprefix("sd_convergent_")] = {
                "sd_convergent": {STATUS_NAMES[k]: int(((status == k) & conv).sum()) for k in STATUS_NAMES},
                "sd_divergent": {STATUS_NAMES[k]: int(((status == k) & ~conv).sum()) for k in STATUS_NAMES},
            }
    return {"cells": int(status.size), "counts": counts,
            "period_histogram": {str(int(v)): int(f) for v, f in zip(values, freq)},
            "components_by_period": components,
            "chaotic_ftle": None if len(ftle) == 0 else {
                "median": float(np.median(ftle)), "q25": float(np.quantile(ftle, 0.25)),
                "q75": float(np.quantile(ftle, 0.75)), "max": float(np.max(ftle))},
            "sd_crosstab": sd}


def gate(certificates, rule):
    qualifying = []
    for record in certificates:
        cert = record["certificate"]
        if (record["kind"] == "periodic" and cert.get("verified") and
                (cert.get("attracting") or not rule["requires_attracting"]) and
                (cert.get("primitive_period") or 0) >= rule["min_period"] and
                (record["borders_chaos"] or not rule["requires_borders_chaos"])):
            qualifying.append({"map_id": record["map_id"], "period": cert["primitive_period"],
                               "component_size": record["size"],
                               "spectral_radius": cert["spectral_radius"],
                               "lr_hidden": record["lr_hidden"], "lr_output": record["lr_output"]})
    return {"positive": bool(qualifying), "qualifying": qualifying}


def draw_map(saved, title, path, sd_key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    status, period, ftle = saved["status"], saved["period"], saved["ftle"]
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    extent = [lo[0], lo[-1], lh[0], lh[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    order = [STATUS_DIVERGENT, STATUS_FIXED, STATUS_PERIODIC, STATUS_CHAOTIC, STATUS_UNRESOLVED]
    index = np.zeros_like(status)
    for i, code in enumerate(order):
        index[status == code] = i
    ax = axes[0]
    ax.imshow(index, origin="lower", extent=extent, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([CLASS_COLORS[c] for c in order]), vmin=-0.5, vmax=len(order) - 0.5)
    ax.set_title("A  Dynamical class (every cell)", loc="left", fontsize=10, color=INK)
    ax.legend(handles=[Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    ax = axes[1]
    periodic = (status == STATUS_PERIODIC) & (period >= 2)
    levels = np.log2(np.where(periodic, period, 1).astype(float))
    shown = np.ma.masked_where(~periodic, levels)
    cmap = ListedColormap(PERIOD_RAMP).with_extremes(bad=NEUTRAL)
    bounds = np.linspace(1, 6, len(PERIOD_RAMP) + 1)
    image = ax.imshow(shown, origin="lower", extent=extent, aspect="auto", interpolation="nearest",
                      cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
    bar = fig.colorbar(image, ax=ax, shrink=0.85)
    bar.set_ticks([1, 2, 3, 4, 5, 6])
    bar.set_ticklabels(["2", "4", "8", "16", "32", "64"])
    bar.set_label("primitive period candidate", fontsize=8, color=INK)
    ax.set_title("B  Recurrence period (periodic cells only)", loc="left", fontsize=10, color=INK)
    if not periodic.any():
        ax.text(0.5, 0.5, "no periodic candidates", transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=MUTED)
    ax = axes[2]
    finite = np.isfinite(ftle)
    limit = float(np.quantile(np.abs(ftle[finite]), 0.99)) if finite.any() else 1.0
    limit = max(limit, 1e-6)
    diverging = LinearSegmentedColormap.from_list("ftle", DIVERGING).with_extremes(
        bad=CLASS_COLORS[STATUS_DIVERGENT])
    image = ax.imshow(np.ma.masked_invalid(ftle), origin="lower", extent=extent, aspect="auto",
                      interpolation="nearest", cmap=diverging,
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    bar = fig.colorbar(image, ax=ax, shrink=0.85, extend="both")
    bar.set_label("finite-time exponent per update", fontsize=8, color=INK)
    ax.set_title("C  Largest finite-time exponent", loc="left", fontsize=10, color=INK)
    ax = axes[3]
    conv = saved[sd_key]
    # Neutrals only: the binary score is not one of panel A's classes.
    ax.imshow(conv.astype(int), origin="lower", extent=extent, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([CLASS_COLORS[STATUS_DIVERGENT], NEUTRAL]), vmin=-0.5, vmax=1.5)
    horizon = sd_key.removeprefix("sd_convergent_")
    ax.set_title(f"D  Published binary score, {horizon} updates", loc="left", fontsize=10, color=INK)
    ax.legend(handles=[Patch(facecolor=NEUTRAL, edgecolor=MUTED, label="scored convergent"),
                       Patch(color=CLASS_COLORS[STATUS_DIVERGENT], label="scored divergent")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    for ax in axes:
        ax.set_xlabel("log10 output-layer learning rate", fontsize=8, color=MUTED)
        ax.set_ylabel("log10 hidden-layer learning rate", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_facecolor(SURFACE)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/trainability/EXP-028-plan.json"))
    parser.add_argument("--raw", default=None)
    parser.add_argument("--summary", default=str(ROOT / "docs/EXP-028-summary.json"))
    parser.add_argument("--figures", default=None)
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    raw = Path(args.raw) if args.raw else (ROOT / plan["output"]["raw_root"] / plan["output"]["attempt"]).resolve()
    figures = Path(args.figures) if args.figures else (
        ROOT / "../neural-shrimp-data/figures/EXP-028" / plan["output"]["attempt"]).resolve()
    receipt = verify_receipt(raw)
    if receipt["plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest():
        raise SystemExit("receipt plan hash differs from the plan file")
    figures.mkdir(parents=True, exist_ok=True)
    maps = {}
    min_size = plan["certification"]["min_component_size"]
    sd_key = f"sd_convergent_{plan['iteration']['sd_horizons'][1]}"
    for stage in ("stage1", "stage2"):
        for path in sorted((raw / stage).glob("*.npz")):
            saved = np.load(path)
            map_id = path.stem
            maps[map_id] = map_summary(saved, min_size)
            maps[map_id]["log10_hidden_range"] = [float(saved["log10_hidden"][0]), float(saved["log10_hidden"][-1])]
            maps[map_id]["log10_output_range"] = [float(saved["log10_output"][0]), float(saved["log10_output"][-1])]
            figure = figures / f"{map_id}.png"
            steps = plan["iteration"]["steps"]
            draw_map(saved, f"EXP-028 {map_id}: {steps:,} full-batch updates per cell (exploratory pilot)",
                     figure, sd_key)
            maps[map_id]["figure"] = {"path": str(figure), "sha256": sha256_file(figure)}
    zooms = {p.stem: json.loads(p.read_text()) for p in sorted((raw / "stage1").glob("*-zoom.json"))}
    certificates = json.loads((raw / "stage3" / "certificates.json").read_text())
    diagnostics = json.loads((raw / "stage3" / "chaos-diagnostics.json").read_text())
    cert_table = [{"map_id": r["map_id"], "kind": r["kind"], "component_period": r["period"],
                   "size": r["size"], "borders_chaos": r["borders_chaos"],
                   "lr_hidden": r["lr_hidden"], "lr_output": r["lr_output"],
                   "verified": r["certificate"]["verified"], "attracting": r["certificate"]["attracting"],
                   "primitive_period": r["certificate"]["primitive_period"],
                   "spectral_radius": r["certificate"]["spectral_radius"],
                   "stability": r["certificate"]["stability"], "reason": r["certificate"].get("reason")}
                  for r in certificates]
    chaos = {}
    for record in diagnostics:
        entry = chaos.setdefault(record["map_id"], {"rerun": 0, "agree": 0})
        entry["rerun"] += 1
        entry["agree"] += int(record["agrees"])
    summary = {
        "experiment": "EXP-028", "label": plan["label"],
        "plan_sha256": receipt["plan_sha256"],
        "receipt_sha256": sha256_file(raw / "receipt.json"),
        "raw": str(raw), "git": receipt["git"], "status": receipt["status"],
        "maps": maps, "zoom_windows": {k: {"window": v["window"]} for k, v in zooms.items()},
        "certificates": cert_table, "chaos_diagnostics": chaos,
        "gate": gate(certificates, plan["gate"]),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": summary["gate"]["positive"], "maps": len(maps),
                      "certificates": len(cert_table)}, indent=2))


if __name__ == "__main__":
    main()
