"""Summarize a completed EXP-029 attempt and draw its maps.

Reads saved evidence only after verifying the receipt. Labels every computed
cell, extracts the dominant transverse Floquet modulus of periodic cells,
compares starts A and B where both were computed, and draws every cell of the
start-A maps. Refuses incomplete attempts and existing outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIVERGED, EQUILIBRIUM, PERIODIC, CHAOTIC, UNRESOLVED, NOT_COMPUTED = -1, 0, 1, 2, 3, -2
NAMES = {DIVERGED: "diverged", EQUILIBRIUM: "equilibrium", PERIODIC: "periodic",
         CHAOTIC: "chaotic_candidate", UNRESOLVED: "unresolved", NOT_COMPUTED: "not_computed"}
COLORS = {EQUILIBRIUM: "#2a78d6", PERIODIC: "#eb6834", CHAOTIC: "#1baf7a",
          UNRESOLVED: "#e1e0d9", DIVERGED: "#52514e", NOT_COMPUTED: "#ffffff"}
PERIOD_RAMP = ["#b7d3f6", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
               "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c23434"]
SURFACE, INK, MUTED, NEUTRAL = "#fcfcfb", "#0b0b0b", "#898781", "#f0efec"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def labels(saved, chaos_threshold):
    out = np.full(saved["period"].shape, UNRESOLVED, dtype=np.int64)
    out[saved["period"] >= 1] = PERIODIC
    out[(saved["period"] == 0) & saved["equilibrium"]] = EQUILIBRIUM
    out[(saved["period"] == 0) & ~saved["equilibrium"] & (saved["ftle"] > chaos_threshold)] = CHAOTIC
    out[saved["diverged"].astype(bool)] = DIVERGED
    out[~saved["computed"]] = NOT_COMPUTED
    return out


def transverse_modulus(saved):
    """Largest multiplier modulus after removing the one closest to 1."""
    mult = saved["multiplier_real"] + 1j * saved["multiplier_imag"]
    distance = np.abs(mult - 1.0)
    neutral = np.nanargmin(np.where(np.isfinite(distance), distance, np.inf), axis=-1)
    moduli = np.abs(mult)
    mask = np.ones_like(moduli, dtype=bool)
    np.put_along_axis(mask, neutral[..., None], False, axis=-1)
    transverse = np.where(mask, moduli, -np.inf).max(axis=-1)
    neutral_error = np.abs(np.take_along_axis(mult, neutral[..., None], axis=-1)[..., 0] - 1.0)
    finite = np.all(np.isfinite(mult), axis=-1)
    return np.where(finite, transverse, np.nan), np.where(finite, neutral_error, np.nan)


def summarize(saved, chaos_threshold):
    lab = labels(saved, chaos_threshold)
    trans, neutral_error = transverse_modulus(saved)
    periodic = lab == PERIODIC
    periods, freq = np.unique(saved["period"][periodic], return_counts=True)
    attracting = periodic & (trans < 1.0)
    return lab, trans, {
        "counts": {NAMES[k]: int((lab == k).sum()) for k in NAMES},
        "period_histogram": {str(int(p)): int(f) for p, f in zip(periods, freq)},
        "periodic_transverse_lt_1": int(attracting.sum()),
        "transverse_quantiles": None if not periodic.any() else
            [float(q) for q in np.nanquantile(trans[periodic], [0.05, 0.25, 0.5, 0.75, 0.95])],
        "max_neutral_multiplier_error": None if not periodic.any() else float(np.nanmax(neutral_error[periodic])),
        "max_floquet_closure": None if not periodic.any() else float(np.nanmax(saved["floquet_closure"][periodic])),
    }


def draw(saved, lab, trans, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    ga, aa = saved["gamma_axis"], saved["alpha_axis"]
    extent = [aa[0], aa[-1], ga[0], ga[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    order = [DIVERGED, EQUILIBRIUM, PERIODIC, CHAOTIC, UNRESOLVED]
    index = np.zeros(lab.shape, dtype=int)
    for i, code in enumerate(order):
        index[lab == code] = i
    axes[0].imshow(index, origin="lower", extent=extent, aspect="auto", interpolation="nearest",
                   cmap=ListedColormap([COLORS[c] for c in order]), vmin=-0.5, vmax=len(order) - 0.5)
    axes[0].set_title("A  Autonomous class (every cell)", loc="left", fontsize=10, color=INK)
    axes[0].legend(handles=[Patch(color=COLORS[c], label=NAMES[c].replace("_", " ")) for c in order],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7, frameon=False)
    periodic = lab == PERIODIC
    levels = np.log2(np.where(periodic, saved["period"], 1).astype(float))
    cmap = ListedColormap(PERIOD_RAMP).with_extremes(bad=NEUTRAL)
    bounds = np.linspace(0, 5, len(PERIOD_RAMP) + 1)
    image = axes[1].imshow(np.ma.masked_where(~periodic, levels), origin="lower", extent=extent,
                           aspect="auto", interpolation="nearest", cmap=cmap,
                           norm=BoundaryNorm(bounds, cmap.N))
    bar = fig.colorbar(image, ax=axes[1], shrink=0.85)
    bar.set_ticks([0, 1, 2, 3, 4, 5])
    bar.set_ticklabels(["1", "2", "4", "8", "16", "32"])
    bar.set_label("x3 maxima per screened cycle", fontsize=8, color=INK)
    axes[1].set_title("B  Screened period (periodic cells)", loc="left", fontsize=10, color=INK)
    ftle = np.where(saved["computed"], saved["ftle"], np.nan)
    finite = np.isfinite(ftle)
    limit = max(float(np.quantile(np.abs(ftle[finite]), 0.99)) if finite.any() else 1.0, 1e-6)
    diverging = LinearSegmentedColormap.from_list("ftle", DIVERGING).with_extremes(bad=COLORS[DIVERGED])
    image = axes[2].imshow(np.ma.masked_invalid(ftle), origin="lower", extent=extent, aspect="auto",
                           interpolation="nearest", cmap=diverging,
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
    bar = fig.colorbar(image, ax=axes[2], shrink=0.85, extend="both")
    bar.set_label("finite-time exponent per time unit", fontsize=8, color=INK)
    axes[2].set_title("C  Largest finite-time exponent", loc="left", fontsize=10, color=INK)
    seq = LinearSegmentedColormap.from_list("mu", PERIOD_RAMP).with_extremes(bad=NEUTRAL)
    image = axes[3].imshow(np.ma.masked_where(~periodic | ~np.isfinite(trans), trans), origin="lower",
                           extent=extent, aspect="auto", interpolation="nearest", cmap=seq, vmin=0, vmax=1)
    bar = fig.colorbar(image, ax=axes[3], shrink=0.85, extend="max")
    bar.set_label("dominant transverse multiplier modulus", fontsize=8, color=INK)
    axes[3].set_title("D  Transverse contraction (periodic cells)", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.set_xlabel("alpha (connection 3 to 1)", fontsize=8, color=MUTED)
        ax.set_ylabel("gamma (connection 1 to 3)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/track-b/EXP-029-plan.json"))
    parser.add_argument("--raw", default=None)
    parser.add_argument("--summary", default=str(ROOT / "docs/EXP-029-summary.json"))
    parser.add_argument("--figures", default=None)
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    raw = Path(args.raw) if args.raw else (ROOT / plan["output"]["raw_root"] / plan["output"]["attempt"]).resolve()
    figures = Path(args.figures) if args.figures else (
        ROOT / "../neural-shrimp-data/figures/EXP-029" / plan["output"]["attempt"]).resolve()
    receipt = json.loads((raw / "receipt.json").read_text())
    bad = [f["path"] for f in receipt["files"] if sha256_file(raw / f["path"]) != f["sha256"]]
    if bad:
        raise SystemExit(f"receipt mismatch: {bad[:5]}")
    if receipt["status"] != "complete":
        raise SystemExit(f"attempt status {receipt['status']!r}; analysis requires a complete attempt")
    if receipt["plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest():
        raise SystemExit("receipt plan hash differs from the plan file")
    if Path(args.summary).exists() or figures.exists():
        raise SystemExit("summary or figure directory already exists; refusing to overwrite")
    figures.mkdir(parents=True)
    threshold = plan["labels"]["chaos_threshold_per_time"]
    jobs, labelled = {}, {}
    for path in sorted((raw / "maps").glob("*.npz")):
        saved = np.load(path)
        lab, trans, info = summarize(saved, threshold)
        jobs[path.stem], labelled[path.stem] = info, (saved, lab)
        if path.stem.endswith("-A"):
            figure = figures / f"{path.stem}.png"
            draw(saved, lab, trans, f"EXP-029 {path.stem}: autonomous network, exploratory survey", figure)
            info["figure"] = {"path": str(figure), "sha256": sha256_file(figure)}
    agreement = {}
    for spec in plan["maps"]:
        a, b = labelled.get(f"{spec['map_id']}-A"), labelled.get(f"{spec['map_id']}-B")
        if a is None or b is None:
            continue
        both = a[0]["computed"] & b[0]["computed"]
        same_class = (a[1] == b[1]) & both
        same_period = same_class & ((a[1] != PERIODIC) | (a[0]["period"] == b[0]["period"]))
        agreement[spec["map_id"]] = {"compared": int(both.sum()), "same_class": int(same_class.sum()),
                                     "same_class_and_period": int(same_period.sum())}
    summary = {"experiment": "EXP-029", "label": plan["label"], "plan_sha256": receipt["plan_sha256"],
               "receipt_sha256": sha256_file(raw / "receipt.json"), "git": receipt["git"],
               "jobs": jobs, "start_agreement": agreement}
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jobs": list(jobs), "start_agreement": agreement}, indent=2))


if __name__ == "__main__":
    main()
