"""Offline phase portraits and six-step return DOTS from three audited P18 cycles.

This samples no new map values and fits no return curve. The beta=0 parent's
missing return-state array remains an explicit recorded-closure limitation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np

import baby_shrimp_memory_orbit_alignment as audit

ROOT, DATA = audit.ROOT, audit.DATA
SOURCE = "tools/plot_baby_shrimp_memory_orbits.py"
SOURCES = (SOURCE, audit.SOURCE)
BETAS = (0., 0.006448531026086656, .007)
LABELS = ("β = 0", "β* = 0.006448531", "β = 0.007")
require = audit.require


def token_period(tokens):
    tokens = np.asarray(tokens)
    require(tokens.shape == (18,) and np.issubdtype(tokens.dtype, np.integer), "18 integer token IDs required")
    return next(q for q in range(1, 19) if 18 % q == 0 and np.array_equal(tokens, np.tile(tokens[:q], 18 // q)))


def closure_record(orbit, point, endpoint=None):
    """Distinguish saved-endpoint closure from the parent's recorded summaries."""
    orbit = np.asarray(orbit)
    require(orbit.shape == (18, 896) and orbit.dtype == np.float64 and np.isfinite(orbit).all(),
            "finite float64 18 by 896 orbit required")
    norm = np.linalg.norm(orbit[0])
    require(norm > 0, "zero initial state")
    lower = np.linalg.norm(orbit[1:] - orbit[0], axis=1) / norm
    primitive = point["primitive"]
    recorded = float(primitive["all_returns"][-1])
    require(primitive["closed"] and primitive["no_lower_period"] and 0 <= recorded < 1e-11
            and np.min(lower) > 1e-8
            and np.allclose(lower, primitive["all_returns"][:-1], rtol=1e-10, atol=1e-13),
            "saved primitive-order/closure evidence differs")
    require(token_period(point["tokens"]) == 6, "token cycle must have primitive period six")
    measured = None
    if endpoint is not None:
        endpoint = np.asarray(endpoint)
        require(endpoint.size == 896 and endpoint.dtype == np.float64 and np.isfinite(endpoint).all(),
                "invalid saved return state")
        measured = float(np.linalg.norm(endpoint.reshape(-1) - orbit[0]) / norm)
        require(measured < 1e-12 and np.isclose(measured, recorded, rtol=1e-5, atol=1e-14),
                "saved endpoint closure differs")
    return {"full_state_period": 18, "token_period": 6, "recorded_relative_closure": recorded,
            "endpoint_relative_closure": measured, "minimum_lower_relative_return": float(np.min(lower)),
            "closure_scope": ("Recomputed from saved return-state.npy; no fresh model evaluation."
                              if endpoint is not None else
                              "Recorded closure only: parent has no saved return-state.npy; not independently recomputed."),
            "endpoint_available": endpoint is not None}


def joint_pca(blocks, components=2):
    """One origin and basis shared by every case; never normalize cases separately."""
    blocks = [np.asarray(block, dtype=np.float64) for block in blocks]
    require(blocks and all(b.ndim == 2 and b.shape[1] == blocks[0].shape[1] and np.isfinite(b).all()
                           for b in blocks), "invalid joint-PCA inputs")
    pooled = np.concatenate(blocks)
    mean = pooled.mean(axis=0)
    _, singular, basis = np.linalg.svd(pooled - mean, full_matrices=False)
    require(len(singular) >= components and singular[0] > 0, "collapsed joint-PCA sample")
    basis = basis[:components].copy()
    # Fix a reproducible sign convention; this does not change temporal order.
    for axis in basis:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    fractions = singular[:components] ** 2 / np.sum(singular ** 2)
    return mean, basis, fractions, [(b - mean) @ basis.T for b in blocks]


def six_step_pairs(coordinates, phase):
    """Exactly three ordered visits and three returns, including 12->0 wrap."""
    coordinates = np.asarray(coordinates)
    require(coordinates.shape == (18,) and np.isfinite(coordinates).all(), "18 finite scalar coordinates required")
    require(isinstance(phase, int) and 0 <= phase < 6, "phase must be 0 through 5")
    source = np.arange(phase, 18, 6)
    target = (source + 6) % 18
    return source, target, np.column_stack([coordinates[source], coordinates[target]])


def align_samples(samples):
    require(len(samples) == 3 and tuple(s["summary"]["parameter"] for s in samples) == BETAS,
            "exact beta=0, beta*, beta=.007 cases required in order")
    aligned, rows = [np.array(samples[0]["orbit"], copy=True)], []
    require(token_period(samples[0]["tokens"]) == 6, "reference token period differs")
    for index, sample in enumerate(samples):
        comparison = audit.compare_orbits(samples[0], sample)
        require(comparison["alignment_margin"] > 1e-10, "ambiguous full-state cyclic alignment")
        require(comparison["tokens_match_at_best_full_state_alignment"], "aligned token phases differ")
        if index:
            aligned.append(np.roll(sample["orbit"], comparison["best_right_roll"], axis=0))
        rows.append(comparison)
    return aligned, rows


def derive(samples):
    aligned, comparisons = align_samples(samples)
    mean, basis, fractions, portraits = joint_pca(aligned)
    arrays = {"portrait_mean": mean, "portrait_basis": basis,
              "portrait_explained_variance": fractions, "portrait_coordinates": np.stack(portraits)}
    phase_rows = []
    for phase in range(6):
        origin, axes, share, _ = joint_pca([orbit[phase::6] for orbit in aligned])
        all_coordinates = [(orbit - origin) @ axes[0] for orbit in aligned]
        pairs = [six_step_pairs(coordinates, phase) for coordinates in all_coordinates]
        arrays[f"phase_{phase}_mean"] = origin
        arrays[f"phase_{phase}_basis"] = axes
        arrays[f"phase_{phase}_pairs"] = np.stack([p[2] for p in pairs])
        arrays[f"phase_{phase}_source_indices"] = pairs[0][0]
        arrays[f"phase_{phase}_target_indices"] = pairs[0][1]
        phase_rows.append({"phase": phase, "common_pc1_explained_variance": float(share[0]),
                           "source_indices": pairs[0][0].tolist(), "target_indices": pairs[0][1].tolist(),
                           "return_pairs_per_case": 3})
    return arrays, {"alignment_to_beta_zero": comparisons,
                    "portrait_explained_variance": fractions.tolist(), "phases": phase_rows,
                    "coordinate_method": "Joint PCA over all three cases, one shared origin/basis for the full portrait and one shared origin/basis per six-step phase. No separate case centering, scaling or curve fit."}


def load_cases(receipt_pairs, *, data=DATA):
    bundles = [audit.load_receipt(path, sha, data=data) for path, sha in receipt_pairs]
    require(len({str(b["path"]) for b in bundles}) == len(bundles), "duplicate audit receipt")
    require(all(b["scientific_metadata"] == bundles[0]["scientific_metadata"] for b in bundles),
            "scientific metadata differs")
    selected, records = [], []
    for beta in BETAS:
        candidates = [(b, s) for b in bundles for s in b["samples"] if s["summary"]["parameter"] == beta]
        require(len(candidates) == 1, f"expected exactly one audited accepted cycle at beta={beta}")
        bundle, sample = candidates[0]
        directory = Path(data) / sample["summary"]["orbit_path"]
        point = audit.read(directory.with_name("point.json"))
        endpoint = directory.with_name("return-state.npy")
        closure = closure_record(sample["orbit"], point,
                                 np.load(endpoint, allow_pickle=False) if endpoint.exists() else None)
        selected.append(sample)
        records.append({**sample["summary"], **closure,
                        "execution_source_commit": bundle["receipt"]["execution_source_commit"],
                        "input_audit_commit": bundle["receipt"]["analysis_commit"],
                        "endpoint_sha256": audit.digest(endpoint) if endpoint.exists() else None})
    return bundles, selected, records


def draw(arrays, derived, records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#2166ac", "#d95f02", "#1b7837")
    phase_colors = ("#332288", "#117733", "#44aa99", "#ddcc77", "#cc6677", "#882255")
    portrait = arrays["portrait_coordinates"]
    lo, hi = portrait.min(axis=(0, 1)), portrait.max(axis=(0, 1))
    pad = np.maximum((hi - lo) * .1, 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for case, ax in enumerate(axes):
        for phase in range(6):
            points = portrait[case, phase::6]
            ax.scatter(points[:, 0], points[:, 1], c=phase_colors[phase], s=45,
                       label=f"Phase {phase}", edgecolors="white", linewidths=.4)
        closure = records[case]
        value = closure["endpoint_relative_closure"] if closure["endpoint_available"] else closure["recorded_relative_closure"]
        mode = "saved endpoint" if closure["endpoint_available"] else "recorded only"
        ax.set(title=f"{LABELS[case]}\nRelative closure {value:.2g} ({mode})",
               xlabel="Common state PC1", ylabel="Common state PC2",
               xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
        ax.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("The saved 18-state orbit at three memory levels", fontsize=19)
    explained = 100 * sum(derived["portrait_explained_variance"])
    fig.text(.5, .09, f"All cases: full-state period 18; token period 6. Shared two-dimensional PCA explains {explained:.2f}% of pooled variance.", ha="center", fontsize=10)
    fig.text(.5, .045, "These are projected saved cycles, not sampled neighborhoods. β = 0 has no archived return state; its closure is recorded evidence.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.98, bottom=.22, top=.79, wspace=.27)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"memory-orbit-portrait.{ext}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 10))
    markers = ("o", "s", "^")
    for phase, ax in enumerate(axes.flat):
        pairs = arrays[f"phase_{phase}_pairs"]
        low, high = float(pairs.min()), float(pairs.max())
        padding = max((high-low)*.12, 1e-12)
        limits = (low-padding, high+padding)
        ax.plot(limits, limits, color="#bdbdbd", lw=.7, zorder=0)
        for case in range(3):
            ax.scatter(pairs[case, :, 0], pairs[case, :, 1], s=65, marker=markers[case],
                       edgecolors=colors[case], facecolors="none", linewidths=1.4, label=LABELS[case])
        share = derived["phases"][phase]["common_pc1_explained_variance"] * 100
        ax.set(title=f"Phase {phase} · common PC1: {share:.1f}%", xlabel="Position at this visit", ylabel="Position six steps later",
               xlim=limits, ylim=limits)
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), ncol=3, frameon=False)
    fig.suptitle("Six-step returns: three discrete points in each band", fontsize=19, y=.975)
    fig.text(.5, .066, "Each case uses one common phase coordinate: visits k, k+6, k+12, including the last-to-first return.", ha="center", fontsize=10)
    fig.text(.5, .035, "Only saved period-18 cycles. No continuous curve, turning-point count, unimodality or invariant-manifold claim follows.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, bottom=.14, top=.855, wspace=.36, hspace=.35)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"memory-six-step-return-dots.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists(), "new output directory outside Git under data store required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {name: audit.digest(ROOT / name) for name in SOURCES}
    for name, sha in source_hashes.items():
        require(hashlib.sha256(audit.git_bytes(commit, name)).hexdigest() == sha, "commit plotting source before execution: " + name)
    bundles, samples, records = load_cases(args.receipt)
    for bundle in bundles:
        require(not out.is_relative_to(bundle["run"]), "output must not alter an immutable raw attempt")
    arrays, derived = derive(samples)
    for bundle in bundles:
        audit.verify_manifest(bundle["run"], bundle["receipt"]["manifest"])
        require(audit.digest(bundle["path"]) == bundle["sha256"], "audit receipt changed during analysis")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "plot-data.npz", **arrays)
    draw(arrays, derived, records, out)
    common = {"id": "BSC3-memory-saved-orbit-comparison", "created_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_commit": commit, "analysis_source_sha256": source_hashes, "threads": 1,
              "inputs": [{"path": str(b["path"].relative_to(DATA)), "sha256": b["sha256"],
                          "complete": b["receipt"]["complete"], "status": b["receipt"]["status"]} for b in bundles],
              "selection": "Exactly the accepted beta=0, beta*=0.006448531026086656 and beta=.007 saddle cycles from the supplied frozen audits.",
              "cases": records, "derivation": derived, "plot_data_sha256": audit.digest(out / "plot-data.npz"),
              "scope": "Saved-orbit comparison only; no model execution or continuous return-map sampling. Cyclic wrap uses the audited periodic-order evidence; the parent beta=0 lacks a saved final return state.",
              "visual_review": "pending"}
    specs = {"memory-orbit-portrait": "Three panels show the same eighteen-step full-state cycle at zero memory, the numerical trace-height root, and memory 0.007. Six colors identify phases modulo six, with three visits per phase. All cases share one joint two-dimensional PCA frame. Per-case closure labels distinguish saved endpoint checks from the parent's recorded-only closure.",
             "memory-six-step-return-dots": "Six panels, one for each phase modulo six, compare exactly three six-step return pairs from each of three saved period-eighteen cycles. Blue circles, orange squares and green triangles distinguish memory levels. Each phase uses a single jointly fitted principal coordinate across cases. The gray diagonal is only a reference; no continuous return curve is drawn."}
    for stem, alt in specs.items():
        record = {**common, "figure": stem, "alt_text": alt, "interval_semantics": "No confidence or uncertainty intervals; these are numerical saved-cycle coordinates.",
                  "artifacts": [{"path": str(out / f"{stem}.{ext}"), "sha256": audit.digest(out / f"{stem}.{ext}")} for ext in ("png", "pdf")]}
        (out / f"{stem}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    artifacts = [{"path": p.name, "sha256": audit.digest(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())]
    (out / "receipt.json").write_text(json.dumps({**common, "artifacts": artifacts}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "cases": len(records), "six_step_pairs_per_case_per_phase": 3, "visual_review": "pending"}))


if __name__ == "__main__":
    main()
