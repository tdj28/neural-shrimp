"""Exploratory A3b analysis: regimes, controls and incremental value of rotation features.

Reads slices written by ``tools/explore_consequence.py simulate`` and

  1. labels every cell with ``neural_shrimp.regimes.classify`` (rules fixed
     in code before the analysis ran);
  2. summarizes each outcome per regime, with ordinary period-2 cells and
     qualified chaotic cells as the controls required by AGENTS.md;
  3. compares nested least-squares models with leave-one-block-out
     cross-validation (blocks of contiguous cells along the slice):
     base = FTLE (positive and negative parts plus indicator classes
     negative / |FTLE| <= 1e-3 / weakly positive, the classify thresholds),
     preconditioned sharpness, log10 rate and log10 oscillation amplitude;
     full = base + rotation and
     locking features (locked and quasi-periodic indicators, rotation number,
     log10 distance to the nearest rational with denominator <= 12, 1/q of the
     locked cycle). Reproducibility outcomes are also compared with a
     transient-amplification covariate added to both models.
     Intervals are moving-block bootstrap percentiles;
  4. checks whether the rotation number can be read off the training loss
     alone (spectral peak of the even-step loss series);
  5. draws a slice panel and a regime summary per slice.
"""

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.regimes import (  # noqa: E402
    block_bootstrap, blocked_cv_r2, classify, partial_correlation,
)

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "consequence"
BEYOND = ("period2", "locked", "other_periodic", "quasiperiodic", "chaotic", "irregular")
STYLE = {  # validated categorical slots 1-3 plus neutrals; markers as secondary encoding
    "fixed": ("#e1e0d9", "o"), "period2": ("#2a78d6", "o"), "locked": ("#eb6834", "D"),
    "quasiperiodic": ("#1baf7a", "s"), "other_periodic": ("#4a3aa7", "^"),
    "chaotic": ("#52514e", "o"), "irregular": ("#a3a29b", "v"), "divergent": ("#f0efec", "x"),
}
BLOCK = 25
FLOOR = 1e-17


def rationals(max_q=12):
    return sorted({Fraction(m, q) for q in range(2, max_q + 1) for m in range(1, q) if Fraction(m, q) <= 0.5})


def derive(d, cfg):
    n = len(d["log10_hidden"])
    deltas, dirs = cfg["deltas"], cfg["directions"]
    rho = d["rho"]
    regime, rho_used = [], np.zeros(n)
    for i in range(n):
        label = classify(int(d["status"][i]), int(d["period"][i]), tuple(d["ftle"][i]),
                         (rho[i, 1], rho[i, 2]), None if not np.isfinite(rho[i, 0]) else rho[i, 0],
                         float(d["separation"][i]), int(d["cycle"][i, 0]))
        regime.append(label)
        if label == "locked":
            rho_used[i] = d["cycle"][i, 0] / d["cycle"][i, 1]
        elif label in ("quasiperiodic", "chaotic", "irregular") and np.isfinite(rho[i, 0]):
            rho_used[i] = rho[i, 0]
    regime = np.array(regime)
    table = rationals()
    dist = np.array([min(abs(r - float(f)) for f in table) if r > 0 else 0.5 for r in rho_used])
    dist[regime == "locked"] = 0.0
    log_dist = np.log10(np.maximum(dist, 1e-6))
    locked_q = np.where(regime == "locked", d["cycle"][:, 1], 0)

    def block(i):
        return slice(i * dirs, (i + 1) * dirs)

    i8 = deltas.index(1e-8)
    med = np.stack([np.nanmedian(d["func_dis"][:, block(i), -1], axis=1) for i in range(len(deltas))], axis=1)
    logmed = np.log10(np.maximum(med, FLOOR))
    slope = np.array([np.polyfit(np.log10(deltas), row, 1)[0] if np.all(np.isfinite(row)) else np.nan
                      for row in logmed])
    ftle_mean = d["ftle"].mean(axis=1)
    f = {
        "log10_hidden": d["log10_hidden"], "regime": regime, "rho": rho_used, "locked_q": locked_q,
        "ftle": ftle_mean, "sharpness": d["sharpness_mean"], "amplitude": np.log10(np.maximum(d["amplitude"], FLOOR)),
        "transient": np.log10(np.maximum(np.nanmedian(d["param_dis"][:, block(0), 0], axis=1), FLOOR) / deltas[0]),
        "features_rotation": {
            "locked": (regime == "locked").astype(float), "quasiperiodic": (regime == "quasiperiodic").astype(float),
            "rho": rho_used, "log10_rational_distance": log_dist,
            "inv_q": np.where(locked_q > 0, 1.0 / np.maximum(locked_q, 1), 0.0)},
    }
    s2 = np.any(np.isfinite(d["tail_test_mean"]))
    outcomes = {
        "repro_func_T40k_d1e-8": logmed[:, i8],
        "repro_slope_vs_delta": slope,
        "repro_after_anneal_d1e-8": np.log10(np.maximum(np.nanmedian(d["anneal_func_dis"][:, block(i8)], axis=1), FLOOR)),
        "tail_train_loss": np.log10(d["tail_train_mean"]),
        "avg_iterate_train_loss": np.log10(d["avg_iterate_train"]),
        "stopping_time_spread": np.log10(np.maximum(d["anneal_func_spread"], FLOOR)),
        "annealed_train_loss": np.log10(np.nanmean(d["anneal_train"], axis=1)),
        "loss_line_fraction": d["line_fraction"],
    }
    if s2:
        outcomes.update({
            "tail_test_mse": np.log10(d["tail_test_mean"]),
            "avg_iterate_test_mse": np.log10(d["avg_iterate_test"]),
            "annealed_test_mse": np.log10(np.nanmean(d["anneal_test"], axis=1)),
        })
    f["outcomes"] = outcomes
    peaks = d["loss_peaks"]
    match = np.full(n, np.nan)
    for i in range(n):
        if regime[i] in ("quasiperiodic", "locked") and np.any(np.isfinite(peaks[i])):
            r = rho_used[i]
            gaps = [min(abs(p - r), abs(p - (1 - r))) for p in peaks[i] if np.isfinite(p)]
            match[i] = min(gaps)
    f["loss_rotation_gap"] = match
    # sensitivity flag (pre-specified): quasi-periodic labels whose two FTLE estimates are both
    # clearly negative may be slowly converging locked orbits missed by the recurrence screen
    f["qp_attracting"] = (regime == "quasiperiodic") & (np.max(d["ftle"], axis=1) < -2e-4)
    return f


def base_features(f):
    """FTLE (continuous parts plus sign classes at the classify thresholds), sharpness, rate, amplitude.

    Pooled slices add one indicator per extra slice (slice-specific intercepts).
    """
    lam = np.clip(f["ftle"], -0.05, 0.1)
    out = [np.maximum(lam, 0), np.minimum(lam, 0), (lam < -1e-3).astype(float),
           (np.abs(lam) <= 1e-3).astype(float), ((lam > 1e-3) & (lam <= 5e-3)).astype(float),
           f["sharpness"], f["log10_hidden"], f["amplitude"]]
    for k in range(1, int(f.get("slice", np.zeros(1)).max()) + 1):
        out.append((f["slice"] == k).astype(float))
    return out


def pool(fs):
    """Concatenate derived slices; ``order`` sorts by slice, then rate, so CV blocks stay within slices."""
    keys = ("log10_hidden", "regime", "rho", "locked_q", "ftle", "sharpness", "amplitude", "transient",
            "loss_rotation_gap", "qp_attracting")
    out = {k: np.concatenate([f[k] for f in fs]) for k in keys}
    out["slice"] = np.concatenate([np.full(len(f["ftle"]), k) for k, f in enumerate(fs)])
    out["order"] = out["slice"] * 10.0 + out["log10_hidden"]
    common = set.intersection(*[set(f["outcomes"]) for f in fs])
    out["outcomes"] = {k: np.concatenate([f["outcomes"][k] for f in fs]) for k in fs[0]["outcomes"] if k in common}
    out["features_rotation"] = {k: np.concatenate([f["features_rotation"][k] for f in fs])
                                for k in fs[0]["features_rotation"]}
    return out


def nested(f, rows, outcome, with_transient=False, draws=300):
    base = base_features(f)
    if with_transient:
        base.append(f["transient"])
    rot = list(f["features_rotation"].values())
    y = f["outcomes"][outcome]
    ok = rows & np.isfinite(y) & np.all(np.isfinite(np.stack(base + rot)), axis=0)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(f.get("order", f["log10_hidden"])[idx])]
    if len(idx) < 4 * BLOCK:
        return None
    B = [b[idx] for b in base]
    R = [r[idx] for r in rot]
    Y = y[idx]
    # drop rotation features that are constant on these rows
    R = [r for r in R if np.ptp(r) > 0]

    def gain(sel):
        blocks = np.arange(len(sel)) // BLOCK
        b = blocked_cv_r2([c[sel] for c in B], Y[sel], blocks)
        full = blocked_cv_r2([c[sel] for c in B] + [c[sel] for c in R], Y[sel], blocks)
        return b, full

    everything = np.arange(len(idx))
    r2_base, r2_full = gain(everything)
    low, high = block_bootstrap(lambda s: np.subtract(*gain(s)[::-1]), len(idx), BLOCK, draws=draws, seed=7)
    partial = {}
    for name, col in f["features_rotation"].items():
        c = col[idx]
        if np.ptp(c) == 0:
            continue
        value = partial_correlation(c, Y, B)
        lo, hi = block_bootstrap(lambda s: partial_correlation(c[s], Y[s], [b[s] for b in B]), len(idx), BLOCK,
                                 draws=draws, seed=11)
        partial[name] = [round(value, 3), round(lo, 3), round(hi, 3)]
    return {"rows": int(len(idx)), "cv_r2_base": round(r2_base, 3), "cv_r2_full": round(r2_full, 3),
            "delta_r2": round(r2_full - r2_base, 3), "delta_r2_ci": [round(low, 3), round(high, 3)],
            "partial_correlations": partial}


def summarize(f):
    out = {}
    for name, y in f["outcomes"].items():
        per = {}
        for label in np.unique(f["regime"]):
            v = y[(f["regime"] == label) & np.isfinite(y)]
            if len(v):
                per[label] = {"n": int(len(v)), "median": round(float(np.median(v)), 4),
                              "q25": round(float(np.percentile(v, 25)), 4), "q75": round(float(np.percentile(v, 75)), 4)}
        out[name] = per
    return out


def slice_figure(f, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = f["log10_hidden"]
    panels = [("rotation\nnumber", f["rho"], False),
              ("largest\nFTLE", f["ftle"], False),
              ("log10 disagreement\nT=40k, delta 1e-8", f["outcomes"]["repro_func_T40k_d1e-8"], False),
              ("log10 disagreement\nafter anneal", f["outcomes"]["repro_after_anneal_d1e-8"], False),
              ("log10 stopping-\ntime spread", f["outcomes"]["stopping_time_spread"], False)]
    loss_key = "tail_test_mse" if "tail_test_mse" in f["outcomes"] else "tail_train_loss"
    panels.append(("log10 tail\n" + ("test MSE" if loss_key == "tail_test_mse" else "train loss"),
                   f["outcomes"][loss_key], False))
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 1.7 * (len(panels) + 1)), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [0.35] + [1] * len(panels)})
    fig.patch.set_facecolor("#fcfcfb")
    band = axes[0]
    for label, (color, _) in STYLE.items():
        sel = f["regime"] == label
        if sel.any():
            band.scatter(x[sel], np.zeros(sel.sum()), c=color, marker="|", s=300, linewidths=1.2)
    band.set_yticks([])
    band.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    for ax, (name, y, _) in zip(axes[1:], panels):
        for label, (color, marker) in STYLE.items():
            if label == "divergent":
                continue
            sel = (f["regime"] == label) & np.isfinite(y)
            if sel.any():
                ax.plot(x[sel], y[sel], marker, color=color, markersize=2.6, linestyle="", label=label.replace("_", " "))
        ax.set_ylabel(name, fontsize=7.5, color="#52514e")
        ax.tick_params(labelsize=7.5, colors="#52514e")
        ax.grid(True, color="#f0efec", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].legend(fontsize=7, frameon=False, ncol=7, loc="upper left", markerscale=2.5)
    axes[-1].set_xlabel("log10 hidden-layer learning rate (output rate fixed)", fontsize=9, color="#52514e")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def regime_figure(f, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(f["outcomes"])
    order = [r for r in ("fixed",) + BEYOND if np.any(f["regime"] == r)]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    for ax, name in zip(axes.ravel(), names):
        y = f["outcomes"][name]
        for k, label in enumerate(order):
            sel = (f["regime"] == label) & np.isfinite(y)
            if not sel.any():
                continue
            color, marker = STYLE[label]
            ax.plot(k + 0.25 * (rng.random(sel.sum()) - 0.5), y[sel], marker, color=color, markersize=2.2,
                    alpha=0.8, linestyle="")
            ax.plot([k - 0.3, k + 0.3], [np.median(y[sel])] * 2, color="#0b0b0b", linewidth=1.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6.5, color="#52514e")
        ax.set_title(name.replace("_", " "), fontsize=8, loc="left", color="#0b0b0b")
        ax.tick_params(labelsize=7, colors="#52514e")
        ax.grid(True, axis="y", color="#f0efec", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("slices", nargs="+", help="slice names under explore/consequence")
    parser.add_argument("--ns", type=float, nargs="+", help="log10 hidden Neimark-Sacker point per slice")
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--pool", action="store_true", help="also fit pooled nested models across the slices")
    parser.add_argument("--report", default="analysis.json")
    args = parser.parse_args()
    report = {}
    derived = []
    for k, name in enumerate(args.slices):
        folder = OUT / name
        meta = json.loads((folder / "metadata.json").read_text())
        d = np.load(folder / "slice.npz")
        f = derive(d, meta["config"])
        derived.append(f)
        labels, counts = np.unique(f["regime"], return_counts=True)
        entry = {"metadata": {k_: meta[k_] for k_ in ("system", "seed", "log10_output", "hidden", "cells", "commit")},
                 "regime_counts": dict(zip(labels.tolist(), counts.tolist())), "per_regime": summarize(f)}
        locked = f["regime"] == "locked"
        q = f["locked_q"][locked]
        per = d["period"][locked]
        entry["locked_period_is_2q_power2"] = int(sum(1 for a, b in zip(per, q)
                                                       if b > 0 and (a // (2 * b)) >= 1 and a % (2 * b) == 0
                                                       and ((a // (2 * b)) & ((a // (2 * b)) - 1)) == 0))
        entry["locked_cells"] = int(locked.sum())
        entry["locked_rotations"] = sorted({f"{int(m)}/{int(qq)}" for m, qq in d["cycle"][locked]})
        gap = f["loss_rotation_gap"]
        for label in ("quasiperiodic", "locked"):
            g = gap[(f["regime"] == label) & np.isfinite(gap)]
            entry[f"loss_peak_matches_rotation_{label}"] = [int((g < 1e-3).sum()), int(len(g))]
        rows_all = np.isin(f["regime"], BEYOND)
        entry["qp_with_negative_ftle"] = int(f["qp_attracting"].sum())
        rows_ns = rows_all & (f["log10_hidden"] > args.ns[k]) if args.ns else None
        models = {}
        for outcome in f["outcomes"]:
            models[outcome] = {"beyond_eos": nested(f, rows_all, outcome, draws=args.draws)}
            if rows_ns is not None:
                models[outcome]["beyond_ns"] = nested(f, rows_ns, outcome, draws=args.draws)
            if outcome.startswith("repro"):
                models[outcome]["beyond_eos_with_transient"] = nested(f, rows_all, outcome, True, draws=args.draws)
            if f["qp_attracting"].any():
                models[outcome]["beyond_eos_without_attracting_qp"] = nested(
                    f, rows_all & ~f["qp_attracting"], outcome, draws=args.draws)
        entry["nested_models"] = models
        p2 = np.flatnonzero(f["regime"] == "period2")
        lk = np.flatnonzero(locked)
        if len(p2) and len(lk):
            entry["period2_control_min_log_rate_distance_to_locked"] = round(
                float(np.min(np.abs(f["log10_hidden"][p2][:, None] - f["log10_hidden"][lk][None, :]))), 4)
        report[name] = entry
        title = (f"{meta['system']}-{meta['seed']}, log10 output {meta['log10_output']:.4f}: regimes and training "
                 "consequences (exploratory)")
        slice_figure(f, title, folder / "slice.png")
        regime_figure(f, title, folder / "regimes.png")
    if args.pool and len(derived) > 1:
        pooled = pool(derived)
        rows = np.isin(pooled["regime"], BEYOND)
        report["pooled"] = {"slices": args.slices, "nested_models": {
            outcome: {"beyond_eos": nested(pooled, rows, outcome, draws=args.draws),
                      "beyond_eos_without_attracting_qp": nested(pooled, rows & ~pooled["qp_attracting"], outcome,
                                                                 draws=args.draws)}
            for outcome in pooled["outcomes"]}}
    target = OUT / args.report
    target.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: {"regime_counts": v["regime_counts"], "locked_rotations": v["locked_rotations"]}
                      for k, v in report.items() if k != "pooled"}, indent=1))


if __name__ == "__main__":
    main()
