"""Readings of run BH (bound hierarchy; run card in docs/research-log.md, 18 Oct 2026 17:00 UTC).

Per completed cell (tools/bound_hierarchy.py output): the block bounds B_k (dyadic k = 1..32 and overlapping windows,
every k <= 32) over a recorded window of N = 32 steps, Lambda_N = B_32, the long-run tangent exponent lambda_long
(2000 steps after the window), the map's label and exponent, the factorization error and the per-step gap
log(||Sigma_p|| ||J_h|| / T) - log||DF||. The readings fixed in the run card:
1. chain: B_k >= B_2k >= ... >= Lambda_N in every cell (violations beyond 1e-12);
2. the largest relative factorization error DF = Sigma_p J_h / T;
3. convergence: the median of B_k - lambda_long against k, per label;
4. accuracy: for each k, the share of cells whose sign of B_k agrees with the sign of lambda_long, and with the map
   label; the smallest k at which the share reaches 95 %, or that no k up to 32 does;
5. the per-step gap, per label.
Also reported: cells where B_k lies below lambda_long (B_k bounds the window's growth, not the long run, so finite
windows can fall below), and agreement of lambda_long with the map label.

Usage: python tools/analyze_bh.py --runs llm-soft-loop/pod64/bh/a-0 llm-soft-loop/pod64/bh/a-1 \
         --json docs/bh/bh-summary.json --out paper/llm/figures/fig-bound-hierarchy.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    cells = []
    for r in args.runs:
        cells += json.loads((DATA / "explore" / r / "bh.json").read_text())["cells"]
    status = {}
    for c in cells:
        status[c.get("status")] = status.get(c.get("status"), 0) + 1
    done = sorted([c for c in cells if c.get("status") == "done"], key=lambda c: c["index"])
    kd = sorted({int(k) for c in done for k in c["B_dyadic"]})
    ko = sorted({int(k) for c in done for k in c["B_overlap"]})
    lab = np.array([c["map_label"] for c in done])
    lam = np.array([c["lambda_long"] for c in done])
    Bd = np.array([[c["B_dyadic"][str(k)] for k in kd] for c in done])
    Bo = np.array([[c["B_overlap"][str(k)] for k in ko] for c in done])
    # 1. chain
    viol = float(max(0.0, np.max(np.diff(Bd, axis=1))))          # B_2k - B_k > 0 is a violation
    lam_N = np.array([c["Lambda_N"] for c in done])
    viol_N = float(max(0.0, np.max(lam_N - Bd[:, -1])))
    # 2. factorization
    fact = float(max(c["factorization_rel_err_max"] for c in done))
    # 3. convergence and 4. accuracy, on the overlapping-window bounds (every k), dyadic for the chain
    out = {"cells_total": len(cells), "status": status, "completed": len(done),
           "completed_by_label": {lb: int((lab == lb).sum()) for lb in np.unique(lab)},
           "chain_violation_max": viol, "Lambda_N_minus_B32_max": viol_N, "chain_monotone_all": all(c["chain_monotone"] for c in done),
           "factorization_rel_err_max": fact, "k_dyadic": kd, "k_overlap": ko}
    conv, acc = {}, {}
    for lb in np.unique(lab):
        m = lab == lb
        conv[lb] = {"median_B_minus_lambda_dyadic": dict(zip(map(str, kd), np.median(Bd[m] - lam[m, None], axis=0).tolist())),
                    "median_B_minus_lambda_overlap": dict(zip(map(str, ko), np.median(Bo[m] - lam[m, None], axis=0).tolist())),
                    "median_lambda_long": float(np.median(lam[m])),
                    "cells_B32_below_lambda_long": int((Bd[m, -1] < lam[m]).sum())}
    sign_lam = np.sign(lam)
    sign_lab = np.where(lab == "chaotic", 1.0, -1.0)
    for name, B, ks in (("overlap", Bo, ko), ("dyadic", Bd, kd)):
        agree_lam = (np.sign(B) == sign_lam[:, None]).mean(0)
        agree_lab = (np.sign(B) == sign_lab[:, None]).mean(0)
        first_lam = next((k for k, a in zip(ks, agree_lam) if a >= 0.95), None)
        first_lab = next((k for k, a in zip(ks, agree_lab) if a >= 0.95), None)
        acc[name] = {"share_sign_agrees_lambda_long": dict(zip(map(str, ks), agree_lam.tolist())),
                     "share_sign_agrees_map_label": dict(zip(map(str, ks), agree_lab.tolist())),
                     "smallest_k_95pct_lambda_long": first_lam, "smallest_k_95pct_map_label": first_lab,
                     "periodic_cells_with_positive_bound": dict(zip(map(str, ks), (B[lab == "periodic"] > 0).sum(0).tolist()))}
    out["convergence"] = conv
    out["accuracy"] = acc
    out["lambda_long_sign_agrees_map_label"] = int((sign_lam == sign_lab).sum())
    out["gap"] = {lb: {"median_of_cell_medians": float(np.median([c["gap_median"] for c in done if c["map_label"] == lb])),
                       "min": float(min(c["gap_min"] for c in done if c["map_label"] == lb))} for lb in np.unique(lab)}
    out["cells"] = [{"index": c["index"], "source": c["source"], "label": c["map_label"], "alpha": c["alpha"],
                     "log10_T": c["log10_temperature"], "map_exponent": c["map_exponent"], "lambda_long": c["lambda_long"],
                     "B": {str(k): c["B_dyadic"][str(k)] for k in kd}, "gap_median": c["gap_median"]} for c in done]
    text = json.dumps(out, indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("cells", "convergence", "accuracy")}, indent=1))
    for lb, v in conv.items():
        print(lb, "median B_k - lambda_long (dyadic):", {k: round(x, 3) for k, x in v["median_B_minus_lambda_dyadic"].items()},
              "| median lambda_long %.3f | B_32 below lambda_long in %d" % (v["median_lambda_long"], v["cells_B32_below_lambda_long"]))
    for name in ("dyadic",):
        print("sign agreement with lambda_long:", {k: round(x, 3) for k, x in acc[name]["share_sign_agrees_lambda_long"].items()})
        print("sign agreement with map label:  ", {k: round(x, 3) for k, x in acc[name]["share_sign_agrees_map_label"].items()})
        print("periodic cells with a positive bound:", acc[name]["periodic_cells_with_positive_bound"])
    print("smallest k reaching 95 % (overlap): lambda_long", acc["overlap"]["smallest_k_95pct_lambda_long"],
          "map label", acc["overlap"]["smallest_k_95pct_map_label"])
    if args.json:
        Path(args.json).write_text(text + "\n")
    if not args.out:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=200)
    col = {"chaotic": "tab:red", "periodic": "tab:blue"}
    ax = axes[0]
    for i, c in enumerate(done):
        ax.plot(ko, Bo[i], "-", color=col[lab[i]], alpha=0.35, lw=0.9)
        ax.plot([ko[-1] * 1.08], [lam[i]], "<", color=col[lab[i]], ms=4, alpha=0.7)
    for lb in col:
        ax.plot(ko, np.median(Bo[lab == lb], axis=0), "-", color=col[lb], lw=2.6, label=f"{lb} cells: median $B_k$")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block length $k$ (steps)")
    ax.set_ylabel("block bound $B_k$ (per step)")
    ax.set_title("(a) the bounds fall with $k$ (triangles: long-run exponent)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for lb in col:
        m = lab == lb
        ax.plot(ko, np.median(Bo[m] - lam[m, None], axis=0), "o-", color=col[lb], ms=3, label=f"{lb}: median $B_k-\\lambda$")
        q1, q3 = np.percentile(Bo[m] - lam[m, None], [25, 75], axis=0)
        ax.fill_between(ko, q1, q3, color=col[lb], alpha=0.15, lw=0)
    kk = np.array(ko, dtype=float)
    ax.plot(kk, np.median(Bo[:, 0] - lam) / kk, "k:", lw=1.2, label="$1/k$ from $k=1$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("block length $k$ (steps)")
    ax.set_ylabel(r"overshoot $B_k-\lambda_{\mathrm{long}}$")
    ax.set_title("(b) how far the bound overshoots the exponent", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(ko, list(acc["overlap"]["share_sign_agrees_lambda_long"].values()), "o-", color="k", ms=3,
            label="sign of $B_k$ = sign of long-run exponent")
    ax.plot(ko, list(acc["overlap"]["share_sign_agrees_map_label"].values()), "s--", color="tab:purple", ms=3,
            label="sign of $B_k$ matches the map label")
    ax.axhline(0.95, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("block length $k$ (steps)")
    ax.set_ylabel("share of cells")
    ax.set_title(f"(c) does the bound's sign predict chaos? ({len(done)} cells)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
