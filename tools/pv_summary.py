"""Run PV summary: does the chaos of the 0.5B soft loop depend on the prompt? (human, 19 Oct 2026: "Did we see how the
chaos changes when we change the prompt? ... both together and both separately"; "the burden is on us to show").

For each condition's 128 x 128 map of the standard plane (pv-<condition>, made with scan --prefix or --start-vector):
- the chaotic share (largest exponent above the chaos level) of the plane, and of the rows in three temperature bands;
- the largest exponent, the share of cells with a detected period, and the leading periods;
- agreement with the base map p00, cell by cell (both chaotic or both not), and the Jaccard index of the chaotic sets.
Every map is gated before it is summarized: the grid must equal the base map's, the record must name the condition's
prefix or start vector, and no exponent may be non-finite. A map that fails a gate is listed as failed, not summarized.

Usage: python tools/pv_summary.py [--json docs/pv/pv-summary.json] [--out paper/llm/figures/fig-prompt-variants.png]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
BANDS = DATA / "explore" / "llm-soft-loop" / "pod64" / "bands"
LABELS = {
    "p00": "base: system prompt + \"Hello!\"", "p01": "system prompt only", "p02": "\"Hello!\" only (Qwen default system)",
    "p03": "neither (Qwen default system only)", "p04": "system words shuffled (1)", "p05": "system words shuffled (2)",
    "p06": "user tokens reversed (\"!Hello\")", "p07": "both shuffled", "p08": "opposite system prompt",
    "p09": "happy system prompt", "p10": "sad system prompt", "p11": "happy user message", "p12": "sad user message",
    "p13": "plain text, no chat markup", "p14": "long system prompt", "p15": "two-turn conversation",
    "p16": "prompt = one Gaussian vector", "p17": "prompt = one random token", "p18": "prompt = one random mixture",
    "s1": "base, start = Gaussian vector", "s2": "base, start = random token", "s3": "base, start = random mixture"}


def load(cond):
    d = BANDS / f"pv-{cond}"
    if not (d / "scan.npz").exists() and (BANDS / f"pv-{cond}b" / "scan.npz").exists():
        d = BANDS / f"pv-{cond}b"           # a rerun (p14: the first run ran out of GPU memory; rerun with --chunk 512)
    z = np.load(d / "scan.npz", allow_pickle=True)
    rec = json.loads((d / "scan.json").read_text())
    return z, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json")
    ap.add_argument("--out")
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    args = ap.parse_args()
    base, brec = load("p00")
    a, t = base["alpha"], base["log10_temperature"]
    bands = {"cold (log10 T < -0.5)": t < -0.5, "middle (-0.5 to 0)": (t >= -0.5) & (t < 0.0), "hot (log10 T >= 0)": t >= 0.0}
    base_chaos = base["lyapunov"] > args.chaos_level
    rows, failed, maps = [], [], {}
    for cond in LABELS:
        try:
            z, rec = load(cond)
        except FileNotFoundError as e:
            failed.append({"condition": cond, "gate": f"missing: {e.filename}"})
            continue
        ra = rec["args"]
        gate = []
        if not (np.array_equal(z["alpha"], a) and np.array_equal(z["log10_temperature"], t)):
            gate.append("grid differs from the base map")
        want = ("start_vector" if cond.startswith("s") else "prefix")
        if not ra.get(want) and cond != "p00":
            gate.append(f"record lacks --{want.replace('_', '-')}")
        lam = z["lyapunov"]
        if not np.all(np.isfinite(lam)):
            gate.append(f"{int((~np.isfinite(lam)).sum())} non-finite exponents")
        if gate:
            failed.append({"condition": cond, "gate": "; ".join(gate)})
            continue
        chaos = lam > args.chaos_level
        per = z["period"]
        top = sorted(((int(q), int((per == q).sum())) for q in np.unique(per[per > 0])), key=lambda x: -x[1])[:4]
        inter, union = int((chaos & base_chaos).sum()), int((chaos | base_chaos).sum())
        row = {"condition": cond, "label": LABELS[cond], "prefix_tokens": rec.get("prefix_tokens"),
               "chaotic_share": round(float(chaos.mean()), 4),
               "chaotic_share_by_band": {k: round(float(chaos[m].mean()), 4) for k, m in bands.items()},
               "max_lambda": round(float(lam.max()), 4), "periodic_share": round(float((per > 0).mean()), 4),
               "leading_periods": top, "agreement_with_base": round(float((chaos == base_chaos).mean()), 4),
               "jaccard_chaotic_with_base": round(inter / union, 4) if union else None}
        rows.append(row)
        maps[cond] = lam
    summary = {"chaos_level": args.chaos_level, "plane": {"alpha": [float(a[0]), float(a[-1])],
               "log10_temperature": [float(t[0]), float(t[-1])], "shape": list(base_chaos.shape)},
               "conditions": rows, "failed": failed}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if args.out and maps:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conds = list(maps)
        ncol = 6
        nrow = int(np.ceil(len(conds) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.5 * nrow), sharex=True, sharey=True)
        ext = [a[0], a[-1], t[0], t[-1]]
        for ax in axs.flat:
            ax.set_visible(False)
        share = {r["condition"]: r["chaotic_share"] for r in rows}
        for ax, cond in zip(axs.flat, conds):
            ax.set_visible(True)
            lam = maps[cond]
            ax.imshow(np.clip(np.where(lam <= args.chaos_level, lam, np.nan), -1.2, 0), origin="lower", extent=ext,
                      aspect="auto", cmap="gray", vmin=-1.2, vmax=0, interpolation="nearest")
            im = ax.imshow(np.where(lam > args.chaos_level, lam, np.nan), origin="lower", extent=ext, aspect="auto",
                           cmap="plasma", vmin=0, vmax=0.5, interpolation="nearest")
            ax.set_title(f"{LABELS[cond]}\n{100 * share[cond]:.0f} % chaotic", fontsize=7.5)
            ax.tick_params(labelsize=6)
        for ax in axs[-1]:
            ax.set_xlabel("steering strength α", fontsize=7)
        for ax in axs[:, 0]:
            ax.set_ylabel("log$_{10}T$", fontsize=7)
        cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.01)
        cb.set_label("largest Lyapunov exponent (chaotic cells; gray: periodic, by contraction)", fontsize=8)
        fig.savefig(args.out, dpi=170, bbox_inches="tight")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
