"""Run NT figure and summary: the near-nilpotent point of the period-5 window continued in log10 T
(tools/nilpotent_track.py). Gates every row before it is drawn: the orbit's Newton residual, the divisor-period
return distance (a split below 1e-7 means a divisor-period orbit), and |c1|, |c2| against the run's --tol-f. Rows
that fail are listed and drawn hollow; rows not converged end a track (no fallback point).

Usage: python tools/plot_nilpotent_track.py --runs llm-soft-loop/pod64/nt/down llm-soft-loop/pod64/nt/up \
         [--json docs/nt/nt-summary.json] [--out paper/llm/figures/fig-head-temperature.png]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def load(run):
    path = DATA / "explore" / run / "track.json"
    rep = json.loads(path.read_text())
    tol = rep["args"]["tol_f"]
    rows = []
    for r in rep["rows"]:
        gate = {"orbit": r["newton"] is not None and r["newton"] < 1e-9,
                "not_divisor": r["split"] is not None and r["split"] > 1e-7,
                "invariants": r["c1"] is not None and max(abs(r["c1"]), abs(r["c2"])) < tol}
        rows.append({**r, "gate": gate, "passed": bool(r["converged"] and all(gate.values()))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        for r in load(run):
            rows.append({**r, "run": run})
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r["log10_T"]):
        key = round(r["log10_T"], 6)
        if key in seen:                     # the start temperature appears in both tracks: keep both, mark repeat
            r = {**r, "repeat": True}
        seen.add(key)
        uniq.append(r)
    summary = {"runs": args.runs, "rows": [{k: r[k] for k in ("run", "log10_T", "a1", "a2", "moduli", "arguments_deg",
                                                               "tokens", "c1", "c2", "newton", "split", "gate",
                                                               "passed")} for r in uniq]}
    ok = [r for r in uniq if r["passed"]]
    if ok:
        lead = np.array([r["moduli"][0] for r in ok])
        lts = np.array([r["log10_T"] for r in ok])
        summary["leading_modulus_range"] = [float(lead.min()), float(lead.max())]
        summary["log10_T_range_passed"] = [float(lts.min()), float(lts.max())]
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
        ax = axes[0]
        for k, (col, lab) in enumerate([("k", r"$|\mu_1|$"), ("tab:blue", r"$|\mu_2|$"), ("tab:orange", r"$|\mu_3|$"),
                                        ("tab:green", r"$|\mu_4|$"), ("tab:red", r"$|\mu_5|$")]):
            xs = [r["log10_T"] for r in uniq if r["moduli"]]
            ys = [r["moduli"][k] for r in uniq if r["moduli"]]
            fc = [col if r["passed"] else "w" for r in uniq if r["moduli"]]
            ax.plot(xs, ys, "-", color=col, lw=0.8)
            ax.scatter(xs, ys, s=18, facecolors=fc, edgecolors=col, label=lab, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}T$")
        ax.set_ylabel("multiplier modulus at the near-nilpotent point")
        ax.legend(fontsize=7)
        ax.set_title("(a) the leading spectrum at the head (hollow: a gate failed)", fontsize=9)
        ax = axes[1]
        xs = [r["a1"] for r in uniq]
        ys = [r["a2"] for r in uniq]
        sc = ax.scatter(xs, ys, c=[r["log10_T"] for r in uniq], cmap="viridis", s=22, zorder=3)
        ax.plot(xs, ys, "-", color="0.5", lw=0.7)
        fig.colorbar(sc, ax=ax, label=r"$\log_{10}T$")
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)")
        ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)")
        ax.ticklabel_format(useOffset=False)
        ax.set_title("(b) the head's position in the plane", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out)


if __name__ == "__main__":
    main()
