"""Figure fig-chaos-twins and table rows for the chaos-zone texts (run CT, tools/chaos_texts.py).

For every recorded cell that passed the chaotic label (no period up to 64, twin exponents above 1e-3 at relative
offsets 1e-6 and 1e-8), the figure draws the relative distance between the loop and its free twin, started 1e-10
away and never renormalized, against the step in units of the cell's Lyapunov time, lambda * t, where every cell's
growth should follow the same line 1e-10 * exp(lambda * t) (dotted). Dots mark the first step where their most
probable tokens differ; the legend gives that step. The script also prints, for each cell, LaTeX-ready texts
of both loops around that step, for the paper's table.

Usage: python tools/plot_chaos_texts.py OUT.png OUT.json RECEIPT.json[:CELLINDEX] ...
       (each argument after the outputs names one receipt and optionally one cell; default: all its chaotic cells)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {"qwen0.5b": "0.5B", "qwen1.5b": "1.5B"}


def tex_token(t):
    out = []
    for ch in t:
        if ch == "\n":
            out.append(r"$\hookleftarrow$")
        elif ch == " ":
            out.append(r"\ ")
        elif ch in "#$%&_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def window(cell, before, after):
    """Base and twin leading tokens from `before` steps before the split to `after` steps after it."""
    k = cell["split_step"] - 1          # index of the first differing step
    lo, hi = max(0, k - before), min(len(cell["steps"]), k + after)
    base = [s["top"][0]["text"] for s in cell["steps"][lo:hi]]
    twin = [s["twin"] for s in cell["steps"][lo:hi]]
    return lo, k, base, twin


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png, out_json = sys.argv[1], sys.argv[2]
    picks = []
    for arg in sys.argv[3:]:
        path, _, idx = arg.partition(":")
        rec = json.loads(Path(path).read_text())
        cells = rec["cells"] if not idx else [rec["cells"][int(idx)]]
        for c in cells:
            picks.append((rec["args"], c, path))
    fig, bx = plt.subplots(figsize=(4.8, 3.0))
    cmap = plt.get_cmap("tab10")
    summary = []
    for i, (args, c, path) in enumerate(picks):
        if not c["chaotic"]:
            raise SystemExit(f"{path}: cell {c['alpha']},{c['log10T']} did not pass the chaotic label")
        lam = float(np.mean(c["lyapunov"]))
        d = np.array([s["log10_distance"] for s in c["steps"]])
        t = np.arange(1, len(d) + 1)
        col = cmap(i % 10)
        concept = {"hostility-vs-kindness": "hostility"}.get(args["concept"], args["concept"])
        steered = f"{concept}, $\\alpha={c['alpha']:g}$, " if c["alpha"] else "unsteered, "
        label = f"{MODEL_LABELS.get(args['model'], args['model'])}, {steered}$\\lambda={lam:.2f}$"
        split = f", split at {c['split_step']}" if c["split_step"] else ", no split"
        bx.plot(lam * t, d, "-", color=col, lw=1.0, label=label + split)
        if c["split_step"]:
            bx.plot(lam * c["split_step"], d[c["split_step"] - 1], "o", color=col, ms=4, mec="black", mew=0.5)
        predicted = float(np.log(1.0 / args["eps_text"]) / lam)
        summary.append({"receipt": path, "model": args["model"], "concept": args["concept"] if args["vectors"] else None,
                        "alpha": c["alpha"], "log10T": c["log10T"], "lyapunov": c["lyapunov"],
                        "split_step": c["split_step"], "order_one_steps_predicted": predicted,
                        "lyapunov_times_at_split": (lam * c["split_step"]) if c["split_step"] else None,
                        "log10_distance_at_split": float(d[c["split_step"] - 1]) if c["split_step"] else None,
                        "first_step_distance_above_0.1": (int(np.argmax(d > -1)) + 1) if (d > -1).any() else None})
        if c["split_step"]:
            lo, k, base, twin = window(c, 14, 10)
            print(f"% {label}: split at step {c['split_step']}, window from step {lo + 1}")
            print("base:", "".join(tex_token(x) for x in base[:k - lo]) + r"\,$\Vert$\," +
                  "".join(tex_token(x) for x in base[k - lo:]))
            print("twin:", "".join(tex_token(x) for x in twin[:k - lo]) + r"\,$\Vert$\," +
                  "".join(tex_token(x) for x in twin[k - lo:]))
    eps = picks[0][0]["eps_text"]
    xs = np.linspace(0, 26, 100)
    bx.plot(xs, np.log10(eps) + xs / np.log(10), ":", color="black", lw=0.9)
    bx.axhline(0, color="gray", lw=0.5)
    bx.set_ylim(-11, 1)
    bx.tick_params(labelsize=7)
    bx.set_ylabel(r"$\log_{10}$ relative distance of the copy", fontsize=8)
    bx.set_xlabel(r"step $t$ in Lyapunov times, $\lambda t$", fontsize=8)
    bx.set_xlim(0, 32)
    bx.legend(fontsize=5.6, frameon=False, loc="lower right")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
