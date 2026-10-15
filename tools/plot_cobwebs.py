"""Figure fig-cobwebs: cobweb diagrams of the 7-step return map along the hostile line (alpha 0.0655,
hostility-vs-kindness at layer 12, Qwen2.5-0.5B-Instruct, float64), through the doubling cascade at the right edge of
the hidden-chaos window: periods 7, 14, 28 and 56 (cells of hidden/tline-a0.0655), and the first chaotic cell below.

Each cell runs from the fixed start for `steps` steps (no discarded transient: the approach to the attractor is part
of the picture). For one phase of the 7-step cycle (the phase with the largest spread over the recorded states), the
states of that phase are projected on their leading principal component s, and the points (s_n, s_(n+1)) of
consecutive visits are drawn with the cobweb of a stretch of visits: from the diagonal to the map and back.

Usage: python tools/plot_cobwebs.py [--device cuda] [--steps 4200] [--out PNG]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"


def pick_cells():
    """The cascade at the window's right edge: the largest-temperature cell of each period 7, 14, 28, 56, then the
    largest-temperature chaotic cell below the period-56 one (hidden/tline-a0.0655, map-level labels)."""
    cells = json.loads((DATA / "hidden/tline-a0.0655/hidden-chaos.json").read_text())["cells"]
    out = []
    for q in (7, 14, 28, 56):
        c = max((c for c in cells if c["period"] == q and c["log10_temperature"] > 0.189), key=lambda c: c["log10_temperature"])
        out.append((f"period {q}", c["log10_temperature"]))
    t56 = out[-1][1]
    ch = [c for c in cells if c["period"] == 0 and min(c["lyapunov"]) > 1e-3 and c["log10_temperature"] < t56]
    c = max(ch, key=lambda c: c["log10_temperature"])
    out.append((f"chaos (lambda {c['lyapunov'][0]:.3f})", c["log10_temperature"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=4200)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--skip", type=int, default=1400, help="steps of the approach left out of the projection")
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-cobwebs.png"))
    args = ap.parse_args()
    cache = DATA / "cobwebs" / "trajectories.npz"
    cells = pick_cells()
    if cache.exists():
        z = np.load(cache)
        S = z["states"]
    else:
        import explore_llm_attractors as ela
        import explore_llm_phase_map as elpm
        import explore_llm_soft_loop as esl
        loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", args.device)
        v = np.asarray(elpm.load_vectors("llm-phase-map/vectors-l12-p1b")["hostility-vs-kindness"], dtype=np.float64)
        X = loop.prepare(np.array([0.0655 * v for _ in cells]))
        temps = loop.torch.as_tensor(10.0 ** np.array([t for _, t in cells]), dtype=loop.state_dtype, device=loop.device)
        S = np.zeros((args.steps, len(cells), X.shape[-1]))
        for i in range(args.steps):
            X, _ = loop.step(X, temps)
            S[i] = X[:, -1].double().cpu().numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, states=S, log10_temperature=np.array([t for _, t in cells]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cobweb(ax, s, start, n, color="tab:red"):
        cx, cy = [s[start]], [s[start]]
        for k in range(start, min(start + n, len(s) - 1)):
            cx += [s[k], s[k + 1]]
            cy += [s[k + 1], s[k + 1]]
        ax.plot(cx, cy, "-", color=color, lw=0.8, zorder=2)

    def frame(ax, x, y):
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo + 1e-30)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ":", color="k", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([])
        ax.set_yticks([])

    rm = np.load(DATA / "retmap/body-edge/return-map.npz")
    panels = len(cells) - 1 + 2                      # the four cascade cells, then two chaotic return maps of run RM
    fig, axes = plt.subplots(1, panels, figsize=(3.4 * panels, 3.7), dpi=170)
    report = []
    skip = args.skip
    for k, ((name, lt), ax) in enumerate(zip(cells[:-1], axes)):
        best = None
        for ph in range(7):
            sk = 70 if name == "period 7" else skip          # period 7 converges early: show its approach
            P = S[ph::7, k][sk // 7:sk // 7 + 80] if name == "period 7" else S[ph::7, k][sk // 7:]
            Pc = P - P.mean(0)
            _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
            s = Pc @ vt[0]
            if best is None or np.ptp(s) > best[0]:
                best = (np.ptp(s), ph, s, float(sv[0] ** 2 / np.sum(sv ** 2)))
        _, ph, s, expl = best
        x, y = s[:-1], s[1:]
        ax.plot(x, y, ".", ms=2, color="0.55", zorder=1)
        cobweb(ax, s, 0, 60)
        q = {"period 7": 1, "period 14": 2, "period 28": 4, "period 56": 8}[name]
        ax.plot(s[-q - 1:-1], s[-q:], "o", ms=4, color="tab:blue", zorder=3)
        frame(ax, x, y)
        ax.set_title(f"{name} (log10 T {lt:.5f})\n{q}-cycle of the 7-step map", fontsize=8)
        ax.set_xlabel(r"$s_n$", fontsize=8)
        if k == 0:
            ax.set_ylabel(r"$s_{n+1}$ (one 7-step return)", fontsize=8)
        report.append({"cell": name, "log10_temperature": lt, "phase": ph, "explained": expl})
    for j, (c, ph, title) in enumerate(((5, 0, "chaos, 7 bands (log10 T 0.18480)"),
                                        (0, 0, "chaos at the body's edge (alpha 0.07136)"))):
        ax = axes[len(cells) - 1 + j]
        s = rm[f"c{c}_p{ph}"]
        x, y = s[:-1], s[1:]
        ax.plot(x, y, ".", ms=1, color="0.55", zorder=1)
        cobweb(ax, s, len(s) // 2, 25)
        frame(ax, x, y)
        ax.set_title(title + "\none-humped map, a chaotic cobweb", fontsize=8)
        ax.set_xlabel(r"$s_n$", fontsize=8)
        report.append({"cell": title, "phase": ph})
    fig.suptitle("Cobwebs on the 7-step return map of the hostility-steered loop (Qwen2.5-0.5B, alpha 0.0655 unless "
                 "stated): gray, visits of one phase; red, the cobweb; blue, the attracting cycle", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
