"""Figure of run PV: where the chaos is found across prompts (human, 19 Oct 2026: "averaging the pixels of all the prompt
chaos maps to show highlights where chaos is found most often").

For every cell of the standard plane (128 x 128 float64 cells, alpha -0.12 to 0.12, log10 T -1.0 to 0.3, philosophy at
layer 12), the fraction of the prompt variants p00 to p18 in which the cell is chaotic (largest exponent above 1e-3).
The three random-start runs (s1 to s3) are left out: they use the base prompt and nearly reproduce its map, so they
would weight it four times. (a) The frequency over the plane. (b) The mean frequency in each temperature row, for all
prompts, for the 16 text prompts and for the 3 prompts made of one random vector.

Usage: python tools/plot_pv_frequency.py --out paper/llm/figures/fig-prompt-frequency.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pv_summary as pvs  # noqa: E402

TEXT = [f"p{k:02d}" for k in range(16)]
VECTOR = ["p16", "p17", "p18"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chaos, used, missing = {}, [], []
    for c in TEXT + VECTOR:
        try:
            z, _ = pvs.load(c)
        except FileNotFoundError:
            missing.append(c)
            continue
        lam = z["lyapunov"]
        if not np.all(np.isfinite(lam)):
            missing.append(c)
            continue
        chaos[c] = lam > args.chaos_level
        used.append(c)
        a, t = z["alpha"], z["log10_temperature"]
    allc = np.array([chaos[c] for c in used], dtype=float)
    txt = np.array([chaos[c] for c in used if c in TEXT], dtype=float)
    vec = np.array([chaos[c] for c in used if c in VECTOR], dtype=float)
    freq = allc.mean(0)
    plt.rcParams.update({"font.size": 8})
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    im = ax.imshow(freq, origin="lower", extent=[a[0], a[-1], t[0], t[-1]], aspect="auto", cmap="magma",
                   vmin=0, vmax=max(0.5, float(freq.max())), interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("fraction of prompts in which the cell is chaotic", fontsize=7)
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    ax.set_title(f"(a) how often each cell is chaotic ({len(used)} prompts)", fontsize=8)
    # Okabe-Ito blue and orange (told apart under colour-vision deficiencies and, by lightness, in grey print), and a
    # dashed line for the random-vector prompts
    for F, lab, col, ls in ((allc, f"all {len(used)} prompts", "k", "-"),
                            (txt, f"{len(txt)} text prompts", "#0072B2", "-"),
                            (vec, f"{len(vec)} random-vector prompts", "#E69F00", "--")):
        if len(F):
            axp.plot(F.mean(0).mean(1), t, color=col, ls=ls, lw=1.4 if col == "k" else 1.0, label=lab)
    axp.set_xlabel("mean fraction of chaotic cells in the row")
    axp.set_ylabel(r"$\log_{10}T$")
    axp.set_ylim(t[0], t[-1])
    axp.set_title("(b) by temperature", fontsize=8)
    axp.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    rowf = freq.mean(1)
    k = int(np.argmax(rowf))
    rec = {"used": used, "missing": missing,
           "share_chaotic_in_at_least_one": float((freq > 0).mean()),
           "share_chaotic_in_half_or_more": float((freq >= 0.5).mean()),
           "share_chaotic_in_all": float((freq == 1).mean()),
           "max_cell_frequency": float(freq.max()),
           "peak_row": {"log10_temperature": float(t[k]), "mean_frequency": float(rowf[k])},
           "mean_frequency_by_band": {"cold (log10 T < -0.5)": float(freq[t < -0.5].mean()),
                                      "middle (-0.5 to 0)": float(freq[(t >= -0.5) & (t < 0)].mean()),
                                      "hot (log10 T >= 0)": float(freq[t >= 0].mean())}}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
