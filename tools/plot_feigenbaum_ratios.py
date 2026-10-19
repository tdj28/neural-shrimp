"""Figure fig-feigenbaum-ratios: finite-order interval ratios of the doubling sequences, from the stored float64 runs.

- Qwen2.5-0.5B crescent (philosophy, layer 12), cut log10 T = -0.14798: flips (cascade/crescent-t-0.14798-v2) and
  superstable points (superstable/crescent-t-0.14798, superstable/crescent-deep2), with worst-case ranges from the
  clean brackets of the imprecise superstable points.
- The same crescent at log10 T = -0.156: flips from the pod run (cascade/crescent-t-0.156-b: 3->6, 6->12, 12->24), the
  local continuations (cascade/crescent-t-0.156-c from 12->24 on, cascade/crescent-t-0.156-d if present), and the
  384->768 flip of run FBD (cascade/crescent-t-0.156-e), whose locator refused at the precision floor of the period-384
  multiplier: its root is the zero of the least-squares line through the logged evaluations (research log, FBD
  result), and the 768->1536 flip of run FBD2 (cascade/crescent-t-0.156-f); superstable points
  (superstable/crescent-t-0.156, crescent-t-0.156-192, crescent-t-0.156-384 if present).
- Qwen2.5-1.5B (philosophy, layer 14) along log10 T at alpha 0.100: flips (cascade/q15-a0.100-p4, edges/q15-a0.100-p16,
  edges/q15-a0.100-p32).

A flip ratio (lambda_n - lambda_{n-1}) / (lambda_{n+1} - lambda_n) is plotted at level n - 1 (the band of its first
interval), a superstable ratio half a level later. Usage: python tools/plot_feigenbaum_ratios.py [--out PNG]
"""
import argparse
import itertools
import json

import numpy as np
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
DELTA = 4.669201609


def ratios(lams):
    return [(lams[i] - lams[i - 1]) / (lams[i + 1] - lams[i]) for i in range(1, len(lams) - 1)]


def flips(run):
    return json.loads((DATA / "cascade" / run / "cascade.json").read_text())["flip_parameters"]


def progress_flips(run):
    """Flip parameters recorded in a cascade run's progress file (runs that stopped after some flips), in order."""
    f = DATA / "cascade" / run / "progress.jsonl"
    if not f.exists():
        return []
    return [json.loads(line)["lambda"] for line in f.read_text().splitlines() if line.startswith('{"flip"')]


def edge_flip(run):
    (w,) = json.loads((DATA / "edges" / run / "edges.json").read_text())["windows"]
    return next(e["lambda"] for e in w["edges"] if e["kind"] == "flip")


def superstable():
    """Superstable points (period, value, clean bracket) of the crescent at log10 T = -0.14798, period 12 on."""
    rows = []
    for run in ("crescent-t-0.14798", "crescent-deep2", "crescent-t-0.14798-768b"):     # 768b: run FBS768n
        f = DATA / "superstable" / run / "superstable.json"
        if not f.exists():
            continue
        for b in json.loads(f.read_text())["bands"]:
            if b.get("superstable") is not None and b["period"] >= 12:
                rows.append((b["period"], b["superstable"], b["clean_bracket"]))
    return sorted(rows)


def worst_case(rows, i):
    """Range of (s_i - s_{i-1}) / (s_{i+1} - s_i) over the corners of the three brackets."""
    vals = [(b - a) / (c - b) for a, b, c in itertools.product(*(rows[j][2] for j in (i - 1, i, i + 1)))]
    return min(vals), max(vals)


def floor_root(run):
    """Zero of the least-squares line g(lambda) through a refused flip search's logged evaluations."""
    ev = [json.loads(line) for line in (DATA / "cascade" / run / "progress.jsonl").read_text().splitlines()
          if line.startswith("{") and '"g"' in line]
    lam = np.array([e["lambda"] for e in ev])
    g = np.array([e["g"] for e in ev])
    c = lam.mean()
    b, a = np.polyfit(lam - c, g, 1)
    return float(c - a / b)


def sequence_156():
    lams = flips("crescent-t-0.156-b")[:2] + flips("crescent-t-0.156-c")
    d = DATA / "cascade/crescent-t-0.156-d/cascade.json"
    if d.exists():
        lams += json.loads(d.read_text())["flip_parameters"]
    elif (DATA / "cascade/crescent-t-0.156-e/progress.jsonl").exists():
        lams.append(floor_root("crescent-t-0.156-e"))
    f = DATA / "cascade/crescent-t-0.156-f/progress.jsonl"       # FBD2: the 768->1536 flip
    if f.exists():
        lams += [json.loads(line)["lambda"] for line in f.read_text().splitlines() if line.startswith('{"flip"')]
    return lams


def superstable_156():
    """Superstable points (period, value, clean bracket) at log10 T = -0.156, period 12 on."""
    rows = []
    for run in ("crescent-t-0.156", "crescent-t-0.156-192", "crescent-t-0.156-384"):
        f = DATA / "superstable" / run / "superstable.json"
        if f.exists():
            for b in json.loads(f.read_text())["bands"]:
                if b.get("superstable") is not None and b["period"] >= 12:
                    rows.append((b["period"], b["superstable"], b["clean_bracket"]))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-feigenbaum-ratios.png"))
    args = ap.parse_args()

    ss = superstable()
    s148 = ratios([r[1] for r in ss])
    f156 = ratios(sequence_156())
    q15 = ratios(flips("q15-a0.100-p4") + [edge_flip("q15-a0.100-p16"), edge_flip("q15-a0.100-p32")]
                 + progress_flips("q15-a0.100-p64") + progress_flips("q15-a0.100-p128"))
    f148 = ratios(flips("crescent-t-0.14798-v2") + progress_flips("crescent-t-0.14798-p96b")
                  + (progress_flips("crescent-t-0.14798-p96d") or progress_flips("crescent-t-0.14798-p96c")))

    fig, ax = plt.subplots(figsize=(4.2, 2.9), dpi=300)
    ax.axhline(DELTA, color="0.3", ls=":", lw=1)
    ax.text(1.45, DELTA + 0.04, r"$\delta=4.669$", ha="left", va="bottom", fontsize=7)
    blue, green, red = "#0072B2", "#D55E00", "#009E73"   # Okabe-Ito: safe under color-vision deficiency
    ax.plot(range(1, len(f148) + 1), f148, "o-", color=blue, label=r"0.5B, $\log_{10}T=-0.148$, flips")
    xs = [n + 1.5 for n in range(1, len(s148) + 1)]
    lo, hi = zip(*(worst_case(ss, i) for i in range(1, len(ss) - 1)))
    err = [[max(v - a, 0) for v, a in zip(s148, lo)], [max(b - v, 0) for v, b in zip(s148, hi)]]
    ax.errorbar(xs, s148, yerr=err, fmt="s--", color=blue, mfc="white", capsize=3,
                label=r"0.5B, $\log_{10}T=-0.148$, superstable")
    top = 5.0
    inside = [(n, v) for n, v in enumerate(f156, start=1) if v <= top]
    ax.plot(*zip(*inside), "D-", color=green, ms=5,
            label=r"0.5B, $\log_{10}T=-0.156$, flips")
    for n, v in enumerate(f156, start=1):         # off-scale ratios: an arrow at the top edge, joined by a dotted line
        if v > top:
            ax.plot([n], [top - 0.06], marker="^", color=green, ms=7, clip_on=False)
            ax.text(n + 0.1, top + 0.01, "%.1f" % v, color=green, fontsize=7, va="bottom", ha="left", clip_on=False)
            if n < len(f156):
                ax.plot([n, n + 1], [top - 0.06, f156[n]], ":", color=green, lw=1.2)
    ss6 = superstable_156()
    s156 = ratios([r[1] for r in ss6])
    if s156:
        lo6, hi6 = zip(*(worst_case(ss6, i) for i in range(1, len(ss6) - 1)))
        wide = [b - a > 1.0 for a, b in zip(lo6, hi6)]    # ranges set by the wide s12, s24 brackets: in the caption
        err6 = [[0 if w else max(v - a, 0) for v, a, w in zip(s156, lo6, wide)],
                [0 if w else max(b - v, 0) for v, b, w in zip(s156, hi6, wide)]]
        ax.errorbar([n + 1.5 for n in range(1, len(s156) + 1)], s156, yerr=err6, fmt="D--", color=green, mfc="white",
                    ms=5, capsize=3, label=r"0.5B, $\log_{10}T=-0.156$, superstable")
    ax.plot(range(1, len(q15) + 1), q15, "^-", color=red, label="1.5B, flips")
    ax.set_xlim(0.6, 6.0 if len(f156) < 6 else len(f156) + 0.4)
    ax.set_ylim(2.6, top)
    ax.set_xlabel(r"doubling level $n$", fontsize=8)
    ax.set_ylabel(r"interval ratio $\delta_n$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps({"flips_-0.148": f148, "superstable_-0.148": s148, "superstable_ranges": list(zip(lo, hi)),
                      "flips_-0.156": f156, "superstable_-0.156": s156,
                      "superstable_-0.156_ranges": list(zip(lo6, hi6)) if s156 else [], "flips_1.5B": q15,
                      "out": args.out}))


if __name__ == "__main__":
    main()
