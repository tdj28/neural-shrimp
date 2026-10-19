"""Figure and summary for a chaotic cell whose leading-token sequence repeats exactly (tools/hidden_chaos.py cell
--trace): the states split by the phase of the repeating text, their bands in a two-dimensional principal-component
projection, and how much the leading token's probability varies within each phase.

Usage: python tools/plot_hidden_bands.py TRACE_DIR OUT.png [--title TITLE]
       (TRACE_DIR explore-relative, containing trace.npz and hidden-chaos.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def bands(states, lead, q):
    """Per phase of the period-q text: centroid, radius (largest distance to the centroid) and size."""
    phase = np.arange(len(lead)) % q
    out = []
    for k in range(q):
        S = states[phase == k]
        c = S.mean(0)
        out.append({"phase": k, "centroid": c, "radius": float(np.max(np.linalg.norm(S - c, axis=1))),
                    "rms": float(np.sqrt(np.mean(np.sum((S - c) ** 2, axis=1)))), "n": int(len(S))})
    return phase, out


def return_map_fit(pk, q, degree=8):
    """Polynomial fit of the return map p_n -> p_{n+1} of one phase's probability; its residual relative to the range,
    its Lyapunov exponent (mean log |f'| along the data), and its critical points inside the data range."""
    x, y = pk[:-1], pk[1:]
    lo, rng = x.min(), x.max() - x.min()
    c = np.polynomial.polynomial.polyfit((x - lo) / rng, y, degree)
    f = lambda t: np.polynomial.polynomial.polyval((np.asarray(t) - lo) / rng, c)
    dc = np.polynomial.polynomial.polyder(c)
    fprime = np.polynomial.polynomial.polyval((x - lo) / rng, dc) / rng
    roots = [float(r.real) * rng + lo for r in np.polynomial.polynomial.polyroots(dc)
             if abs(r.imag) < 1e-9 and 0 <= r.real <= 1]
    curv = [float(np.polynomial.polynomial.polyval((r - lo) / rng, np.polynomial.polynomial.polyder(dc)) / rng ** 2)
            for r in roots]
    return {"f": f, "degree": degree, "residual_rel": float(np.sqrt(np.mean((y - f(x)) ** 2)) / (y.max() - y.min())),
            "exponent_per_repetition": float(np.mean(np.log(np.abs(fprime)))), "critical_points": roots,
            "curvature_at_critical_points": curv}


def main():
    import hidden_chaos as hc
    import explore_llm_attractors as ela

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("out")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    base = ela.DATA / "explore" / args.trace_dir
    tr = np.load(base / "trace.npz")
    js = json.loads((base / "hidden-chaos.json").read_text())
    info, model = js["cells"][0], js["args"]["model"]
    lead, pmax, ent, states = tr["lead"], tr["pmax"], tr["entropy"], tr["states"]
    q = hc.symbolic_period(lead.tolist())
    if q == 0:
        raise SystemExit("the leading-token sequence is not periodic: not a repeating-text cell")
    phase, B = bands(states, lead, q)
    cents = np.array([b["centroid"] for b in B])
    sep = min(float(np.linalg.norm(cents[i] - cents[j])) for i in range(q) for j in range(i + 1, q))
    scale = float(np.mean(np.linalg.norm(states, axis=1)))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ela.MODELS[model], cache_dir=ela.CACHE)
    text = [tok.decode([int(t)]) for t in lead[:q]]
    per_phase = [{"phase": k, "token": text[k], "radius_rel": B[k]["radius"] / scale, "rms_rel": B[k]["rms"] / scale,
                  "pmax_min": float(pmax[phase == k].min()), "pmax_max": float(pmax[phase == k].max()),
                  "entropy_min": float(ent[phase == k].min()), "entropy_max": float(ent[phase == k].max())}
                 for k in range(q)]
    summary = {"log10_temperature": info["log10_temperature"], "alpha": info["alpha"], "lyapunov": info["lyapunov"],
               "eps": info["eps"], "state_period": info["period"], "text_period": q, "text": text,
               "min_centroid_separation_rel": sep / scale, "max_band_radius_rel": max(b["radius"] for b in B) / scale,
               "per_phase": per_phase, "steps": int(len(lead))}
    widest = int(np.argmax([b["radius"] for b in B]))
    fit = return_map_fit(pmax[phase == widest], q)
    summary["return_map"] = {"phase": widest, "token": text[widest], **{k: v for k, v in fit.items() if k != "f"},
                             "full_map_exponent_per_repetition": [float(l) * q for l in info["lyapunov"]]}
    (base / "bands-summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_phase"}, ensure_ascii=False))
    for r in per_phase:
        print(json.dumps(r, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = states - states.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 4.8))
    # Okabe-Ito colours (told apart under colour-vision deficiencies), ordered so that the closest bands of the
    # period-7 cell differ most, also in grey print; one marker per phase as well
    colors = ["#0072B2", "#CC79A7", "#000000", "#56B4E9", "#E69F00", "#D55E00", "#009E73"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    cmap = lambda k: colors[k % len(colors)]
    for k in range(q):
        m = phase == k
        ax.scatter(P[m, 0], P[m, 1], s=36, marker=markers[k % len(markers)], linewidths=0, color=cmap(k),
                   label=f"{k}: " + repr(text[k]).strip("'"), rasterized=True)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend(title="phase: leading token", fontsize=7, title_fontsize=8, markerscale=1, loc="best", frameon=False)
    ax.set_title(f"(a) all states, by phase of the repeating text (period {q})", fontsize=9)
    big = int(np.argmax([b["radius"] for b in B]))          # the widest band, on its own principal components
    Sb = states[phase == big]
    Xb = Sb - Sb.mean(0)
    _, sb, Vb = np.linalg.svd(Xb, full_matrices=False)
    Pb = Xb @ Vb[:2].T
    bx.scatter(Pb[:, 0], Pb[:, 1], s=3, color=cmap(big), rasterized=True)
    bx.set_xlabel("band principal component 1")
    bx.set_ylabel("band principal component 2")
    bx.set_title(f"(b) inside one band: phase {big}, leading token " + repr(text[big]), fontsize=9)
    pk = pmax[phase == big]
    cx.scatter(pk[:-1], pk[1:], s=4, color=cmap(big), rasterized=True)
    fit = return_map_fit(pk, q)
    xs = np.linspace(pk.min(), pk.max(), 400)
    cx.plot(xs, fit["f"](xs), color="k", lw=0.8)
    lam = float(np.mean(info["lyapunov"])) * q
    cx.xaxis.set_major_locator(plt.MaxNLocator(5))
    cx.text(0.27, 0.05, f"degree-{fit['degree']} fit: residual {fit['residual_rel']:.0e} of range\n"
            f"exponent of the fitted map {fit['exponent_per_repetition']:.3f} per repetition\n"
            f"full map (twins) {lam:.3f} per {q} steps", transform=cx.transAxes, fontsize=7.5)
    cx.set_xlabel(f"probability of {repr(text[big])} at repetition n")
    cx.set_ylabel(f"... at repetition n+1")
    cx.set_title("(c) the text repeats; this probability does not", fontsize=9)
    fig.suptitle(args.title or (f"Qwen2.5-0.5B-Instruct, hostility direction (layer 12), alpha {info['alpha']}, "
                                f"log10 T {info['log10_temperature']:.5f}: largest exponent "
                                f"{min(info['lyapunov']):.3f}"), fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=250)


if __name__ == "__main__":
    main()
