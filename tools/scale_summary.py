"""Scaling series (run SC; predictions frozen in docs/scaling-predictions.md): per model, the unsteered chaotic share
of the temperatures in log10 T [-1, 0.3] (the common range) and over each sweep's full range, the largest exponent,
the scoring of predictions P1 and P2, and a figure of the share against parameter count per family.

Chaotic = exponent above 1e-3 and no period (each sweep's own maximum period). Sweeps missing on disk are reported as
missing and left out of the scoring of the predictions they belong to (never imputed).

Usage: python tools/scale_summary.py OUT.png OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
P = "llm-soft-loop/pod64/"
# family, label, parameters (billions), sweep folders (explore-relative; several are joined)
MODELS = [
    ("Qwen2.5", "0.5B", 0.49, [P + "scan64/q05-alpha0"]),
    ("Qwen2.5", "1.5B", 1.54, [P + "big/qwen1.5b-alpha0-wide"]),
    ("Qwen2.5", "3B", 3.09, [P + "big/qwen3b-alpha0-wide"]),
    ("Qwen2.5", "7B", 7.62, [P + "controls/q7-alpha0"]),
    ("Qwen2.5", "14B", 14.7, [P + "big/qwen14b-alpha0-wide"]),
    ("Qwen2.5", "32B", 32.5, [P + "big/qwen32b-alpha0-wide"]),
    ("Qwen2.5", "72B", 72.7, [P + "big/qwen72b-alpha0", P + "big/qwen72b-alpha0-highT"]),
    ("Llama-3.1", "8B", 8.03, [P + "llama8b-hires/alpha0"]),
    ("Llama-3.1", "70B", 70.6, [P + "big/llama3.1-70b-alpha0"]),
    ("Llama-3.3", "70B", 70.6, [P + "big/llama3.3-70b-alpha0"]),
    ("OLMo-2", "1B", 1.48, [P + "big/olmo2-1b-alpha0-wide"]),
    ("OLMo-2", "7B", 7.30, [P + "controls/o-alpha0c"]),
    ("OLMo-2", "13B", 13.7, [P + "big/olmo2-13b-alpha0-wide"]),
    ("OLMo-2", "32B", 32.2, [P + "big/olmo2-32b-alpha0-wide"]),
    ("Qwen3.5 MoE", "35B-A3B", 35.0, [P + "controls/m1t-alpha0/scan"]),
]


def load(folders):
    import explore_llm_attractors as ela

    lt, L, per, H = [], [], [], []
    for f in folders:
        p = ela.DATA / "explore" / f / "scan.npz"
        if not p.exists():
            return None
        z = np.load(p)
        lt.append(z["log10_temperature"])
        L.append(z["lyapunov"].ravel())
        per.append(z["period"].ravel())
        H.append(z["entropy"].ravel())
    lt, L, per, H = (np.concatenate(v) for v in (lt, L, per, H))
    o = np.argsort(lt, kind="stable")
    lt, L, per, H = lt[o], L[o], per[o], H[o]
    # joined sweeps can share an endpoint (Qwen2.5-72B: both include log10 T = 0.3); keep the first copy so that every
    # temperature counts once, and stop if the duplicate's labels disagree
    keep = np.ones(len(lt), bool)
    keep[1:] = np.abs(np.diff(lt)) > 1e-9
    for i in np.flatnonzero(~keep):
        if (L[i] > 1e-3 and per[i] <= 0) != (L[i - 1] > 1e-3 and per[i - 1] <= 0):
            raise SystemExit(f"joined sweeps disagree at log10 T {lt[i]}")
    if not np.all(np.isfinite(L[keep])):
        raise SystemExit("non-finite exponent in a sweep")
    return lt[keep], L[keep], per[keep], H[keep]


def decades(lt, mask):
    """Decades of log10 T covered by the cells in mask (each cell weighted by its grid spacing)."""
    return float((np.gradient(lt) * mask).sum())


def latex_rows(rows):
    """Body rows of the paper's scaling table (Table tab:scale)."""
    out = []
    for r in rows:
        name = f"{r['family']}-{r['model']}" if not r["family"].startswith("Qwen3.5") else "Qwen3.5-35B-A3B"
        if r["missing"]:
            out.append(f"{name} & -- & \\multicolumn{{6}}{{c}}{{missing}} \\\\")
            continue
        lo, hi = r["log10T_range"]
        out.append(f"{name} & {r['params_b']:.1f} & {r['chaotic_core']}/{r['temperatures_core']} "
                   f"({100 * r['share_core']:.1f}) & {r['unresolved_core']} & ${lo:.1f}$ to ${hi:.1f}$ "
                   f"& {r['chaos_decades']:.3f} & {r['intermediate_entropy_decades']:.3f} & ${r['max_exponent']:+.2f}$ \\\\")
    return "\n".join(out) + "\n"


def main():
    out_png, out_json = sys.argv[1], sys.argv[2]
    rows = []
    for family, label, params, folders in MODELS:
        got = load(folders)
        if got is None:
            rows.append({"family": family, "model": label, "params_b": params, "missing": True})
            continue
        lt, L, per, H = got
        chaotic = (L > 1e-3) & (per <= 0)
        core = lt <= 0.3 + 1e-9
        mid = (H > 1) & (H < 10)
        rows.append({"family": family, "model": label, "params_b": params, "missing": False,
                     "temperatures_core": int(core.sum()), "chaotic_core": int(chaotic[core].sum()),
                     "unresolved_core": int(((per <= 0) & ~chaotic)[core].sum()),
                     "share_core": float(chaotic[core].mean()),
                     "temperatures_all": int(lt.size), "chaotic_all": int(chaotic.sum()),
                     "log10T_range": [float(lt.min()), float(lt.max())],
                     "max_exponent": float(L.max()),
                     "chaos_decades": decades(lt, chaotic), "intermediate_entropy_decades": decades(lt, mid),
                     "chaotic_log10T": [round(float(x), 4) for x in lt[chaotic]]})
    by = {(r["family"], r["model"]): r for r in rows}

    def share(fam, m):
        r = by.get((fam, m))
        return None if r is None or r["missing"] else r["share_core"]

    p1 = {}
    for fam, ref, larger in (("Qwen2.5", "7B", ("14B", "32B")), ("OLMo-2", "7B", ("13B", "32B"))):
        for m in larger:
            a, b = share(fam, m), share(fam, ref)
            p1[f"{fam} {m} < {ref}"] = None if a is None or b is None else bool(a < b)
    p2 = {}
    for fam, order in (("Qwen2.5", ("7B", "14B", "32B", "72B")), ("OLMo-2", ("7B", "13B", "32B"))):
        vals = [share(fam, m) for m in order]
        p2[fam + " " + " >= ".join(order)] = None if any(v is None for v in vals) else bool(
            all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))
    summary = {"rows": rows, "P1": p1, "P2": p2,
               "P1_holds": None if any(v is None for v in p1.values()) else all(p1.values()),
               "P2_holds": None if any(v is None for v in p2.values()) else all(p2.values())}
    Path(out_json).write_text(json.dumps(summary, indent=1))
    Path(out_json).with_suffix(".tex").write_text(latex_rows(rows))
    for r in rows:
        if r["missing"]:
            print(f"{r['family']:12s} {r['model']:8s} missing")
        else:
            print(f"{r['family']:12s} {r['model']:8s} {r['params_b']:6.2f}B  core {r['chaotic_core']:3d}/{r['temperatures_core']:3d} "
                  f"({100 * r['share_core']:5.1f} %)  all {r['chaotic_all']:3d}/{r['temperatures_all']:3d}  max {r['max_exponent']:+.3f}")
    print(json.dumps({"P1": p1, "P2": p2}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scipy.stats import spearmanr

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"Qwen2.5": ("tab:blue", "o"), "Llama-3.1": ("tab:red", "s"), "Llama-3.3": ("tab:orange", "D"),
              "OLMo-2": ("tab:green", "^"), "Qwen3.5 MoE": ("tab:purple", "*")}
    for fam, (colour, marker) in styles.items():
        got = [r for r in rows if r["family"] == fam and not r["missing"]]
        if not got:
            continue
        pts = sorted((r["params_b"], r["chaos_decades"]) for r in got)
        x, y = zip(*pts)
        ms = 7 if marker != "*" else 11
        ax.plot(x, y, "-", color=colour, marker=marker, label=fam, ms=ms)
        bx.plot([r["intermediate_entropy_decades"] for r in got], [r["chaos_decades"] for r in got], marker,
                color=colour, ms=ms, label=fam)
        for r in got:
            moe = fam.startswith("Qwen3.5")
            bx.annotate("MoE" if moe else r["model"], (r["intermediate_entropy_decades"], r["chaos_decades"]),
                        fontsize=6, xytext=(3, -9) if moe else (3, 2), textcoords="offset points")
    ok = [r for r in rows if not r["missing"]]
    x = [r["intermediate_entropy_decades"] for r in ok]
    y = [r["chaos_decades"] for r in ok]
    rho_b = spearmanr(x, y).statistic
    rho_n = spearmanr(np.log([r["params_b"] for r in ok]), y).statistic
    summary["spearman"] = {"breadth": float(rho_b), "log_params": float(rho_n), "models": len(ok)}
    Path(out_json).write_text(json.dumps(summary, indent=1))
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions)")
    ax.set_ylabel("chaotic decades of log10 T (unsteered sweep)")
    ax.set_title(f"against size (Spearman {rho_n:+.2f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    bx.set_xlabel("decades of log10 T with next-token entropy 1-10 nats (on the orbits)")
    bx.set_title(f"against the intermediate-entropy breadth (Spearman {rho_b:+.2f})", fontsize=10)
    fig.suptitle("Unsteered soft-token maps of %d models: how much of the temperature axis is chaotic" % len(ok),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)


if __name__ == "__main__":
    main()
