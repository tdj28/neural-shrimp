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

    lt, L, per = [], [], []
    for f in folders:
        p = ela.DATA / "explore" / f / "scan.npz"
        if not p.exists():
            return None
        z = np.load(p)
        lt.append(z["log10_temperature"])
        L.append(z["lyapunov"].ravel())
        per.append(z["period"].ravel())
    return np.concatenate(lt), np.concatenate(L), np.concatenate(per)


def main():
    out_png, out_json = sys.argv[1], sys.argv[2]
    rows = []
    for family, label, params, folders in MODELS:
        got = load(folders)
        if got is None:
            rows.append({"family": family, "model": label, "params_b": params, "missing": True})
            continue
        lt, L, per = got
        chaotic = (L > 1e-3) & (per <= 0)
        core = lt <= 0.3 + 1e-9
        rows.append({"family": family, "model": label, "params_b": params, "missing": False,
                     "temperatures_core": int(core.sum()), "chaotic_core": int(chaotic[core].sum()),
                     "share_core": float(chaotic[core].mean()),
                     "temperatures_all": int(lt.size), "chaotic_all": int(chaotic.sum()),
                     "log10T_range": [float(lt.min()), float(lt.max())],
                     "max_exponent": float(L.max()),
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

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = {"Qwen2.5": ("tab:blue", "o"), "Llama-3.1": ("tab:red", "s"), "Llama-3.3": ("tab:orange", "D"),
              "OLMo-2": ("tab:green", "^"), "Qwen3.5 MoE": ("tab:purple", "*")}
    for fam, (colour, marker) in styles.items():
        pts = sorted((r["params_b"], 100 * r["share_core"]) for r in rows if r["family"] == fam and not r["missing"])
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "-", color=colour, marker=marker, label=fam, ms=7 if marker != "*" else 11)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions)")
    ax.set_ylabel("chaotic temperatures in log10 T [-1, 0.3] (%)")
    ax.set_title("Unsteered soft-token maps: chaos against model size", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250)


if __name__ == "__main__":
    main()
