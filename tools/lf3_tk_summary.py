"""Summaries of runs LF3 (LFM3 soft-token maps) and TK (top-k loops of Qwen2.5-0.5B), from the retrieved folders.

Definitions as in the paper: chaotic = exponent above 1e-3 and no period up to the scan's maximum; unresolved =
neither a period nor a positive exponent; chaotic decades and intermediate-entropy breadth as in scale_summary.py;
median network factor over the run-D cells with mean entropy in [0.3, 6) nats, as in plot_damping.py. TK adds:
irregular = positive exponent and no period in a top-k cell whose twin crossed a top-k set boundary (set_switch_steps
> 0), since such an exponent can contain jumps; and, sampled, the token-flip fraction per step.

Usage: python tools/lf3_tk_summary.py lf3|tk OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import scale_summary as ss  # noqa: E402

P = "llm-soft-loop/pod64/"
EXPLORE = ela.DATA / "explore"


def labels(L, per):
    chaotic = (L > 1e-3) & (per <= 0)
    return chaotic, (per <= 0) & ~chaotic


def lf3():
    out = {}
    for model in ("lfm3-1.7b-base", "lfm3-1.7b"):
        rec = {}
        got = ss.load([P + f"lfm3/{model}-alpha0"])
        if got is None:
            rec["sweep"] = "missing"
        else:
            lt, L, per, H = got
            chaotic, unres = labels(L, per)
            core = lt <= 0.3 + 1e-9
            rec["sweep"] = {"temperatures_core": int(core.sum()), "chaotic_core": int(chaotic[core].sum()),
                            "unresolved_core": int(unres[core].sum()), "temperatures_all": int(lt.size),
                            "chaotic_all": int(chaotic.sum()), "unresolved_all": int(unres.sum()),
                            "max_exponent": float(L.max()), "chaos_decades": ss.decades(lt, chaotic),
                            "intermediate_entropy_decades": ss.decades(lt, (H > 1) & (H < 10)),
                            "chaotic_log10T": [round(float(x), 4) for x in lt[chaotic]],
                            "chaotic_exponents": [round(float(x), 3) for x in L[chaotic]]}
        for part in ("phil", "host"):
            f = EXPLORE / P / f"lfm3/{model}-{part}" / "scan.npz"
            if not f.exists():
                rec[part] = "missing"
                continue
            z = np.load(f)
            L, per = z["lyapunov"], z["period"]
            chaotic, unres = labels(L, per)
            rec[part] = {"cells": int(L.size), "chaotic": int(chaotic.sum()), "share": float(chaotic.mean()),
                         "unresolved": int(unres.sum()), "max_exponent": float(L.max()),
                         "periods": {int(q): int((per == q).sum()) for q in np.unique(per)}}
        f = EXPLORE / P / f"damping/d1/{model}" / "damping.json"
        if f.exists():
            cells = json.loads(f.read_text())["cells"]
            H = np.array([c["mean_entropy"] for c in cells])
            g = np.array([c["mean_log_g_net"] for c in cells])
            sel = (H >= 0.3) & (H < 6)
            rec["damping"] = {"median_g_net_0.3_6": float(np.median(g[sel])) if sel.any() else None,
                              "cells_0.3_6": int(sel.sum()),
                              "max_closure": max(c["max_closure"] for c in cells)}
        else:
            rec["damping"] = "missing"
        out[model] = rec
    return out


def tk():
    out = {"soft": {}, "sampled": {}, "divergence": {}}
    for k in (0, 1, 2, 3, 5, 10, 20, 50):
        d = EXPLORE / P / f"topk/soft-k{k}"
        if not (d / "scan.npz").exists():
            out["soft"][k] = "missing"
            continue
        z, t = np.load(d / "scan.npz"), np.load(d / "topk.npz")
        L, per, sw = z["lyapunov"], z["period"], t["set_switch_steps"]
        chaotic, unres = labels(L, per)
        irregular = chaotic & (sw > 0)
        rec = {}
        for j, alpha in enumerate(z["alpha"]):
            rec[f"alpha={alpha:.4f}"] = {"temperatures": int(L.shape[0]), "chaotic": int((chaotic & ~irregular)[:, j].sum()),
                                         "irregular": int(irregular[:, j].sum()), "unresolved": int(unres[:, j].sum()),
                                         "periodic": int((per[:, j] > 0).sum()), "max_exponent": float(L[:, j].max()),
                                         "cells_with_set_switches": int((sw[:, j] > 0).sum()),
                                         "chaos_decades": ss.decades(z["log10_temperature"], (chaotic & ~irregular)[:, j])}
        out["soft"][k] = rec
    for name in ["k0", "k2", "k3", "k5", "k10", "k20", "k50", "k20-eps1e-2", "k20-eps1e-4", "k20-eps1e-8"]:
        d = EXPLORE / P / f"topk/sampled-{name}"
        if not (d / "topk.npz").exists():
            out["sampled"][name] = "missing"
            continue
        z, t = np.load(d / "scan.npz"), np.load(d / "topk.npz")
        flips = t["token_flip_steps"] / int(t["samples"])
        per = z["period"]
        out["sampled"][name] = {f"alpha={a:.4f}": {"mean_flip_fraction": float(flips[:, j].mean()),
                                                   "max_flip_fraction": float(flips[:, j].max()),
                                                   "periodic_cells": int((per[:, j] > 0).sum()),
                                                   "max_exponent": float(z["lyapunov"][:, j].max())}
                                for j, a in enumerate(z["alpha"])}
        out["sampled"][name]["eps"] = float(t["eps"])
    for name in ("soft", "soft-k20", "sampled-k20", "sampled"):
        f = EXPLORE / P / f"topk/divergence-{name}" / "divergence.json"
        out["divergence"][name] = json.loads(f.read_text())["summary"] if f.exists() else "missing"
    return out


def main():
    which, path = sys.argv[1], sys.argv[2]
    rec = lf3() if which == "lf3" else tk()
    Path(path).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:6000])


if __name__ == "__main__":
    main()
