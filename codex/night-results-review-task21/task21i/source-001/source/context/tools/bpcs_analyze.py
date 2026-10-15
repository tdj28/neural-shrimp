"""Analysis of run BPCS (steered arm of the behaviour test; research log, 15 Oct, BPCS entries), written before its
data: does moving a prompt's loop into chaos by a small steering push change the accuracy of its ordinary answers?

Units: the first 1000 BPC prompts under four conditions (unsteered, from BPC; philosophy, hostility-vs-kindness and a
norm-matched random direction at alpha 0.01, block 14). Per prompt and condition: the loop label at log10 T 0.40
(chaotic: visible or hidden chaos; periodic; other labels excluded) and y = fraction of 10 sampled answers correct.

Primary: least squares y = beta * chaotic + prompt fixed effects + condition fixed effects; beta with a 95 % bootstrap
interval over prompts (10,000) and a permutation test that permutes the labels among each prompt's conditions
(10,000). Only prompts whose label differs between conditions inform beta; their number is reported. Secondary: the
edge bins of BPC (deep periodic lambda <= -0.3, edge |lambda| <= 0.1, deep chaotic lambda >= 0.3) on the accuracy
residual after both sets of fixed effects.

Usage: python tools/bpcs_analyze.py --out-json PATH (run names as in the BPCS entries)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/bp"
LT = "0.4"
CONDITIONS = {"unsteered": ("bpc-label-*", "bpc-gen2"), "philosophy": ("bpcs-philos-label-*", "bpcs-philos-gen"),
              "hostility": ("bpcs-hostil-label-*", "bpcs-hostil-gen"), "random": ("bpcs-random-label-*", "bpcs-random-gen")}


def within(y, prompt, cond, ncond):
    """Residualise y on prompt and condition fixed effects (alternating projections)."""
    r = y.astype(float).copy()
    for _ in range(50):
        for g in (prompt, cond):
            means = np.bincount(g, weights=r) / np.maximum(np.bincount(g), 1)
            r = r - means[g]
    return r


def fe_beta(y, x, prompt, cond, ncond):
    ry, rx = within(y, prompt, cond, ncond), within(x.astype(float), prompt, cond, ncond)
    den = float(rx @ rx)
    return float(ry @ rx / den) if den > 0 else float("nan")


def primary(y, x, prompt, cond, ncond, n_boot=10000, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    est = fe_beta(y, x, prompt, cond, ncond)
    groups = {}
    for i, p in enumerate(prompt):
        groups.setdefault(p, []).append(i)
    keys = list(groups)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        newp = np.concatenate([[j] * len(groups[keys[k]]) for j, k in enumerate(pick)])
        boots.append(fe_beta(y[idx], x[idx], newp, cond[idx], ncond))
    perm = 0
    for _ in range(n_perm):
        xp = x.copy()
        for g in groups.values():
            xp[g] = rng.permutation(x[g])
        perm += abs(fe_beta(y, xp, prompt, cond, ncond)) >= abs(est)
    varying = sum(1 for g in groups.values() if len(set(x[g])) > 1)
    return {"beta": est, "ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
            "p_permutation": (perm + 1) / (n_perm + 1), "prompts": len(groups), "prompts_with_varying_label": varying,
            "rows": int(len(y))}


def load():
    rows = []
    for c, (lab_glob, gen) in CONDITIONS.items():
        labels = {}
        for f in glob.glob(str(DATA / lab_glob / "labels.jsonl")):
            for line in open(f):
                r = json.loads(line)
                labels[r["prompt"]] = r["cells"][0]
        g = json.loads((DATA / gen / "generations.json").read_text())["sampled"][LT]
        for p, cell in labels.items():
            if p >= 1000 or str(p) not in g:
                continue
            s = g[str(p)]
            rows.append({"prompt": p, "condition": c, "class": cell["class"], "exponent": float(cell["lyapunov"][0]),
                         "y": sum(x["correct"] for x in s) / len(s)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    rows = load()
    conds = sorted({r["condition"] for r in rows})
    keep = [r for r in rows if r["class"] in ("periodic", "visible chaos", "hidden chaos")]
    pid = {p: i for i, p in enumerate(sorted({r["prompt"] for r in keep}))}
    y = np.array([r["y"] for r in keep])
    x = np.array([r["class"] != "periodic" for r in keep]).astype(int)
    prompt = np.array([pid[r["prompt"]] for r in keep])
    cond = np.array([conds.index(r["condition"]) for r in keep])
    res = {"conditions": conds, "rows_by_condition": {c: sum(1 for r in keep if r["condition"] == c) for c in conds},
           "chaotic_share_by_condition": {c: float(np.mean([r["class"] != "periodic" for r in keep if r["condition"] == c]))
                                          for c in conds},
           "primary": primary(y, x, prompt, cond, len(conds))}
    allp = {p: i for i, p in enumerate(sorted({r["prompt"] for r in rows}))}
    ya = np.array([r["y"] for r in rows])
    ra = within(ya, np.array([allp[r["prompt"]] for r in rows]), np.array([conds.index(r["condition"]) for r in rows]),
                len(conds))
    ex = np.array([r["exponent"] for r in rows])
    bins = {"deep_periodic": ex <= -0.3, "edge": np.abs(ex) <= 0.1, "deep_chaotic": ex >= 0.3}
    res["edge_bins"] = {k: {"n": int(m.sum()), "mean_residual": float(ra[m].mean()) if m.any() else None}
                        for k, m in bins.items()}
    Path(args.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
