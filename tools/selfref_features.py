"""Frozen feature-selection rule for the self-referential dynamics study (docs/selfref-dynamics-design.md).

Input: maxima-SAE.npz from `llm_sae_readout.py profile-csv` on the designed corpus (per-text maximum activation of
every feature). The rule was fixed before any steering run:
- each category's texts are split by their order in the corpus into a selection half (even positions) and a
  held-out half (odd positions);
- per half, m_f[c] is the mean over category c's texts of feature f's per-text maximum;
- for a target set S, margin_f = mean over c in S of m_f[c] minus the largest m_f[c] over the competitor categories
  (all categories outside S and outside EXCLUDED);
- on the selection half, live features (nonzero somewhere) are ranked by margin and the top TOP with a positive
  margin are candidates;
- a candidate is kept only if its held-out margin is positive and ranks within the top HELD_OUT_SHARE of live features.
Sets: deception = deception_cover_story, tactical_misdirection, dishonesty_confession; roleplay = roleplay_persona,
persona_maintenance, fictional_pretending. EXCLUDED for both: the two sets and false_self_attribution. Reported as
comparators: each kept feature's +-3 index neighbours and RANDOM random live features (seed 0) with their held-out
margins.

Usage: python tools/selfref_features.py select MAXIMA_NPZ OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

SETS = {"deception": ("deception_cover_story", "tactical_misdirection", "dishonesty_confession"),
        "roleplay": ("roleplay_persona", "persona_maintenance", "fictional_pretending")}
EXCLUDED = tuple(c for s in SETS.values() for c in s) + ("false_self_attribution",)
TOP, HELD_OUT_SHARE, RANDOM, SEED = 3, 0.01, 24, 0


def category_means(maxima, categories, rows):
    cats = sorted(set(categories))
    return cats, np.stack([maxima[rows & (categories == c)].mean(0) for c in cats])      # (categories, features)


def margins(means, cats, target):
    idx = [cats.index(c) for c in target]
    comp = [k for k, c in enumerate(cats) if c not in target and c not in EXCLUDED]
    if not comp:
        raise SystemExit("no competitor categories")
    return means[idx].mean(0) - means[comp].max(0)


def select(maxima, categories):
    categories = np.asarray(categories)
    n = len(categories)
    position = np.zeros(n, dtype=int)
    for c in set(categories):
        where = np.nonzero(categories == c)[0]
        position[where] = np.arange(len(where))
    sel_rows, held_rows = position % 2 == 0, position % 2 == 1
    cats, m_sel = category_means(maxima, categories, sel_rows)
    _, m_held = category_means(maxima, categories, held_rows)
    missing = [c for s in SETS.values() for c in s if c not in cats]
    if missing:
        raise SystemExit(f"corpus lacks categories {missing}")
    live = np.nonzero(maxima.max(0) > 0)[0]
    rng = np.random.default_rng(SEED)
    out = {"rule": __doc__.split("Usage:")[0].strip(), "texts": int(n), "live_features": int(len(live)), "sets": {}}
    for name, target in SETS.items():
        g_sel, g_held = margins(m_sel, cats, target), margins(m_held, cats, target)
        order = live[np.argsort(-g_sel[live], kind="stable")]
        held_rank = {int(f): r for r, f in enumerate(live[np.argsort(-g_held[live], kind="stable")])}
        limit = HELD_OUT_SHARE * len(live)
        cands = [int(f) for f in order[:TOP] if g_sel[f] > 0]
        kept, report = [], []
        for f in cands:
            ok = bool(g_held[f] > 0 and held_rank[f] < limit)
            report.append({"feature": f, "selection_margin": float(g_sel[f]), "held_out_margin": float(g_held[f]),
                           "held_out_rank": held_rank[f], "kept": ok,
                           "neighbours": {str(f + d): float(g_held[f + d]) for d in (-3, -2, -1, 1, 2, 3)
                                          if 0 <= f + d < maxima.shape[1]}})
            if ok:
                kept.append(f)
        randoms = rng.choice(live, size=min(RANDOM, len(live)), replace=False)
        out["sets"][name] = {"categories": list(target), "candidates": report, "kept": kept,
                             "random_comparators": {str(int(f)): float(g_held[f]) for f in randoms}}
    return out


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "select":
        raise SystemExit(__doc__)
    out = Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f"{out} exists; selections are never overwritten")
    z = np.load(sys.argv[2])
    res = select(z["maxima"], z["category"])
    res["input"] = sys.argv[2]
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {"kept": v["kept"]} for k, v in res["sets"].items()}))


if __name__ == "__main__":
    main()
