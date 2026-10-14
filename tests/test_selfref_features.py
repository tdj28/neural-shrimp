"""Tests for tools/selfref_features.py: the frozen selection rule picks set-specific features that generalize to
the held-out half and drops one that is specific only on the selection half."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import selfref_features as sf  # noqa: E402

CATS = ["ai_identity_disclaimer", "deception_cover_story", "direct_consciousness_claim", "dishonesty_confession",
        "false_self_attribution", "fictional_pretending", "hedged_cautious_style", "honesty_correction",
        "neutral_factual_control", "persona_maintenance", "refusal_safety_disclaimer", "roleplay_persona",
        "self_ref_mindfulness", "tactical_misdirection"]


def corpus(n_per=10, F=200, seed=0):
    rng = np.random.default_rng(seed)
    cats = np.repeat(CATS, n_per)
    M = np.abs(rng.normal(0, 0.05, size=(len(cats), F))).astype(np.float32)
    M[:, 150:] = 0.0                                                  # dead features
    dec = np.isin(cats, sf.SETS["deception"])
    rol = np.isin(cats, sf.SETS["roleplay"])
    M[dec, 7] += 1.0                                                  # deception feature, both halves
    M[rol, 42] += 1.0                                                 # roleplay feature, both halves
    M[np.isin(cats, ["neutral_factual_control"]), 7] += 0.3           # competitor activity, smaller
    pos = np.zeros(len(cats), dtype=int)
    for c in CATS:
        w = np.nonzero(cats == c)[0]
        pos[w] = np.arange(len(w))
    M[dec & (pos % 2 == 0), 99] += 2.0                                # specific on the selection half only
    return M, cats


def test_rule_selects_and_validates():
    M, cats = corpus()
    res = sf.select(M, cats)
    d, r = res["sets"]["deception"], res["sets"]["roleplay"]
    assert 99 in [c["feature"] for c in d["candidates"]] and 99 not in d["kept"]      # fails the held-out half
    assert d["kept"] == [7]
    assert r["kept"][0] == 42
    assert len(d["random_comparators"]) == 24 and all(int(f) < 150 for f in d["random_comparators"])
    assert res["live_features"] == 150
