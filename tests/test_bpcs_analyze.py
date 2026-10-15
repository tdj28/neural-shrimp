"""The BPCS within-prompt analysis recovers a planted label effect and ignores prompt-level confounding."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bpcs_analyze as bs  # noqa: E402


def synthetic(effect, seed=0, n=400, k=4):
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    prompt = np.repeat(np.arange(n), k)
    cond = np.tile(np.arange(k), n)
    # hard prompts are chaotic more often (a confound across prompts), conditions shift labels within prompts
    x = (difficulty[prompt] + 0.8 * rng.normal(size=n * k) + 0.3 * cond > 0.6).astype(int)
    y = np.clip(0.6 - 0.15 * difficulty[prompt] + effect * x + 0.05 * rng.normal(size=n * k), 0, 1)
    return y, x, prompt, cond, k


def test_recovers_planted_within_prompt_effect():
    y, x, p, c, k = synthetic(-0.1)
    r = bs.primary(y, x, p, c, k, n_boot=200, n_perm=200)
    assert r["ci95"][1] < 0 and r["p_permutation"] < 0.05


def test_prompt_confounding_alone_gives_no_effect():
    y, x, p, c, k = synthetic(0.0, seed=1)
    raw = y[x == 1].mean() - y[x == 0].mean()
    assert raw < -0.05
    r = bs.primary(y, x, p, c, k, n_boot=200, n_perm=200)
    assert r["ci95"][0] < 0 < r["ci95"][1]
