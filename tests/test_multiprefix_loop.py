"""MultiPrefixSoftLoop: cells with different prompts (and prefix lengths) in one batch reproduce, row by row, the
single-prefix SoftLoop with each prompt as its prefix (behavior_prompts.label's per-prompt path)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"
needs_model = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")

PROMPTS = (0, 50, 101, 104)      # docs/bp-tasks.json: a sum, a product and two NQ; prefixes of 46, 44, 49, 58 tokens
LOG_TS = [0.25, 0.40]
EPS = [1e-6, 1e-9]


class ToyLoop:
    """A smooth nonlinear map of (B, 1, 2) states whose rows are independent, with the temperature as a gain."""

    def __init__(self):
        import torch

        self.torch, self.state_dtype, self.device = torch, torch.float64, torch.device("cpu")
        self.A = torch.tensor([[1.2, 0.3], [-0.4, 0.9]], dtype=torch.float64)

    def prepare(self, rows):
        return self.torch.full((len(rows), 1, 2), 0.3, dtype=self.state_dtype)

    def step(self, X, temps):
        T = self.torch.as_tensor(temps, dtype=self.state_dtype)[:, None, None]
        Y = self.torch.tanh(X @ self.A.T * T)
        y = Y[:, 0, 0]
        return Y, self.torch.softmax(self.torch.stack([y, -y, 0 * y], -1), -1)


def test_twin_groups_reproduce_separate_calls():
    """Cells drawn in groups get the twins of separate calls on those groups (without the groups, the later cells
    would get other directions and other finite-horizon exponents); the default is one group."""
    import hidden_chaos as hc

    lts = [0.0, 0.1, 0.0, 0.1]
    args = (np.zeros(2), lts, [0.0] * 4, 5, 30, [1e-4, 1e-6], 1e-8)
    grouped = hc.run_batch(ToyLoop(), *args, twin_groups=[2, 2])
    separate = hc.run_batch(ToyLoop(), np.zeros(2), lts[:2], [0.0] * 2, *args[3:]) * 2
    assert np.allclose([r["lyapunov"] for r in grouped], [r["lyapunov"] for r in separate], rtol=0, atol=1e-12)
    ungrouped = hc.run_batch(ToyLoop(), *args)
    assert ungrouped == hc.run_batch(ToyLoop(), *args, twin_groups=[4])
    assert max(abs(a - b) for a, b in zip(ungrouped[2]["lyapunov"], grouped[2]["lyapunov"])) > 1e-6
    with pytest.raises(ValueError):
        hc.run_batch(ToyLoop(), *args, twin_groups=[2, 1])


@pytest.fixture(scope="module")
def loops():
    import behavior_prompts as bp
    import explore_llm_soft_loop as esl

    ref = esl.SoftLoop("qwen0.5b", 12, 4, "open", ela.OPENER, 1, dtype="float64")
    multi = esl.MultiPrefixSoftLoop.from_loop(ref)            # the same weights, no second copy
    items = {it["prompt"]: it for it in json.loads((ROOT / "docs/bp-tasks.json").read_text())["items"]}
    ids = [bp.prefix_ids(ref.sm.tok, items[p]) for p in PROMPTS]
    assert len({len(p) for p in ids}) == len(ids)            # four different prefix lengths
    return ref, multi, ids


def use_prefix(ref, ids):
    """The per-prompt path of behavior_prompts.label: overwrite the single prefix."""
    ref.prefix_ids = ref.torch.tensor([ids], device=ref.device)
    ref.L0 = ref.prefix_ids.shape[1]


@needs_model
def test_logits_and_tangents_match_per_prompt_loop(loops):
    """Start windows are identical, and the logits of a step after the padded prefixes (with per-row steering in the
    prefix pass, at a perturbed window) equal the single-prefix logits to rounding (|logits| about 20); so do the
    forward-mode tangents of step_jvp. The cache is cropped back to the padded prefix length after every step."""
    ref, multi, ids = loops
    torch = ref.torch
    n, d = len(ids), ref.E.shape[1]
    rng = np.random.default_rng(0)
    steer = rng.normal(size=(n, d)) * 0.3
    pert = torch.as_tensor(rng.normal(size=(n, 1, d)) * 1e-2)
    V = torch.as_tensor(rng.normal(size=(n, 1, d)))
    temps = np.full(n, 10 ** 0.25)
    multi.set_prefixes(ids)
    X0 = multi.prepare(np.zeros((n, d)))
    z0 = multi.logits(X0)
    X1 = multi.prepare(steer) + pert
    z1 = multi.logits(X1)
    Y1, _, J1 = multi.step_jvp(X1, temps, V)
    assert multi.L0 == max(len(p) for p in ids)
    assert multi.cache.get_seq_length() == multi.L0
    for g, p in enumerate(ids):
        use_prefix(ref, p)
        X = ref.prepare(np.zeros((1, d)))
        assert torch.equal(X0[g], X[0])
        assert float((z0[g] - ref.logits(X)[0]).abs().max()) < 1e-11
        X = ref.prepare(steer[g:g + 1]) + pert[g:g + 1]
        assert float((z1[g] - ref.logits(X)[0]).abs().max()) < 1e-11
        Y, _, J = ref.step_jvp(X, temps[:1], V[g:g + 1])
        assert float((Y1[g] - Y[0]).abs().max()) < 1e-13
        assert float((J1[g] - J[0]).norm() / J[0].norm()) < 1e-10
    assert torch.equal(z1, multi.logits(X1))                  # autonomous: the cache is back at the prefixes
    with pytest.raises(ValueError):                           # rows not a multiple of the cells
        multi.prepare(np.zeros((n + 1, d)))


@needs_model
def test_window_of_two_matches_per_prompt_loop(loops):
    """A window of K = 2 fed-back embeddings (positions L_r and L_r + 1, causal within the window) after the padded
    prefixes gives the single-prefix logits to rounding."""
    import copy

    import explore_llm_soft_loop as esl

    ref, _, ids = loops
    torch = ref.torch
    ref2 = copy.copy(ref)                                    # the same model, window 2
    ref2.K = 2
    multi = esl.MultiPrefixSoftLoop.from_loop(ref2)
    n, d = len(ids), ref.E.shape[1]
    pert = torch.as_tensor(np.random.default_rng(1).normal(size=(n, 2, d)) * 1e-2)
    multi.set_prefixes(ids)
    z = multi.logits(multi.prepare(np.zeros((n, d))) + pert)
    for g, p in enumerate(ids):
        use_prefix(ref2, p)
        X = ref2.prepare(np.zeros((1, d)))
        assert X.shape == (1, 2, d)
        assert float((z[g] - ref2.logits(X + pert[g:g + 1])[0]).abs().max()) < 1e-11


@needs_model
def test_rotary_tables_see_true_positions(loops):
    """RoPE scores depend on relative positions only, so equal logits cannot tell true positions from a uniformly
    shifted row: record the position ids that reach the rotary module. Prefix tokens sit at 0 .. L_r - 1 of their own
    prompt (the padding before them is masked) and the fed-back token at L_r, every step."""
    ref, multi, ids = loops
    seen = []

    def hook(module, args, kwargs):
        seen.append((kwargs.get("position_ids", args[1] if len(args) > 1 else None)).clone())

    handle = multi.model.model.rotary_emb.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        multi.set_prefixes(ids)
        X = multi.prepare(np.zeros((2 * len(ids), ref.E.shape[1])))      # two rows per cell
        for _ in range(2):
            X, _ = multi.step(X, np.full(2 * len(ids), 10 ** 0.25))
    finally:
        handle.remove()
    lengths = [len(p) for p in ids] * 2
    Lp = max(lengths)
    prefix, steps = seen[0], seen[1:]
    assert prefix.shape == (2 * len(ids), Lp) and len(steps) == 2
    for r, L in enumerate(lengths):
        assert prefix[r, Lp - L:].tolist() == list(range(L))
        assert multi.prefix_mask[r].tolist() == [0] * (Lp - L) + [1] * L
        assert all(s[r].tolist() == [L] for s in steps)


@needs_model
def test_labels_match_per_prompt_path(loops):
    """hidden_chaos labels of all cells in one batch (twin directions drawn per prompt) agree with one prompt at a time:
    identical periods, classes and leading tokens, exponents equal up to rounding. Rounding differences of about
    1e-15 in the state cannot be removed (the padded attention and the larger matrix products sum in another order)
    and are amplified as after any other change of the batch: by the local growth in chaotic cells, and relative to
    the twin separation at the 1e-9 offset. Measured on 12 cells of six prompts at transient 30, record 40 (8
    threads): the existing path, repeated, is bitwise reproducible, but against itself with duplicated cells (the
    same twins, a larger batch) its exponents differ by up to 2.4e-9 (offset 1e-6) and 6.9e-7 (offset 1e-9) in
    non-chaotic cells and 1.6e-5 in chaotic ones; the batched loop differs from it by up to 4.0e-9, 1.9e-6 and 7.2e-5.
    The bounds below leave a margin of more than ten over those."""
    import hidden_chaos as hc

    ref, multi, ids = loops
    v0 = np.zeros(ref.E.shape[1])
    transient, record, tol = 10, 20, 1e-8
    seq = []
    for p in ids:
        use_prefix(ref, p)
        seq += hc.run_batch(ref, v0, LOG_TS, [0.0] * len(LOG_TS), transient, record, EPS, tol)
    multi.set_prefixes([p for p in ids for _ in LOG_TS])
    bat = hc.run_batch(multi, v0, LOG_TS * len(ids), [0.0] * (len(LOG_TS) * len(ids)), transient, record, EPS, tol,
                       twin_groups=[len(LOG_TS)] * len(ids))
    assert len(bat) == len(seq)
    for c, (a, b) in enumerate(zip(seq, bat)):
        assert a["log10_temperature"] == b["log10_temperature"]
        assert a["period"] == b["period"] and hc.classify(a) == hc.classify(b), c
        assert a["lead_tokens"] == b["lead_tokens"], c
        chaotic = max(a["lyapunov"]) > 0
        bounds = (1e-3, 1e-3) if chaotic else (1e-7, 1e-4)        # offsets 1e-6 and 1e-9
        for x, y, bound in zip(a["lyapunov"], b["lyapunov"], bounds):
            assert abs(x - y) < bound, f"cell {c}: exponents {a['lyapunov']} against {b['lyapunov']}"
