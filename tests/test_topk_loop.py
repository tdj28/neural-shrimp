import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")


@pytest.fixture(scope="module")
def loop():
    import topk_loop
    return topk_loop.TopKLoop("qwen0.5b", 12, 4, "open", "Hello!", 1, dtype="float64")


def start(loop, n, seed=0, scale=1e-2):
    torch = loop.torch
    rng = np.random.default_rng(seed)
    X = loop.prepare(np.zeros((n, loop.E.shape[1])))
    return X + torch.as_tensor(rng.normal(size=tuple(X.shape)) * scale)


def test_whole_vocabulary_is_the_soft_token_map(loop):
    import explore_llm_soft_loop as esl

    torch = loop.torch
    loop.top_k, loop.sample = 0, False
    X = start(loop, 2)
    temps = np.array([0.8, 1.3])
    Y, p = loop.step(X, temps)
    Y0, p0 = esl.SoftLoop.step(loop, X, temps)
    assert torch.equal(Y, Y0) and torch.equal(p, p0)


def test_top1_is_greedy_and_topk_weights_live_on_k_tokens(loop):
    torch = loop.torch
    X = start(loop, 2, seed=1)
    z = loop.logits(X)
    loop.top_k, loop.sample = 1, False
    Y, p = loop.step(X, np.array([0.7, 2.0]))
    for r in range(2):
        assert torch.equal(Y[r, -1], loop.E[int(torch.argmax(z[r]))])
    loop.top_k = 5
    Y, p = loop.step(X, np.array([0.7, 2.0]))
    vals, idx = torch.topk(z, 5, dim=-1)
    q = torch.softmax(vals / torch.tensor([[0.7], [2.0]], dtype=torch.float64), dim=-1)
    assert ((p > 0).sum(-1) == 5).all() and torch.allclose(p.sum(-1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(Y[:, -1], (q[:, :, None] * loop.E[idx]).sum(1), atol=1e-13, rtol=0)


def test_sampled_rows_with_shared_noise_draw_the_same_token_from_the_top_k(loop):
    torch = loop.torch
    loop.top_k, loop.sample, loop.paired, loop.transient = 3, True, True, 0
    X = start(loop, 1, seed=2)
    XX = X.repeat(4, 1, 1)                     # two cells, each with an identical twin (offset eps = 0)
    loop.prepare(np.zeros((4, loop.E.shape[1])))
    loop.eps = 0.0
    z = loop.logits(XX)
    top = set(torch.topk(z[0], 3).indices.tolist())
    for _ in range(5):
        Y, p = loop.step(XX, np.array([1.5] * 4))
        tok = [int(torch.argmin((loop.E - Y[r, -1]).norm(dim=-1))) for r in range(4)]
        assert tok[0] == tok[2] and tok[1] == tok[3] and set(tok) <= top
    assert loop.flip.sum() == 0
    loop.paired, loop.sample, loop.eps = True, False, 1e-6


def test_twins_across_a_set_boundary_are_counted(loop):
    """A twin far from its trajectory has a different top-k set at some step; identical twins never do."""
    torch = loop.torch
    loop.top_k, loop.sample, loop.paired, loop.transient = 2, False, True, 0
    X = start(loop, 1, seed=3)
    loop.prepare(np.zeros((2, loop.E.shape[1])))
    same = torch.cat([X, X])
    loop.step(same, np.array([1.0, 1.0]))
    assert loop.switch.sum() == 0
    other = torch.cat([X, loop.E[[0]][:, None, :]])
    loop.step(other, np.array([1.0, 1.0]))
    assert loop.switch.sum() == 1
