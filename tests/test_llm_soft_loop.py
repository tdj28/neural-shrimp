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
    import explore_llm_soft_loop as esl
    return esl.SoftLoop("qwen0.5b", 12, 4, "open", "Hello!", 2, dtype="float64")


def test_cached_step_matches_full_recompute(loop):
    torch = loop.torch
    rng = np.random.default_rng(0)
    d = loop.E.shape[1]
    steer = rng.normal(size=(2, d)) * 0.5
    X = loop.prepare(steer) + torch.as_tensor(rng.normal(size=(2, loop.K, d)) * 1e-2)
    z_cached = loop.logits(X)
    assert loop.cache.get_seq_length() == loop.L0          # cropped back to the prefix
    prefix = loop.model.get_input_embeddings()(loop.prefix_ids)[0].to(torch.float64)
    full = torch.cat([prefix.repeat(2, 1, 1), X], dim=1)
    loop.sm.steer = torch.as_tensor(steer)[:, None, :]
    try:
        with torch.no_grad():
            z_full = loop.model(inputs_embeds=full).logits[:, -1, :].to(torch.float64)
    finally:
        loop.sm.steer = None
    assert torch.allclose(z_cached, z_full, atol=1e-8, rtol=0)
    z_again = loop.logits(X)                                 # the map is autonomous: same input, same output
    assert torch.equal(z_cached, z_again)


def test_low_temperature_feeds_back_the_greedy_token(loop):
    torch = loop.torch
    X = loop.prepare(np.zeros((1, loop.E.shape[1])))
    z = loop.logits(X)
    X1, p = loop.step(X, np.array([1e-4]))
    greedy = int(torch.argmax(z[0]))
    assert torch.allclose(X1[0, -1], loop.E[greedy], atol=1e-10)
