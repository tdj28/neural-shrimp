import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")


@pytest.fixture(scope="module")
def sm():
    import explore_llm_phase_map as elpm
    return elpm.SteeredModel("qwen0.5b", 12, 4)


def test_zero_steering_changes_nothing(sm):
    torch = sm.torch
    enc = sm.tok(["Hello there, how are you?"], return_tensors="pt")
    with torch.no_grad():
        base = sm.model(**enc).logits
        sm.steer = torch.zeros((1, 1, sm.model.config.hidden_size), dtype=torch.float64)
        try:
            zero = sm.model(**enc).logits
        finally:
            sm.steer = None
    assert torch.equal(base, zero)


def test_steering_shifts_the_layer_output_by_the_vector(sm):
    torch = sm.torch
    enc = sm.tok(["Hello there, how are you?"], return_tensors="pt")
    rng = np.random.default_rng(0)
    v = rng.normal(size=sm.model.config.hidden_size)
    with torch.no_grad():
        base = sm.model(**enc, output_hidden_states=True).hidden_states[13]
        sm.steer = torch.as_tensor(v)[None, None, :]
        try:
            moved = sm.model(**enc, output_hidden_states=True).hidden_states[13]
        finally:
            sm.steer = None
    np.testing.assert_allclose((moved - base).double().numpy()[0], np.tile(v, (base.shape[1], 1)), atol=1e-4)
