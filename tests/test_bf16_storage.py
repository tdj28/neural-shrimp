"""bfloat16 storage with float64 compute (Engine weights="bf16") must reproduce the float64 map exactly."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen2.5-0.5B-Instruct not cached")
def test_bf16_storage_matches_float64_bitwise():
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    v = elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]

    def run(weights):
        loop = esl.SoftLoop("qwen0.5b", 12, 8, "open", ela.OPENER, 1, "float64", "cpu", weights)
        X = loop.prepare((0.071674 * v)[None])
        out = []
        for _ in range(8):
            X, _ = loop.step(X, np.array([10 ** -0.115]))
            out.append(X[0, -1].numpy().copy())
        return np.array(out), loop

    a, _ = run("native")
    b, loop = run("bf16")
    assert loop.dtype == loop.torch.float64
    assert loop.model.lm_head.weight.dtype == loop.torch.float64
    assert {m.weight.dtype for m in loop.model.model.layers[0].modules()
            if isinstance(m, loop.torch.nn.Linear)} == {loop.torch.bfloat16}
    assert (a == b).all()
