"""Building the prefix cache in sub-batches (large models spread over GPUs) must give the same map."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen2.5-0.5B-Instruct not cached")
def test_prefix_cache_in_sub_batches_matches_one_pass():
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    v = elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    loop = esl.SoftLoop("qwen0.5b", 12, 8, "open", ela.OPENER, 1, "float64", "cpu")
    rows = np.array([a * v for a in (0.0, 0.03, 0.06, 0.0717, 0.09)])
    temps = 10.0 ** np.array([-0.3, -0.115, -0.115, -0.115, 0.1])

    def run(prefix_batch):
        loop.prefix_batch = prefix_batch
        X = loop.prepare(rows)
        for _ in range(10):
            X, _ = loop.step(X, temps)
        return X.numpy().copy(), loop.cache.get_seq_length()

    a, la = run(None)
    b, lb = run(2)
    assert la == lb == loop.L0
    assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a))
