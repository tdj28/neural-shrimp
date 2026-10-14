"""The network and readout factors of tools/gain_split.py add up to the exponent."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Qwen2.5-0.5B-Instruct not cached")
def test_factors_add_up_to_the_exponent():
    import explore_llm_soft_loop as esl
    import gain_split as gs

    loop = esl.SoftLoop("qwen0.5b", 0, 8, "open", ela.OPENER, 1, "float64", "cpu")
    cells = gs.run(loop, np.array([0.0, 0.2]), transient=20, record=30, eps=1e-6)
    for c in cells:
        assert abs(c["log_g_net"] + c["log_g_read"] + c["boundary_term"] - c["lyapunov"]) < 1e-9
