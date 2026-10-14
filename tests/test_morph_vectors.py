"""tools/morph_vectors.py: constant norm along the great circle, exact endpoints, uniform angle steps."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import morph_vectors as mv  # noqa: E402


def test_slerp_constant_norm_and_endpoints():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64) * 3.0, rng.normal(size=64)
    s, omega, frames = mv.slerp_frames(a, b, 11)
    norms = np.array([np.linalg.norm(f) for f in frames])
    assert np.allclose(norms, np.linalg.norm(a), rtol=1e-12)
    assert np.allclose(frames[0], a) and np.allclose(frames[-1], b / np.linalg.norm(b) * np.linalg.norm(a))
    ua = a / np.linalg.norm(a)
    angles = [np.arccos(np.clip(f @ ua / np.linalg.norm(f), -1, 1)) for f in frames]
    assert np.allclose(np.diff(angles), omega / 10, rtol=1e-9)
