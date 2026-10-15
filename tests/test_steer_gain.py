"""The readout statistics of tools/steer_gain.py against a dense computation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
torch = pytest.importorskip("torch")
import steer_gain as sg  # noqa: E402


def dense(p, E, T):
    """Eigenvalues of (1/T) E^T (diag p - p p^T) E over the whole vocabulary."""
    return np.linalg.eigvalsh(E.T @ (np.diag(p) - np.outer(p, p)) @ E / T)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trace_and_largest_eigenvalue_match_dense(seed):
    rng = np.random.default_rng(seed)
    V, d, n = 50, 8, 3
    E = rng.normal(size=(V, d))
    z = rng.normal(size=(n, V)) * 2.0
    T = np.array([0.5, 1.0, 2.0])
    p = np.exp(z / T[:, None])
    p /= p.sum(1, keepdims=True)
    F = p @ E
    u = rng.normal(size=(n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = lambda a: torch.tensor(a)                                              # noqa: E731
    r1, tr, gap, u2, change = sg.readout_stats(t(p), t(z), t(F), t(E), t((E ** 2).sum(1)), t(T), t(u), 400)
    for c in range(n):
        ref = dense(p[c], E, T[c])
        assert tr[c].item() == pytest.approx(ref.sum(), rel=1e-10)
        assert r1[c].item() == pytest.approx(ref[-1], rel=1e-8)
        zs = np.sort(z[c])[::-1]
        assert gap[c].item() == pytest.approx((zs[0] - zs[1]) / T[c])
    assert float(change.max()) < 1e-10


def test_stable_rank_counts_competing_tokens():
    """Two tokens sharing the mass with orthogonal embeddings: the readout has rank one (stable rank 1); three equal
    tokens: rank two with equal eigenvalues (stable rank 2). The start vector is random: the all-ones vector lies in
    the readout's null space."""
    E = np.eye(6)
    for k, expect in ((2, 1.0), (3, 2.0)):
        p = np.full((1, 6), 1e-14)
        p[0, :k] = (1 - 6e-14) / k
        z = np.log(p)
        F = p @ E
        t = lambda a: torch.tensor(a)                                          # noqa: E731
        r1, tr, *_ = sg.readout_stats(t(p), t(z), t(F), t(E), t((E ** 2).sum(1)), t(np.array([1.0])),
                                      t(np.random.default_rng(k).normal(size=(1, 6))), 200)
        assert tr[0].item() / r1[0].item() == pytest.approx(expect, rel=1e-6)
