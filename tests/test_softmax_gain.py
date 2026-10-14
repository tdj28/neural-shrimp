import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import softmax_gain as sg


def test_covariance_matches_explicit_embedding_covariance():
    rng = np.random.default_rng(1)
    E = torch.as_tensor(rng.normal(size=(50, 6)))
    p = torch.softmax(torch.as_tensor(rng.normal(size=50)) * 3, dim=0)
    S = sg.covariance(SimpleNamespace(E=E, torch=torch), p)
    mean = (p[:, None] * E).sum(0)
    explicit = sum(p[t] * torch.outer(E[t] - mean, E[t] - mean) for t in range(50))
    assert torch.allclose(S, explicit, atol=1e-12)


def test_softmax_jacobian_factorises_for_tied_linear_readout():
    # F(x) = E^T softmax(E h(x) / T) with h linear: DF = (1/T) Sigma_p J_h
    rng = np.random.default_rng(2)
    E = torch.as_tensor(rng.normal(size=(40, 5)))
    A = torch.as_tensor(rng.normal(size=(5, 5)))
    T = 0.7
    x = torch.as_tensor(rng.normal(size=5))
    F = lambda y: torch.softmax(E @ (A @ y) / T, dim=0) @ E
    J = torch.autograd.functional.jacobian(F, x)
    p = torch.softmax(E @ (A @ x) / T, dim=0)
    S = sg.covariance(SimpleNamespace(E=E, torch=torch), p)
    assert torch.allclose(J, S @ A / T, atol=1e-12)


def test_rank_of_covariance_is_below_support_size():
    rng = np.random.default_rng(3)
    E = torch.as_tensor(rng.normal(size=(30, 8)))
    p = torch.zeros(30, dtype=torch.float64)
    p[[2, 7, 11]] = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    ev = torch.linalg.eigvalsh(sg.covariance(SimpleNamespace(E=E, torch=torch), p))
    assert (ev > 1e-10).sum() == 2
