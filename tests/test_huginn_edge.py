"""huginn_edge recovers the analytic stability edge of a relaxed affine map with Huginn's critical pair
(modulus 0.97, 0.156 of a revolution per step): lam_c = 2 (1 - Re mu) / |1 - mu|^2."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_edge as he  # noqa: E402


class AffineLoop:
    def __init__(self, mu_abs, turns):
        th = 2 * math.pi * turns
        self.M = mu_abs * torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float64)
        self.star = torch.tensor([10.0, 10.0], dtype=torch.float64)
        self.torch = torch

    def F(self, s):                     # s: (B, T=1, 2); fixed point star
        return self.star + (s - self.star) @ self.M.T


def args(**kw):
    a = dict(lam_lo=1.0, lam_hi=1.28, coarse_step=0.04, coarse_steps=96, fine_points=6, fine_steps=192, margin=1e-3)
    a.update(kw)
    return SimpleNamespace(**a)


def test_recovers_the_analytic_edge():
    loop = AffineLoop(0.97, 0.156)
    mu = 0.97 * complex(math.cos(2 * math.pi * 0.156), math.sin(2 * math.pi * 0.156))
    lam_c = 2 * (1 - mu.real) / abs(1 - mu) ** 2
    h0 = (loop.star + torch.tensor([1.0, 0.0], dtype=torch.float64))[None, None, :]
    rec = he.edge_of(loop, h0, args())
    import json
    json.dumps(rec)                      # every record must serialize (run HEdev failed on a numpy bool)
    assert rec["outcome"] == "edge located", rec["outcome"]
    assert rec["bracket"][0] <= lam_c <= rec["bracket"][1]
    assert abs(rec["lam_edge"] - lam_c) < 3e-3


def test_unstable_native_point_is_reported_not_hidden():
    loop = AffineLoop(1.01, 0.156)
    h0 = (loop.star + torch.tensor([1.0, 0.0], dtype=torch.float64))[None, None, :]
    rec = he.edge_of(loop, h0, args())
    import json
    json.dumps(rec)
    assert rec["outcome"].startswith("native unstable") and "lam_edge" not in rec
