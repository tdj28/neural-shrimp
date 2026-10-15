import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bifurcation_curves as bc  # noqa: E402
import cusp_locate as cl  # noqa: E402

torch = pytest.importorskip("torch")


def cubic(x, theta):
    """x -> x + t1 + t2 (x - 1) - (x - 1)^3: fold curve (t1, t2) = (-2 u^3, 3 u^2) with u = x - 1, cusp at x = 1 and
    theta = 0, b = -3 u, c = -1 (the state norm is 1 there, as in the soft-token maps, so relative steps behave)."""
    return x + theta[0] + theta[1] * (x - 1) - (x - 1) ** 3


def traced():
    fam = bc.Family(cubic, (1,), torch.float64)
    records, reason, states = bc.trace_curve(fam, np.array([1.52]), [-0.25, 0.75], "fold", fix=1, direction=1,
                                             steps=200, ds=0.02, ds_max=0.05, bounds=((-1.0, 1.0), (-1.0, 1.0)))
    us = np.array([s[0] - 1 for s in states])
    k = int(np.flatnonzero(np.sign(us[1:]) != np.sign(us[:-1]))[0])      # b = -3u changes sign between k, k+1
    return fam, records, states, k


def test_locates_the_analytic_cusp_with_its_cubic_and_regular_unfolding():
    fam, records, states, k = traced()
    tp = records[k]["parameter_tangent"]
    res = cl.locate(fam, states[k], records[k]["theta"], states[k + 1], records[k + 1]["theta"], fix=1,
                    direction=int(np.sign(tp[0])), rounds=3, per_round=5, param=0, param_scale=1.0,
                    nf_steps=(1e-2, 3e-3, 1e-3))
    fin = res["final"]
    assert res["located"], res["checks"]
    assert max(abs(v) for v in fin["theta"]) < 1e-6 and abs(res["state"][0] - 1) < 1e-4
    assert abs(fin["c"] + 1) < 1e-6 and abs(fin["det_J"] - 1) < 1e-6
    assert all(br["gap"] < prev["gap"] for prev, br in zip(res["brackets"], res["brackets"][1:]))


def test_refuses_a_bracket_without_a_sign_change():
    fam, records, states, k = traced()
    with pytest.raises(ValueError):
        cl.locate(fam, states[k - 3], records[k - 3]["theta"], states[k - 2], records[k - 2]["theta"], fix=1,
                  direction=1, param=0, param_scale=1.0, nf_steps=(1e-2, 3e-3, 1e-3))
