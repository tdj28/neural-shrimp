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


def traced_cubic():
    fam = bc.Family(cubic, (1,), torch.float64)
    records, reason, states = bc.trace_curve(fam, np.array([1.52]), [-0.25, 0.75], "fold", fix=1, direction=1,
                                             steps=200, ds=0.02, ds_max=0.05, bounds=((-1.0, 1.0), (-1.0, 1.0)))
    us = np.array([s[0] - 1 for s in states])
    k = int(np.flatnonzero(np.sign(us[1:]) != np.sign(us[:-1]))[0])      # b = -3u changes sign between k, k+1
    return fam, records, states, k


def test_locates_the_analytic_cusp_with_its_cubic_and_regular_unfolding():
    fam, records, states, k = traced_cubic()
    res = cl.locate(fam, states[k], records[k]["theta"], states[k + 1], records[k + 1]["theta"],
                    records[k]["parameter_tangent"], test="cusp", kind="fold", rounds=3, per_round=5, param=0,
                    param_scale=1.0, nf_steps=(1e-2, 3e-3, 1e-3))
    fin = res["final"]
    assert res["located"], res["checks"]
    assert max(abs(v) for v in fin["theta"]) < 1e-6 and abs(res["state"][0] - 1) < 1e-4
    assert abs(fin["c"] + 1) < 1e-6 and abs(fin["det_J"] - 1) < 1e-6
    assert all(br["gap"] < prev["gap"] for prev, br in zip(res["brackets"], res["brackets"][1:]))


def test_reversal_of_the_projection_is_the_cusp():
    fam, records, states, _ = traced_cubic()
    k = next(r["index"] for r in records if r["cusp_candidate"]) - 1     # the tangent reverses between k and k+1
    res = cl.locate(fam, states[k], records[k]["theta"], states[k + 1], records[k + 1]["theta"],
                    records[k]["parameter_tangent"], test="reversal", kind="fold", recl=records[k],
                    recr=records[k + 1], rounds=3, per_round=5)
    assert res["located"], res["checks"]
    assert max(abs(v) for v in res["final"]["theta"]) < 1e-4


def fold_flip(z, theta):
    """(x, y) -> (x + t1 + t2 / 2 + u^2, 1 - (1 + t2) w + w^3) with u = x - 1, w = y - 1: fold curve t1 = -t2 / 2 at
    x = y = 1 (state norm away from zero, as in the soft-token maps), along which the second multiplier -(1 + t2)
    crosses -1 at t2 = 0, a fold-flip point with orthogonal eigenvectors."""
    u, w = z[0] - 1, z[1] - 1
    return torch.stack([z[0] + theta[0] + 0.5 * theta[1] + u ** 2, 1 - (1 + theta[1]) * w + w ** 3])


def test_crossing_locates_and_types_a_fold_flip_point():
    fam = bc.Family(fold_flip, (2,), torch.float64)
    records, reason, states = bc.trace_curve(fam, np.array([1.01, 1.0]), [0.15, -0.3], "fold", fix=1, direction=-1,
                                             steps=14, ds=0.05, ds_max=0.05)
    rad = np.array([r["other_spectral_radius"] for r in records])
    k = int(np.flatnonzero(np.sign(rad[1:] - 1) != np.sign(rad[:-1] - 1))[0])
    res = cl.locate(fam, states[k], records[k]["theta"], states[k + 1], records[k + 1]["theta"],
                    records[k]["parameter_tangent"], test="crossing", kind="fold", recl=records[k],
                    recr=records[k + 1], rounds=3, per_round=5)
    fin = res["final"]
    assert res["located"], res["checks"]
    assert abs(fin["theta"][1]) < 1e-4 and abs(fin["theta"][0] + 0.5 * fin["theta"][1]) < 1e-9
    assert fin["type"] == "fold-flip (+1 and -1)" and abs(fin["eigenvector_angle_degrees"] - 90) < 1e-6


def test_refuses_a_bracket_without_a_sign_change():
    fam, records, states, k = traced_cubic()
    with pytest.raises(ValueError):
        cl.locate(fam, states[k - 3], records[k - 3]["theta"], states[k - 2], records[k - 2]["theta"],
                  records[k - 3]["parameter_tangent"], test="cusp", kind="fold", param=0, param_scale=1.0,
                  nf_steps=(1e-2, 3e-3, 1e-3))


def test_results_serialize_to_json():
    import json

    fam, records, states, k = traced_cubic()
    res = cl.locate(fam, states[k], records[k]["theta"], states[k + 1], records[k + 1]["theta"],
                    records[k]["parameter_tangent"], test="cusp", kind="fold", rounds=1, per_round=5, param=0,
                    param_scale=1.0, nf_steps=(1e-2, 3e-3, 1e-3))
    res.pop("state")
    json.loads(json.dumps(res, default=cl._plain))
    assert all(type(v) is bool for v in res["checks"].values()) and type(res["located"]) is bool
