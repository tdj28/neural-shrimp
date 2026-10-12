import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")


@pytest.fixture(scope="module")
def setup():
    import explore_llm_soft_loop as esl
    loop = esl.SoftLoop("qwen0.5b", 12, 8, "open", "Hello!", 1, dtype="float64")
    v = np.random.default_rng(3).normal(size=loop.E.shape[1])
    v *= 190.0 / np.linalg.norm(v)                                     # the size of the concept vectors
    fam = esl.soft_family(loop, v, period=2, alpha_scale=10.0, chunk=512)
    theta = np.array([-0.1, 0.07 * 10.0])
    orb = esl.SoftOrbit(loop, 0.07 * v, 10.0 ** -0.1)
    x = orb.iterate(orb.start, 5)                                      # a state on the steered trajectory
    return esl, loop, v, fam, theta, orb, fam.to_np(x)


def test_family_map_matches_soft_orbit_steps(setup):
    esl, loop, v, fam, theta, orb, x = setup
    r = fam.residual(x, theta)                                         # via SoftOrbit (cropped prefix cache)
    with loop.torch.no_grad():
        y = fam.to_np(fam.phi(fam.to_x(x), fam.to_theta(theta)))       # via the differentiable map (fresh prefix)
    assert np.allclose(y - x, r, atol=1e-12, rtol=0)


def test_parameter_derivatives_include_the_steered_prefix(setup):
    """D_theta Phi by forward-mode AD agrees with central differences in log10 T and in alpha (whose steering
    also changes the prefix cache), converging like h^2."""
    esl, loop, v, fam, theta, orb, x = setup
    P = fam.param_jacobian(x, theta)
    for i in range(2):
        errs = []
        for h in (1e-4, 1e-5):
            e = np.zeros(2)
            e[i] = h
            fd = (fam.residual(x, theta + e) - fam.residual(x, theta - e)) / (2 * h)
            errs.append(np.linalg.norm(fd - P[:, i]) / np.linalg.norm(P[:, i]))
        assert errs[1] < 1e-6 and errs[1] < errs[0] / 30, errs


def test_gradient_of_the_contracted_jacobian_matches_differences(setup):
    """grad_s (reverse mode through a forward-mode JVP) agrees with central differences of
    s = w^T D_x Phi v along a random direction in (x, theta)."""
    esl, loop, v, fam, theta, orb, x = setup
    torch = loop.torch
    rng = np.random.default_rng(4)
    vv, ww = rng.normal(size=fam.n), rng.normal(size=fam.n)
    dx, dth = rng.normal(size=fam.n) * np.linalg.norm(x) / np.sqrt(fam.n), np.array([0.3, -0.5])
    gx, gt = fam.grad_s(x, theta, vv, ww)

    def s(xx, th):
        with fam.context():
            _, jv = torch.func.jvp(lambda u: fam.phi(u, fam.to_theta(th)), (fam.to_x(xx),), (fam.to_x(vv),))
        return float((fam.to_x(ww) * jv).sum())

    exact = gx @ dx + gt @ dth
    errs = []
    for h in (1e-4, 1e-5):
        fd = (s(x + h * dx, theta + h * dth) - s(x - h * dx, theta - h * dth)) / (2 * h)
        errs.append(abs(fd - exact) / abs(exact))
    assert errs[1] < 1e-5 and errs[1] < errs[0] / 30, errs


def test_dense_monodromy_matches_a_jvp_of_the_map(setup):
    esl, loop, v, fam, theta, orb, x = setup
    torch = loop.torch
    M = fam.monodromy(x, theta)
    d = np.random.default_rng(5).normal(size=fam.n)
    with fam.context():
        _, jv = torch.func.jvp(lambda u: fam.phi(u, fam.to_theta(theta)), (fam.to_x(x),), (fam.to_x(d),))
    assert np.allclose(M @ d, fam.to_np(jv), atol=1e-10 * np.linalg.norm(M @ d), rtol=0)


def synthetic_tline(points, crossings):
    return {"points": [{"index": i, "lambda": lam, "spectral_radius": rho, **extra}
                       for i, (lam, rho, extra) in enumerate(points)], "crossings": crossings}


def test_curve_start_picks_the_boundary_and_refuses_mismatches():
    import explore_llm_soft_loop as esl

    states = np.arange(5 * 3, dtype=float).reshape(5, 1, 3)
    fold = synthetic_tline([(-0.12, 0.5, {}), (-0.11, 0.8, {}), (-0.105, 0.97, {}),
                            (-0.1049, 1.02, {"turned_since_previous": True, "turn_lambda_estimate": -0.1048}),
                            (-0.106, 1.2, {})],
                           [{"between_log10_T": [-0.105, -0.1049], "type": "real +1 (fold)"}])
    kind, i, x0, lam = esl.curve_start(fold, states, "auto", -1)
    assert kind == "fold" and i == 3 and lam == -0.1049 and np.array_equal(x0, states[3].ravel())
    flip = synthetic_tline([(-0.12, 0.5, {}), (-0.13, 0.9, {}), (-0.135, 0.99, {}), (-0.14, 1.05, {})],
                           [{"between_log10_T": [-0.135, -0.14], "type": "real -1 (flip)"}])
    assert esl.curve_start(flip, states, "auto", -1)[:2] == ("flip", 2)
    with pytest.raises(SystemExit):                       # a turn whose crossing is not a real +1
        esl.curve_start(dict(fold, crossings=[{"between_log10_T": [-0.105, -0.1049],
                                               "type": "complex pair (Neimark-Sacker)"}]), states, "fold", -1)
    with pytest.raises(SystemExit):
        esl.curve_start(flip, states, "fold", -1)
    with pytest.raises(SystemExit):
        esl.curve_start(dict(flip, crossings=[]), states, "auto", -1)


def test_mixed_precision_family_takes_residuals_from_the_reference(setup):
    """With a float64 reference, residuals are float64-exact while the monodromy comes from the float32 map."""
    esl, loop, v, fam, theta, orb, x = setup
    lo = esl.SoftLoop("qwen0.5b", 12, 8, "open", "Hello!", 1, dtype="float32")
    mixed = esl.soft_family(lo, v, period=2, alpha_scale=10.0, chunk=512, reference=loop)
    assert np.allclose(mixed.residual(x, theta), fam.residual(x, theta), atol=1e-14, rtol=0)
    M32, M64 = mixed.monodromy(x, theta), fam.monodromy(x, theta)
    err = np.linalg.norm(M32 - M64) / np.linalg.norm(M64)
    assert 1e-9 < err < 1e-3, err                                      # float32 derivatives, not float64 ones


def test_curve_points_prefers_the_newer_of_result_and_progress(tmp_path):
    import json
    import os

    import explore_llm_soft_loop as esl

    run = tmp_path / "curve-x"
    run.mkdir()
    rows = [{"index": i, "log10_temperature": -0.1 + i * 1e-3, "alpha": 0.07, "critical_multiplier": [-1.0, 0.0],
             "cusp_candidate": False} for i in range(3)]
    (run / "progress.jsonl").write_text("".join(f"16:00:0{i} {json.dumps(r)}\n" for i, r in enumerate(rows)))
    kind, pts = esl.curve_points(run)
    assert kind == "flip" and len(pts) == 3                            # running: progress only
    (run / "curve.json").write_text(json.dumps({"kind": "fold", "points": rows[:1]}))
    os.utime(run / "progress.jsonl", (1, 1))
    assert esl.curve_points(run) == ("fold", rows[:1])                 # finished: the result
    os.utime(run / "curve.json", (0, 0))
    assert esl.curve_points(run)[0] == "flip"                          # stale result from an earlier run


def test_exact_jvp_family_takes_jvps_from_the_reference(setup):
    """exact_jvp: Jacobian-vector products (g, refined multipliers) in float64 from the reference, the dense
    monodromy still in float32."""
    esl, loop, v, fam, theta, orb, x = setup
    lo = esl.SoftLoop("qwen0.5b", 12, 8, "open", "Hello!", 1, dtype="float32")
    mixed = esl.soft_family(lo, v, period=2, alpha_scale=10.0, chunk=512, reference=loop, exact_jvp=True)
    d = np.random.default_rng(6).normal(size=fam.n)
    j64 = fam.jvp_x(x, theta, d)
    assert np.allclose(mixed.jvp_x(x, theta, d), j64, atol=1e-12 * np.linalg.norm(j64), rtol=0)
    plain = esl.soft_family(lo, v, period=2, alpha_scale=10.0, chunk=512, reference=loop)
    err = np.linalg.norm(plain.jvp_x(x, theta, d) - j64) / np.linalg.norm(j64)
    assert 1e-9 < err < 1e-2, err                                     # the float32 JVP is not the reference one
    with pytest.raises(ValueError):
        esl.soft_family(lo, v, period=2, exact_jvp=True)
