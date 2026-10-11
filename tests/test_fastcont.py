import json
import math
from pathlib import Path

import numpy as np
import pytest

from neural_shrimp.fastcont import FoldCurve, OrbitCurve, TwoRateMap, divisor_separations, newton
from neural_shrimp.trainability import make_s2

ROOT = Path(__file__).resolve().parents[1]
NODE = (ROOT.parent / "neural-shrimp-data/explore/s2-period18-boundary-curves/"
        "attempt-006-ray32-fold/run/node-+1-010.json")


@pytest.fixture(scope="module")
def fmap():
    net, x, y, theta0 = make_s2(28011)
    return TwoRateMap(net, x, y), theta0


def test_step_derivatives_match_central_differences(fmap):
    f, theta0 = fmap
    p = np.array([-0.08, 1.82])
    _, jac, dp = f.step_jac(theta0, p)
    h = 1e-6
    for j in range(f.n):
        e = np.zeros(f.n)
        e[j] = h
        col = (f.step(theta0 + e, p) - f.step(theta0 - e, p)) / (2 * h)
        np.testing.assert_allclose(jac[:, j], col, rtol=1e-6, atol=1e-8)
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        col = (f.step(theta0, p + e) - f.step(theta0, p - e)) / (2 * h)
        np.testing.assert_allclose(dp[:, k], col, rtol=1e-6, atol=1e-8)


def test_period_map_and_jvp_agree(fmap):
    f, theta0 = fmap
    p = np.array([-0.08, 1.80])
    theta = f.iterate(theta0, p, 50)
    t, m, dp, points = f.period_map(theta, p, 3)
    np.testing.assert_allclose(t, points[-1])
    q = np.random.default_rng(0).normal(size=f.n)
    t2, mq = f.monodromy_vector(theta, p, 3, q)
    np.testing.assert_allclose(t2, t, atol=1e-14)
    np.testing.assert_allclose(mq, m @ q, rtol=1e-10, atol=1e-12)
    h = 1e-6
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        col = (f.period_map(theta, p + e, 3)[0] - f.period_map(theta, p - e, 3)[0]) / (2 * h)
        np.testing.assert_allclose(dp[:, k], col, rtol=1e-5, atol=1e-8)


def test_fold_jacobian_matches_central_differences(fmap):
    f, theta0 = fmap
    p = np.array([-0.08, 1.80])
    theta = f.iterate(theta0, p, 50)
    q = np.random.default_rng(1).normal(size=f.n)
    q /= np.linalg.norm(q)
    system = FoldCurve(f, 2, sigma=1.0)
    z = np.concatenate([theta, q, p])
    res, jac, _, _ = system.residual_jacobian(z)
    h = 1e-6
    for j in range(len(z)):
        e = np.zeros(len(z))
        e[j] = h
        col = (system.residual_jacobian(z + e)[0] - system.residual_jacobian(z - e)[0]) / (2 * h)
        np.testing.assert_allclose(jac[:, j], col, rtol=1e-4, atol=1e-6)


def test_orbit_curve_recovers_a_fixed_period_orbit(fmap):
    # positive control: an attracting period-2 orbit at a rate pair below the
    # Neimark-Sacker curve is a regular point of the one-parameter orbit curve
    f, theta0 = fmap
    p = [-0.08, 1.74]
    theta = f.iterate(theta0, p, 20000)
    curve = OrbitCurve(f, 2, fixed=p[0])
    z, history = newton(curve, np.concatenate([theta, [p[1]]]),
                        extra=lambda w: (np.eye(f.n + 1)[-1], w[-1] - p[1]))
    assert z is not None and history[-1] < 1e-11
    points = f.orbit(z[:f.n], curve.params(z[-1]), 2)
    assert divisor_separations(points, 2)[1] > 1e-3


@pytest.mark.skipif(not NODE.exists(), reason="saved carried fold node not available")
def test_saved_window_fold_node_is_a_fold_root(fmap):
    f, _ = fmap
    node = json.loads(NODE.read_text())
    point = [float(v) for v in node["point"]]
    theta, q = np.array(point[:13]), np.array(point[15:28])
    p = [math.log10(point[14]), math.log10(point[13])]
    system = FoldCurve(f, 18, sigma=1.0)
    res, _, m, points = system.residual_jacobian(np.concatenate([theta, q / np.linalg.norm(q), p]))
    assert np.linalg.norm(res[:13]) < 1e-8
    assert np.linalg.norm(res[13:26]) < 1e-7
    assert min(divisor_separations(points, 18).values()) > 1e-2
