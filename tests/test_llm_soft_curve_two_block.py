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
    """Two-block family as in run TB: block 4 steered along v, block 16 along w, at a fixed temperature."""
    import explore_llm_soft_loop as esl
    loop = esl.SoftLoop("qwen0.5b", 4, 8, "open", ela.OPENER, 1, dtype="float64")
    loop.sm.add_steer_layer(16)
    rng = np.random.default_rng(7)
    v, w = rng.normal(size=(2, loop.E.shape[1]))
    v *= 190.0 / np.linalg.norm(v)                                     # the size of the concept vectors
    w *= 190.0 / np.linalg.norm(w)
    log_t = -0.1
    fam = esl.soft_family(loop, v, period=2, alpha_scale=10.0, chunk=512, second=(16, w), log_temperature=log_t)
    theta = np.array([-0.12 * 10.0, 0.10 * 10.0])
    loop.extra = (16, np.array([0.10 * w]))
    orb = esl.SoftOrbit(loop, -0.12 * v, 10.0 ** log_t)
    x = orb.iterate(orb.start, 5)                                      # a state on the steered trajectory
    return esl, loop, v, w, fam, theta, fam.to_np(x)


def test_two_block_family_needs_the_hook_and_temperature(setup):
    esl, loop, v, w, fam, theta, x = setup
    with pytest.raises(ValueError):
        esl.soft_family(loop, v, period=2, second=(16, w))             # no fixed temperature
    with pytest.raises(ValueError):
        esl.soft_family(loop, v, period=2, second=(20, w), log_temperature=-0.1)   # layer 20 has no hook


def test_two_block_map_matches_soft_orbit_steps(setup):
    """The differentiable map (fresh prefix with both blocks) and SoftOrbit's cached steps agree, and the second
    block changes the map."""
    esl, loop, v, w, fam, theta, x = setup
    r = fam.residual(x, theta)
    with loop.torch.no_grad():
        y = fam.to_np(fam.phi(fam.to_x(x), fam.to_theta(theta)))
    assert np.allclose(y - x, r, atol=1e-12, rtol=0)
    r0 = fam.residual(x, theta * np.array([1.0, 0.0]))                  # alpha2 = 0
    assert np.linalg.norm(r - r0) > 1e-6 * np.linalg.norm(x)
    assert loop.sm.steer is None and loop.sm.steer_extra == {}         # hooks are cleared after the map


def test_two_block_parameter_derivatives_include_both_prefixes(setup):
    """D_theta Phi by forward-mode AD agrees with central differences in alpha1 and alpha2 (both blocks also steer
    the prefix cache), converging like h^2."""
    esl, loop, v, w, fam, theta, x = setup
    P = fam.param_jacobian(x, theta)
    for i in range(2):
        errs = []
        for h in (1e-4, 1e-5):
            e = np.zeros(2)
            e[i] = h
            fd = (fam.residual(x, theta + e) - fam.residual(x, theta - e)) / (2 * h)
            errs.append(np.linalg.norm(fd - P[:, i]) / np.linalg.norm(P[:, i]))
        assert errs[1] < 1e-6 and errs[1] < errs[0] / 30, errs


def test_two_block_dense_monodromy_matches_a_jvp_of_the_map(setup):
    esl, loop, v, w, fam, theta, x = setup
    torch = loop.torch
    M = fam.monodromy(x, theta)
    d = np.random.default_rng(5).normal(size=fam.n)
    with fam.context():
        _, jv = torch.func.jvp(lambda u: fam.phi(u, fam.to_theta(theta)), (fam.to_x(x),), (fam.to_x(d),))
    assert np.allclose(M @ d, fam.to_np(jv), atol=1e-10 * np.linalg.norm(M @ d), rtol=0)
