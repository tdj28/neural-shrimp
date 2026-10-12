import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402

MODEL_DIR = ela.CACHE / "models--Qwen--Qwen2.5-0.5B-Instruct"
pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files not present")


@pytest.fixture(scope="module")
def loop():
    import explore_llm_soft_loop as esl
    return esl.SoftLoop("qwen0.5b", 12, 4, "open", "Hello!", 2, dtype="float64")


def test_cached_step_matches_full_recompute(loop):
    torch = loop.torch
    rng = np.random.default_rng(0)
    d = loop.E.shape[1]
    steer = rng.normal(size=(2, d)) * 0.5
    X = loop.prepare(steer) + torch.as_tensor(rng.normal(size=(2, loop.K, d)) * 1e-2)
    z_cached = loop.logits(X)
    assert loop.cache.get_seq_length() == loop.L0          # cropped back to the prefix
    prefix = loop.model.get_input_embeddings()(loop.prefix_ids)[0].to(torch.float64)
    full = torch.cat([prefix.repeat(2, 1, 1), X], dim=1)
    loop.sm.steer = torch.as_tensor(steer)[:, None, :]
    try:
        with torch.no_grad():
            z_full = loop.model(inputs_embeds=full).logits[:, -1, :].to(torch.float64)
    finally:
        loop.sm.steer = None
    assert torch.allclose(z_cached, z_full, atol=1e-8, rtol=0)
    z_again = loop.logits(X)                                 # the map is autonomous: same input, same output
    assert torch.equal(z_cached, z_again)


def test_low_temperature_feeds_back_the_greedy_token(loop):
    torch = loop.torch
    X = loop.prepare(np.zeros((1, loop.E.shape[1])))
    z = loop.logits(X)
    X1, p = loop.step(X, np.array([1e-4]))
    greedy = int(torch.argmax(z[0]))
    assert torch.allclose(X1[0, -1], loop.E[greedy], atol=1e-10)


def test_orbit_jacobian_ad_matches_central_differences(loop):
    """Forward-mode JVPs of one soft step (window shift included) agree with central differences, and leave
    the stored prefix cache unchanged."""
    import explore_llm_soft_loop as esl

    torch = loop.torch
    rng = np.random.default_rng(1)
    d = loop.E.shape[1]
    orb = esl.SoftOrbit(loop, rng.normal(size=d) * 0.5, 0.7, chunk=512)
    X = orb.start + torch.as_tensor(rng.normal(size=tuple(orb.start.shape)) * 1e-2)
    V = torch.as_tensor(rng.normal(size=(3,) + orb.shape))
    V = V / V.flatten(1).norm(dim=1)[:, None, None]
    ad = orb.jvp_rows(X, V).detach()
    errors = []
    for h_rel in (1e-3, 1e-4):                                # central differences converge like h^2 in float64
        h = h_rel * float(X.norm())
        Y = orb.f(torch.cat([X + h * V, X - h * V]))
        fd = (Y[:3] - Y[3:]) / (2 * h)
        errors.append(float((ad[:, -1] - fd[:, -1]).norm() / fd[:, -1].norm()))
    assert errors[1] < 1e-6 and errors[1] < errors[0] / 50, errors
    assert orb.caches[3].get_seq_length() == loop.L0
    J = orb.jacobian(X, method="ad")
    assert J.shape == (orb.n, orb.n)
    assert np.allclose(J @ V[0].reshape(-1).numpy(), ad[0].reshape(-1).numpy(), rtol=0, atol=1e-9 * float(ad[0].norm()))


def test_float64_model_keeps_float64_norms(loop):
    """A float64 model must not round to float32 inside its RMSNorm layers (the Hugging Face default upcast)."""
    torch = loop.torch
    norm = loop.model.model.norm
    x = torch.randn(1, 1, norm.weight.shape[0], dtype=torch.float64)
    y1, y2 = norm(x), norm(x * (1 + 1e-12))
    assert y1.dtype == torch.float64
    assert 0 < float((y2 - y1).abs().max()) < 1e-9


def test_soft_phi_matches_steps_and_temperature_derivative(loop):
    """The differentiable p-fold map used for continuation in temperature equals p soft steps, and its
    forward-mode derivative in log10 T matches central differences."""
    import explore_llm_soft_loop as esl

    torch = loop.torch
    rng = np.random.default_rng(2)
    d = loop.E.shape[1]
    orb = esl.SoftOrbit(loop, rng.normal(size=d) * 0.5, 10 ** -0.2, chunk=512)
    X = orb.start + torch.as_tensor(rng.normal(size=tuple(orb.start.shape)) * 1e-2)
    phi = orb.phi(2)
    lt = torch.tensor(-0.2, dtype=torch.float64)
    with torch.no_grad():
        y = phi(X, lt)
    assert torch.allclose(y, orb.f(orb.f(X)), atol=1e-12, rtol=0)
    _, dy = torch.func.jvp(phi, (X, lt), (torch.zeros_like(X), torch.tensor(1.0, dtype=torch.float64)))
    h = 1e-5
    with torch.no_grad():
        fd = (phi(X, lt + h) - phi(X, lt - h)) / (2 * h)
    assert float((dy - fd).norm() / fd.norm()) < 1e-6
    assert orb.cache1.get_seq_length() == loop.L0


def test_subspace_multipliers_match_dense_eigenvalues(loop):
    """Block subspace iteration with batched JVPs recovers the leading eigenvalues of the dense Jacobian."""
    import explore_llm_soft_loop as esl

    torch = loop.torch
    rng = np.random.default_rng(3)
    d = loop.E.shape[1]
    orb = esl.SoftOrbit(loop, rng.normal(size=d) * 0.5, 10 ** 0.1, chunk=512)
    X = orb.start + torch.as_tensor(rng.normal(size=tuple(orb.start.shape)) * 1e-2)
    vals, ok = orb.subspace_multipliers(X, 1, k=6, iters=60)
    dense = np.linalg.eigvals(orb.jacobian(X))
    dense = dense[np.argsort(-np.abs(dense))]
    assert ok
    assert np.allclose(np.abs(vals[:3]), np.abs(dense[:3]), rtol=1e-6, atol=0)
