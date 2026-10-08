import numpy as np
import pytest
import torch

from neural_shrimp import cifar


def tiny(seed=3, lam=1e-2):
    rng = np.random.default_rng(seed)
    model = cifar.MLP(d=5, width=3, k=4)
    x = rng.normal(size=(7, 5))
    y = cifar.one_hot(rng.integers(0, 4, size=7))[:, :4]
    problem = cifar.Problem(model, x, y, lam)
    theta = torch.as_tensor(rng.normal(size=model.n_parameters) * 0.7)
    return model, problem, theta, rng


def test_gradient_and_hvp_match_finite_differences():
    model, problem, theta, rng = tiny()
    v = torch.as_tensor(rng.normal(size=model.n_parameters))
    h = 1e-6
    fd_grad = (problem.loss(theta + h * v) - problem.loss(theta - h * v)) / (2 * h)
    assert abs(float(fd_grad) - float(problem.grad(theta) @ v)) < 1e-7
    fd_hvp = (problem.grad(theta + h * v) - problem.grad(theta - h * v)) / (2 * h)
    np.testing.assert_allclose(problem.hvp(theta, v).numpy(), fd_hvp.numpy(), atol=1e-6)


def test_two_step_jvp_matches_finite_differences():
    model, problem, theta, rng = tiny()
    eta = 0.3
    v = torch.as_tensor(rng.normal(size=model.n_parameters))
    h = 1e-6

    def f2(t):
        return problem.update(problem.update(t, eta), eta)

    fd = (f2(theta + h * v) - f2(theta - h * v)) / (2 * h)
    np.testing.assert_allclose(problem.jvp_two_step(theta, v, eta).numpy(), fd.numpy(), atol=1e-6)


def test_arnoldi_multipliers_match_dense_eigenvalues():
    model, problem, theta, rng = tiny(seed=5, lam=1e-3)
    eta = 0.9

    def f2(t):
        return problem.update(problem.update(t, eta), eta)

    dense = torch.func.jacfwd(f2)(theta).numpy()
    values = np.linalg.eigvals(dense)
    out = cifar.two_step_multipliers(problem, theta.numpy(), eta, k=4, tol=1e-12)
    lm = np.array([complex(*v) for v in out["modes"]["LM"]])
    np.testing.assert_allclose(np.sort(np.abs(lm))[::-1], np.sort(np.abs(values))[::-1][:4], rtol=1e-8)
    li = np.array([complex(*v) for v in out["modes"]["LI"]])
    assert np.isclose(li.imag.max(), values.imag.max(), rtol=1e-8, atol=1e-10)


def test_newton_period2_finds_a_fixed_point_solution():
    model, problem, theta, rng = tiny(seed=7, lam=5e-2)
    eta = 0.05
    for _ in range(4000):
        theta = problem.update(theta, eta)
    solved, history = cifar.newton_period2(problem, theta + 1e-4, eta, tol=1e-12)
    assert history[-1]["relative_residual"] < 1e-11
    assert float((problem.update(solved, eta) - solved).norm()) < 1e-9


def test_balanced_subsets_are_disjoint_balanced_and_reproducible():
    labels = np.repeat(np.arange(10), 300)
    a, b = cifar.balanced_subsets(labels, 28201, size=512)
    assert len(a) == len(b) == 512 and not set(a) & set(b)
    counts = np.bincount(labels[a], minlength=10)
    assert counts.min() == 51 and counts.max() == 52 and counts.sum() == 512
    np.testing.assert_array_equal(np.bincount(labels[b], minlength=10), counts)
    a2, b2 = cifar.balanced_subsets(labels, 28201, size=512)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    with pytest.raises(ValueError):
        cifar.balanced_subsets(labels, 29001)


def test_rotation_from_samples_recovers_a_planted_rotation():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(50, 2)))[0]
    k = np.arange(4000)
    angle = 2 * np.pi * 0.1234 * k
    curve = np.stack([np.cos(angle) + 0.1 * np.cos(2 * angle), np.sin(angle)], 1) @ basis.T
    rho, share = cifar.rotation_from_samples(curve + 3.0)
    assert abs(rho - 0.1234) < 1e-4 and share > 0.99


def load_driver():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'tools/explore_cifar.py'
    spec = importlib.util.spec_from_file_location('explore_cifar_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LinearProblem:
    def update(self, theta, eta):
        return 0.5 * theta

    def jvp_two_step(self, theta, vector, eta, theta_next=None):
        return 0.25 * vector


def test_newton_final_diagnostic_matches_returned_state_after_last_update():
    initial = torch.ones(4, dtype=torch.float64)
    solved, history = cifar.newton_period2(LinearProblem(), initial, 0.1, max_newton=1)
    actual = float((0.25 * solved - solved).norm()) / max(1., float(solved.norm()))
    assert history[-1]['relative_residual'] == pytest.approx(actual, abs=1e-30)
    assert history[-1]['converged']
    assert solved.norm() < 1e-12
    zero, history = cifar.newton_period2(LinearProblem(), torch.zeros(4), .1)
    assert history[-1]['converged'] and history[-1]['relative_residual'] == 0


def test_failed_gmres_does_not_apply_an_unqualified_step(monkeypatch):
    import scipy.sparse.linalg
    monkeypatch.setattr(scipy.sparse.linalg, 'gmres', lambda *a, **k: (np.ones(4), 1))
    initial = torch.ones(4, dtype=torch.float64)
    solved, history = cifar.newton_period2(LinearProblem(), initial, .1)
    assert torch.equal(solved, initial)
    assert history[-1]['status'] == 'linear_solver_failed'
    assert not history[-1]['converged']


def test_arnoldi_seed_is_reproducible_and_bad_ritz_pairs_fail(monkeypatch):
    import scipy.sparse.linalg
    starts = []
    def fake(operator, **kwargs):
        starts.append(kwargs['v0'].copy())
        return np.array([2., 3.]), np.eye(6)[:, :2]
    monkeypatch.setattr(scipy.sparse.linalg, 'eigs', fake)
    operator = scipy.sparse.linalg.aslinearoperator(np.diag([2., 3., 4., 5., 6., 7.]))
    first = cifar.arnoldi_spectrum(operator, k=2)
    second = cifar.arnoldi_spectrum(operator, k=2)
    assert first['qualified'] and second['qualified']
    np.testing.assert_array_equal(starts[0], starts[2])
    assert not np.array_equal(starts[0], starts[1])
    bad = scipy.sparse.linalg.aslinearoperator(np.eye(6))
    result = cifar.arnoldi_spectrum(bad, k=2)
    assert not result['qualified'] and result['dominant_complex'] is None


def test_partial_arnoldi_convergence_is_not_qualified(monkeypatch):
    import scipy.sparse.linalg
    def failure(*args, **kwargs):
        raise scipy.sparse.linalg.ArpackNoConvergence('bounded failure', np.array([1.]), np.eye(6)[:, :1])
    monkeypatch.setattr(scipy.sparse.linalg, 'eigs', failure)
    result = cifar.arnoldi_spectrum(scipy.sparse.linalg.aslinearoperator(np.eye(6)), k=2)
    assert not result['qualified']
    assert not result['diagnostics']['LM']['arpack_converged']


def test_arnoldi_matvec_budget_and_newton_deadline_are_enforced():
    from scipy.sparse.linalg import aslinearoperator
    import time
    with pytest.raises(RuntimeError, match='matvec budget'):
        cifar.arnoldi_spectrum(aslinearoperator(np.diag(np.arange(1., 7.))), k=2, max_matvec=0)
    with pytest.raises(TimeoutError, match='wall-time'):
        cifar.newton_period2(LinearProblem(), torch.ones(4), .1, deadline=time.monotonic() - 1)


@pytest.mark.parametrize('residual,separation,converged', [(np.nan, 1., False), (1e-3, 1., False), (0., 0., True)])
def test_driver_rejects_failed_or_collapsed_orbits_before_arnoldi(monkeypatch, residual, separation, converged):
    driver = load_driver()
    monkeypatch.setattr(driver, 'certify', lambda *a, **k: (torch.ones(6),
        [{'relative_residual': residual, 'converged': converged}], separation))
    def forbidden(*a, **k):
        raise AssertionError('Arnoldi must not run')
    monkeypatch.setattr(cifar, 'two_step_multipliers', forbidden)
    _, row = driver.evaluate_orbit(None, torch.ones(6), .1, 2)
    assert row['stop_reason'] and 'top_kind' not in row


def test_failed_bisection_keeps_unresolved_bracket_and_incremental_evidence(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import json
    driver = load_driver()
    monkeypatch.setattr(driver, 'subset_a_problem', lambda *a: (SimpleNamespace(data_identity={'mock': True}), None))
    calls = []
    def evaluate(problem, theta, eta, k, **kwargs):
        i = len(calls)
        calls.append(i)
        if i == 2:
            return theta, {'stop_reason': 'corrector_not_converged', 'relative_residual': .2}
        return theta, {'stop_reason': None, 'estimated_stability': 'stable' if i == 0 else 'unstable',
                       'top': [.9 if i == 0 else 1.1, .01], 'top_modulus': .9 if i == 0 else 1.1,
                       'top_kind': 'complex', 'theta_over_2pi': .1}
    monkeypatch.setattr(driver, 'evaluate_orbit', evaluate)
    state = tmp_path / 'state.npy'
    np.save(state, np.ones(6))
    output = tmp_path / 'attempt'
    args = SimpleNamespace(threads=1, lam=.01, state=str(state), eta=.1, output_ratio=1.,
                           k=2, step=.1, max_steps=2, bisect_tol=1e-6, output=str(output))
    driver.cmd_orbit(args)
    saved = json.loads((output / 'orbit.json').read_text())
    assert saved['status'] == 'unresolved' and saved['crossing'] is None
    assert len(saved['rows']) == 2 and len(saved['bisection_rows']) == 1
    assert saved['bracket']['log10_eta_upper'] > saved['bracket']['log10_eta_lower']
    assert len(list(output.glob('state-*.npy'))) == 2
    assert any(json.loads(line)['kind'] == 'bisection' for line in (output / 'events.jsonl').read_text().splitlines())
    with pytest.raises(FileExistsError):
        driver.cmd_orbit(args)


def test_crossing_angle_refers_to_top_ritz_value_not_an_unrelated_complex_mode(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import json
    driver = load_driver()
    monkeypatch.setattr(driver, 'subset_a_problem', lambda *a: (SimpleNamespace(data_identity={'mock': True}), None))
    count = []
    def evaluate(problem, theta, eta, k, **kwargs):
        stable = not count
        count.append(1)
        return theta, {'stop_reason': None, 'estimated_stability': 'stable' if stable else 'unstable',
                       'top': [.9 if stable else 1.1, 0.], 'top_modulus': .9 if stable else 1.1,
                       'top_kind': 'plus', 'theta_over_2pi': .123}
    monkeypatch.setattr(driver, 'evaluate_orbit', evaluate)
    state = tmp_path / 'state.npy'
    np.save(state, np.ones(6))
    output = tmp_path / 'attempt'
    args = SimpleNamespace(threads=1, lam=.01, state=str(state), eta=.1, output_ratio=1.,
                           k=2, step=.1, max_steps=2, bisect_tol=1., output=str(output))
    driver.cmd_orbit(args)
    result = json.loads((output / 'orbit.json').read_text())
    assert result['crossing']['type'] == 'fold_candidate'
    assert result['crossing']['theta_over_2pi'] == 0.


@pytest.mark.parametrize('field,value', [('eta', 0.), ('eta', np.nan), ('output_ratio', -1.),
    ('step', -.1), ('step', np.nan), ('step', np.inf), ('bisect_tol', 0.), ('max_seconds', -1.),
    ('max_steps', 0), ('newton_steps', -1), ('gmres_restart', 0), ('arnoldi_maxiter', 0),
    ('arnoldi_max_matvec', 0), ('stability_margin', np.nan), ('stability_margin', -.1)])
def test_invalid_orbit_controls_rejected_before_output_creation(tmp_path, field, value):
    from types import SimpleNamespace
    driver = load_driver()
    args = SimpleNamespace(eta=.1, output_ratio=1., step=.1, bisect_tol=1e-6,
                           max_steps=2, lam=.01, output=str(tmp_path/'unused'))
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        driver.cmd_orbit(args)
    assert not (tmp_path/'unused').exists()


def test_corrected_state_survives_spectral_timeout(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import json
    driver = load_driver()
    x = np.arange(12., dtype=np.float64).reshape(4, 3)
    identity = {'x': driver.array_identity(x), 'y': driver.array_identity(x[:, :1]),
                'subset_a_record_sha256': 'mock-only'}
    monkeypatch.setattr(driver, 'subset_a_problem', lambda *a: (SimpleNamespace(data_identity=identity), None))
    corrected = torch.arange(6., dtype=torch.float64)
    monkeypatch.setattr(driver, 'certify', lambda *a, **k: (corrected,
        [{'relative_residual': 0., 'converged': True}], .1))
    def timeout(*a, **k):
        raise TimeoutError('controlled spectral timeout')
    monkeypatch.setattr(cifar, 'two_step_multipliers', timeout)
    state = tmp_path/'initial.npy'; np.save(state, np.ones(6))
    output = tmp_path/'attempt'
    args = SimpleNamespace(threads=1, lam=.01, eta=.1, output_ratio=1., step=.1,
        max_steps=2, bisect_tol=1e-6, k=2, state=str(state), output=str(output))
    with pytest.raises(TimeoutError):
        driver.cmd_orbit(args)
    result = json.loads((output/'orbit.json').read_text())
    assert result['data_identity'] == identity
    assert result['status'] == 'failed_or_interrupted' and result['crossing'] is None
    saved = result['last_corrector']
    assert saved['spectrum_status'] == 'pending' and 'estimated_stability' not in saved
    assert cifar.sha256_file(output/saved['state_file']) == saved['state_sha256']
    np.testing.assert_array_equal(np.load(output/saved['state_file']), corrected.numpy())


def test_circle_is_excluded_from_bounded_pilot():
    with pytest.raises(NotImplementedError, match='disabled'):
        load_driver().cmd_circle(None)
