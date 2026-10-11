"""Synthetic controls only: no CIFAR/model/gradient/HVP or target inputs opened."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
from neural_shrimp import cifar_linear_diagnostic as linear
from neural_shrimp import cifar_restart_diagnostic as fixed
ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_explicit_zero_and_fixed_one_cycle(monkeypatch):
    seen = {}
    def gmres(op, rhs, **kwargs):
        seen.update(kwargs)
        assert np.array_equal(kwargs['x0'], np.zeros_like(rhs))
        for i in range(200):
            op @ rhs
            kwargs['callback'](.9)
        op @ rhs  # SciPy's end-of-cycle true check.
        return rhs, 1
    monkeypatch.setattr(linear, 'gmres', gmres)
    _, _, _, result = linear.solve_direction(lambda v: v, np.ones(3), restart=200,
        cycles=1, max_matvec=202, explicit_zero_start=True)
    assert result['matvec_calls'] == 202 and result['inner_iterations'] == 200
    assert seen['maxiter'] == 1 and seen['restart'] == 200
    assert seen['callback_type'] == 'pr_norm' and seen['rtol'] == 1e-10 and seen['atol'] == 0.
    assert 'M' not in seen
    assert result['gmres_info'] == 1 and result['forcing_test_pass']


def test_old_default_omits_x0(monkeypatch):
    def gmres(op, rhs, **kwargs):
        assert 'x0' not in kwargs
        assert kwargs['restart'] == 50 and kwargs['maxiter'] == 4
        return rhs, 0
    monkeypatch.setattr(linear, 'gmres', gmres)
    linear.solve_direction(lambda v: v, np.ones(2))
    with pytest.raises(ValueError, match='must be bool'):
        linear.solve_direction(lambda v: v, np.ones(2), explicit_zero_start=1)


def test_prefix_persist_before_reject_and_no_relaxation():
    rows = []
    prefix = fixed.Prefix([.9]*200, rows.append)
    for i in range(50):
        prefix(dict(kind='gmres_inner', iteration=i+1, estimated_relative_residual=.9))
    prefix.complete()
    prefix(dict(kind='gmres_inner', iteration=51, estimated_relative_residual=.5))
    mismatch = fixed.Prefix([.9]*200, rows.append)
    with pytest.raises(ValueError, match='prefix mismatch'):
        mismatch(dict(kind='gmres_inner', iteration=1, estimated_relative_residual=.8))
    assert mismatch.values == [.8] and rows[-1]['estimated_relative_residual'] == .8
    with pytest.raises(ValueError, match='fewer than 50'):
        mismatch.complete()


def test_operation_caps_and_failures():
    events = []
    budget = fixed.Budget(events.append)
    for kind, cap in fixed.CAPS.items():
        budget.attempted[kind] = cap - 1
        assert budget.call(kind, lambda: 2) == 2
        with pytest.raises(ValueError, match='operation cap'):
            budget.call(kind, lambda: pytest.fail('over budget'))
        assert budget.attempted[kind] == cap
    budget = fixed.Budget(events.append)
    with pytest.raises(RuntimeError):
        budget.call('hvp', lambda: (_ for _ in ()).throw(RuntimeError('synthetic')))
    assert budget.attempted['hvp'] == 1 and budget.returned['hvp'] == 0
    assert events[-1]['kind'] == 'operation_started'


def test_nested_hvp_gradient_excluded_from_direct_count():
    class Algebra:
        def grad(self, theta): return theta
        def hvp(self, theta, v): self.grad(theta); return v
        def jvp_two_step(self, theta, v): return self.hvp(theta, self.hvp(theta, v))
    class Counted(fixed.CountedProblemMixin, Algebra): pass
    problem = Counted(); problem.budget = fixed.Budget(lambda row: None)
    problem.grad(2)
    problem.hvp(2, 3)
    for _ in range(203): problem.jvp_two_step(2, 3)
    assert problem.budget.attempted == dict(action=0, j2vp=203, hvp=407, direct_gradient=1)
    assert problem._hvp_depth == 0


def test_deadline_sticky_term_and_monitored_rss(monkeypatch):
    budget = fixed.Budget(lambda row: None)
    budget.deadline = 0.
    with pytest.raises(TimeoutError): budget.check()
    budget.deadline = time.monotonic() + 10
    monkeypatch.setattr(fixed, 'STOP_REQUESTED', False)
    fixed.request_stop(None, None)
    with pytest.raises(InterruptedError): budget.check()
    monkeypatch.setattr(fixed, 'STOP_REQUESTED', False)
    class Usage: ru_maxrss = fixed.RSS_BYTES//1024 + 1
    monkeypatch.setattr(fixed.resource, 'getrusage', lambda which: Usage())
    with pytest.raises(MemoryError): budget.check()


def test_store_budget_final_reserve_and_bitwise_replay(tmp_path):
    store = fixed.Store(tmp_path/'store', limit=8192, reserve=4096)
    with pytest.raises(ValueError, match='storage cap'): store.write('large', b'x'*4097)
    driver = module('diagnose_cifar_restart')
    with pytest.raises(ValueError, match='byte-exact'):
        driver.replay(store, 'initial-map1', np.array([-0.]), {'initial-map1': np.array([0.])})
    assert np.signbit(np.load(store.root/'initial-map1.npy')[0])
    store.finish(dict(status='failed'))
    assert json.loads((store.root/'manifest.json').read_text())['files']['initial-map1.npy'] == fixed.entry(store.root/'initial-map1.npy')
    with pytest.raises(FileExistsError): fixed.Store(store.root)


def test_returned_direction_survives_failed_independent_check(monkeypatch, tmp_path):
    def gmres(op, rhs, **kw): return rhs, 7
    monkeypatch.setattr(linear, 'gmres', gmres)
    store = fixed.Store(tmp_path/'store')
    def fail(v): raise RuntimeError('independent action failed')
    with pytest.raises(RuntimeError, match='independent action'):
        linear.solve_direction(fail, np.ones(2), explicit_zero_start=True,
            save_direction=lambda p, info: store.array('direction', p))
    assert np.array_equal(np.load(store.root/'direction.npy'), -np.ones(2))


def test_source_guard_before_output(monkeypatch, tmp_path):
    driver = module('diagnose_cifar_restart')
    monkeypatch.setattr(driver, 'sources', lambda: (_ for _ in ()).throw(ValueError('dirty source')))
    with pytest.raises(ValueError, match='dirty source'):
        driver.run(tmp_path/'out', 'a'*40)
    assert not (tmp_path/'out').exists()


def test_baseline_manifest_before_any_model(monkeypatch):
    driver = module('diagnose_cifar_restart')
    monkeypatch.setattr(driver, 'verify_frozen', lambda *a: (_ for _ in ()).throw(ValueError('manifest')))
    monkeypatch.setattr(driver.prior, 'load_inputs', lambda: pytest.fail('must not load reference'))
    with pytest.raises(ValueError, match='manifest'): driver.inputs()


def fake_launch(monkeypatch, tmp_path, code, seconds=2):
    launcher = module('launch_cifar_restart')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    monkeypatch.setattr(launcher, 'committed_sources', lambda: {'synthetic': 'bound'})
    monkeypatch.setattr(launcher, 'command', lambda out, review: [sys.executable, '-c', code])
    monkeypatch.setattr(launcher.os, 'getpriority', lambda *a: 10)
    monkeypatch.setattr(launcher, 'SUPERVISOR_SECONDS', seconds)
    return launcher, lambda: launcher.launch(tmp_path/'out', tmp_path/'launcher', head, head)


def test_launcher_owned_failure_and_log_cap(monkeypatch, tmp_path):
    launcher, call = fake_launch(monkeypatch, tmp_path, "import os;os.write(1,b'x'*10000)")
    monkeypatch.setattr(launcher, 'LOG_LIMIT', 64)
    receipt = call()
    assert receipt['log_truncated'] and not receipt['successful']
    assert (tmp_path/'launcher'/'stdout.log').stat().st_size == 64
    assert not receipt['owned_live_pids_remaining']
    assert (tmp_path/'launcher').stat().st_mode & 0o222 == 0


@pytest.mark.parametrize('code', [
    'import os,time;pid=os.fork();time.sleep(5) if pid==0 else os._exit(0)',
    'import os,time;os.close(1);os.close(2);time.sleep(5)'])
def test_launcher_inherited_pipe_and_no_pipe_hang(monkeypatch, tmp_path, code):
    _, call = fake_launch(monkeypatch, tmp_path, code, seconds=.2)
    start = time.monotonic(); receipt = call()
    assert time.monotonic()-start < 3
    assert receipt['failure'] == 'supervisor_timeout_or_inherited_pipe'
    assert not receipt['successful']


def test_launcher_rss_is_monitored_and_fails_closed(monkeypatch, tmp_path):
    launcher, call = fake_launch(monkeypatch, tmp_path, 'import time;time.sleep(5)')
    actual = launcher.group_status
    def high(group):
        rows = actual(group)
        for row in rows: row['rss_bytes'] = launcher.RSS_BYTES + 1
        return rows
    monkeypatch.setattr(launcher, 'group_status', high)
    receipt = call()
    assert receipt['failure'] == 'monitored_group_rss_limit' and not receipt['successful']


def test_launcher_fixed_command():
    launcher = module('launch_cifar_restart')
    cmd = launcher.command(Path('/tmp/example'), 'a'*40)
    assert cmd[:7] == ['timeout', '--signal=TERM', '--kill-after=5s', '60s', 'nice', '-n', '10']
    assert '--review-commit' in cmd


def synthetic_driver(monkeypatch, tmp_path, *, bad_final=False, early=False):
    driver = module('diagnose_cifar_restart')
    monkeypatch.setattr(driver, 'sources', lambda: {'synthetic': 'bound'})
    monkeypatch.setattr(driver, 'environment', lambda: {'synthetic': True})
    class Vector(np.ndarray):
        def numpy(self): return np.asarray(self)
        def norm(self): return np.linalg.norm(self)
    def tensor(x): return np.asarray(x).view(Vector)
    class Algebra:
        def __init__(self, *a): pass
        def grad(self, theta): return theta*.1
        def update(self, theta, eta): return theta-eta*self.grad(theta)
        def hvp(self, theta, v): self.grad(theta); return v*.1
        def jvp_two_step(self, theta, v, eta, theta_next=None):
            one = v-eta*self.hvp(theta, v)
            two = one-eta*self.hvp(theta_next, one)
            if bad_final and self.budget.attempted['j2vp'] > 1: two[:] = np.nan
            return two
    class Counted(fixed.CountedProblemMixin, Algebra): pass
    monkeypatch.setattr(driver, 'CountedProblem', Counted)
    monkeypatch.setattr(driver.cifar, 'as_tensor', tensor)
    monkeypatch.setattr(driver.cifar, 'MLP', lambda **kw: None)
    theta = tensor([1., 2.]); eta = .1
    first = theta-eta*theta*.1; second=first-eta*first*.1
    direction = tensor([1., 0.]); one=direction-eta*direction*.1
    arrays = dict(theta=theta.numpy(), x=np.ones((2, 2)), y=np.ones((2, 1)),
        directions=np.asarray([direction]*3), gradient=(theta*.1).numpy(),
        hvp=np.asarray([direction*.1]*3), j2vp=np.asarray([one-eta*one*.1]*3))
    saved = {'initial-theta': theta.numpy(), 'initial-map1': first.numpy(),
             'initial-map2': second.numpy(), 'initial-residual': (second-theta).numpy()}
    baseline = dict(estimated_relative_residual_history=[.9]*200, true_forcing_ratio=.9)
    monkeypatch.setattr(driver, 'inputs', lambda: (arrays, dict(model={}, lam=.01, eta=eta), saved, baseline, {'synthetic': True}))
    def gmres(operator, rhs, **kw):
        for _ in range(1 if early else 50): kw['callback'](.9)
        return rhs, 1
    monkeypatch.setattr(linear, 'gmres', gmres)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return driver, lambda: driver.run(tmp_path/'synthetic', head)


def test_driver_all_four_trials_and_no_orbit_claim(monkeypatch, tmp_path):
    _, call = synthetic_driver(monkeypatch, tmp_path)
    result = call()
    assert result['comparison_qualified'] and result['status'] == 'diagnostic_complete'
    assert result['operation_attempted'] == dict(action=1, j2vp=2, hvp=5, direct_gradient=11)
    assert len(result['trials']) == 4 and not result['accepted_orbit'] and not result['spectrum_requested']
    assert all(not row['accepted_orbit'] and not row['applied_to_continuation'] for row in result['trials'])
    assert not result['linear']['forcing_test_pass']


def test_driver_nonfinite_final_action_retained(monkeypatch, tmp_path):
    _, call = synthetic_driver(monkeypatch, tmp_path, bad_final=True)
    with pytest.raises(ValueError, match='finite real vector'): call()
    folder = tmp_path/'synthetic'
    assert (folder/'direction.npy').exists()
    assert np.isnan(np.load(folder/'applied-direction.npy')).all()
    assert np.isnan(np.load(folder/'true-linear-residual.npy')).all()
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'failed_or_interrupted' and not result['comparison_qualified']


def test_driver_early_return_saved_but_unqualified(monkeypatch, tmp_path):
    _, call = synthetic_driver(monkeypatch, tmp_path, early=True)
    with pytest.raises(ValueError, match='fewer than 50'): call()
    folder = tmp_path/'synthetic'
    assert (folder/'direction.npy').exists() and not (folder/'applied-direction.npy').exists()
    assert json.loads((folder/'result.json').read_text())['operation_attempted']['action'] == 0


def test_derivative_parity_failure_blocks_solve(monkeypatch, tmp_path):
    driver, call = synthetic_driver(monkeypatch, tmp_path)
    original = driver.inputs
    def bad_inputs():
        arrays, *remaining = original()
        arrays['gradient'] = np.array([99., 99.])
        return (arrays, *remaining)
    monkeypatch.setattr(driver, 'inputs', bad_inputs)
    monkeypatch.setattr(linear, 'gmres', lambda *a, **kw: pytest.fail('parity failure must block solve'))
    with pytest.raises(ValueError, match='parity failed'): call()
    assert (tmp_path/'synthetic'/'reference-gradient.npy').exists()
    assert not (tmp_path/'synthetic'/'direction.npy').exists()


def test_launcher_partial_manifest_still_retained(monkeypatch, tmp_path):
    out = tmp_path/'out'
    code = 'from pathlib import Path;p=Path(' + repr(str(out)) + ");p.mkdir();(p/'manifest.json').write_text('{')"
    _, call = fake_launch(monkeypatch, tmp_path, code)
    receipt = call()
    assert not receipt['successful'] and 'raw_validation_error' in receipt
    assert (tmp_path/'launcher'/'receipt.json').exists()
    assert (out/'manifest.json').read_text() == '{'
    assert out.stat().st_mode & 0o222 == 0


def test_complete_input_inventory_rejects_directory_and_symlink(monkeypatch, tmp_path):
    driver = module('diagnose_cifar_restart')
    folder=tmp_path/'input'; folder.mkdir(); (folder/'nested').mkdir()
    monkeypatch.setattr(driver.prior, 'verify_folder', lambda *a: pytest.fail('must reject before manifest access'))
    with pytest.raises(ValueError, match='regular files'): driver.verify_frozen(folder, 'a'*64)
    (folder/'nested').rmdir(); (folder/'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='regular files'): driver.verify_frozen(folder, 'a'*64)
