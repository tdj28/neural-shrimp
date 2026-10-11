"""One fixed CPU restart200 comparison, source-only until separately authorized."""
import os
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREADS:
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
import scipy
import torch
from scipy.sparse.linalg import gmres
import diagnose_cifar_linear_solve as prior
from neural_shrimp import cifar
from neural_shrimp.cifar_linear_diagnostic import solve_direction, evaluate_trials
from neural_shrimp.cifar_restart_diagnostic import (Budget, CountedProblemMixin,
    Prefix, Store, SETTINGS, byte_equal, require, request_stop)
BASELINE = Path('/data2/neural-shrimp-data/explore/cifar/linear-diagnostic/attempt-001')
BASELINE_SHA = 'd47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3'
BASELINE_SOURCE = 'a9830e714d259597a53cee6ef7be41a9f07bb205'
GMRES_SHA = 'cf8316712dfd558ce3153637c6176d285161de1347309ab7f6139abc80122a06'
VERSIONS = dict(torch='2.14.1+cu130', numpy='2.5.3', scipy='1.18.1')
SOURCES = ('src/neural_shrimp/__init__.py', 'src/neural_shrimp/controls.py',
    'src/neural_shrimp/models.py', 'src/neural_shrimp/cifar.py',
    'src/neural_shrimp/cifar_linear_diagnostic.py', 'src/neural_shrimp/cifar_restart_diagnostic.py',
    'tools/diagnose_cifar_linear_solve.py', 'tools/diagnose_cifar_restart.py', 'tools/launch_cifar_restart.py',
    'docs/plans/cifar-restart-sensitivity-decision.md', 'docs/plans/cifar-restart-sensitivity-decision.json',
    'docs/plans/cifar-restart-sensitivity-implementation.md')


def sources():
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'clean checkout required')
    inventory = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        require(data == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT), 'uncommitted source: ' + name)
        inventory[name] = hashlib.sha256(data).hexdigest()
    return inventory


def environment():
    versions = dict(torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__)
    digest = hashlib.sha256(inspect.getsource(gmres).encode()).hexdigest()
    require(versions == VERSIONS and digest == GMRES_SHA, 'frozen library or GMRES source mismatch')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice >=10 required')
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Torch threads must be one')
    require(all(os.environ.get(k) == '1' for k in THREADS), 'BLAS thread environment')
    return dict(**versions, gmres_python_source_sha256=digest, python=platform.python_version(),
        pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0), device='cpu', gpu=False,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        thread_env={k: os.environ[k] for k in THREADS})


def verify_frozen(folder, digest):
    require(not folder.is_symlink() and folder.is_dir(), 'frozen input directory')
    require(all(p.is_file() and not p.is_symlink() for p in folder.iterdir()), 'frozen inventory must contain regular files only')
    return prior.verify_folder(folder, digest)


def inputs():
    verify_frozen(BASELINE, BASELINE_SHA)
    verify_frozen(prior.BUNDLE, prior.BUNDLE_MANIFEST_SHA)
    verify_frozen(prior.OLD, prior.OLD_MANIFEST_SHA)
    arrays, basis, old = prior.load_inputs()
    baseline = json.loads((BASELINE / 'result.json').read_text())
    linear = json.loads((BASELINE / 'linear.json').read_text())
    require(baseline['status'] == 'diagnostic_complete' and baseline['source_commit'] == BASELINE_SOURCE, 'baseline source/status')
    require(baseline['settings']['restart'] == 50 and baseline['settings']['cycles'] == 4 and
            baseline['settings']['max_matvec'] == 205 and linear['inner_iterations'] == 200 and
            linear['matvec_calls'] == 205, 'baseline fixed solver settings/counts')
    require(baseline['input_identity']['data_identity'] == basis['data_identity'] and
            baseline['input_identity']['state_sha256'] == prior.STATE_SHA, 'baseline input identity')
    require(all(baseline[k] == value for k, value in VERSIONS.items()) and
            baseline['gmres_python_source_sha256'] == GMRES_SHA, 'baseline numerical implementation')
    saved = {name: np.load(BASELINE / (name + '.npy'), allow_pickle=False)
             for name in ('initial-theta', 'initial-map1', 'initial-map2', 'initial-residual')}
    require(byte_equal(saved['initial-theta'], arrays['theta']), 'baseline theta bytes')
    identity = dict(bundle=str(prior.BUNDLE), bundle_manifest_sha256=prior.BUNDLE_MANIFEST_SHA,
        old_corrector=str(prior.OLD), old_manifest_sha256=prior.OLD_MANIFEST_SHA,
        baseline=str(BASELINE), baseline_manifest_sha256=BASELINE_SHA, baseline_source_commit=BASELINE_SOURCE,
        state_sha256=prior.STATE_SHA, data_identity=basis['data_identity'], model=basis['model'],
        eta=basis['eta'], lam=basis['lam'], reference_source_commit=basis['source_commit'])
    return arrays, basis, saved, linear, identity


class CountedProblem(CountedProblemMixin, cifar.Problem):
    pass


def replay(store, name, actual, saved):
    store.array(name, actual)  # Mismatching output remains inspectable.
    require(byte_equal(actual, saved[name]), 'byte-exact initial replay mismatch: ' + name)


def run(out, review_commit):
    inventory = sources()  # Before output creation.
    require(isinstance(review_commit, str) and re.fullmatch('[0-9a-f]{40}', review_commit), 'full review commit required')
    subprocess.run(['git', 'cat-file', '-e', review_commit + '^{commit}'], cwd=ROOT, check=True)
    runtime = environment()
    store = Store(out)
    started = time.monotonic()
    budget = Budget(store.event)
    result = dict(schema='cifar-restart-sensitivity-v1', status='running', evidence_rung=0,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=inventory, review_commit=review_commit, runtime=runtime, settings=SETTINGS,
        started_utc=datetime.now(timezone.utc).isoformat(), accepted_orbit=False,
        applied_to_continuation=False, spectrum_requested=False, comparison_qualified=False)
    prefix = None
    try:
        store.json('runtime-start.json', result)
        arrays, basis, saved, baseline, identity = inputs()
        result['input_identity'] = identity
        store.json('input-identity.json', identity)
        budget.check()
        problem = CountedProblem(cifar.MLP(**basis['model']), arrays['x'], arrays['y'], basis['lam'])
        problem.budget = budget
        theta, eta = cifar.as_tensor(arrays['theta']), basis['eta']
        replay(store, 'initial-theta', theta.numpy(), saved)
        first = problem.update(theta, eta)
        replay(store, 'initial-map1', first.numpy(), saved)
        second = problem.update(first, eta)
        replay(store, 'initial-map2', second.numpy(), saved)
        residual = (second - theta).numpy()
        replay(store, 'initial-residual', residual, saved)
        result['initial'] = dict(residual_norm=float(np.linalg.norm(residual)),
            relative_residual=float(np.linalg.norm(residual)) / max(1., float(theta.norm())),
            relative_separation=float((first - theta).norm()) / max(1., float(theta.norm())))
        store.json('initial.json', result['initial'])
        v = cifar.as_tensor(arrays['directions'][0])
        result['reference_parity'] = {}
        for name, calculate, expected in (
            ('gradient', lambda: problem.grad(theta), arrays['gradient']),
            ('hvp', lambda: problem.hvp(theta, v), arrays['hvp'][0]),
            ('j2vp', lambda: problem.jvp_two_step(theta, v, eta, theta_next=first), arrays['j2vp'][0])):
            actual = calculate().numpy()
            store.array('reference-' + name, actual)
            result['reference_parity'][name] = prior.parity(actual, expected)
        store.json('parity.json', result['reference_parity'])
        prefix = Prefix(baseline['estimated_relative_residual_history'], store.event)
        final_action = False
        def action(vector):
            def calculate():
                applied = problem.jvp_two_step(theta, cifar.as_tensor(vector), eta, theta_next=first).numpy() - vector
                if final_action:
                    # Retain returned Ap/e before the core's finite/true-residual checks.
                    store.array('applied-direction', applied)
                    store.array('true-linear-residual', applied + residual)
                return applied
            return budget.call('action', calculate)
        def save_returned(step, info):
            nonlocal final_action
            store.array('direction', step)
            store.json('returned-direction.json', dict(gmres_info=info, shape=list(step.shape),
                finite=bool(np.isfinite(step).all()), pending_independent_true_check=True))
            prefix.complete()  # Early return is retained but cannot qualify this comparison.
            final_action = True
        step, applied, error, linear = solve_direction(action, residual,
            **{k: SETTINGS[k] for k in ('restart', 'cycles', 'rtol', 'forcing_limit', 'max_matvec', 'explicit_zero_start')},
            deadline=budget.deadline, event=prefix, save_direction=save_returned)
        result['linear'] = linear
        store.json('linear.json', linear)
        def evaluate(trial, index):
            t = cifar.as_tensor(trial)
            one = problem.update(t, eta)
            store.array(f'trial-{index}-map1', one.numpy())
            two = problem.update(one, eta)
            store.array(f'trial-{index}-map2', two.numpy())
            r = (two - t).numpy()
            store.array(f'trial-{index}-residual', r)
            return dict(residual=r, relative_separation=float((one - t).norm()) / max(1., float(t.norm())))
        result['trials'] = evaluate_trials(theta.numpy(), residual, step, linear, evaluate,
            armijo=SETTINGS['armijo'], deadline=budget.deadline,
            save_state=lambda index, trial: store.array(f'trial-{index}-theta', trial), event=store.event)
        budget.check()
        result.update(status='diagnostic_complete', comparison_qualified=True,
            candidate_trial_indices=[row['index'] for row in result['trials'] if row['inexact_step_candidate']],
            baseline_true_forcing_ratio=baseline['true_forcing_ratio'])
    except BaseException as error:
        result.update(status='failed_or_interrupted', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            operation_attempted=budget.attempted, operation_returned=budget.returned,
            retained_prefix_values=[] if prefix is None else prefix.values,
            finished_utc=datetime.now(timezone.utc).isoformat())
        store.finish(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--review-commit', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run(args.out, args.review_commit)
    print(json.dumps(dict(status=result['status'], comparison_qualified=result['comparison_qualified'],
                         accepted_orbit=False, elapsed_seconds=result['elapsed_seconds'])))

if __name__ == '__main__':
    main()
