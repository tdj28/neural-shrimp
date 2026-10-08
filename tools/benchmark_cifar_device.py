"""Prepare a subset-A CPU reference bundle; probe identical float64 derivatives.

No continuation. No automatic provider launch. The probe needs only the bundle
and these source files, not CIFAR archives or held-out subset indices. CPU is
the default device. Each invocation requires an exclusive output directory and
an external process timeout; internal deadlines stop between operations.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy
import torch
from scipy.sparse.linalg import gmres

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neural_shrimp import cifar
from neural_shrimp.cifar_device import DeviceProblem, HostTwoStepOperator, parity_metrics

SOURCES = ('src/neural_shrimp/__init__.py', 'src/neural_shrimp/controls.py',
           'src/neural_shrimp/models.py', 'src/neural_shrimp/cifar.py',
           'src/neural_shrimp/cifar_device.py', 'tools/explore_cifar.py',
           'tools/benchmark_cifar_device.py')


def hashes():
    return {name: cifar.sha256_file(ROOT/name) for name in SOURCES}


def commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('device benchmark wall-time budget exhausted')


def manifest(folder):
    return {path.name: {'sha256': cifar.sha256_file(path), 'bytes': path.stat().st_size}
            for path in sorted(folder.iterdir()) if path.is_file() and path.name != 'manifest.json'}


def prepare(args, record, deadline):
    # Only preparation on zeta needs the original loader; probe is self-contained.
    from explore_cifar import subset_a_problem
    problem, model = subset_a_problem(args.lam)
    theta = np.load(args.state, allow_pickle=False)
    if theta.shape != (model.n_parameters,) or theta.dtype != np.float64 or not np.isfinite(theta).all():
        raise ValueError('expected finite float64 owned state matching model')
    rng = np.random.default_rng(28206)
    directions = rng.normal(size=(3, model.n_parameters))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    state = cifar.as_tensor(theta)
    first = problem.update(state, args.eta)
    hvps, j2vps = [], []
    for direction in directions:
        check_deadline(deadline)
        vector = cifar.as_tensor(direction)
        hvps.append(problem.hvp(state, vector).numpy())
        j2vps.append(problem.jvp_two_step(state, vector, args.eta, theta_next=first).numpy())
    arrays = dict(x=problem.x.numpy(), y=problem.y.numpy(), theta=theta, directions=directions,
        gradient=problem.grad(state).numpy(), hvp=np.asarray(hvps), j2vp=np.asarray(j2vps))
    if not all(np.isfinite(a).all() and a.dtype == np.float64 for a in arrays.values()):
        raise ValueError('reference must be finite float64')
    np.savez(args.output/'reference.npz', **arrays)
    record.update(eta=args.eta, lam=args.lam, model={'d': model.d, 'width': model.width, 'k': model.k},
        state=str(args.state), state_sha256=cifar.sha256_file(args.state),
        data_identity=problem.data_identity, direction_seed=28206, reference_source='existing CPU Problem',
        reference_sha256=cifar.sha256_file(args.output/'reference.npz'))


def probe(args, record, deadline):
    entries = json.loads((args.bundle/'manifest.json').read_text())
    if not {'result.json', 'reference.npz'} <= entries.keys():
        raise ValueError('incomplete reference bundle manifest')
    for name in ('result.json', 'reference.npz'):
        path = args.bundle/name
        if path.stat().st_size != entries[name]['bytes'] or cifar.sha256_file(path) != entries[name]['sha256']:
            raise ValueError('reference bundle hash mismatch')
    basis = json.loads((args.bundle/'result.json').read_text())
    if basis['status'] != 'complete' or basis['source_sha256'] != hashes():
        raise ValueError('reference source mismatch or incomplete preparation')
    with np.load(args.bundle/'reference.npz', allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    if not all(a.dtype == np.float64 and np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('reference bundle must contain finite float64 arrays')
    model = cifar.MLP(**basis['model'])
    problem = DeviceProblem(model, arrays['x'], arrays['y'], basis['lam'], device=args.device)
    theta = problem.tensor(arrays['theta'])
    directions = [problem.tensor(v) for v in arrays['directions']]
    first = problem.update(theta, basis['eta'])
    problem.synchronize()
    record.update(bundle=str(args.bundle), bundle_manifest_sha256=cifar.sha256_file(args.bundle/'manifest.json'),
        reference_source_commit=basis['source_commit'], data_identity=basis['data_identity'],
        eta=basis['eta'], lam=basis['lam'], parameters=model.n_parameters, samples=problem.n,
        device=str(problem.device), dtype=str(theta.dtype), scipy_location='CPU',
        gpu_name=torch.cuda.get_device_name(problem.device) if problem.device.type == 'cuda' else None,
        parity={}, timings={})
    if problem.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(problem.device)
    def measure(name, functions, expected):
        seconds, observed = [], []
        for fn in functions:
            check_deadline(deadline)
            problem.synchronize(); started = time.perf_counter()
            value = fn(); problem.synchronize()
            seconds.append(time.perf_counter()-started)
            observed.append(value.detach().cpu().numpy().copy())
        record['timings'][name] = {'seconds': seconds, 'median_seconds': float(np.median(seconds)),
            'scope': 'synchronized device operation, inputs resident; output validation transfer excluded'}
        record['parity'][name] = parity_metrics(np.asarray(observed), expected)
        write_json(args.output/'result.json', record)
    measure('gradient', [lambda: problem.grad(theta)]*3, np.repeat(arrays['gradient'][None, :], 3, axis=0))
    measure('hvp', [lambda v=v: problem.hvp(theta, v) for v in directions], arrays['hvp'])
    measure('j2vp', [lambda v=v: problem.jvp_two_step(theta, v, basis['eta'], theta_next=first)
                    for v in directions], arrays['j2vp'])
    bridge = HostTwoStepOperator(problem, theta, basis['eta'])
    bridge_values = []
    for direction in arrays['directions']:
        check_deadline(deadline)
        bridge_values.append(bridge.matvec(direction))
    record['parity']['host_bridge_j2vp'] = parity_metrics(np.asarray(bridge_values), arrays['j2vp'])
    record['host_bridge'] = bridge.measurements
    if not all(entry['passed'] for entry in record['parity'].values()):
        raise ValueError('CPU/reference parity failed; no GMRES attempted')
    operator = HostTwoStepOperator(problem, theta, basis['eta'], subtract_identity=True)
    # All derivative calls, including GMRES ones, check the remaining wall budget.
    from scipy.sparse.linalg import LinearOperator
    def bounded(vector):
        check_deadline(deadline)
        return operator.matvec(vector)
    bounded_operator = LinearOperator(operator.shape, matvec=bounded, dtype=np.float64)
    residual = (problem.update(first, basis['eta'])-theta).detach().cpu().numpy()
    history = []
    started = time.perf_counter()
    step, info = gmres(bounded_operator, -residual, rtol=1e-8, restart=5, maxiter=1,
                      callback=history.append, callback_type='pr_norm')
    record['gmres_one_cycle'] = {'seconds': time.perf_counter()-started, 'info': int(info),
        'restart': 5, 'maxiter': 1, 'relative_residual_history': history, 'step_finite': bool(np.isfinite(step).all()),
        'matvec_calls': len(operator.measurements), 'phase_measurements': operator.measurements,
        'scope': 'timing and convergence diagnostic; correction is never applied'}
    record['peak_cuda_allocated_bytes'] = (torch.cuda.max_memory_allocated(problem.device)
        if problem.device.type == 'cuda' else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--state', type=Path, required=True)
    prepare_parser.add_argument('--lam', type=float, default=.01)
    prepare_parser.add_argument('--eta', type=float, default=.00827753)
    probe_parser = sub.add_parser('probe')
    probe_parser.add_argument('--bundle', type=Path, required=True)
    probe_parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    for child in (prepare_parser, probe_parser):
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--max-seconds', type=float, default=60)
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('max-seconds must be finite and positive')
    if args.action == 'prepare' and (not np.isfinite(args.eta) or args.eta <= 0
                                    or not np.isfinite(args.lam) or args.lam < 0):
        parser.error('eta must be finite positive; lam finite nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    record = {'status': 'running', 'action': args.action, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit(), 'source_sha256': hashes(), 'host': platform.node(), 'pid': os.getpid(),
        'nice': os.getpriority(os.PRIO_PROCESS, 0), 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(), 'torch': torch.__version__,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'python': platform.python_version(), 'cuda_runtime': torch.version.cuda, 'load_average': os.getloadavg(),
        'tf32': False, 'claim_scope': 'engineering operation/parity pilot; no orbit or stability claim'}
    write_json(args.output/'result.json', record)
    try:
        (prepare if args.action == 'prepare' else probe)(args, record, started+args.max_seconds)
        record['status'] = 'complete'
    except BaseException as error:
        record['status'] = 'failed_or_interrupted'; record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        record['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.output/'result.json', record)
        write_json(args.output/'manifest.json', manifest(args.output))
    print(json.dumps({'output': str(args.output), 'status': record['status'], 'elapsed_seconds': record['elapsed_seconds']}))


if __name__ == '__main__':
    main()
