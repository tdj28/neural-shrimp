"""Independent saved-state CIFAR residual audit; no Newton, spectrum, or trajectory run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

SOURCE = '7de5dcfb5eb50ed7829bc3b69b43c6e2c4f889cd'
DATA = Path('/data2/neural-shrimp-data')
ATTEMPT = DATA / 'explore/cifar/corrector-only/attempt-001'
ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(p):
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def gradient(theta, x, y, ridge):
    """Explicit chain rule, independently implemented without producer/Torch imports."""
    d, h, k = x.shape[1], 32, y.shape[1]
    a, b, c = d*h, d*h+h, d*h+h+h*k
    w1, b1 = theta[:a].reshape(d, h), theta[a:b]
    w2, b2 = theta[b:c].reshape(h, k), theta[c:]
    hidden = np.tanh(x @ w1 + b1)
    err = (hidden @ w2 + b2 - y) / len(x)
    back = (err @ w2.T) * (1-hidden*hidden)
    return np.concatenate([(x.T @ back).ravel(), back.sum(0),
                           (hidden.T @ err).ravel(), err.sum(0)]) + ridge*theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'reviewer_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              'audited_commit': SOURCE, 'pid': os.getpid(), 'nice': os.getpriority(os.PRIO_PROCESS, 0),
              'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
              'numpy': np.__version__, 'method': 'independent NumPy analytic gradient; exactly two local map evaluations at saved state; no correction/spectrum/trajectory extension',
              'checks': {}}
    for value in result['thread_environment'].values():
        assert value == '1'
    manifest = json.loads((ATTEMPT/'manifest.json').read_text())
    result['attempt_manifest'] = identity(ATTEMPT/'manifest.json')
    for name, meta in manifest.items():
        assert sha(ATTEMPT/name) == meta['sha256'] and (ATTEMPT/name).stat().st_size == meta['bytes'], name
    assert set(manifest) == {p.name for p in ATTEMPT.iterdir() if p.name != 'manifest.json'}
    orbit = json.loads((ATTEMPT/'orbit.json').read_text())
    assert orbit['source_commit'] == SOURCE
    for name, expected in orbit['source_sha256'].items():
        data = subprocess.check_output(['git','show',SOURCE+':'+name], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == expected, name
    events = [json.loads(s) for s in (ATTEMPT/'events.jsonl').read_text().splitlines()]
    launcher_path = DATA/'logs/cifar-corrector-only-attempt-001.launcher.json'
    log_path = launcher_path.with_suffix('').with_suffix('.log')
    launcher = json.loads(launcher_path.read_text())
    assert sha(log_path) == launcher['log_sha256']
    assert launcher['source_commit'] == SOURCE and launcher['exit_code'] == 0
    assert launcher['threads'] == 1 and not launcher['gpu_requested'] and not launcher['paid_resources']
    assert '--corrector-only' in launcher['command']
    result['launcher'] = identity(launcher_path)
    result['launcher_log'] = identity(log_path)
    result['elapsed_producer_seconds'] = launcher['elapsed_seconds']
    row = orbit['rows'][0]
    assert len(orbit['rows']) == 1 and not orbit['bisection_rows'] and orbit['crossing'] is None
    assert row['newton_iterations'] == 0 and len(row['newton_history']) == 1
    assert row['newton_history'][0]['status'] == 'linear_solver_failed'
    assert row['newton_history'][0]['gmres_info'] == 4
    assert row['spectrum_status'] == 'not_requested'
    assert not any('arnoldi' in e['kind'] or 'spectrum' in e['kind'] for e in events)
    source_state = Path(orbit['args']['state'])
    state_paths = sorted(ATTEMPT.glob('state-*.npy'))
    assert len(state_paths) == 4
    assert all(sha(p) == sha(source_state) == orbit['state_sha256'] for p in state_paths)
    theta = np.load(state_paths[-1])
    assert theta.dtype == np.float64 and theta.shape == (98666,)
    result['input_state'] = identity(source_state)
    result['saved_states'] = [identity(p) for p in state_paths]
    # Read the archive containers, then select A only. Never read the B index list.
    subset_path = DATA/'explore/cifar/subsets/subset-A.json'
    subset = json.loads(subset_path.read_text())
    indices = np.asarray(subset['indices'], dtype=np.int64)
    assert sha(subset_path) == orbit['data_identity']['subset_a_record_sha256']
    assert hashlib.sha256(indices.tobytes()).hexdigest() == subset['sha256_int64'] == orbit['data_identity']['subset_a_indices_sha256_int64']
    pixels, labels = [], []
    for number in range(1, 6):
        with (DATA/f'datasets/cifar10/cifar-10-batches-py/data_batch_{number}').open('rb') as handle:
            batch = pickle.load(handle, encoding='bytes')
        pixels.append(np.asarray(batch[b'data'], dtype=np.uint8))
        labels.extend(batch[b'labels'])
    selected = np.concatenate(pixels)[indices].astype(np.float64)
    selected_labels = np.asarray(labels)[indices]
    scale = selected.std(0); scale[scale == 0] = 1
    x = (selected-selected.mean(0))/scale
    y = np.eye(10)[selected_labels]
    for name, array in [('x', x), ('y', y)]:
        expected = orbit['data_identity'][name]
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256']
        assert list(array.shape) == expected['shape'] and array.dtype.str == expected['dtype']
    eta, ridge = orbit['args']['eta'], orbit['args']['lam']
    assert eta == .00827753 and ridge == .01 and orbit['args']['output_ratio'] == 1
    first = theta - eta*gradient(theta,x,y,ridge)
    second = first - eta*gradient(first,x,y,ridge)
    denominator = max(1.0, np.linalg.norm(theta))
    residual, separation = np.linalg.norm(second-theta)/denominator, np.linalg.norm(first-theta)/denominator
    assert abs(residual-row['relative_residual']) < 1e-12
    assert abs(separation-row['separation']) < 1e-12
    assert residual > 1e-8 and separation > 1e-7
    result['recomputed'] = {'relative_closure': float(residual), 'separation': float(separation),
                            'closure_difference': float(residual-row['relative_residual']),
                            'separation_difference': float(separation-row['separation']),
                            'closure_gate': 1e-8, 'separation_gate': 1e-7,
                            'closure_pass': False, 'separation_pass': True}
    result['checks'] = {'all_manifest_bytes': True, 'source_blobs': True, 'launcher_log': True,
                        'all_returned_states_identical_to_input': True, 'zero_accepted_newton_steps': True,
                        'zero_arnoldi_evidenced_by_corrector_only_source_and_events': True,
                        'selected_A_data_identity_matches': True,
                        'holdout_scope': 'Only A selected/evaluated; raw archive containers include other rows. B index file and reserved seeds never used.'}
    result['verdict'] = 'pass: accurately reported unresolved corrector; no accepted orbit or stability inference'
    result['elapsed_audit_seconds'] = time.monotonic()-started
    target = args.output/'review.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verdict':result['verdict'], **result['recomputed'], 'output':str(target)}))


if __name__ == '__main__':
    main()
