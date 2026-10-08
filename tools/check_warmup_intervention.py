"""Check warmup artifact hashes and replay retained state transitions locally.

This is a floating-point consistency audit, not a rigorous certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.warmup import branch_access


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output):
    manifest = json.loads((output / "manifest.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    if manifest['status'] != 'complete' or metadata['status'] != 'complete':
        raise ValueError('run did not complete')
    for entry in manifest['files']:
        path = output / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'artifact mismatch: {path}')
    for entry in metadata['inputs']:
        path = Path(entry['path'])
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'input mismatch: {path}')
    net, x, y, _ = SYSTEMS['S2'](28011)
    errors, loss_errors, checked = [], [], 0
    access = []
    for path in sorted(output.glob('*.npz')):
        with np.load(path) as archive:
            trace = {key: archive[key] for key in archive.files}
            if path.stem.endswith("-common"):
                access.extend({"file": path.name, "start": start, **branch_access(trace, j)}
                              for j, start in enumerate(("near_orbit", "ordinary_initialization")))
            indices = np.unique(np.r_[0, np.arange(1, len(trace['state']), 997), len(trace['state']) - 1])
            for index in indices:
                previous = trace['initial_state'] if index == 0 else trace['state'][index - 1]
                expected = trace['state'][index]
                rates = net.rate_vector(trace['rates'][index, :, 0], trace['rates'][index, :, 1])
                finite = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(expected), axis=1)
                if not finite.any():
                    continue
                state = previous[finite].copy()
                losses = []
                for _ in range(2):
                    losses.append(net.loss(state, x, y))
                    state -= rates[finite] * net.gradient(state, x, y)
                errors.append(float(np.max(np.abs(state - expected[finite]))))
                loss_errors.append(float(np.max(np.abs(np.stack(losses, axis=1) - trace['loss'][index, finite]))))
                checked += int(finite.sum())
    if not errors or max(errors) > 1e-12 or max(loss_errors) > 1e-12:
        raise ValueError('saved-state replay exceeded 1e-12 absolute tolerance')
    alpha_errors, closures = [], []
    for path in sorted(output.glob('reference-*.json')):
        refs = json.loads(path.read_text())['references']
        targets = np.linspace(-metadata['args']['alpha'], metadata['args']['alpha'], len(refs))
        for ref, target in zip(refs, targets):
            orbit = np.asarray(ref['orbit'])
            rates = net.rate_vector([10 ** ref['h']], [10 ** ref['o']])[0]
            closure = orbit[1] - rates * net.gradient(orbit[1:2], x, y)[0] - orbit[0]
            closures.append(float(np.max(np.abs(closure))))
            alpha_errors.append(abs(ref['alpha'] - target))
    if max(closures) > 1e-9 or max(alpha_errors) > 2e-9:
        raise ValueError('reference closure or multiplier matching failed')
    return {'artifact_root': str(output), 'source_commit': metadata['source_commit'],
            'manifest_sha256': digest(output / 'manifest.json'), 'hash_checked_files': len(manifest['files']),
            'sampled_transition_cells': checked, 'max_state_replay_error': max(errors),
            'max_loss_replay_error': max(loss_errors), 'max_reference_closure': max(closures),
            'max_log_multiplier_matching_error': max(alpha_errors), 'branch_accessibility_post_run_annotation': access,
            'status': 'passed_float64_consistency_audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.output), indent=2))


if __name__ == '__main__':
    main()
