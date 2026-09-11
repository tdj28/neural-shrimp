#!/usr/bin/env python3
"""Re-derive the recurrent-network result summary from immutable receipts."""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
from run_hopfield import verify, sha

ROOT = Path(__file__).resolve().parents[1]


def derive():
    directory = ROOT / 'evidence/EXP-004/attempt-001'
    verify(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    selected = json.loads((directory / 'selected.json').read_text())
    counts = {}
    for grid in ('overview', 'focus', 'transect'):
        census = Counter()
        for path in sorted((directory / grid).glob('row-*.npz')):
            with np.load(path, allow_pickle=False) as row:
                census.update(map(int, row['period']))
        counts[grid] = {'cells': sum(census.values()), 'candidate_peak_counts': dict(sorted(census.items()))}
    representatives = {}
    for role, item in selected.items():
        certificate = item['certificate']
        representatives[role] = {
            'alpha': item['alpha'], 'gamma': item['gamma'],
            'grid_peak_period': item['candidate_period'],
            'refinements': item['refinements'],
            'certificate': {k: certificate[k] for k in ('verified', 'attracting', 'reason', 'period', 'relative_closure', 'peak_count', 'multipliers', 'transverse_spectral_radius') if k in certificate},
            'behavior_metrics': item['behavior']['metrics'],
        }
    return {
        'scope': 'Numerical reproduction of a published fixed-weight three-neuron recurrent network; selected orbits verified, exact bifurcation topology and task utility untested.',
        'experiment': 'EXP-004', 'source_commit': receipt['source_commit'],
        'receipt': {'path': str((directory/'receipt.json').relative_to(ROOT)), 'sha256': sha(directory/'receipt.json')},
        'census': counts, 'total_grid_and_transect_cells': sum(v['cells'] for v in counts.values()),
        'verified_attracting_representatives': sum(bool(v['certificate']['verified']) and bool(v['certificate']['attracting']) for v in representatives.values()),
        'representatives': representatives,
        'wall_seconds': receipt['wall_seconds'], 'paid_compute_usd': 0, 'paid_review_usd': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    payload = derive()
    expected = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n'
    target = ROOT / 'evidence/hopfield-summary.json'
    if args.write:
        target.write_text(expected)
    elif target.read_text() != expected:
        raise ValueError('Derived neural-result summary drift')
    print(json.dumps({'verified': True, 'cells': payload['total_grid_and_transect_cells'], 'verified_attracting_orbits': payload['verified_attracting_representatives']}))


if __name__ == '__main__':
    main()
