"""Generate or check the compact scalar-mechanism receipt from hash-bound local raw evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(data_root):
    attempts = []
    for name in ('attempt-001-unit', 'attempt-002-pilot'):
        directory = data_root/'explore/scalar-mechanism'/name
        manifest_path = directory/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest['files']:
            path = directory/entry['path']
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError(f'raw artifact mismatch: {path}')
        attempts.append({'relative_path': str(directory.relative_to(data_root)),
                         'manifest_sha256': sha(manifest_path), 'manifest_bytes': manifest_path.stat().st_size,
                         'source_commit': manifest['metadata']['source_commit'],
                         'plan_sha256': manifest['metadata']['plan_sha256'], 'files': len(manifest['files'])})
    results = json.loads((data_root/attempts[1]['relative_path']/'results.json').read_text())['results']
    table, balanced_errors, ns_residuals, coefficient_errors, slopes = [], [], [], [], []
    representative = None
    for row in results:
        item = {k: row[k] for k in ('ridge', 'ratio', 'outcome')}
        if row['ratio'] == 1:
            analytic = (np.sqrt(5)-1)/(row['target']-row['ridge'])
            balanced_errors.append(abs(row['eta_boundary']-analytic))
        if row['outcome'] == 'neimark_sacker':
            item.update(eta=row['ns_root']['eta'], rotation=row['coefficient_analytic']['theta_over_2pi'],
                        coefficient=row['coefficient_analytic']['d'])
            ns_residuals.append(row['ns_root']['residual'])
            coefficient_errors.extend(abs(row['coefficient_analytic']['d']-x['d'])
                                      for x in row['coefficient_finite_difference'])
            slopes.extend(x['d_modulus_d_eta'] for x in row['transversality'])
            if abs(row['ridge']-.2) < 1e-12 and abs(row['ratio']-np.sqrt(8)) < 1e-12:
                representative = row
        table.append(item)
    probes = representative['circle_probes']
    amplitudes = [p['rms_amplitude'] for p in probes]
    deltas = [p['delta'] for p in probes]
    exponent = float(np.polyfit(np.log(deltas), np.log(amplitudes), 1)[0])
    reverse = next(row for row in results if row['ratio'] < 1)
    reciprocity = {'rate_residual': abs(reverse['ns_root']['eta']-np.sqrt(8)*representative['ns_root']['eta']),
                   'angle_residual': abs(reverse['coefficient_analytic']['theta_over_2pi']-representative['coefficient_analytic']['theta_over_2pi']),
                   'both_coefficients_negative': bool(reverse['coefficient_analytic']['d'] < 0 and representative['coefficient_analytic']['d'] < 0)}
    return {'label': 'exploratory', 'evidence_rung': 0, 'data_root': str(data_root), 'attempts': attempts,
            'generator_sha256': sha(Path(__file__)),
            'counts': dict(Counter(row['outcome'] for row in results)), 'table': table,
            'max_balanced_flip_rate_error': max(balanced_errors), 'max_ns_root_residual': max(ns_residuals),
            'max_analytic_fd_coefficient_error': max(coefficient_errors), 'crossing_slope_range': [min(slopes), max(slopes)],
            'representative': {'ridge': .2, 'ratio': float(np.sqrt(8)), 'eta': representative['ns_root']['eta'],
                               'rotation': representative['coefficient_analytic']['theta_over_2pi'],
                               'coefficient': representative['coefficient_analytic']['d'],
                               'ns_root': representative['ns_root'], 'probes': probes,
                               'amplitude_exponent': exponent}, 'reciprocity': reciprocity,
            'scope': 'First attracting 2-cycle from one deterministic initial direction per case; fixed target 1.2; not all attractors, no interval proof, no neural-network utility claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/data2/neural-shrimp-data'))
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = derive(args.data_root)
    if args.write:
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    else:
        if json.loads(args.receipt.read_text()) != result:
            raise ValueError('receipt differs from hash-bound raw evidence')
    print(json.dumps({'counts': result['counts'], 'amplitude_exponent': result['representative']['amplitude_exponent'],
                      'max_ns_root_residual': result['max_ns_root_residual'],
                      'max_analytic_fd_coefficient_error': result['max_analytic_fd_coefficient_error'],
                      'receipt_matches_raw_evidence': True}))


if __name__ == '__main__':
    main()
