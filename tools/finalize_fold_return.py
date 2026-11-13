"""Reverify FGR1 evidence and combine its first-return and cycle figures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fold_global_return import backup_file, sha, write_json
from audit_fold_global_return import render


def main(args):
    audit, cycle, out = map(Path, (args.audit, args.cycle, args.out))
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((audit / 'audit.json').read_text())
    rows = json.loads((audit / 'rows.json').read_text())
    verification = json.loads((cycle / 'verification.json').read_text())
    run, backup = Path(summary['raw_path']), Path(summary['backup_path'])
    for root, copy in [(run, backup), (cycle, backup / 'cycle-verification')]:
        assert sha(root / 'manifest.json') == sha(copy / 'manifest.json')
        for rec in json.loads((root / 'manifest.json').read_text()):
            for base in (root, copy):
                file = base / rec['path']
                assert file.stat().st_size == rec['bytes'] and sha(file) == rec['sha256']
    assert verification['run_manifest_sha256'] == summary['manifest_sha256'] == sha(run / 'manifest.json')
    assert verification['plan_sha256'] == summary['plan_sha256']
    assert verification['precision']['passed']
    assert verification['status'] == 'numerically attracting primitive period-three orbit'
    with np.load(cycle / 'orbit-and-monodromy.npz') as z:
        orbit, M = z['orbit'], z['monodromy']
        spectrum = np.linalg.eigvals(M)
        assert abs(float(np.max(np.abs(spectrum))) - verification['spectral_radius']) < 1e-10
        assert M.shape == (896, 896) and orbit.shape == (3, 896)
        assert np.isfinite(M).all() and np.isfinite(orbit).all()
        matches = [min(np.linalg.norm(x-y)/np.linalg.norm(y) for y in orbit) for x in z['candidate_end_states']]
        assert max(matches) < 1e-9
    summary['cycle_verification'] = verification
    render(out, rows, summary, .00393259341682185)
    with np.load(run / 'inputs.npz') as z:
        seeds, left = z['seeds'], z['left']
        delta = seeds[77] - seeds[64]
        matched = {'baseline_row': 64, 'perturbed_row': 77,
                   'baseline_return_updates': rows[64]['entry_update'],
                   'relative_full_state_change': float(np.linalg.norm(delta)/np.linalg.norm(seeds[64])),
                   'slow_coordinate_change': float(delta @ left),
                   'baseline_v_over_U': rows[64]['exit_v_over_U'],
                   'perturbed_outcome': 'numerically attracting period-three orbit'}
    compact_predictions = {k:{name:value for name,value in rec.items() if name.endswith(('RMSE', 'MAE'))}
                           for k,rec in summary['prediction']['models'].items()}
    receipt = {k:summary[k] for k in ('source_commit', 'plan_sha256', 'manifest_sha256', 'raw_path',
                'backup_path', 'all_manifest_hashes_verified_on_both_disks', 'files_verified_per_disk',
                'bytes_per_disk', 'counts', 'actual_model_updates_across_rows', 'batch_iterations',
                'wall_seconds', 'return_updates_range', 'entry_v_over_U_range', 'phase_at_return_counts', 'baseline_replay')}
    receipt.update({'status': 'completed exploratory first-return pilot and numerical cycle verification',
                    'finalizer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'physical_disks': {'primary': '/dev/nvme0n1', 'backup': '/dev/sda'},
                    'audit_path': str(audit), 'cycle_path': str(cycle), 'figures_path': str(out),
                    'audit_sha256': sha(audit / 'audit.json'), 'rows_sha256': sha(audit / 'rows.json'),
                    'cycle_verification_sha256': sha(cycle / 'verification.json'),
                    'cycle_manifest_sha256': sha(cycle / 'manifest.json'), 'cycle_status': verification['status'],
                    'cycle_spectral_radius': verification['spectral_radius'],
                    'cycle_closure_relative_residual': verification['primitive_closure'][2],
                    'cycle_lag1_lag2_relative_residuals': verification['primitive_closure'][:2],
                    'cycle_leading_multipliers': verification['leading_multipliers'],
                    'cycle_max_phase_match_relative_error': max(matches),
                    'cycle_readout_top1': [r[0] for r in verification['readout']],
                    'cycle_derivative_precision': verification['precision'],
                    'matched_control': matched, 'grouped_holdout_predictions': compact_predictions,
                    'holdout_groups': [3,7], 'holdout_rows': 32, 'all_holdout_rows_returned': True,
                    'figures_sha256': {p.name:sha(p) for p in out.glob('*.png')},
                    'limitations': ['One parameter setting, eight exits from one original trajectory.',
                        'Designed perturbations are not a random population sample or basin-volume estimate.',
                        'First-return stopping does not establish sustained reinjection or the later fate of returned rows.',
                        'Cycle verification uses floating-point numerics, not interval-certified bounds.',
                        'Period three in this high-dimensional map does not establish chaos.',
                        'Neither tested polynomial return model gave a reliable held-out predictor.']})
    write_json(out / 'receipt.json', receipt)
    for p in sorted(out.iterdir()):
        backup_file(p, out, backup / 'final')
    if args.receipt:
        write_json(args.receipt, receipt)
    print(json.dumps({'receipt': str(out / 'receipt.json'), 'counts': receipt['counts'],
                      'spectral_radius': verification['spectral_radius'], 'matched_control': matched}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--cycle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--receipt')
    main(parser.parse_args())
