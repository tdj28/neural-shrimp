"""Audit of the twin demonstration: separation of nudged copies and their capture times (reads saved states only).

python tools/fold_twins_audit.py --run <twin run> --out <audit dir> --backup <backup dir>
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json

CYCLE = Path('/data2/neural-shrimp-data/audits/fold-global-return/20261113-cycle-01')
THRESHOLD = 1e-3      # separation at which a nudged copy counts as having drifted away
ON = 1e-3             # distance to the cycle at which a copy counts as captured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    a = p.parse_args()
    run, out, backup = Path(a.run), Path(a.out), Path(a.backup)
    out.mkdir(parents=True, exist_ok=False); backup.mkdir(parents=True, exist_ok=False)
    res = json.loads((run/'result.json').read_text()); sel = json.loads((run/'selection.json').read_text())
    labels = sel['labels']; tv = np.array(res['valid_through'])
    with np.load(CYCLE/'orbit-and-monodromy.npz') as z:
        orbit = z['orbit']
    on = np.linalg.norm(orbit, axis=1)
    shards = sorted((run/'trajectories').glob('updates-*.npz'), key=lambda q: int(q.stem.split('-')[1]))
    with np.load(run/'trajectories'/'initial.npz') as z:
        x0 = z['states']
    ref = np.array([k-k % 4 for k in range(len(labels))])
    g0 = np.linalg.norm(x0-x0[ref], axis=1)/np.linalg.norm(x0[ref], axis=1)
    gaps, d3 = [], []
    for f in shards:
        with np.load(f) as s:
            S = s['states']
        for x in S:
            gaps.append(np.linalg.norm(x-x[ref], axis=1)/np.linalg.norm(x[ref], axis=1))
            d3.append((np.linalg.norm(x[:, None, :]-orbit[None], axis=2)/on[None]).min(axis=1))
    gaps, d3 = np.array(gaps), np.array(d3); steps = np.arange(1, len(gaps)+1)
    copies = []
    for k, lab in enumerate(labels):
        live = steps <= min(tv[k], tv[ref[k]])
        g = gaps[live, k]
        cross = int(steps[live][np.argmax(g > THRESHOLD)]) if (g > THRESHOLD).any() else None
        hit = np.flatnonzero((d3[:, k] < ON) & (steps <= tv[k]))
        copies.append({'source_row': lab['source_row'], 'eps': lab['eps'], 'initial_relative_gap': float(g0[k]),
                       'outcome': res['outcomes'][k]['kind'], 'valid_through': int(tv[k]),
                       'first_within_1e-3_of_cycle': int(steps[hit[0]]) if len(hit) else None,
                       'final_distance_to_cycle': float(d3[tv[k]-1, k]),
                       'first_gap_above_1e-3': cross if lab['eps'] > 0 else None,
                       'implied_growth_per_update': (float(np.log(THRESHOLD/lab['eps'])/cross)
                                                     if lab['eps'] > 0 and cross else None)})
    rates = [c['implied_growth_per_update'] for c in copies if c['implied_growth_per_update']]
    cross = [c['first_gap_above_1e-3'] for c in copies if c['first_gap_above_1e-3']]
    audit = {'status': 'exploratory demonstration; descriptive',
             'run': str(run), 'run_manifest_sha256': sha(run/'manifest.json'),
             'cycle_sha256': sha(CYCLE/'orbit-and-monodromy.npz'), 'copies': copies,
             'nudged_copies': sum(c['eps'] > 0 for c in copies),
             'nudged_that_drifted_beyond_1e-3': len(cross),
             'drift_time_range': [min(cross), max(cross)] if cross else None,
             'implied_growth_range': [min(rates), max(rates)] if rates else None,
             'captured': sum(c['first_within_1e-3_of_cycle'] is not None for c in copies),
             'outcomes': {k: sum(c['outcome'] == k for c in copies) for k in {c['outcome'] for c in copies}},
             'capture_times': [c['first_within_1e-3_of_cycle'] for c in copies],
             'reference': 'FGR3 complete tangent-growth windows: mean 0.0734 per update, range 0.02744 to 0.11295'}
    write_json(out/'audit.json', audit)
    np.savez_compressed(out/'gaps.npz', gaps=gaps.astype(np.float32), d3=d3.astype(np.float32))
    files = []
    for f in ('audit.json', 'gaps.npz'):
        backup_file(out/f, out, backup); files.append({'path': f, 'bytes': (out/f).stat().st_size, 'sha256': sha(out/f)})
    write_json(out/'manifest.json', files); backup_file(out/'manifest.json', out, backup)
    print(json.dumps({k: v for k, v in audit.items() if k != 'copies'}, indent=1))


if __name__ == '__main__':
    main()
