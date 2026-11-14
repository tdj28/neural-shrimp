"""Compact receipt for the transient-to-cycle figures (FGR5 series and the twin demonstration).

Binds the twin run and both audits by manifest hash on both disks, the figure files and the numbers quoted.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha, write_json
from collect_fold_return_followup_receipt import both

RAW = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/global-return')
AUD = Path('/data2/neural-shrimp-data/audits/fold-global-return')
ARC = Path('/archive/neural-shrimp-backups')
FIGS = ['fig-fold-transient-capture', 'fig-fold-twins-capture']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT/'docs/fold-transient-figures.json'))
    p.add_argument('--figure-start', type=int, required=True, help='FGR5 row of the twin start drawn in the figure')
    a = p.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    run = '20261114-twins-01'; series = '20261114-fgr5-series-01'; twins = '20261114-twins-audit-01'
    s = json.loads((AUD/series/'meta.json').read_text()); t = json.loads((AUD/twins/'audit.json').read_text())
    z = np.load(AUD/series/'series.npz'); d3, valid = z['d3'], z['valid_through']
    captured = int(sum((d3[i, :valid[i]+1] < 1e-3).any() for i in range(len(d3))))
    dc = z['dc']; pair = np.fmin(dc, np.concatenate([dc[:, :1], dc[:, :-1]], axis=1)); closest = []
    for i in range(len(dc)):                       # slow passages of at least 20 updates before capture
        hit = np.flatnonzero(d3[i, :valid[i]+1] < 1e-3); end = hit[0] if len(hit) else valid[i]+1
        e = np.flatnonzero(np.diff(np.r_[0, (pair[i, :end] < 0.03).astype(int), 0]) != 0)
        closest += [float(np.nanmin(dc[i, u:w])) for u, w in zip(e[::2], e[1::2]) if w-u >= 20]
    on_cycle_at_stop = int(sum(d3[i, valid[i]] < 1e-12 for i in range(len(d3))))
    figs = {}
    for name in FIGS:
        for suffix in ('', '-mobile'):
            for ext in ('pdf', 'png'):
                f = ROOT/'paper/llm/figures'/f'{name}{suffix}.{ext}'
                figs[f.name] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
    receipt = {
        'study': 'Transient-to-cycle figures at the post-fold setting: FGR5 series and twin demonstration',
        'receipt_source_commit': head,
        'plan': {'file': 'docs/fold-return-twins-plan.json', 'sha256': sha(ROOT/'docs/fold-return-twins-plan.json')},
        'twin_run': {'raw': str(RAW/run), 'backup': str(ARC/run), 'manifest_sha256': both(RAW/run, ARC/run),
                     'source_commit': json.loads((RAW/run/'metadata.json').read_text())['source_commit']},
        'audits': {k: {'path': str(AUD/v), 'backup': str(ARC/v), 'manifest_sha256': both(AUD/v, ARC/v)}
                   for k, v in (('series', series), ('twins', twins))},
        'numbers': {
            'fresh_trajectories': int(len(d3)), 'fresh_captured_within_1e-3': captured,
            'fresh_still_wandering_at_end': int(len(d3))-captured,
            'fresh_rows_below_1e-12_at_stop': on_cycle_at_stop,
            'cycle_spectral_radius': s['spectral_radius'], 'cycle_contraction_per_update': s['contraction_per_update'],
            'cycle_factor_per_3_updates': float(np.exp(3*s['contraction_per_update'])),
            'slow_passages': len(closest), 'passage_closest_approach_median': float(np.median(closest)),
            'passage_closest_approach_5_95': np.percentile(closest, [5, 95]).tolist(),
            'figure_start': a.figure_start,
            'example_row': 13, 'example_capture_update': int(np.flatnonzero(d3[13, :valid[13]+1] < 1e-3)[0]),
            'twins': {k: t[k] for k in ('nudged_copies', 'nudged_that_drifted_beyond_1e-3', 'drift_time_range',
                                        'implied_growth_range', 'captured', 'outcomes', 'capture_times')},
            'twin_copies': t['copies'],
            'fgr3_growth_reference': {'complete_windows': 26, 'mean_per_update': 0.0734,
                                      'range_per_update': [0.02744, 0.11295],
                                      'source': 'docs/fold-return-autonomous.json at 6343dc481f275afcb73b81b338a1597cbf2f21ed'}},
        'figures': figs,
        'limits': ('Descriptive. One prompt and parameter setting; the twin run has four starts. The regime colors use '
                   'fixed thresholds: a slow passage when this or the previous state lies within 3% of the channel '
                   'centre, captured when within 1e-3 of the cycle. Unnudged copies sit in another batch than in FGR5, '
                   'so they do not reproduce the FGR5 trajectories beyond a few hundred updates.'),
    }
    write_json(Path(a.out), receipt)
    print(json.dumps(receipt['numbers']['twins'], indent=1))


if __name__ == '__main__':
    main()
