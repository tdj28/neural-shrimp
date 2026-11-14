"""Compact receipt for the FGR3 follow-up (unwrapped return, FGR4/FGR4b zoom, frozen FGR5 test).

Binds every raw run and audit by manifest hash on both disks, the plans, the frozen prediction and score, the key
numbers quoted in the papers and blog, and the figure files. Reads files only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha, write_json

RAW = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/global-return')
AUD = Path('/data2/neural-shrimp-data/audits/fold-global-return')
ARC = Path('/archive/neural-shrimp-backups')
RUNS = ['20261114-fgr4-zoom-part0', '20261114-fgr4-zoom-part1', '20261114-fgr4b-zoom-part0',
        '20261114-fgr4b-zoom-part1', '20261114-fgr5-fresh-01']
AUDITS = {'unwrap': '20261114-unwrap-01', 'zoom': '20261114-fgr4-zoom-audit-02',
          'prediction': '20261114-fgr5-prediction-01', 'score': '20261114-fgr5-score-01',
          'robustness': '20261114-fgr5-robustness-01'}
FIGS = ['fig-fold-return-unwrapped', 'fig-fold-return-zoom', 'fig-fold-return-escape-test']


def both(local, backup):
    a, b = sha(local/'manifest.json'), sha(backup/'manifest.json')
    if a != b:
        raise ValueError(f'Backup manifest differs: {local}')
    for f in json.loads((local/'manifest.json').read_text()):
        for base in (local, backup):
            p = base/f['path']
            if p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
                raise ValueError(f'Manifest entry fails: {p}')
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT/'docs/fold-return-followup.json'))
    p.add_argument('--article-figure', default='/data2/chaos-journal-neural-shrimp/fig-fold-return-followup.pdf')
    args = p.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    runs = {}
    for r in RUNS:
        d = RAW/r
        complete = json.loads((d/'complete.json').read_text())
        runs[r] = {'raw': str(d), 'backup': str(ARC/r), 'manifest_sha256': both(d, ARC/r),
                   'complete': complete, 'source_commit': json.loads((d/'metadata.json').read_text())['source_commit']}
    audits = {k: {'path': str(AUD/v), 'backup': str(ARC/v), 'manifest_sha256': both(AUD/v, ARC/v)}
              for k, v in AUDITS.items()}
    uw = json.loads((AUD/AUDITS['unwrap']/'audit.json').read_text())
    zoom = json.loads((AUD/AUDITS['zoom']/'audit.json').read_text())
    pred = json.loads((ROOT/'docs/fold-return-fgr5-prediction.json').read_text())
    score = json.loads((ROOT/'docs/fold-return-fgr5-score.json').read_text())
    rob = json.loads((AUD/AUDITS['robustness']/'robustness.json').read_text())
    if sha(ROOT/'docs/fold-return-fgr5-prediction.json') != score['prediction_sha256']:
        raise ValueError('Score was not made against the committed prediction')
    lr = uw['one_cycle']['resolved_long_runs']; un = uw['one_cycle']['unresolved']
    sw = uw['one_cycle']['resolved_short_windows']
    levels = {str(l['delta']): {k: l[k] for k in ('pairs', 'uncertain', 'fraction')}
              for l in zoom['uncertainty_time']['levels']}
    figs = {}
    for name in FIGS:
        for suffix in ('', '-mobile'):
            for ext in ('pdf', 'png'):
                f = ROOT/'paper/llm/figures'/f'{name}{suffix}.{ext}'
                figs[f.name] = {'bytes': f.stat().st_size, 'sha256': sha(f)}
    art = Path(args.article_figure)
    figs['article:'+art.name] = {'bytes': art.stat().st_size, 'sha256': sha(art)}
    receipt = {
        'study': 'FGR3 follow-up: unwrapped return, FGR4/FGR4b zoom, frozen FGR5 test',
        'receipt_source_commit': head,
        'plans': {f: sha(ROOT/'docs'/f) for f in ('fold-return-fgr4-plan.json', 'fold-return-fgr4b-plan.json',
                                                  'fold-return-fgr5-plan.json')},
        'prediction': {'file': 'docs/fold-return-fgr5-prediction.json',
                       'sha256': sha(ROOT/'docs/fold-return-fgr5-prediction.json'),
                       'committed_before_fresh_outcomes_were_read': '22b11cd26c6f87b476203e93da51a5eeb70fbe46'},
        'score': {'file': 'docs/fold-return-fgr5-score.json', 'sha256': sha(ROOT/'docs/fold-return-fgr5-score.json')},
        'runs': runs, 'audits': audits,
        'numbers': {
            'unwrap': {'long_runs': uw['resolved_runs_at_least_0.02_wide'],
                       'fresh_departures': uw['one_cycle']['all']['departures'],
                       'long_run_returns': lr['returns'], 'long_run_time_correct_unwrapped': lr['unwrapped_time_correct'],
                       'long_run_time_correct_fgr3_table': lr['table_time_correct'],
                       'long_run_median_tau_error': lr['unwrapped_tau_error_quantiles_50_90_99'][0],
                       'M_departures': un['departures'], 'M_captures': un['captures'],
                       'short_window_returns': sw['returns'], 'short_window_time_correct': sw['unwrapped_time_correct'],
                       'table_stops_in_M': uw['table_stops']-uw['table_stops_in_resolved_intervals'],
                       'table_stops': uw['table_stops'], 'odd_time_table_returns': uw['odd_time_table_returns']},
            'zoom': {'evaluations': zoom['evaluations'], 'bases_inside_M': zoom['bases_inside_M'],
                     'levels_time_criterion': levels,
                     'alpha_time': zoom['uncertainty_time']['alpha'],
                     'alpha_time_bootstrap_95': zoom['uncertainty_time']['alpha_bootstrap_95'],
                     'alpha_tau': zoom['uncertainty_tau']['alpha'],
                     'alpha_tau_bootstrap_95': zoom['uncertainty_tau']['alpha_bootstrap_95'],
                     'outcome_kinds': zoom['outcome_kinds'],
                     'fgr4_replicates_same_outcome_and_time': 16, 'fgr4_replicates_bitwise_identical': 5,
                     'fgr4_replicate_max_next_q_shift_U': 2.5910997347722287e-08},
            'prediction': {'kappa': pred['kappa'], 'kappa_bootstrap_2.5_50_97.5': pred['kappa_bootstrap_95'],
                           'sensitivity_bins': pred['sensitivity_bins'], 'bases_used': pred['bases_used']},
            'score': {k: score[k] for k in score if k not in ('status',)},
            'robustness': {k: rob[k] for k in ('kappa_trajectory_bootstrap_95', 'departures_per_trajectory_range')}},
        'figures': figs,
        'limits': ('One prompt, steering direction and parameter setting. M and the unwrapping rule were defined after '
                   'the FGR3 fresh trajectories were seen; FGR5 tests them on new trajectories. An exponent measured '
                   'between 1e-10 and 1e-4 in q shows structure across those scales, not an infinite fractal. Exact '
                   'reruns in another batch slot change 9 of 466 outcomes inside M.'),
    }
    write_json(Path(args.out), receipt)
    print(json.dumps(receipt['numbers']['score'], indent=1)[:400])


if __name__ == '__main__':
    main()
