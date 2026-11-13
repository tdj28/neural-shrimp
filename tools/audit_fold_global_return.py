"""CPU-only audit and figures from immutable FGR1 states, including stop padding."""
import argparse
from collections import Counter
import json
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from fold_global_return import backup_file, sha, write_json
from fold_panel_data import channel_mask, coordinates


def independent_return(x, centre, left, right, width):
    v, transverse = coordinates(x, centre, left, right, 1.)
    mask = channel_mask(v, transverse, width)
    # Direct array check, independent of the runtime streak counter.
    candidates = np.flatnonzero(mask[4:] & mask[2:-2] & mask[:-4]) + 4
    if not len(candidates):
        return None
    confirmation = int(candidates[0])
    entry = confirmation - 4
    return dict(entry_update=entry, confirmed_update=confirmation,
                entry_v=float(v[entry]), entry_transverse=float(transverse[entry]))


def independent_recurrence(x):
    if len(x) < 192:
        return None
    y = x[-128:]
    norm = np.maximum(np.linalg.norm(y, axis=1), 1e-12)
    for lag in range(1, 65):
        error = float(np.max(np.sqrt(np.sum((y - x[-128-lag:-lag])**2, axis=1)) / norm))
        if error <= 1e-9:
            return {'lag': lag, 'max_relative_error': error}
    return None


def quadratic(features):
    columns = [np.ones(len(features))]
    columns.extend(features.T)
    columns.extend(features[:, i] * features[:, j]
                   for i in range(features.shape[1]) for j in range(i, features.shape[1]))
    return np.column_stack(columns)


def prediction_checks(seeds, rows, outcomes, centre, left, right, width):
    """Fixed exploratory comparison; PCA and scaling see development exits only."""
    train = np.array([r['split'] == 'development' and o['kind'] == 'return'
                      for r, o in zip(rows, outcomes)])
    test = np.array([r['split'] == 'test' and o['kind'] == 'return'
                     for r, o in zip(rows, outcomes)])
    if train.sum() < 20 or not test.any():
        return {'status': 'insufficient returned development/test rows'}
    v = (seeds - centre) @ left
    residual = seeds - centre - v[:, None] * right
    mean = residual[train].mean(axis=0)
    _, singular, vh = np.linalg.svd(residual[train] - mean, full_matrices=False)
    pc = (residual - mean) @ vh[:3].T
    full = np.column_stack([v / width, pc])
    mu, sd = full[train].mean(axis=0), full[train].std(axis=0)
    sd[sd < 1e-14] = 1.
    full = (full - mu) / sd
    targets = np.array([[o['entry_v'] / width, o['entry_update']] if o['kind'] == 'return'
                        else [np.nan, np.nan] for o in outcomes])
    models = {'constant': np.ones((len(seeds), 1)),
              'critical_quadratic': quadratic(full[:, :1]),
              'critical_plus_three_transverse_PCs_quadratic': quadratic(full)}
    output = {'status': 'exploratory grouped holdout', 'train_rows': int(train.sum()),
              'test_rows': int(test.sum()), 'test_anchors': [3, 7],
              'ridge': .01, 'transverse_PC_variance_fraction':
              (singular[:3]**2 / np.sum(singular**2)).tolist(), 'models': {}}
    for name, design in models.items():
        penalty = .01 * np.eye(design.shape[1]); penalty[0, 0] = 0.
        coef = np.linalg.solve(design[train].T @ design[train] + penalty,
                               design[train].T @ targets[train])
        prediction = design[test] @ coef
        error = prediction - targets[test]
        output['models'][name] = {
            'entry_v_over_U_RMSE': float(np.sqrt(np.mean(error[:, 0]**2))),
            'return_updates_RMSE': float(np.sqrt(np.mean(error[:, 1]**2))),
            'entry_v_over_U_MAE': float(np.mean(np.abs(error[:, 0]))),
            'return_updates_MAE': float(np.mean(np.abs(error[:, 1]))),
            'parameters_per_target': design.shape[1],
            'test_row_ids': np.flatnonzero(test).tolist(), 'predictions': prediction.tolist(),
            'observations': targets[test].tolist()}
    return output


def audit(args):
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((run / 'metadata.json').read_text())
    backup = Path(metadata['backup'])
    manifest = json.loads((run / 'manifest.json').read_text())
    complete = json.loads((run / 'complete.json').read_text())
    assert sha(run / 'manifest.json') == complete['manifest_sha256'] == sha(backup / 'manifest.json')
    assert complete['backup_manifest_sha256'] == complete['manifest_sha256']
    for f in manifest:
        for base in [run, backup]:
            p = base / f['path']
            assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], str(p)
    plan = json.loads((run / 'plan.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    gate = json.loads((run / 'replay-gate.json').read_text())
    assert gate['passed'] and max(gate['one_update_max_abs'], gate['two_updates_max_abs']) < 1e-11
    assert sha(run / 'plan.json') == metadata['plan_sha256']
    assert sha(run / 'inputs.npz') == plan['inputs_sha256']
    assert plan['epsilon'] == 1e-5 and plan['source']['log10_temperature'] == -.115
    assert plan['source']['model'] == 'qwen0.5b' and plan['source']['layer'] == 12
    assert plan['source']['concept'] == 'philosophy' and plan['source']['window'] == 1
    original = json.loads((ROOT / 'docs/fold-panel-revision.json').read_text())['natural']['input_hashes']
    assert original == plan['input_hashes']
    source_hashes = {'fold_global_return.py': metadata['source_sha256'], **metadata['runtime_source_sha256']}
    for name, digest in source_hashes.items():
        content = subprocess.check_output(['git', 'show', metadata['source_commit'] + ':tools/' + name], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    with np.load(run / 'inputs.npz') as z:
        data = {k: z[k] for k in z.files}
    n, d = data['seeds'].shape
    assert n == 128 and d == 896 and len(plan['rows']) == n
    arrays = [data['seeds'][None]]
    expected_start = 1
    for path in sorted(run.glob('updates-*.npz')):
        start, end = map(int, path.stem.split('-')[1:])
        assert start == expected_start
        expected_start = end + 1
        with np.load(path) as z:
            states, ids, probs = z['states'], z['top_ids'], z['top_probabilities']
            assert states.shape == (end-start+1, n, d) and states.dtype == np.float64
            assert ids.shape == probs.shape == (end-start+1, n, 8)
            assert np.isfinite(states).all() and np.isfinite(probs).all()
            assert (probs >= 0).all() and (probs <= 1).all() and (probs.sum(axis=-1) <= 1. + 1e-12).all()
            assert (ids >= 0).all() and (ids < 151936).all()
            arrays.append(states)
    all_states = np.concatenate(arrays)
    audited, saved_baseline = [], []
    for i, (row, observed, length) in enumerate(zip(plan['rows'], result['outcomes'], result['valid_through'])):
        assert row['id'] == i and 1 <= length <= plan['max_updates']
        x = all_states[:length + 1, i]
        # Completed rows are padding only after this length, never recurrence evidence.
        if length + 1 < len(all_states):
            assert np.array_equal(all_states[length+1:, i], np.broadcast_to(x[-1], all_states[length+1:, i].shape))
        ret = independent_return(x, data['centre'], data['left'], data['right'], plan['width'])
        if observed['kind'] == 'return':
            assert ret is not None and length == ret['confirmed_update']
            for k in ret:
                assert abs(ret[k] - observed[k]) < 1e-13, (i, k)
        elif observed['kind'] == 'outside_recurrence_candidate':
            assert ret is None
            rec = independent_recurrence(x)
            assert rec and rec['lag'] == observed['lag']
            v, t = coordinates(x[-128:], data['centre'], data['left'], data['right'], 1.)
            assert not channel_mask(v, t, plan['width']).any()
        else:
            assert observed['kind'] == 'unresolved_at_horizon' and length == plan['max_updates'] and ret is None
        v0, t0 = coordinates(x[:1], data['centre'], data['left'], data['right'], 1.)
        audited.append({**row, **observed, 'valid_updates': length, 'exit_v_over_U': float(v0[0] / plan['width']),
                        'exit_transverse': float(t0[0]),
                        'entry_v_over_U': None if ret is None else ret['entry_v'] / plan['width']})
        if row['condition'] == 'baseline':
            anchor = plan['anchors'][row['anchor']]
            events = json.loads((run / 'saved-events.json').read_text())['events']
            target = next(e for e in events if e['entry_pair'] == anchor['saved_next_entry_pair'])
            expected = 2 * target['entry_pair'] + target['entry_phase'] - (2 * anchor['pair'] + anchor['phase'])
            saved_baseline.append({'anchor': row['anchor'], 'saved_return_updates': expected,
                                   'measured_return_updates': None if ret is None else ret['entry_update'],
                                   'agrees': ret is not None and ret['entry_update'] == expected})
    assert all(r['agrees'] for r in saved_baseline)
    returned = [r for r in audited if r['kind'] == 'return']
    prediction = prediction_checks(data['seeds'], plan['rows'], result['outcomes'],
                                   data['centre'], data['left'], data['right'], plan['width'])
    matched = []
    for r in audited:
        if r['condition'] not in ('exit_transverse', 'random_transverse'):
            continue
        base = audited[16*r['anchor']]
        gap = abs(r['exit_v_over_U'] - base['exit_v_over_U'])
        assert gap < 1e-10
        matched.append({'row': r['id'], 'anchor': r['anchor'], 'condition': r['condition'],
                        'exit_v_over_U_gap': gap, 'baseline_outcome': base['kind'], 'outcome': r['kind'],
                        'return_time_difference': r.get('entry_update', 0) - base.get('entry_update', 0)
                        if r['kind'] == base['kind'] == 'return' else None,
                        'entry_v_over_U_difference': r['entry_v_over_U'] - base['entry_v_over_U']
                        if r['kind'] == base['kind'] == 'return' else None})
    summary = {'status': 'audited exploratory first returns', 'source_commit': metadata['source_commit'],
               'plan_sha256': sha(run / 'plan.json'), 'manifest_sha256': sha(run / 'manifest.json'),
               'raw_path': str(run), 'backup_path': str(backup), 'all_manifest_hashes_verified_on_both_disks': True,
               'files_verified_per_disk': len(manifest), 'bytes_per_disk': sum(r['bytes'] for r in manifest),
               'replay_gate': gate, 'counts': dict(Counter(r['kind'] for r in audited)),
               'actual_model_updates_across_rows': sum(result['valid_through']),
               'batch_iterations': len(all_states) - 1, 'wall_seconds': result['total_wall_seconds'],
               'return_updates_range': [min(r['entry_update'] for r in returned), max(r['entry_update'] for r in returned)] if returned else None,
               'entry_v_over_U_range': [min(r['entry_v_over_U'] for r in returned), max(r['entry_v_over_U'] for r in returned)] if returned else None,
               'phase_at_return_counts': dict(Counter(r['entry_update'] % 2 for r in returned)),
               'baseline_replay': saved_baseline, 'prediction': prediction, 'matched_critical_coordinate': matched,
               'limitations': plan['limitations'] + ['First-return stopping cannot determine later escape or sustained intermittency.']}
    write_json(out / 'rows.json', audited)
    write_json(out / 'audit.json', summary)
    render(out, audited, summary, plan['width'])
    write_json(out / 'figures.json', {p.name: sha(p) for p in out.glob('*.png')})
    for p in sorted(out.iterdir()):
        if p.is_file():
            backup_file(p, out, backup / 'audit')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('prediction', 'baseline_replay', 'replay_gate', 'matched_critical_coordinate')}, indent=2))


def render(out, rows, summary, width):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), layout='constrained')
    fig.suptitle('Where states return after leaving the slow channel', fontsize=20, fontweight='bold')
    ret = [r for r in rows if r['kind'] == 'return']
    markers = {'baseline': '*', 'critical': 'o', 'exit_transverse': '^', 'random_transverse': 's', 'combined': 'D'}
    labels = {'baseline': 'Observed exit, replayed', 'critical': 'Shift along slow direction',
              'exit_transverse': 'Shift across slow direction', 'random_transverse': 'Random transverse shift',
              'combined': 'Combined shift'}
    cmap = plt.get_cmap('viridis')
    lo, hi = summary['return_updates_range'] or [0, 1]
    norm = matplotlib.colors.Normalize(lo, max(hi, lo + 1))
    for kind, marker in markers.items():
        rr = [r for r in ret if r['condition'] == kind]
        for ax, field in [(axes[0], 'exit_v_over_U'), (axes[1], 'entry_update')]:
            ax.scatter([r[field] for r in rr], [r['entry_v_over_U'] for r in rr],
                       c=[r['entry_update'] for r in rr], cmap=cmap, norm=norm, marker=marker,
                       s=110 if kind == 'baseline' else 40, edgecolors='black' if kind == 'baseline' else 'none',
                       linewidths=.6, alpha=.9, label=labels[kind])
    for ax in axes:
        ax.axhline(0, lw=.8, color='.5')
        ax.axhspan(-1, 1, color='#fff2db', alpha=.45, zorder=-2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel('Re-entry position in the channel, v / U')
        ax.grid(alpha=.18)
    axes[0].set_title('A  Exit position → next re-entry', loc='left', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Starting exit position, v / U')
    axes[1].set_title('B  How long the excursion takes', loc='left', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Model updates from exit sample to re-entry')
    axes[1].legend(fontsize=9, loc='upper right')
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 shrink=.72, label='Updates to re-entry')
    counts = summary['counts']
    text = (f"{counts.get('return', 0)} returned · {counts.get('outside_recurrence_candidate', 0)} outside-channel recurrence candidates · "
            f"{counts.get('unresolved_at_horizon', 0)} unresolved · 128 trials in total\n"
            "Plots show returned trials. Eight observed exits with 16 conditions each; both cycle phases checked.\n"
            "Qwen2.5-0.5B · α = αc + 0.00001 · log₁₀T = −0.115 · Patched float64 CPU\n"
            "Channel entry requires |v| < U and transverse remainder < 0.05 for three same-phase readings.\n"
            "A finite first-return study; it does not establish permanent trapping, escape, or chaos.")
    fig.supxlabel(text, fontsize=10)
    fig.savefig(out / 'global-returns.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
    styles = {'return': ('#0072b2', 'o', 'Returned to channel'),
              'outside_recurrence_candidate': ('#d55e00', 's', 'Recurrence outside channel (candidate)'),
              'unresolved_at_horizon': ('#555555', '>', 'Unresolved at horizon')}
    for kind, (color, marker, label) in styles.items():
        rr = [r for r in rows if r['kind'] == kind]
        if rr:
            when = [r['entry_update'] if kind == 'return' else r['valid_updates'] for r in rr]
            ax.scatter(when, [r['id'] for r in rr], s=23, c=color, marker=marker,
                       label=f"{label}: {len(rr)}")
    for i in range(8):
        if i % 2 == 0:
            ax.axhspan(i*16-.5, (i+1)*16-.5, color='.94', zorder=-2)
    ax.set_yticks(np.arange(8)*16+7.5, [f'Exit {i+1}' for i in range(8)])
    ax.set_ylim(128, -1)
    ax.set_xlabel('Model updates since starting at the exit sample')
    ax.set_ylabel('Eight observed exits; sixteen nearby starting states per exit')
    ax.set_title('Every trial: return, outside recurrence, or unresolved', loc='left', fontsize=17, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=.2)
    fig.supxlabel('Return time marks the first of three same-phase channel readings. Recurrence time marks detection.\n'
                  'Full states checked in float64; finite recurrence does not establish an attracting orbit.', fontsize=10)
    fig.savefig(out / 'all-trial-outcomes.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    audit(p.parse_args())
