"""Render publication views of the completed FGR2 audits; never run a model.

The input receipts, audit files and per-row records are hash-checked. Outputs
are derived copies; the original figures and immutable histories stay intact.
Use --verify to regenerate in a temporary directory and compare every byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inputs():
    receipt = ROOT / 'docs/fold-return-mechanism.json'
    r = read(receipt)
    paths = {}
    sources = {str(receipt): sha(receipt)}
    for key, binding in r['bindings'].items():
        p = Path(binding['audit_path'])
        assert sha(p / 'audit.json') == binding['audit_sha256'], key
        files = read(p / 'files.json')
        for name in ('rows.json', 'pairs.json', 'events.json', 'natural-exits.npz'):
            if name in files:
                assert sha(p / name) == files[name]['sha256'], (key, name)
                sources[str(p / name)] = sha(p / name)
        sources[str(p / 'audit.json')] = sha(p / 'audit.json')
        paths[key] = p
    rows = read(paths['line'] / 'rows.json')
    pairs = [p for p in read(paths['complete_cycle'] / 'pairs.json') if p['split'] == 'test']
    routes = [p for p in read(paths['scalar_excursion'] / 'pairs.json') if p['split'] == 'test']
    events = read(paths['line'] / 'events.json')
    with np.load(paths['line'] / 'natural-exits.npz') as z:
        channel = z['channel']
    assert len(rows) == 64 and len(pairs) == 28 and len(routes) == 31
    assert sum(p['original_route'] == 'return' for p in routes) == len(pairs)
    assert all(p['same_route'] for p in routes)
    assert sum(p['kind'] == 'reference_p3' for p in rows) == r['line_scan']['counts']['reference_p3']
    assert max(abs(p['next_v_error']) for p in pairs) == r['complete_return_cycle']['max_next_departure_v_over_U_error']
    assert all(p['cycle_time_error'] == 0 for p in pairs)
    return r, rows, pairs, events, channel, sources


def save(fig, out, stem):
    for ext in ('png', 'pdf'):
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
        fig.savefig(out / f'{stem}.{ext}', dpi=240, metadata=meta)
    plt.close(fig)


def figures(out):
    r, rows, pairs, events, channel, sources = inputs()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    groups = {}
    for e in events:
        groups.setdefault(e['row'], []).append(e)
    background = [(a['exit_v_over_U'], b['exit_v_over_U'])
                  for group in groups.values() for a, b in zip(group, group[1:])
                  if b['visit'] == a['visit'] + 1]
    points = np.array(background)
    display = np.zeros(channel.shape)
    display[channel | np.c_[channel[:, 1:], np.zeros(len(rows), dtype=bool)]] = 1
    for i, row in enumerate(rows):
        if row['kind'] == 'reference_p3':
            display[i, row['first_128_match_start']:row['valid_updates'] + 1] = 2
        display[i, row['valid_updates'] + 1:] = np.nan
    cmap = ListedColormap(['#e9edf1', '#db9e3b', '#42728f'])
    cmap.set_bad('white')
    for mobile in (False, True):
        suffix = '-mobile' if mobile else ''
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(5.4, 9.0) if mobile else (11, 4.8),
                                 layout='constrained')
        a, b = axes.ravel()
        a.scatter(points[:, 0], points[:, 1], s=4, c='#bbb', alpha=.55,
                  label='Other measured cycles', rasterized=True)
        q0 = [p['v0'] for p in pairs]
        a.scatter(q0, [p['observed_next_v'] for p in pairs], s=25, c='#086a92', label='Original state')
        a.scatter(q0, [p['predicted_next_v'] for p in pairs], s=58, facecolors='none',
                  edgecolors='#bd643a', label='Reconstructed state')
        a.set(xlabel=r'This departure, $v/U$', ylabel=r'Next departure, $v/U$',
              title='(a) Both states evolved by the full network')
        a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
        b.scatter(q0, [abs(p['next_v_error']) for p in pairs], s=25, c='#086a92')
        b.set(xlabel=r'This departure, $v/U$', ylabel=r'Next-departure difference (in units of $U$)',
              yscale='log', title='(b) Differences for held-out pairs')
        for ax in (a, b):
            ax.grid(alpha=.16)
        save(fig, out, 'fig-fold-return-map' + suffix)

        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(5.4, 9.2) if mobile else (11, 4.8),
                                 gridspec_kw={} if mobile else {'width_ratios': [1.8, 1]},
                                 layout='constrained')
        a, b = axes.ravel()
        a.imshow(display, origin='lower', aspect='auto', interpolation='nearest',
                 extent=[0, channel.shape[1] - 1, -.5/63, 1+.5/63], cmap=cmap, vmin=0, vmax=2,
                 rasterized=True)
        a.set(xlabel='Individual model updates', ylabel=r'Starting-state interpolation $s$',
              title='(a) Repeated passages can end in capture')
        a.legend(handles=[Patch(color='#db9e3b', label='Slow channel'),
                          Patch(color='#42728f', label='Period-3 match'),
                          Patch(color='#e9edf1', label='Elsewhere')], fontsize=8,
                 loc='upper center', bbox_to_anchor=(.5, -.17), ncol=3)
        for kind, color, marker, label in [('reference_p3', '#42728f', 'o', 'Period-3 capture'),
                                         ('unresolved_at_horizon', '#ad563c', '^', 'Unresolved at 4,096')]:
            selected = [row for row in rows if row['kind'] == kind]
            b.scatter([row['s'] for row in selected], [row['visit_count'] for row in selected],
                      c=color, marker=marker, s=26, label=label)
        b.set(xlabel=r'Starting-state interpolation $s$', ylabel='Persistent channel visits',
              title='(b) Visits before stopping')
        b.grid(alpha=.16)
        b.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.17))
        save(fig, out, 'fig-fold-return-fates' + suffix)
    claims = {
        'line_starts': len(rows), 'horizon': channel.shape[1] - 1,
        'captured': sum(row['kind'] == 'reference_p3' for row in rows),
        'unresolved': sum(row['kind'] == 'unresolved_at_horizon' for row in rows),
        'returned_at_least_once': sum(row['visit_count'] > 0 for row in rows),
        'returned_then_captured': sum(row['kind'] == 'reference_p3' and row['visit_count'] > 0 for row in rows),
        'heldout_routes': r['scalar_reconstruction']['test_pairs'],
        'heldout_returning_pairs': len(pairs),
        'heldout_P3_pairs': r['scalar_reconstruction']['original_P3_test_pairs'],
        'max_entry_error_U': r['scalar_reconstruction']['max_return_v_over_U_error'],
        'max_next_departure_error_U': max(abs(p['next_v_error']) for p in pairs),
        'max_cycle_time_error': max(abs(p['cycle_time_error']) for p in pairs),
        'cycle_duration_range': [min(p['cycle_updates'] for p in pairs), max(p['cycle_updates'] for p in pairs)],
        'max_initial_relative_reconstruction_error': r['scalar_reconstruction']['max_test_initial_relative_state_error'],
        'constant_entry_rmse_U': r['simple_prediction_comparisons']['models']['constant']['equal_row_RMSE'][0],
        'interpolation_entry_rmse_U': r['simple_prediction_comparisons']['models']['v_linear_interpolation']['equal_row_RMSE'][0],
    }
    for stem in ('fig-fold-return-map', 'fig-fold-return-fates'):
        map_plot = stem.endswith('map')
        per = {
            'figure_id': stem,
            'description': ('Original and scalar-reconstructed states, each advanced through the unchanged full network.'
                            if map_plot else 'All 64 constructed starts; full histories end at verified P3 matching or the fixed horizon.'),
            'alt_text': ('Reconstructed natural exit states reproduce the next departure of all 28 returning held-out pairs; both trajectories use the full network.'
                         if map_plot else 'Of 64 constructed initial states, 27 approach period 3, including 26 after returning; 37 remain unresolved at 4096 updates.'),
            'sources': sources, 'generator': {'path': 'tools/publish_fold_return_figures.py', 'sha256': sha(__file__)},
            'matplotlib': matplotlib.__version__, 'claims': claims,
            'selection': 'All test returning pairs; gray background is all adjacent eligible exit pairs.' if map_plot else 'All initial states, in original interpolation order; stopped histories are blank.',
            'uncertainty': 'Finite deterministic exploratory sample, no population confidence interval.',
            'limits': r['limits'] + ['Full network evaluates the return; no standalone scalar simulator.', 'No persistent-chaos claim from returns.'],
            'outputs': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.glob(stem + '*'))},
            'accessibility': 'Captions describe all categories; open and filled markers distinguish reconstruction; triangle and circle distinguish final outcomes.'
        }
        (out / (stem + '.receipt.json')).write_text(json.dumps(per, indent=2) + '\n')
    return claims


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            claims = figures(Path(tmp))
            for item in Path(tmp).iterdir():
                assert item.read_bytes() == (args.out / item.name).read_bytes(), item.name
        print('All publication figures and receipts regenerate byte-for-byte.')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        claims = figures(args.out)
    print(json.dumps(claims, indent=2))


if __name__ == '__main__':
    main()
