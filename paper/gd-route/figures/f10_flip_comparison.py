"""Plot all six saved doubled-cycle proposals; no model or continuation calls."""
import json
import math
import common as c

CASES = [
    (13, 's2-period18-flip/attempt-002-target', 'run',
     'bd34d93e8430ed17b0705725ef27c384c7e8ae4853e20f8363042cfcc493123c',
     'unstable', 'attracting', 1e6, 'Subcritical', c.FLIP),
    (51, 's2-period18-other-flips/attempt-002', 'run/ray-51',
     '9a199a50d93954a836ac5b642eb4f035afefcf3628c59a8429702e9e089306db',
     'attracting', 'unstable', 1e4, 'Supercritical', c.FOLD),
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(case):
    ray, relative, sub, pin, stability, parent_stability, _, _, _ = case
    root = c.EXPLORE / relative
    manifest = root / 'manifest.json'
    require(c.sha256(manifest) == pin, 'external numerical manifest pin')
    files = json.loads(manifest.read_text())['files']
    inputs = [manifest]
    values = {}
    for name in ('flip.json', 'doubled.json'):
        path = root / sub / name
        entry = files[str(path.relative_to(root))]
        require(c.sha256(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], 'saved file binding')
        values[name] = json.loads(path.read_text())
        inputs.append(path)
    flip, rows = values['flip.json'], values['doubled.json']
    require(flip['qualified'] and len(rows) == 3, 'fixed qualified flip and all three proposals')
    points = []
    for index, row in enumerate(rows):
        require(row['qualified'] and all(row['gates'].values()), 'qualified numerical proposal')
        for key in ('qualification', 'opposite_qualification'):
            require(row[key]['qualified'] and row[key]['stability'] == stability, 'two-phase numerical qualification')
            require([d['divisor'] for d in row[key]['divisors']] == [1, 2, 3, 4, 6, 9, 12, 18], 'primitive36 divisor set')
        require(row['parent']['qualification']['stability'] == parent_stability, 'parent classification')
        require(row['delta_s'] == row['s'] - flip['s'], 'saved signed detuning')
        # The solver's controlled projection, not a norm. Read already saved
        # native states only; no forward evaluation supplies the middle phase.
        amplitude = math.fsum(p * (x - y) / 2 for p, x, y in
                              zip(flip['left'], row['state'], row['data']['orbit'][18]))
        require(math.isfinite(amplitude) and amplitude > 0 and
                abs(amplitude - row['amplitude_requested']) < 1e-8, 'retained amplitude constraint')
        points.append({'slot': index, 's': row['s'], 'delta_s': row['delta_s'],
                       'projected_amplitude': amplitude, 'requested_amplitude': row['amplitude_requested'],
                       'stability': stability, 'parent_stability': parent_stability,
                       'spectral_radius': row['qualification']['spectral_radius'],
                       'opposite_spectral_radius': row['opposite_qualification']['spectral_radius']})
    return {'ray': ray, 'numerical_flip_s': flip['s'], 'points': points}, inputs


def main():
    c.style()
    fig, axes = c.plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True, layout='constrained')
    inputs, derived = [], []
    for ax, case, letter in zip(axes, CASES, ('A', 'B')):
        data, paths = load(case)
        inputs.extend(paths); derived.append(data)
        ray, _, _, _, stability, _, factor, criticality, color = case
        points = data['points']
        ax.scatter([p['delta_s'] * factor for p in points],
                   [p['projected_amplitude'] * 1e3 for p in points],
                   s=46, marker='o', facecolors='none' if stability == 'unstable' else color,
                   edgecolors=color, linewidths=1.3, label=f'Period 36: {stability}', zorder=3)
        ax.scatter([0], [0], s=28, marker='D', color=c.INK, label='Local flip', zorder=3)
        ax.set_title(f'{letter}  {criticality} (ray {ray})')
        power = 6 if factor == 1e6 else 4
        ax.set_xlabel(r'Signed detuning $s-s_{*,\mathrm{num}}$ ' + rf'$(\times 10^{{-{power}}})$')
        ax.set_ylim(-.08, 1.75)
        ax.set_xlim((-3.9, .3) if ray == 13 else (-.3, 4.9))
        ax.grid(axis='y', color=c.NEUTRAL, linewidth=.6)
        ax.legend(loc='upper right' if ray == 13 else 'upper left', frameon=False)
    axes[0].set_ylabel(r'Projected amplitude $a$ $(\times 10^{-3})$')
    c.save(fig, 'f10_flip_comparison', inputs, __file__,
           'Exploratory numerical proposals; local flip types are separately interval-certified',
           'Two panels show all three numerical primitive36 cycles on each selected ray. Ray13 has unstable '
           'cycles at negative detuning; ray51 has attracting cycles at positive detuning. Vertical amplitude '
           'is a left-eigenvector projection, not a norm. Axes have different detuning scales, and no lines '
           'interpolate the six points. A diamond marks each numerical flip origin.',
           {'cases': derived, 'amplitude_definition': 'p_star^T(theta-F^18(theta))/2, p_star^T q_star=1',
            'x_scale_multipliers': [1e6, 1e4], 'y_scale_multiplier': 1e3, 'all_six_saved_points': True},
           'No new model evaluation, decomposition or orbit correction. Points are numerical, not six '
           'individual interval certificates; separate local proofs establish opposite criticality. '
           'No interpolated branch or validated connection to the center is claimed.')


if __name__ == '__main__':
    main()
