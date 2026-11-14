"""Figure: the frozen FGR5 test (reads the score, robustness and prediction receipts; no computation)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, RED = '#1f6785', '#c8641e', '#a43b43'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', required=True); p.add_argument('--robustness', required=True)
    p.add_argument('--prediction', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    score = json.loads(Path(a.score).read_text()); rob = json.loads(Path(a.robustness).read_text())
    pred = json.loads(Path(a.prediction).read_text()); out = Path(a.out)
    km = np.array(rob['kaplan_meier_departures'])
    lo, mid, hi = pred['kappa_bootstrap_95']
    n = np.arange(0, km[:, 0].max()+5)
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(5.5, 9) if mobile else (11, 4.4), layout='constrained')
        x, y = axes.ravel()
        x.fill_between(n, (1-hi)**n, (1-lo)**n, color=BLUE, alpha=.15, lw=0, label='prediction, 95 % bootstrap band')
        x.plot(n, (1-pred['kappa'])**n, '-', color=BLUE, lw=1.5, label=fr"prediction $(1-\kappa)^n$, $\kappa$ = {pred['kappa']:.4f}")
        x.step(km[:, 0], km[:, 1], where='post', color=ORANGE, lw=2, label=f"{rob['trajectories']} fresh trajectories (Kaplan-Meier)")
        x.set(xlim=(0, n.max()), ylim=(0, 1.02), xlabel='Departures from the slow channel',
              ylabel='Share of trajectories not yet ended', title='(a) Survival, predicted before the runs')
        x.grid(alpha=.15); x.legend(fontsize=8, frameon=False, loc='upper right')
        x.spines[['top', 'right']].set_visible(False)
        rows = [('Predicted from the zoom\n(1024 one-cycle runs)', pred['kappa'], [lo, hi], BLUE),
                ('Observed, Poisson interval', score['kappa_observed'], score['kappa_observed_95'], ORANGE),
                ('Observed, resampling\nwhole trajectories', rob['kappa_observed'], rob['kappa_trajectory_bootstrap_95'], ORANGE)]
        for k, (label, v, ci, c) in enumerate(rows):
            y.errorbar(v, -k, xerr=[[v-ci[0]], [ci[1]-v]], fmt='o', color=c, ms=7, capsize=4)
        y.set_yticks([0, -1, -2], [r[0] for r in rows], fontsize=8.5)
        y.set(xlim=(0, .03), ylim=(-2.6, .6), xlabel='Share of departures that end the transient',
              title='(b) Escape rate per departure')
        y.grid(alpha=.15, axis='x'); y.spines[['top', 'right']].set_visible(False)
        y.text(.98, .03, f"{score['terminations_after_a_departure']} endings after {score['departures_with_observed_next_outcome']:,} departures",
               transform=y.transAxes, ha='right', fontsize=8)
        name = 'fig-fold-return-escape-test'+('-mobile' if mobile else '')
        for ext in ('png', 'pdf'):
            meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
            fig.savefig(out/f'{name}.{ext}', dpi=240, metadata=meta)
        plt.close(fig)


if __name__ == '__main__':
    main()
