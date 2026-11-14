"""Figure: FGR4/FGR4b zoom of the unresolved interval and the FGR5 escape-rate prediction (reads audits only)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, RED, GRAY = '#1f6785', '#c8641e', '#a43b43', '#e9e9e9'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True); p.add_argument('--prediction', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    audit = json.loads((Path(args.audit)/'audit.json').read_text())
    pred = json.loads(Path(args.prediction).read_text())
    out = Path(args.out)
    u = audit['uncertainty_time']
    lv = [l for l in u['levels'] if l['delta'] > 0]
    ctrl = [l for l in u['levels'] if l['delta'] == 0]
    d = np.array([l['delta'] for l in lv]); f = np.array([l['fraction'] for l in lv])
    n = np.array([l['pairs'] for l in lv]); k = np.array([l['uncertain'] for l in lv])
    se = np.sqrt(np.maximum(f*(1-f), 1e-12)/n)
    fit = [l for l in lv if l['delta'] <= 1e-4 and l['uncertain'] >= 10]
    x = np.log10([l['delta'] for l in fit]); y = np.log10([l['fraction'] for l in fit])
    slope, icept = np.polyfit(x, y, 1)
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(5.5, 9) if mobile else (11, 4.6), layout='constrained')
        a, b = axes.ravel()
        a.errorbar(d, f, yerr=1.96*se, fmt='o', color=BLUE, ms=6, capsize=3, label='neighbors with a different return time')
        xx = np.logspace(np.log10(d.min())-.3, -4, 50)
        a.plot(xx, 10**icept*xx**slope, '-', color=BLUE, lw=1.2, label=fr'fit: $\delta^{{{slope:.2f}}}$')
        a.plot(xx, f[d == 1e-4][0]*(xx/1e-4), '--', color='#888', lw=1, label=r'a smooth map would fall as $\delta^{1}$')
        if ctrl:
            c = ctrl[0]
            a.axhline(c['fraction'], color=RED, ls=':', lw=1.4,
                      label=f"exact rerun in another batch slot: {c['uncertain']} of {c['pairs']} differ")
        a.set(xscale='log', yscale='log', ylim=(1e-3, 1.3), xlabel=r'Distance $\delta$ between neighboring starts (units of $q$)',
              ylabel='Share of pairs', title='(a) Inside M, structure at every sampled scale')
        a.grid(alpha=.15, which='both'); a.legend(fontsize=7.5, loc='lower right', frameon=False)
        a.spines[['top', 'right']].set_visible(False)
        edges = np.linspace(*pred['domain'], pred['bins']+1); mid = (edges[1:]+edges[:-1])/2
        w = edges[1]-edges[0]
        b.bar(mid, pred['termination_by_bin'], width=w*.9, color=RED, alpha=.85)
        b.axvspan(3.0503125, 3.56, color=GRAY, zorder=0)
        b.set(xlim=pred['domain'], xlabel=r'Current departure $q$', ylabel='Share of uniform starts that end the transient',
              title='(b) Where departures end, from 1024 uniform starts')
        b.text(.97, .62, fr"escape estimate $\kappa$ = {pred['kappa']:.4f} per departure", transform=b.transAxes,
               ha='right', va='top', fontsize=9)
        b.text(3.305, -.006, 'M', ha='center', va='top', fontsize=10, color='#555')
        b.grid(alpha=.15, axis='y'); b.spines[['top', 'right']].set_visible(False)
        name = 'fig-fold-return-zoom'+('-mobile' if mobile else '')
        for ext in ('png', 'pdf'):
            meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
            fig.savefig(out/f'{name}.{ext}', dpi=240, metadata=meta)
        plt.close(fig)


if __name__ == '__main__':
    main()
