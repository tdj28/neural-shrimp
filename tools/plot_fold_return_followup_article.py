"""Three-panel figure for the Chaos article: unwrapped return, zoom ladder, frozen survival test (reads receipts)."""
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
    for k in ('unwrap', 'zoom', 'score', 'robustness', 'prediction', 'out'):
        p.add_argument('--'+k, required=True)
    a = p.parse_args()
    uw = Path(a.unwrap)
    audit = json.loads((uw/'audit.json').read_text())
    rows = json.loads((uw/'table-unwrapped.json').read_text())
    fresh = json.loads((uw/'fresh-scored.json').read_text())
    zoom = json.loads((Path(a.zoom)/'audit.json').read_text())['uncertainty_time']
    rob = json.loads(Path(a.robustness).read_text()); pred = json.loads(Path(a.prediction).read_text())
    runs = [(r['lo'], r['hi']) for r in audit['resolved_runs_at_least_0.02_wide']]
    m_lo, m_hi = runs[0][1], runs[1][0]
    fig, (x, y, z) = plt.subplots(1, 3, figsize=(7.1, 2.55), layout='constrained')
    plt.rcParams.update({'font.size': 7})
    x.axvspan(m_lo, m_hi, color=GRAY, zorder=0)
    fr = [s for s in fresh if s['kind'] == 'return']
    x.plot([s['q'] for s in fr], [s['tau'] for s in fr], 'o', ms=2.2, mfc='none', mec=ORANGE, mew=.5)
    ret = [r for r in rows if r['kind'] == 'return']
    x.plot([r['q'] for r in ret], [r['tau'] for r in ret], '.', ms=1.6, color=BLUE)
    x.set_xlabel(r'Departure $q$', fontsize=7); x.set_ylabel(r'Next departure time $\tau$', fontsize=7)
    x.set_title('(a) Return, wraps undone', fontsize=7.5)
    x.text((m_lo+m_hi)/2, 112, 'M', ha='center', fontsize=7, color='#555')
    lv = [l for l in zoom['levels'] if l['delta'] > 0]; ctrl = [l for l in zoom['levels'] if l['delta'] == 0]
    d = np.array([l['delta'] for l in lv]); f = np.array([l['fraction'] for l in lv]); n = np.array([l['pairs'] for l in lv])
    y.errorbar(d, f, yerr=1.96*np.sqrt(f*(1-f)/n), fmt='o', ms=3, color=BLUE, capsize=2, lw=.8)
    fit = [l for l in lv if l['delta'] <= 1e-4 and l['uncertain'] >= 10]
    s, c = np.polyfit(np.log10([l['delta'] for l in fit]), np.log10([l['fraction'] for l in fit]), 1)
    xx = np.logspace(-10.3, -4, 20); y.plot(xx, 10**c*xx**s, '-', color=BLUE, lw=.8)
    y.plot(xx, f[d == 1e-4][0]*xx/1e-4, '--', color='#888', lw=.7)
    if ctrl:
        y.axhline(ctrl[0]['fraction'], color=RED, ls=':', lw=1)
    y.set(xscale='log', yscale='log', ylim=(1e-3, 1.3))
    y.set_xlabel(r'Neighbor distance $\delta$', fontsize=7); y.set_ylabel('Share with different outcome', fontsize=7)
    y.set_title(fr'(b) Inside M: $\delta^{{{s:.2f}}}$', fontsize=7.5)
    km = np.array(rob['kaplan_meier_departures']); lo, _, hi = pred['kappa_bootstrap_95']
    k = np.arange(0, km[:, 0].max()+5)
    z.fill_between(k, (1-hi)**k, (1-lo)**k, color=BLUE, alpha=.15, lw=0)
    z.plot(k, (1-pred['kappa'])**k, '-', color=BLUE, lw=1)
    z.step(km[:, 0], km[:, 1], where='post', color=ORANGE, lw=1.3)
    z.set(xlim=(0, k.max()), ylim=(0, 1.02))
    z.set_xlabel('Departures', fontsize=7); z.set_ylabel('Not yet ended', fontsize=7)
    z.set_title('(c) Frozen survival test', fontsize=7.5)
    for ax in (x, y, z):
        ax.tick_params(labelsize=6.5); ax.spines[['top', 'right']].set_visible(False)
    out = Path(a.out)
    for ext in ('pdf', 'png'):
        meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
        fig.savefig(out/f'fig-fold-return-followup.{ext}', dpi=300, metadata=meta)


if __name__ == '__main__':
    main()
