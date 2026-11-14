"""Figure: the FGR3 return with the wraps undone (reads the 20261114-unwrap-01 audit; no computation)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AUDIT = Path('/data2/neural-shrimp-data/audits/fold-global-return/20261114-unwrap-01')
BLUE, ORANGE, GRAY = '#1f6785', '#c8641e', '#e9e9e9'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    out = Path(p.parse_args().out)
    audit = json.loads((AUDIT/'audit.json').read_text())
    rows = json.loads((AUDIT/'table-unwrapped.json').read_text())
    fresh = json.loads((AUDIT/'fresh-scored.json').read_text())
    runs = [(r['lo'], r['hi']) for r in audit['resolved_runs_at_least_0.02_wide']]
    ret = [r for r in rows if r['kind'] == 'return']
    q = np.array([r['q'] for r in ret]); tau = np.array([r['tau'] for r in ret])
    fr = [s for s in fresh if s['kind'] == 'return']
    fq = np.array([s['q'] for s in fr]); ftau = np.array([s['tau'] for s in fr])
    long = [s for s in fr if s['unwrapped']['kind'] == 'return' and any(a <= s['q'] <= b for a, b in runs)]
    lq = np.array([s['q'] for s in long]); err = np.array([abs(s['unwrapped']['tau']-s['tau']) for s in long])
    wrong = np.array([s['unwrapped']['updates'] != s['updates'] for s in long])
    stops = [r['q'] for r in rows if r['kind'] != 'return']
    caps = [s['q'] for s in fresh if s['kind'] == 'reference_p3']
    m_lo, m_hi = runs[0][1], runs[1][0]
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1, 1 if mobile else 2,
                                 figsize=(5.5, 9) if mobile else (11, 4.6), layout='constrained')
        a, b = axes.ravel()
        a.axvspan(m_lo, m_hi, color=GRAY, zorder=0)
        a.plot(fq, ftau, 'o', ms=4.5, mfc='none', mec=ORANGE, mew=.8, label=f'{len(fr)} returns on 16 fresh trajectories')
        a.plot(q, tau, '.', ms=3.5, color=BLUE, label=f'{len(ret)} table returns')
        top = max(tau.max(), ftau.max())+4
        a.plot(stops, np.full(len(stops), top), '|', ms=8, color='#a43b43', label='table stops (29 period 3, 1 other)')
        a.plot(caps, np.full(len(caps), top+3), 'x', ms=5, color='#333', label='last departure before a fresh capture')
        a.set(xlim=(2.99, 4.13), xlabel=r'Current departure $q=v/U$',
              ylabel=r'Next departure time $\tau$ (channel steps)',
              title='(a) With the wraps undone, the return is one curve outside M')
        a.text((m_lo+m_hi)/2, tau.min()-1, 'M', ha='center', va='top', fontsize=11, color='#555')
        a.legend(fontsize=7.5, loc='lower right', frameon=False)
        a.spines[['top', 'right']].set_visible(False)
        b.semilogy(lq[~wrong], np.maximum(err[~wrong], 1e-7), 'o', ms=3.5, mfc='none', mec=ORANGE, mew=.8,
                   label='return time correct')
        if wrong.any():
            b.semilogy(lq[wrong], np.maximum(err[wrong], 1e-7), 'x', ms=6, color='#333', label='return time wrong')
        b.axvspan(m_lo, m_hi, color=GRAY, zorder=0)
        b.set(xlim=(2.99, 4.13), xlabel=r'Current departure $q$', ylabel=r'$|\tau$ error$|$ (channel steps)',
              title='(b) Fresh returns from the two resolved runs')
        b.grid(alpha=.15); b.legend(fontsize=8, loc='upper center', frameon=False)
        b.spines[['top', 'right']].set_visible(False)
        b.text(.98, .04, f'{len(long)} returns; median error {np.median(err):.5f}', transform=b.transAxes,
               ha='right', fontsize=8)
        name = 'fig-fold-return-unwrapped'+('-mobile' if mobile else '')
        for ext in ('png', 'pdf'):
            meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
            fig.savefig(out/f'{name}.{ext}', dpi=240, metadata=meta)
        plt.close(fig)


if __name__ == '__main__':
    main()
