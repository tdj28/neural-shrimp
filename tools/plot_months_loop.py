"""Figure for the months pilot (reads static.json and, if given, analysis.json; no computation).

(a) the 12 month and the 12 number embeddings on their own first two principal axes; (b) first-harmonic share and the
end-to-start distance ratio across layers; (c) the loop: months visited and exponent against temperature; (d) the
loop's month states in the plane of the month circle; (e) readout covariance of the actual competitors against the
same probabilities on random tokens.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
FULL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
        'December']
BLUE, ORANGE, GRAY, RED = '#1f6785', '#c8641e', '#9a9a9a', '#a43b43'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static', required=True); p.add_argument('--analysis'); p.add_argument('--out', required=True)
    a = p.parse_args()
    st = json.loads(Path(a.static).read_text())
    an = json.loads(Path(a.analysis).read_text()) if a.analysis else None
    out = Path(a.out)
    for mobile in (False, True):
        n = 6 if an else 2
        if mobile:
            fig, ax = plt.subplots(n, 1, figsize=(5.6, 4.3*n), layout='constrained')
        else:
            fig, ax = plt.subplots(2 if an else 1, 3, figsize=(15, 9.2 if an else 4.6), layout='constrained')
        ax = np.ravel(ax)
        x = ax[0]
        m = np.array(st['sets']['months']['embeddings']['pc12_coordinates'])
        q = np.array(st['sets']['numbers']['embeddings']['pc12_coordinates'])
        for pts, col, lab, names, closed in ((m, BLUE, 'months', MONTHS, True),
                                             (q, ORANGE, 'numbers one to twelve', [str(k+1) for k in range(12)], False)):
            sc = pts/np.abs(pts).max()
            xs, ys = (np.r_[sc[:, 0], sc[:1, 0]], np.r_[sc[:, 1], sc[:1, 1]]) if closed else (sc[:, 0], sc[:, 1])
            x.plot(xs, ys, '-', color=col, lw=.8, alpha=.6)
            x.plot(sc[:, 0], sc[:, 1], 'o', color=col, ms=5, label=lab)
            for (u, v), t in zip(sc, names):
                x.annotate(t, (u, v), xytext=(3, 3), textcoords='offset points', fontsize=7, color=col)
        x.set(aspect='equal', xlabel='first principal axis (scaled)', ylabel='second principal axis (scaled)',
              title='(a) Input embeddings: months close, numbers do not')
        x.legend(fontsize=8, frameon=False, loc='lower right'); x.spines[['top', 'right']].set_visible(False)
        y = ax[1]
        for name, col in (('months', BLUE), ('weekdays', GRAY), ('numbers', ORANGE)):
            h = st['sets'][name]['hidden_states']
            y.plot([r['layer'] for r in h], [r['wrap_over_median_adjacent'] for r in h], 'o-', ms=3, color=col, label=name)
        y.axhline(1, color='#bbb', lw=.8, ls=':')
        y.set(xlabel='Layer (0 = input embeddings)', ylabel='last-to-first distance / median neighbor distance',
              title='(b) Closure through the network')
        y.legend(fontsize=8, frameon=False); y.spines[['top', 'right']].set_visible(False)
        if an:
            cells = an['cells']
            lt = np.array([c['log10_temperature'] for c in cells])
            z = ax[2]
            grid = np.array([[1.0 if FULL[k] in c['months_visited'] else 0.0 for k in range(12)] for c in cells])
            dlt = lt[1]-lt[0]
            z.imshow(grid.T, aspect='auto', origin='lower', cmap='Blues', vmin=0, vmax=1.3,
                     extent=(lt[0]-dlt/2, lt[-1]+dlt/2, -.5, 11.5), interpolation='nearest')
            z.set_yticks(range(12), MONTHS, fontsize=7)
            z.set(xlabel=r'$\log_{10} T$', ylabel='months written (dark), last 720 updates',
                  title='(c) Which months the loop writes')
            e = ax[3]
            lam = np.array([np.nan if c['lyapunov'] is None else c['lyapunov'] for c in cells])
            per = np.array([c['state_period'] for c in cells])
            e.plot(lt, lam, 'o-', ms=3, color=BLUE, lw=1)
            col = [c for c in cells if c['twin_collapsed']]
            if col:
                e.plot([c['log10_temperature'] for c in col], [-6.5]*len(col), 'v', ms=4, color=GRAY,
                       label='twin collapsed (superstable)')
            e.plot(lt[per == 0], lam[per == 0], 'o', ms=6, mfc='none', mec=RED, label='no period up to 120')
            e.axhline(0, color='#888', lw=.8, ls='--')
            for t, pq in zip(lt, per):
                if pq and (t == lt[0] or pq != per[list(lt).index(t)-1]):
                    e.annotate(str(pq), (t, 0.6), fontsize=7, ha='center', color='#333')
            e.set(xlabel=r'$\log_{10} T$', ylabel='largest Lyapunov exponent per update', ylim=(-7, 1),
                  title='(d) Exponent; numbers: state period')
            e.legend(fontsize=7, frameon=False, loc='lower right'); e.spines[['top', 'right']].set_visible(False)
            w = ax[4]
            pick = [c for c in cells if len(c['month_plane_projection']) > 10 and c['month_count'] >= 6]
            sel = [pick[k] for k in np.linspace(0, len(pick)-1, 4).astype(int)]
            cols = ['#08306b', '#2171b5', '#6baed6', '#9ecae1']
            for c, cl in zip(sel, cols):
                pr = np.array(c['month_plane_projection'])
                w.plot(pr[:, 0], pr[:, 1], '.-', ms=4, lw=.5, color=cl, label=fr"$\log_{{10}}T$ = {c['log10_temperature']:.2f}")
            w.set(aspect='equal', xlabel='first month axis', ylabel='second month axis',
                  title='(e) Loop states at month steps, month-circle plane')
            w.legend(fontsize=7, frameon=False); w.spines[['top', 'right']].set_visible(False)
            v = ax[5]
            ok = np.array([c['month_steps'] >= 50 for c in cells])           # enough updates that write a month
            ratio = np.array([c['cov_trace_ratio_median'] for c in cells])
            v.plot(lt[ok], ratio[ok], 'o-', ms=3, color=BLUE, label='updates whose leading token is a month')
            v.axhline(1, color='#888', lw=.8, ls='--')
            v.set(xlabel=r'$\log_{10} T$', ylabel='readout covariance: actual / random words',
                  ylim=(0, 1.1), title='(f) Close competitors damp the readout')
            v.legend(fontsize=7, frameon=False, loc='upper left')
            v.spines[['top', 'right']].set_visible(False)
        name = 'fig-months-loop'+('-mobile' if mobile else '')
        for ext in ('png', 'pdf'):
            meta = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Software': 'Matplotlib'}
            fig.savefig(out/f'{name}.{ext}', dpi=200, metadata=meta)
        plt.close(fig)


if __name__ == '__main__':
    main()
