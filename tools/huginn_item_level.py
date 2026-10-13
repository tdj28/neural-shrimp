"""Exploratory (development data only): does each ARC question's fixed-point spectrum predict how its own readout
behaves under relaxation? H2 readouts (train 0-99, steps 24-32, lam 1.0/1.10/1.15/1.25) vs spectra of train 0-29."""
import json, sys, glob
import numpy as np
sys.path.insert(0, 'tools')
import huginn_arc_analysis as ha
from scipy.stats import spearmanr
rows = [json.loads(l) for l in open('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/huginn/arc-readout-dev/rows.jsonl')]
spec = {}
for q in range(30):
    f = f'/data2/neural-shrimp-data/explore/huginn-relax/arc-dev-spectra/q{q:04d}/fixed-spectrum.json'
    if q < 4:
        f = f'/data2/neural-shrimp-data/explore/huginn-relax/arc-dev-spectrum-{q}/fixed-spectrum.json'
    try:
        s = json.load(open(f))
    except FileNotFoundError:
        continue
    if s.get('fixed_point_residual_F', 0.0) > 1e-4:
        print('skip non-fixed', q); continue
    spec[q] = ha.eigenvalues(s)
print('spectra', len(spec))
table = []
for r in rows:
    q = r['index']
    if q not in spec: continue
    steps = sorted(r['by_steps'], key=int)
    gl = np.array([r['by_steps'][k]['loglik'][r['gold']] for k in steps])
    ll = np.array([r['by_steps'][k]['loglik'] for k in steps])
    others = np.delete(ll, r['gold'], axis=1).max(1)
    margin = gl - others
    osc = np.mean(np.abs(np.diff(margin)))
    choice = ll.argmax(1)
    flips = int((np.diff(choice) != 0).sum())
    res32 = r['native_residual'][-1]
    lam = r['lam']
    rho = ha.rho(spec[q], lam)
    lc = ha.lam_critical(spec[q])[0]
    table.append((q, lam, rho, lc, osc, flips, res32))
t = np.array(table, float)
for lam in (1.0, 1.10, 1.15, 1.25):
    m = t[:, 1] == lam
    unstable = t[m, 2] >= 1
    print(f'lam {lam}: n {m.sum()}, predicted unstable {unstable.sum()}; '
          f'residual32 median stable {np.median(t[m][~unstable, 6]) if (~unstable).any() else float("nan"):.3g} '
          f'unstable {np.median(t[m][unstable, 6]) if unstable.any() else float("nan"):.3g}; '
          f'osc median stable {np.median(t[m][~unstable, 4]) if (~unstable).any() else float("nan"):.3g} '
          f'unstable {np.median(t[m][unstable, 4]) if unstable.any() else float("nan"):.3g}; '
          f'spearman(rho, residual32) {spearmanr(t[m, 2], t[m, 6]).statistic:.2f}, (rho, osc) {spearmanr(t[m, 2], t[m, 4]).statistic:.2f}; '
          f'flips any: stable {np.mean(t[m][~unstable, 5] > 0) if (~unstable).any() else float("nan"):.2f} unstable {np.mean(t[m][unstable, 5] > 0) if unstable.any() else float("nan"):.2f}')
# pooled over lam: AUC of (rho >= 1) and of rho for 'any flip' and for large oscillation
import p1_detection as pd
print('pooled AUC rho -> any flip', round(pd.auc((t[:, 5] > 0).astype(int), t[:, 2]), 3),
      'rho -> osc > 0.3', round(pd.auc((t[:, 4] > 0.3).astype(int), t[:, 2]), 3),
      'spearman(rho, log residual32)', round(spearmanr(t[:, 2], np.log(t[:, 6])).statistic, 3))
