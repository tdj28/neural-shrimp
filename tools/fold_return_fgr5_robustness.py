"""Post-test robustness of the FGR5 score (disclosed; not part of the frozen criteria).

Resamples whole trajectories (the independent units) for the observed per-departure termination rate, and compares
the Kaplan-Meier survival in departures with the frozen prediction (1 - kappa)^n.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha
from fold_return_escape import TERMINAL, save_all


def per_trajectory(pairs):
    out = {}
    for p in pairs:
        d = out.setdefault(p['trajectory'], {'departures': 0, 'terminated': False})
        if p['kind'] in ('return',)+TERMINAL:
            d['departures'] += 1
        if p['kind'] in TERMINAL:
            d['terminated'] = True
    return [out[k] for k in sorted(out)]


def kaplan_meier(trajs):
    times = sorted({t['departures'] for t in trajs if t['terminated']})
    s, curve = 1.0, [(0, 1.0)]
    for n in times:
        at_risk = sum(t['departures'] >= n for t in trajs)
        events = sum(t['departures'] == n and t['terminated'] for t in trajs)
        s *= 1-events/at_risk; curve.append((n, s))
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', required=True); p.add_argument('--prediction', required=True)
    p.add_argument('--out', required=True); p.add_argument('--backup', required=True)
    a = p.parse_args()
    pairs = json.loads((Path(a.score)/'fresh-pairs.json').read_text())
    pred = json.loads(Path(a.prediction).read_text())
    trajs = per_trajectory(pairs)
    D = np.array([t['departures'] for t in trajs]); C = np.array([t['terminated'] for t in trajs], dtype=float)
    rng = np.random.default_rng(20261116); boot = []
    for _ in range(10000):
        i = rng.integers(0, len(trajs), len(trajs))
        boot.append(C[i].sum()/D[i].sum())
    km = kaplan_meier(trajs)
    res = {'status': 'post-test robustness, disclosed; not a frozen criterion',
           'bindings': {'score': sha(Path(a.score)/'score.json'), 'prediction': sha(Path(a.prediction))},
           'trajectories': len(trajs), 'departures': int(D.sum()), 'terminations': int(C.sum()),
           'departures_per_trajectory_range': [int(D.min()), int(D.max())],
           'kappa_observed': float(C.sum()/D.sum()),
           'kappa_trajectory_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
           'kappa_predicted': pred['kappa'], 'kappa_predicted_bootstrap_95': pred['kappa_bootstrap_95'],
           'kaplan_meier_departures': km,
           'predicted_survival_at_km_times': [(n, (1-pred['kappa'])**n) for n, _ in km]}
    save_all(a.out, a.backup, {'robustness.json': res})
    print(json.dumps({k: v for k, v in res.items() if 'survival' not in k and 'kaplan' not in k}, indent=1))


if __name__ == '__main__':
    main()
