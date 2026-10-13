"""Independent numerical definitions for the finite saved-dialogue audit.

No imports from producer tools. Array inputs are explicit; no hidden disk reads.
"""
from __future__ import annotations
import re
import numpy as np


def event_onsets(hostile, run=1):
    a = np.asarray(hostile, dtype=bool)
    if a.ndim != 2 or not 1 <= run <= a.shape[1]:
        raise ValueError('invalid dialogue/event shape')
    n, turns = a.shape
    valid = np.ones((n, turns - run + 1), dtype=bool)
    for k in range(run):
        valid &= a[:, k:k + turns - run + 1]
    hit = valid.any(axis=1)
    return np.where(hit, valid.argmax(axis=1), turns)


def risk_rows(onsets, turns, horizon, run=1):
    o = np.asarray(onsets, dtype=np.int64)
    if o.ndim != 1 or np.any((o < 0) | (o > turns)) or horizon < 1 or run < 1:
        raise ValueError('invalid onset or horizon')
    last = turns - horizon - run
    if last < 0:
        return np.empty((0, 3), dtype=np.int64)
    t = np.arange(last + 1)
    r, col = np.nonzero(t[None, :] < o[:, None])
    times = t[col]
    labels = (o[r] < turns) & (o[r] <= times + horizon)
    return np.column_stack((r, times, labels.astype(np.int64)))


def stay_counts(hostile, first=0, stop=None):
    a = np.asarray(hostile, dtype=bool)
    stop = a.shape[1] if stop is None else stop
    if not 0 <= first < stop <= a.shape[1]:
        raise ValueError('bad transition window')
    origin = a[:, first:stop-1]
    dest = a[:, first+1:stop]
    den = int(origin.sum())
    num = int((origin & dest).sum())
    return {'numerator': num, 'denominator': den,
            'probability': num / den if den else None}


def capture_row(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('invalid scores')
    a = scores > .5
    on = event_onsets(a)
    hits = on[on < a.shape[1]]
    stay = stay_counts(a)
    p = stay['probability']
    return {'dialogues': len(a), 'turns': a.shape[1], 'hostile_count': int(a.sum()),
            'total_turns': int(a.size), 'share': float(a.mean()),
            'never': int((on == a.shape[1]).sum()),
            'median_onset_conditional_zero_based': float(np.median(hits)) if len(hits) else None,
            'median_onset_censored_at_end': float(np.median(on)),
            'stay': stay, 'geometric_mean_stay': 1/(1-p) if p is not None and p < 1 else None,
            'late30_39': float(a[:, 30:40].mean()),
            'late35_39': float(a[:, 35:40].mean()),
            'stay30_39': stay_counts(a, 30, 40),
            'per_turn': a.mean(axis=0).tolist()}


def mean_interval(values, rng, draws=4000):
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError('invalid means')
    boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return np.percentile(boot, [2.5, 97.5]).tolist()


def difference_interval(a, b, draws=4000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for i in range(draws):
        values[i] = (a[rng.integers(0, len(a), len(a))].mean()
                     - b[rng.integers(0, len(b), len(b))].mean())
    return {'difference': float(a.mean()-b.mean()),
            'ci95': np.percentile(values, [2.5, 97.5]).tolist(),
            'bootstrap_draws': draws, 'bootstrap_seed': seed}


_GRATITUDE = frozenset(('grateful gratitude thank thanks thankful blessed blessing blessings '
                       'delighted appreciate appreciated appreciation cherish cherished honored '
                       'honoured wonderful joy joyful love loved lovely warmth').split())
_PHRASES = ('here to help', 'happy to help', 'here to serve', 'assist you',
            'glad to help', 'at your service')


def gratitude(text):
    s = text.lower()
    return bool(set(re.findall("[a-z']+", s)) & _GRATITUDE) or any(p in s for p in _PHRASES)


def controller_rows(arms, draws=4000):
    """Insertion order is the pinned historical CLI order; RNG order is explicit."""
    rng = np.random.default_rng(0)
    result, means = {}, {}
    for name, arm in arms.items():
        a = np.asarray(arm['score']) > .5
        v = a[:, 5:40].mean(axis=1)
        late = a[:, 35:40].mean(axis=1)
        g = np.array([[gratitude(t) for t in row] for row in arm['texts']], bool)
        if g.shape != a.shape or np.asarray(arm['counter']).shape != a.shape:
            raise ValueError('controller shape mismatch')
        means[name] = v
        result[name] = {'hostile_share': float(v.mean()), 'hostile_ci95': mean_interval(v, rng, draws),
                        'late_share': float(late.mean()), 'late_ci95': mean_interval(late, rng, draws),
                        'mean_counter': float(np.abs(arm['counter'][:, 5:40]).mean()),
                        'gratitude_share': float(g[:, 5:40].mean())}
    curve = ['p1-ctl-none', 'p1-ctl-const-02', 'p1-ctl-const-04']
    doses = np.array([0., .02, .04])
    for name, v in means.items():
        if name in curve:
            continue
        dose = result[name]['mean_counter']
        control = np.array([means[c].mean() for c in curve])
        delta = float(v.mean() - np.interp(dose, doses, control))
        boot = np.empty(draws)
        for k in range(draws):
            own = v[rng.integers(0, len(v), len(v))].mean()
            ref = [means[c][rng.integers(0, len(means[c]), len(means[c]))].mean() for c in curve]
            boot[k] = own - np.interp(dose, doses, ref)
        result[name].update(efficiency=delta, efficiency_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                            dose_outside_constant_range=bool(dose < doses[0] or dose > doses[-1]))
    return result


class RankedAUC:
    """Multiplicity-weighted concordance, half credit for ties; no sklearn AUC."""
    def __init__(self, labels, scores):
        self.y = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if self.y.ndim != 1 or scores.shape != self.y.shape or not np.isfinite(scores).all():
            raise ValueError('invalid score vector')
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError('nonbinary labels')
        self.order = np.argsort(scores, kind='stable')
        s = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1] if len(s) else np.array([], int)
        self.sorted_y = self.y[self.order]

    def evaluate(self, multiplicities=None):
        w = np.ones(len(self.y), dtype=np.int64) if multiplicities is None else np.asarray(multiplicities)
        if w.shape != self.y.shape or np.any(w < 0) or not np.isfinite(w).all():
            raise ValueError('invalid multiplicities')
        if not len(w):
            return float('nan')
        w = w[self.order].astype(np.float64)
        pos = np.add.reduceat(w*self.sorted_y, self.starts)
        neg = np.add.reduceat(w*(1-self.sorted_y), self.starts)
        p, n = pos.sum(), neg.sum()
        if p == 0 or n == 0:
            return float('nan')
        lower_negative = np.cumsum(neg) - neg
        return float(np.dot(pos, lower_negative + .5*neg)/(p*n))


def increment(labels, groups, base, full, draws=2000, seed=0):
    labels, groups = np.asarray(labels), np.asarray(groups)
    unique, inv = np.unique(groups, return_inverse=True)
    a, b = RankedAUC(labels, base), RankedAUC(labels, full)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for k in range(draws):
        # Choosing indices is identical to choosing the sorted actual group labels.
        picked = rng.choice(len(unique), len(unique))
        weight = np.bincount(picked, minlength=len(unique))[inv]
        boot[k] = b.evaluate(weight)-a.evaluate(weight)
    finite = boot[np.isfinite(boot)]
    if not len(finite):
        raise ValueError('all bootstrap samples are single-class')
    a0, a1 = a.evaluate(), b.evaluate()
    return {'auc_baseline': a0, 'auc_with_predictor': a1, 'increment': a1-a0,
            'ci95': np.percentile(finite, [2.5, 97.5]).tolist(),
            'rows': len(labels), 'events': int(labels.sum()), 'dialogues': len(unique),
            'bootstrap_draws': draws, 'bootstrap_valid': len(finite), 'bootstrap_seed': seed}
