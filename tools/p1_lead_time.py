"""Lead time of dynamical predictors over a text classifier (misalignment proxy P1, docs/misalignment-dynamics-plan.md).

For sampled self-dialogue arms (explore_llm_msm sample, scored by tools/hostility_score.py), the event is the first
hostile turn of a replica (classifier label above the threshold). At every turn t before the onset (all turns for a
replica that never turns hostile), the label for horizon h is whether the onset falls in turns t+1 .. t+h. Predictors
available at turn t:

- classifier: the classifier's continuous score at turn t (the reactive baseline: its own rise before the threshold);
- forecast: the probability of reaching the hostile states within h steps from turn t's Markov state, with the
  hostile states made absorbing in a transition matrix estimated on training dialogues (a finite-horizon committor);
- probe: a logistic probe of hostility on the turn's PCA-reduced activation (fit on training turns);
- variance and autocorrelation: the variance and lag-1 autocorrelation of the probe's logit over the last w turns
  (early-warning indicators of critical slowing down).

Baselines that see the same turns (added before any ladder data, following the literature audit in
docs/related-work-draft.md): the classifier's change since the previous turn, its maximum so far, the turn index, and
the projection of the turn's activation on the fixed steering axis (a persona-axis score). Combined models: a logistic
regression on the baseline features, and one on baseline plus dynamical features (forecast, probe, variance,
autocorrelation), each fit on training turns per horizon; the incremental value of dynamics is the difference of their
test AUCs.

Each predictor is scored by the area under the ROC curve on test dialogues, with a dialogue-level bootstrap. The
Markov model (PCA, k-means states, transition matrix, hostile states), the probe and the combined models are fit on
the training arms only.

Usage:
  python tools/p1_lead_time.py --train ARM [ARM ...] --test ARM [ARM ...] --out llm-msm/p1-lead-time
  (ARM: explore-relative folders containing sample.json, activations.npy and hostility.npz)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LABEL = 0                                   # index of "toxic" in hostility_score.LABELS


def onsets(score, threshold):
    """First turn with score above the threshold for each replica (len = turns when it never happens)."""
    hostile = score > threshold
    return np.where(hostile.any(1), hostile.argmax(1), score.shape[1])


def horizon_labels(onset, turns, h):
    """(replica, turn, label) for every turn before the onset: label 1 when the onset is in turns t+1 .. t+h."""
    rows = []
    for r, o in enumerate(onset):
        for t in range(min(o, turns)):
            if t + 1 > turns - 1:
                continue
            rows.append((r, t, int(t < o <= t + h)))
    return np.array(rows, dtype=int).reshape(-1, 3)


def transition_matrix(labels, k, lag=1):
    C = np.zeros((k, k))
    for row in labels:
        for a, b in zip(row[:-lag], row[lag:]):
            C[a, b] += 1
    C += 1e-6                                              # unvisited rows: near-uniform, avoids division by zero
    return C / C.sum(1, keepdims=True)


def hit_probability(T, hostile, h):
    """P(reach a hostile state within h steps | start state), hostile states absorbing; shape (h + 1, k)."""
    k = T.shape[0]
    A = T.copy()
    A[hostile] = 0.0
    A[hostile, hostile] = 1.0
    p = hostile.astype(float)
    out = [p.copy()]
    for _ in range(h):
        p = A @ p
        out.append(p.copy())
    return np.array(out)


def rolling_stats(x, w):
    """Variance and lag-1 autocorrelation of x[r, t-w+1 .. t] for every (r, t >= w - 1); nan before."""
    R, N = x.shape
    var = np.full((R, N), np.nan)
    ac = np.full((R, N), np.nan)
    for t in range(w - 1, N):
        seg = x[:, t - w + 1: t + 1]
        var[:, t] = seg.var(1)
        a, b = seg[:, :-1] - seg[:, :-1].mean(1, keepdims=True), seg[:, 1:] - seg[:, 1:].mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        ac[:, t] = np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-300), np.nan)
    return var, ac


def auc(y, s):
    """Area under the ROC curve (Mann-Whitney), nan when one class is missing; nan scores are dropped."""
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):                                   # average ranks over ties
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Model:
    """PCA + k-means Markov states, transition matrix, hostile states and a logistic probe, fit on training arms."""

    def __init__(self, acts, score, threshold, dims=20, states=40, hostile_fraction=0.5, seed=0):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        R, N, d = acts.shape
        flat = acts.reshape(-1, d)
        self.pca = PCA(n_components=dims, random_state=seed).fit(flat)
        Y = self.pca.transform(flat)
        self.km = KMeans(n_clusters=states, n_init=10, random_state=seed).fit(Y)
        lab = self.km.predict(Y).reshape(R, N)
        self.T = transition_matrix(lab, states)
        hostile_turn = (score > threshold).reshape(-1)
        frac = np.array([hostile_turn[lab.reshape(-1) == s].mean() if (lab == s).any() else 0.0 for s in range(states)])
        self.hostile = frac >= hostile_fraction
        self.hostile_fraction_by_state = frac
        self.probe = LogisticRegression(max_iter=2000, C=1.0).fit(Y, hostile_turn.astype(int)) \
            if 0 < hostile_turn.sum() < len(hostile_turn) else None

    def features(self, acts):
        R, N, d = acts.shape
        Y = self.pca.transform(acts.reshape(-1, d))
        states = self.km.predict(Y).reshape(R, N)
        logit = self.probe.decision_function(Y).reshape(R, N) if self.probe is not None else np.zeros((R, N))
        return states, logit


BASELINE = ("classifier", "classifier_change", "classifier_max", "turn", "axis")
DYNAMICAL = ("forecast", "probe", "variance", "autocorrelation")


def predictors(model, acts, score, horizons, window, axis):
    """All predictors as (replica, turn) arrays, and the per-horizon forecast."""
    states, logit = model.features(acts)
    var, ac = rolling_stats(logit, window)
    R, N = score.shape
    change = np.concatenate([np.zeros((R, 1)), np.diff(score, axis=1)], axis=1)
    hist_max = np.maximum.accumulate(score, axis=1)
    turn = np.broadcast_to(np.arange(N, dtype=float), (R, N))
    proj = acts @ axis if axis is not None else np.zeros((R, N))
    hits = hit_probability(model.T, model.hostile, max(horizons))
    base = {"classifier": score, "classifier_change": change, "classifier_max": hist_max, "turn": turn, "axis": proj,
            "probe": logit, "variance": var, "autocorrelation": ac}
    return base, {h: hits[h][states] for h in horizons}


def design(base, forecast_h, r, t, names):
    cols = [(forecast_h if n == "forecast" else base[n])[r, t] for n in names]
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0)


class Combined:
    """Logistic regressions on the baseline features and on baseline plus dynamical features, per horizon."""

    def __init__(self, base, forecast, onset, turns, horizons):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.fits = {}
        for h in horizons:
            rows = horizon_labels(onset, turns, h)
            if len(rows) == 0 or rows[:, 2].min() == rows[:, 2].max():
                continue
            r, t, y = rows[:, 0], rows[:, 1], rows[:, 2]
            for name, feats in (("baseline", BASELINE), ("baseline+dynamics", BASELINE + DYNAMICAL)):
                X = design(base, forecast[h], r, t, feats)
                sc = StandardScaler().fit(X)
                lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X), y)
                self.fits[(h, name)] = (feats, sc, lr)

    def score(self, base, forecast, h, r, t):
        out = {}
        for name in ("baseline", "baseline+dynamics"):
            if (h, name) in self.fits:
                feats, sc, lr = self.fits[(h, name)]
                out["combined:" + name] = lr.decision_function(sc.transform(design(base, forecast[h], r, t, feats)))
        return out


def evaluate(model, acts, score, threshold, horizons, window, boot, seed=0, axis=None, combined=None):
    base, forecast = predictors(model, acts, score, horizons, window, axis)
    R, N = score.shape
    onset = onsets(score, threshold)
    rng = np.random.default_rng(seed)
    out = {}
    for h in horizons:
        rows = horizon_labels(onset, N, h)
        if len(rows) == 0:
            out[h] = {"n": 0}
            continue
        r, t, y = rows[:, 0], rows[:, 1], rows[:, 2]
        preds = {k: v[r, t] for k, v in base.items()}
        preds["forecast"] = forecast[h][r, t]
        if combined is not None:
            preds.update(combined.score(base, forecast, h, r, t))
        res = {"n": int(len(y)), "positives": int(y.sum()), "auc": {k: auc(y, v) for k, v in preds.items()}}
        reps = np.unique(r)
        draws = {k: [] for k in preds}
        for _ in range(boot):
            pick = rng.choice(reps, size=len(reps), replace=True)
            idx = np.concatenate([np.flatnonzero(r == p) for p in pick])
            for k, v in preds.items():
                draws[k].append(auc(y[idx], v[idx]))
        res["auc_ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if boot else None
                           for k, v in draws.items()}
        if "combined:baseline" in res["auc"] and "combined:baseline+dynamics" in res["auc"]:
            res["incremental_auc"] = res["auc"]["combined:baseline+dynamics"] - res["auc"]["combined:baseline"]
            if boot:
                d = np.array(draws["combined:baseline+dynamics"]) - np.array(draws["combined:baseline"])
                res["incremental_auc_ci95"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
        out[h] = res
    return {"replicas": int(R), "turns": int(N), "captured": int((onset < N).sum()),
            "onset_quantiles": np.quantile(onset[onset < N], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
            if (onset < N).any() else None, "horizons": out}


def load(arm):
    import explore_llm_soft_loop as esl

    src = esl.DATA / "explore" / arm
    acts = np.load(src / "activations.npy").astype(np.float64)
    score = np.load(src / "hostility.npz")["classifier"][:, :, LABEL].astype(np.float64)
    return acts, score


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dims", type=int, default=20)
    parser.add_argument("--states", type=int, default=40)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--vectors", default="llm-phase-map/vectors-l12-p1b")
    parser.add_argument("--axis", default="hostility-vs-kindness", help="fixed steering axis for the axis baseline")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    axis = elpm.load_vectors(args.vectors)[args.axis]
    axis = axis / np.linalg.norm(axis)
    tr = [load(a) for a in args.train]
    acts = np.concatenate([a for a, _ in tr])
    score = np.concatenate([s for _, s in tr])
    model = Model(acts, score, args.threshold, args.dims, args.states)
    base, forecast = predictors(model, acts, score, args.horizons, args.window, axis)
    combined = Combined(base, forecast, onsets(score, args.threshold), score.shape[1], args.horizons)
    result = {"args": vars(args), "hostile_states": int(model.hostile.sum()),
              "train": evaluate(model, acts, score, args.threshold, args.horizons, args.window, 0, axis=axis,
                                combined=combined), "test": {}}
    for arm in args.test:
        a, s = load(arm)
        result["test"][arm] = evaluate(model, a, s, args.threshold, args.horizons, args.window, args.boot, axis=axis,
                                       combined=combined)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "lead-time.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "args"}))


if __name__ == "__main__":
    main()
