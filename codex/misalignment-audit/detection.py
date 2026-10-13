"""Own feature construction and model orchestration; no producer imports."""
from __future__ import annotations
import warnings
import numpy as np
from reducers import event_onsets, risk_rows, increment

PREDICTORS = ('D1_forecast', 'D2_susceptibility', 'D3_state_probe', 'D4_token_max', 'all')


def make_table(arms, horizon, run):
    fields = {k: [] for k in ('y', 'group', 'base', 'extra', 'state', 'arm', 'turn')}
    offset = 0
    for index, arm in enumerate(arms):
        score = np.asarray(arm['score'], dtype=np.float64)
        rows = risk_rows(event_onsets(score > .5, run), score.shape[1], horizon, run)
        r, t, y = rows.T
        previous = np.zeros_like(score)
        previous[:, 1:] = score[:, :-1]
        maxima = np.maximum.accumulate(score, axis=1)
        base = np.column_stack((score[r, t], score[r, t]-previous[r, t], maxima[r, t],
                                t, arm['axis'][r, t], np.full(len(r), arm['dose'])))
        tok = arm['tok'][r, t]
        if np.isinf(tok).any():
            raise ValueError('infinite token projection: no silent nonfinite selection')
        tokenmax = np.where(np.isfinite(tok).any(axis=1), np.max(np.where(np.isnan(tok), -np.inf, tok), axis=1), 0)
        extra = np.column_stack((arm['q'][r, t], arm['q_plus'][r, t]-arm['q'][r, t], tokenmax))
        state = arm['pre'][r, t+1].astype(np.float64)
        if not (np.isfinite(base).all() and np.isfinite(extra).all() and np.isfinite(state).all()):
            raise ValueError('nonfinite feature')
        for k, v in (('y', y), ('group', r+offset), ('base', base), ('extra', extra),
                     ('state', state), ('arm', np.full(len(r), index)), ('turn', t)):
            fields[k].append(v)
        offset += len(score)
    return {k: np.concatenate(v, axis=0) for k, v in fields.items()}


def fit_logistic(x, y, diagnostics):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if np.unique(y).size != 2:
        raise ValueError('training set lacks one class')
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=5000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(scaler.transform(x), y)
    diagnostics.append({'n_train': len(y), 'n_features': x.shape[1],
                        'iterations': model.n_iter_.tolist(), 'parameters': model.get_params(),
                        'warnings': [str(w.message) for w in caught]})
    return scaler, model


def score_model(pair, x):
    scaler, model = pair
    return model.decision_function(scaler.transform(x))


def fit_pca(x, diagnostics):
    from sklearn.decomposition import PCA
    model = PCA(n_components=20, random_state=0).fit(x)
    diagnostics.append({'pca_n_train': len(x), 'n_features': x.shape[1],
                        'components': 20, 'solver': model._fit_svd_solver})
    return model


def predictor_features(tab, state, predictor):
    if predictor == 'D3_state_probe':
        extra = state
    elif predictor == 'all':
        extra = np.column_stack((tab['extra'], state))
    else:
        col = {'D1_forecast': 0, 'D2_susceptibility': 1, 'D4_token_max': 2}[predictor]
        extra = tab['extra'][:, [col]]
    return np.column_stack((tab['base'], extra))


def crossfit(tab, diagnostics):
    groups = np.unique(tab['group'])
    permuted = np.random.default_rng(0).permutation(groups)
    fold_by_group = dict(zip(permuted.tolist(), (np.arange(len(groups)) % 5).tolist()))
    fold = np.array([fold_by_group[g] for g in tab['group']], dtype=int)
    outputs = {k: np.full(len(fold), np.nan) for k in ('baseline', *PREDICTORS)}
    for k in range(5):
        train = {key: val[fold != k] for key, val in tab.items()}
        test = {key: val[fold == k] for key, val in tab.items()}
        if set(train['group']) & set(test['group']):
            raise AssertionError('dialogue leakage')
        pca = fit_pca(train['state'], diagnostics)
        ztr, zte = pca.transform(train['state']), pca.transform(test['state'])
        model = fit_logistic(train['base'], train['y'], diagnostics)
        outputs['baseline'][fold == k] = score_model(model, test['base'])
        for predictor in PREDICTORS:
            xtr = predictor_features(train, ztr, predictor)
            xte = predictor_features(test, zte, predictor)
            model = fit_logistic(xtr, train['y'], diagnostics)
            outputs[predictor][fold == k] = score_model(model, xte)
    if not all(np.isfinite(v).all() for v in outputs.values()):
        raise ValueError('incomplete crossfit scores')
    return outputs, fold


def frozen_fit(train, diagnostics):
    pca = fit_pca(train['state'], diagnostics)
    full = np.column_stack((train['base'], pca.transform(train['state'])))
    models = (fit_logistic(train['base'], train['y'], diagnostics),
              fit_logistic(full, train['y'], diagnostics))
    scores = [score_model(model, x) for model, x in zip(models, (train['base'], full))]
    thresholds = [float(np.quantile(s[train['y'] == 0], .9)) for s in scores]
    return pca, models, thresholds


def frozen_score(fitted, test):
    pca, models, thresholds = fitted
    x = (test['base'], np.column_stack((test['base'], pca.transform(test['state']))))
    return [score_model(model, value) for model, value in zip(models, x)], thresholds


def operating_point(labels, scores, thresholds):
    y = np.asarray(labels)
    result = {}
    for name, values, threshold in zip(('baseline', 'with_state'), scores, thresholds):
        result[name] = float(np.mean(values[y == 1] > threshold))
        result['false_positive_rate_'+name] = float(np.mean(values[y == 0] > threshold))
    return result


def evaluate_l7(tab, draws=2000):
    diagnostics = []
    scores, fold = crossfit(tab, diagnostics)
    rows = []
    for predictor in PREDICTORS:
        row = increment(tab['y'], tab['group'], scores['baseline'], scores[predictor], draws)
        row['predictor'] = predictor
        rows.append(row)
    return rows, scores, fold, diagnostics


def evaluate_l8(train, test, draws=2000):
    diagnostics = []
    fitted = frozen_fit(train, diagnostics)
    scores, thresholds = frozen_score(fitted, test)
    rows = []
    for name, mask in [('pooled', np.ones(len(test['y']), bool)),
                       ('p1-conf040', test['arm'] == 0), ('p1-conf045', test['arm'] == 1)]:
        row = increment(test['y'][mask], test['group'][mask], scores[0][mask], scores[1][mask], draws)
        row.update(test=name, flagged_at_train_10pct_fpr=operating_point(test['y'][mask],
                     [s[mask] for s in scores], thresholds), thresholds=thresholds)
        rows.append(row)
    return rows, scores, diagnostics
