"""Causal drift-aware diagnostic baselines; no orbit or new-information claim."""
from collections import deque

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PENALTIES = (.01, .1, 1., 10.)
WINDOW = 32
HORIZON = 16
FAMILIES = ('B0', 'B1', 'B2', 'B3')


def plane_coherence(a, b, c, floor=1e-4):
    """Cosine between adjacent oriented 2-planes, from three unit directions."""
    ab, bc, ac = float(a@b), float(b@c), float(a@c)
    p, q = max(0., 1-ab*ab), max(0., 1-bc*bc)
    if min(p, q) < floor:
        return np.nan
    return float(np.clip((ab*bc-ac)/np.sqrt(p*q), -1., 1.))


def ar_forecast(losses, order):
    y = np.asarray(losses, dtype=float)
    if len(y) < WINDOW:
        return np.nan
    y = y[-WINDOW:]
    design = np.column_stack([np.ones(len(y)-order)] +
                              [y[order-j-1:len(y)-j-1] for j in range(order)])
    beta = np.linalg.lstsq(design, y[order:], rcond=None)[0]
    return float(np.r_[1., y[-order:][::-1]]@beta)


class CausalFeatures:
    """Only past/current observations. All gradients are actual objective gradients."""
    def __init__(self):
        self.losses, self.norms = deque(maxlen=32), deque(maxlen=32)
        self.units, self.steps = deque(maxlen=4), deque(maxlen=4)
        self.grams, self.planes = deque(maxlen=8), deque(maxlen=8)
        self.previous_gradient = None
        self.previous_predictions = [np.nan, np.nan]
        self.previous_gnorm = np.nan
        self.names = None
        self.groups = None

    def observe(self, loss, gradient, theta, previous_step=None):
        g, theta = np.asarray(gradient, dtype=float), np.asarray(theta, dtype=float)
        if not np.isfinite(loss) or not np.isfinite(g).all() or not np.isfinite(theta).all():
            raise ValueError('features require finite current state/loss/gradient')
        self.losses.append(float(loss)); self.norms.append(float(np.linalg.norm(theta)))
        gn = float(np.linalg.norm(g))
        self.units.append(-g/gn if gn > 1e-30 else np.full_like(g, np.nan))
        stepnorm = np.nan if previous_step is None else float(np.linalg.norm(previous_step))
        self.steps.append(stepnorm)
        if self.previous_gradient is None or previous_step is None or stepnorm <= 1e-30:
            secant = np.nan
        else:
            secant = float((g-self.previous_gradient)@previous_step/(stepnorm*stepnorm))
        losses = np.asarray(self.losses)
        forecast = [ar_forecast(losses, k) for k in (1, 2)]
        enough = len(losses) == WINDOW
        median = float(np.median(losses)) if enough else np.nan
        mad = float(np.median(abs(losses-median))) if enough else np.nan
        slope = float(np.polyfit(np.arange(WINDOW), losses, 1)[0]) if enough else np.nan
        cosine = float(self.units[-1]@self.units[-2]) if len(self.units) > 1 else np.nan
        plane = plane_coherence(*list(self.units)[-3:]) if len(self.units) >= 3 else np.nan
        self.planes.append(plane)
        gram = np.full(6, np.nan)
        if len(self.units) == 4:
            directions = np.asarray(self.units)
            gram = (directions@directions.T)[np.triu_indices(4, 1)]
        self.grams.append(gram)
        mean = np.full(6, np.nan); variation = mean.copy()
        if len(self.grams) == 8 and np.isfinite(self.grams).all():
            mean, variation = np.mean(self.grams, axis=0), np.std(self.grams, axis=0)
        plane_mean, plane_std = np.nan, np.nan
        if len(self.planes) == 8 and np.isfinite(self.planes).all():
            plane_mean, plane_std = float(np.mean(self.planes)), float(np.std(self.planes))
        entries = []
        def add(group, name, value):
            entries.append((group, name, float(value)))
        # Common geometry-availability mask prevents missingness from favoring B3.
        add('common', 'plane_unavailable', not np.isfinite(plane))
        add('common', 'plane_history_unavailable', not np.isfinite(plane_mean))
        for name, value in [('ce', loss), ('slope32', slope), ('median32', median), ('mad32', mad)]:
            add('B0', name, value)
        for i in range(2):
            add('B0', f'ar{i+1}_forecast', forecast[i])
            add('B0', f'ar{i+1}_previous_absolute_error', abs(loss-self.previous_predictions[i]))
        for name, value in [('gradient_norm', gn), ('gradient_norm_change', gn-self.previous_gnorm),
                            ('consecutive_cosine', cosine), ('secant_curvature', secant),
                            ('parameter_norm_drift32', (self.norms[-1]-self.norms[0])/max(self.norms[0], 1e-30) if enough else np.nan)]:
            add('B1', name, value)
        for i in range(6):
            for label, values in [('gram', gram), ('gram_mean8', mean), ('gram_std8', variation)]:
                add('B2', f'{label}_{i}', values[i])
        # Preserve the full causal history used by B3's nonlinear moments.
        # Entry zero is already present as gram_0..5 above.
        grams = list(self.grams)
        for lag in range(1, 8):
            previous_gram = grams[-lag-1] if len(grams) > lag else np.full(6, np.nan)
            for i, value in enumerate(previous_gram):
                add('B2', f'gram_history_lag{lag}_{i}', value)
        steps = list(self.steps)
        for lag in range(1, 4):
            ratio = steps[-1]/steps[-lag-1] if len(steps) > lag and steps[-lag-1] > 1e-30 else np.nan
            add('B2', f'step_ratio_lag{lag}', ratio)
        for name, value in [('plane_coherence', plane), ('plane_mean8', plane_mean), ('plane_std8', plane_std)]:
            add('B3', name, value)
        self.previous_gradient = g.copy(); self.previous_gnorm = gn
        self.previous_predictions = forecast
        self.groups = [row[0] for row in entries]; self.names = [row[1] for row in entries]
        return np.asarray([row[2] for row in entries])


def burst_labels(losses, guard_step=None):
    """Future labels only for offline scoring; NaN means censored/unavailable."""
    losses = np.asarray(losses, dtype=float)
    labels = np.full(len(losses), np.nan)
    for t in range(WINDOW-1, len(losses)):
        if guard_step is not None and t < guard_step <= t+HORIZON:
            labels[t] = 1
        elif t+HORIZON < len(losses):
            history = losses[t-WINDOW+1:t+1]
            median = np.median(history)
            threshold = median+max(.1, 3*1.4826*np.median(abs(history-median)))
            labels[t] = float(np.max(losses[t+1:t+HORIZON+1]) > threshold)
    return labels


def family_columns(groups, family):
    allowed = {'common', *FAMILIES[:FAMILIES.index(family)+1]}
    return np.asarray([i for i, g in enumerate(groups) if g in allowed])


def fit_logistic(X, y, penalty):
    """Training-only imputation/scaling; intercept unpenalized, mean CE objective."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if len(X) == 0 or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('nonempty finite binary labels required')
    count = np.isfinite(X).sum(0)
    mean = np.divide(np.nansum(X, axis=0), count, out=np.zeros(X.shape[1]), where=count>0)
    filled = np.where(np.isfinite(X), X, mean)
    scale = np.maximum(filled.std(0), 1e-8)
    Z = np.c_[np.ones(len(X)), (filled-mean)/scale]
    def objective(w):
        logits = Z@w
        value = np.mean(np.logaddexp(0, logits)-y*logits)+.5*penalty*(w[1:]@w[1:])
        grad = Z.T@(expit(logits)-y)/len(y)
        grad[1:] += penalty*w[1:]
        return value, grad
    start = np.zeros(Z.shape[1])
    start[0] = np.log((y.sum()+.5)/(len(y)-y.sum()+.5))
    fitted = minimize(objective, start, jac=True, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError('logistic fit failed: '+fitted.message)
    return {'mean':mean.tolist(), 'scale':scale.tolist(), 'weights':fitted.x.tolist(),
            'penalty':float(penalty), 'training_rows':len(y), 'fit_iterations':int(fitted.nit)}


def predict(model, X):
    X = np.asarray(X, float); mean = np.asarray(model['mean']); scale = np.asarray(model['scale'])
    Z = np.c_[np.ones(len(X)), (np.where(np.isfinite(X),X,mean)-mean)/scale]
    return expit(Z@np.asarray(model['weights']))


def prediction_metrics(y, probability, threshold=.95):
    """Average precision groups score ties; no independent-step inference."""
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) == 0:
        return {'rows':0, 'brier':None, 'average_precision':None, 'false_alarms_per_100':None}
    ap = None
    if y.sum() > 0 and y.sum() < len(y):
        order = np.argsort(-probability, kind='stable'); ys = y[order]
        ends = np.r_[np.flatnonzero(np.diff(probability[order])), len(y)-1]
        true = np.cumsum(ys)[ends]; precision = true/(ends+1)
        recall = true/y.sum(); ap = float(np.sum(np.diff(np.r_[0.,recall])*precision))
    return {'rows':len(y), 'positive_rows':int(y.sum()), 'brier':float(np.mean((probability-y)**2)),
            'average_precision':ap, 'false_alarms_per_100':float(100*np.mean((probability>=threshold)&(y==0)))}
