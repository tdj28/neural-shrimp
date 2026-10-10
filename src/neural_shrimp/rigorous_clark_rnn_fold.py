"""Native leaky-RNN local folds using the unchanged generic Arb fold engine.

Mixed jets differentiate the map itself. No finite differences enter proof gates.
The active parameter is delta; J0 and gain remain exact fixed constants.
"""
from copy import deepcopy
from flint import arb, ctx
from .rigorous_orbits import exact
from .rigorous_folds import _proof

class FoldClarkRNN:
    def __init__(self, inputs):
        if inputs.get('schema') != 'clark-rnn-fold-map-v1':
            raise ValueError('fold map schema')
        self.n = len(inputs['j0'])
        if not 1 <= self.n <= 16 or any(len(r) != self.n for r in inputs['j0']):
            raise ValueError('matrix dimension')
        self.j = [[exact(x) for x in r] for r in inputs['j0']]
        self.g = exact(inputs['gain'])
        if not self.g.is_finite() or not self.g > 0 or not all(x.is_finite() for r in self.j for x in r):
            raise ValueError('finite exact model constants')

    def jet_step(self, x, delta):
        if len(x) != self.n:
            raise ValueError('state dimension')
        t = [a.tanh() for a in x]
        return [(1-delta)*x[i] + delta*self.g*sum((t[k]*self.j[i][k] for k in range(self.n)), 0)
                for i in range(self.n)]

def relabel(record, n):
    r = deepcopy(record)
    if 'gates' in r and 'positive_hidden_rate' in r['gates']:
        r['gates']['positive_delta'] = r['gates'].pop('positive_hidden_rate')
    if 'hidden_rate_second_derivative' in r:
        r['delta_second_derivative'] = r.pop('hidden_rate_second_derivative')
    r['schema'] = 'clark-rnn-fold-result-v1'
    return r

def check_rnn_fold(inputs, *, precision=256, radius_exponent=-80, newton_steps=5,
                   max_power=512, tick=lambda: None, observe=lambda kind,record: None):
    if inputs.get('schema') != 'clark-rnn-fold-input-v1':
        raise ValueError('fold input schema')
    for value, lo, hi in ((precision,128,2048),(radius_exponent,-1024,-1),
                          (newton_steps,0,12),(max_power,1,4096),(inputs['period'],1,64)):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError('invalid bounded fold settings')
    with ctx.workprec(precision):
        model = FoldClarkRNN(inputs['map_inputs'])
        z = [exact(x) for x in inputs['approximation']]
        if len(z) != 2*model.n+1 or not all(x.is_finite() for x in z) or not 0 < z[model.n] < 1:
            raise ValueError('augmented approximation or open leak domain')
        r = _proof(model,z,inputs['period'],radius_exponent,newton_steps,max_power,tick,
                   lambda kind,record: observe(kind,relabel(record,model.n)))
        r = relabel(r,model.n)
        b = r['box'][model.n]
        delta = arb(arb(tuple(b['mid'])),arb(tuple(b['rad'])))
        r['gates']['delta_below_one'] = bool(delta < 1)
        r['passed'] = bool(all(r['gates'].values()))
        r['evidence_rung'] = 3 if r['passed'] else 0
        if not r['passed'] and r['status'] == 'primitive_nondegenerate_fold_with_attracting_complement':
            r['status'] = 'fold_gates_incomplete'
        r['settings']['precision'] = precision
        r['exact_statement'] = {'period':inputs['period'],'dimension':model.n,
            'augmented_dimension':2*model.n+1,'parameter':'leak delta in (0,1); exact fixed gain and J0',
            'scope':'exact-real discrete-map local fold only',
            'normalization':'q^T q=1 and p^T q=1; a=p^T R_delta, c=p^T DxxR[q,q]; normal form quadratic coefficient c/2',
            'not_claimed':['attraction of the neutral fold','global branch identity or connection to body certificate',
                           'whole shrimp geometry','continuous-time bifurcation or inference utility']}
        return r
