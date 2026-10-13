"""Fixed capture, hysteresis, and exit-control replay; no model fitting."""
import numpy as np
from reducers import capture_row, difference_interval, controller_rows
from inputs import CAPTURE, HISTORY, CONTROL, load_basic

POOLS = {
    'ramp': ('p1-ramp-s1','p1-ramp-s2'),
    'fresh040': ('p1-h040','p1-c040-s2'),
    'fresh050': ('p1-h050','p1-h050s2'),
    'fresh060': ('p1-h060','p1-h060s2'),
    'fresh030': ('p1-c030-s1','p1-c030-s2'),
    'fresh000': ('p1-free',),
    'hold040': ('p1-exit040-s1','p1-exit040-s2'),
    'hold030': ('p1-exit030','p1-exit030-s2'),
    'hold020': ('p1-exit020',),
    'hold000': ('p1-exit000','p1-exit000-s2'),
    'holdm020': ('p1-exitm020','p1-exitm020-s2'),
}
CONTRASTS = (('ramp','fresh040'),('ramp','fresh050'),('hold040','fresh040'),('hold030','fresh030'))


def calculate(data, draws=4000):
    capture = {name:capture_row(data[name]['score']) for name in CAPTURE}
    history = {name:capture_row(data[name]['score']) for name in HISTORY}
    pools = {name:np.concatenate([data[a]['score'] for a in arms]) for name,arms in POOLS.items()}
    pooled = {name:capture_row(score) for name,score in pools.items()}
    contrasts = {a+'-minus-'+b:difference_interval((pools[a][:,35:40]>.5).mean(axis=1),
                 (pools[b][:,35:40]>.5).mean(axis=1),draws) for a,b in CONTRASTS}
    control = controller_rows({name:data[name] for name in CONTROL},draws)
    return {'capture':capture,'history_seed_rows':history,'history_pooled':pooled,
            'history_contrasts':contrasts,'controller':control,
            'actual_sample_args':{name:arm.get('args',{'synthetic_only':True}) for name,arm in data.items()},
            'conventions':{'hostile':'toxic > 0.5','onset':'zero-based, conditional on event',
                           'late':'35 through 39 inclusive','stay_late':'origins 30 through 38',
                           'bootstrap_hysteresis':'independent samples, 4000, seed0 reset per contrast',
                           'bootstrap_controller':'4000, seed0 shared, fixed historical CLI order'}}


def load():
    return {a:load_basic(a,a in CONTROL) for a in CAPTURE+HISTORY+CONTROL}


def synthetic():
    rng=np.random.default_rng(9174)
    data={}
    for a in CAPTURE+HISTORY+CONTROL:
        score=rng.uniform(size=(128,40))
        data[a]={'score':score,'counter':rng.choice([-.06,0.],size=(128,40)),
                 'texts':[['neutral wording']*40 for _ in range(128)]}
    return data
