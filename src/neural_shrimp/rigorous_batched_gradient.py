"""Aggregate counters around unchanged per-chart mean-value derivative objects."""
from copy import deepcopy
from flint import ctx
from .rigorous_connection import require
from .rigorous_gradient_enclosure import MeanValueHiddenRateMap


class BatchedMeanValueHiddenRateMap:
    """Fresh72-call objects, same exact inputs/precision, strict aggregate cap.

    The historical per-instance180-call limit remains untouched. Counts include
    attempted-but-not-returned operations even if a nested unit call raises.
    No completed block is discarded before its counters are accumulated.
    """
    FIELDS=('map_attempted','map_returned','unit_attempted','unit_returned')
    def __init__(self,inputs,max_calls,*,factory=MeanValueHiddenRateMap):
        require(type(max_calls) is int and 1<=max_calls<=1152,'aggregate map cap')
        require(128<=ctx.prec<=512,'constructor precision')
        self.inputs=deepcopy(inputs);self.max_calls=max_calls;self.factory=factory;self.precision=ctx.prec
        self.completed=dict.fromkeys(self.FIELDS,0);self.instances=0
        self.base=self._new(min(72,max_calls));self.n=self.base.n;self.hidden=self.base.hidden;self.output=self.base.output

    def _new(self,cap):
        require(ctx.prec==self.precision,'unchanged constructor precision')
        result=self.factory(deepcopy(self.inputs),max_calls=cap);self.instances+=1
        return result

    def counts(self):
        current=self.base.counts()
        return {**{key:self.completed[key]+current[key] for key in self.FIELDS},
                'map_cap':self.max_calls,'unit_cap':2*self.max_calls}

    def step(self,theta,hidden_rate):
        require(ctx.prec==self.precision,'unchanged derivative precision')
        require(self.counts()['map_attempted']<self.max_calls,'aggregate map evaluations')
        local=self.base.counts()
        if local['map_attempted']==local['map_cap']:
            require(local['map_returned']==local['map_cap'] and local['unit_attempted']==local['unit_returned']==2*local['map_cap'],'completed derivative block before rotation')
            remaining=self.max_calls-(self.completed['map_attempted']+local['map_attempted'])
            candidate=self._new(min(72,remaining))
            require(candidate.n==self.n and candidate.hidden==self.hidden and candidate.output==self.output,'identical native model and output rate')
            for key in self.FIELDS:self.completed[key]+=local[key]
            self.base=candidate
        return self.base.step(theta,hidden_rate)
