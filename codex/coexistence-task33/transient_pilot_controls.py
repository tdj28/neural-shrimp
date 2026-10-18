"""Tiny actual propagation-loop controls; no model evaluation."""
from flint import arb, ctx
from transient_pilot import propagate
ctx.prec=2048
saved={}; seen=[]
def save(n,v): saved[n]=v
def forward(x):
    seen.append(x[0])
    return [2*x[0]+1],[arb(0)]
x=arb(1, '0.01')
r=propagate([x],forward,3,save)
assert len(saved)==4 and len(seen)==3
assert seen[0].rad()>0 and seen[1].rad()>=2*x.rad()
assert r[0].contains(arb(15, '0.08'))
assert saved['state-003.json']['state'][0]['rad'][0]>0
for n in (0,21,True):
    try: propagate([x],forward,n,save)
    except ValueError: pass
    else: raise AssertionError('invalid step count accepted')
for f in (lambda x:([],[]), lambda x:([arb('nan')],[arb(0)]),lambda x:([x[0]],[arb(1)])):
    try: propagate([x],f,1,save)
    except ValueError: pass
    else: raise AssertionError('invalid output accepted')
print('7 propagation controls passed: radius chain, 3 step bounds, 3 output refusals')
