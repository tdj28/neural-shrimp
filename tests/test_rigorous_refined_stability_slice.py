import pytest
from flint import ctx,arb,arb_mat
import neural_shrimp.rigorous_refined_stability_slice as m
from neural_shrimp.rigorous_arclength_connection import arclength_tube
from neural_shrimp.rigorous_orbits import ball_json
from neural_shrimp.rigorous_connection import restore_ball

class NativeFold:
    n=1
    hidden=1
    def step(self,x,h):return [(1-h)*x[0]],arb_mat([[1-h]]),[-x[0]]
    def jet_step(self,x,h):return [(1-h)*x[0]]

def chart():
    with ctx.workprec(256):
        a=NativeFold()
        return arclength_tube(a,a,['0','1/4'],normal=['0','1'],origin=['0','1/4'],parameter='0',weights=['1','1'],period=1,
          halfwidth_exponent=-4,radius_exponent=-12,newton_steps=0,anchor={'state_box':[ball_json(arb(0)),ball_json(arb('1/4'))]})

def test_complete_refined_native_synthetic():
    with ctx.workprec(256):
        events=[]
        out=m.refined_stability_slice(NativeFold(),NativeFold(),NativeFold(),chart(),'1/32','1/16',observe=lambda k,r:events.append((k,r)))
        assert out['passed'] and out['status']=='complete_refined_slice_bounds'
        assert sum(k.endswith('slice_intersection') for k,r in events)==6
        a=next(r for k,r in events if k=='base_physical_inputs');b=next(r for k,r in events if k=='refined_inputs')
        assert all(restore_ball(y).contains(restore_ball(x)) for x,y in zip(a['actual_whole_box'],b['actual_whole_box']))
        assert [r['phase'] for k,r in events if k=='refined_phase']==[0]
        assert out['tangent_refinement']['passed']

def test_failed_prerequisite_never_acquires_refined_jet(monkeypatch):
    monkeypatch.setattr(m,'stability_slice',lambda *a,**k:{'passed':False,'status':'inconclusive_uniform_tangent'})
    out=m.refined_stability_slice(NativeFold(),NativeFold(),object(),chart(),'1/32','1/16')
    assert out['status']=='inconclusive_base_prerequisites'

def test_failed_refinement_stops_after_durable_record(monkeypatch):
    monkeypatch.setattr(m,'refine_tangent',lambda *a:{'passed':False})
    events=[]
    out=m.refined_stability_slice(NativeFold(),NativeFold(),NativeFold(),chart(),'1/32','1/16',observe=lambda k,r:events.append(k))
    assert not out['passed'] and events[-1]=='tangent_refinement' and 'refined_phase' not in events

def test_storage_failure_propagates():
    def observer(k,r):
        if k=='tangent_refinement':raise OSError('retention failed')
    with pytest.raises(OSError,match='retention failed'):m.refined_stability_slice(NativeFold(),NativeFold(),NativeFold(),chart(),'1/32','1/16',observe=observer)
