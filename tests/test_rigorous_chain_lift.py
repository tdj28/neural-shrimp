import copy
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_chain_lift import native_lift,quadratic_membership
from neural_shrimp.rigorous_orbits import vector_json

class Jordan:
    n=2
    def cusp_step(self,x,a,b):return [x[0]+x[1],x[1]]

def fixture():
    box=[arb(0),arb(0),arb(1),arb(1),arb(1),arb(0),arb(0),arb(1)/3]
    normal=['0']*14;normal[-2]='1';origin=['0']*14
    lift=native_lift(Jordan(),box,3,normal,origin)
    center=[arb(0),arb(0),arb(1),arb(0)]*3+[arb(1),arb(1)]
    slope=[arb(0)]*14;slope[-2]=arb(1)
    chart={'schema':'neural-shrimp-quadratic-eigenpair-locus-v1','passed':True,'gates':dict.fromkeys(('uniform_normalized_eigenpair_locus','primitive_period','anchor_identified'),True),'period':3,'dimension':2,'sigma':1,'shooting_dimension':14,'normal':normal,'origin':origin,'parameter_center':'1','parameter_endpoints':['1/2','3/2'],'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json([arb(0)]*14),'weights':['1']*14,'settings':{'halfwidth_exponent':-1,'radius_exponent':-8},'endpoint_root_boxes':[]}
    return lift,chart

def test_native_phase_order_and_projection_are_exact():
    with ctx.workprec(256):
        lift,c=fixture();assert lift['closure_contains_zero']
        from neural_shrimp.rigorous_eigenpair_march import decode
        assert decode(lift['parameter'])==1
        assert all(decode(v)==expected for v,expected in zip(lift['box'],[0,0,1,0]*3+[1,1]))
        assert quadratic_membership(lift,c)['passed']

@pytest.mark.parametrize('i',[0,2,4,8,12,13])
def test_all_native_state_mode_phase_rate_coordinates_are_required(i):
    lift,c=fixture();lift['box'][i]=vector_json([arb(2)])[0]
    assert not quadratic_membership(lift,c)['passed']

def test_unknown_gauge_interval_not_midpoint_is_tested():
    lift,c=fixture();lift['box'][-2]=vector_json([arb(1,arb(1)/2)])[0]
    assert not quadratic_membership(lift,c)['parameter_inside']

def test_curvature_is_required_in_predictor_membership():
    lift,c=fixture();lift['box'][-2]=vector_json([arb(5)/4])[0];lift['box'][0]=vector_json([arb(1)/16])[0]
    curve=[arb(0)]*14;curve[0]=arb(2);c['curvature']=vector_json(curve)
    assert quadratic_membership(lift,c)['passed']
    c['curvature']=vector_json([arb(0)]*14)
    assert not quadratic_membership(lift,c)['passed']

@pytest.mark.parametrize('kind',['gauge','sign','failed','closure'])
def test_mismatched_or_unproved_conventions_rejected(kind):
    lift,c=fixture()
    if kind=='gauge':lift['normal']=list(lift['normal']);lift['normal'][0]='1'
    elif kind=='sign':lift['sigma']=-1
    elif kind=='failed':c['passed']=False
    else:lift['closure_contains_zero']=False
    with pytest.raises(ValueError):quadratic_membership(lift,c)

def test_interrupt_and_invalid_native_dimensions():
    def stop():raise TimeoutError('fixed cap')
    with pytest.raises(TimeoutError):native_lift(Jordan(),[arb(1)]*8,3,['1']*14,['0']*14,tick=stop)
    with pytest.raises(ValueError):native_lift(Jordan(),[arb(1)]*8,19,['1']*78,['0']*78)
