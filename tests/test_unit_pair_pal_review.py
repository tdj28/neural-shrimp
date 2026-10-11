import sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
import review_unit_pair_pal_math as A
from neural_shrimp.rigorous_orbits import vector_json

def test_direct_convolution_retains_all_cubic_terms():
    assert A.mul([1,2,3,4],[5,6,7,8])==[5,16,34,60]
def test_entire_series_at_origin_derivatives_through_three():
    with ctx.workprec(256):
        t=[arb(0),arb(1),arb(0),arb(0)]
        for offset in (0,1):
            values=A.terminal_series(t,offset)
            from math import factorial
            for j,v in enumerate(values):
                lo,hi=A.M.interval(A.ball_json(v));assert lo<=F((-1)**j,factorial(2*j+offset))<=hi
def test_anchor_uses_full_parameter_projection_and_strict_box():
    c=[arb(0)]*3;s=[arb(1),arb(0),arb(0)];a=[arb(0)]*3;normal=s;rr=[arb(1)]*3
    good=vector_json([arb(0),arb(0),arb(0)])
    assert A.anchor_membership(good,c,s,a,normal,c,arb(0),arb(1),rr)['passed']
    bad=vector_json([arb(0),arb(1),arb(0)])
    assert not A.anchor_membership(bad,c,s,a,normal,c,arb(0),arb(1),rr)['passed']
    outside=vector_json([arb(2),arb(0),arb(0)])
    assert not A.anchor_membership(outside,c,s,a,normal,c,arb(0),arb(1),rr)['passed']


def test_actual_serialized_jet_schema_and_full_input_rejection():
    from neural_shrimp.rigorous_eigenpair_path import PathJet
    from neural_shrimp.rigorous_orbits import ball_json
    with ctx.workprec(256):
        c=list(map(arb,[1,2,1,0,0,1,1,1,0]));s=[arb(0)]*8+[arb(1)];a=[arb(0)]*9;d=arb(0)
        events=[]
        for j,name in [(1,'q'),(2,'r')]:
            inputs=[PathJet([c[i],s[i],0,0],[c[2*j+i],s[2*j+i],0,0])for i in range(2)]+[PathJet([c[-3],0,0,0]),PathJet([c[-2],0,0,0])]
            events.append(dict(phase=0,direction=name,d=ball_json(d),inputs=[v.serialized()for v in inputs],returns=[v.serialized()for v in inputs[:2]]))
        got=A.path_coefficients(c,s,a,d,events,2,1,s,c,arb(0))
        assert got.nrows()==9 and got.ncols()==4
        assert all(got[i,j].is_zero()for i in [0,1,6,7,8]for j in range(4))
        events[0]['inputs'][0][0][0]=ball_json(arb(9))
        with pytest.raises(ValueError):A.path_coefficients(c,s,a,d,events,2,1,s,c,arb(0))


def test_exact_gauge_endpoint_join_rejects_changed_gauge_and_boundary():
    import copy
    from neural_shrimp.rigorous_orbits import ball_json
    n=705
    old=dict(shooting_dimension=n,normal=['1']+['0']*(n-1),origin=['0']*n,parameter_center='0',parameter_endpoints=['-1','1'],settings={'halfwidth_exponent':0})
    new=dict(old,parameter_center='2',parameter_endpoints=['1','3'],center=vector_json([arb(2)]+[arb(0)]*(n-1)),slope=vector_json([arb(1)]+[arb(0)]*(n-1)),curvature=vector_json([arb(0)]*n),coordinate_radii=vector_json([arb(1)]*n))
    endpoint=vector_json([arb(1)]+[arb(0)]*(n-1))
    assert A.exact_gauge_join(old,endpoint,new)['passed']
    bad=copy.deepcopy(new);bad['normal'][1]='1'
    with pytest.raises(ValueError,match='identical exact'):A.exact_gauge_join(old,endpoint,bad)
    bad=copy.deepcopy(new);bad['parameter_center']='4';bad['parameter_endpoints']=['3','5']
    with pytest.raises(ValueError,match='ordered common'):A.exact_gauge_join(old,endpoint,bad)
    bad=endpoint[:];bad[1]=ball_json(arb(1))
    assert not A.exact_gauge_join(old,bad,new)['passed']
    bad=endpoint[:];bad[0]=ball_json(arb(0))
    with pytest.raises(ValueError,match='gauge consistent'):A.exact_gauge_join(old,bad,new)
