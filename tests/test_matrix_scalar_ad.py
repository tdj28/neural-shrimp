import sys
from pathlib import Path
import pytest
from flint import arb,ctx
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews'),str(Path(__file__).resolve().parents[1]/'tools')]
from review_matrix_scalar_ad import Jet,objective_ad
from review_c1_evidence import objective_ad as prior_ad
from neural_shrimp.rigorous_orbits import BallTanhMap

def model(d,w,b):
    return dict(schema='neural-shrimp-c1-input-v1',network=dict(d=d,width=w,bias=b,activation='tanh',in_scale='1/2',out_scale='1/4',loss_scale='3/2',ridge='1/100'),x=[['1/3']*d,['-2/5']*d],y=['1/7','-1/9'],lr_hidden='1/8',lr_output='1/16')

def test_exact_polynomial_cross_derivatives_and_no_mutation():
    x,y=Jet(arb(2),2,0),Jet(arb(3),2,1);z=(x*x)*y+x*y*y+7
    assert z.v==37 and z.g[0,0]==21 and z.g[1,0]==16
    assert z.h[0,0]==6 and z.h[0,1]==z.h[1,0]==10 and z.h[1,1]==4
    assert x.v==2 and x.g[0,0]==1 and x.h[0,0]==0

def test_closed_tanh_scalar_derivatives():
    with ctx.workprec(192):
        x=Jet(arb(1)/3,1,0);y=x.tanh();t=x.v.tanh();a=1-t*t
        assert y.v.overlaps(t)and y.g[0,0].overlaps(a)and y.h[0,0].overlaps(-2*t*a)

@pytest.mark.parametrize('d,w,b',[(2,3,True),(2,3,False),(8,8,False)])
def test_complete_point_and_box_maps_overlap_separate_analytic_derivative(d,w,b):
    with ctx.workprec(192):
        inp=model(d,w,b);n=d*w+w+(w+1 if b else 0)
        for rad in [arb(0),arb((1,-100))]:
            state=[arb(arb(i+1)/37,rad)for i in range(n)]
            a,j=objective_ad(inp,state);v,k=BallTanhMap(inp).step(state)
            assert all(x.overlaps(y)for x,y in zip(a,v))
            assert all(j[i,l].overlaps(k[i,l])for i in range(n)for l in range(n))
            if b:
                q,h=prior_ad(inp,state)
                assert all(x.overlaps(y)for x,y in zip(a,q))and all(j[i,l].overlaps(h[i,l])for i in range(n)for l in range(n))

def test_unbiased_zero_origin_has_exact_mixed_loss_derivative():
    with ctx.workprec(128):
        inp=model(1,1,False);inp['x']=[['2']];inp['y']=['3'];inp['network'].update(in_scale='1',out_scale='1',loss_scale='1',ridge='0')
        v,j=objective_ad(inp,[arb(0),arb(0)])
        assert v==[arb(0),arb(0)]and j[0,0]==j[1,1]==1
        assert j[0,1]==arb(12)/8 and j[1,0]==arb(12)/16
        with pytest.raises(ValueError):objective_ad(inp,[arb(0)])
