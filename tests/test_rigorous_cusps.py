import numpy as np
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_cusps import CuspJet,augmented_cusp,TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,dyadic
from neural_shrimp.rigorous_folds import augmented,FoldTanhMap

class ScalarCusp:
    n=1
    def cusp_step(self,x,a,b):return [x[0]+(a-1)+(b-1)*x[0]+x[0]*x[0]*x[0]]

def floats(m):return np.array([[float(m[i,j]) for j in range(m.ncols())] for i in range(m.nrows())])

def test_scalar_cusp_augmented_polynomial_and_all_jacobian_columns():
    with ctx.workprec(256):
        z=[exact(s) for s in ['1/5','9/10','11/10','4/5','-3/10']];x,a,b,q,h=z
        v,j,d=augmented_cusp(ScalarCusp(),z,1)
        target=[(a-1)+(b-1)*x+x*x*x,(b-1+3*x*x)*q,(b-1+3*x*x)*h+6*x*q*q,(q*q-1)/2,q*h]
        deriv=[[b-1+3*x*x,1,x,0,0],[6*x*q,0,q,b-1+3*x*x,0],[6*x*h+6*q*q,0,h,12*x*q,b-1+3*x*x],[0,0,0,q,0],[0,0,0,h,q]]
        assert all(v[i,0].overlaps(target[i]) for i in range(5))
        assert all(j[i,k].overlaps(arb(deriv[i][k])) for i in range(5) for k in range(5))
        v,j,d=augmented_cusp(ScalarCusp(),[arb(0),arb(1),arb(1),arb(1),arb(0)],1)
        assert all(v[i,0].is_zero() for i in range(5)) and not j.det().contains(0)

@pytest.mark.parametrize('period',[1,3])
def test_tanh_composition_full_augmented_jacobian_finite_difference(period):
    # Only scalar division is expressed as multiplication in the test model.
    class Map:
        n=2
        def cusp_step(self,x,a,b):return [(x[0]*a+x[1]*.5).tanh()+b*x[1],(x[1]*b-x[0]*(arb(1)/3)).tanh()+a*x[0]]
    z=np.array([.1,-.2,.4,.3,.7,-.4,.2,.5]);eps=1e-6
    with ctx.workprec(256):
        def calc(x):return augmented_cusp(Map(),[exact(dyadic(v)) for v in x],period)
        value,j,_=calc(z);jac=floats(j)
        for k in range(len(z)):
            step=np.eye(len(z))[k]*eps;plus=calc(z+step)[0];minus=calc(z-step)[0]
            difference=(floats(plus)-floats(minus))[:,0]/(2*eps)
            np.testing.assert_allclose(jac[:,k],difference,rtol=3e-7,atol=2e-9)

def test_tanh_gd_first_two_blocks_match_unchanged_mixedjet_and_restore_rates():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'d':1,'width':1,'bias':True,'activation':'tanh','in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[['1/3'],['-1/4']],'y':['1/7','-1/9'],'lr_hidden':'1/10','lr_output':'1/20'}
    with ctx.workprec(256):
        m=TwoRateCuspMap(inp);old=m.rates;n=m.n;x=[exact(str(i)+'/20') for i in range(n)];q=[exact('1/3')]*n;h=[exact('-1/7')]*n
        c,j,d=augmented_cusp(m,x+[exact('1/10'),exact('1/20')]+q+h,3)
        a,b,data=augmented(FoldTanhMap(inp),x+[exact('1/10')]+q,3)
        assert m.rates is old
        for i in range(2*n):assert c[i,0].overlaps(a[i])
        for i in range(n):
            for k in range(n):assert d['direction_gradients'][i,k].overlaps(b[n+i,k])

def test_dimension_mismatch_and_interruption():
    with pytest.raises(ValueError):augmented_cusp(ScalarCusp(),[arb(0)],1)
    def stop():raise TimeoutError('unit')
    with pytest.raises(TimeoutError):augmented_cusp(ScalarCusp(),[arb(0),arb(1),arb(1),arb(1),arb(0)],1,stop)
    a=CuspJet.variable(arb(0),2,0);b=CuspJet.variable(arb(1),3,1)
    with pytest.raises(ValueError):a+b
    with pytest.raises(ValueError):a*b


def test_scalar_cusp_full_local_proof_and_cubic_unfolding_constants():
    from neural_shrimp.rigorous_cusps import cusp_proof
    from neural_shrimp.rigorous_connection import restore_ball
    with ctx.workprec(256):
        r=cusp_proof(ScalarCusp(),[arb(0),arb(1),arb(1),arb(1),arb(0)],1,newton_steps=1)
        assert r['passed'] and all(r['gates'].values())
        assert restore_ball(r['cubic_coefficient']).contains(1)
        assert restore_ball(r['unfolding_determinant']).contains(1)

@pytest.mark.parametrize('values,expected', [([0,.5,2],True),([0,-1,2],False),([0,1,2],False)])
def test_unit_circle_exclusion_allows_unstable_modes_but_rejects_neutral(values,expected):
    from flint import arb_mat
    from neural_shrimp.rigorous_cusps import spectral_exclusion
    with ctx.workprec(256):
        m=arb_mat([[arb(values[i]) if i==j else arb(0) for j in range(3)] for i in range(3)])
        assert spectral_exclusion(m)['passed']==expected

def test_complex_pair_unit_circle_exclusion():
    from flint import arb_mat
    from neural_shrimp.rigorous_cusps import spectral_exclusion
    with ctx.workprec(256):
        for radius,expected in [(exact('1/2'),True),(arb(1),False),(arb(2),True)]:
            m=arb_mat([[arb(0),-radius],[radius,arb(0)]])
            assert spectral_exclusion(m)['passed']==expected
