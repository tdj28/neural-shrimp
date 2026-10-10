import numpy as np
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_generalized_chain import augmented_chain,chain_root_proof
from neural_shrimp.rigorous_orbits import exact,dyadic
from neural_shrimp.rigorous_cusps import TwoRateCuspMap

class NormalForm:
    n=2
    def cusp_step(self,x,a,b):return [x[0]+x[1],x[1]+a-1+(b-1)*x[1]+x[0]*x[0]+x[0]*x[1]*3]

def array(m):return np.array([[float(m[i,j])for j in range(m.ncols())]for i in range(m.nrows())])
def point():return [arb(0),arb(0),arb(1),arb(1),arb(1),arb(0),arb(0),arb(1)]

def test_polynomial_generalized_chain_all_jacobian_blocks():
    with ctx.workprec(256):
        z=[exact(x)for x in ['1/5','-1/7','11/10','9/10','4/5','-2/5','1/3','2/3']];x,y,a,b,q,s,r,t=z
        v,j,_=augmented_chain(NormalForm(),z,1);ax=2*x+3*y;ay=b-1+3*x
        expected=[y,a-1+(b-1)*y+x*x+3*x*y,s,ax*q+ay*s,t-q,ax*r+ay*t-s,(q*q+s*s-1)/2,q*r+s*t]
        jac=[[0,1,0,0,0,0,0,0],[ax,ay,1,y,0,0,0,0],[0,0,0,0,0,1,0,0],[2*q+3*s,3*q,0,s,ax,ay,0,0],[0,0,0,0,-1,0,0,1],[2*r+3*t,3*r,0,t,0,-1,ax,ay],[0,0,0,0,q,s,0,0],[0,0,0,0,r,t,q,s]]
        assert all(v[i,0].overlaps(expected[i])for i in range(8))
        assert all(j[i,k].overlaps(arb(jac[i][k]))for i in range(8)for k in range(8))
        v,j,_=augmented_chain(NormalForm(),point(),1)
        assert all(v[i,0].is_zero()for i in range(8)) and not j.det().contains(0)

@pytest.mark.parametrize('period',[1,3])
def test_composed_tanh_jacobian_all_columns(period):
    class Map:
        n=2
        def cusp_step(self,x,a,b):return [(x[0]*a+x[1]*.5).tanh()+b*x[1],(x[1]*b-x[0]*(arb(1)/3)).tanh()+a*x[0]]
    z=np.array([.1,-.2,.4,.3,.7,-.4,.2,.5]);eps=1e-6
    with ctx.workprec(256):
        def calc(x):return augmented_chain(Map(),[exact(dyadic(v))for v in x],period)
        _,j,_=calc(z)
        for k in range(len(z)):
            step=np.eye(len(z))[k]*eps
            diff=(array(calc(z+step)[0])-array(calc(z-step)[0]))[:,0]/(2*eps)
            np.testing.assert_allclose(array(j)[:,k],diff,rtol=3e-7,atol=2e-9)


def test_interval_root_and_fixed_point_period_rejection():
    with ctx.workprec(256):
        result=chain_root_proof(NormalForm(),point(),1,newton_steps=1)
        assert result['passed'] and all(result['gates'].values())
        # The same fixed point is not a primitive period-two orbit. q stays
        # unit length, and the generalized vector scales by half for F^2.
        z=point();z[-1]=arb(1)/2
        result=chain_root_proof(NormalForm(),z,2)
        assert result['gates']['normalized_augmented_root_unique']
        assert not result['gates']['primitive_period'] and not result['passed']


def test_gd_direct_return_and_chain_identity_matches_old_derivative():
    from neural_shrimp.rigorous_orbits import BallTanhMap,eye
    inputs={'schema':'neural-shrimp-c1-input-v1','network':{'d':1,'width':1,'bias':True,'activation':'tanh','in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[['1/3'],['-1/4']],'y':['1/7','-1/9'],'lr_hidden':'1/10','lr_output':'1/20'}
    with ctx.workprec(256):
        model=TwoRateCuspMap(inputs);oldrates=model.rates;n=model.n;x=[exact(str(i)+'/20')for i in range(n)];q=[exact('1/3')]*n;r=[exact('-1/7')]*n
        v,j,data=augmented_chain(model,x+[exact('1/10'),exact('1/20')]+q+r,3)
        state=x;mon=eye(n);original=BallTanhMap(inputs)
        for _ in range(3):state,derivative=original.step(state);mon=derivative*mon
        assert model.rates is oldrates
        for i in range(n):
            assert v[i,0].overlaps(state[i]-x[i])
            assert v[n+i,0].overlaps(sum(((mon[i,k]-int(i==k))*q[k]for k in range(n)),arb(0)))
            assert v[2*n+i,0].overlaps(sum(((mon[i,k]-int(i==k))*r[k]for k in range(n)),arb(0))-q[i])
            for k in range(n):assert data['monodromy'][i,k].overlaps(mon[i,k])


def test_semisimple_and_simple_one_do_not_satisfy_chain():
    class Identity:
        n=2
        def cusp_step(self,x,a,b):return x
    class Simple:
        n=2
        def cusp_step(self,x,a,b):return [x[0],x[1]*.5]
    for model in (Identity(),Simple()):
        v,_,_=augmented_chain(model,point(),1)
        assert not v[4,0].contains(0)


def test_invalid_dimensions_period_settings_and_interrupt():
    with pytest.raises(ValueError):augmented_chain(NormalForm(),[arb(0)],1)
    for period in (0,65,True):
        with pytest.raises(ValueError):augmented_chain(NormalForm(),point(),period)
    for keyword in ({'newton_steps':True},{'radius_exponent':True}):
        with pytest.raises(ValueError):chain_root_proof(NormalForm(),point(),1,**keyword)
    def stop():raise TimeoutError('synthetic stop')
    with pytest.raises(TimeoutError):augmented_chain(NormalForm(),point(),1,stop)
