from fractions import Fraction
import numpy as np
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair import (entire_pair,terminal_matrix,phase,assemble,
    proposal,native_lift,unit_pair_tube,anchor_test)
from neural_shrimp.rigorous_orbits import exact,dyadic,eye,norm_inf_upper
from neural_shrimp.rigorous_generalized_chain import augmented_chain

class Polynomial:
    n=2
    def cusp_step(self,z,h,o):
        x,y=z
        return [x+y,y+(h-1)+x*x*exact('1/2')+(o-1)*y+2*x*y]

class Primitive:
    n=3
    def cusp_step(self,z,h,o):
        x,y,w=z
        return [x+y*exact('1/2'),y+(h-1)*exact('1/2')+x*x*exact('1/2')+(o-1)*y*exact('1/2')+2*x*y,
                w*exact('-3/2')+w*w*w*exact('1/2')]

def floats(m):return np.array([[float(m[i,j])for j in range(m.ncols())]for i in range(m.nrows())])
def arr(z):return [exact(dyadic(v))for v in z]
def chain(model):
    n=model.n;x=[arb(0)]*n
    if n==3:x[-1]=arb(1)
    return x+[arb(1),arb(1)]+[arb(1)]+[arb(0)]*(n-1)+[arb(0),arb(1)]+[arb(0)]*(n-2)

def settings(model,period):
    flat,lift=native_lift(model,chain(model),period)
    return [str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in flat],dict(
        period=period,parameter='1/1024',halfwidth_exponent=-10,radius_exponent=-12,newton_steps=2,
        anchor={'box':flat,'dimension':model.n,'period':period,'parameter':'0'})

@pytest.mark.parametrize('t',['0','1/4','-1/4','1/16384','-1/16384'])
def test_entire_series_and_two_derivatives(t):
    with ctx.workprec(256):
        tt=exact(t);c,e,d=entire_pair(tt)
        if tt>0:
            a=tt.sqrt();assert c.contains(a.cos())and e.contains(a.sin()/a)
        elif tt<0:
            a=(-tt).sqrt();assert c.contains(a.cosh())and e.contains(a.sinh()/a)
        else:assert c.contains(1)and e.contains(1)and d.is_zero()
        cp,ep,dp=entire_pair(tt,1);cpp,epp,dpp=entire_pair(tt,2)
        assert (2*cp+e).contains(0)
        assert (2*tt*ep+e-c).contains(0)
        assert (2*tt*epp+3*ep-cp).contains(0)
        assert (2*cpp+ep).contains(0)
        assert (dp-e-tt*ep).contains(0)and(dpp-2*ep-tt*epp).contains(0)
        assert (c*c+tt*e*e).contains(1)

def test_entire_interval_spans_zero_and_invalid_inputs():
    with ctx.workprec(256):
        interval=arb(0,exact('1/8'));c,e,d=entire_pair(interval)
        for v in ['-1/8','0','1/8']:
            cc,ee,dd=entire_pair(exact(v));assert c.contains(cc)and e.contains(ee)and d.contains(dd)
        with pytest.raises(ValueError):entire_pair(exact('1/2'))
        with pytest.raises(ValueError):entire_pair(arb(0),3)

def test_same_zero_equations_as_generalized_chain_full_permutation():
    with ctx.workprec(256):
        m=Polynomial();z=chain(m);z=[v+exact(str(i+1)+'/100')for i,v in enumerate(z)]
        v,j,_=augmented_chain(m,z,1)
        order=[0,1,4,5,6,7,2,3];flat=[z[i]for i in order]
        w,k,_,ht=assemble(m,flat,1,arb(0))
        for i in range(8):
            assert w[i,0].contains(v[i,0])
            for l in range(8):assert k[i,l].contains(j[i,order[l]])
        assert ht.nrows()==8

@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_all_native_jacobian_columns_parameter_and_condensed_inverse(model,period):
    with ctx.workprec(256):
        ini,kw=settings(model,period);z=np.array([float(Fraction(v))for v in ini]);t=exact('1/1024')
        g,j,local,ht=assemble(model,arr(z),period,t)
        for k in range(len(z)):
            perturb=np.eye(len(z))[k]*1e-6
            fd=(floats(assemble(model,arr(z+perturb),period,t)[0])-floats(assemble(model,arr(z-perturb),period,t)[0]))[:,0]/2e-6
            np.testing.assert_allclose(fd,floats(j)[:,k],atol=5e-10,rtol=1e-7)
        eps=exact('1/1000000')
        fd=(floats(assemble(model,arr(z),period,t+eps)[0])-floats(assemble(model,arr(z),period,t-eps)[0]))[:,0]/2e-6
        np.testing.assert_allclose(fd,floats(ht)[:,0],atol=5e-10,rtol=1e-7)
        pre,small=proposal(j,local,model.n,t)
        assert small.nrows()==3*model.n+2
        assert norm_inf_upper(eye(len(z))-pre*j)<arb('1e-55')
        if period==2:assert local[0][0].det().contains(0)

@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_uniform_positive_interval_with_original_exact_jordan_anchor(model,period):
    with ctx.workprec(256):
        ini,kw=settings(model,period);r,data=unit_pair_tube(model,ini,**kw)
        assert r['passed'],r['gates']
        assert r['parameter_endpoints']==['0','1/512']
        assert all(r['coordinate_inclusions'])and all(r['anchor_check']['coordinate_inclusions'])
        if period==2:assert r['divisors'][0]['excluded']
        else:
            c,_,_=entire_pair(exact('1/512'));x=2*(c-1)
            assert data['endpoints'][1][0].contains(x)
            assert data['endpoints'][1][-2].contains(1-x*x/2)
            assert data['endpoints'][1][-1].contains(1-x)
        assert 'complementary_spectrum'not in r['gates']
        # First chart's rigorously returned endpoint is the entire next anchor.
        dd=exact('1/512');next_initial=[str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in [a+data['slope'][i,0]*dd for i,a in enumerate(data['center'])]]
        kw.update(parameter='3/1024',anchor={'box':data['endpoints'][1],'period':period,'dimension':model.n,'parameter':'1/512'})
        rr,_=unit_pair_tube(model,next_initial,**kw);assert rr['passed']

def test_lift_all_native_state_and_two_modes_then_wrong_anchor_fails():
    with ctx.workprec(256):
        m=Primitive();flat,lift=native_lift(m,chain(m),2)
        assert lift['closure_contains_zero']and len(flat)==20
        assert flat[6]==0 and flat[7]==1 and flat[15]==exact('1/2')and flat[16]==1
        ini,kw=settings(m,2);kw['anchor']['box'][16]+=1
        r,_=unit_pair_tube(m,ini,**kw);assert not r['passed']and not r['gates']['anchor_identified']

def test_repeated_cycle_is_not_primitive_even_if_full_root_passes():
    with ctx.workprec(256):
        m=Primitive();ini,kw=settings(m,2)
        # Four shooting phases have the generalized return chain r halved.
        z=chain(m);z[-3:]=[v/2 for v in z[-3:]];flat,_=native_lift(m,z,4)
        exactini=[str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in flat]
        kw.update(period=4,anchor={'box':flat,'dimension':3,'period':4,'parameter':'0'})
        r,_=unit_pair_tube(m,exactini,**kw)
        assert r['gates']['uniform_normalized_unit_pair']and not r['gates']['primitive_period']

def test_both_direction_calls_counted_and_failure_observed_before_rejection():
    calls=[];events=[]
    m=Polynomial();phase(m,[arb(0)]*2,[arb(1),arb(0)],[arb(0),arb(1)],[arb(1)]*2,
                          tick=lambda:calls.append(1),observe=lambda k,v:events.append((k,v)))
    assert len(calls)==2 and [v['direction']for k,v in events]==['q','r']
    class Bad(Polynomial):
        def cusp_step(self,*args):return [args[0][0]*arb('nan')]*2
    events=[]
    with pytest.raises(ValueError):phase(Bad(),[arb(0)]*2,[arb(1)]*2,[arb(1)]*2,[arb(1)]*2,observe=lambda k,v:events.append((k,v)))
    assert len(events)==1 and events[0][1]['value'][0][0]=={'finite':False}

def test_interrupt_precedes_actual_native_call_and_bad_rates_rejected():
    class Never(Polynomial):
        def cusp_step(self,*_):raise AssertionError('called')
    def stop():raise TimeoutError('owned cap')
    with pytest.raises(TimeoutError):phase(Never(),[arb(0)]*2,[arb(1)]*2,[arb(1)]*2,[arb(1)]*2,tick=stop)
    with pytest.raises(ValueError):phase(Never(),[arb(0)]*2,[arb(1)]*2,[arb(1)]*2,[arb(0)]*2)
