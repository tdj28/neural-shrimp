from fractions import Fraction
import numpy as np
import pytest
from flint import arb, arb_mat, ctx
from neural_shrimp.rigorous_eigenpair_locus import (
    assemble, proposal, phase, locus_tube, two_charts, exact_mid)
from neural_shrimp.rigorous_orbits import exact, dyadic, eye, norm_inf_upper


class CoupledCycle:
    """Nontrivially conjugated scalar primitive2, critical product sigma."""
    n=2
    def __init__(self,sigma=1,other='1/2'):
        self.sigma=sigma; self.other=exact(other)
    def cusp_step(self,z,h,o):
        x,y=z[0]+z[1]/2 if not hasattr(z[1], 'g') else z[0]+z[1]*exact('1/2'),z[1]
        # Values f(1)=-1,f(-1)=1; f'(1)=2,f'(-1)=sigma/2.
        c=exact('3/8') if self.sigma==1 else exact('5/8')
        d=exact('9/8') if self.sigma==1 else exact('7/8')
        f=-c+(-1-d)*x+c*x*x+d*x*x*x+(h-1)+(o-1)*x
        yy=self.other*y
        return [f-yy*exact('1/2'),yy]


def settings(model,period=2,sigma=1):
    initial=['1','0','1','0','-1','0','2','0','1','1']
    normal=['0']*len(initial);normal[-1]='1'
    return initial,dict(period=period,sigma=sigma,normal=normal,origin=initial,parameter='0',
        halfwidth_exponent=-14,radius_exponent=-16,newton_steps=2,
        anchor={'box':[exact(v) for v in initial],'period':period,'sigma':sigma,'dimension':model.n})


def floats(a):return np.array([[float(a[i,j]) for j in range(a.ncols())] for i in range(a.nrows())])


@pytest.mark.parametrize('sigma',[-1,1])
def test_coupled_native_shooting_and_all_jacobian_columns(sigma):
    with ctx.workprec(256):
        m=CoupledCycle(sigma);ini,kw=settings(m,sigma=sigma)
        z=np.array([float(Fraction(v)) for v in ini]);normal=[exact(v) for v in kw['normal']];origin=[exact(v) for v in ini]
        def get(z):return assemble(m,[exact(dyadic(v)) for v in z],2,sigma,normal,origin,arb(0))
        r,j,local=get(z)
        assert all(r[i,0].is_zero() for i in range(len(z)))
        for k in range(len(z)):
            step=np.eye(len(z))[k]*1e-6
            fd=(floats(get(z+step)[0])-floats(get(z-step)[0]))[:,0]/2e-6
            np.testing.assert_allclose(fd,floats(j)[:,k],rtol=1e-7,atol=2e-9)
        c,small=proposal(j,local,sigma)
        assert small.nrows()==6
        assert norm_inf_upper(eye(len(z))-c*j)<arb('1e-60')
        exact_inv=j.inv()
        assert max(abs(float((c-exact_inv)[i,k])) for i in range(len(z)) for k in range(len(z)))<1e-60


@pytest.mark.parametrize('sigma',[-1,1])
def test_two_primitive_coupled_charts_and_exact_endpoint_join(sigma):
    with ctx.workprec(256):
        m=CoupledCycle(sigma);ini,kw=settings(m,sigma=sigma)
        r=two_charts(m,ini,**kw)
        assert r['passed'] and len(r['charts'])==2
        assert all(c['gates']['primitive_period'] and c['divisors'][0]['excluded'] for c in r['charts'])
        assert r['charts'][0]['parameter_endpoints'][1]==r['charts'][1]['parameter_endpoints'][0]
        assert 'simple critical multiplier' in r['charts'][0]['not_claimed']


def test_condensation_does_not_invert_singular_native_jacobian():
    with ctx.workprec(256):
        m=CoupledCycle(1,'0');ini,kw=settings(m)
        z=[exact(v) for v in ini]
        _,j,local=assemble(m,z,2,1,[exact(v) for v in kw['normal']],z,arb(0))
        assert local[0][0].det().is_zero()
        c,_=proposal(j,local,1)
        assert norm_inf_upper(eye(len(z))-c*j)<arb('1e-60')


class Cusp:
    n=1
    def cusp_step(self,z,h,o):
        x=z[0];return [x+(h-1)+(o-1)*x+x*x*x]


def test_affine_chart_can_cross_cusp_without_cubic_or_fold_gate():
    with ctx.workprec(256):
        ini=['0','1','1','1'];normal=['1','0','0','0']
        r,_=locus_tube(Cusp(),ini,period=1,sigma=1,normal=normal,origin=ini,parameter='0',
            halfwidth_exponent=-12,radius_exponent=-16,
            anchor={'box':[exact(v) for v in ini],'period':1,'sigma':1,'dimension':1})
        assert r['passed']
        # Actual fold locus h=1+2x^3,o=1-3x^2 has zero projected rate tangent at cusp.
        assert all(v['mid'][0]==0 for v in r['slope'][-2:][0])
        assert r['genericity'].startswith('not evaluated')


class OtherMode:
    n=2
    def __init__(self,other):self.other=other
    def cusp_step(self,z,h,o):return [z[0]+(h-1)+z[0]*z[0],(self.other+o-1)*z[1]]


def test_complement_minus_one_crossing_allowed_but_rank_loss_exposed():
    with ctx.workprec(256):
        ini=['0','0','1','0','1','1'];kw=dict(period=1,sigma=1,normal=['0','0','0','0','0','1'],origin=ini,parameter='0',
            halfwidth_exponent=-12,radius_exponent=-16,anchor={'box':[exact(v) for v in ini],'period':1,'sigma':1,'dimension':2})
        r,_=locus_tube(OtherMode(-1),ini,**kw)
        assert r['passed']
        with pytest.raises((ZeroDivisionError,ValueError)):
            locus_tube(OtherMode(1),ini,**kw)


def test_wrong_native_phase_or_critical_sign_anchor_rejected():
    with ctx.workprec(256):
        m=CoupledCycle();ini,kw=settings(m)
        kw['anchor']['sigma']=-1
        with pytest.raises(ValueError,match='anchor convention'):locus_tube(m,ini,**kw)
        _,kw=settings(m);kw['anchor']['box'][0]=arb(-1)
        r=two_charts(m,ini,**kw)
        assert not r['passed'] and len(r['charts'])==1 and not r['charts'][0]['anchor_check']['passed']


def test_repeated_native_cycle_excluded_even_when_locus_root_passes():
    with ctx.workprec(256):
        m=CoupledCycle();ini,kw=settings(m)
        expanded=ini[:-2]*2+ini[-2:];kw.update(period=4,normal=['0']*(len(expanded)-1)+['1'],origin=expanded)
        kw['anchor'].update(period=4,box=[exact(v) for v in expanded])
        r,_=locus_tube(m,expanded,**kw)
        assert r['gates']['uniform_normalized_eigenpair_locus']
        assert not r['gates']['primitive_period']


def test_phase_return_observed_before_nonfinite_rejection():
    class Bad:
        n=1
        def cusp_step(self,z,h,o):return [z[0]*arb('nan')]
    events=[]
    with pytest.raises(ValueError,match='finite local return'):
        assemble(Bad(),[arb(0),arb(1),arb(1),arb(1)],1,1,[arb(0)]*3+[arb(1)],[arb(0)]*4,arb(0),observe=lambda k,r:events.append((k,r)))
    assert events[0][0]=='phase_return' and events[0][1]['value'][0][0]=={'finite':False}


def test_interrupt_before_model_call_and_invalid_rate():
    called=[]
    class Never:
        n=1
        def cusp_step(self,*args):called.append(1);raise AssertionError
    def stop():raise TimeoutError('unit')
    with pytest.raises(TimeoutError):assemble(Never(),[arb(0),arb(1),arb(1),arb(1)],1,1,[arb(0)]*4,[arb(0)]*4,arb(0),tick=stop)
    assert not called
    with pytest.raises(ValueError,match='positive'):phase(Never(),[arb(0)],[arb(1)],[arb(0),arb(1)])


def test_exact_midpoint_and_precision_guard():
    with ctx.workprec(256):
        a=arb((3,-180));assert exact(exact_mid(a))==a
        m=CoupledCycle();ini,kw=settings(m)
        with pytest.raises(ValueError,match='precision'):locus_tube(m,ini,precision=53,**kw)


def test_two_actual_rate_tanh_adapter_and_exception_restores_rates():
    from neural_shrimp.rigorous_cusps import TwoRateCuspMap,augmented_cusp
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'d':1,'width':1,'bias':True,'activation':'tanh','in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[['1/3'],['-1/4']],'y':['1/7','-1/9'],'lr_hidden':'1/10','lr_output':'1/20'}
    with ctx.workprec(256):
        m=TwoRateCuspMap(inp);old=m.rates;n=m.n
        x=[exact(str(i)+'/20') for i in range(n)];q=[exact('1/3')]*n;rates=[exact('1/10'),exact('1/20')]
        v,t,e=phase(m,x,q,rates)
        value,j,data=augmented_cusp(m,x+rates+q+[arb(0)]*n,1)
        for i in range(n):
            assert v[i,0].overlaps(value[i,0]+x[i])
            assert v[n+i,0].overlaps(value[n+i,0]+q[i])
            for k in range(n):
                assert t[i,k].overlaps(j[i,k]+int(i==k))
                assert t[n+i,k].overlaps(j[n+i,k])
            for k in range(2):
                assert e[i,k].overlaps(j[i,n+k]);assert e[n+i,k].overlaps(j[n+i,n+k])
        assert m.rates is old


def test_profile_has_no_target_mode_and_strict_resource_projection():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'tools/profile_eigenpair_locus.py'
    spec=importlib.util.spec_from_file_location('eigenprofile',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.projection([1.,1.],[.001]*12,100000,1000)['resource_admitted']
    assert not m.projection([100.,100.],[.001]*12,100000,1000)['resource_admitted']
    for invalid in (-1.,0.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):m.projection([invalid,1.],[.001]*12,100000,1000)
    for required in ('src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_cusps.py'):
        assert required in m.SOURCES
