"""Independent polynomial/series/path controls; no empirical inputs."""
from fractions import Fraction
import pytest
from flint import arb,arb_mat,arb_series,ctx
from neural_shrimp.rigorous_eigenpair_path import PathJet,path_coefficients,quadratic_tube,quadratic_two_charts
from neural_shrimp.rigorous_eigenpair_locus import assemble,locus_tube
from neural_shrimp.rigorous_orbits import exact
from test_rigorous_eigenpair_locus import CoupledCycle,Cusp,settings


def test_product_mixed_top_coefficient_and_truncation():
    x=PathJet([1,2,3,4],[5,6,7,8]);y=PathJet([9,10,11,12],[13,14,15,16]);z=x*y
    assert z.a[3]==1*12+2*11+3*10+4*9
    assert z.b[3]==1*16+2*15+3*14+4*13+5*12+6*11+7*10+8*9
    t=PathJet([0,1,0,0]);e=PathJet([0]*4,[1,0,0,0])
    assert (t*t*t*e).b[3]==1 and all(v==0 for v in (t*t*t*t).a)
    assert all(v==0 for v in (e*e).a+(e*e).b)


@pytest.mark.parametrize('mid',['0','2/7','-3/5'])
def test_tanh_all_eight_coefficients_against_independent_series(mid):
    with ctx.workprec(256):
        a=[exact(mid),exact('3/7'),exact('-2/9'),exact('1/11')];b=[exact('4/5'),exact('-1/8'),exact('2/13'),exact('3/17')]
        j=PathJet(a,b).tanh();aa=arb_series(a,4);bb=arb_series(b,4);ex=(2*aa).exp();t=(ex-1)/(ex+1);tb=(1-t*t)*bb
        for i in range(4):
            assert j.a[i].overlaps(t[i]) and j.b[i].overlaps(tb[i])
            assert abs(j.a[i]-t[i]).upper()<arb('1e-65') and abs(j.b[i]-tb[i]).upper()<arb('1e-65')


def test_interval_jet_contains_point_specializations():
    with ctx.workprec(256):
        wide=PathJet([arb(exact('1/3'),exact('1/100')),arb(2),arb(-1),arb(0)],[arb(exact('2/3'),exact('1/100')),arb(1),arb(0),arb(0)]).tanh()
        for c in ('97/300','1/3','103/300'):
            for q in ('197/300','2/3','203/300'):
                p=PathJet([exact(c),2,-1,0],[exact(q),1,0,0]).tanh()
                assert all(wide.a[i].contains(p.a[i]) and wide.b[i].contains(p.b[i]) for i in range(4))


class CoupledPolynomial:
    n=2
    def cusp_step(self,z,h,o):
        x,y=z;return [x+x*x*y+h*x*y+o*x*x*x,y+x*y*y+h*o*x]


def test_full_path_coefficients_against_independent_series_and_gradient():
    with ctx.workprec(256):
        z=[exact(str(Fraction(i+1,7))) for i in range(6)];s=[exact(str(Fraction((-1)**i*(i+1),11))) for i in range(6)];t=[exact(str(Fraction(i-2,13))) for i in range(6)]
        nn=[arb(0)]*5+[arb(1)];d=exact('1/17');sc=exact('1/19')
        coeff=path_coefficients(CoupledPolynomial(),z,s,t,d,1,-1,nn,z,sc)
        v=[arb_series([z[i]+s[i]*d+t[i]*d*d/2,s[i]+t[i]*d,t[i]/2,0],4) for i in range(6)]
        x,y,q,r,h,o=v
        f=x+x*x*y+h*x*y+o*x*x*x;g=y+x*y*y+h*o*x
        fq=(1+2*x*y+h*y+3*o*x*x)*q+(x*x+h*x)*r
        gq=(y*y+h*o)*q+(1+2*x*y)*r
        reference=[f-x,g-y,fq+q,gq+r,(q*q+r*r-1)/2,o-z[-1]-arb_series([sc+d,1],4)]
        for i,ref in enumerate(reference):
            for k in range(4):assert coeff[i,k].overlaps(ref[k]) and abs(coeff[i,k]-ref[k]).upper()<arb('1e-60')
        # Independent old full Jacobian checks the linear coefficient at d=0.
        old,j,_=assemble(CoupledPolynomial(),z,1,-1,nn,z,sc)
        at0=path_coefficients(CoupledPolynomial(),z,s,t,arb(0),1,-1,nn,z,sc)
        js=j*arb_mat([[v] for v in s]);js[js.nrows()-1,0]-=1
        assert all(at0[i,0].overlaps(old[i,0]) and at0[i,1].overlaps(js[i,0]) for i in range(6))


@pytest.mark.parametrize('sigma',[-1,1])
def test_primitive_coupled_two_charts_native_joins_and_call_count(sigma):
    with ctx.workprec(256):
        model=CoupledCycle(sigma);ini,kw=settings(model,sigma=sigma);calls=[];events=[]
        r=quadratic_two_charts(model,ini,**kw,tick=lambda:calls.append(1),observe=lambda k,v:events.append(k))
        assert r['passed'] and len(r['charts'])==2 and len(calls)==2*(7*2+1)
        assert events.count('path_phase_return')==12 and events.count('path_assembly_return')==6
        assert r['charts'][0]['parameter_endpoints'][1]==r['charts'][1]['parameter_endpoints'][0]
        assert all(c['anchor_check']['passed'] and c['divisors'][0]['excluded'] for c in r['charts'])


def test_quadratic_path_improves_known_cusp_remainder_without_omitting_defects():
    with ctx.workprec(256):
        ini=['0','1','1','1'];kw=dict(period=1,sigma=1,normal=['1','0','0','0'],origin=ini,parameter='0',halfwidth_exponent=-6,radius_exponent=-16,anchor={'box':[exact(v) for v in ini],'period':1,'sigma':1,'dimension':1})
        old,_=locus_tube(Cusp(),ini,**kw);new,_=quadratic_tube(Cusp(),ini,**kw)
        assert not old['gates']['uniform_normalized_eigenpair_locus'] and new['passed']
        # Exact branch rates h=1+2*x^3,o=1-3*x^2: proposed rate curvatures0,-6.
        from neural_shrimp.rigorous_eigenpair_locus import exact_mid
        assert new['curvature'][-2]['mid'][0]==0
        assert exact(str(new['curvature'][-1]['mid'][0]))*arb((1,new['curvature'][-1]['mid'][1]))==-6
        assert 'quadratic_center_coefficients' in new and 'quadratic_whole_coefficients' in new


def test_center_defects_are_retained_for_inexact_proposal():
    with ctx.workprec(256):
        ini=['1/1024','1','1','1'];kw=dict(period=1,sigma=1,normal=['1','0','0','0'],origin=ini,parameter='0',halfwidth_exponent=-12,radius_exponent=-16,newton_steps=0)
        r,_=quadratic_tube(Cusp(),ini,**kw)
        assert any(v[0]['mid'][0]!=0 for v in r['quadratic_center_coefficients'])
        assert not r['passed'] and not r['gates']['anchor_identified']


def test_path_failing_return_and_interrupt_are_retained():
    class Bad:
        n=1
        def cusp_step(self,z,h,o):return [z[0]*arb('nan')]
    z=[arb(0),arb(1),arb(1),arb(1)];events=[]
    with pytest.raises(ValueError,match='finite path return'):
        path_coefficients(Bad(),z,[arb(0)]*4,[arb(0)]*4,arb(0),1,1,[arb(0)]*3+[arb(1)],z,arb(0),observe=lambda k,v:events.append((k,v)))
    assert events[0][0]=='path_phase_return' and events[0][1]['returns'][0][0][0]=={'finite':False}
    class Never:
        n=1
        def cusp_step(self,*a):raise AssertionError('must not evaluate')
    def stop():raise TimeoutError('before call')
    with pytest.raises(TimeoutError):path_coefficients(Never(),z,[arb(0)]*4,[arb(0)]*4,arb(0),1,1,[arb(0)]*3+[arb(1)],z,arb(0),tick=stop)


def test_biased_tanh_model_direction_and_parameter_cross_terms():
    from neural_shrimp.rigorous_cusps import TwoRateCuspMap
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'d':1,'width':1,'bias':True,'activation':'tanh','in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[['1/3'],['-1/4']],'y':['1/7','-1/9'],'lr_hidden':'1/10','lr_output':'1/20'}
    with ctx.workprec(256):
        model=TwoRateCuspMap(inp);n=model.n;z=[exact(str(Fraction(i,20))) for i in range(n)]+[exact('1/3')]*n+[exact('1/10'),exact('1/20')]
        ss=[exact(str(Fraction(i-3,30))) for i in range(len(z))];tt=[exact('1/7')]*len(z);nn=[arb(0)]*(len(z)-1)+[arb(1)]
        pc=path_coefficients(model,z,ss,tt,arb(0),1,1,nn,z,arb(0));g,j,_=assemble(model,z,1,1,nn,z,arb(0));js=j*arb_mat([[v] for v in ss]);js[js.nrows()-1,0]-=1
        for i in range(len(z)):assert pc[i,0].overlaps(g[i,0]) and pc[i,1].overlaps(js[i,0])


def test_interval_third_path_coefficients_contain_independent_point_samples():
    with ctx.workprec(256):
        z=[exact(str(Fraction(i+1,8))) for i in range(6)];s=[exact('1/8')]*6;t=[exact('-1/16')]*6;nn=[arb(0)]*5+[arb(1)]
        wide=path_coefficients(CoupledPolynomial(),z,s,t,arb(0,exact('1/64')),1,1,nn,z,arb(0))
        for d in [exact('-1/64'),arb(0),exact('1/64')]:
            point=path_coefficients(CoupledPolynomial(),z,s,t,d,1,1,nn,z,arb(0))
            assert all(wide[i,3].contains(point[i,3]) for i in range(6))


def test_synthetic_profile_caps_counts_and_invalid_cost_fail_closed():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'tools/profile_eigenpair_path.py';spec=importlib.util.spec_from_file_location('pathprofile',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    out=p.projection([1.,1.],[.001]*12,[.002]*12,100000,1000)
    assert out['admitted'] and out['two_chart_seconds']==2*48.10717263398692+10
    for v in (-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError):p.projection([v,1.],[.001]*12,[.002]*12,100000,1000)
    assert not p.projection([60.,60.],[.001]*12,[.002]*12,100000,1000)['admitted']
    assert not p.projection([1.,1.],[.001]*12,[.002]*12,2*1024**2,1000)['admitted']
    assert 'native_calls' not in out and len(p.SOURCES)==len(set(p.SOURCES))
