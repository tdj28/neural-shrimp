"""Synthetic controls of flip AD, signs and proof gates. No selected target data."""
from fractions import Fraction
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_flips import FlipJet,FlipTanhMap,augmented,_proof,check_flip
from neural_shrimp.rigorous_orbits import BallTanhMap,exact


def restore(b):return arb(arb(tuple(b['mid'])),arb(tuple(b['rad'])))


def test_exact_product_third_and_mixed_derivatives():
    # x(t)=2+3t, y(t)=5+7t; x^2*y=20+88t+129t²+63t³.
    x=FlipJet.variable(arb(2),2,0,arb(3));y=FlipJet.variable(arb(5),2,1,arb(7))
    f=x*x*y
    assert f.f==20 and f.d==88 and f.d2==258 and f.d3==378
    assert list(f.g)==[20,4] and list(f.m)==[58,12]


def test_tanh_and_exp_third_chain():
    with ctx.workprec(200):
        x=FlipJet.variable(arb(1)/3,1,0,arb(2));f=(x*x).tanh()
        u=x.f*x.f;t=u.tanh();s=1-t*t
        d=4*x.f;d2=arb(8)
        expected=3*(-2*t*s)*d*d2+s*(6*t*t-2)*d*d*d
        assert f.d3.overlaps(expected)
        e=(x*x).exp()
        assert e.d3.overlaps(u.exp()*(3*d*d2+d*d*d))
        assert e.m[0].overlaps(u.exp()*(4+8*x.f*x.f))


def inputs(bias=True):
    return {'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':1,'bias':bias,'in_scale':'3/2','out_scale':'2/3','loss_scale':'3/4','ridge':'1/20'},'x':[['1/3','-1/4'],['-2/5','1/2']],'y':['1/7','-2/9'],'lr_hidden':'1/8','lr_output':'1/9'}


def family():
    return {'schema':'neural-shrimp-exact-log-ray-v1','ell':['-1','-3/2'],'direction':['1/4','-2/7']}


@pytest.mark.parametrize('bias',[False,True])
def test_two_rate_map_and_jacobian_against_existing_formula(bias):
    with ctx.workprec(256):
        inp=inputs(bias);new=FlipTanhMap(inp,family());old=BallTanhMap(inp);n=old.n
        s=arb(1)/5;rates=new.rates_at(s);old.rates=[rates[0]]*old.hidden+[rates[1]]*(n-old.hidden)
        x=[arb(i+1)/17 for i in range(n)];q=[arb(i+2)/23 for i in range(n)]
        h,j,data=augmented(new,x+[s]+q,1);end,m=old.step(x)
        for i in range(n):
            assert (h[i]+x[i]).overlaps(end[i])
            for k in range(n):
                assert j[i,k].overlaps(m[i,k]-int(i==k))
                assert j[n+i,n+1+k].overlaps(m[i,k]+int(i==k))
            d=exact(family()['direction'][int(i>=old.hidden)])
            expected=(end[i]-x[i])*arb(10).log()*d
            assert data['parameter_derivative'][i].overlaps(expected)
        # Both layer parameter derivatives must be genuinely nonzero here.
        assert any(not data['parameter_derivative'][i].contains(0) for i in range(old.hidden))
        assert any(not data['parameter_derivative'][i].contains(0) for i in range(old.hidden,n))


def test_composed_third_and_mixed_columns_independent_high_precision_differences():
    # Test only: old separately written analytic Jacobian; no FD in proof gates.
    with ctx.workprec(384):
        inp=inputs();new=FlipTanhMap(inp,family());n=new.n
        x=[arb(i+1)/17 for i in range(n)];q=[arb(i+2)/23 for i in range(n)];s=arb(1)/5
        _,j,data=augmented(new,x+[s]+q,2);eps=arb((1,-36));tol=arb((1,-60))
        def action(xx,ss):
            old=BallTanhMap(inp);rates=new.rates_at(ss);old.rates=[rates[0]]*old.hidden+[rates[1]]*(n-old.hidden)
            _,m,_,_=old.iterate(xx,2)
            return [sum((m[i,k]*q[k] for k in range(n)),arb(0)) for i in range(n)]
        zero=action(x,s);plus=action([a+eps*b for a,b in zip(x,q)],s);minus=action([a-eps*b for a,b in zip(x,q)],s)
        for i in range(n):
            assert abs((plus[i]-minus[i])/(2*eps)-data['second_directional'][i])<tol
            assert abs((plus[i]-2*zero[i]+minus[i])/(eps*eps)-data['third_directional'][i])<tol
            assert data['second_directional'][i].overlaps(sum((data['mixed_directional'][i,k]*q[k] for k in range(n)),arb(0)))
        for col in range(n+1):
            xx=list(x);ss=s
            if col<n:xx[col]+=eps
            else:ss+=eps
            plus=action(xx,ss);xx=list(x);ss=s
            if col<n:xx[col]-=eps
            else:ss-=eps
            minus=action(xx,ss)
            for i in range(n):assert abs((plus[i]-minus[i])/(2*eps)-j[n+i,col])<tol


class PolynomialFlip:
    n=2
    def __init__(self,a=0,c=1,shift=0,nonnormal=False,unstable=False,tangent=False):
        self.a,self.c,self.shift,self.nonnormal,self.unstable,self.tangent=a,c,shift,nonnormal,unstable,tangent
    def jet_step(self,x,s):
        # x=P(u+shift*s,y), P=[[1,2],[0,1]] if nonnormal.
        r=x[0]-(2*x[1] if self.nonnormal else 0);u=r-self.shift*s
        rate=s*s if self.tangent else s
        f=self.shift*s-(1+rate)*u+self.a*u*u+self.c*u*u*u
        stable=x[1]*(arb(3)/2 if self.unstable else arb(1)/2)
        return [f+(2*stable if self.nonnormal else 0),stable]


def proof(model,period=1):
    return _proof(model,[arb(0),arb(0),arb(0),arb(1),arb(0)],period,-80,0,128)


def test_known_supercritical_flip_and_doubled_side():
    with ctx.workprec(256):
        r=proof(PolynomialFlip())
        assert r['passed'] and r['criticality']=='supercritical' and r['doubled_side_in_s']=='greater'
        assert restore(r['alpha']).contains(-1) and restore(r['cubic_b']).contains(1)
        assert r['complement_witness_power']==1


def test_quadratic_elimination_changes_cubic_sign():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(a=2,c=-3))
        assert r['passed'] and r['criticality']=='supercritical'
        assert restore(r['third_directional'][0]).contains(-18)
        assert restore(r['cubic_b']).contains(1) # c+a², not raw cubic c.


def test_nonnormal_left_vector_and_moving_branch_total_crossing():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(a=2,c=-3,shift=3,nonnormal=True))
        assert r['passed']
        assert restore(r['left_vector'][0]).contains(1) and restore(r['left_vector'][1]).contains(-2)
        assert restore(r['orbit_branch_derivative'][0]).contains(3)
        # Partial derivative at fixed state is -13; total branch derivative -1.
        assert restore(r['mixed_directional'][0][2]).contains(-13)
        assert restore(r['alpha']).contains(-1) and restore(r['cubic_b']).contains(1)


def test_subcritical_sign_and_opposite_doubled_side():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(c=-1))
        assert r['passed'] and r['criticality']=='subcritical' and r['doubled_side_in_s']=='less'


def test_zero_cubic_does_not_pass():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(a=2,c=-4))
        assert r['gates']['augmented_root_unique'] and r['gates']['total_multiplier_crossing']
        assert not r['gates']['cubic_nondegeneracy'] and not r['passed']


def test_unstable_complement_does_not_pass():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(unstable=True))
        assert r['gates']['augmented_root_unique'] and r['gates']['cubic_nondegeneracy']
        assert not r['gates']['other_modes_attracting'] and not r['passed']


def test_period_multiple_excluded_even_with_flip_of_iterate():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(),period=3)
        assert r['gates']['augmented_root_unique']
        assert [d['divisor'] for d in r['divisors']]==[1]
        assert not r['gates']['primitive_period'] and not r['passed']


def test_zero_crossing_augmented_root_unproved():
    with ctx.workprec(256),pytest.raises(ZeroDivisionError):proof(PolynomialFlip(tangent=True))


def test_defective_minus_one_cannot_pass():
    class Defective:
        n=2
        def jet_step(self,x,s):return [-(1+s)*x[0]+x[1],-x[1]]
    with ctx.workprec(256),pytest.raises(ZeroDivisionError):proof(Defective())


def test_deadline_propagates_and_invalid_settings_fail_closed():
    def stop():raise TimeoutError('synthetic timeout')
    with pytest.raises(TimeoutError):augmented(PolynomialFlip(),[arb(0)]*3+[arb(1),arb(0)],1,stop)
    with pytest.raises(ValueError):check_flip({'schema':'neural-shrimp-flip-input-v1','period':18},precision=True)


def test_invalid_family_rejected():
    with pytest.raises(ValueError):FlipTanhMap(inputs(),{'schema':'neural-shrimp-exact-log-ray-v1','ell':['0','0'],'direction':['0','0']})


def test_phase_frame_linear_conjugacy_normalization():
    # A cheap exact phase-frame control: T=[[2,1],[0,3]], k=||Tq||=2.
    # This checks normalization covariance, not a new selected orbit phase.
    base=PolynomialFlip(a=2,c=-3,shift=3,nonnormal=True)
    class Conjugate:
        n=2
        def jet_step(self,x,s):
            u=[x[0]*(arb(1)/2)-x[1]*(arb(1)/6),x[1]*(arb(1)/3)]
            f=base.jet_step(u,s)
            return [2*f[0]+f[1],3*f[1]]
    with ctx.workprec(256):
        old=proof(base);new=proof(Conjugate())
        assert old['passed'] and new['passed']
        assert restore(new['alpha']).overlaps(restore(old['alpha']))
        assert restore(new['cubic_b']).overlaps(restore(old['cubic_b'])/4)
        assert restore(new['left_vector'][0]).contains(1)
        assert restore(new['left_vector'][1]).contains(-arb(5)/3)
        assert restore(new['orbit_branch_derivative'][0]).contains(6)


def test_unproved_coefficient_sign_is_not_labeled_criticality():
    with ctx.workprec(256):
        r=proof(PolynomialFlip(unstable=True))
        assert r['cubic_sign']=='positive'
        assert r['criticality']=='unproved' and r['doubled_side_in_s']=='unproved'
