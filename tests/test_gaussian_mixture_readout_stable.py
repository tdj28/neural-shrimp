"""Pure synthetic recovery tests; never load saved target arrays or scores."""
from pathlib import Path
import sys

import numpy as np
import pytest
from flint import arb,ctx
from scipy.special import ndtr

_checkout=Path(__file__).resolve().parents[1]
_tools=_checkout/'tools'
if not (_tools/'gaussian_mixture_readout.py').is_file():
    _tools=Path('/data2/neural-shrimp/tools')  # External pre-freeze development only.
sys.path.insert(0,str(_tools))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gaussian_mixture_readout as original
import gaussian_mixture_readout_stable as stable


def vector(v):return np.array(v,dtype=np.float64)


def score(a,b,c,mt,mz,sign=1):
    d=stable.define_classifier(vector([a,c]),vector([b,0]),1.)
    return stable.probability(d,vector([mt,mz]),sign)


@pytest.mark.parametrize('b',[1e-2,1e-4,1e-7,1e-10])
@pytest.mark.parametrize('ratio',[1e-2,1e-4,1e-6])
def test_small_signal_across_orders_of_magnitude(b,ratio):
    a=.05*b;c=ratio*a;mt=.002;mz=-.01
    r=score(a,b,c,mt,mz)
    assert r['complete'] and r['qualified'],r
    assert r['interval'][1]-r['interval'][0]<=1e-4
    assert all(x['maximum_point_absolute_error']<=1e-15 for x in r['diagnostics']['nominal_runs'])
    assert abs(r['diagnostics']['nominal_runs'][0]['value']-r['diagnostics']['nominal_runs'][1]['value'])<=1e-8
    if b<=1e-4:
        weak_second_limit=ndtr((a*mt+c*mz)/np.hypot(a,c))
        assert abs(r['nominal']-weak_second_limit)<1e-9


@pytest.mark.parametrize('b',[1-1e-8,1.,1+1e-8])
def test_near_cancellation_b_equals_one(b):
    r=score(1e-6,b,1e-12,.02,-.3)
    assert r['qualified'],r
    assert r['diagnostics']['exact_rank']==2


def test_high_separation_and_precision_escalation():
    r=score(1e6,1e6,1e-30,0.,.1)
    assert r['qualified'],r
    assert r['diagnostics']['precision_bits'] in [256,512]
    assert abs(r['nominal']-ndtr(.1))<1e-12
    assert any('point-CDF' in a['error'] for a in r['diagnostics']['attempts'])


@pytest.mark.parametrize('a,b,c,mt,mz',[
    (1.,2.,.5,3.,.5),(.1,1.,.3,.9,.7),(.7,.4,.8,-.5,.2),(2.,1.,1.5,-1.,-.3)])
def test_original_certificate_unchanged_in_well_conditioned_cases(a,b,c,mt,mz):
    d=stable.define_classifier(vector([a,c]),vector([b,0]),1.)
    mean=vector([mt,mz]);prior=original.probability(d,mean,1);new=stable.probability(d,mean,1)
    assert prior['qualified'] and new['qualified']
    assert prior['interval']==new['interval']
    assert prior['diagnostics']['enclosure']==new['diagnostics']['enclosure']
    assert prior['diagnostics']['transition_roots']==new['diagnostics']['transition_roots']
    assert abs(prior['nominal']-new['nominal'])<1e-10


@pytest.mark.parametrize('A,B',[
    ([0,0],[1,0]),([1,0],[0,0]),([1,0],[0,2]),([1,0],[2,0]),([100,0],[100,0])])
def test_scalar_linear_orthogonal_definitions_and_results_identical(A,B):
    A=vector(A);B=vector(B);d=stable.define_classifier(A,B,1.)
    assert d['schema']==original.define_classifier(A,B,1.)['schema']
    assert stable.probability(d,vector([.3,.2]),1)==original.probability(d,vector([.3,.2]),1)
    assert stable.linear_probability(vector([1,0]),vector([.3,.2]),1,1.)==original.linear_probability(vector([1,0]),vector([.3,.2]),1,1.)


def test_declared_negative_one_mode_remains_exact_prior_recipe():
    B=vector([20,1]);A=vector([1,2]);d=stable.define_classifier(A,B,1.,one_mode_rho=-.3)
    assert d['branch']=='collinear'
    mean=[dict(coefficient=-.3,vector=B),dict(coefficient=1.,vector=B)]
    assert stable.probability(d,mean,1)==original.probability(d,mean,1)


def test_no_frozen_global_monkeypatch_or_precision_leak():
    prior=original._nominal;precision=ctx.prec
    assert score(5e-9,1e-7,5e-13,.002,-.01)['qualified']
    assert original._nominal is prior and ctx.prec==precision


def test_point_error_exhaustion_retains_failed_precision_attempts(monkeypatch):
    monkeypatch.setattr(stable,'POINT_ERROR_LIMIT',0.)
    r=score(1.,2.,.5,3.,.5)
    assert not r['complete'] and not r['qualified'] and r['interval'] is not None
    assert len(r['diagnostics']['attempts'])==3
    assert all('point-CDF error budget' in a['error'] for a in r['diagnostics']['attempts'])
    assert all(a['diagnostics']['failed_point_run']['point_attempts']>=1
               for a in r['diagnostics']['attempts'])


def test_synthetic_old_jitter_failure_is_repaired_without_threshold_relaxation():
    A=vector([5e-6,5e-8]);B=vector([1e-4,0]);mu=vector([.002,-.01])
    d=original.define_classifier(A,B,1.)
    old=original.probability(d,mu,1);new=stable.probability(d,mu,1)
    assert not old['qualified'] and old['interval'] is not None
    assert new['qualified']
    # Old nominal failure continued through512bits; new success stops at128.
    # Compare the certificate at the same precision, not two ladder endpoints.
    same_precision=next(a for a in old['diagnostics']['attempts']
                        if a['precision_bits']==new['diagnostics']['precision_bits'])
    assert new['interval']==same_precision['partial_interval']
    assert new['diagnostics']['nominal_runs'][0]['epsabs']==1e-9
    assert new['diagnostics']['nominal_runs'][1]['epsabs']==1e-11
    for name in ['probability_width','quad_epsabs','quad_limits','quad_agreement','quad_max_error','precision_bits']:
        assert stable.contract()[name]==original.contract()[name]


def test_direct_point_calculation_removes_binary64_log2_jitter():
    with ctx.workprec(128):
        coefficients=list(map(original._ball,[5e-9,1e-7,5e-13,0.,0.]))
        value,error=stable._point_probability(1e-6,*coefficients,1)
        old=ndtr(original._g_float(1e-6,5e-9,1e-7)/(2*5e-13))
        assert abs(value-old)>1e-7
        assert error<=1e-15
        assert abs(value-ndtr(.01))<1e-14


def test_algebraic_small_signal_identity_and_direct_logcosh_agree():
    # Independent tanh identity is safe in these nonsaturated synthetic cases.
    with ctx.workprec(256):
        for av,bv,tv in [(1e-8,1e-7,.002),(1e-6,1.,.02),(.3,.7,2.)]:
            a,b,t=map(original._ball,(av,bv,tv))
            direct=2*a*t+(b*(t-a)).cosh().log()-(b*(t+a)).cosh().log()
            identity=2*a*t-2*((b*t).tanh()*(a*b).tanh()).atanh()
            assert direct.overlaps(identity)
            assert direct.overlaps(original._g(t,a,b))


def test_tiny_signal_orientation_and_signed_mean_symmetry():
    a,b,c=5e-9,1e-7,5e-13
    A=vector([a,c]);B=vector([b,0]);mu=vector([.002,-.01])
    vals=[]
    for nuisance,mean,sign in [(B,mu,1),(-B,mu,1),(B,-mu,-1)]:
        r=stable.probability(stable.define_classifier(A,nuisance,1.),mean,sign)
        assert r['qualified'];vals.append(r['nominal'])
    assert max(vals)-min(vals)<1e-12


def test_binary64_rotated_synthetic_small_signal():
    a,b,c=5e-9,1e-7,5e-13
    e=vector([.6,.8,0]);v=vector([-.8,.6,0])
    d=stable.define_classifier(a*e+c*v,b*e,1.)
    result=stable.probability(d,.002*e-.01*v,1)
    assert result['qualified'],result
    assert abs(result['nominal']-score(a,b,c,.002,-.01)['nominal'])<1e-10
