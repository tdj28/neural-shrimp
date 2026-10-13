"""Small CPU-only inverse-secant proposal experiments and refusal tests."""
from pathlib import Path
import sys
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import inverse_secant
from inverse_secant import InverseSecant
from shooting_correction import correct


@pytest.fixture(autouse=True)
def precision():
    old=ctx.threads;ctx.threads=1
    with ctx.workprec(256):yield
    ctx.threads=old


def rows(values):return [[arb(v)] for v in values]
def kernel(**kwargs):return InverseSecant([arb_mat([[0]]),arb_mat([[0]])],arb_mat([[1]]),**kwargs)
def evidence():return {'synthetic_complete_tuple':True}


def test_exact_cyclic_inverse_and_correction_sign_match_existing_kernel():
    A=[arb_mat([[1]]),arb_mat([[arb(1)/2]])];B=arb_mat([[2]])
    k=InverseSecant(A,B);v=[arb(3),arb(-2)];got=k.baseline(v)
    # J=[[1,-1],[-1,1/2]]; J R v=v.
    assert got[0]-got[1]==v[0] and -got[0]+got[1]/2==v[1]
    original=correct(rows([0,0]),rows([3,-2]),A,B)
    assert all(got[i]==-original['record']['corrections'][i][0]['mid'][0]*arb((1,original['record']['corrections'][i][0]['mid'][1])) for i in range(2))


def test_rank_one_secant_is_exact_when_all_quantities_dyadic():
    k=kernel();record=k.update(rows([0,0]),rows([0,0]),rows([1,0]),rows([0,-2]),evidence=evidence())
    assert record['accepted']
    assert k.apply([arb(0),arb(-2)])==[arb(1),arb(0)]
    assert len(k.history)==1 and all(v.is_exact() for pair in k.history for col in pair for v in col)


def test_collinear_secants_are_safe_and_cap_restart_is_explicit():
    k=kernel(max_history=2)
    assert k.update(rows([0,0]),rows([0,0]),rows([1,0]),rows([0,-2]),evidence=evidence())['accepted']
    assert k.update(rows([0,0]),rows([0,0]),rows([2,0]),rows([0,-4]),evidence=evidence())['accepted']
    rejected=k.update(rows([0,0]),rows([0,0]),rows([4,0]),rows([0,-8]),evidence=evidence())
    assert rejected['reason']=='history_cap_restart' and not k.history
    assert k.propose(rows([1,1]),rows([1,1]),evidence=evidence())['record']['reason']=='baseline_no_history'
    assert len(k.decisions)==4


@pytest.mark.parametrize('new_h,reason',[(rows([0,0]),'zero_secant'),
    ([[arb(1,2)],[arb(-1,2)]],'noise_dominated_secant')])
def test_zero_and_noisy_secants_reset_to_baseline(new_h,reason):
    k=kernel();r=k.update(rows([0,0]),rows([0,0]),rows([1,1]),new_h,evidence=evidence())
    assert not r['accepted'] and r['reason']==reason and not k.history
    assert 'residual_difference' in r and r['certificate_passed'] is False


def test_very_small_dyadic_secant_scaled_without_binary64_underflow():
    k=kernel();small=arb((1,-2000))
    r=k.update(rows([0,0]),rows([0,0]),[[small],[arb(0)]],[[arb(0)],[-2*small]],evidence=evidence())
    assert r['accepted'] and r['scale_power']==1999
    got=k.apply([arb(0),-2*small])
    assert got[0]==small and got[1]==0


def test_full_residual_radii_remain_in_witnesses_and_proposal():
    k=kernel();r=k.update(rows([0,0]),[[arb(0,(1,-100))],[arb(0,(1,-100))]],
        rows([1,0]),[[arb(0,(1,-100))],[arb(-2,(1,-100))]],evidence=evidence())
    assert r['accepted']
    assert all(v['rad'][0]>0 for v in r['residual_difference'])
    result=k.propose(rows([1,1]),[[arb(1,(1,-100))],[arb(-1,(1,-100))]],evidence=evidence())
    assert all(v.is_exact() for row in result['centers'] for v in row)
    assert any(v['rad'][0]>0 for v in result['record']['chosen_correction'])


def test_update_norm_cap_and_step_cap_have_deterministic_baseline_fallbacks():
    k=kernel(update_norm_limit=1)
    r=k.update(rows([0,0]),rows([0,0]),rows([100,0]),rows([0,-1]),evidence=evidence())
    assert r['reason']=='update_norm_cap' and not k.history
    k=kernel(update_norm_limit=1024,step_ratio_limit=2)
    assert k.update(rows([0,0]),rows([0,0]),rows([100,0]),rows([0,-1]),evidence=evidence())['accepted']
    out=k.propose(rows([1,1]),rows([0,-1]),evidence=evidence())
    assert out['record']['reason']=='step_cap_baseline_fallback' and not k.history
    assert out['centers']==rows([0,1])
    assert out['record']['accelerated_correction'][0]['mid']!=out['record']['baseline_correction'][0]['mid']


@pytest.mark.parametrize('fault',['centers','residuals','radius_center','nonfinite','evidence'])
def test_incomplete_or_nonfinite_evidence_refused(fault):
    k=kernel();c0=rows([0,0]);h0=rows([0,0]);c1=rows([1,1]);h1=rows([2,2]);meta=evidence()
    if fault=='centers':c1.pop()
    if fault=='residuals':h0[0].append(arb(0))
    if fault=='radius_center':c1[0][0]=arb(1,1)
    if fault=='nonfinite':h1[1][0]=arb('nan')
    if fault=='evidence':meta={}
    with pytest.raises(ValueError):k.update(c0,h0,c1,h1,evidence=meta)


def iterate(function,initial,*,accelerate,steps=8):
    k=kernel(max_history=4,step_ratio_limit=16)
    c=rows(initial);h=function(c);previous=None
    traces=[]
    for _ in range(steps):
        if accelerate and previous is not None:k.update(previous[0],previous[1],c,h,evidence=evidence())
        old=(c,h);proposal=k.propose(c,h,evidence=evidence());c=proposal['centers'];h=function(c);previous=old
        traces.append(max(float(abs(v.mid())) for row in h for v in row))
    return traces,k


def test_nonlinear_known_root_and_linear_error_families_remain_finite():
    cases={
        'low_rank':lambda c:[[c[0][0]/4-c[1][0]],[-c[0][0]]],
        'full_rank':lambda c:[[c[0][0]/4-c[1][0]],[c[1][0]/2-c[0][0]]],
        'nonnormal':lambda c:[[2*c[0][0]-c[1][0]],[c[1][0]/16-c[0][0]]],
        'nonlinear':lambda c:[[c[0][0]/4+c[0][0]*c[0][0]/32-c[1][0]],
                              [c[1][0]/4+c[1][0]*c[1][0]/32-c[0][0]]],
    }
    for name,function in cases.items():
        fixed,_=iterate(function,[arb(1)/4,-arb(1)/8],accelerate=False)
        fast,k=iterate(function,[arb(1)/4,-arb(1)/8],accelerate=True)
        assert all(v>=0 and v<1e10 for v in fixed+fast)
        assert len(k.history)<=4 and all(d['certificate_passed'] is False for d in k.decisions)
        print(name,'baseline',fixed[-1],'broyden',fast[-1],'ratio',fast[-1]/fixed[-1] if fixed[-1] else None)


def test_runtime_builds_only_phase_vectors_never_a_global_square(monkeypatch):
    native=inverse_secant.arb_mat
    k=InverseSecant([native(2,2)]*4,native([[1,0],[0,1]]))
    shapes=[]
    def audited(*args):
        value=native(*args);shapes.append((value.nrows(),value.ncols()))
        assert value.nrows()<=2 and value.ncols()==1
        return value
    monkeypatch.setattr(inverse_secant,'arb_mat',audited)
    c0=[[arb(0),arb(0)] for _ in range(4)]
    c1=[[arb(1),arb(-1)] for _ in range(4)]
    h1=[[arb(-2),arb(2)] for _ in range(4)]
    assert k.update(c0,c0,c1,h1,evidence=evidence())['accepted']
    k.propose(c1,h1,evidence=evidence())
    assert shapes and len(k.history)==1 and len(k.history[0][0])==8


def test_unchanged_centers_rejected_and_evidence_is_snapshotted():
    k=kernel();meta={'run':'synthetic','nested':{'complete':True}}
    result=k.update(rows([0,0]),rows([0,0]),rows([0,0]),rows([1,1]),evidence=meta)
    assert result['reason']=='zero_center_step'
    meta['nested']['complete']=False
    assert result['evidence']['nested']['complete'] is True


@pytest.mark.parametrize('kwargs',[{'max_history':0},{'max_history':9},{'max_history':True},
    {'signal_ratio':1},{'step_ratio_limit':arb(1)/2},{'update_norm_limit':arb(1,1)}])
def test_invalid_caps_refused(kwargs):
    with pytest.raises(ValueError):kernel(**kwargs)
