import copy
from pathlib import Path
import sys

import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
import secant_restore as restore
from inverse_secant import InverseSecant


@pytest.fixture(autouse=True)
def precision():
    old=ctx.prec;ctx.prec=256
    yield
    ctx.prec=old


def model():return [arb_mat([[arb(1)/2]]) for _ in range(2)],arb_mat([[arb(5)/4]])
def h(c):return [[c[0][0]/2-c[1][0]],[c[1][0]/2-c[0][0]]]
def centers(i):return [[arb(i)],[arb(i*i+i)]]


def records(count=3):
    A,B=model();kernel=InverseSecant(A,B,**restore.CAPS)
    decisions=[]
    for i in range(count):decisions.append(kernel.update(centers(i),h(centers(i)),centers(i+1),h(centers(i+1)),evidence={'index':i}))
    proposal=kernel.propose(centers(count),h(centers(count)),evidence={'last':count})['record']
    return kernel,decisions,proposal


@pytest.mark.parametrize('count',[0,1,2,3,4,5,6])
def test_actual_kernel_history_roundtrips_after_proposal(count):
    kernel,decisions,proposal=records(count)
    history,audit=restore.restore_factors(decisions,proposal,period=2,dimension=1)
    assert history==kernel.history
    assert audit['active_history_after_proposal']==len(kernel.history)
    assert audit['certificate_passed'] is False
    assert audit['historical_dense_arithmetic_replayed'] is False
    if count==5:
        assert decisions[-1]['reason']=='history_cap_restart' and not history
    cloned=restore.install_factors(*model(),history)
    for v in ([arb(1),arb(0)],[arb(-3),arb(2)]):
        for a,b in zip(kernel.apply(v),cloned.apply(v)):assert a.overlaps(b)


def test_proposal_step_fallback_clears_accepted_factors():
    A=[arb_mat([[0]])]*2;B=arb_mat([[1]]);kernel=InverseSecant(A,B,**restore.CAPS)
    old=[[arb(0)],[arb(0)]];new=[[arb(10)],[arb(0)]]
    residual=[[arb(0)],[arb(1)]]
    decision=kernel.update(old,old,new,residual,evidence={'synthetic':1})
    assert decision['accepted']
    proposal=kernel.propose(new,residual,evidence={'synthetic':2})['record']
    assert proposal['reason']=='step_cap_baseline_fallback'
    history,audit=restore.restore_factors([decision],proposal,period=2,dimension=1)
    assert history==[] and audit['active_history_after_proposal']==0


def test_rejected_update_with_retained_factors_does_not_install_them():
    kernel=InverseSecant([arb_mat([[0]])]*2,arb_mat([[1]]),**restore.CAPS)
    old=[[arb(0)],[arb(0)]];new=[[arb(100000)],[arb(0)]];residual=[[arb(0)],[arb(1)]]
    decision=kernel.update(old,old,new,residual,evidence={'synthetic':1})
    assert not decision['accepted'] and decision['reason']=='update_norm_cap' and 'u_midpoints' in decision
    proposal=kernel.propose(new,residual,evidence={'synthetic':2})['record']
    history,_=restore.restore_factors([decision],proposal,period=2,dimension=1)
    assert history==[]


@pytest.mark.parametrize('fault',['prior','active','accepted_bool','unknown_reason','history_cap','factor_radius',
    'factor_shape','factor_midpoint','update_bound','signal','denominator','negative_denominator_radius',
    'proposal_clear','proposal_choice','proposal_candidate','step_cap','accelerated_upper','baseline_lower','cap_inequality'])
def test_corrupted_restoration_record_refused(fault):
    _,decisions,proposal=records(2)
    decisions=copy.deepcopy(decisions);proposal=copy.deepcopy(proposal)
    d=decisions[0]
    if fault=='prior':d['prior_history']=1
    if fault=='active':d['active_history']=0
    if fault=='accepted_bool':d['accepted']=1
    if fault=='unknown_reason':d['reason']='silently_retried'
    if fault=='history_cap':d['max_history']=5
    if fault=='factor_radius':d['u_midpoints'][0]['rad']=[1,-300]
    if fault=='factor_shape':d['u_midpoints'].pop()
    if fault=='factor_midpoint':d['u_enclosures'][0]['mid']=[1,8]
    if fault=='update_bound':d['update_norm_upper']={'mid':[0,0],'rad':[0,0]}
    if fault=='signal':d['signal']={'mid':[0,0],'rad':[0,0]}
    if fault=='denominator':d['denominator']={'mid':[-1,0],'rad':[0,0]}
    if fault=='negative_denominator_radius':d['denominator']['rad']=[-1,-300]
    if fault=='proposal_clear':proposal['active_history']=0
    if fault=='proposal_choice':proposal['chosen_correction']=copy.deepcopy(proposal['chosen_correction']);proposal['chosen_correction'][0]['mid']=[1,12]
    if fault=='proposal_candidate':proposal['next_centers'][0]['mid']=[1,12]
    if fault=='step_cap':proposal['step_ratio_limit']={'mid':[1,0],'rad':[0,0]}
    if fault=='accelerated_upper':proposal['accelerated_step_upper']={'mid':[0,0],'rad':[0,0]}
    if fault=='baseline_lower':proposal['baseline_step_lower']={'mid':[1,100],'rad':[0,0]}
    if fault=='cap_inequality':proposal['accelerated_step_upper']={'mid':[1,100],'rad':[0,0]}
    with pytest.raises(ValueError):restore.restore_factors(decisions,proposal,period=2,dimension=1)


def test_installed_kernel_copies_factor_vectors_and_matrices():
    _,decisions,proposal=records(1);history,_=restore.restore_factors(decisions,proposal,period=2,dimension=1)
    A,B=model();kernel=restore.install_factors(A,B,history)
    original=history[0][0][0]
    kernel.history[0][0][0]=arb(99);kernel.A[0][0,0]=arb(98);kernel.B[0,0]=arb(97)
    assert history[0][0][0]==original and A[0][0,0]==arb(1)/2 and B[0,0]==arb(5)/4


def test_no_input_mutation_and_strict_fixed_caps():
    _,decisions,proposal=records(2);before=copy.deepcopy((decisions,proposal))
    restore.restore_factors(decisions,proposal,period=2,dimension=1)
    assert (decisions,proposal)==before
    with pytest.raises(ValueError):restore.restore_factors(decisions,proposal,period=True,dimension=1)
    with pytest.raises(ValueError):restore.install_factors(*model(),[],caps={**restore.CAPS,'max_history':8})
