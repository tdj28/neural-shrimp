import copy
from fractions import Fraction
import importlib.util
from pathlib import Path
import pytest
from flint import ctx
from test_eigenpair_reanchor import parent
from neural_shrimp.eigenpair_partial_repair import repair_inputs, repair
from neural_shrimp.rigorous_eigenpair_march import next_inputs
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube
from neural_shrimp.rigorous_orbits import ball_json


def fixture():
    model,p=parent();initial,kw=next_inputs(p);f,_=quadratic_tube(model,initial,**kw)
    f=copy.deepcopy(f);f['passed']=False;f['gates']['uniform_normalized_eigenpair_locus']=False
    old=[Fraction(w)*Fraction(2)**f['settings']['radius_exponent'] for w in f['weights']]
    proposal={'old_exact_radii':list(map(str,old)),'exact_radii':[str(v*Fraction(10001,10000)) for v in old]}
    return model,p,f,proposal


def test_native_fresh_proof_uses_qualified_parent_not_failed_center():
    with ctx.workprec(256):
        m,p,f,proposal=fixture();events=[];ticks=[];calls=[]
        # Failed output never supplies a state, mode or parameter anchor.
        f['center']=['invalid failed center'];f['endpoint_root_boxes']=['invalid failed root']
        def tube(model,initial,**kw):
            calls.append((initial,kw));return quadratic_tube(model,initial,**kw)
        result=repair(m,p,f,proposal,tick=lambda:ticks.append(1),observe=lambda k,v:events.append(k),tube=tube)
        assert result['passed'] and result['parameter_endpoints']==f['parameter_endpoints']
        expected,kw=next_inputs(p)
        assert calls[0][0]==expected
        actual=calls[0][1]['anchor'];wanted=kw['anchor']
        assert {k:v for k,v in actual.items() if k!='box'}=={k:v for k,v in wanted.items() if k!='box'}
        assert list(map(ball_json,actual['box']))==list(map(ball_json,wanted['box']))
        assert len(ticks)==15 and events[0]=='repair_proposal' and events[-1]=='chart_return'
        assert all(Fraction(w)*Fraction(2)**result['settings']['radius_exponent']>=Fraction(old) for w,old in zip(result['weights'],proposal['exact_radii']))


@pytest.mark.parametrize('case',['already_passed','wrong_gate','wrong_period','wrong_sign','wrong_normal','wrong_interval','wrong_precision','old_identity','shrink','radius_nonexact','radius_mismatch','dimension'])
def test_invalid_failed_context_or_proposal_rejected_before_model(case):
    with ctx.workprec(256):
        m,p,f,proposal=fixture()
        if case=='already_passed':f['passed']=True
        if case=='wrong_gate':f['gates']['anchor_identified']=False
        if case=='wrong_period':f['period']+=1
        if case=='wrong_sign':f['sigma']=-1
        if case=='wrong_normal':f['normal'][0]='100'
        if case=='wrong_interval':f['parameter_endpoints'][0]='100'
        if case=='wrong_precision':f['settings']['precision']=128
        if case=='old_identity':proposal['old_exact_radii'][0]='1'
        if case=='shrink':proposal['exact_radii'][0]=str(Fraction(proposal['old_exact_radii'][0])/2)
        if case=='radius_nonexact':f['coordinate_radii'][0]['rad']=[1,-100]
        if case=='radius_mismatch':f['coordinate_radii'][0]['mid']=[1,0]
        if case=='dimension':f['coordinate_radii'].pop()
        with pytest.raises(ValueError):repair_inputs(p,f,proposal)


def test_proposal_durability_failure_prevents_model_call():
    with ctx.workprec(256):
        m,p,f,proposal=fixture();calls=[]
        def observe(k,v):raise OSError('durability')
        with pytest.raises(OSError):repair(m,p,f,proposal,observe=observe,tube=lambda *a,**kw:calls.append(1))
        assert not calls


def test_source_inventory_and_cost_gate():
    path=Path(__file__).parents[1]/'tools/repair_eigenpair_partial.py'
    spec=importlib.util.spec_from_file_location('repair_driver',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    assert set(d.I.SOURCES)<=set(d.SOURCES)
    assert all((path.parents[1]/n).is_file() for n in d.SOURCES)
    old=dict(passed=False,ticks=508,calls=4,status='qualified_prefix_then_unresolved',peak_rss_kib=1139348,elapsed_seconds=85.3,stored_bytes_before_final=174048933)
    assert d.cost_gate(old)['seconds']<90
    old['elapsed_seconds']=200
    with pytest.raises(ValueError):d.cost_gate(old)
