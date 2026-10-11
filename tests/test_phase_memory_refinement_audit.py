"""Independent audit synthetic tests; never reads either target attempt."""
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/phase_memory_refinement_audit'),str(ROOT/'tools'),str(ROOT/'tests')]
import audit_refinement as audit
import phase_memory_refinement_evaluation as producer
import test_phase_memory_refinement_evaluation as fixture


def test_full_independent_selection_method_and_cell_census():
    cases=fixture.census();actual=audit.specs(cases)
    assert actual==producer.comparison_specs(cases) and len(actual)==2380
    assert len({x['id'] for x in actual})==2380
    assert len(audit.paired_cells(cases,{}))==3744
    assert all(row['status']=='failed' for row in audit.paired_cells(cases,{}))


@pytest.mark.parametrize('method',audit.METHODS)
def test_independent_scipy_clock_score_replays_fabricated_producer(method):
    case=fixture.case();ctx=fixture.context(case);packet=fixture.packet(case,method,ctx)
    evidence=audit.codec.Audit();strict=audit.codec.Audit()
    fresh=audit.score(packet,case,ctx,evidence);saved=producer.score_packet(packet,case,ctx)
    audit.tree_replay(saved,fresh,strict,evidence,'score')
    assert not evidence.failures and not strict.failures
    assert fresh['task']['method']==method


def test_independent_raw_late_final_and_crossing_diagnostics_match():
    spec,left,right,a,b=fixture.pair(fixture.case(sham=True))
    right['states'][10,0]=1e-6;right['states'][20,1]=np.nextafter(1e-6,np.inf)
    right['states'][9600,2]=2e-6;right['states'][-1,2]=3e-6
    gain=a['readouts']['1']['noise_gain'];b['readouts']['4']['q']=.02*1e-4*gain
    saved=producer.compare_pair(spec,left,right,a,b)
    fresh=audit.comparison(spec,{left['task']['id']:left,right['task']['id']:right},{a['task']['id']:a,b['task']['id']:b})
    evidence=audit.codec.Audit();strict=audit.codec.Audit();audit.tree_replay(saved,fresh,strict,evidence,'comparison')
    assert not evidence.failures and not strict.failures
    assert fresh['state_difference']['first_threshold_crossing_time']==.5
    assert fresh['activation_difference']['first_threshold_crossing_time']==.25
    assert fresh['first_completed_readout_threshold_crossing']['delay_cycles']==4
    assert fresh['late_window']['samples']==3201


def test_missing_method_is_explicit_unqualified_comparison():
    spec=fixture.pair()[0];fresh=audit.comparison(spec,{}, {})
    assert fresh==dict(spec=spec,complete=False,qualified=False)


def test_arithmetic_difference_remains_separate_from_scientific_flags():
    evidence=audit.codec.Audit();strict=audit.codec.Audit()
    audit.tree_replay({'qualified':True,'value':1e-11},{'qualified':True,'value':0.},strict,evidence,'row')
    assert not evidence.failures and len(strict.failures)==1
    audit.tree_replay({'qualified':False},{'qualified':True},strict,evidence,'changed gate')
    assert len(evidence.failures)==1


def test_sampled_crossing_times_are_exact_even_inside_scalar_replay_tolerance():
    evidence=audit.codec.Audit();strict=audit.codec.Audit()
    audit.tree_replay({'first_threshold_crossing_time':.025+1e-15},{'first_threshold_crossing_time':.025},strict,evidence,'state')
    assert len(evidence.failures)==1 and not strict.failures


def test_erf_gaussian_accuracy_matches_independent_formula_and_below_chance():
    actual,z=audit.accuracy(-1e-5,1e-5,.2,1,1e-4)
    assert actual==pytest.approx(.5*(1+math.erf(-.5/math.sqrt(2))),abs=1e-16) and z==[-.5,-.5]
    assert audit.accuracy(0.,0.,.2,-1,1e-5)[0]==.5


def test_all_paired_cells_keep_true_same_method_signs_and_failure_denominator():
    cases=fixture.census();scores={}
    for c in cases:
        if c['kind']=='sham':continue
        for method in audit.METHODS:
            scores[c['id']+'--'+method]=fixture.fabricated_score(c,method,q=-c['sign']*1e-5)
    rows=audit.paired_cells(cases,scores)
    assert len(rows)==3744 and all(x['status']=='complete' and x['balanced_accuracy']<.5 for x in rows)
    removed=next(iter(scores));del scores[removed]
    partial=audit.paired_cells(cases,scores)
    assert len(partial)==3744 and sum(x['status']=='failed' for x in partial)==12


def test_truthful_solver_and_exact_pulse_forgery_are_evidence_failures():
    c=fixture.case();ctx=fixture.context(c);p=fixture.packet(c,audit.NEW_METHODS[-1],ctx)
    p['solver']['id']='dop853';p['realized_delta']=p['realized_delta']+np.array([.1,0.,0.])
    evidence=audit.codec.Audit();audit.score(p,c,ctx,evidence)
    assert any('truthful solver' in x['check'] for x in evidence.failures)
    assert any('realized pulse' in x['check'] for x in evidence.failures)


def test_report_numpy_values_serialize_safely_without_relaxing_nan():
    value={'qualified':np.bool_(False),'count':np.int64(425),'vector':np.array([1.,2.])}
    assert json.loads(json.dumps(value,default=audit.json_default,allow_nan=False))==dict(qualified=False,count=425,vector=[1.,2.])
    with pytest.raises(ValueError):json.dumps({'x':np.float64(np.nan)},default=audit.json_default,allow_nan=False)
