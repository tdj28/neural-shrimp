"""Independent synthetic integration of EXP021's explicit numerical adapter."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_mixture_readout_evaluation as stable
import gaussian_mixture_readout as frozen_core
import mixture_readout_evaluation as frozen_evaluation
import run_phase_memory_recovery as codec
from test_refined_two_write_integration import pipeline
from test_mixture_readout_evaluation import fixture
from test_mixture_readout_integration import direct_density_accuracy


def test_adapter_preserves_entire_statistical_census():
    assert stable.cell_census() == frozen_evaluation.cell_census()
    assert sum(len(stable.calculation_census(c)) for c in stable.cell_census()) == 4704
    assert stable.assemble_cell is frozen_evaluation.assemble_cell
    assert stable.summarize is frozen_evaluation.summarize
    original = frozen_evaluation.contract()
    updated = stable.contract()
    assert all(updated[k] == v for k,v in original.items())
    assert updated['statistical_design_changed'] is False


def test_new_nominal_calls_do_not_patch_frozen_modules():
    old_probability, old_nominal = frozen_core.probability, frozen_core._nominal
    original_hash = codec.packet_digest(frozen_core.define_classifier(
        np.array([1e-8,2e-13]), np.array([1e-7,0.]), 1.))
    definition = stable.stable.define_classifier(np.array([1e-8,2e-13]), np.array([1e-7,0.]), 1.)
    result = stable.stable.probability(definition, np.array([.002,-.01]), 1)
    assert result['qualified'], result
    assert codec.packet_digest(definition) == original_hash
    assert frozen_core.probability is old_probability and frozen_core._nominal is old_nominal
    assert max(r['maximum_point_absolute_error'] for r in result['diagnostics']['nominal_runs']) <= 1e-15


@pytest.mark.parametrize('s1,s2',[(-1,-1),(-1,1),(1,-1),(1,1)])
def test_new_nominal_agrees_with_direct_four_density_oracle(s1,s2):
    A=np.array([1.3,.7]);B=np.array([2.1,0.]);mean=s1*A+s2*B+np.array([.17,-.23])
    oracle=direct_density_accuracy(A,B,mean,s1)
    result=stable.stable.probability(stable.stable.define_classifier(A,B,1.),mean,s1)
    assert result['qualified'],result
    assert result['interval'][0]-2e-9 <= oracle <= result['interval'][1]+2e-9
    assert abs(result['nominal']-oracle)<2e-9


def test_tiny_signal_cell_keeps_all40_slots_and_original_linear_replay():
    cell,sealed,scores=fixture();A=np.zeros(96);B=A.copy();A[:2]=[1e-12,2e-17];B[0]=1e-11
    for recipe,row in sealed['definitions'].items():
        row['A']=None if recipe=='one_mode' else A.copy();row['B']=B.copy()
        row['definition']=stable.stable.define_classifier(A,B,1e-4,one_mode_rho=row['rho'])
    for method,rows in scores.items():
        for row in rows:
            first,second=row['task']['first_sign'],row['task']['second_sign']
            mean=first*A+second*B
            row['windows'][cell['window']]['activation_displacement']=mean
            row['windows'][cell['window']]['primary']=frozen_evaluation.old._project(sealed['h'],mean,first)
    saved_before=codec.packet_digest(sealed);recorded=[]
    result=stable.evaluate_cell(cell,sealed,scores,progress_fn=recorded.append)
    assert result['complete'] and result['qualified'],result
    assert len(result['calculations'])==len(recorded)==40
    assert result['gates']['original_linear_replay']
    assert all(r['result']['qualified'] for r in recorded)
    assert codec.packet_digest(sealed)==saved_before
    single=stable.evaluate_calculation(cell,sealed,scores,stable.calculation_census(cell)[0])
    assert codec.packet_digest(single)==codec.packet_digest(recorded[0])


def test_same_analytic_oscillator_definitions_and_complete_new_pipeline(pipeline,tmp_path):
    p=pipeline;cell=next(c for c in stable.cell_census(p['phases'])
        if c['context_id']==p['context']['id'] and c['window']=='k2')
    outer=dict(context_id=cell['context_id'],status='complete',qualified=True,
        acquisition=p['physical'],evaluation=p['seal'])
    sealed=stable.seal_cell(cell,outer,p['references']['rk45'])
    original=frozen_evaluation.seal_cell(cell,outer,p['references']['rk45'])
    assert codec.packet_digest(sealed)==codec.packet_digest(original)
    file=codec.save_packet(tmp_path,'definitions',sealed);sealed=codec.load_packet(file['path'])
    scores={m:[copy.deepcopy(r) for r in p['scores'].values() if r['task']['method']==m
        and r['task']['arm']=='identity' and r['task']['first_sign']!=0 and r['task']['second_sign']!=0]
        for m in stable.methods_for_cell(cell)}
    result=stable.evaluate_cell(cell,sealed,scores)
    assert result['complete'] and result['qualified'] and result['secondary_qualified'],result
    assert len(result['calculations'])==40
    file=codec.save_packet(tmp_path,'result',result)
    assert codec.packet_digest(codec.load_packet(file['path']))==codec.packet_digest(result)
