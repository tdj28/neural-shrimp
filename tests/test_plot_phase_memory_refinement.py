"""Synthetic sealed summaries only; no target reads, integration or fitting."""
import copy
import json
import math
from pathlib import Path
import sys
import pytest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import plot_phase_memory_refinement as p


def case(setting,phase,direction=None,dose=None,sign=None):
    stem=f'{setting}--p{phase}'
    if direction is None:return dict(id=stem+'--sham',setting=setting,phase_index=phase,kind='sham',dose_index=None,epsilon=None,direction=None,sign=None)
    return dict(id=stem+f'--d{dose}--{direction}--s{sign:+d}',setting=setting,phase_index=phase,kind='pulse',dose_index=dose,epsilon=(1e-4,1e-3,1e-2)[dose],direction=direction,sign=sign)


def fixtures():
    contexts=[('core',3,'phase_write'),('core',3,'neuron_one'),('core',7,'phase_write'),('core',7,'neuron_one'),('failed-near-edge',2,'phase_write'),('failed-near-edge',2,'neuron_one'),('failed-near-edge',3,'phase_write'),('failed-near-edge',3,'neuron_one'),('failed-near-edge',4,'phase_write'),('failed-near-edge',5,'phase_write'),('failed-near-edge',5,'neuron_one'),('failed-near-edge',6,'phase_write'),('failed-near-edge',6,'neuron_one')]
    cases=[case(s,ph,d,e,sign) for s,ph,d in contexts for e in range(3) for sign in (-1,1)]
    cases += [case(s,ph) for s,ph in sorted({(s,ph) for s,ph,d in contexts})]
    failed=[c['id'] for c in cases if c['dose_index']==2][:18]
    cats={c['id']:'matched-sham' if c['kind']=='sham' else 'original-failed-strong-sign' if c['id'] in failed else 'matched-strong-opposite-sign' if c['dose_index']==2 else 'matched-lower-dose-both-signs' for c in cases}
    manifest=dict(source_commit='a'*40,plan_sha256='b'*64,plan=dict(selection=dict(cases=cases,failed_signed_ids=failed,selection_categories=cats)))
    terminal=[]
    for c in cases:
        for mi,m in enumerate(p.METHODS):
            task=dict(c,id=c['id']+'--'+m,method=m)
            q=0. if c['kind']=='sham' else c['sign']*(c['epsilon']+mi*1e-6)
            score=dict(task=task,status='complete',orientation=1,readouts={str(d):dict(q=q,noise_gain=.1) for d in p.DELAYS})
            terminal.append(dict(task=task,status='complete',score=score))
    comparisons=[];caseindex={c['id']:c for c in cases}
    def difference(peak,start=0.):
        return dict(rms=peak/2,maximum_absolute=peak,maximum_time=320.,maximum_component=0,value_at_maximum=peak,component_rms=[peak/2,0.,0.],first_threshold_crossing_time=None)
    for spec in p.pair_specs(cases):
        c=caseindex[spec['case_id']];i,j=[p.METHODS.index(m) for m in spec['pair']]
        peak=0. if c['kind']=='sham' else abs(i-j)*(1. if c['dose_index']==2 and i<3 else 1e-9)
        act=peak/10;state=difference(peak);activation=difference(act)
        if peak>1e-6:state['first_threshold_crossing_time']=100.
        if act>1e-7:activation['first_threshold_crossing_time']=100.025
        late_state=difference(peak,240.);late_act=difference(act,240.)
        if peak>1e-6:late_state['first_threshold_crossing_time']=240.
        if act>1e-7:late_act['first_threshold_crossing_time']=240.
        qdiff=0. if c['kind']=='sham' else abs(i-j)*1e-6/(1e-4*.1)
        readouts={str(d):dict(standardized_mean_difference=qdiff,passed=qdiff<=.01,completed_window_end=float(d),last_sample_time=d-1+31.5/32) for d in p.DELAYS}
        first=None if qdiff<=.01 else dict(delay_cycles=1,completed_window_end=1.,last_sample_time=31.5/32)
        comparisons.append(dict(spec=spec,complete=True,qualified=peak<=1e-6 and act<=1e-7 and qdiff<=.01,error=None,state_difference=state,activation_difference=activation,
            late_window=dict(start=240.,end=320.,samples=3201,state_difference=late_state,activation_difference=late_act),last_state_difference=[peak,0.,0.],last_activation_difference=[act,0.,0.],readouts=readouts,first_completed_readout_threshold_crossing=first,state_pass=peak<=1e-6,activation_pass=act<=1e-7))
    ba=[]
    for s,ph,d in contexts:
        for e in range(3):
            for mi,m in enumerate(p.METHODS):
                for delay in p.DELAYS:
                    for sigma in p.NOISES:
                        z=((1e-4,1e-3,1e-2)[e]+mi*1e-6)/(sigma*.1)
                        accuracy=.5*(1+math.erf(z/math.sqrt(2)))
                        ba.append(dict(setting=s,phase_index=ph,direction=d,dose_index=e,method=m,delay_cycles=delay,sigma=sigma,status='complete',balanced_accuracy=accuracy,phase_only_balanced_accuracy=accuracy,full_linear_balanced_accuracy=accuracy,zero_balanced_accuracy=.5,standardized_correct_means=[z,z]))
    groups={g:dict(required=sum(r['spec']['group']==g for r in comparisons),complete=sum(r['spec']['group']==g for r in comparisons),passed=sum(r['spec']['group']==g and r['qualified'] for r in comparisons)) for g in p.GROUPS}
    summary=dict(experiment_id='EXP-014-refinement',original_unchanged=True,complete=True,source_valid=True,error=None,postflight_error=None,analysis=dict(counts=p.EXPECTED_COUNTS,setting_means=None,original_study_requalified=False,terminal_rows=terminal,comparisons=comparisons,paired_readouts=ba,comparison_groups=groups,complete=True))
    return summary,manifest


@pytest.fixture
def data():return fixtures()


def test_complete_denominators_and_saturation_stays_visible(data):
    summary,manifest=data;out=p.collect(summary,manifest)
    assert out['counts']==p.EXPECTED_COUNTS and out['analysis_complete']
    assert len(out['cases'])==85 and len(out['pairs'])==2380 and len(out['signed_readouts'])==2720 and len(out['paired_readouts'])==3744
    assert sum(c['original_failed_sign'] for c in out['cases'])==18
    strong=[c for c in out['cases'] if c['case']['dose_index']==2]
    assert all(c['maximum_readout_difference']>.01 and c['maximum_primary_ba_method_change']==0. for c in strong)
    assert all(c['maximum_primary_ba_method_change'] is None and c['primary_ba_cells_required']==0 for c in out['cases'] if c['case']['kind']=='sham')
    assert out['display']['means'] is None


@pytest.mark.parametrize('bad',['cases','duplicate_case','failed_signs','category','terminal','duplicate_terminal','comparison','duplicate_comparison','pair_group','ba','duplicate_ba','boolean_phase','nan','qualification','crossing','threshold_onset','counts','means','requalification'])
def test_corrupt_or_omitted_visible_evidence_rejected(data,bad):
    summary,manifest=data;science=summary['analysis'];row=science['comparisons'][0]
    if bad=='cases':manifest['plan']['selection']['cases'].pop()
    elif bad=='duplicate_case':manifest['plan']['selection']['cases'][-1]=manifest['plan']['selection']['cases'][0]
    elif bad=='failed_signs':manifest['plan']['selection']['failed_signed_ids'].pop()
    elif bad=='category':manifest['plan']['selection']['selection_categories'][manifest['plan']['selection']['cases'][0]['id']]='invented'
    elif bad=='terminal':science['terminal_rows'].pop()
    elif bad=='duplicate_terminal':science['terminal_rows'][-1]=science['terminal_rows'][0]
    elif bad=='comparison':science['comparisons'].pop()
    elif bad=='duplicate_comparison':science['comparisons'][-1]=science['comparisons'][0]
    elif bad=='pair_group':row['spec']['group']='new-new'
    elif bad=='ba':science['paired_readouts'].pop()
    elif bad=='duplicate_ba':science['paired_readouts'][-1]=science['paired_readouts'][0]
    elif bad=='boolean_phase':science['paired_readouts'][0]['phase_index']=True
    elif bad=='nan':row['state_difference']['maximum_absolute']=float('nan')
    elif bad=='qualification':row['qualified']=not row['qualified']
    elif bad=='crossing':row['first_completed_readout_threshold_crossing']=None
    elif bad=='threshold_onset':row['state_difference']['first_threshold_crossing_time']=.0001
    elif bad=='counts':science['counts']=dict(science['counts'],pair_rows=2379)
    elif bad=='means':science['setting_means']={'core':.9}
    else:science['original_study_requalified']=True
    with pytest.raises((ValueError,KeyError)):p.collect(summary,manifest)


def test_unavailable_pair_keeps_fixed_denominator_and_marker(data):
    summary,manifest=data;science=summary['analysis'];old=science['comparisons'][0];spec=old['spec'];group=spec['group']
    science['comparisons'][0]=dict(spec=spec,complete=False,qualified=False,error='synthetic missing raw')
    science['comparison_groups'][group]['complete']-=1
    science['comparison_groups'][group]['passed']-=int(old['qualified'])
    science['complete']=False;summary['complete']=False
    out=p.collect(summary,manifest);first=out['cases'][0]['groups'][group]
    assert not out['analysis_complete'] and first['unavailable']==1 and first['required']==3
    assert len(out['pairs'])==2380


def seal(tmp_path,summary,manifest):
    source=tmp_path/'source';source.mkdir()
    for name,value in [('summary.json',summary),('manifest.json',manifest),('postflight.json',dict(error=None,source_bindings_unchanged=True))]:
        (source/name).write_text(json.dumps(value))
    artifacts=[p.binding(path,source) for path in sorted(source.iterdir())]
    receipt=dict(source_commit=manifest['source_commit'],plan_sha256=manifest['plan_sha256'],experiment_id='EXP-014-refinement',status='complete',artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts))
    (source/'receipt.json').write_text(json.dumps(receipt));return source


def test_full_receipt_bindings_and_immutable_output(data,tmp_path):
    summary,manifest=data;source=seal(tmp_path,summary,manifest)
    saved,plan,provenance=p.load_saved(source);assert provenance['artifacts_verified']==3
    assert p.collect(saved,plan)['analysis_complete']
    (source/'extra.txt').write_text('unbound')
    with pytest.raises(ValueError,match='inventory'):p.load_saved(source)
    (source/'extra.txt').unlink();(source/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):p.load_saved(source)


def test_actual_report_export_and_figures(data,tmp_path):
    summary,manifest=data;source=seal(tmp_path,summary,manifest);saved,plan,provenance=p.load_saved(source)
    output=tmp_path/'report';result=p.write_report(saved,plan,provenance,output,source_directory=source)
    assert result['analysis_complete'] and (output/'refinement-diagnostics.png').stat().st_size>10000 and (output/'refinement-diagnostics.pdf').stat().st_size>10000
    assert len((output/'signed-readouts.csv').read_text().splitlines())==2721
    assert len((output/'case-groups.csv').read_text().splitlines())==256
    assert 'no noisy trials' in (output/'README.md').read_text()
    receipt=p.read(output/'receipt.json')
    assert receipt['artifacts']==[p.binding(path,output) for path in sorted(output.iterdir()) if path.name!='receipt.json']
    with pytest.raises(FileExistsError):p.write_report(saved,plan,provenance,output)
    with pytest.raises(ValueError):p.write_report(saved,plan,provenance,source/'nested',source_directory=source)


def test_partial_report_plot_does_not_create_averages(data,tmp_path):
    summary,manifest=data;science=summary['analysis'];old=science['comparisons'][0];g=old['spec']['group']
    science['comparisons'][0]=dict(spec=old['spec'],complete=False,qualified=False,error='synthetic partial')
    science['comparison_groups'][g]['complete']-=1;science['comparison_groups'][g]['passed']-=int(old['qualified'])
    science['complete']=False;summary['complete']=False
    out=p.write_report(summary,manifest,{'synthetic':True},tmp_path/'partial')
    report=p.read(tmp_path/'partial/report.json');assert not out['analysis_complete'] and report['display']['means'] is None
    assert report['cases'][0]['groups'][g]['unavailable']==1
