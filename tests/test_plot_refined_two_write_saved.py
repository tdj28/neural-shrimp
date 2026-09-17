from pathlib import Path
import copy
import importlib.util
import csv
import json
import numpy as np
import pytest

p=Path(__file__).resolve().parent/'plot_refined_two_write_saved.py'
if not p.exists():p=Path(__file__).resolve().parents[1]/'tools/plot_refined_two_write_saved.py'
spec=importlib.util.spec_from_file_location('plot_refined_two_write_saved',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    balances=[];scores=[];contexts={}
    for setting in m.SETTINGS:
        for phase in range(24):
            cid=f'{setting}--p{phase:02d}';ok=phase!=9;split='development' if phase<8 else 'heldout'
            contexts[cid]=dict(qualified=ok,complete=ok)
            for arm in m.ARMS:
                for window in m.WINDOWS:
                    balances.append(dict(context_id=cid,setting=setting,phase_index=phase,split=split,arm=arm,window=window,
                        complete=ok,qualified=ok,with_second=.6 if ok else None,without_second=.7 if ok else None,interference_loss=.1 if ok else None))
                for first in (-1,0,1):
                    for second in (-1,0,1):
                        for method in (('rk4-coarse','rk4-fine','dop853-independent') if phase in (0,8) else ('rk4-coarse','rk4-fine')):
                            task=dict(setting=setting,phase_index=phase,split=split,arm=arm,first_sign=first,second_sign=second,method=method)
                            scores.append(dict(task=task,task_id=f'{cid}--{arm}--s1{first:+d}--s2{second:+d}--{method}',complete=ok,qualified=ok,
                                windows=dict(k4=dict(times=(np.array(3+(np.arange(32)+.5)/32)*(1. if setting=='core' else 2.)).tolist(),primary=dict(signed_margin=float(first*second),projected_mean=float(second),probability_correct=.6)))))
    return dict(synthetic_fixture=True,experiment_id='EXP-019',complete=False,source_valid=True,reused_regression_phases=True,fresh_holdout=False,analysis=dict(settings={s:dict(qualified=False,means=None) for s in m.SETTINGS},primary=None,counts=dict(histories=1296,trace_tasks=2700,comparisons=1404),balanced_accuracies=balances,scores=scores,contexts=contexts))


def test_census_and_missing_columns():
    balances,fine,contexts=m.collect(fixture())
    assert len(balances)==576 and len(fine)==1296 and len(contexts)==48
    a=m.matrix(balances,'core',[('identity','k4','with_second')])
    assert a.shape==(1,24) and np.isnan(a[0,9]) and np.sum(np.isfinite(a))==23
    z=m.margin_matrix(fine,contexts,'core')
    assert z.shape==(6,24) and np.all(np.isnan(z[:,9]))
    assert np.all(z[::2,0]==-1) and np.all(z[1::2,0]==1)


@pytest.mark.parametrize('edit',[
    lambda s:s['analysis']['balanced_accuracies'].pop(),
    lambda s:s['analysis']['balanced_accuracies'].__setitem__(1,s['analysis']['balanced_accuracies'][0]),
    lambda s:s['analysis']['scores'].pop(),
    lambda s:s['analysis']['contexts'].pop('core--p00'),
    lambda s:s['analysis']['balanced_accuracies'][0].update(with_second=np.nan),
    lambda s:s['analysis']['balanced_accuracies'][0].update(qualified=False),
    lambda s:s['analysis']['balanced_accuracies'][9*12].update(with_second=.6),
    lambda s:s['analysis']['counts'].update(comparisons=1080),
])
def test_refuses_dropped_or_invented_cells(edit):
    s=fixture();edit(s)
    with pytest.raises(ValueError):m.collect(s)


def test_all_unavailable_is_retained():
    s=fixture()
    for row in s['analysis']['balanced_accuracies']:row.update(complete=False,qualified=False,with_second=None,without_second=None,interference_loss=None)
    for row in s['analysis']['contexts'].values():row.update(complete=False,qualified=False)
    for row in s['analysis']['scores']:row.update(complete=False,qualified=False,windows={})
    balances,fine,contexts=m.collect(s)
    assert np.isnan(m.matrix(balances,'core',[('identity','k4','with_second')])).all()
    assert np.isnan(m.margin_matrix(fine,contexts,'core')).all()


def test_synthetic_render_preserves_csv_denominators(tmp_path):
    result=m.render(fixture(),tmp_path)
    assert result['phase_rows']==result['signed_margin_rows']==576
    assert len((tmp_path/'phase-readouts.csv').read_text().splitlines())==577
    assert len((tmp_path/'signed-margins-at-4T.csv').read_text().splitlines())==577
    assert all(Path(p).stat().st_size>10000 for p in result['images'])
    assert result['periods']=={'core':1.,'failed-near-edge':2.}
    rows=list(csv.DictReader((tmp_path/'phase-readouts.csv').open()))
    absent=[r for r in rows if r['phase_index']=='9']
    assert len(absent)==24 and all(r['plotted']=='False' for r in absent)
    assert all(r['reused_regression_phase']=='True' for r in rows)
    edge=next(r for r in rows if r['setting']=='failed-near-edge' and r['window']=='k4')
    assert float(edge['window_start'])==6. and float(edge['window_end'])==8.


def complete_fixture():
    s=fixture();s['complete']=True
    for row in s['analysis']['balanced_accuracies']:
        row.update(complete=True,qualified=True,with_second=.6,without_second=.7,interference_loss=.1)
    for row in s['analysis']['contexts'].values():row.update(complete=True,qualified=True)
    for row in s['analysis']['scores']:row.update(complete=True,qualified=True)
    s['analysis']['settings']={setting:dict(qualified=True,means=[]) for setting in m.SETTINGS}
    s['analysis']['primary']={'estimate_percentage_points':0.}
    return s


def test_complete_qualified_scope_is_supported_without_plotting_means(tmp_path):
    s=complete_fixture();result=m.render(s,tmp_path)
    assert result['qualified_contexts']=={'core':24,'failed-near-edge':24}
    assert result['setting_means_plotted'] is False
    assert 'All 48 contexts qualified' in result['status']
    assert result['reused_regression_phases'] and result['fresh_holdout'] is False


def test_lost_analysis_keeps_planned_cells_and_unknown_periods(tmp_path):
    s=fixture();s['analysis']=None
    result=m.render(s,tmp_path)
    assert result['phase_rows']==576 and result['signed_margin_rows']==576
    assert result['periods']=={'core':None,'failed-near-edge':None}
    assert result['qualified_contexts']=={'core':0,'failed-near-edge':0}
    assert 'Analysis unavailable' in result['status']


def test_complete_but_unqualified_context_remains_masked_and_csv_retains_value():
    s=fixture()
    for r in s['analysis']['balanced_accuracies']:
        if r['phase_index']==9:r.update(complete=True,with_second=.9,without_second=.95,interference_loss=.05)
    for setting in m.SETTINGS:s['analysis']['contexts'][f'{setting}--p09']['complete']=True
    balances,_,_=m.collect(s)
    assert balances['core',9,'identity','k4']['with_second']==.9
    assert np.isnan(m.matrix(balances,'core',[('identity','k4','with_second')])[0,9])


def test_failed_source_suppresses_all_numerical_cells():
    s=complete_fixture();s['source_valid']=False
    balances,_,contexts=m.collect(s)
    assert np.isnan(m.matrix(balances,'core',[('identity','k4','with_second')])).all()
    assert not any(r['qualified'] for r in contexts.values())
    assert all(r['saved_qualified'] for r in balances.values())


def test_existing_figure_outputs_are_never_overwritten(tmp_path):
    target=tmp_path/'identity-readout-by-phase.png';target.write_bytes(b'immutable existing artifact')
    with pytest.raises(FileExistsError):m.render(fixture(),tmp_path)
    assert target.read_bytes()==b'immutable existing artifact'


@pytest.mark.parametrize('edit',[
    lambda s:s.update(experiment_id='EXP-016'),
    lambda s:s.update(fresh_holdout=True),
    lambda s:s['analysis']['settings']['core'].update(means=[]),
    lambda s:s['analysis'].update(primary={'estimate_percentage_points':12.}),
    lambda s:s['analysis']['scores'][0]['task'].update(method='forged'),
    lambda s:s['analysis']['scores'][0]['task'].update(first_sign=True),
    lambda s:s['analysis']['balanced_accuracies'][0].update(split='heldout'),
])
def test_scope_favorable_aggregate_and_identity_tampering_rejected(edit):
    s=fixture();edit(s)
    with pytest.raises(ValueError):m.collect(s)


def test_saved_clock_inconsistency_rejected():
    s=fixture();_,fine,_=m.collect(s)
    row=next(iter(fine.values()));row['windows']['k4']['times'][0]+=.1
    with pytest.raises(ValueError):m.periods_from_saved_clocks(fine)


def sealed_fixture(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();summary=raw/'summary.json'
    summary.write_text(json.dumps(fixture())+'\n')
    receipt=dict(experiment_id='EXP-019',source_commit='a'*40,plan_sha256='b'*64,
        artifacts=[dict(path='summary.json',bytes=summary.stat().st_size,sha256=m.sha(summary))])
    (raw/'receipt.json').write_text(json.dumps(receipt)+'\n')
    return raw,m.sha(raw/'receipt.json')


def test_receipt_bound_run_retains_outputs_and_optional_audit_without_inventing_verdict(tmp_path):
    raw,digest=sealed_fixture(tmp_path)
    audit=tmp_path/'audit.json';audit.write_text('{"strict_arithmetic_pass":false}\n')
    result=m.run(raw,tmp_path/'figures',digest,audit_report=audit,expected_audit=m.sha(audit))
    assert result['raw_receipt']['sha256']==digest
    assert result['associated_audit_report']['sha256']==m.sha(audit)
    assert 'neither asserts its verdict' in result['audit_scope']
    assert len(result['outputs'])==4
    assert result['suppression']['setting_means_never_plotted']
    assert (tmp_path/'figures'/'receipt.json').exists()
    with pytest.raises(FileExistsError):m.run(raw,tmp_path/'figures',digest)


@pytest.mark.parametrize('kind',['summary','receipt','duplicate'])
def test_hash_or_summary_census_change_is_rejected_before_output(tmp_path,kind):
    raw,digest=sealed_fixture(tmp_path)
    if kind=='summary':(raw/'summary.json').write_text('{}\n')
    elif kind=='receipt':digest='0'*64
    else:
        receipt=m.read(raw/'receipt.json');receipt['artifacts']*=2
        (raw/'receipt.json').write_text(json.dumps(receipt));digest=m.sha(raw/'receipt.json')
    with pytest.raises(ValueError):m.run(raw,tmp_path/'figures',digest)
    assert not (tmp_path/'figures').exists()


def test_no_output_inside_a_sealed_evidence_directory(tmp_path):
    raw,digest=sealed_fixture(tmp_path)
    with pytest.raises(ValueError):m.run(raw,raw/'new',digest)
    other=tmp_path/'other';other.mkdir();(other/'receipt.json').write_text('{}\n')
    with pytest.raises(ValueError):m.run(raw,other/'new',digest)
