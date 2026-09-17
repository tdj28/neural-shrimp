"""Synthetic summaries/figures only; never read a live EXP014 outcome."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import plot_phase_memory as p


def synthetic_summary():
    rows=[]
    for setting,phase,dose,direction,method,delay,sigma in p.expected_cells():
        value=.52+.01*phase+.03*dose+.02*p.DIRECTIONS.index(direction)/delay
        rows.append(dict(setting=setting,phase_index=phase,dose_index=dose,direction=direction,method=method,
            delay_cycles=delay,sigma=sigma,status='complete',balanced_accuracy=value,
            phase_only_balanced_accuracy=value-.03,full_linear_balanced_accuracy=value-.01,zero_balanced_accuracy=.5))
    terminal=[];comparisons=[];ba=[]
    pairs=((p.METHODS[0],p.METHODS[1]),(p.METHODS[1],p.METHODS[2]))
    for setting in p.SETTINGS:
        cases=[]
        for phase in range(8):
            cases.extend(f'{setting}--p{phase}--d{dose}--{direction}--s{sign:+d}' for dose,direction,sign in p.itertools.product(range(3),p.DIRECTIONS,(-1,1)))
            cases.append(f'{setting}--p{phase}--sham')
        terminal.extend(dict(id=case+'--'+method,setting=setting,status='complete',sham_pass=True) for case,method in p.itertools.product(cases,p.METHODS))
        comparisons.extend(dict(ids=[case+'--'+a,case+'--'+b],setting=setting,qualified=True) for case in cases for a,b in pairs)
        ba.extend(dict(setting=setting,phase_index=phase,dose_index=dose,direction=direction,methods=list(methods),delay_cycles=delay,sigma=sigma,qualified=True)
                  for phase,dose,direction,methods,delay,sigma in p.itertools.product(range(8),range(3),p.DIRECTIONS,pairs,p.DELAYS,p.NOISES))
    science=dict(paired_readouts=rows,settings={s:dict(complete=True,qualified=True) for s in p.SETTINGS},terminal_rows=terminal,
        trace_comparisons=comparisons,balanced_accuracy_comparisons=ba,prerequisites={s:dict(reference=True,propagators=[True]*8) for s in p.SETTINGS})
    return dict(experiment_id='EXP-014',qualified=True,complete=True,error=None,postflight_error=None,scientific=science)


@pytest.fixture()
def summary():return synthetic_summary()


def test_all_cells_and_exact_eight_phase_means(summary):
    report=p.collect(summary)
    assert len(report['rows'])==5184 and len(report['means'])==648
    assert all(r['phase_count']==8 for r in report['means'])
    assert all(abs(r['actual_minus_phase_only']-.03)<1e-15 for r in report['means'])
    assert all(v['display_qualified'] for v in report['settings'].values())


def test_failed_setting_gets_no_means_while_other_can_qualify(summary):
    summary['qualified']=False;summary['scientific']['settings']['core']['qualified']=False
    report=p.collect(summary)
    assert not report['settings']['core']['display_qualified']
    assert report['settings']['failed-near-edge']['display_qualified']
    assert len(report['means'])==324 and all(r['setting']=='failed-near-edge' for r in report['means'])
    assert len(report['rows'])==5184


@pytest.mark.parametrize('mutation',['missing','duplicate','nan','outofrange','boolphase','terminal','sham','comparison','bacomparison','prerequisite','postflight','producer'])
def test_incomplete_or_forged_qualification_suppresses_means(summary,mutation):
    science=summary['scientific']
    if mutation=='missing':science['paired_readouts'].pop(0)
    elif mutation=='duplicate':science['paired_readouts'].append(copy.deepcopy(science['paired_readouts'][0]))
    elif mutation=='nan':science['paired_readouts'][0]['balanced_accuracy']=float('nan')
    elif mutation=='outofrange':science['paired_readouts'][0]['balanced_accuracy']=1.1
    elif mutation=='boolphase':science['paired_readouts'][0]['phase_index']=False
    elif mutation=='terminal':science['terminal_rows'][0]['status']='failed'
    elif mutation=='sham':science['terminal_rows'][0]['sham_pass']=False
    elif mutation=='comparison':science['trace_comparisons'][0]['qualified']=False
    elif mutation=='bacomparison':science['balanced_accuracy_comparisons'][0]['qualified']=False
    elif mutation=='prerequisite':science['prerequisites']['core']['reference']=False
    elif mutation=='postflight':summary['postflight_error']='drift'
    else:summary['error']='failed'
    report=p.collect(summary)
    assert len(report['rows'])==5184 and not report['settings']['core']['display_qualified']
    assert not any(r['setting']=='core' for r in report['means'])


def test_real_png_pdf_report_paths_for_qualified_and_diagnostic_settings(summary,tmp_path):
    summary['qualified']=False;summary['scientific']['settings']['core']['qualified']=False
    next(r for r in summary['scientific']['paired_readouts'] if r['setting']=='core' and r['method']==p.PRIMARY and r['sigma']==1e-4)['status']='unavailable'
    summary['scientific']['trace_comparisons'][0]['qualified']=False
    output=tmp_path/'report'
    result=p.write_report(summary,{'synthetic':True},output)
    assert not result['display_qualified']['core'] and result['display_qualified']['failed-near-edge']
    assert len(list(output.glob('*.png')))==9 and len(list(output.glob('*.pdf')))==9
    assert all(path.stat().st_size>1000 for path in output.glob('*.png'))
    saved=json.loads((output/'report.json').read_text())
    assert len(saved['rows'])==5184 and all(r['setting']!='core' for r in saved['means'])
    assert 'DIAGNOSTIC ONLY' in (output/'README.md').read_text()
    with pytest.raises(FileExistsError):p.write_report(summary,{},output)


def test_output_never_inside_raw_or_repository(summary,tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();raw=tmp_path/'raw';raw.mkdir();monkeypatch.setattr(p,'ROOT',root)
    for destination in (root/'reports',raw/'reports'):
        with pytest.raises(ValueError):p.write_report(summary,{},destination,source_directory=raw)


def test_hash_bound_summary_manifest_postflight_loading(summary,tmp_path):
    source='a'*40;plan='b'*64
    for name,value in [('summary.json',summary),('manifest.json',dict(source_commit=source,plan_sha256=plan)),('postflight.json',dict(error=None))]:
        (tmp_path/name).write_text(json.dumps(value))
    artifacts=[dict(path=x.name,bytes=x.stat().st_size,sha256=p.sha(x)) for x in sorted(tmp_path.iterdir())]
    receipt=dict(source_commit=source,plan_sha256=plan,artifacts=artifacts,experiment_id='EXP-014')
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    restored,provenance=p.load_saved(tmp_path)
    assert restored==summary and len(provenance['inputs'])==3
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash/bytes'):p.load_saved(tmp_path)


def test_failed_comparison_marked_in_diagnostic_rows_without_means(summary):
    summary['scientific']['trace_comparisons'][0]['qualified']=False
    report=p.collect(summary)
    failed=[row for row in report['rows'] if row['numerical_failure']]
    assert len(failed)==36  # One signed pair×3methods×4delays×3noise.
    assert all((r['setting'],r['phase_index'],r['dose_index'],r['direction'])==('core',0,0,'phase_write') for r in failed)
    assert not report['settings']['core']['display_qualified']
