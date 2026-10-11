"""Synthetic saved-summary tests; no producer imports or research data reads."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import plot_stable_mixture_readout_saved as plot


def fixture():
    cells=[];means={s:[] for s in plot.SETTINGS}
    for si,setting in enumerate(plot.SETTINGS):
        for phase in range(24):
            for wi,window in enumerate(plot.WINDOWS):
                cid=f'{setting}--p{phase:02d}';ident=cid+'--'+window
                cell=dict(id=ident,context_id=cid,setting=setting,phase_index=phase,
                    phase_fraction=(phase+.25)/24,split='development' if phase<8 else 'heldout',window=window)
                old=.51+.03*si+.001*phase-.005*wi
                full=old+.07;one=old+.025
                group=dict(cell_id=ident,method='rk4-fine',original_linear=old,original_linear_interval=[old,old],
                    primary=dict(complete=True,qualified=True,nominal=full,interval=[full-2e-5,full+2e-5]),
                    one_mode=dict(complete=True,qualified=True,nominal=one,interval=[one-2e-5,one+2e-5]))
                cells.append(dict(id=ident,cell=cell,complete=True,qualified=True,secondary_complete=True,
                    secondary_qualified=True,groups=[group]))
    settings={}
    for setting in plot.SETTINGS:
        records={r:[] for r in ['primary','one_mode']}
        for recipe in records:
            for window in plot.WINDOWS:
                selected=[p['groups'][0] for p in cells if p['cell']['setting']==setting and p['cell']['window']==window]
                records[recipe].append(dict(window=window,phases=24,
                    original_linear=sum(p['original_linear'] for p in selected)/24,
                    nominal_accuracy=sum(p[recipe]['nominal'] for p in selected)/24,
                    accuracy_interval=[sum(p[recipe]['interval'][i] for p in selected)/24 for i in (0,1)]))
        settings[setting]=dict(cells=72,qualified=True,secondary_qualified=True,
            means=records['primary'],secondary_means=records['one_mode'])
    return dict(experiment_id='EXP-021',complete=True,qualified=True,source_valid=True,
        analysis=dict(cells=cells,settings=settings,primary=dict(nominal_gain=.07)))


def suppress_setting(saved,setting='core'):
    saved['qualified']=False;saved['analysis']['primary']=None
    status=saved['analysis']['settings'][setting]
    status.update(qualified=False,secondary_qualified=False,means=None,secondary_means=None)
    for packet in saved['analysis']['cells']:
        if packet['cell']['setting']==setting:
            packet['qualified']=False;packet['secondary_qualified']=False
            for recipe in ('primary','one_mode'):packet['groups'][0][recipe]['qualified']=False


def test_exact_census_and_no_phase_selection():
    data=plot.collect(fixture())
    assert len(data['phase_rows'])==432 and len(data['aggregate_rows'])==18
    assert all(r['displayed'] for r in data['phase_rows'])
    for setting in plot.SETTINGS:
        for window in plot.WINDOWS:
            for recipe in plot.RECIPES:
                assert {r['phase_index'] for r in data['phase_rows'] if r['setting']==setting and r['window']==window and r['recipe']==recipe}==set(range(24))


def test_unqualified_setting_suppresses_means_not_known_old_phase_values():
    saved=fixture();suppress_setting(saved)
    data=plot.collect(saved)
    assert all(not r['displayed'] for r in data['aggregate_rows'] if r['setting']=='core')
    assert all(r['displayed'] for r in data['phase_rows'] if r['setting']=='core' and r['recipe']=='original_linear')
    assert all(not r['displayed'] for r in data['phase_rows'] if r['setting']=='core' and r['recipe']!='original_linear')
    assert not data['primary_reported']


def test_secondary_failure_keeps_primary_figure_values():
    saved=fixture();status=saved['analysis']['settings']['core']
    status.update(secondary_qualified=False,secondary_means=None)
    for p in saved['analysis']['cells']:
        if p['cell']['setting']=='core':
            p['secondary_qualified']=False;p['groups'][0]['one_mode']['qualified']=False
    data=plot.collect(saved)
    assert all(r['displayed'] for r in data['aggregate_rows'] if r['setting']=='core' and r['recipe']!='one_mode')
    assert all(not r['displayed'] for r in data['aggregate_rows'] if r['setting']=='core' and r['recipe']=='one_mode')


def test_invalid_source_suppresses_all_values():
    saved=fixture();saved.update(source_valid=False,qualified=False,complete=False)
    data=plot.collect(saved)
    assert not any(r['displayed'] for r in data['phase_rows']+data['aggregate_rows'])
    assert all(r['nominal'] is not None for r in data['phase_rows'])


def test_unavailable_analysis_uses_bound_planned_positions_without_values():
    original=fixture();planned=[p['cell'] for p in original['analysis']['cells']]
    original.update(analysis={'status':'unavailable'},complete=False,qualified=False)
    data=plot.collect(original,planned)
    assert len(data['phase_rows'])==432
    assert not any(r['displayed'] for r in data['phase_rows']+data['aggregate_rows'])
    assert all(r['nominal'] is None for r in data['phase_rows'])


@pytest.mark.parametrize('mutate',[
    lambda x:x['analysis']['cells'].pop(),
    lambda x:x['analysis']['cells'].__setitem__(1,copy.deepcopy(x['analysis']['cells'][0])),
    lambda x:x['analysis']['cells'][0]['cell'].update(phase_index=True),
    lambda x:x['analysis']['cells'][0]['cell'].update(phase_fraction=float('nan')),
    lambda x:x['analysis']['cells'][0]['cell'].update(phase_fraction=.99),
    lambda x:x['analysis']['cells'][0]['groups'][0].update(method='rk4-coarse'),
    lambda x:x['analysis']['cells'][0]['groups'][0]['primary'].update(interval=[-.1,1]),
    lambda x:x['analysis']['cells'][0]['groups'][0]['primary'].update(nominal=float('nan')),
    lambda x:x['analysis']['cells'][0].update(complete=False),
    lambda x:x['analysis']['settings']['core'].update(qualified=False),
    lambda x:x['analysis']['settings']['core']['means'][0].update(phases=23),
    lambda x:x['analysis']['settings']['core']['means'][0].update(nominal_accuracy=.99),
])
def test_malformed_ledger_rejected(mutate):
    saved=fixture();mutate(saved)
    with pytest.raises(ValueError):plot.collect(saved)


def write_raw(tmp_path):
    raw=tmp_path/'synthetic-raw';raw.mkdir()
    source='a'*40;plan='b'*64
    saved=fixture()
    manifest=dict(source_commit=source,plan_sha256=plan,plan={'cells':[p['cell'] for p in saved['analysis']['cells']]})
    for name,obj in [('summary.json',saved),('manifest.json',manifest)]:
        (raw/name).write_text(json.dumps(obj))
    inventory=[dict(path=n,bytes=(raw/n).stat().st_size,sha256=plot.digest(raw/n)) for n in ['summary.json','manifest.json']]
    receipt=dict(experiment_id='EXP-021',source_commit=source,plan_sha256=plan,qualified=True,status='complete',artifacts=inventory)
    (raw/'receipt.json').write_text(json.dumps(receipt))
    return raw,dict(expected_receipt=plot.digest(raw/'receipt.json'),expected_source=source,expected_plan=plan)


def test_actual_render_hash_binding_and_csv_census(tmp_path):
    raw,args=write_raw(tmp_path);out=tmp_path/'rendered'
    result=plot.render(raw,out,**args)
    assert result['phase_rows']==432 and result['aggregate_rows']==18
    assert len((out/'phase-readouts.csv').read_text().splitlines())==433
    assert len((out/'aggregate-readouts.csv').read_text().splitlines())==19
    assert all((out/p['path'].split('/')[-1]).stat().st_size==p['bytes'] for p in result['outputs'])
    assert all(p['sha256']==plot.digest(p['path']) for p in result['outputs'])
    with pytest.raises(ValueError):plot.render(raw,out,**args)


def test_failed_setting_render_is_explicit(tmp_path):
    saved=fixture();suppress_setting(saved)
    plot.figures(plot.collect(saved),tmp_path)
    assert (tmp_path/'phase-resolved-readouts.png').stat().st_size>10000
    assert (tmp_path/'full-cohort-readout-averages.png').stat().st_size>10000


@pytest.mark.parametrize('bad',['hash','source','plan','input','nested'])
def test_evidence_and_destination_guards(tmp_path,bad):
    raw,args=write_raw(tmp_path);out=tmp_path/'output'
    if bad=='hash':args['expected_receipt']='0'*64
    if bad=='source':args['expected_source']='0'*40
    if bad=='plan':args['expected_plan']='0'*64
    if bad=='input':(raw/'summary.json').write_text('{}')
    if bad=='nested':out=raw/'figure-output'
    with pytest.raises(ValueError):plot.render(raw,out,**args)
    assert not out.exists()
