"""Synthetic EXP020 lifecycle tests: no historical arrays or target decoders.

The real codecs, phase/slot census and owned scheduler are retained. Synthetic
metadata files use opaque bytes; numerical probability routines are never used.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
HEAD='a'*40


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


@pytest.fixture
def runner(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('synthetic_mixture_runner',TOOLS/'run_mixture_readout.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.ROOT=tmp_path/'checkout';r.ROOT.mkdir();r.OLD=tmp_path/'old';r.OLD.mkdir()
    r.binding.__defaults__=(r.ROOT,);r.SOURCE_PATHS=['synthetic-source.py']
    (r.ROOT/r.SOURCE_PATHS[0]).write_text('# synthetic source\n')
    r.evaluation=SimpleNamespace(**{name:getattr(r.evaluation,name) for name in ('cell_census','calculation_census','summarize')},contract=lambda:{'synthetic':True})
    relatives=['manifest.json','summary.json','analysis.json','analysis.npz','input-seal.json','input-seal.npz','reference-seal-index.json','global-reference-gate.json']
    for setting in r.CONTRACT['settings']:
        for phase in range(24):
            cid=f'{setting}--p{phase:02d}'
            relatives += [f'seals/{cid}.{ext}' for ext in ('json','npz')]
            relatives += [f'references/payloads/{cid}--reference--{method}/packet.{ext}' for method in ('dop853','rk45') for ext in ('json','npz')]
    for name in relatives:
        p=r.OLD/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'opaque synthetic bytes')
    write(r.OLD/'manifest.json',dict(source_commit=r.OLD_SOURCE,plan_sha256=r.OLD_PLAN_SHA,plan={'phases':r.phases()}))
    artifacts=[r.binding(r.OLD/name,r.OLD) for name in relatives]
    write(r.OLD/'receipt.json',dict(status='complete',source_commit=r.OLD_SOURCE,plan_sha256=r.OLD_PLAN_SHA,artifacts=artifacts))
    r.OLD_RECEIPT_SHA=r.legacy.sha(r.OLD/'receipt.json')
    compact=r.ROOT/r.COMPACT;compact.parent.mkdir(parents=True,exist_ok=True);compact.write_bytes(b'opaque finite-outcome summary')
    r.COMPACT_SHA=r.legacy.sha(compact)
    q=dict(complete=True,source_valid=True,contexts_planned=48,contexts_qualified=48,scores_planned=2700,scores_complete=2700,
        comparisons_planned=1404,comparisons_qualified=1404,all_primary_setting_prerequisites_qualified=True,global_reference_barrier_pass=True,
        frozen_saved_replay_pass=True,owned_process_cleanup_verified=True,reference_evidence_and_gate_pass=True,finite_evidence_and_gate_pass=True,
        finite_strict_arithmetic_pass=True,reference_alternate_strict_arithmetic_pass=False,reference_alternate_failed_comparisons=165,strict_thresholds_unchanged=True)
    write(r.ROOT/r.CERTIFICATE,dict(source_commit=r.OLD_SOURCE,plan_sha256=r.OLD_PLAN_SHA,raw_receipt={'sha256':r.OLD_RECEIPT_SHA},
        compact_summary={'sha256':r.COMPACT_SHA},raw_summary={'sha256':r.legacy.sha(r.OLD/'summary.json')},no_performance_outcomes=True,qualification=q,audit_bindings=[]))
    r.CERTIFICATE_SHA=r.legacy.sha(r.ROOT/r.CERTIFICATE)
    write(r.ROOT/'docs/EXP-020-runtime-provenance.json',{'synthetic':True})
    return r


def seal(cell):
    return dict(id=cell['id'],cell=cell,context_id=cell['context_id'],complete=True,status='complete',
        definitions={k:dict(available=True,complete=True,definition={'synthetic':True}) for k in ('primary','alternate','one_mode')},
        h=np.array([1.,0.]),baseline=np.zeros(2))


def terminal(task,status='not_started_synthetic',**fields):
    return dict(id=task['id'],case_id=task['case']['id'],kind=task['kind'],method=task['method']['id'],status=status,qualified=False,**fields)


def test_complete_census_fixed_phases_and_4704_slots(runner):
    r=runner;specs=r.tasks();cells=[t['case'] for t in specs]
    assert r.phases()==r.previous.phase_rows()
    assert len(cells)==len({c['id'] for c in cells})==144
    slots=sum([r.evaluation.calculation_census(c) for c in cells],[])
    assert len(slots)==len({s['id'] for s in slots})==4704
    assert sum(s['family']=='finite' and s['recipe']=='primary' for s in slots)==1200
    assert sum(s['family']=='finite' and s['recipe']=='one_mode' for s in slots)==1200
    assert sum(s['family']=='alternate' for s in slots)==576
    assert sum(s['family']=='reference' for s in slots)==1728
    assert r.CONTRACT['auxiliary_linear_reference_calculations']==1728
    assert r.CONTRACT['mixture_plus_auxiliary_probability_calculations']==4704+1728
    assert {c['phase_index'] for c in cells if c['split']=='heldout'}==set(range(8,24))
    assert r.CONTRACT['fresh_holdout'] is False


def test_metadata_preparation_hashes_299_files_without_array_or_outcome_decode(runner,monkeypatch):
    r=runner
    def forbidden(*args,**kwargs):pytest.fail('Metadata preparation decoded arrays or outcomes')
    monkeypatch.setattr(r.codec,'load_packet',forbidden);monkeypatch.setattr(r.np,'load',forbidden)
    original=r.read
    def read(path):
        if Path(path) in [r.OLD/'summary.json',r.OLD/'analysis.json',r.ROOT/r.COMPACT]:forbidden()
        return original(path)
    r.read=read
    plan=r.build_plan()
    assert len(plan['historical_inputs'])==299
    assert r.validate(plan)['probability_calculations']==4704
    bad=copy.deepcopy(plan);bad['cells'][0]['phase_fraction']+=.01
    with pytest.raises(ValueError):r.validate(bad)


@pytest.mark.parametrize('change',['raw_array','raw_summary','receipt','compact','certificate','qualification'])
def test_prior_binding_or_qualification_changes_rejected(runner,change):
    r=runner
    if change=='raw_array':(r.OLD/'analysis.npz').write_bytes(b'changed')
    elif change=='raw_summary':(r.OLD/'summary.json').write_bytes(b'changed')
    elif change=='receipt':write(r.OLD/'receipt.json',{})
    elif change=='compact':(r.ROOT/r.COMPACT).write_bytes(b'changed')
    elif change=='certificate':write(r.ROOT/r.CERTIFICATE,{})
    else:
        p=r.ROOT/r.CERTIFICATE;v=r.read(p);v['qualification']['contexts_qualified']=47;write(p,v);r.CERTIFICATE_SHA=r.legacy.sha(p)
    with pytest.raises((ValueError,KeyError)):r.old_inputs()


def source_fixture(r,monkeypatch):
    path=r.ROOT/r.PLAN_REL;write(path,{'synthetic':True})
    write(r.ROOT/r.REVIEW_REL,dict(status='ready_for_exploratory_execution',plan_sha256=r.legacy.sha(path)))
    state=dict(dirty='',head=HEAD,branch='codex/synthetic',remote=HEAD,origin='git@github-personal:tdj28/neural-shrimp.git')
    committed={name:(r.ROOT/name).read_bytes() for name in [*r.SOURCE_PATHS,r.PLAN_REL,r.REVIEW_REL]}
    monkeypatch.setattr(r.legacy,'git',lambda *args:state[{'status':'dirty','rev-parse':'head','branch':'branch','remote':'origin','ls-remote':'remote'}[args[0]]])
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:committed[cmd[-1].split(':',1)[1]])
    return path,state,committed


def test_exact_committed_pushed_source_guard(runner,monkeypatch):
    path,_,_=source_fixture(runner,monkeypatch)
    assert runner.source_guard({},path,HEAD)['source_commit']==HEAD


@pytest.mark.parametrize('field,value',[('dirty','?? extra.py'),('head','b'*40),('branch','main'),('remote','c'*40),('origin','https://elsewhere/repo')])
def test_source_guard_rejects_unfrozen_states(runner,monkeypatch,field,value):
    path,state,_=source_fixture(runner,monkeypatch);state[field]=value
    with pytest.raises(ValueError):runner.source_guard({},path,HEAD)


def test_source_guard_requires_exact_review(runner,monkeypatch):
    r=runner;path,_,committed=source_fixture(r,monkeypatch)
    review=r.read(r.ROOT/r.REVIEW_REL);review['plan_sha256']='f'*64;write(r.ROOT/r.REVIEW_REL,review)
    committed[r.REVIEW_REL]=(r.ROOT/r.REVIEW_REL).read_bytes()
    with pytest.raises(ValueError,match='Exact-plan'):r.source_guard({},path,HEAD)


def test_external_exclusive_attempt_cannot_extend_old_or_alias_checkout(runner,tmp_path):
    r=runner;raw=tmp_path/'raw';path=r.external_attempt(raw,'attempt-001')
    with pytest.raises(FileExistsError):r.external_attempt(raw,'attempt-001')
    prior=tmp_path/'EXP-017'/'attempt-001';prior.mkdir(parents=True)
    for bad in [r.ROOT,r.ROOT/'new',r.OLD,r.OLD/'nested',prior/'nested',path]:
        with pytest.raises(ValueError):r.external_attempt(bad,'attempt-002')
    other=tmp_path/'aliased';other.mkdir();(other/'EXP-020').symlink_to(r.ROOT,target_is_directory=True)
    with pytest.raises(ValueError):r.external_attempt(other,'attempt-001')


def definitions_fixture(r,path):
    cells=[t['case'] for t in r.tasks()];definitions={c['id']:seal(c) for c in cells}
    write(path/'input-index.json',{'synthetic':True})
    for ident,p in definitions.items():r.codec.save_packet(path/'definitions',ident,p)
    write(path/'definition-index.json',r.definition_index(path,cells,definitions))
    return cells,definitions


def test_all432_definitions_roundtrip_before_barrier(runner,tmp_path):
    cells,definitions=definitions_fixture(runner,tmp_path/'attempt')
    gate=runner.barrier_packet(tmp_path/'attempt',cells,definitions)
    assert gate['passed'] and gate['classifier_definitions']==432
    definitions.pop(cells[-1]['id'])
    with pytest.raises(ValueError):runner.barrier_packet(tmp_path/'attempt',cells,definitions)


@pytest.mark.parametrize('change',['array_bytes','valid_replacement','index','identity'])
def test_definition_barrier_reloads_disk_arrays_not_cached_packets(runner,tmp_path,change):
    r=runner;path=tmp_path/'attempt';cells,definitions=definitions_fixture(r,path);cell=cells[0]
    p=path/'definitions'/f'{cell["id"]}.json'
    if change=='array_bytes':p.with_suffix('.npz').write_bytes(b'corrupt')
    elif change=='valid_replacement':
        packet=copy.deepcopy(definitions[cell['id']]);packet['h'][0]=2.
        p.unlink();p.with_suffix('.npz').unlink();r.codec.save_packet(p.parent,p.stem,packet)
    elif change=='index':write(path/'definition-index.json',{})
    else:definitions[cell['id']]['context_id']='wrong'
    with pytest.raises((ValueError,KeyError,OSError)):r.barrier_packet(path,cells,definitions)


def runtime_fixture(r,tmp_path,monkeypatch):
    plan=r.build_plan();path=r.ROOT/r.PLAN_REL;write(path,plan)
    write(r.ROOT/r.REVIEW_REL,{'synthetic':True})
    provenance=dict(source_commit=HEAD,pushed_branch='codex/synthetic',review=r.binding(r.ROOT/r.REVIEW_REL))
    monkeypatch.setattr(r,'source_guard',lambda *a:provenance)
    monkeypatch.setattr(r,'runtime_check',lambda:None)
    monkeypatch.setattr(r,'resource_check',lambda *a,**k:{})
    monkeypatch.setattr(r.legacy,'git',lambda *a:HEAD if a[0]=='rev-parse' else '')
    for key in r.legacy.THREAD_ENV:monkeypatch.setenv(key,'1')
    r.reference_input=lambda cell:({}, {})
    r.evaluation.seal_cell=lambda cell,*args:seal(cell)
    r.evaluation.summarize=lambda cells,definitions,results:dict(complete=False,qualified=False,definitions_present=len(definitions),results_present=len(results),counts={'cells':144,'total_new_probability_calculations':4704})
    r.finite_observations=lambda plan,directory,cells,defs:{c['id']:{'synthetic':True} for c in cells}
    r.runtime=SimpleNamespace(execute=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('synthetic dispatch stop')))
    return plan,path,tmp_path/'output'


@pytest.mark.parametrize('stage',['launch','input_event','saved_definition','gate_event','dispatch'])
def test_failed_stages_preserve_full_denominators_and_available_definition_prefix(runner,tmp_path,monkeypatch,stage):
    r=runner;plan,path,raw=runtime_fixture(r,tmp_path,monkeypatch);original_event=r.event;original_save=r.codec.save_packet
    if stage=='launch':r.resource_check=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('headroom'))
    if stage in ('input_event','gate_event'):
        wanted='reference_inputs_bound' if stage=='input_event' else 'classifiers_sealed'
        def event(directory,name,**fields):
            if name==wanted:raise r.ProducerStop('synthetic pre-event interruption')
            return original_event(directory,name,**fields)
        r.event=event
    if stage=='saved_definition':
        def save(directory,name,packet):
            result=original_save(directory,name,packet)
            if Path(directory).name=='definitions':raise r.ProducerStop('after packet before assignment')
            return result
        monkeypatch.setattr(r.codec,'save_packet',save)
    outcome=r.run(path,raw,'attempt-001',HEAD);directory=Path(outcome['directory'])
    assert outcome['summary']['complete'] is False
    assert len(r.read(directory/'all-terminal-records.json'))==144
    assert sum(len(v) for v in r.read(directory/'planned-calculations.json').values())==4704
    analysis=r.codec.load_packet(directory/'analysis.json')
    expected={'launch':0,'input_event':0,'saved_definition':1,'gate_event':144,'dispatch':144}[stage]
    assert analysis['summary']['definitions_present']==expected
    assert r.read(directory/'receipt.json')['artifacts']==r.inventory(directory)
    if stage in ('launch','input_event','saved_definition','gate_event'):assert not (directory/'observations').exists()


def test_corrupted_global_gate_blocks_every_finite_read(runner,tmp_path,monkeypatch):
    r=runner;_,path,raw=runtime_fixture(r,tmp_path,monkeypatch);original=r.read
    def read(p):
        value=original(p)
        if Path(p).name=='global-definition-gate.json':value['classifier_definitions']=431
        return value
    r.read=read;r.finite_observations=lambda *a:pytest.fail('Finite access occurred after corrupted gate')
    result=r.run(path,raw,'attempt-001',HEAD)
    assert not result['summary']['complete']
    assert 'readback differs' in result['summary']['error']
    assert not (Path(result['directory'])/'observations').exists()


def test_finite_access_occurs_only_after_all432_definitions_and_matching_event(runner,tmp_path,monkeypatch):
    r=runner;_,path,raw=runtime_fixture(r,tmp_path,monkeypatch);seen=[]
    def finite(plan,directory,cells,defs):
        event=r.require_barrier(directory,cells,defs)
        assert event['classifier_definitions']==432 and len(list((directory/'definitions').glob('*.json')))==144
        seen.append(True)
        return {c['id']:{'synthetic':True} for c in cells}
    r.finite_observations=finite
    r.run(path,raw,'attempt-001',HEAD)
    assert seen==[True]


def test_authoritative_timeout_retains_one_returned_calculation_and_never_promotes_packet(runner,tmp_path):
    r=runner;task=r.tasks()[0];cell=task['case'];sealed=seal(cell)
    d=r.codec.save_packet(tmp_path/'inputs','definition',sealed)
    o=r.codec.save_packet(tmp_path/'inputs','observation',dict(cell=cell,scores={}))
    slots=r.evaluation.calculation_census(cell)
    def evaluate(cell,sealed,scores,progress_fn):
        progress_fn(dict(slots[0],result={'synthetic':True}))
        time.sleep(10)
    def assemble(cell,sealed,scores,calculations):
        # Deliberately claim complete to prove the parent's timeout is authoritative.
        return dict(id=cell['id'],cell=cell,complete=True,qualified=True,calculations=calculations)
    r.evaluation.evaluate_cell=evaluate;r.evaluation.assemble_cell=assemble
    context=dict(definition_path=d['path'],definition_semantic_sha256=d['semantic_sha256'],observation_path=o['path'],observation_semantic_sha256=o['semantic_sha256'])
    rows=r.runtime.execute([task],{cell['id']:context},tmp_path/'cells',{},workers=1,deadline=time.monotonic()+5,per_task=.15,worker_fn=r.worker)
    assert rows[0]['status']=='task_timeout'
    assert r.read(tmp_path/'cells/cleanup.json')['all_original_identities_absent'] is True
    trusted,partial=r.collect(tmp_path,[task],rows)
    assert trusted=={} and partial[0]['packet_complete'] is True
    assert partial[0]['returned_calculation_ids']==[slots[0]['id']]
    assert partial[0]['qualified'] is False


def test_complete_parent_requires_exact_full_calculation_journal(runner,tmp_path):
    r=runner;task=r.tasks()[0];cell=task['case'];stage=tmp_path/'cells';dest=stage/'payloads'/task['id']
    slots=[dict(s,result={'synthetic':True}) for s in r.evaluation.calculation_census(cell)]
    packet=dict(id=cell['id'],cell=cell,complete=True,qualified=True,calculations=slots)
    saved=r.codec.save_packet(dest,'packet',packet)
    row=terminal(task,'complete',pid=os.getpid(),packet_path=str(Path(saved['path']).relative_to(stage)),packet_sha256=saved['sha256'],packet_semantic_sha256=saved['semantic_sha256'])
    for slot in slots[:-1]:r.work.append(dest/'calculation-journal.jsonl',slot)
    with pytest.raises((ValueError,AssertionError)):r.collect(tmp_path,[task],[row])
    r.work.append(dest/'calculation-journal.jsonl',slots[-1])
    assert set(r.collect(tmp_path,[task],[row])[0])=={cell['id']}


def test_missing_and_duplicate_parent_ledgers_rejected(runner,tmp_path):
    r=runner;specs=r.tasks();rows=[terminal(t) for t in specs]
    assert r.verify_lifecycle(tmp_path,specs,rows)==[]
    with pytest.raises(ValueError):r.verify_lifecycle(tmp_path,specs,rows[:-1])
    rows[-1]=rows[0]
    with pytest.raises(ValueError):r.verify_lifecycle(tmp_path,specs,rows)


def test_failed_attempt_prefix_can_verify_without_any_probability_replay(runner,tmp_path,monkeypatch):
    r=runner;plan,path,raw=runtime_fixture(r,tmp_path,monkeypatch)
    r.resource_check=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('headroom'))
    result=r.run(path,raw,'attempt-001',HEAD)
    committed={name:(r.ROOT/name).read_bytes() for name in [*r.SOURCE_PATHS,r.PLAN_REL,r.REVIEW_REL]}
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:committed[cmd[-1].split(':',1)[1]])
    r.evaluation.evaluate_cell=lambda *a,**k:pytest.fail('Missing outcomes were computed')
    r.evaluation.evaluate_calculation=r.evaluation.evaluate_cell
    report=r.verify(result['directory'])
    assert report['replayed_cells']==report['replayed_returned_partial_calculations']==0
    assert report['complete'] is False and report['planned_calculations']==4704
