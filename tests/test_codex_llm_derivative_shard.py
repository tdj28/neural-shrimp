"""Synthetic final-domain shard admission and live-input integration only."""
import copy
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from flint import arb,arb_mat,ctx
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import derivative_shard as driver
from ball_jet import Jet

def pack(m=0,r=0):return dict(mid=driver.domain._pair(Fraction(m)),rad=driver.domain._pair(Fraction(r)))


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    for key,value in [('PERIOD',2),('DIMENSION',4),('COLUMNS',2),('VOCABULARY',7),('LAYERS',2)]:
        monkeypatch.setattr(driver,key,value)
    old=ctx.threads;ctx.threads=1
    with ctx.workprec(1024):yield
    ctx.threads=old


def fixture(tmp_path,monkeypatch):
    import torch
    owned=tmp_path/'owned';owned.mkdir();prefix=owned/'prefix';prefix.mkdir()
    monkeypatch.setattr(driver,'OWNED',owned)
    assets={'synthetic':'a'*64};prefix_ids=(1,2,3)
    mapping=driver.map_definition(assets,prefix_ids)
    centers=[[pack(i+j+Fraction(1,2**1700)) for j in range(4)] for i in range(2)]
    definition=dict(schema=driver.DEFINITION_SCHEMA,role=driver.coverage.ROLE,period=2,dimension=4,
        centers=centers,radii=[pack(Fraction(1,2**1800))]*2,map=mapping,
        map_sha256=driver.domain.canonical_hash(mapping),selection_provenance={'synthetic_only':True})
    path=owned/'final-definition.json';driver.write_json(path,definition)
    bundle=driver.source_bundle()
    plan=dict(schema=driver.SCHEMA,definition_path=str(path),definition_sha256=driver.sha(path),
        map_sha256=driver.domain.canonical_hash(mapping),backend_sha256=driver.domain.canonical_hash(driver.backend_definition(torch.__version__)),
        source_sha256=bundle['source_sha256'],evaluator_sha256=driver.domain.canonical_hash(bundle),
        prefix_attempt=str(prefix),prefix_receipt_sha256='b'*64,prefix_map_source_sha256='c'*64,prefix_cache_sha256='d'*64,
        phase_index=0,start_column=0,precision=1024,token_chunk=1024,envelope=dict(driver.ENVELOPE))
    return owned,definition,plan,assets,prefix_ids


def admitted(plan,assets,prefix_ids):
    import torch
    return driver.admit_definition(plan,assets,prefix_ids,torch.__version__)


def test_definition_uses_independent_source_map_backend_and_exact_global_data(tmp_path,monkeypatch):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    got,binding,raw=admitted(plan,assets,ids)
    assert got==definition and raw==Path(plan['definition_path']).read_bytes()
    assert binding['map_sha256']==driver.domain.canonical_hash(driver.map_definition(assets,ids))
    assert binding['evaluator_sha256']==driver.domain.canonical_hash(driver.source_bundle())
    assert binding['phase_domain_sha256']!=binding['shooting_domain_sha256']
    assert got['centers'][0][0]['mid']==[1,-1700]


@pytest.mark.parametrize('fault',['definition_hash','map_hash','backend_hash','source_hash','bundle_hash',
    'asset','prefix','alpha','steering','prototype','schema','dimension','period','missing_coordinate',
    'float_center','nonexact_center','zero_radius','provenance','unknown_field'])
def test_admission_refuses_bad_trust_anchors_and_definition(tmp_path,monkeypatch,fault):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    if fault=='definition_hash':plan['definition_sha256']='0'*64
    if fault=='map_hash':plan['map_sha256']='0'*64
    if fault=='backend_hash':plan['backend_sha256']='0'*64
    if fault=='source_hash':plan['source_sha256'][driver.SOURCE_PATHS[0]]='0'*64
    if fault=='bundle_hash':plan['evaluator_sha256']='0'*64
    if fault=='asset':definition['map']['pinned_assets']['synthetic']='0'*64
    if fault=='prefix':definition['map']['prefix_ids'].append(9)
    if fault=='alpha':definition['map']['alpha_ratio']=[1,4]
    if fault=='steering':definition['map']['steering_layer']=11
    if fault=='prototype':definition['role']='prototype_box'
    if fault=='schema':definition['schema']='codex-mixed-precision-phase-domain-v1'
    if fault=='dimension':definition['dimension']=3
    if fault=='period':definition['period']=3
    if fault=='missing_coordinate':definition['centers'][0].pop()
    if fault=='float_center':definition['centers'][0][0]['mid']=[1.0,0]
    if fault=='nonexact_center':definition['centers'][0][0]['rad']=[1,-2000]
    if fault=='zero_radius':definition['radii'][0]=pack(0)
    if fault=='provenance':definition['selection_provenance']={}
    if fault=='unknown_field':definition['numerically_verified']=True
    if fault not in ('definition_hash','map_hash','backend_hash','source_hash','bundle_hash'):
        path=Path(plan['definition_path']);driver.write_json(path,definition);plan['definition_sha256']=driver.sha(path)
    with pytest.raises(ValueError):admitted(plan,{'synthetic':'a'*64},ids)


@pytest.mark.parametrize('fault',['phase_bool','phase_range','start_bool','start_alignment','start_range','precision_bool',
    'precision','token_chunk','envelope_bool','seconds','missing_source','extra_source','unknown','outside_definition'])
def test_fixed_plan_scope_refusals(tmp_path,monkeypatch,fault):
    _,_,plan,_,_=fixture(tmp_path,monkeypatch)
    if fault=='phase_bool':plan['phase_index']=False
    if fault=='phase_range':plan['phase_index']=2
    if fault=='start_bool':plan['start_column']=False
    if fault=='start_alignment':plan['start_column']=1
    if fault=='start_range':plan['start_column']=4
    if fault=='precision_bool':plan['precision']=True
    if fault=='precision':plan['precision']=2048
    if fault=='token_chunk':plan['token_chunk']=64
    if fault=='envelope_bool':plan['envelope']['cpu_threads']=True
    if fault=='seconds':plan['envelope']['timeout_seconds']=3601
    if fault=='missing_source':plan['source_sha256'].pop(driver.SOURCE_PATHS[0])
    if fault=='extra_source':plan['source_sha256']['other']='a'*64
    if fault=='unknown':plan['ignore_failed_columns']=True
    if fault=='outside_definition':
        path=tmp_path/'outside.json';path.write_text('{}');plan['definition_path']=str(path)
    with pytest.raises(ValueError):driver.validate_plan(plan)


class FakeBall:
    def __init__(self,*args,fail=False,bad_direction=False,skip_layer=False,head_fault=None,**kwargs):
        self.fail=fail;self.bad_direction=bad_direction;self.skip_layer=skip_layer;self.head_fault=head_fault
    def forward(self,state,directions):
        live=[Jet(x,tuple(directions[i,k] for k in range(driver.COLUMNS))) for i,x in enumerate(state)]
        if self.bad_direction:live[0]=Jet(live[0].value,(arb(-1),)*driver.COLUMNS)
        for layer in range(driver.LAYERS):
            if self.skip_layer and layer==driver.LAYERS-1:continue
            self.progress('step_start',layer,[live])
            if self.fail:raise ValueError('synthetic evaluator failure')
            self.progress('step_end',layer,[live])
        from streamed_head import argmax_diagnostic
        self.last_head_diagnostics=argmax_diagnostic([arb(i) for i in range(driver.VOCABULARY)])
        if self.head_fault=='bad_margin':self.last_head_diagnostics['strict_margin_established']=False
        if self.head_fault=='inconsistent_extrema':self.last_head_diagnostics['candidate_lower']=pack(100)
        if self.head_fault=='nonpoint_endpoint':self.last_head_diagnostics['candidate_lower']=pack(6,1)
        if self.head_fault=='bad_vocabulary':self.last_head_diagnostics['vocabulary_count']-=1
        if self.head_fault=='missing_diagnostic':self.last_head_diagnostics=None
        for kind in ('head_logits','head_derivatives'):
            for first in range(0,driver.VOCABULARY,1024):
                if self.head_fault=='missing_head':continue
                last=min(first+1024,driver.VOCABULARY)
                if self.head_fault=='partial_vocabulary':last-=1
                self.head_progress(kind,first,last)
        return [2*x for x in state],directions*2


def make_shard(tmp_path,definition,plan,binding,*,ball=None):
    out=tmp_path;out.mkdir()
    _,state,directions,records=driver.construct_inputs(definition,plan['phase_index'],plan['start_column'])
    driver.write_json(out/'direction-inputs.json',dict(directions=records))
    shard=driver.evaluate_shard(ball or FakeBall(),definition,binding,plan,state,directions,records,out,lambda *a,**k:None)
    return shard,state


def test_all_shards_feed_collector_and_last_live_inputs_contain_exact_centers(tmp_path,monkeypatch):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    definition,binding,_=admitted(plan,assets,ids)
    shards=[]
    for phase in range(2):
        for start in (0,2):
            local=dict(plan,phase_index=phase,start_column=start)
            shard,state=make_shard(tmp_path/f'shard-{phase}-{start}',definition,local,binding)
            assert shard['actual_inputs']==[driver.domain.pack_ball(v) for v in state]
            assert shard['actual_inputs_sha256']==driver.domain.canonical_hash(shard['actual_inputs'])
            assert shard['direction_indices']==list(range(start,start+2))
            shards.append(shard)
    matrices,report=driver.coverage.collect_derivatives(definition['centers'],definition['radii'],shards,period=2,dimension=4,
        **{'expected_'+name:binding[name] for name in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256')},expected_precision=1024)
    assert report['coverage_complete'] and report['certificate_passed'] is False
    for matrix in matrices:
        for i in range(4):
            for j in range(4):assert matrix[i,j]==2*int(i==j)


@pytest.mark.parametrize('fault',['failure','bad_direction','missing_layer','narrow_live'])
def test_live_inputs_persist_before_failure_or_are_refused(tmp_path,monkeypatch,fault):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    definition,binding,_=admitted(plan,assets,ids)
    out=tmp_path/'new';out.mkdir();_,state,directions,records=driver.construct_inputs(definition,0,0)
    driver.write_json(out/'direction-inputs.json',dict(directions=records))
    ball=FakeBall(fail=fault=='failure',bad_direction=fault=='bad_direction',skip_layer=fault=='missing_layer')
    if fault=='narrow_live':state=[v.mid() for v in state]
    with pytest.raises(ValueError):driver.evaluate_shard(ball,definition,binding,plan,state,directions,records,out,lambda *a,**k:None)
    assert not (out/'derivative-shard.json').exists()
    if fault in ('failure','missing_layer'):assert (out/'actual-inputs.json').is_file()


@pytest.mark.parametrize('fault',['primal_shape','derivative_shape','primal_nonfinite','derivative_nonfinite'])
def test_incomplete_or_nonfinite_outputs_refused(fault):
    v=[arb(0)]*4;m=arb_mat(4,2)
    if fault=='primal_shape':v.pop()
    if fault=='derivative_shape':m=arb_mat(3,2)
    if fault=='primal_nonfinite':v[0]=arb('nan')
    if fault=='derivative_nonfinite':m[0,0]=arb('nan')
    with pytest.raises(ValueError):driver.capture_outputs(v,m)


@pytest.mark.parametrize('fault',[None,'prefix_hash','evaluator_failure'])
def test_driver_fake_full_flow_and_failure_status_without_real_model(tmp_path,monkeypatch,fault):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'));import qwen_map
    owned,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    source=tmp_path/'source';source.mkdir();path=source/'plan.json';driver.write_json(path,plan)
    out=owned/'fresh'/'run'
    model=SimpleNamespace(d=4,layers=2,steering_layer=12,config={'vocab_size':7},
        weights={'w':torch.zeros(1,dtype=torch.float64)},vector=torch.zeros(4,dtype=torch.float64))
    calls=[]
    class Ball(FakeBall):
        def __init__(self,*a,**k):super().__init__(fail=fault=='evaluator_failure')
        def forward(self,*a):calls.append(1);return super().forward(*a)
    monkeypatch.setattr(driver,'HERE',source)
    monkeypatch.setattr(driver,'StreamedHeadQwenJet',Ball)
    monkeypatch.setattr(driver,'load_prefix',lambda *a,**k:dict(prefix_sha256='e'*64 if fault=='prefix_hash' else plan['prefix_cache_sha256']))
    monkeypatch.setattr(qwen_map,'PINNED_SHA256',assets);monkeypatch.setattr(qwen_map,'PREFIX_IDS',ids)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda:model)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(os,'getpriority',lambda *a:10);monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['derivative_shard','--output',str(out),'--plan',str(path)])
    if fault:
        with pytest.raises(ValueError):driver.main()
    else:driver.main()
    status=driver.read_json(out/'status.json')
    assert status['certificate_passed'] is False
    assert driver.sha(out/'candidate-definition.json')==plan['definition_sha256']
    if fault=='prefix_hash':assert not calls
    else:assert calls==[1]
    if fault:assert status['status']=='final_derivative_shard_failed'
    else:
        assert status['status']=='final_derivative_shard_completed' and status['columns_completed']==2
        assert status['shard_sha256']==driver.sha(out/'derivative-shard.json')


def test_isolated_help():
    result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'codex/certify-llm/derivative_shard.py'),'--help'],
                          capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('fault',['missing_head','partial_vocabulary','bad_margin','bad_vocabulary','missing_diagnostic','inconsistent_extrema','nonpoint_endpoint'])
def test_incomplete_or_inconsistent_head_diagnostics_refuse_completion(tmp_path,monkeypatch,fault):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    definition,binding,_=admitted(plan,assets,ids)
    with pytest.raises(ValueError):
        make_shard(tmp_path/'head-fault',definition,plan,binding,ball=FakeBall(head_fault=fault))
    assert (tmp_path/'head-fault'/'actual-inputs.json').exists()
    if fault!='partial_vocabulary':
        assert (tmp_path/'head-fault'/'derivative-shard.json').exists()
    assert not (tmp_path/'head-fault'/'head-diagnostics.json').exists()


def test_head_witness_bound_to_full_actual_phase_without_orbit_claim(tmp_path,monkeypatch):
    _,definition,plan,assets,ids=fixture(tmp_path,monkeypatch)
    definition,binding,_=admitted(plan,assets,ids)
    shard,_=make_shard(tmp_path/'head',definition,plan,binding)
    path=tmp_path/'head'/'head-diagnostics.json';note=driver.read_json(path)
    assert shard['head_diagnostics_file_sha256']==driver.sha(path)
    assert note['design_sha256']==binding['design_sha256']
    assert note['map_sha256']==binding['map_sha256']
    assert note['actual_inputs_sha256']==shard['actual_inputs_sha256']
    assert note['head_passes']==[['head_logits',0,7],['head_derivatives',0,7]]
    assert note['diagnostic']['candidate_index']==6
    assert note['diagnostic']['strict_margin_established']
    assert note['certificate_passed'] is False
