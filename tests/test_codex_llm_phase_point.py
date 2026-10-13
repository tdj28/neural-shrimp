import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import phase_point as worker
from ball_qwen import Dual


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    for key,value in [('PERIOD',2),('DIMENSION',4),('LAYERS',2)]:monkeypatch.setattr(worker,key,value)
    old=ctx.prec,ctx.threads;ctx.prec=2048;ctx.threads=1
    yield
    ctx.prec,ctx.threads=old


def request_fixture(tmp_path,monkeypatch):
    import torch
    owned=tmp_path/'owned';owned.mkdir();prefix=owned/'prefix';prefix.mkdir()
    monkeypatch.setattr(worker,'OWNED',owned)
    assets={'synthetic':'a'*64};ids=(1,2)
    centers=[[arb(i*4+j)+arb((1,-1700)) for j in range(4)] for i in range(2)]
    records=worker.saved.center_records(centers)
    mapping=worker.map_spec(assets,ids);backend=worker.backend(torch.__version__)
    request=dict(schema=worker.SCHEMA,source_commit='a'*40,
        source_sha256={path:worker.base.digest(ROOT/path) for path in worker.SOURCES},
        map=mapping,map_sha256=worker.canonical(mapping),backend=backend,backend_sha256=worker.canonical(backend),
        centers=records,centers_sha256=worker.canonical(records),prefix_attempt=str(prefix),
        prefix_receipt_sha256='b'*64,prefix_map_source_sha256='c'*64,prefix_cache_sha256='d'*64,
        precision=2048,selection_provenance={'synthetic':True})
    return owned,request,centers,assets,ids


def admit(request,assets,ids):
    import torch
    return worker.admit(request,assets,ids,torch.__version__,'a'*40)


def test_admission_keeps_every_exact_phase_coordinate(tmp_path,monkeypatch):
    owned,request,centers,assets,ids=request_fixture(tmp_path,monkeypatch)
    decoded,prefix=admit(request,assets,ids)
    assert decoded==centers and prefix==owned/'prefix'
    assert worker.saved.center_records(decoded)==request['centers']


@pytest.mark.parametrize('fault',['source','bundle','map','asset','backend','precision','centers_hash',
    'missing_coordinate','radius','noncanonical','provenance','prefix_hash','prefix_outside','extra','map_alias','backend_alias'])
def test_bad_request_refused(tmp_path,monkeypatch,fault):
    _,request,centers,assets,ids=request_fixture(tmp_path,monkeypatch)
    if fault=='source':request['source_commit']='b'*40
    if fault=='bundle':request['source_sha256'][worker.SOURCES[0]]='0'*64
    if fault=='map':request['map_sha256']='0'*64
    if fault=='map_alias':request['map']['dimension']=4.0
    if fault=='backend_alias':request['backend']['threads']=True
    if fault=='asset':assets={'other':'0'*64}
    if fault=='backend':request['backend']['precision']=1024
    if fault=='precision':request['precision']=True
    if fault=='centers_hash':request['centers_sha256']='0'*64
    if fault=='missing_coordinate':request['centers'][0].pop()
    if fault=='radius':request['centers'][0][0]['rad']=[1,-1900]
    if fault=='noncanonical':
        m,e=request['centers'][0][0]['mid'];request['centers'][0][0]['mid']=[m*2,e-1]
    if fault in ('missing_coordinate','radius','noncanonical'):request['centers_sha256']=worker.canonical(request['centers'])
    if fault=='provenance':request['selection_provenance']={}
    if fault=='prefix_hash':request['prefix_cache_sha256']=''
    if fault=='prefix_outside':request['prefix_attempt']=str(tmp_path)
    if fault=='extra':request['ignore_failed_phase']=True
    with pytest.raises(ValueError):admit(request,assets,ids)


class FakeBall:
    def __init__(self,*args,fault=None,**kwargs):self.fault=fault
    def forward(self,center):
        live=[Dual(v,arb(0)) for v in center]
        if self.fault=='shifted_live':live[0]=Dual(center[0]+arb((1,-1800)),arb(0))
        if self.fault=='live_tangent':live[0]=Dual(center[0],arb(1))
        for i in range(worker.LAYERS):
            if self.fault=='missing_layer' and i==worker.LAYERS-1:continue
            self.progress('step_start',i,[live])
            if self.fault=='evaluation_failure':raise ValueError('synthetic evaluation failure')
            self.progress('step_end',i,[live])
        values=[v*2 for v in center];tangents=[arb(0)]*len(center)
        if self.fault=='short_output':values.pop()
        if self.fault=='nonfinite':values[0]=arb('nan')
        if self.fault=='output_tangent':tangents[0]=arb(1)
        return values,tangents


def test_point_input_live_input_successor_and_output_bound(tmp_path,monkeypatch):
    _,_,centers,_,_=request_fixture(tmp_path,monkeypatch)
    out=tmp_path/'phase';out.mkdir()
    result=worker.evaluate_phase(FakeBall(),centers,1,out,lambda *a,**k:None)
    inputs=worker.saved.read_json(out/'input.json');live=worker.saved.read_json(out/'live-input.json')
    assert inputs['phase_index']==1 and inputs['center']==live['center']
    assert inputs['next_center']==worker.saved.center_records(centers)[0]
    assert result['point_sha256']==worker.base.digest(out/'point.json')
    assert result['input_sha256']==worker.base.digest(out/'input.json')
    values=worker.saved.read_json(out/'point.json')['value']
    assert worker.saved.decoded_vector(values,4)==[v*2 for v in centers[1]]


@pytest.mark.parametrize('fault',['shifted_live','live_tangent','missing_layer','evaluation_failure','short_output','nonfinite','output_tangent'])
def test_incomplete_or_changed_phase_never_succeeds(tmp_path,monkeypatch,fault):
    _,_,centers,_,_=request_fixture(tmp_path,monkeypatch);out=tmp_path/'phase';out.mkdir()
    with pytest.raises(ValueError):worker.evaluate_phase(FakeBall(fault=fault),centers,0,out,lambda *a,**k:None)
    assert (out/'input.json').is_file() and not (out/'point.json').exists()
    if fault not in ('shifted_live','live_tangent'):assert (out/'live-input.json').is_file()


@pytest.mark.parametrize('index',[True,-1,2])
def test_invalid_phase_index_refused(tmp_path,monkeypatch,index):
    _,_,centers,_,_=request_fixture(tmp_path,monkeypatch);out=tmp_path/'phase';out.mkdir()
    with pytest.raises(ValueError):worker.evaluate_phase(FakeBall(),centers,index,out,lambda *a,**k:None)
    assert not list(out.iterdir())


@pytest.mark.parametrize('fault',[None,'prefix_hash','evaluation_failure','completed_attempt','request_bytes'])
def test_complete_fake_worker_and_failure_status(tmp_path,monkeypatch,fault):
    import torch
    sys.path.insert(0,str(ROOT/'codex/crosscheck'));import qwen_map
    owned,request,_,assets,ids=request_fixture(tmp_path,monkeypatch)
    candidate=owned/'fresh'/'run'/'candidate-01';candidate.mkdir(parents=True)
    path=candidate/'request.json';worker.base.write_json(path,request);out=candidate/'phase-00'
    if fault=='completed_attempt':(owned/'fresh'/'receipt.json').write_text('{}')
    model=SimpleNamespace(d=4,layers=2,steering_layer=12,config={'vocab_size':151936},
        weights={'w':torch.zeros(1,dtype=torch.float64)},vector=torch.zeros(4,dtype=torch.float64))
    calls=[];limits=[]
    if fault=='request_bytes':
        original_read=Path.read_bytes
        monkeypatch.setattr(Path,'read_bytes',lambda p:original_read(p)+b' ' if p==path else original_read(p))
    class Ball(FakeBall):
        def __init__(self,*a,**k):super().__init__(fault='evaluation_failure' if fault=='evaluation_failure' else None)
        def forward(self,*a):calls.append(1);return super().forward(*a)
    monkeypatch.setattr(worker.base,'BallQwen',Ball)
    monkeypatch.setattr(worker.base,'load_prefix',lambda *a,**k:dict(prefix_sha256='e'*64 if fault=='prefix_hash' else request['prefix_cache_sha256']))
    monkeypatch.setattr(qwen_map,'PINNED_SHA256',assets);monkeypatch.setattr(qwen_map,'PREFIX_IDS',ids)
    monkeypatch.setattr(qwen_map.ManualQwenSoftMap,'from_pinned_files',lambda:model)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(worker.resource,'getrlimit',lambda *a:(7*1024**3,18*1024**3))
    monkeypatch.setattr(worker.resource,'setrlimit',lambda kind,pair:limits.append(pair))
    monkeypatch.setattr(os,'getpriority',lambda *a:10);monkeypatch.setenv('CERT_SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(sys,'argv',['phase_point','--request',str(path),'--request-sha256',worker.base.digest(path),'--phase','0','--output',str(out)])
    if fault:
        with pytest.raises(ValueError):worker.main()
    else:worker.main()
    if fault in ('completed_attempt','request_bytes'):
        assert not out.exists() and not calls and not limits;return
    assert limits==[(7*1024**3,7*1024**3)]
    status=worker.saved.read_json(out/'status.json')
    assert status['certificate_passed'] is False
    if fault=='prefix_hash':assert not calls
    else:assert calls==[1]
    assert status['status']==('phase_point_failed' if fault else 'phase_point_completed')
