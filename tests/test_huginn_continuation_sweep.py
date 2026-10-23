import sys,json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import huginn_continuation_sweep as s
import huginn_incremental_retrieval as h
from test_huginn_incremental_retrieval import setup_collector,sha


def test_refinement_prefers_first_label_change_and_does_not_call_zero_chaos():
    rows=[{'period':[2,2],'growth':[[0,0],[0,0]]},{'period':[2,0],'growth':[[0,0],[0,0]]},{'period':[4,4],'growth':[[2,2],[2,2]]}]
    i,why=s.refinement_interval(rows)
    assert i==0 and 'unresolved' in why
    with pytest.raises(ValueError):s.refinement_interval([])


def test_no_period_change_uses_growth_and_first_tie():
    rows=[{'period':[0,0],'growth':[[x,x],[x,x]]} for x in [0,1,2]]
    assert s.refinement_interval(rows)[0]==0
    x=s.interior_grid(1.27,1.275,17)
    assert len(x)==17 and x[0]>1.27 and x[-1]<1.275 and (np.diff(x)>0).all()


def test_two_offset_growth_known_linear_map_and_base_independence():
    import torch
    torch.set_num_threads(1)
    base=torch.tensor([[[2.,3.]],[[4.,5.]]],dtype=torch.float64)
    radius=torch.tensor([[1e-3,2e-3],[3e-3,4e-3]],dtype=torch.float64)
    direction=torch.tensor([1.,0.],dtype=torch.float64)
    twins=base[None]+radius[:,:,None,None]*direction
    got,g,logs=s.growth_step(base,twins,radius,lambda x:1.2*x)
    assert torch.equal(got,1.2*base)
    np.testing.assert_allclose(logs,np.log(1.2),atol=1e-12)
    np.testing.assert_allclose((g-got).flatten(2).norm(dim=2),radius,atol=1e-12)
    with pytest.raises(ValueError,match='separation'):s.growth_step(base,twins,radius,lambda x:torch.zeros_like(x))


def test_continuation_streams_before_setting_completion_and_requires_all_shards(tmp_path):
    c,reader=setup_collector(tmp_path)
    c.plan=dict(experiment='HPC2',max_settings=2,batch=1,eps=[.001,.0003],transient=1024,measure=1024,checkpoint_steps=1024)
    root=c.identity['remote_data'];planbytes=json.dumps(c.plan).encode()
    reader.files[root+'/plan.json']=planbytes
    reader.files['/root/neural-shrimp/'+c.identity['plan']]=planbytes
    c.sources[c.identity['plan']]={'sha256':sha(planbytes),'bytes':len(planbytes)}
    reader.files[root+'/runtime.json']=json.dumps(dict(source=c.identity['source'],plan_sha256=sha(planbytes))).encode()
    names=h.continuation_artifacts(c.plan)
    for name in names:reader.files[root+'/'+name]=name.encode()
    reader.snapshot_value=dict(ready=True,chunks=[],artifacts=['states-0000-t-01024.npz'])
    assert c.step()=='partial'
    assert (c.out/'data/states-0000-t-01024.npz').exists()
    with pytest.raises(ValueError):c.names(dict(chunks=[],artifacts=['../secret']))
    with pytest.raises(ValueError):c.names(dict(chunks=[],artifacts=['benchmark.npz','benchmark.npz']))
    for name in ['chunk-0000.npz','chunk-0001.npz','map.npz','summary.json','progress.jsonl']:reader.files[root+'/'+name]=name.encode()
    required=names|{'plan.json','runtime.json','steering-direction.npy','chunk-0000.npz','chunk-0001.npz','map.npz','summary.json','progress.jsonl'}
    manifest={name:sha(reader.files[root+'/'+name]) for name in required}
    reader.files[root+'/manifest.json']=json.dumps(manifest).encode()
    reader.snapshot_value=dict(ready=True,chunks=[0,1],artifacts=sorted(names),manifest=manifest)
    for _ in range(20):
        if c.step()=='complete':break
    else:pytest.fail('collector did not complete')
    assert set(json.loads((c.out/'data/manifest.json').read_text()))==required
    reader.snapshot_value['artifacts'].remove('initial-0001.npz')
    with pytest.raises(ValueError,match='Incomplete'):c.step()
    c.close()


@pytest.mark.parametrize('gated',[False,True])
def test_complete_synthetic_worker_retains_all_states_and_selection(tmp_path,monkeypatch,capsys,gated):
    import argparse,types,torch
    import explore_latent_loop as latent
    torch.set_num_threads(1)
    # Exercise the complete control/output path on CPU; no pretrained model or GPU.
    old_to=torch.Tensor.to;old_tensor=torch.tensor
    def cpu_to(self,*args,**kw):
        if args and args[0]=='cuda':args=('cpu',*args[1:])
        if kw.get('device')=='cuda':kw['device']='cpu'
        return old_to(self,*args,**kw)
    def cpu_tensor(*args,**kw):
        if kw.get('device')=='cuda':kw['device']='cpu'
        return old_tensor(*args,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to);monkeypatch.setattr(torch,'tensor',cpu_tensor)
    for name in ['set_per_process_memory_fraction','synchronize']:monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'synthetic CPU')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    class FakeLoop:
        def __init__(self,*a,**k):
            self.token_ids=[1];self.e=self.freqs=None
            self.model=types.SimpleNamespace(config=types.SimpleNamespace(test_time_noise=0),core_block_forward=lambda h,*a:(self.F(h),None))
            self.tok=types.SimpleNamespace(decode=lambda ids:str(ids[0]))
        def setup(self,p):return torch.ones(1,12,5280)
        def F(self,h):return .2*h+1
        def set_steering(self,*a):pass
        def logits(self,h):return torch.arange(20,dtype=torch.float32)
    monkeypatch.setattr(latent,'HuginnLoop',FakeLoop);monkeypatch.setattr(latent,'DATA',tmp_path)
    anchor=np.ones((1,12,5280),dtype=np.float32);reference=np.concatenate([anchor,.5*anchor+.5*(.2*anchor+1)])
    np.savez(tmp_path/'input.npz',anchor=anchor,reference=reference,direction=np.ones(5280))
    plan=dict(experiment='HPC2',batch=1,input_path='input.npz',input_sha256=s.sha256(tmp_path/'input.npz'),max_compute_seconds=3600,prompt='test',prompt_ids=[1],model_revision=latent.REVISIONS['huginn'],steering=0.,anchor_update_size=.5,lambda_range=[1.27,1.28],checkpoint_steps=256,transient=256,measure=256,max_settings=3,coarse_settings=2,refinement_settings=1,eps=[.001,.0003],recurrence_tolerance=3e-5)
    if gated:
        plan.update(retrieval_barrier=True,schedule='bidirectional',max_settings=4,refinement_settings=2)
        import huginn_evidence_gate as gate
        actual_wait=gate.wait_for_ack
        acknowledged=[]
        def verified_ack(root,payload,deadline):
            assert payload['index']==len(acknowledged)
            for name,digest in payload['files'].items():assert s.sha256(root/name)==digest
            # The worker cannot have started the next setting while waiting.
            assert not (root/f"initial-{payload['index']+1:04d}.npz").exists()
            ack=root/'.retrieval-acks'/f"{payload['index']:04d}.json";ack.parent.mkdir(exist_ok=True)
            ack.write_text(json.dumps(payload));actual_wait(root,payload,deadline)
            acknowledged.append(payload)
        monkeypatch.setattr(gate,'wait_for_ack',verified_ack)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    s.execute(argparse.Namespace(out='test',plan=path,source_commit='a'*40),plan)
    root=tmp_path/'explore/test';summary=json.loads((root/'summary.json').read_text())
    assert len(summary['rows'])==plan['max_settings'] and all(row['period']==[1,1] for row in summary['rows'])
    assert summary['rows'][2]['stage']==('descending' if gated else 'refined')
    manifest=json.loads((root/'manifest.json').read_text())
    assert set(manifest)=={p.name for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}
    for name,digest in manifest.items():assert s.sha256(root/name)==digest
    for i in range(plan['max_settings']):
        with np.load(root/f'states-{i:04d}-m-00256.npz') as z:
            assert z['states'].shape==(256,2,12,5280) and z['log_growth'].shape==(256,2,2)

    # Feed actual worker output through the production watchdog decision path.
    import runpod_scan_group as group
    from test_runpod_scan_group import protected_files
    control=tmp_path/'controller'
    config,pod,attempt,phase,binding,status=protected_files(control,monkeypatch)
    config.update(hard_minutes=900,retrieval_minutes=90,no_progress_minutes=20)
    phase.update(compute_stopped=False,phase='computing',
                 hard_deadline_epoch=pod['created_epoch']+54000,
                 compute_deadline_epoch=pod['created_epoch']+48600)
    binding.update(hard_deadline_epoch=phase['hard_deadline_epoch'],deadline_epoch=phase['hard_deadline_epoch']-60)
    (attempt/'phase.json').write_text(json.dumps(phase))
    (attempt/'incremental/identity.json').write_text(json.dumps(binding))
    tracker=group.ProgressWatch(pod['created_epoch'])
    logfile=attempt/'runner-stdout.txt';logfile.write_text('')
    now=pod['created_epoch']+1900
    for line in capsys.readouterr().out.splitlines():
        with logfile.open('a') as f:f.write(line+'\n')
        value,archive=group.progress(control,pod['name'])
        snapshot=group.retrieval_snapshot(control,pod,config)
        tracker.observe(value,snapshot,now)
        assert group.watch_reason(config,pod,dict(finished=False,updated_epoch=now),True,now,tracker,snapshot,protected=True,archive_ready=archive) is None
        now+=15
    assert tracker.numeric is not None
    assert group.watch_reason(config,pod,dict(finished=False,updated_epoch=now),True,now+1201,tracker,snapshot,protected=True,archive_ready=False)=='numerical progress stalled'

    # Transfer the actual generated artifacts through the production collector.
    c,reader=setup_collector(tmp_path/'transport')
    c.plan=plan
    remote=c.identity['remote_data']
    for file in root.iterdir():
        if file.is_file():reader.files[remote+'/'+file.name]=file.read_bytes()
    reader.files['/root/neural-shrimp/'+c.identity['plan']]=path.read_bytes()
    c.sources[c.identity['plan']]={'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
    c.close()
    c=h.Collector(tmp_path/'bound-transport',c.identity,c.sources,'b'*64,plan,reader)
    artifacts=sorted(h.continuation_artifacts(plan))
    reader.snapshot_value=dict(ready=True,chunks=list(range(plan['max_settings'])),artifacts=artifacts,manifest=manifest)
    if gated:
        reader.snapshot_value['barrier']=acknowledged[-1]
        reader.acknowledge=lambda root,payload: None
    for _ in range(30):
        if c.step()=='complete':break
    else:pytest.fail('actual worker artifacts were not completely retrieved')
    for name,digest in manifest.items():assert h.digest_file(c.out/'data'/name)==digest
    import huginn_retrieval_guard as guard
    assert guard.verified_completion(c.out,c.identity,'b'*64,0)['retrieval_complete']
    c.close()
