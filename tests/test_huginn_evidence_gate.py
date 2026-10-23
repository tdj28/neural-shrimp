import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from huginn_evidence_gate import wait_for_ack
import huginn_incremental_retrieval as retrieval
import runpod_scan_group as group
from test_runpod_scan_group import protected_config


def test_no_advancement_without_exact_ack_and_original_deadline(tmp_path):
    payload={'index':0,'source':'a'*40,'files':{'chunk':'b'*64}}
    ticks=iter([0.,0.,1.,1.,2.])
    with pytest.raises(TimeoutError):
        wait_for_ack(tmp_path,payload,2.,clock=lambda:next(ticks),sleep=lambda _:None)
    p=tmp_path/'.retrieval-acks/0000.json';p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError):wait_for_ack(tmp_path,payload,2.,clock=lambda:0.)
    p.write_text(json.dumps(payload));wait_for_ack(tmp_path,payload,2.,clock=lambda:0.)


def test_missing_endpoint_metadata_keeps_known_connection_but_not_foreign_identity():
    identity={'pod_id':'owned','name':'name'}
    rows=iter([dict(id='owned',name='name',runtime={'ports':[dict(privatePort=22,isIpPublic=True,ip='127.0.0.1',publicPort=2222)]}),dict(id='owned',name='name'),dict(id='other',name='name')])
    reader=retrieval.SSHReader(identity,100,status_lookup=lambda _:next(rows))
    assert reader.refresh();base=reader.base.copy()
    assert reader.refresh() and reader.base==base
    with pytest.raises(ValueError):reader.refresh()


def test_evidence_wait_or_collector_loss_never_triggers_early_deletion():
    c=protected_config();c['evidence_gate']=True
    pod=dict(created_epoch=0,cost_per_hr=.5);state=dict(finished=False,updated_epoch=0)
    tracker=group.ProgressWatch(0)
    for snapshot in [dict(bound=True,active=False,complete=False),dict(bound=True,active=True,compute_stopped=True,complete=False)]:
        assert group.watch_reason(c,pod,state,False,5000,tracker,snapshot,protected=True,archive_ready=False) is None
        assert group.watch_reason(c,pod,state,False,7140,tracker,snapshot,protected=True,archive_ready=False)=='original hard lifetime termination reserve reached'


def test_recovery_requires_stopped_proof_and_verified_files(tmp_path,monkeypatch):
    from unittest.mock import Mock
    import huginn_retrieval_guard as guard
    g=guard.ProtectedRun.__new__(guard.ProtectedRun)
    g.evidence_gate=True;g.numerical_started=True;g.out=tmp_path;g.identity={'remote_data':'/owned','source':'a'*40}
    g.retrieval_deadline=100;g.compute_deadline=90;g.collector_restarts=0
    g.collector=Mock();g.collector.poll.return_value=None;g.write=Mock()
    data=tmp_path/'incremental/data';data.mkdir(parents=True);(data/'chunk.npz').write_bytes(b'evidence')
    files={'chunk.npz':guard.digest_file(data/'chunk.npz')}
    (tmp_path/'incremental/receipts.jsonl').write_text(json.dumps(dict(event='setting_acknowledged',index=0,files=files))+'\n')
    barrier=dict(index=0,source='a'*40,files=files)
    snapshots=iter([dict(barrier=barrier,stopped=None),dict(barrier=barrier,stopped=dict(index=0,source='a'*40))])
    monkeypatch.setattr(retrieval.SSHReader,'snapshot',lambda *a:next(snapshots))
    monkeypatch.setattr(guard.subprocess,'run',Mock())
    t=[0.];sleeps=[]
    def sleep(n):t[0]+=n;sleeps.append(n)
    result=g.salvage_before_termination(['fake-ssh'],clock=lambda:t[0],sleep=sleep)
    assert result['verified_boundary']==0 and sleeps==[10]


def test_unrecoverable_transport_cannot_extend_original_deadline(tmp_path,monkeypatch):
    from unittest.mock import Mock
    import huginn_retrieval_guard as guard
    g=guard.ProtectedRun.__new__(guard.ProtectedRun);g.evidence_gate=True;g.numerical_started=True
    g.identity={'remote_data':'/owned'};g.retrieval_deadline=20.;g.write=Mock()
    monkeypatch.setattr(guard.subprocess,'run',Mock(side_effect=OSError('unreachable')))
    t=[0.]
    result=g.salvage_before_termination(['fake-ssh'],clock=lambda:t[0],sleep=lambda n:t.__setitem__(0,t[0]+n))
    assert result['deadline_reached'] and t[0]==20


def test_collector_ack_requires_every_setting_file_and_matching_hash(tmp_path):
    from test_huginn_incremental_retrieval import setup_collector,sha
    c,reader=setup_collector(tmp_path)
    plan=dict(experiment='HPC2',retrieval_barrier=True,max_settings=1,batch=1,eps=[.001,.0003],transient=1024,measure=1024,checkpoint_steps=1024)
    c.plan=plan;root=c.identity['remote_data'];planbytes=json.dumps(plan).encode()
    reader.files[root+'/plan.json']=planbytes
    reader.files['/root/neural-shrimp/'+c.identity['plan']]=planbytes
    c.sources[c.identity['plan']]={'sha256':sha(planbytes),'bytes':len(planbytes)}
    reader.files[root+'/runtime.json']=json.dumps(dict(source=c.identity['source'],plan_sha256=sha(planbytes))).encode()
    names=retrieval.continuation_artifacts(plan)
    for name in names:reader.files[root+'/'+name]=name.encode()
    required=names|{'plan.json','runtime.json','steering-direction.npy','chunk-0000.npz'}
    payload=dict(index=0,source=c.identity['source'],files={n:sha(reader.files[root+'/'+n]) for n in required})
    reader.snapshot_value=dict(ready=True,chunks=[0],artifacts=sorted(names),barrier=payload)
    acknowledged=[];reader.acknowledge=lambda r,p:acknowledged.append(p)
    c.step();assert not acknowledged
    for _ in range(10):
        c.step()
        if acknowledged:break
    assert acknowledged==[payload]
    c.acknowledged.clear();(c.out/'data/states-0000-m-01024.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):c.step()
    assert len(acknowledged)==1
    c.close()


def test_emergency_cleanup_waits_for_evidence_or_original_deadline(tmp_path,monkeypatch):
    from test_runpod_scan_group import protected_files,creation
    c,pod,*_=protected_files(tmp_path,monkeypatch);c['evidence_gate']=True
    monkeypatch.setattr(group,'ledger',lambda:[creation(cost_per_hr=.5)])
    cutoff=pod['created_epoch']+60*c['hard_minutes']-60
    t=[cutoff-30];deletions=[]
    monkeypatch.setattr(group.time,'time',lambda:t[0])
    monkeypatch.setattr(group.time,'sleep',lambda n:t.__setitem__(0,t[0]+n))
    monkeypatch.setattr(group,'terminate_owned',lambda *a:deletions.append(t[0]))
    group.cleanup(tmp_path,pod,'watchdog lost',c)
    assert deletions==[cutoff]


def test_new_sweep_without_controller_gate_rejected_before_rental():
    c=protected_config();cmd=c['commands'][0]
    i=cmd.index('--jobset-args')+1
    cmd[i]=cmd[i].replace('docs/huginn-chaos-map-plan.json','docs/huginn-chaos-transition-plan.json')
    with pytest.raises(ValueError,match='before rental'):group.validate(c)


def test_gated_numerical_failure_enters_recovery_without_idle_spend(tmp_path):
    from unittest.mock import Mock
    from test_huginn_retrieval_guard import fake_guard
    g=fake_guard(tmp_path);g.evidence_gate=True;g.collector.pid=123
    g.collector.poll.return_value=None
    remote=Mock();remote.poll.return_value=1
    with pytest.raises(RuntimeError,match='recover verified boundary'):
        g.supervise(remote,clock=lambda:50.,sleep=lambda _:pytest.fail('must enter recovery immediately'))
