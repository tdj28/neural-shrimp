"""Prospective controls: synthetic gradients/artifacts; future target output unopened."""
import copy,importlib.util,json,signal,subprocess
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import utility_dose_audit as a
from neural_shrimp.period2_branch_io import Store,canonical,file_entry,identity,load_record
from neural_shrimp.utility_dose import dose_prediction
ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_scalar_gradient_all_coordinates_against_toy_loss_finite_difference():
    theta=np.linspace(-.25,.35,13);x=np.array([[.2,.5],[-.3,.7],[.8,-.1]]);y=np.array([.1,-.2,.3])
    def loss(t):
        h=np.tanh(x@t[:6].reshape(2,3)+t[6:9]);e=h@t[9:12]+t[12]-y
        return .5*np.mean(e*e)+.0005*np.dot(t,t)
    fd=[]
    for i in range(13):
        d=np.zeros(13);d[i]=1e-6;fd.append((loss(theta+d)-loss(theta-d))/2e-6)
    g=a.reference_gradient(theta,x,y);np.testing.assert_allclose(g,fd,rtol=2e-7,atol=2e-10)
    z=np.zeros_like(x);g=a.reference_gradient(theta,z,y)
    np.testing.assert_array_equal(g[:6],.001*theta[:6])
    from neural_shrimp.gdmaps import TwoLayerNet
    np.testing.assert_allclose(a.reference_gradient(theta,x,y),TwoLayerNet(2,3,ridge=.001).gradient(theta[None,:],x,y)[0],rtol=1e-13,atol=1e-14)

def test_training_data_hashes_without_validation_or_test():
    x,y=a.training_only()
    assert a.sha(repr(x.shape).encode()+x.tobytes())=='306db8f39a2d813dcabfdaa8cf975954c40f766fd7271c37a8a6dec1730f8f0e'
    assert a.sha(repr(y.shape).encode()+y.tobytes())=='6ddb5c9fb9e4facd87ca74bcf8fb4917a9708064b1d4fe056503f74016d36f38'

@pytest.mark.parametrize('g',[np.array([3.,4.]),np.array([1.,np.sqrt(24.)]),np.array([1.,np.nextafter(np.sqrt(24.),np.inf)]),np.array([0.,1.]),np.array([np.nan,1.]),np.array([1e308,1e308]),np.array([1e-320,1.])])
def test_independent_dose_oracle_and_failure_branches(g):
    rates=np.array([2.,2.]) if np.all(g==1e308) else np.ones(2)
    expected=dose_prediction(g,rates,1);observed=a.dose_oracle(g,rates,1);a.compare(observed,expected)
    if observed['arm_predictions']:
        s=g*rates
        for arm in observed['arm_predictions'].values():
            np.testing.assert_allclose(arm['predicted_update_increment']+s,arm['treatment_minus_sham_increment'],atol=1e-14,rtol=1e-12)

def test_every_entry_and_exact_decisions_checked():
    value=a.dose_oracle(np.linspace(-1.,2.,13),np.array([2.,.5]));bad=copy.deepcopy(value)
    bad['arm_predictions']['hidden']['predicted_update_increment'][8]+=.001
    with pytest.raises(ValueError):a.compare(bad,value)
    bad=copy.deepcopy(value);bad['status']='different'
    with pytest.raises(ValueError):a.compare(bad,value)
    assert not a.bit_equal(np.array([0.]),np.array([-0.]))

def write_inventory(folder):
    files={p.name:file_entry(p) for p in folder.iterdir() if p.name!='manifest.json'}
    (folder/'manifest.json').write_bytes(canonical({'files':files,'status':'synthetic'}))
    return file_entry(folder/'manifest.json')['sha256']

@pytest.mark.parametrize('damage',['rewrite_rehash','extra','symlink','unsafe'])
def test_external_manifest_pin_and_inventory_fail_closed(tmp_path,damage):
    root=tmp_path/'evidence';root.mkdir();(root/'a.json').write_text('{}');pin=write_inventory(root)
    assert a.inventory(root,pin,4096)
    if damage=='rewrite_rehash':(root/'a.json').write_text('{"value":1}');write_inventory(root)
    elif damage=='extra':(root/'extra').write_text('new')
    elif damage=='symlink':(root/'a.json').unlink();(root/'a.json').symlink_to(root/'manifest.json')
    else:(root/'manifest.json').write_bytes(canonical({'files':{'../a.json':{}},'status':'synthetic'}));pin=file_entry(root/'manifest.json')['sha256']
    with pytest.raises(ValueError):a.inventory(root,pin,4096)

def envelope():
    raw={'manifest.json':{'bytes':10,'sha256':'a'*64},'result.json':{'bytes':20,'sha256':'b'*64}}
    execution='c'*40;review=a.PRODUCER_REVIEW;threads={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    launch=dict(schema='utility-dose-launcher-v1',source_commit=execution,source_review_commit=review,sources={},output=str(a.TARGET),launcher=str(a.TARGET_LAUNCHER),command=['timeout','--signal=TERM','--kill-after=5s','10s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python','tools/explore_utility_dose_feasibility.py','--out',str(a.TARGET),'--review-commit',review],external_seconds=10,kill_grace_seconds=5,total_raw_bytes=a.TOTAL_BYTES,store_bytes=a.STORE_BYTES,launcher_bytes=a.LAUNCHER_BYTES,nice=10,thread_env=threads,launcher_pid=101,child_pid=102)
    runtime=dict(source_commit=execution,source_review_commit=review,input_review_commit=a.INPUT_REVIEW,input_manifest_sha256=a.INPUT_MANIFEST,sources={},nice=10,thread_env=threads,pid=103,limits=dict(internal_seconds=5,external_seconds=10,kill_grace_seconds=5,gradients=4,total_bytes=a.TOTAL_BYTES,store_bytes=a.STORE_BYTES,launcher_bytes=a.LAUNCHER_BYTES))
    producer=dict(schema='utility-dose-feasibility-v1',status='fixed_candidate_infeasible',gradient_attempts=4,gradient_returns=4,forks=[{}]*4,elapsed_seconds=.2)
    receipt=dict(source_commit=execution,source_review_commit=review,child_pid=102,status='process_finished',process_exit=0,failure=None,log_truncated=False,within_store_cap=True,elapsed_seconds=.4,raw_file_count=2,raw_bytes=30,raw_inventory_sha256=a.sha(canonical(raw)),raw_manifest=raw['manifest.json'],raw_result=raw['result.json'])
    return raw,launch,runtime,producer,receipt,execution

@pytest.mark.parametrize('damage',['none','count','runtime','source','review','command','raw_bytes','manifest','scope','pid'])
def test_complete_source_process_envelope(damage):
    raw,launch,runtime,producer,receipt,execution=envelope()
    if damage=='count':producer['gradient_returns']=3
    if damage=='runtime':producer['elapsed_seconds']=6.
    if damage=='source':runtime['source_commit']='e'*40
    if damage=='review':receipt['source_review_commit']='f'*40
    if damage=='command':launch['command'][3]='20s'
    if damage=='raw_bytes':receipt['raw_bytes']=31
    if damage=='manifest':receipt['raw_manifest']={}
    if damage=='scope':runtime['limits']['gradients']=5
    if damage=='pid':receipt['child_pid']=104
    if damage=='none':a.validate_envelope(raw,launch,runtime,producer,receipt,execution)
    else:
        with pytest.raises(ValueError):a.validate_envelope(raw,launch,runtime,producer,receipt,execution)

def test_records_require_complete_exact_consumed_inventory(tmp_path):
    s=Store(tmp_path/'records');s.json('runtime-start.json',{});s.emit('bound_inputs',{'a':np.ones(3)});s.finish({'status':'synthetic'})
    manifest=json.loads((s.root/'manifest.json').read_text());assert a.records(s.root,manifest)[0][0]=='bound_inputs'
    bad=copy.deepcopy(manifest);bad['files']['unconsumed.npy']={}
    with pytest.raises(ValueError):a.records(s.root,bad)
    meta_path=s.root/'00000-bound_inputs.json';meta=json.loads(meta_path.read_text());meta['arrays_file']='other.npz';meta_path.write_bytes(canonical(meta))
    with pytest.raises(ValueError):a.records(s.root,manifest)

@pytest.mark.parametrize('stop',['deadline','failure','term'])
def test_auditor_preserves_returned_gradient_on_stop(tmp_path,monkeypatch,stop):
    runner=module('audit_utility_dose_feasibility');out=tmp_path/'audit';clock=[0.]
    monkeypatch.setattr(runner,'OUT',out);monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}});monkeypatch.setattr(runner.os,'getpriority',lambda *_:10);monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    def fake(raw,launcher,execution,store,check,stats):
        stats['reference_gradient_attempts']=stats['reference_gradient_returns']=1
        store.emit('independent_gradient_return',{'index':0,'gradient':np.arange(13,dtype=float)},final=True)
        if stop=='deadline':clock[0]=6.
        elif stop=='failure':raise ValueError('synthetic corrupted prediction')
        else:signal.getsignal(signal.SIGTERM)(None,None)
        check();raise AssertionError('must stop')
    monkeypatch.setattr(runner,'audit_saved',fake)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=runner.run(out,'a'*40,'b'*64,'c'*64,head)
    assert result['status']==('audit_failed_or_interrupted' if stop=='failure' else 'audit_resource_stop') and result['fixed_candidate_feasible'] is None
    assert result['reference_gradient_returns']==1 and (out/'manifest.json').exists()
    kind,payload=load_record(out,'00000-independent_gradient_return.json');assert kind=='independent_gradient_return';np.testing.assert_array_equal(payload['gradient'],np.arange(13))
    assert all(not p.stat().st_mode&0o222 for p in out.iterdir())

def test_launcher_fixed_limits_and_external_pins():
    launcher=module('launch_utility_dose_audit');cmd=launcher.command(Path('/synthetic'),'a'*40,'b'*40,'c'*64,'d'*64)
    assert cmd[:7]==['timeout','--signal=TERM','--kill-after=5s','10s','nice','-n','10']
    assert cmd[-6:]==['--execution-commit','b'*40,'--raw-manifest','c'*64,'--launcher-manifest','d'*64]
    assert launcher.STORE_BYTES+launcher.LAUNCHER_BYTES==1048576 and launcher.SUPERVISOR_SECONDS==15

def test_complete_synthetic_saved_outcome_replay(tmp_path,monkeypatch):
    """A complete toy producer bundle; never reads the real utility target paths."""
    from datetime import datetime,timedelta
    from neural_shrimp.gdmaps import TwoLayerNet
    producer=module('explore_utility_dose_feasibility');raw=tmp_path/'producer';launchdir=tmp_path/'launcher';launchdir.mkdir()
    x=np.array([[.2,.5],[-.3,.7],[.8,-.1]]);y=np.array([.1,-.2,.3])
    net=TwoLayerNet(2,3,ridge=.001)
    forks=[dict(point=i//2,origin='near_orbit' if i%2==0 else 'ordinary_initialization',native_state=np.linspace(-.25,.35,13)+i*.01,native_rates=np.array([2.,.5]),state={'sha256':'synthetic'}) for i in range(4)]
    gate=dict(projected_seconds=1+1000*4*63.64927253895439/1232000,formula='1+1000*4*63.64927253895439/1232000',limitation='synthetic fixture')
    source_inventory={name:file_entry(ROOT/name) for name in a.PRODUCER_FILES}
    monkeypatch.setattr(producer,'sources',lambda:source_inventory)
    monkeypatch.setattr(producer,'bound_inputs',lambda:(net,x,y,forks,gate))
    monkeypatch.setattr(producer.os,'getpriority',lambda *_:10)
    p_result=producer.run(raw,a.PRODUCER_REVIEW)
    p_runtime=json.loads((raw/'runtime-start.json').read_text());execution=p_runtime['source_commit']
    entries={p.name:file_entry(p) for p in raw.iterdir()};raw_pin=file_entry(raw/'manifest.json')['sha256']
    monkeypatch.setattr(a,'TARGET',raw);monkeypatch.setattr(a,'TARGET_LAUNCHER',launchdir)
    _,launch,_,_,receipt,_=envelope()
    launch.update(source_commit=execution,sources=source_inventory,started_utc=(datetime.fromisoformat(p_runtime['started_utc'])-timedelta(seconds=.1)).isoformat())
    # command's --out value is position10, while final element is review SHA.
    launch['command'][10]=str(raw)
    receipt.update(source_commit=execution,raw_file_count=len(entries),raw_bytes=sum(e['bytes'] for e in entries.values()),raw_inventory_sha256=a.sha(canonical(entries)),raw_manifest=entries['manifest.json'],raw_result=entries['result.json'],elapsed_seconds=p_result['elapsed_seconds']+.2,finished_utc=(datetime.fromisoformat(p_runtime['started_utc'])+timedelta(seconds=1)).isoformat())
    launch['launcher_pid']=p_runtime['pid']+10;launch['child_pid']=p_runtime['pid']+11;receipt['child_pid']=launch['child_pid']
    (launchdir/'launcher-source.py').write_bytes((ROOT/'tools/launch_utility_dose_feasibility.py').read_bytes());(launchdir/'stdout.log').write_bytes(b'synthetic\n')
    receipt['stdout']=file_entry(launchdir/'stdout.log')
    (launchdir/'runtime-start.json').write_bytes(canonical(launch));(launchdir/'receipt.json').write_bytes(canonical(receipt))
    lmanifest={'status':'process_finished','files':{p.name:file_entry(p) for p in launchdir.iterdir()}}
    (launchdir/'manifest.json').write_bytes(canonical(lmanifest));launch_pin=file_entry(launchdir/'manifest.json')['sha256']
    bound=[dict(point=f['point'],origin=f['origin'],state=f['native_state'],rates=f['native_rates'],state_sha256='synthetic') for f in forks]
    monkeypatch.setattr(a,'bound_original_inputs',lambda:(x,y,bound,{'files':{}}))
    s=Store(tmp_path/'audit');stats={'reference_gradient_attempts':0,'reference_gradient_returns':0}
    outcome=a.audit_saved(raw_pin,launch_pin,execution,s,lambda:None,stats)
    assert outcome['status']=='audit_passed' and outcome['fixed_candidate_feasible']==p_result['candidate_feasible']
    assert stats['reference_gradient_attempts']==stats['reference_gradient_returns']==4
    assert max(v['max_gradient_absolute_error'] for v in outcome['forks'])<1e-14
    s.finish(outcome);assert (s.root/'manifest.json').exists()
