"""Synthetic audit parser/math controls; never opens certificate target files."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_s1r72_certificate as subject


def ball(n=0,e=0,r=0,re=0):return {'mid':[n,e],'rad':[r,re],'display':'ignored'}


def test_proximity_uses_entire_box_not_center():
    inp={'approximation':['0']*72};box=[ball() for _ in range(72)]
    box[0]=ball(r=1,re=-30)
    result=subject.proximity(inp,box)
    assert result['passed'] and Fraction(result['normalized_max_distance'])==Fraction(1,2**30)
    box[0]=ball(r=1,re=-20)
    assert not subject.proximity(inp,box)['passed']


def test_proximity_normalization_and_exact_threshold():
    inp={'approximation':['10000000']+['0']*71};box=[ball() for _ in range(72)]
    box[0]=ball(10000001)
    value=subject.proximity(inp,box)
    assert value['passed'] and Fraction(value['normalized_max_distance'])==Fraction(1,10_000_000)
    box[0]=ball(10000002)
    assert not subject.proximity(inp,box)['passed']
    # Neither proximity result asserts that the saved seed belongs to the box.
    assert value['coordinates'][0]['absolute_distance_bound']=='1'


@pytest.mark.parametrize('bad',[{'finite':False},ball(r=-1),ball(n=True),{'mid':['1',0],'rad':[0,0]},ball(e=100001)])
def test_invalid_ball_fails_closed(bad):
    box=[ball() for _ in range(72)];box[0]=bad
    with pytest.raises(ValueError):subject.proximity({'approximation':['0']*72},box)


def profile_values():
    return {'passed':True,'map_seconds':.1,'projected_seconds':12.8,'max_rss_kib':100,
            'map_calls_in_checker':14,'safety_factor':2,'overhead_allowance_seconds':10,'evidence_rung':0},{'elapsed_seconds':.3,'max_rss_kib':120}


@pytest.mark.parametrize('patch',[{'map_seconds':float('nan')},{'map_seconds':-1},
                                  {'projected_seconds':1},{'max_rss_kib':float('inf')},
                                  {'map_calls_in_checker':1},{'evidence_rung':3}])
def test_profile_projection_and_semantics(patch):
    result,end=profile_values();assert subject.check_profile_measurements(result,end)==12.8
    result.update(patch)
    with pytest.raises(ValueError):subject.check_profile_measurements(result,end)


def make_inventory(folder):
    (folder/'a').write_text('bound')
    m={'files':{'a':subject.entry(folder/'a')},'total_bytes':5,'within_byte_cap':True}
    (folder/'manifest.json').write_bytes(subject.canonical(m))
    return subject.sha((folder/'manifest.json').read_bytes())


def test_inventory_exact_hash_coverage_and_totals(tmp_path):
    digest=make_inventory(tmp_path);subject.inventory(tmp_path,digest,{'a'})
    (tmp_path/'extra').write_text('unlisted')
    with pytest.raises(ValueError):subject.inventory(tmp_path,digest,{'a'})
    (tmp_path/'extra').unlink();(tmp_path/'a').write_text('changed')
    with pytest.raises(ValueError):subject.inventory(tmp_path,digest,{'a'})


def test_inventory_rejects_symlink_and_rehashed_total(tmp_path):
    digest=make_inventory(tmp_path)
    external=tmp_path.parent/(tmp_path.name+'-external');external.write_text('bound')
    (tmp_path/'a').unlink();(tmp_path/'a').symlink_to(external)
    with pytest.raises(ValueError):subject.inventory(tmp_path,digest)
    (tmp_path/'a').unlink();(tmp_path/'a').write_text('bound')
    m=subject.read(tmp_path/'manifest.json');m['total_bytes']=-1
    (tmp_path/'manifest.json').write_bytes(subject.canonical(m))
    with pytest.raises(ValueError):subject.inventory(tmp_path,subject.sha((tmp_path/'manifest.json').read_bytes()))


def exact_fixture():
    network=dict(d=8,width=8,bias=False,activation='tanh',in_scale=.5,out_scale=.125,loss_scale=1.,ridge=.001)
    saved={'x':np.zeros((72,8)),'y':np.zeros(72)};saved['x'][0,0]=.1
    node={'pair':np.stack([np.arange(72,dtype=np.float64)/10,-np.arange(72,dtype=np.float64)/10]),'rates':np.array([.1,.2])}
    def q(v):
        f=Fraction(float(v));return f'{f.numerator}/{f.denominator}'
    en=deepcopy(network)
    for k in ('in_scale','out_scale','loss_scale','ridge'):en[k]=q(en[k])
    inp=dict(schema='neural-shrimp-c1-input-v1',period=2,network=en,x=[[q(v) for v in row] for row in saved['x']],
             y=[q(v) for v in saved['y']],approximation=[q(v) for v in node['pair'][0]],lr_hidden=q(.1),lr_output=q(.2))
    return inp,saved,node,network


def test_actual_dyadics_not_decimal_reinterpretation_or_other_phase():
    inp,saved,node,model=exact_fixture();subject.compare_exact_inputs(inp,saved,node,model)
    bad=deepcopy(inp);bad['lr_hidden']='1/10'
    with pytest.raises(ValueError):subject.compare_exact_inputs(bad,saved,node,model)
    bad=deepcopy(inp);bad['approximation']=[str(Fraction(float(v))) for v in node['pair'][1]]
    with pytest.raises(ValueError):subject.compare_exact_inputs(bad,saved,node,model)
    bad=deepcopy(inp);bad['x'][0][0]='1/10'
    with pytest.raises(ValueError):subject.compare_exact_inputs(bad,saved,node,model)


def launcher_fixture(folder,action='profile',reverse_options=False):
    source=subject.PROFILE_SOURCE if action=='profile' else subject.COLLECTION_SOURCE
    argv=['timeout','--signal=TERM','--kill-after=5s','25s' if action=='profile' else '110s','nice','-n','10',
          '/data2/neural-shrimp/.venv/bin/python','tools/certify_s1r_period2.py',action,'--out',str(folder)]
    if action=='collect':
        argv=argv+['--profile',str(subject.PROFILE)] if not reverse_options else argv[:-2]+['--profile',str(subject.PROFILE),'--out',str(folder)]
    start={'argv':argv,'source_commit':source,'source_review_commit':subject.REVIEW,'pid':42,'start_utc':'2026-10-08T00:00:00+00:00',
           'thread_env':{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    log=Path(str(folder)+'.launcher.log');log.write_text('synthetic log')
    end={**start,'exit_code':0,'elapsed_seconds':1.,'end_utc':'2026-10-08T00:00:01+00:00',
         'log':dict(path=str(log),**subject.entry(log))}
    sp=Path(str(folder)+'.launch-start.json');ep=Path(str(folder)+'.launcher.json')
    sp.write_bytes(subject.canonical(start));ep.write_bytes(subject.canonical(end))
    runtime={'elapsed_seconds':.5,'started_utc':'2026-10-08T00:00:00.1+00:00','pid':43}
    return source,runtime,sp,ep


@pytest.mark.parametrize('action,reverse',[('profile',False),('collect',False),('collect',True)])
def test_launcher_exact_scope_binds_log_exit_and_commands(tmp_path,action,reverse):
    folder=tmp_path/'synthetic';source,runtime,sp,ep=launcher_fixture(folder,action,reverse)
    subject.launcher(folder,action,source,runtime)
    end=subject.read(ep);end['exit_code']=124;ep.write_bytes(subject.canonical(end))
    with pytest.raises(ValueError):subject.launcher(folder,action,source,runtime)


def test_launcher_rehashed_extra_parameter_rejected(tmp_path):
    folder=tmp_path/'synthetic';source,runtime,sp,ep=launcher_fixture(folder)
    for path in (sp,ep):
        value=subject.read(path);value['argv']+=['--precision','512'];path.write_bytes(subject.canonical(value))
    with pytest.raises(ValueError):subject.launcher(folder,'profile',source,runtime)


def test_one_replay_and_proximity_failure_keeps_verified_result(monkeypatch,tmp_path):
    calls=[]
    def replay(path):calls.append(path);return {'passed':True,'evidence_rung':3}
    monkeypatch.setattr(subject.verify_certificates,'verify',replay)
    inp={'approximation':['0']*72};box=[ball() for _ in range(72)];box[0]=ball(r=1,re=-10)
    value=subject.replay_and_proximity(inp,{'result':{'box':box}},lambda name,result:subject.save(tmp_path,name,result))
    assert len(calls)==1 and not value['passed']
    assert subject.read(tmp_path/'stage-02-verifier.json')['passed']
    assert not subject.read(tmp_path/'stage-03-exact-proximity.json')['passed']


def test_failed_replay_never_runs_proximity(monkeypatch,tmp_path):
    def fail(path):raise ValueError('synthetic verifier failure')
    monkeypatch.setattr(subject.verify_certificates,'verify',fail)
    monkeypatch.setattr(subject,'proximity',lambda *args:pytest.fail('proximity should not execute'))
    with pytest.raises(ValueError):subject.replay_and_proximity({}, {},lambda name,result:subject.save(tmp_path,name,result))
    assert list(tmp_path.iterdir())==[]


def test_audit_write_is_exclusive_and_cap_enforced(tmp_path,monkeypatch):
    subject.save(tmp_path,'stage.json',{'retained':'stage'})
    with pytest.raises(FileExistsError):subject.save(tmp_path,'stage.json',{})
    monkeypatch.setattr(subject,'CAP',1)
    with pytest.raises(ValueError):subject.save(tmp_path,'next.json',{})
    assert subject.read(tmp_path/'stage.json')=={'retained':'stage'}


def seal_nested(folder):
    receipt=subject.read(folder/'receipt.json')
    receipt['files']={name:subject.entry(folder/name) for name in ('inputs.json','certificate.json','runtime-start.json','runtime-end.json')}
    (folder/'receipt.json').write_bytes(subject.canonical(receipt))
    (folder.parent/'result.json').write_bytes(subject.canonical(receipt))


def nested_fixture(folder):
    folder.mkdir()
    inputs={'synthetic':'inputs'}
    source=subject.COLLECTION_SOURCE
    hashes={name:subject.sha((subject.ROOT/name).read_bytes()) for name in subject.producer.certify_c1.SOURCES}
    start=dict(source_commit=source,source_sha256=hashes,pid=123,cpu_threads=1,nice=10,gpu='not used',paid_compute=False)
    end={**start,'elapsed_seconds':.2,'max_rss_kib':120}
    cert={'schema':'neural-shrimp-c1-certificate-v1','runtime':start,'inputs_sha256':subject.sha(subject.canonical(inputs)),
          'result':dict(settings=subject.SETTINGS,passed=True,evidence_rung=3,exact_statement=dict(primitive_period=2,dimension=72,existence_and_local_uniqueness=True,primitive_period_excluded_divisors=True,attraction=True))}
    receipt=dict(schema='neural-shrimp-c1-receipt-v1',external_root=str(folder.resolve()),source_commit=source,source_sha256=hashes,elapsed_seconds=.2,max_rss_kib=120)
    for name,obj in [('inputs.json',inputs),('runtime-start.json',start),('runtime-end.json',end),('certificate.json',cert),('receipt.json',receipt)]:
        (folder/name).write_bytes(subject.canonical(obj))
    seal_nested(folder)
    return inputs,{'source_commit':source,'pid':123},{'elapsed_seconds':.4,'max_rss_kib':130}


@pytest.mark.parametrize('which',['settings','statement','runtime_pid','memory','source_inventory','input','receipt_time'])
def test_nested_binding_rehashed_semantic_corruption(tmp_path,which):
    folder=tmp_path/'certificate';inp,start,end=nested_fixture(folder)
    subject.nested_binding(folder,start,end,inp)
    if which in ('settings','statement','runtime_pid'):
        path=folder/'certificate.json';value=subject.read(path)
        if which=='settings':value['result']['settings']['newton_steps']=6
        elif which=='statement':value['result']['exact_statement']['attraction']=False
        else:value['runtime']['pid']=999
    elif which=='memory':
        path=folder/'runtime-end.json';value=subject.read(path);value['max_rss_kib']=131
    elif which=='source_inventory':
        path=folder/'runtime-start.json';value=subject.read(path);value['source_sha256']={}
    elif which=='input':
        path=folder/'inputs.json';value={'different':'inputs'}
    else:
        path=folder/'receipt.json';value=subject.read(path);value['elapsed_seconds']=.3
    path.write_bytes(subject.canonical(value));seal_nested(folder)
    with pytest.raises(ValueError):subject.nested_binding(folder,start,end,inp)


def runtime_fixture(folder):
    hashes={name:subject.sha((subject.ROOT/name).read_bytes()) for name in subject.producer.SOURCES}
    start=dict(source_commit=subject.PROFILE_SOURCE,source_sha256=hashes,settings=subject.SETTINGS,action='profile',pid=1,cpu_threads=1,nice=10,gpu=False,paid=False)
    end={**start,'elapsed_seconds':1.,'max_rss_kib':100,'status':'profile_passed'}
    for name,value in [('runtime-start.json',start),('runtime-end.json',end)]:(folder/name).write_bytes(subject.canonical(value))


@pytest.mark.parametrize('patch',[{'status':'failed'},{'pid':2},{'elapsed_seconds':21},{'max_rss_kib':float('nan')}])
def test_runtime_completion_source_and_caps(tmp_path,patch):
    runtime_fixture(tmp_path)
    subject.runtime_pair(tmp_path,'profile',subject.PROFILE_SOURCE,'profile_passed')
    path=tmp_path/'runtime-end.json';value=subject.read(path);value.update(patch)
    import json
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):subject.runtime_pair(tmp_path,'profile',subject.PROFILE_SOURCE,'profile_passed')


def test_audit_self_source_guard_rejects_dirty_dependency(monkeypatch):
    def fake_git(commit,name):
        raw=(subject.ROOT/name).read_bytes()
        return raw+b'changed' if name==subject.SELF_SOURCES[0] else raw
    monkeypatch.setattr(subject,'git_bytes',fake_git)
    with pytest.raises(ValueError,match='uncommitted'):subject.self_binding()


def test_original_record_rejects_unreferenced_arrays_and_wrong_metadata(tmp_path):
    def write(extra=False,bad_shape=False):
        a=np.array([1.,2.]);values={'a':a}
        if extra:values['b']=np.array([3.])
        np.savez(tmp_path/'arrays.npz',**values)
        meta=dict(array='a',dtype=a.dtype.str,shape=[99] if bad_shape else list(a.shape),order='C',sha256=subject.sha(a.tobytes()))
        obj=dict(kind='node',arrays_file='arrays.npz',arrays_sha256=subject.sha((tmp_path/'arrays.npz').read_bytes()),payload={'a':meta})
        (tmp_path/'record.json').write_bytes(subject.canonical(obj))
    write();kind,value=subject.decode_record(tmp_path,'record.json');assert kind=='node' and value['a'].tolist()==[1.,2.]
    write(extra=True)
    with pytest.raises(ValueError,match='unused'):subject.decode_record(tmp_path,'record.json')
    write(bad_shape=True)
    with pytest.raises(ValueError,match='metadata'):subject.decode_record(tmp_path,'record.json')
