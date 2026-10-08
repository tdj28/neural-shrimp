"""Prospective fixed-attempt audit: one C1 replay, exact dyadic proximity, no search."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import flint
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import certify_s1r_period2 as producer
import verify_certificates
from neural_shrimp.trainability import SYSTEMS, array_sha256

REVIEWED='2c00544ca35a5cf4edb28bacc0a80e285e284073'
REVIEW='426ce4fe6509ba7862a6c315386f09de85538e9b'
PROFILE_SOURCE='750233fc03b0a24dbdba32a2e49435931495fdab'
COLLECTION_SOURCE='dc2732da3b76dcf85c4c86051d4e096f5a3c903e'
BASE=Path('/data2/neural-shrimp-data/certificates/s1r-period2')
PROFILE=BASE/'profile-001';COLLECTION=BASE/'attempt-001'
PROFILE_MANIFEST='850a055d40541e1ebbaa118e773b41f036d8ae3b5f6a4691044ec19312ea073e'
COLLECTION_MANIFEST='7a5f43508b4a24dc22b809eec8b22834dfa40a603a3f7b28af20d0d1988ad21e'
INPUT_SHA='b05c7456ef9db17aec5bd1f72e76b0f5b3be3f5d6050e8bd950c1067888ff443'
BRANCH_RAW=Path('/data2/neural-shrimp-data/explore/s1r-branch-association/attempt-001')
BRANCH_MANIFEST='45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
SETTINGS={'precision':256,'radius_exponent':-80,'newton_steps':5,'max_power':4096}
CAP=32*1024**2
SELF_SOURCES=['tools/audit_s1r72_certificate.py','tests/test_s1r72_certificate_audit.py',
              'docs/reviews/s1r72-certificate-outcome-card.md']


def require(value,message):
    if not value:raise ValueError(message)


def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def entry(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':sha(raw)}


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def inventory(folder,expected_sha,expected_names=None,cap=CAP):
    require(folder.is_dir() and not folder.is_symlink(),'real artifact directory')
    manifest=folder/'manifest.json';require(not manifest.is_symlink(),'manifest symlink')
    require(sha(manifest.read_bytes())==expected_sha,'manifest identity')
    m=read(manifest);files=m['files'];require(isinstance(files,dict),'manifest files')
    actual=set()
    for p in folder.rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'unsafe artifact entry')
        if p.is_file():actual.add(str(p.relative_to(folder)))
    require(actual==set(files)|{'manifest.json'},'complete artifact inventory')
    if expected_names is not None:require(set(files)==set(expected_names),'expected artifact inventory')
    for name,value in files.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'safe artifact path')
        require(entry(folder/relative)==value,'artifact size/hash '+name)
    total=sum(v['bytes'] for v in files.values())
    require(total+manifest.stat().st_size<=cap,'cumulative artifact cap')
    if 'total_bytes' in m:require(m['total_bytes']==total and m['within_byte_cap'] is True,'manifest total/cap')
    return m


@lru_cache(None)
def git_bytes(commit,name):
    return subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)


def verify_sources(values,commit,names):
    require(set(values)==set(names),'complete source inventory')
    for name in names:
        expected=sha(git_bytes(REVIEWED,name))
        require(values[name]==expected==sha(git_bytes(commit,name))==sha((ROOT/name).read_bytes()),'bound source '+name)


def runtime_pair(folder,action,source,expected_status):
    start=read(folder/'runtime-start.json');end=read(folder/'runtime-end.json')
    require(end=={**start,**{k:end[k] for k in ('elapsed_seconds','max_rss_kib','status')}},'runtime completion consistency')
    require(start['source_commit']==source and end['status']==expected_status,'runtime source/status')
    verify_sources(start['source_sha256'],source,producer.SOURCES)
    require(start['settings']==SETTINGS and start['action']==action,'settings/action')
    require(type(start['pid']) is int and start['pid']>0 and start['cpu_threads']==1 and
            type(start['nice']) is int and start['nice']>=10 and start['gpu'] is False and start['paid'] is False,'runtime scope')
    require(finite(end['elapsed_seconds']) and 0<end['elapsed_seconds']<=(20 if action=='profile' else 90),'runtime elapsed cap')
    require(finite(end['max_rss_kib']) and end['max_rss_kib']>0,'runtime RSS')
    return start,end


def check_profile_measurements(result,end):
    values=[result[k] for k in ('map_seconds','projected_seconds','max_rss_kib')]
    require(all(finite(v) for v in values),'finite profile measurements')
    require(0<result['map_seconds']<=end['elapsed_seconds']<=20 and
            0<result['max_rss_kib']<=end['max_rss_kib']<=1024**2,'profile measured caps')
    require([result[k] for k in ('map_calls_in_checker','safety_factor','overhead_allowance_seconds')]==[14,2,10],'profile cost constants')
    projection=28*result['map_seconds']+10
    require(result['passed'] is True and result['projected_seconds']==projection<=90,'recomputed profile gate')
    require(result['evidence_rung']==0,'profile is not proof')
    return projection


def launcher(folder,action,source,runtime):
    prefix=str(folder);start_path=Path(prefix+'.launch-start.json');end_path=Path(prefix+'.launcher.json');log_path=Path(prefix+'.launcher.log')
    require(all(p.is_file() and not p.is_symlink() for p in (start_path,end_path,log_path)),'launcher paths')
    start=read(start_path);end=read(end_path)
    require(all(end.get(k)==v for k,v in start.items()),'launcher start/end binding')
    expected=['timeout','--signal=TERM','--kill-after=5s','25s' if action=='profile' else '110s','nice','-n','10',
              '/data2/neural-shrimp/.venv/bin/python','tools/certify_s1r_period2.py',action,'--out',str(folder)]
    variants=[expected]
    if action=='collect':
        variants=[expected+['--profile',str(PROFILE)],expected[:-2]+['--profile',str(PROFILE),'--out',str(folder)]]
    require(start['argv'] in variants,'exact launcher command')
    require(start['source_commit']==source and start['source_review_commit']==REVIEW,'launcher source/review')
    require(type(start['pid']) is int and start['pid']>0 and end['exit_code']==0,'launcher process/exit')
    require(start['thread_env']=={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'launcher thread limits')
    require(end['log']==dict(path=str(log_path),**entry(log_path)),'launcher log binding')
    cap=30 if action=='profile' else 115
    require(finite(end['elapsed_seconds']) and runtime['elapsed_seconds']<=end['elapsed_seconds']<=cap,'launcher elapsed cap')
    times=[datetime.fromisoformat(v) for v in (start['start_utc'],runtime['started_utc'],end['end_utc'])]
    require(all(v.tzinfo is not None for v in times) and times[0]<=times[1]<=times[2],'launcher chronological binding')
    return {'files':{str(p):entry(p) for p in (start_path,end_path,log_path)},'pid':start['pid'],
            'child_pid':runtime['pid'],'elapsed_seconds':end['elapsed_seconds'],'exit_code':end['exit_code']}


def decode_record(folder,name):
    obj=read(folder/name);arrays={};used=set()
    if 'arrays_file' in obj:
        n=obj['arrays_file'];require(Path(n).name==n,'original array path')
        require(sha((folder/n).read_bytes())==obj['arrays_sha256'],'original array hash')
        with np.load(folder/n,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    def decode(value):
        if isinstance(value,dict) and 'array' in value:
            a=arrays[value['array']];used.add(value['array'])
            require(a.dtype.str==value['dtype'] and list(a.shape)==value['shape'] and value['order']=='C','original array metadata')
            require(sha(np.ascontiguousarray(a).tobytes())==value['sha256'],'original coordinate hash')
            return a
        if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    result=decode(obj['payload']);require(used==set(arrays),'unused original arrays');return obj['kind'],result


def compare_exact_inputs(inputs,saved,node,network):
    require(inputs['schema']=='neural-shrimp-c1-input-v1' and type(inputs['period']) is int and inputs['period']==2,'exact input schema/period')
    exact_network=dict(network)
    for k in ('in_scale','out_scale','loss_scale','ridge'):
        q=Fraction(float(network[k]));exact_network[k]=f'{q.numerator}/{q.denominator}'
    require(inputs['network']==exact_network,'exact network/scales')
    require(saved['x'].shape==(72,8) and saved['y'].shape==(72,) and node['pair'].shape==(2,72) and node['rates'].shape==(2,),'selected dimensions')
    for name,a in [('x',saved['x']),('y',saved['y']),('state',node['pair']),('rates',node['rates'])]:
        require(a.dtype==np.float64 and np.isfinite(a).all(),'binary64 finite '+name)
    require(bool(np.all(node['rates']>0)),'selected rates positive')
    def q(value):
        require(isinstance(value,str),'rational string');return Fraction(value)
    require([[q(v) for v in row] for row in inputs['x']]==[[Fraction(float(v)) for v in row] for row in saved['x']],'exact input data')
    require([q(v) for v in inputs['y']]==[Fraction(float(v)) for v in saved['y']],'exact target data')
    require([q(v) for v in inputs['approximation']]==[Fraction(float(v)) for v in node['pair'][0]],'selected exact phase')
    require([q(inputs[k]) for k in ('lr_hidden','lr_output')]==[Fraction(float(v)) for v in node['rates']],'actual saved rates')


def original_input_binding(inputs):
    inventory(BRANCH_RAW,BRANCH_MANIFEST)
    kind,saved=decode_record(BRANCH_RAW,'00000-inputs.json');kind2,node=decode_record(BRANCH_RAW,'00082-node.json')
    require(kind=='inputs' and kind2=='node' and node['id']=='node-0004' and node['purpose']=='hidden_backoff' and
            node['accepted'] is True and node['checks']['passed'] is True,'selected node identity')
    require(node['params'].dtype==np.float64 and node['params'].tobytes()==np.array([1.123569870288833,.6271428571428572]).tobytes(),'selected coordinates')
    plan_path=ROOT/'experiments/trainability/EXP-028-plan.json'
    spec=next(v for v in read(plan_path)['maps'] if v['map_id']=='S1R-28002')
    require(spec['system']=='S1R' and spec['seed']==28002,'unprotected selected model')
    net,x,y,initial=SYSTEMS['S1R'](28002)
    require(asdict(net)==saved['model']==spec['network'],'regenerated model')
    realized={k:array_sha256(v) for k,v in [('x',x),('y',y),('theta0',initial)]}
    require(realized==spec['data_sha256'],'frozen dataset identity')
    for k,v in [('x',x),('y',y),('theta0',initial)]:
        require(saved[k].shape==v.shape and saved[k].dtype==v.dtype and saved[k].tobytes()==v.tobytes(),'regenerated data '+k)
    compare_exact_inputs(inputs,saved,node,spec['network'])
    p=inputs['provenance']
    require(p['raw_root']==str(BRANCH_RAW) and p['raw_manifest_sha256']==BRANCH_MANIFEST and p['phase']==0 and
            p['node_record']=='00082-node.json' and p['node_id']=='node-0004','exact provenance')
    require(p['node_record_sha256']==sha((BRANCH_RAW/'00082-node.json').read_bytes()) and
            p['plan_sha256']==sha(plan_path.read_bytes()) and p['data_sha256']==realized and p['map_id']=='S1R-28002','provenance identities')
    return {'original_manifest_sha256':BRANCH_MANIFEST,'node':'node-0004','phase':0,'dimension':72,'data_sha256':realized,
            'exact_inputs_sha256':sha(canonical(inputs)),'map_evaluated':False}


def dyadic_pair(pair):
    require(isinstance(pair,list) and len(pair)==2 and all(type(v) is int for v in pair),'dyadic mantissa/exponent')
    n,e=pair;require(abs(e)<=100000,'bounded serialized exponent')
    return Fraction(n)*(Fraction(2)**e)


def proximity(inputs,box):
    seed=[Fraction(v) for v in inputs['approximation']]
    require(len(seed)==72 and len(box)==72,'proximity dimension')
    scale=max(Fraction(1),*(abs(v) for v in seed));rows=[];largest=Fraction(0)
    for i,(s,b) in enumerate(zip(seed,box)):
        require(isinstance(b,dict) and 'mid' in b and 'rad' in b and b.get('finite',True) is not False,'finite ball representation')
        m=dyadic_pair(b['mid']);r=dyadic_pair(b['rad']);require(r>=0,'nonnegative ball radius')
        bound=abs(m-s)+r;largest=max(largest,bound)
        rows.append(dict(coordinate=i,seed=str(s),midpoint=str(m),radius=str(r),absolute_distance_bound=str(bound),normalized_distance_bound=str(bound/scale)))
    value=largest/scale;cap=Fraction(1,10_000_000)
    return dict(passed=value<=cap,normalized_max_distance=str(value),threshold=str(cap),normalization=str(scale),
                coordinates=rows,scope='Whole-box proximity only; no branch identity, interval continuation or assertion that the seed belongs to the box.')


def nested_binding(folder,outer_start,outer_end,inputs):
    receipt=read(folder/'receipt.json');start=read(folder/'runtime-start.json');end=read(folder/'runtime-end.json');cert=read(folder/'certificate.json')
    require(receipt['schema']=='neural-shrimp-c1-receipt-v1' and receipt['external_root']==str(folder.resolve()),'nested receipt/root')
    required={'inputs.json','certificate.json','runtime-start.json','runtime-end.json'}
    require(set(receipt['files'])==required,'nested inventory')
    for name,value in receipt['files'].items():require(entry(folder/name)==value,'nested hash '+name)
    require((folder/'inputs.json').read_bytes()==canonical(inputs),'nested exact input identity')
    require(start['source_commit']==outer_start['source_commit']==receipt['source_commit'] and start['pid']==outer_start['pid'],'nested execution identity')
    verify_sources(start['source_sha256'],start['source_commit'],producer.certify_c1.SOURCES)
    require(start['source_sha256']==receipt['source_sha256'] and cert['runtime']==start,'nested source/runtime binding')
    require(end=={**start,**{k:end[k] for k in ('elapsed_seconds','max_rss_kib')}},'nested completion metadata')
    require(start['cpu_threads']==1 and start['nice']>=10 and start['gpu']=='not used' and start['paid_compute'] is False,'nested resource scope')
    require(finite(end['elapsed_seconds']) and 0<end['elapsed_seconds']<=outer_end['elapsed_seconds'] and
            finite(end['max_rss_kib']) and 0<end['max_rss_kib']<=outer_end['max_rss_kib'],'nested runtime bounds')
    require(receipt['elapsed_seconds']==end['elapsed_seconds'] and receipt['max_rss_kib']==end['max_rss_kib'],'nested receipt measurements')
    require(cert['schema']=='neural-shrimp-c1-certificate-v1' and cert['inputs_sha256']==sha(canonical(inputs)),'nested certificate input')
    result=cert['result'];require(result['settings']==SETTINGS,'fixed proof settings')
    statement=result['exact_statement']
    require(result['passed'] is True and result['evidence_rung']==3 and statement['primitive_period']==2 and statement['dimension']==72 and
            all(statement[k] is True for k in ('existence_and_local_uniqueness','primitive_period_excluded_divisors','attraction')),'all proof gates')
    require(read(folder.parent/'result.json')==receipt,'wrapper/nested result identity')
    return cert


def bindings():
    pnames={'inputs.json','runtime-start.json','runtime-end.json','result.json'}
    pmanifest=inventory(PROFILE,PROFILE_MANIFEST,pnames)
    names=pnames|{'certificate/'+v for v in ('inputs.json','runtime-start.json','runtime-end.json','certificate.json','receipt.json')}
    cm=inventory(COLLECTION,COLLECTION_MANIFEST,names)
    require(pmanifest['status']=='profile_passed' and cm['status']=='certificate_passed','completed passing attempt statuses')
    ps,pe=runtime_pair(PROFILE,'profile',PROFILE_SOURCE,'profile_passed');cs,ce=runtime_pair(COLLECTION,'collect',COLLECTION_SOURCE,'certificate_passed')
    raw=(COLLECTION/'inputs.json').read_bytes();require(sha(raw)==INPUT_SHA and raw==(PROFILE/'inputs.json').read_bytes(),'fixed/profile exact inputs')
    inputs=json.loads(raw);require(raw==canonical(inputs),'canonical exact inputs')
    projected=check_profile_measurements(read(PROFILE/'result.json'),pe)
    require(cs['profile']==dict(root=str(PROFILE.resolve()),manifest=entry(PROFILE/'manifest.json'),result=entry(PROFILE/'result.json')),'collection profile reference')
    launch={action:launcher(folder,action,source,end) for action,folder,source,end in
            [('profile',PROFILE,PROFILE_SOURCE,pe),('collect',COLLECTION,COLLECTION_SOURCE,ce)]}
    original=original_input_binding(inputs);cert=nested_binding(COLLECTION/'certificate',cs,ce,inputs)
    report=dict(profile_manifest_sha256=PROFILE_MANIFEST,collection_manifest_sha256=COLLECTION_MANIFEST,
                original_inputs=original,profile_projection_seconds=projected,launchers=launch,
                profile_runtime=pe,collection_runtime=ce,nested_receipt=entry(COLLECTION/'certificate/receipt.json'),
                proof_environment=cert['runtime'])
    return report,inputs,cert


def replay_and_proximity(inputs,cert,write):
    replay=verify_certificates.verify(COLLECTION/'certificate/receipt.json')
    require(replay['passed'] is True,'verifier passed')
    write('stage-02-verifier.json',replay)
    near=proximity(inputs,cert['result']['box']);write('stage-03-exact-proximity.json',near)
    return near


def self_binding():
    names=SELF_SOURCES+producer.SOURCES
    for name in names:require((ROOT/name).read_bytes()==git_bytes('HEAD',name),'uncommitted audit dependency '+name)
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sha256={name:sha((ROOT/name).read_bytes()) for name in names})


def save(folder,name,value):
    raw=canonical(value);total=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    require(total+len(raw)<=CAP,'audit raw cap')
    with (folder/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(a.out.resolve().parent==Path('/data2/neural-shrimp-data/audits/s1r-period2').resolve(),'exclusive audit storage parent')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    identity=self_binding();a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report=dict(status='running',audit_source=identity,started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),cpu_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=False,paid=False,verifier_replays=0,python=platform.python_version(),numpy=np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__)
    save(a.out,'runtime-start.json',report)
    def stop(*_):raise TimeoutError('bounded independent audit interrupted')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,100)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        bound,inputs,cert=bindings();save(a.out,'stage-01-bindings.json',bound)
        report['verifier_replays']=1
        near=replay_and_proximity(inputs,cert,lambda name,value:save(a.out,name,value))
        report.update(status='passed' if near['passed'] else 'certificate_passed_proximity_failed',
                      mathematical_certificate_passed=True,selected_state_proximity_passed=near['passed'],
                      normalized_max_distance=near['normalized_max_distance'],scope='Exact-real72D primitive period-two existence/local uniqueness/attraction plus separately assessed selected-state proximity. Shared Arb checker/theorems; no formalization, branch identity, geometry, float-implementation or utility claim.')
    except BaseException as error:
        report.update(status='failed_or_incomplete',error=type(error).__name__+': '+str(error));save(a.out,'failure.json',report)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(a.out,'audit.json',report)
        files={p.name:entry(p) for p in sorted(a.out.iterdir()) if p.is_file()}
        save(a.out,'manifest.json',dict(files=files,status=report['status'],total_bytes=sum(v['bytes'] for v in files.values())))
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'],verifier_replays=report['verifier_replays'],manifest=entry(a.out/'manifest.json')),indent=2))
    return 0 if report['status'] in ('passed','certificate_passed_proximity_failed') else 2

if __name__=='__main__':raise SystemExit(main())
