"""Bounded S1R saved-node adapter for the unchanged C1 Arb checker."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from flint import ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp.rigorous_orbits import BallTanhMap, canonical_bytes, dyadic, exact, sha256
from neural_shrimp.trainability import SYSTEMS, array_sha256
from audit_s1r_branch_outcome import inventory, record, RAW, MANIFEST, require
import certify_c1

NODE='00082-node.json'
SETTINGS=dict(precision=256,radius_exponent=-80,newton_steps=5,max_power=4096)
SOURCES=certify_c1.SOURCES+[
    'tools/certify_s1r_period2.py','tools/audit_s1r_branch_outcome.py',
    'src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py',
    'src/neural_shrimp/models.py','tests/test_s1r_period2_certificate.py',
    'docs/plans/s1r-period2-certificate.md','experiments/trainability/EXP-028-plan.json']
MAX_BYTES=32*1024*1024


def source_binding():
    result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes()
        require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT), 'uncommitted source '+name)
        result[name]=sha256(data)
    return result


def exact_inputs(model,x,y,theta,rates,provenance):
    require(x.shape==(72,8) and y.shape==(72,) and theta.shape==(72,) and rates.shape==(2,), 'S1R dimensions')
    require(all(a.dtype==np.float64 and np.isfinite(a).all() for a in (x,y,theta,rates)), 'finite binary64 inputs')
    require(bool(np.all(rates>0)), 'positive rates')
    network=dict(model)
    for name in ('in_scale','out_scale','loss_scale','ridge'):network[name]=dyadic(network[name])
    return dict(schema='neural-shrimp-c1-input-v1',network=network,
        x=[[dyadic(v) for v in row] for row in x],y=[dyadic(v) for v in y],
        lr_hidden=dyadic(rates[0]),lr_output=dyadic(rates[1]),period=2,
        approximation=[dyadic(v) for v in theta],provenance=provenance)


def prepare():
    inventory(RAW,MANIFEST)
    _,saved=record(RAW,'00000-inputs.json');kind,node=record(RAW,NODE)
    require(kind=='node' and node['id']=='node-0004' and node['accepted'] is True, 'selected qualified node')
    require(node['purpose']=='hidden_backoff' and node['checks']['passed'] is True,'node qualification')
    require(node['pair'].shape==(2,72), 'pair dimensions')
    require(node['params'].tobytes()==np.array([1.123569870288833,.6271428571428572],dtype=np.float64).tobytes(), 'selected coordinates')
    plan_path=ROOT/'experiments/trainability/EXP-028-plan.json'
    spec=next(m for m in json.loads(plan_path.read_bytes())['maps'] if m['map_id']=='S1R-28002')
    require(spec['seed']==28002 and spec['system']=='S1R','development model')
    net,x,y,theta0=SYSTEMS['S1R'](28002)
    realized={name:array_sha256(a) for name,a in [('x',x),('y',y),('theta0',theta0)]}
    require(realized==spec['data_sha256'] and asdict(net)==spec['network']==saved['model'],'frozen model/data')
    for name,a in [('x',x),('y',y),('theta0',theta0)]:
        require(saved[name].dtype==a.dtype and saved[name].shape==a.shape and saved[name].tobytes()==a.tobytes(),'saved data '+name)
    provenance=dict(raw_root=str(RAW),raw_manifest_sha256=MANIFEST,node_record=NODE,node_id=node['id'],phase=0,
        node_record_sha256=sha256((RAW/NODE).read_bytes()),plan_sha256=sha256(plan_path.read_bytes()),
        map_id=spec['map_id'],data_sha256=realized,
        selection='Previously cross-checked attracting local-segment base node0004, selected before this proof attempt; no new search.',
        semantics='Saved binary64 data, ridge, rates and approximation converted to exact dyadic rationals; tanh and GD then exact real. The log10 coordinates are provenance, not recomputed rates.')
    return exact_inputs(saved['model'],saved['x'],saved['y'],node['pair'][0],node['rates'],provenance)


def entry(path):
    data=path.read_bytes();return dict(bytes=len(data),sha256=sha256(data))


def manifest(folder,status):
    files={str(p.relative_to(folder)):entry(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'manifest.json'}
    size=sum(v['bytes'] for v in files.values())
    payload=dict(schema='s1r-period2-attempt-v1',status=status,files=files,total_bytes=size,within_byte_cap=size<=MAX_BYTES)
    (folder/'manifest.json').write_bytes(canonical_bytes(payload))
    return payload


def profile_gate(profile,sources,inputs):
    m=json.loads((profile/'manifest.json').read_bytes())
    require(m['status']=='profile_passed' and m['within_byte_cap'] is True,'completed resource profile')
    actual={str(p.relative_to(profile)) for p in profile.rglob('*') if p.is_file()}
    require(actual==set(m['files'])|{'manifest.json'},'profile inventory')
    for name,e in m['files'].items():
        path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'profile path')
        require(not (profile/path).is_symlink() and entry(profile/path)==e,'profile hash '+name)
    require(m['total_bytes']==sum(v['bytes'] for v in m['files'].values())<=MAX_BYTES,'profile byte cap')
    start=json.loads((profile/'runtime-start.json').read_bytes());end=json.loads((profile/'runtime-end.json').read_bytes())
    result=json.loads((profile/'result.json').read_bytes())
    for meta in (start,end):
        require(meta['source_sha256']==sources and meta['settings']==SETTINGS,'profile source/settings')
        require(meta['action']=='profile' and meta['cpu_threads']==1 and meta['nice']>=10 and meta['gpu'] is False and meta['paid'] is False,'profile runtime scope')
    require(end['status']=='profile_passed' and end['source_commit']==start['source_commit'] and end['pid']==start['pid'],'completed profile runtime')
    require((profile/'inputs.json').read_bytes()==canonical_bytes(inputs),'profile input identity')
    def finite_number(value):return type(value) in (int,float) and math.isfinite(value)
    seconds=result['map_seconds'];elapsed=end['elapsed_seconds']
    rss=result['max_rss_kib'];end_rss=end['max_rss_kib']
    require(all(finite_number(v) for v in (seconds,elapsed,rss,end_rss,result['projected_seconds'])),'finite profile measurements')
    require(0<seconds<=elapsed<=20 and 0<rss<=end_rss<=1024*1024,'profile runtime limits')
    require(result['map_calls_in_checker']==14 and result['safety_factor']==2 and result['overhead_allowance_seconds']==10,'profile projection constants')
    projected=2*14*seconds+10
    require(result['passed'] is True and result['projected_seconds']==projected<=90,'profile gate')
    return dict(root=str(profile.resolve()),manifest=entry(profile/'manifest.json'),result=entry(profile/'result.json'))


def deadline(*_):raise TimeoutError('bounded S1R certificate deadline')


def execute(action,out,profile=None):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    require(action in ('profile','collect'),'unknown action')
    sources=source_binding()
    require(action!='collect' or profile is not None,'profile required before collect')
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status='incomplete';result={}
    runtime=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=sources,started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        nice=os.getpriority(os.PRIO_PROCESS,0),cpu_threads=1,gpu=False,paid=False,settings=SETTINGS,action=action)
    (out/'runtime-start.json').write_bytes(canonical_bytes(runtime))
    prior=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,20 if action=='profile' else 90)
    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_BYTES,MAX_BYTES))
    try:
        inputs=prepare();(out/'inputs.json').write_bytes(canonical_bytes(inputs))
        if action=='profile':
            # One map/Hessian call at the already known state, not a correction or proof verdict.
            with ctx.workprec(SETTINGS['precision']):
                model=BallTanhMap(inputs);theta=[exact(x) for x in inputs['approximation']]
                tick=time.monotonic();state,jac=model.step(theta);seconds=time.monotonic()-tick
                require(all(v.is_finite() for v in state) and all(jac[i,j].is_finite() for i in range(72) for j in range(72)),'finite profile')
            projected=2*14*seconds+10
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result=dict(passed=projected<=90 and rss<=1024*1024,map_seconds=seconds,
                projected_seconds=projected,max_rss_kib=rss,map_calls_in_checker=14,safety_factor=2,overhead_allowance_seconds=10,
                evidence_rung=0,scope='Resource timing only; no proof outcome or continuation.')
            status='profile_passed' if result['passed'] else 'profile_failed'
        else:
            runtime['profile']=profile_gate(profile,sources,inputs)
            (out/'runtime-start.json').write_bytes(canonical_bytes(runtime))
            code=certify_c1.run(inputs,out/'certificate',SETTINGS)
            result=json.loads((out/'certificate/receipt.json').read_bytes())
            status='certificate_passed' if code==0 else 'certificate_failed'
        (out/'result.json').write_bytes(canonical_bytes(result))
    except Exception as error:
        status='failed';result=dict(passed=False,evidence_rung=0,error=type(error).__name__+': '+str(error))
        (out/'failure.json').write_bytes(canonical_bytes(result))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        runtime.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,status=status)
        (out/'runtime-end.json').write_bytes(canonical_bytes(runtime));m=manifest(out,status)
    print(json.dumps(dict(status=status,external_root=str(out.resolve()),manifest=entry(out/'manifest.json'),
        elapsed_seconds=runtime['elapsed_seconds'],within_byte_cap=m['within_byte_cap'],passed=result.get('passed',False)),indent=2))
    return 0 if result.get('passed',False) and m['within_byte_cap'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['profile','collect'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path)
    a=p.parse_args();return execute(a.action,a.out,a.profile)

if __name__=='__main__':raise SystemExit(main())
