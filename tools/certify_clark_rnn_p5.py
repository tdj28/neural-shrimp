#!/usr/bin/env python3
"""Bounded exact-native RNN orbit pilot with complete origin/source bindings."""
import argparse,importlib.util,json,os,platform,resource,signal,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import flint
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('store',ROOT/'tools/explore_clark_rnn.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
from neural_shrimp.rigorous_orbits import canonical_bytes,dyadic
from neural_shrimp.rigorous_clark_rnn import check_rnn_orbit
BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn');PIN='20d1171d8ccfcc41743d85b8757c5c8affbd15b1c393d6f5209af971b363f77f'
EXTRA=['src/neural_shrimp/rigorous_clark_rnn.py','src/neural_shrimp/rigorous_orbits.py','tools/certify_clark_rnn_p5.py','tools/launch_clark_rnn_p5_certificate.py','docs/plans/clark-rnn-p5-certificate.md']
SETTINGS={'precision':256,'radius_exponent':-80,'newton_steps':5,'max_power':512}
THREADS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']

def verify(root,pin):
    if S.sha(root/'manifest.json')!=pin:raise ValueError('manifest pin')
    inv=json.loads((root/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}:raise ValueError('inventory')
    for name,d in inv.items():
        p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('input bytes')
    return inv

def target_inputs():
    raw=BASE/'p5-boundary-slices-001';inv=verify(raw,PIN);bind=json.loads((raw/'run/input-binding.json').read_text());ref=BASE/'refine-p5-001/run/p5/inputs.npz'
    if S.sha(ref)!=bind['source_input_sha256']:raise ValueError('exact matrix origin')
    files=[p for p in (raw/'run').glob('*-native_eigen.json') if json.loads(p.read_text())['tag']=='g16-seed']
    if len(files)!=1:raise ValueError('unique predefined body seed')
    desc=files[0];meta=json.loads(desc.read_text());arrays=desc.with_name(meta['array_file'])
    if S.sha(arrays)!=meta['array_sha256']:raise ValueError('seed identity')
    with np.load(ref) as f:j=f['j']
    with np.load(arrays) as f:theta=f['theta']
    if j.shape!=(4,4) or theta.shape!=(4,) or meta['gain']!=10.02 or meta['delta']!=.956:raise ValueError('fixed body target')
    inputs={'schema':'clark-rnn-c1-input-v1','j0':[[dyadic(x) for x in row] for row in j],'gain':dyadic(meta['gain']),'delta':dyadic(meta['delta']),'period':5,'approximation':[dyadic(x) for x in theta]}
    return inputs,{'raw_manifest_sha256':PIN,'raw':str(raw),'seed_descriptor':str(desc),'seed_descriptor_sha256':S.sha(desc),'seed_arrays_sha256':S.sha(arrays),'matrix_input':str(ref),'matrix_input_sha256':S.sha(ref),'proposal_execution_commit':'a623ef362a5c01331f1e0dccf5d80d77ff37947a','dimension':4,'period':5,'meaning':'exact-real map with binary64 dyadic constants, not a floating-point recurrence theorem'}

def pair(a):return Fraction(a[0])*Fraction(2)**a[1]
def proximity(inputs,result):
    seed=[Fraction(x) for x in inputs['approximation']];scale=max(Fraction(1),*(abs(x) for x in seed));bounds=[]
    for x,b in zip(seed,result['box']):
        lo=pair(b['mid'])-pair(b['rad']);hi=pair(b['mid'])+pair(b['rad']);bounds.append(max(abs(lo-x),abs(hi-x))/scale)
    maximum=max(bounds)
    return {'exact_fraction':str(maximum),'cap':'1/10000000','passed':maximum<Fraction(1,10**7),'scope':'Whole returned root box is near the selected floating seed; this proves proximity, not branch/geometry ancestry.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['profile','target','replay'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile',type=Path);ap.add_argument('--profile-sha256');ap.add_argument('--bundle',type=Path);ap.add_argument('--manifest-sha256');a=ap.parse_args()
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('CPU1 nice10')
    a.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=a.out;S.RAW_CAP=8*1024**2;t=time.monotonic();interrupted=[False];signal.signal(signal.SIGTERM,lambda *_:interrupted.__setitem__(0,True))
    def check():
        if interrupted[0]:raise InterruptedError('SIGTERM')
        if time.monotonic()-t>30:raise TimeoutError('30s internal cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
    runtime={'source':identity,'pid':os.getpid(),'started_unix':time.time(),'mode':a.mode,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'python_flint':flint.__version__,'platform':platform.platform(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ.get(k) for k in THREADS},'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024};S.save(a.out/'runtime-start.json',runtime);result={'status':'incomplete'}
    try:
        if runtime['mem_available_bytes']<1024**3:raise MemoryError('host headroom')
        if a.mode=='profile':
            inputs={'schema':'clark-rnn-c1-input-v1','j0':[[str(Fraction(1,4)) if i==j else '0' for j in range(4)] for i in range(4)],'gain':'1','delta':'1/2','period':5,'approximation':['0']*4};times=[]
            for i in range(2):
                check();start=time.monotonic();q=check_rnn_orbit(inputs,**SETTINGS);S.save(a.out/f'profile-{i}.json',q);times.append(time.monotonic()-start)
            projection=8*max(times)+5;result.update(status='completed',unit_seconds=times,projected_seconds=projection,resource_pass=projection<=30,synthetic_primitive_failure_expected=True)
        else:
            if a.mode=='target':
                if a.profile is None or a.profile_sha256 is None:raise ValueError('profile required')
                verify(a.profile,a.profile_sha256);p=json.loads((a.profile/'result.json').read_text());rs=json.loads((a.profile/'runtime-start.json').read_text())
                if p['status']!='completed' or not p['resource_pass'] or rs['mode']!='profile' or rs['settings']!=SETTINGS or rs['source']['sha256']!=identity['sha256']:raise ValueError('profile identity/status')
                if any(rs[k]!=runtime[k] for k in ['python','numpy','python_flint','platform','threads']):raise ValueError('profile runtime')
                times=p['unit_seconds'];projection=8*max(times)+5
                if not all(np.isfinite(z) and z>0 for z in times+[p['elapsed_seconds'],p['peak_rss_kib']]) or len(times)!=2 or projection!=p['projected_seconds'] or projection>30 or p['elapsed_seconds']>30 or p['peak_rss_kib']>448*1024 or rs['nice']<10:raise ValueError('profile cost')
                inputs,origin=target_inputs();S.save(a.out/'origin.json',origin);S.save(a.out/'profile-binding.json',{'path':str(a.profile),'manifest_sha256':a.profile_sha256,'projection':projection})
            else:
                if a.bundle is None or a.manifest_sha256 is None:raise ValueError('replay inputs')
                verify(a.bundle,a.manifest_sha256);inputs=json.loads((a.bundle/'inputs.json').read_text())
            S.save(a.out/'inputs.json',inputs);check();proof=check_rnn_orbit(inputs,**SETTINGS);S.save(a.out/'proof.json',proof);near=proximity(inputs,proof);S.save(a.out/'whole-box-proximity.json',near);check()
            same=None
            if a.mode=='replay':same=canonical_bytes(proof)==canonical_bytes(json.loads((a.bundle/'proof.json').read_text()))
            result.update(status='completed',proof_passed=proof['passed'],proximity_passed=near['passed'],replay_identical=same,inputs_sha256=S.sha(a.out/'inputs.json'),odd_companion_distinct=proof['odd_companion_distinct'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);S.save(a.out/'result.json',result,final=True);S.manifest(a.out)
    return 0 if result['status']=='completed' and (a.mode=='profile' or result.get('proof_passed') and result.get('proximity_passed') and result.get('replay_identical') is not False) else 2
if __name__=='__main__':sys.exit(main())
