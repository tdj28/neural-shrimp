#!/usr/bin/env python3
"""One source-bound local cusp attempt per preselected saved fold bracket."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip,importlib.util,json,resource,signal,sys,time,subprocess
from pathlib import Path
from fractions import Fraction
import numpy as np
import flint
from flint import ctx,arb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('supervisor',ROOT/'tools/explore_s2_boundary_curves.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
S.SOURCES+=['src/neural_shrimp/rigorous_cusps.py','tests/test_rigorous_cusps.py','tools/certify_s2_cusps.py','docs/plans/s2-period18-cusp-pilot.md']
S.__file__=__file__
from neural_shrimp.rigorous_cusps import TwoRateCuspMap,augmented_cusp,cusp_proof
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json,matrix_json
from neural_shrimp.rigorous_connection import restore_ball
BASE=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves')
PINS={'cusp-a':('attempt-003-target','1a1b58e678f11db81ca7adb2821ca2ff84056768734073a11e70e004813a71f9','node-+1-031.json'),
      'cusp-b':('attempt-004-fold-plus','231b1d448fd06b09a180861d4e67569d3ef0f42967557ba4e24dd7b4780284f6','node-+1-027.json')}
SETTINGS={'precision':256,'radius_exponent':-80,'newton_steps':8,'period':18}

def verify(base,pin):
    if S.sha(base/'manifest.json')!=pin:raise ValueError('fixed manifest identity')
    inv=json.loads((base/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p!=base/'manifest.json'}:raise ValueError('complete inventory')
    for name,row in inv.items():
        p=base/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or S.sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes'] or p.stat().st_mode&0o222:raise ValueError('immutable artifact identity')

def exact_string(value):
    mantissa,exponent=value.mid().man_exp()
    return str(Fraction(int(mantissa))*Fraction(2)**int(exponent))

def inputs(panel):
    name,pin,filename=PINS[panel];base=BASE/name;verify(base,pin)
    node=json.loads((base/'run'/filename).read_text());model=json.loads((base/'run/inputs.json').read_text())
    end=json.loads((base/'launcher-result.json').read_text())
    if not node['qualification']['qualified'] or model['sigma']!=1 or end['exit_code'] or end['owned_group_live_pids']:raise ValueError('qualified completed fold required')
    found=None
    with gzip.open(base/'run/evaluations.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            if row['kind']=='evaluation_return' and row['z']==node['point']:found=row
    if found is None:raise ValueError('saved analytic Jacobian required')
    z=[exact(s) for s in node['point_exact_dyadics']];q=np.array([float(v.mid()) for v in z[15:]])
    j=np.asarray(found['data']['jacobian']);a=np.asarray(found['data']['monodromy'])-np.eye(13)
    border=np.block([[a,q[:,None]],[q[None,:],np.zeros((1,1))]])
    curvature=np.linalg.solve(border,np.r_[-j[13:26,:13]@q,0])[:13]
    if not np.isfinite(curvature).all():raise ValueError('finite curvature proposal')
    z += [exact(dyadic(v)) for v in curvature]
    return model['map_inputs'],z,{'parent':str(base),'manifest_sha256':pin,'node':filename,'node_sha256':S.sha(base/'run'/filename),'saved_evaluation_id':found['id'],'meaning':'proposal at preselected simple-mode coefficient bracket; no interval ancestry or cusp assertion from floating solve'}

def synthetic():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
    z=[exact(str(Fraction(i,128))) for i in range(13)]+[exact('1/100')]*2+[exact('1/4')]*13+[arb(0)]*13
    return inp,z

def execute(args):
    identity=S.source();out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];ticks=[0];serial=[0];used=[0];seed=[None]
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2))
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10')
    limit=15 if args.mode=='profile' else 90
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-start>limit:raise TimeoutError('bounded cusp worker')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise MemoryError('512MiB RSS cap')
    def save(name,row):
        data=(json.dumps(row,indent=2,allow_nan=False)+'\n').encode();used[0]+=len(data)
        if used[0]>30*1024**2:raise OSError('30MiB ordinary output allowance')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    def tick():
        check();ticks[0]+=1
        if ticks[0]>(36 if args.mode=='profile' else 180):raise RuntimeError('native-step cap')
    def observe(kind,row):
        save(f'evaluation-{serial[0]:03d}-{kind}.json',row);serial[0]+=1;check()
        if kind=='newton_proposal':
            state=[restore_ball(v) for v in row['center']];scales=[1]*13+[10,.1]+[1]*13
            if any(abs(state[i]-seed[0][i])/exact(dyadic(scales[i]))>=exact('1/10') for i in range(28)):raise ValueError('proposal leaves fixed0.1 scaled neighborhood')
    runtime={'source':identity,'mode':args.mode,'panel':args.panel,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':{k:os.environ[k] for k in S.THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'started_unix':time.time(),'meminfo':Path('/proc/meminfo').read_text()}
    save('runtime-start.json',runtime)
    for p in S.SOURCES:
        target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes());used[0]+=target.stat().st_size
    result={'status':'started','passed':False};code=2
    try:
        with ctx.workprec(256):
            if args.mode=='profile':
                inp,z=synthetic();model=TwoRateCuspMap(inp);times=[]
                for i in range(2):
                    t=time.monotonic();v,j,_=augmented_cusp(model,z,18,tick);save(f'profile-{i}.json',{'value':matrix_json(v),'jacobian':matrix_json(j)});times.append(time.monotonic()-t)
                result.update(status='synthetic_profile_completed',evaluation_seconds=times,projected_seconds=20*max(times)+5,resource_pass=20*max(times)+5<85)
                code=0 if result['resource_pass'] else 2
            else:
                if not args.profile or not args.profile_manifest:raise ValueError('bound profile required')
                verify(args.profile,args.profile_manifest);profile=json.loads((args.profile/'outcome.json').read_text());prior=json.loads((args.profile/'runtime-start.json').read_text())
                if profile['status']!='synthetic_profile_completed' or not profile['resource_pass'] or profile['native_steps']!=36 or prior['mode']!='profile' or prior['source']['files']!=identity['files']:raise ValueError('profile identity/status')
                for key in ('settings','python','numpy','flint','flint_library','threads'):
                    if runtime[key]!=prior[key]:raise ValueError('profile runtime equality')
                times=profile['evaluation_seconds']
                if len(times)!=2 or not all(np.isfinite(v) and v>0 for v in times) or 20*max(times)+5>=85 or profile['peak_rss_kib']>=512*1024:raise ValueError('measured cost gate')
                inp,z,origin=inputs(args.panel);seed[0]=z;save('inputs.json',{'map_inputs':inp,'approximation':[exact_string(v) for v in z],'exact_approximation':vector_json(z),'period':18});save('origin.json',origin);save('profile-binding.json',{'path':str(args.profile),'manifest_sha256':args.profile_manifest})
                proof=cusp_proof(TwoRateCuspMap(inp),z,18,radius_exponent=-80,newton_steps=8,tick=tick,observe=observe);save('proof.json',proof);check()
                scales=[1]*13+[10,.1]+[1]*13;near=all(abs(restore_ball(proof['box'][i])-z[i])/exact(dyadic(scales[i]))<exact('1/10') for i in range(28));save('whole-box-proximity.json',{'passed':near,'scaled_cap':'1/10','scales':scales,'scope':'finite proposal neighborhood only; curvature variables excluded; no branch identity'})
                result.update(status='cusp_proof_completed',passed=bool(proof['passed'] and near),gates=proof['gates'],whole_box_proximity=near);code=0 if result['passed'] else 2
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_steps=ticks[0],observations=serial[0],exit_code=code);S.save(out/'outcome.json',result);S.save(out/'manifest.json',{'files':S.inventory(out)});print(json.dumps(result));return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--panel',choices=['cusp-a','cusp-b'],default='cusp-a');p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((execute if a.execute else S.launch)(a))
