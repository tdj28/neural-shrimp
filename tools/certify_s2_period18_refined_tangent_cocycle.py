"""Exactly18 new mixed calls on the unchanged doubled slice/refined tangent."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import argparse,json,math,platform,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import analyze_s2_period18_tangent_variation as prior
io=prior.io
from neural_shrimp.rigorous_connection import require,restore_ball
from neural_shrimp.rigorous_connection_frames import restore_matrix
from neural_shrimp.rigorous_orbits import exact,eye,matrix_json,vector_json
from neural_shrimp.rigorous_correlated_cocycle import total_jacobian_derivative
from neural_shrimp.rigorous_norm_cocycle import framed_cocycle
CARD='docs/plans/s2-period18-refined-tangent-cocycle.md'
SOURCES=list(dict.fromkeys(prior.SOURCES+['tools/certify_s2_period18_refined_tangent_cocycle.py','tools/launch_s2_period18_refined_tangent_cocycle.py','tests/test_s2_period18_refined_tangent_cocycle.py',CARD]))
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-tangent-variation-2026-10-10/attempt-001/run')
PIN='00740edbfdbf33ccc47af79ecc3751d239176c9160e66b5ad888b4a4491a2e77'
EXECUTION='db95fc152ce4253c598719b05a85480aa2bdd28a'
CAP=64*1024**2;io.CAP=CAP

class DurabilityError(RuntimeError):pass

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(data)
    return result

def target_input():
    rt,out=io.authenticate(PARENT,PIN,EXECUTION)
    require(rt['parent_manifest']==prior.PIN and out['conditional_bound_passed'],'exact passed tangent dependency')
    art,aout=io.authenticate(prior.PARENT,prior.PIN,prior.EXECUTION)
    require(art['mode']=='target' and aout['result_status']=='complete_slice_bounds' and aout['stages']==85,'complete unchanged doubled derivative acquisition')
    inp=io.read(prior.PARENT/'inputs.json.gz');physical=io.read(prior.PARENT/'stage-009-physical_inputs.json.gz');refined=io.read(PARENT/'tangent-result.json.gz')
    require(refined['passed'] and refined['center_derivative']['passed'] and refined['second_derivative']['passed'],'complete true tangent proof gates')
    require(inp['exact_endpoints']==physical['exact_endpoints']==prior.acquisition.ENDPOINTS,'unchanged exact interval')
    old=io.read(prior.PARENT/'slice-result.json.gz')
    require(vector_json([restore_ball(v) for v in old['whole_localization']['final_state_box']])==physical['actual_whole_box'],'same actual whole root domain')
    tangent=restore_matrix(refined['refined_derivative_box']);require(tangent.nrows()==235 and tangent.ncols()==1,'full235D tangent')
    frames=io.read(prior.PARENT/'stage-029-frames.json.gz')
    require(frames['cyclic_last_index']==0 and [v['phase'] for v in frames['witnesses']]==list(range(18)),'same native cyclic frames')
    centers=[]
    for j in range(18):
        row=io.read(prior.PARENT/f'stage-{10+j:03d}-phase_derivatives.json.gz');require(row['phase']==j,'native center Jacobian phase');centers.append(row['center_J'])
    return {'model_inputs':inp['model_inputs'],'exact_endpoints':inp['exact_endpoints'],'whole_box':physical['actual_whole_box'],
      'whole_tangent':vector_json([tangent[i,0] for i in range(235)]),'center_J':centers,
      'bases':[matrix_json(restore_matrix(v['B']).transpose()) for v in frames['witnesses']],
      'halfwidth':'1/2048','bindings':{'original_manifest':prior.PIN,'tangent_manifest':PIN},'scientific_claim':False}

def profile_input():
    inp=io.io.profile_input();model=inp['model_inputs'];n=13;p=18
    # Same unrelated zero-state ridge model as the earlier full profile.
    # Its exact root is theta=0 for all h; the fixed output rate is1/4.
    net=model['network'];hidden=(net['d']+int(net['bias']))*net['width']
    require(hidden==9 and exact(net['ridge'])==1,'synthetic ridge identity')
    j=eye(n)*exact('3/4');j[n-1,n-1]=exact('1/2')
    return {'model_inputs':model,'exact_endpoints':['-1/2048','1/2048'],
      'whole_box':vector_json([arb(0)]*(p*n)+[arb(exact('1/4'),exact('1/2048'))]),
      'whole_tangent':vector_json([arb(0)]*(p*n)+[arb(1)]),'center_J':[matrix_json(j)]*p,
      'bases':[matrix_json(eye(n))]*p,'halfwidth':'1/2048','scientific_claim':False}

def acquire(model,inp,tick=lambda:None,observe=lambda k,r:None):
    box=[restore_ball(v) for v in inp['whole_box']];tangent=[restore_ball(v) for v in inp['whole_tangent']]
    n=model.n;p=len(inp['center_J']);require(p==len(inp['bases']) and 1<=p<=18 and len(box)==len(tangent)==p*n+1,'full native inventories')
    require(all(v.is_finite() for v in box+tangent) and box[-1]>0,'finite actual refined derivative inputs')
    observe('actual_inputs',{'exact_endpoints':inp['exact_endpoints'],'actual_whole_box':vector_json(box),'actual_uniform_tangent':vector_json(tangent),'halfwidth':inp['halfwidth']})
    rows=[]
    for j in range(p):
        tick();whole,prime=total_jacobian_derivative(model,box[j*n:(j+1)*n],box[-1],tangent[j*n:(j+1)*n],tangent[-1])
        row={'phase':j,'center_J':restore_matrix(inp['center_J'][j]),'whole_Jprime':prime};rows.append(row)
        observe('refined_phase',{'phase':j,'center_J':inp['center_J'][j],'whole_J_mixed_diagnostic':matrix_json(whole),'whole_total_Jprime':matrix_json(prime)})
    return framed_cocycle(rows,[restore_matrix(v) for v in inp['bases']],exact(inp['halfwidth']),tick=tick,observe=observe)

def profile_gate(path,pin,outer_pin):
    require(io.entry(path.parent/'manifest.json')['sha256']==outer_pin,'profile outer pin');io.verify_inventory(path.parent,io.read(path.parent/'manifest.json')['files'])
    launch=io.read(path.parent/'launcher-result.json')
    require(launch['mode']=='profile' and launch['source_sha256']==sources() and launch['exit_code']==0 and launch['owned_live_pids']==[],'profile source/exit/ownership')
    for key,cap in [('elapsed_seconds',22),('max_group_rss_kib',1024**2)]:
        v=launch[key];require(type(v) in (int,float) and math.isfinite(v) and 0<v<cap,'positive bounded launcher '+key)
    require(io.entry(path/'manifest.json')['sha256']==pin,'profile worker pin');io.verify_inventory(path,io.read(path/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json')
    require(rt['mode']=='profile' and rt['source_sha256']==sources() and out['exit_code']==0 and out['passed'] and out['stages']==75,'complete positive75stage profile')
    for k,v in {'python':platform.python_version(),'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,'precision':256,'internal_seconds':10,'mixed_call_cap':18,'model_C1_calls':0}.items():require(rt.get(k)==v,'same library/settings '+k)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10 and out['mixed_counts']=={'attempted':18,'returned':18,'cap':18},'all18 actual mixed calls')
    elapsed=out['elapsed_seconds'];rss=out['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<10 and 4*elapsed+5<30,'measured time gate')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<rss<512*1024,'doubled RSS gate')
    estimate=max(6*io.total(path),16*1024**2);require(estimate<56*1024**2,'measured raw gate')
    return {'worker_manifest':pin,'outer_manifest':outer_pin,'projected_seconds':4*elapsed+5,'projected_bytes':estimate,'scope':'engineering only'}

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();limit=10 if mode=='profile' else 30;serial=0;stopped=False;model=None;code=2
    rt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),
      'started_utc':io.now(),'precision':256,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1'),'python':platform.python_version(),
      'python_flint':io.flint.__version__,'flint':io.flint.__FLINT_VERSION__,'numpy':io.numpy.__version__,'scipy':io.scipy.__version__,'internal_seconds':limit,
      'mixed_call_cap':18,'model_C1_calls':0,'rss_cap_kib':1024**2,'raw_cap':CAP,'tangent_manifest':PIN if mode=='target' else None}
    io.save(out/'runtime-start.json',rt);outcome={'status':'started','passed':False,'scientific_claim':False}
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
    def tick():
        if stopped or time.monotonic()-started>=limit:raise TimeoutError('fixed mixed time cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('fixed RSS cap')
        if io.total(out)>56*1024**2:raise OSError('ordinary raw cap')
    def observe(kind,row):
        nonlocal serial
        try:
            name=f'stage-{serial:03d}-{kind}';payload=io.save(out/(name+'.json.gz'),row);io.save(out/(name+'-payload.json'),payload);serial+=1
        except Exception as exc:raise DurabilityError('failed durable stage:'+str(exc))from exc
        tick()
    try:
        if mode=='target':io.save(out/'cost-gate.json',profile_gate(profile,profile_pin,profile_outer_pin))
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else target_input();payload=io.save(out/'inputs.json.gz',inp);io.save(out/'inputs-payload.json',payload);tick()
            model=io.CountedMixedMap(inp['model_inputs']);require(model.n==13,'native13D')
            result=acquire(model,inp,tick,observe);result['exact_endpoints']=inp['exact_endpoints']
            payload=io.save(out/'result.json.gz',result);io.save(out/'result-payload.json',payload);outcome['passed']=result['passed'];tick()
        code=0;outcome['status']='fixed_refined_cocycle_completed'
    except BaseException as exc:outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['mixed_counts']=model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stages=serial,exit_code=code,finished_utc=io.now())
        io.save(out/'outcome.json',outcome);io.save(out/'manifest.json',{'files':io.inventory(out)})
    print(json.dumps(outcome));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
