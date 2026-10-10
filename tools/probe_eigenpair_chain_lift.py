#!/usr/bin/env python3
"""One bounded authenticated native lift of the isolated generalized-chain root."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_eigenpair_generalized_chain as C
from neural_shrimp.rigorous_chain_lift import native_lift,quadratic_membership
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=C.BASE/'native-lift';PARENT=C.BASE/'attempt001-target'
PINS={'source':'b5a62752ba770154009ce694a09f1ede2826a5bd','outer':'bfb722e8c08dd4ad23c198f8020d8042fb804750f70bcef58994fbb0fed4d951','worker':'e108e6e965c8dfa70a986242bed7232674b051a2b233b2da7f106ecf3bc6e141'}
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/rigorous_chain_lift.py','tests/test_rigorous_chain_lift.py','tools/probe_eigenpair_chain_lift.py','tools/launch_eigenpair_chain_lift.py','tests/test_eigenpair_chain_lift_driver.py','docs/plans/eigenpair-chain-lift.md','docs/reviews/eigenpair-generalized-chain-target.json']))
SETTINGS={'precision':384,'period':18,'native_dimension':13,'shooting_dimension':470}
CAP=64*1024**2
read=C.read;entry=C.entry;inventory=C.inventory

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    result={n:entry(ROOT/n)for n in SOURCES}
    for n,row in result.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
    return result

def qualified_root(proof,out):
    require(out['passed'] is True and out['status']=='chain_proof_completed' and out['ticks']==360 and out['whole_box_proximity']is True,'completed fixed chain outcome')
    require(proof['schema']=='neural-shrimp-generalized-chain-root-v1' and proof['passed']is True,'chain certificate schema')
    require(proof['gates']==dict.fromkeys(('normalized_augmented_root_unique','positive_actual_rates','primitive_period'),True),'all same-root gates')
    require(out['gates']==proof['gates'],'outcome proof identity')
    require((proof['period'],proof['dimension'],proof['augmented_dimension'])==(18,13,41),'exact native root dimensions')
    require(proof['settings']=={'radius_exponent':-80,'newton_steps':8},'fixed chain proof settings')
    require(len(proof['box'])==41 and proof['coordinate_inclusions']==[True]*41 and decode(proof['contraction'])<1,'strict root certificate')
    require([r['divisor']for r in proof['divisors']]==[1,2,3,6,9] and all(r['excluded']is True for r in proof['divisors']),'native primitive18 certificate')
    box=[decode(v)for v in proof['box']];require(box[13]>0 and box[14]>0,'positive physical rates')
    return box

def bind_root():
    identities={'outer':inventory(PARENT,PINS['outer']),'worker':inventory(PARENT/'run',PINS['worker'])}
    rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');inp=read(PARENT/'run/inputs.json');proof=read(PARENT/'run/proof.json')
    require(rt['commit']==la['commit']==PINS['source'] and rt['mode']==la['mode']=='target','exact completed chain source')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None,'completed chain ownership')
    require(rt['settings']==C.SETTINGS and rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'original chain settings')
    require(set(rt['sources'])==set(C.SOURCES),'exact chain source inventory')
    for n,row in rt['sources'].items():
        require(entry(ROOT/n)==entry(PARENT/'run/source'/n)==row,'unchanged archived chain source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',PINS['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical chain source '+n)
    box=qualified_root(proof,out)
    model,seed,diagnostic,binding=C.bind_event()
    require(inp=={'map_inputs':model,'seed':vector_json(seed),'diagnostic':diagnostic,'bindings':binding},'exact chain seed/model/native parent binding')
    event=C.read(C.B.T.C.input_path(C.PARENT,binding['accepted_file']))['chart']
    identities.update(parent=str(PARENT),pins=PINS,proof=entry(PARENT/'run/proof.json'),inputs=entry(PARENT/'run/inputs.json'),native_binding=binding)
    return model,box,event,identities

def cost(out):
    require(out['passed']is True and out['status']=='synthetic_lift_completed' and out['ticks']==36,'two full native synthetic lifts')
    times=out['evaluation_seconds'];require(len(times)==2 and all(C.finite(t,10) and t>0 for t in times),'finite measured native lift cost')
    forecast=4*max(times)+15
    require(forecast<30,'measured target lift with4x time and15s binding reserve')
    return forecast

def profile_receipt(args,identity):
    require(args.profile==BASE/'attempt001-profile/run','fixed completed lift profile')
    inventory(args.profile,args.profile_manifest);inventory(args.profile.parent,args.profile_outer_manifest)
    rt=read(args.profile/'runtime.json');out=read(args.profile/'outcome.json');la=read(args.profile.parent/'launcher-result.json')
    require(rt['mode']==la['mode']=='profile' and rt['commit']==la['commit'] and rt['sources']==identity,'same-source profile')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None,'completed owned profile')
    for n,row in identity.items():
        require(entry(args.profile/'source'/n)==row,'archived profile source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profile source '+n)
    for k,v in {'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same profile '+k)
    require(rt['nice']>=10 and C.finite(out['elapsed_seconds'],30) and C.finite(out['peak_rss_kib'],2*1024**2),'bounded profile')
    return cost(out)

def main(args):
    identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');require(args.out==BASE/f'attempt001-{args.mode}'/'run','exclusive lift path')
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=[0];stop=[False];used=[0];serial=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    def check():
        if stop[0] or time.monotonic()-start>=30:raise TimeoutError('30s lift worker limit')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB sampled self RSS')
    def save(name,value,final=False):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();require(used[0]+len(raw)<CAP if final else used[0]+len(raw)<CAP-2*1024**2,'aggregate raw cap')
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():ticks[0]+=1;require(ticks[0]<=(36 if args.mode=='profile'else 18),'fixed native step ceiling');check()
    def observe(kind,row):save(f'event-{serial[0]:03d}-{kind}.json',row);serial[0]+=1;check()
    save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':args.mode,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid()})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
    result={'passed':False,'status':'unresolved','scope':'Authenticated native lifting and gauge projection only. A chart-membership failure is not branch disconnection; no unfolding or stability claim.'}
    try:
        with ctx.workprec(384):
            if args.mode=='profile':
                inp,box=C.synthetic();normal=['0']*470;normal[-2]='1';origin=['0']*470;times=[]
                save('inputs.json',{'map_inputs':inp,'box':vector_json(box),'normal':normal,'origin':origin,'synthetic_only':True})
                for i in range(2):
                    t=time.monotonic();lift=native_lift(TwoRateCuspMap(inp),box,18,normal,origin,tick,observe);save(f'lift-{i}.json',lift);times.append(time.monotonic()-t)
                result.update(passed=True,status='synthetic_lift_completed',evaluation_seconds=times)
                result['forecast_seconds']=cost({**result,'ticks':ticks[0]})
            else:
                forecast=profile_receipt(args,identity);save('profile_binding.json',{'worker':str(args.profile),'worker_manifest':args.profile_manifest,'outer_manifest':args.profile_outer_manifest,'forecast_seconds':forecast})
                inp,box,event,bindings=bind_root();save('inputs.json',{'map_inputs':inp,'box':vector_json(box),'native_event':event,'bindings':bindings});check()
                lift=native_lift(TwoRateCuspMap(inp),box,18,event['normal'],event['origin'],tick,observe);save('lift.json',lift)
                require(lift['closure_contains_zero'],'conditional proved chain must close inside propagated intervals')
                member=quadratic_membership(lift,event);save('existing_chart_membership.json',member)
                result.update(passed=True,status='authenticated_native_lift_completed',parameter=lift['parameter'],existing_chart_membership=member['passed'])
    except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],observations=serial[0],stored_bytes_before_final=used[0])
    if not C.finite(result['elapsed_seconds'],30) or not C.finite(result['peak_rss_kib'],2*1024**2):result['passed']=False
    save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
