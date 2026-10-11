"""Fixed26 independent AD proof panel with a measured three-phase profile."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
import flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(Path(__file__).parent)]
import review_census13_ad_math as A
from review_c1_evidence import objective_ad
from neural_shrimp.rigorous_orbits import vector_json,matrix_json
need=A.need
BASE=Path('/data2/neural-shrimp-data/audits/census13-independent-ad')
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/census13-panel/attempt003-target')
PINS=dict(source='4d15d945',outer='0ad52e1408d9427a3ecbbdb0230c0aa3eaf32785cb54feee479177e0a3c3f314',worker='80d9338971a56e90e379084d0f290de47f7a83a972e61e5323eb5bbf479d4e37')
FILES=['tools/reviews/review_census13_ad_math.py','tools/reviews/review_census13_ad_driver.py','tools/reviews/launch_census13_ad_review.py','tests/test_census13_ad_review.py','docs/plans/census13-independent-ad.md','docs/plans/census13-panel-inputs.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def inventory(root,pin):
    need(entry(root/'manifest.json')['sha256']==pin,'literal source manifest');v=read(root/'manifest.json')['files'];need(set(v)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'entire immutable inventory')
    for n,info in v.items():need(entry(root/n)==info,'immutable source file '+n)
def forecast(times,sizes):
    need(len(times)==len(sizes)==3 and all(x>0 for x in times)and all(x>0 for x in sizes),'three AD fullphase profiles')
    return dict(seconds=2*1050*max(times)+45,bytes=2*1050*max(sizes)+64*1024**2)

def main(mode,profile_outer=None,profile_worker=None):
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out=BASE/f'attempt001-{mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;calls=0;returned=0;limit=60 if mode=='profile'else 600;result=dict(passed=False,mode=mode,rows=[])
    def stop(*_):raise TimeoutError('fixed AD proof deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
    def save(name,v,closing=False):
        nonlocal used
        b=(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if used+len(b)>=(512 if closing else 504)*1024**2:raise MemoryError('512MiB raw cap with closing reserve')
        used+=len(b);p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        return len(b)
    def step(inp,state,row):
        nonlocal calls,returned
        if time.monotonic()-start>=limit:raise TimeoutError('AD panel deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('AD panel1GiB RSS')
        need(calls<(3 if mode=='profile'else 1050),'fixed native AD phase cap');calls+=1;v,j=objective_ad(inp,state);returned+=1
        size=save(f'row-{row}/native-{calls:05d}.json',dict(input=vector_json(state),value=vector_json(v),jacobian=matrix_json(j)))
        need(all(x.is_finite()for x in v)and all(x.is_finite()for a in j.tolist()for x in a),'finite complete AD return')
        return v,j,size
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();ident={n:entry(ROOT/n)for n in FILES}
        for n,v in ident.items():need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed AD dependencies');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
        rt=dict(commit=source,sources=ident,precision=1024,python=sys.version,flint=flint.__version__,flint_library=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),pid=os.getpid());save('runtime.json',rt)
        inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker']);oldrt=read(PARENT/'run/runtime.json');old=read(PARENT/'run/outcome.json');launch=read(PARENT/'launcher-result.json')
        need(oldrt['commit'].startswith(PINS['source'])and launch['source_commit']==oldrt['commit']and launch['exit_code']==0 and not launch['owned_live_pids']and old['all_orbit_proofs_passed']and old['native_returned']==4200,'completed fixed producer')
        for n,v in oldrt['sources'].items():need(entry(PARENT/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',oldrt['commit']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical producer source')
        card=read(ROOT/'docs/plans/census13-panel-inputs.json');indices=card['new_indices'];need([v['index']for v in old['rows']]==indices and len(indices)==26,'same complete fixed26 panel')
        data={i:read(PARENT/f'run/row-{i}/inputs.json')for i in indices};need(sum(v['period']for v in data.values())==525 and all(v['network']['bias']and v['network']['d']==2 and v['network']['width']==3 for v in data.values()),'fixed13D AD scope and1050call count');save('input-binding.json',dict(producer=str(PARENT),pins=PINS,indices=indices,map_input_hashes={str(i):entry(PARENT/f'run/row-{i}/inputs.json')for i in indices}))
        with ctx.workprec(1024):
            if mode=='profile':
                proof=read(PARENT/'run/row-139/proof.json');times=[];sizes=[]
                for k in [0,16,33]:
                    state=[A.R.original_ball(v).mid()for v in proof['orbit_enclosures'][k]];beg=time.monotonic();_,_,size=step(data[139],state,139);times.append(time.monotonic()-beg);sizes.append(size)
                cost=forecast(times,sizes);result.update(passed=cost['seconds']<600 and cost['bytes']<512*1024**2,times=times,sizes=sizes,forecast=cost)
            else:
                pr=BASE/'attempt001-profile';inventory(pr,profile_outer);inventory(pr/'run',profile_worker);prof=read(pr/'run/outcome.json');prt=read(pr/'run/runtime.json');pl=read(pr/'launcher-result.json')
                need(prof['passed']and pl['exit_code']==0 and not pl['owned_live_pids']and all(prt[k]==rt[k]for k in ['sources','precision','python','flint','flint_library','threads']),'same measured AD source/runtime')
                need(read(pr/'run/input-binding.json')==read(out/'input-binding.json'),'same profile target input binding');cost=forecast(prof['times'],prof['sizes']);need(cost==prof['forecast']and cost['seconds']<600 and cost['bytes']<512*1024**2,'measured admission');save('profile-binding.json',dict(root=str(pr),outer=profile_outer,worker=profile_worker,forecast=cost))
                for i in indices:
                    inp=data[i];proposal=read(PARENT/f'run/row-{i}/proof.json');save(f'row-{i}/inputs.json',inp);beg=time.monotonic();proof=A.prove(inp,proposal,lambda state:step(inp,state,i)[:2]);save(f'row-{i}/proof.json',proof);result['rows'].append(dict(index=i,period=inp['period'],passed=proof['passed'],gates=proof['gates'],contraction=proof['contraction'],attraction_power=proof['attraction_power'],seconds=time.monotonic()-beg));gc.collect()
                result.update(passed=all(v['passed']for v in result['rows'])and len(result['rows'])==26,proofs_passed=sum(v['passed']for v in result['rows']),forecast=cost)
    except BaseException as exc:
        import traceback;result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_attempted=calls,native_returned=returned);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()}),True)
    print(json.dumps({k:v for k,v in result.items()if k!='rows'}));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');a=p.parse_args();raise SystemExit(main(a.mode,a.profile_outer,a.profile_worker))
