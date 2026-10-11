"""Fixed complete13D census representative panel; unchanged C1 proof kernel."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import ctx
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_c1 as C
import certify_census_long_period as B
from neural_shrimp import rigorous_orbits as K
SETTINGS=dict(precision=768,radius_exponent=-320,newton_steps=6,max_power=4096)
BASE=Path('/data2/neural-shrimp-data/explore/certificates/census13-panel')
CARD='docs/plans/census13-panel-inputs.json'
SOURCES=list(dict.fromkeys(B.SOURCES+['tools/certify_census13_panel.py','tools/launch_census13_panel.py','tests/test_census13_panel.py',CARD,'docs/plans/census13-panel.md','certificates/c1-row35-receipt.json','certificates/s2-period18-receipt.json']))
CAP=1024*1024**2

def need(v,s):
    if not v:raise ValueError(s)
def selected(summary,audit,rows):
    keys={(v['map_id'],v['component'])for v in summary['gate']['qualifying']}&{(v['map_id'],v['component'])for v in audit['audit_gate']['qualifying']}
    eligible=[i for i,v in enumerate(rows)if(v['map_id'],v['component'])in keys]
    need(len(keys)==len(eligible)==61,'all61 unique agreed representatives')
    return [i for i in eligible if len(rows[i]['certificate']['orbit'][0])==13]
def forecast(times,sizes):
    need(len(times)==len(sizes)==3 and all(np.isfinite(v)and v>0 for v in times)and all(type(v)is int and v>0 for v in sizes),'three measured full native returns')
    return dict(seconds=2*4200*max(times)+30,bytes=2*4200*max(sizes)+64*1024**2)
def proximity(box,reference):
    need(len(box)==len(reference)==13,'full13 representative proximity');scale=max(F(1),*(abs(F(x))for x in reference));distance=F(0)
    for row,x in zip(box,reference):
        need(row.get('finite',True),'finite box');m=F(row['mid'][0])*F(2)**row['mid'][1];radius=F(row['rad'][0])*F(2)**row['rad'][1];need(radius>=0,'nonnegative radius');distance=max(distance,(abs(m-F(x))+radius)/scale)
    need(distance<=F(1,10**7),'whole proof box close to selected representative');return str(distance)

def main(mode,profile_outer=None,profile_worker=None):
    need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    out=BASE/f'attempt003-{mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;calls=0;returned=0;current=None
    limit=60 if mode=='profile'else 600
    def stop(*_):raise TimeoutError('fixed census13 deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
    def save(name,value,closing=False):
        nonlocal used
        data=K.canonical_bytes(value)
        if used+len(data)>=CAP-(0 if closing else 8*1024**2):raise MemoryError('1GiB panel raw cap, reserved8MiB closing inventory')
        used+=len(data);p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        return len(data)
    def check():
        if time.monotonic()-start>=limit:raise TimeoutError('worker deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1024**2:raise MemoryError('worker1GiB RSS')
    result=dict(passed=False,mode=mode,rows=[],native_cap=3 if mode=='profile'else 4200)
    Original=K.BallTanhMap
    try:
        source={n:B.entry(ROOT/n)for n in SOURCES}
        for n,v in source.items():need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed source');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
        runtime=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=source,settings=SETTINGS,python=sys.version,numpy=np.__version__,flint=flint.__version__,flint_library=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0));save('runtime.json',runtime)
        card=B.load(ROOT/CARD);pins=B.load(ROOT/'docs/plans/census-long-period-inputs.json')['pins'];paths={k:B.located(v['path'])for k,v in pins.items()}
        for k,p in paths.items():need(B.entry(p)['sha256']==pins[k]['sha256'],'exact census input '+k)
        rows=B.load(paths['certificates']);indices=selected(B.load(paths['summary']),B.load(paths['audit']),rows)
        need(indices==card['all13_indices']and len(indices)==28 and card['new_indices']==[i for i in indices if i not in [35,111]],'complete fixed13D panel')
        prepared={i:C.prepare(paths['certificates'],paths['plan'],i,card['selection'])for i in indices}
        reused=[]
        for i,receiptpath in [(35,'certificates/c1-row35-receipt.json'),(111,'certificates/s2-period18-receipt.json')]:
            rec=B.load(ROOT/receiptpath);oldroot=Path(rec['external_root']);prior=B.load(oldroot/'inputs.json')
            need(rec['passed']and rec['statement']['attraction']and rec['statement']['existence_and_local_uniqueness']and rec['statement']['primitive_period_excluded_divisors'],'reuse existing attracting primitive proof')
            for n,v in rec['files'].items():need(B.entry(oldroot/n)==v,'existing proof receipt inventory')
            for k in ['schema','network','x','y','lr_hidden','lr_output','period']:need(prior[k]==prepared[i][k],'exact existing representative map '+k)
            oldproof=B.load(oldroot/'certificate.json');need(oldproof['result']['passed'],'reused raw proof passed');distance=proximity(oldproof['result']['box'],prepared[i]['approximation']);reused.append(dict(index=i,whole_box_proximity=distance,receipt=B.entry(ROOT/receiptpath),proof=B.entry(oldroot/'certificate.json'),root=str(oldroot)))
        need(sum(prepared[i]['period']for i in card['new_indices'])==525 and max(card['new_indices'],key=lambda i:prepared[i]['period'])==139,'fixed4200phase count and largest-period profile')
        save('input-binding.json',dict(pins=pins,all13=indices,new=card['new_indices'],already_proved=reused,scope='Complete13D representative orbit subset only; no interval window census or topology.'))
        if mode=='profile':
            inp=prepared[139];save('profile-inputs.json',inp);orbit=np.asarray(rows[139]['certificate']['orbit']);times=[];sizes=[]
            with ctx.workprec(768):
                model=Original(inp)
                for phase in [0,16,33]:
                    check();state=[K.exact(K.dyadic(v))for v in orbit[phase]];beg=time.monotonic();calls+=1;end,J=model.step(state);returned+=1
                    size=save(f'profile-{phase}.json',dict(state=K.vector_json(state),end=K.vector_json(end),jacobian=K.matrix_json(J)));times.append(time.monotonic()-beg);sizes.append(size)
                    need(all(v.is_finite()for v in end)and all(J[i,j].is_finite()for i in range(13)for j in range(13)),'finite full profile')
            estimate=forecast(times,sizes);result.update(passed=estimate['seconds']<600 and estimate['bytes']<CAP,times=times,sizes=sizes,forecast=estimate)
        else:
            profile=BASE/'attempt003-profile/run';need(B.entry(profile.parent/'manifest.json')['sha256']==profile_outer and B.entry(profile/'manifest.json')['sha256']==profile_worker,'literal committed profile manifest pins');outer=B.load(profile.parent/'launcher-result.json');need(outer['exit_code']==0 and outer['owned_live_pids']==[],'successful terminated profile');need(B.load(profile/'input-binding.json')==B.load(out/'input-binding.json'),'identical profile input binding');m=B.load(profile/'manifest.json')['files'];need(set(m)=={str(p.relative_to(profile))for p in profile.rglob('*')if p.is_file()and p!=profile/'manifest.json'}and all(B.entry(profile/n)==v for n,v in m.items()),'complete measured profile inventory')
            pr=B.load(profile/'outcome.json');rt=B.load(profile/'runtime.json');need(pr['passed']and rt['sources']==source and all(rt[k]==runtime[k]for k in ['settings','python','numpy','flint','flint_library','threads']),'same measured source/library')
            estimate=forecast(pr['times'],pr['sizes']);need(estimate==pr['forecast']and estimate['seconds']<600 and estimate['bytes']<CAP,'recomputed fixed fullpanel resource gate');save('profile-binding.json',dict(root=str(profile),manifest=B.entry(profile/'manifest.json'),forecast=estimate))
            class RetainingMap(Original):
                def step(self,state):
                    nonlocal calls,returned
                    check();need(calls<4200,'fixed fullpanel phase cap');calls+=1;value,J=super().step(state);returned+=1
                    save(f'row-{current}/native-{calls:05d}.json',dict(input=K.vector_json(state),value=K.vector_json(value),jacobian=K.matrix_json(J)))
                    return value,J
            K.BallTanhMap=RetainingMap
            for current in card['new_indices']:
                check();inp=prepared[current];save(f'row-{current}/inputs.json',inp);before=calls;beg=time.monotonic();row=dict(index=current,period=inp['period'],passed=False)
                try:
                    proof=K.check_orbit(inp,**SETTINGS);save(f'row-{current}/proof.json',proof);distance=proximity(proof['box'],inp['approximation']);row.update(passed=proof['passed'],statement=proof['exact_statement'],contraction=proof['contraction_norm_upper'],attraction_power=proof['attraction_witness_power'],attraction_bound=proof['power_bounds'][-1]['norm_inf_upper'],proximity=distance)
                except (TimeoutError,MemoryError):raise
                except Exception as e:row.update(error=type(e).__name__+': '+str(e))
                finally:
                    row.update(native_calls=calls-before,seconds=time.monotonic()-beg);save(f'row-{current}/outcome.json',row);result['rows'].append(row)
            need(len(result['rows'])==26,'all fixed representatives attempted');result.update(passed=True,all_orbit_proofs_passed=all(x['passed']for x in result['rows']),proofs_passed=sum(x['passed']for x in result['rows']),forecast=estimate)
    except BaseException as e:
        import traceback;result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        K.BallTanhMap=Original;signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_attempted=calls,native_returned=returned,current_index=current);save('outcome.json',result,closing=True);save('manifest.json',dict(files={str(p.relative_to(out)):B.entry(p)for p in out.rglob('*')if p.is_file()}),closing=True)
    print(json.dumps({k:v for k,v in result.items()if k!='rows'}));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');a=p.parse_args();raise SystemExit(main(a.mode,a.profile_outer,a.profile_worker))
