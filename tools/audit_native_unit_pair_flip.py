"""Bounded independent saved55 arithmetic and fresh480-bit same-root replay."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for n in THREADS:os.environ[n]='1'
import hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
import flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp import junction_audit as A
from neural_shrimp.rigorous_unit_pair_flip import augmented,root_proof
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import ball_json,matrix_json
import certify_native_unit_pair_flip as B
PARENT=B.BASE/'attempt002-target'
OUT=Path('/data2/neural-shrimp-data/audits/native-unit-pair-flip-independent/attempt002/run')
PIN='9c7c871fd974d752bef6139a8c0fa2b49496f495eaf54fc22ba37cf61c3cc6c3'
WORKER='24ff31605b67a2c990fef915866396ce1fb59ae3aedd31f8f327399712ccbe17'
EXECUTION='714813fa28110cbeb5fac57833c833ea19a3ed0e'
OWN=['src/neural_shrimp/junction_audit.py','tools/audit_native_unit_pair_flip.py','tools/launch_native_unit_pair_flip_audit.py','tests/test_junction_audit.py','docs/plans/native-unit-pair-flip-independent.md']

def main():
    A.need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;calls=0
    def stop(*_):raise TimeoutError('60s local junction audit')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    def save(name,value):
        nonlocal used
        data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();A.need(used+len(data)<32*1024**2,'32MiB raw cap');used+=len(data);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    def tick():
        nonlocal calls
        calls+=1;A.need(calls<=162 and time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'fixed162phases/60s/1GiB')
    def summary(root,proposal,label):
        a=A.root_arithmetic(root);save(label+'-root-arithmetic.json',dict(inclusions=a['inclusions'],contraction=ball_json(a['contraction']),proper_divisors=a['divisors']));b=A.spectrum_crossing(root,proposal)
        return dict(inclusions=a['inclusions'],contraction=ball_json(a['contraction']),proper_divisors=a['divisors'],complement_bounds=[ball_json(v)for v in b['complement_bounds']],crossing=ball_json(b['crossing']))
    result=dict(passed=False)
    try:
        B.inventory(PARENT,PIN);B.inventory(PARENT/'run',WORKER)
        runtime=B.read(PARENT/'run/runtime.json');launch=B.read(PARENT/'launcher-result.json');outcome=B.read(PARENT/'run/outcome.json')
        A.need(runtime['commit']==launch['commit']==EXECUTION and outcome['passed']and outcome['native_calls']==324 and launch['exit_code']==0 and launch['owned_live_pids']==[],'exact completed producer')
        A.need(runtime['settings']==dict(precision=384,period=18,newton_steps=4,radius_exponent=-80),'fixed original settings')
        sources=list(dict.fromkeys(list(runtime['sources'])+OWN));identities={n:B.entry(ROOT/n)for n in sources}
        for n,record in runtime['sources'].items():
            A.need(B.entry(PARENT/'run/source'/n)==identities[n]==record and hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+n],cwd=ROOT)).hexdigest()==record['sha256'],'historical/current source binding')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for n,record in identities.items():
            A.need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==record['sha256'],'committed audit dependencies');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
        save('runtime.json',dict(commit=commit,sources=identities,pid=os.getpid(),precision=480,python=sys.version,python_flint=flint.__version__,flint=str(getattr(flint,'__FLINT_VERSION__','unavailable')),threads=dict.fromkeys(THREADS,'1')))
        with ctx.workprec(480):
            model,_,binding=B.bind_candidate();A.need(binding==B.read(PARENT/'run/inputs.json'),'exact map/candidate/bracket chain')
            save('binding.json',dict(root=str(PARENT),outer=PIN,worker=WORKER,source=EXECUTION,map_inputs=binding['map_inputs']))
            old=B.read(PARENT/'run/root.json');proposal=B.read(PARENT/'run/same-root-complement.json')
            A.need(old['dimension']==13 and old['period']==18 and old['settings']==dict(newton_steps=4,radius_exponent=-80),'literal original root')
            saved=summary(old,proposal,'saved');save('saved-arithmetic.json',saved)
            profile_root=B.BASE/'attempt002-profile';pb=B.read(PARENT/'run/profile-binding.json')
            B.inventory(profile_root,pb['outer']);B.inventory(profile_root/'run',pb['worker'])
            fixture=B.read(profile_root/'run/synthetic-inputs.json');pm=TwoRateCuspMap(fixture['map_inputs']);seed=[A.decode(v).mid()for v in fixture['seed']]
            begin=time.monotonic();pv,pj,_=augmented(pm,seed,18,tick);elapsed=time.monotonic()-begin
            A.need(all(pv[i,0].is_finite()for i in range(55))and all(pj[i,j].is_finite()for i in range(55)for j in range(55)),'finite fresh480 synthetic profile')
            forecast=4*elapsed+15;save('profile.json',dict(seconds=elapsed,forecast=forecast,value=matrix_json(pv),jacobian=matrix_json(pj)))
            A.need(forecast<60-(time.monotonic()-start),'measured remaining budget')
            def observe(kind,value):save('fresh-'+kind+'.json',value)
            center=[A.decode(v)for v in old['center']]
            fresh,_=root_proof(model,center,18,radius_exponent=-84,newton_steps=0,tick=tick,observe=observe)
            save('fresh-root.json',fresh);A.need(fresh['passed'],'fresh480 strict native root')
            replay=summary(fresh,proposal,'fresh');save('fresh-arithmetic.json',replay)
            same=A.strict_nested(old['box'],fresh['box']);save('root-identity.json',dict(fresh_inside_original=same,original_box=old['box'],fresh_box=fresh['box']))
        result.update(passed=True,saved=saved,fresh=replay,same_unique_root=True,coordinates=55,scope='Independent saved interval root/complement/crossing arithmetic plus fresh480-bit zero-Newton shared-kernel native proof in a strictly nested box. Local unit-pair/simple-flip junction; no generic normal form or global connection.')
    except BaseException as exc:
        import traceback;result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,native_calls=calls,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=result['passed']and result['seconds']<60 and result['peak_rss_kib']<1024**2;save('outcome.json',result);save('manifest.json',dict(files={str(p.relative_to(OUT)):B.entry(p)for p in OUT.rglob('*')if p.is_file()}))
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
