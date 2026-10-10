"""One bounded exact-input RNN replay, nested proof, and independent native check."""
import hashlib,json,math,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from neural_shrimp.rigorous_clark_rnn import check_rnn_orbit
from neural_shrimp.rigorous_orbits import canonical_bytes
import review_clark_rnn_c1_math as R
RAW=Path('/data2/neural-shrimp-data/explore/certificates/clark-rnn-p5-2026-10-09/attempt-001')
OUTER='a3119c6c6df5481723695b97135760494bb47b84c032fc14b8d3e22a0ca7fb8a';RUN='065e3ce5ef74ac958fe05dddff314aa40731df37edad1195f87bf45ada0b7ccb'
RECEIPT='certificates/clark-rnn-p5-receipt.json';RECEIPT_HASH='6e3ebe75108e46acbd452b1a456bc699b7cd6ea07c57b7ddee399b6769956b29'
EXEC='bbed3bfbc6e3caf331cb78aa4ded7f4fe6e45527';INPUT='9dca2ba53b93563842c95b43b92581705ba9b0f83c2259a0185bd3e8261c2ce2'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/reviews/review_clark_rnn_c1.py','tools/reviews/review_clark_rnn_c1_math.py','tools/reviews/launch_clark_rnn_c1_review.py','tests/reviews/test_clark_rnn_c1_independent.py','docs/plans/clark-rnn-c1-independent-review.md',RECEIPT,'src/neural_shrimp/rigorous_clark_rnn.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py']
def entry(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_bytes())
def inventory(root,pin):
    R.need(entry(root/'manifest.json')['sha256']==pin,'manifest identity');m=read(root/'manifest.json')['files'];paths={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};R.need(set(paths)==set(m),'complete inventory')
    for n,p in paths.items():R.need(not p.is_symlink()and not(p.stat().st_mode&0o222)and entry(p)==m[n],'immutable hash '+n)
    return {'manifest_sha256':pin,'files':len(paths),'bytes':sum(x['bytes']for x in m.values())}
def native_inputs(origin):
    root=Path(origin['raw']);inventory(root,origin['raw_manifest_sha256']);desc=Path(origin['seed_descriptor']);R.need(entry(desc)['sha256']==origin['seed_descriptor_sha256'],'seed descriptor');d=read(desc)
    a=desc.with_name(d['array_file']);ref=Path(origin['matrix_input']);R.need(entry(a)['sha256']==d['array_sha256']==origin['seed_arrays_sha256']and entry(ref)['sha256']==origin['matrix_input_sha256'],'native arrays')
    R.need(d['tag']=='g16-seed'and d['gain']==10.02 and d['delta']==.956,'fixed selected body seed')
    R.need(read(root/'run/input-binding.json')['source_input_sha256']==entry(ref)['sha256'],'original matrix identity')
    with np.load(ref)as f:j=f['j']
    with np.load(a)as f:x=f['theta']
    R.need(j.shape==(4,4)and x.shape==(4,),'four-dimensional native arrays')
    def exact(v):return str(Fraction.from_float(float(v)))
    return {'schema':'clark-rnn-c1-input-v1','j0':[[exact(v)for v in row]for row in j],'gain':exact(d['gain']),'delta':exact(d['delta']),'period':5,'approximation':[exact(v)for v in x]}
def main(out):
    R.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');R.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'passed':False};stopped=[False]
    def save(n,v):
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True);b=canonical_bytes(v);R.need(sum(q.stat().st_size for q in out.rglob('*')if q.is_file())+len(b)<8*1024**2,'8MiB cap')
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():R.need(not stopped[0]and time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'time/RSS gate')
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,30)
    save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in FILES},'pid':os.getpid(),'python':sys.version,'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)})
    for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        R.need(entry(ROOT/RECEIPT)['sha256']==RECEIPT_HASH,'canonical receipt pin');receipt=read(ROOT/RECEIPT);result['inventory']=inventory(RAW,OUTER);inventory(RAW/'run',RUN)
        run=RAW/'run';rt=read(run/'runtime-start.json');launch=read(RAW/'launch-end.json');status=read(run/'result.json');original=read(run/'proof.json');inp=read(run/'inputs.json')
        R.need(rt==receipt['runtime']and original==receipt['proof']and inp==receipt['inputs']and launch==receipt['launcher']and status==receipt['result'],'canonical evidence equality')
        R.need(rt['source']['commit']==launch['source_commit']==EXEC and rt['mode']=='target','producer execution')
        for n,h in rt['source']['sha256'].items():R.need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==h,'source '+n)
        for n in FILES:
            if n in rt['source']['sha256']:R.need(entry(ROOT/n)['sha256']==rt['source']['sha256'][n],'unchanged replay source '+n)
        R.need(rt['python']==sys.version and rt['numpy']==np.__version__ and rt['python_flint']==flint.__version__ and rt['platform']==platform.platform(),'same replay library environment')
        R.need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10 and launch['exit_code']==0 and launch['owned_group_empty']and status['status']=='completed','producer completed')
        for pid in (rt['pid'],launch['launcher_pid'],launch['child_pid']):R.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
        for v,b in ((status['elapsed_seconds'],30),(status['peak_rss_kib'],448*1024),(launch['elapsed_seconds'],50)):R.need(type(v)in(int,float)and math.isfinite(v)and 0<=v<b,'bounded producer resources')
        pb=read(run/'profile-binding.json');pr=Path(pb['path']);inventory(pr,pb['manifest_sha256']);prt=read(pr/'runtime-start.json');ps=read(pr/'result.json');R.need(prt['source']['sha256']==rt['source']['sha256']and prt['settings']==rt['settings']and ps['resource_pass']and ps['projected_seconds']==pb['projection']<=30,'profile gate')
        R.need(prt['mode']=='profile'and ps['status']=='completed'and prt['nice']>=10 and all(prt[k]==rt[k]for k in ('python','numpy','python_flint','platform','threads')),'profile runtime equality')
        times=ps['unit_seconds'];R.need(len(times)==2 and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in times+[ps['elapsed_seconds'],ps['peak_rss_kib']])and 8*max(times)+5==ps['projected_seconds']and ps['elapsed_seconds']<30 and ps['peak_rss_kib']<448*1024,'finite measured profile gate')
        R.need(entry(run/'inputs.json')['sha256']==INPUT and native_inputs(read(run/'origin.json'))==inp,'independent exact seed/model reconstruction');save('input-binding.json',{'input_sha256':INPUT,'receipt_sha256':RECEIPT_HASH,'original_source':rt['source'],'origin':read(run/'origin.json')});tick()
        replay=check_rnn_orbit(inp,**rt['settings']);save('replay.json',replay);tick();R.need(canonical_bytes(replay)==canonical_bytes(original),'byte-identical original replay')
        nested=check_rnn_orbit(inp,precision=384,radius_exponent=-100,newton_steps=5,max_power=512);save('nested-proof.json',nested);tick();R.need(nested['passed']and nested['odd_companion_distinct']and R.strict_nested(original['box'],nested['box']),'stronger nested proof in original box')
        independent=R.independent_proof(inp,nested,384);save('independent-exp-proof.json',independent);tick();R.need(independent['passed'],'independent exponential-form proof')
        seed=list(map(Fraction,inp['approximation']));scale=max(Fraction(1),*map(abs,seed));near=max(max(abs(a-x),abs(b-x))/scale for x,(a,b)in zip(seed,map(R.endpoints,original['box'])));R.need(str(near)==receipt['whole_box_proximity']['exact_fraction']and near<Fraction(1,10**7),'whole original-box proximity')
        result.update(passed=True,byte_replay=True,nested_box_strict=True,independent_exp_proof=True,primitive_period=5,dimension=4,odd_companion_distinct=True,whole_box_proximity=str(near),scope='one directly enclosed activity cycle and distinct odd companion; not a training certificate or window theorem')
    except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        out.chmod(0o555)
    print(json.dumps(result),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
