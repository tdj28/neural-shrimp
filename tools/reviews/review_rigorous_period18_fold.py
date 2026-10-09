"""One source-bound post-outcome replay and nested check of the selected fold.

No search, orbit continuation, eigensolver or extra target is permitted.
The original failed radius attempt remains a distinct immutable input.
"""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,hashlib,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_folds import check_fold
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,eye,norm_inf_upper
BASE=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09')
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/c1-2026-10-07/attempt-003/inputs.json')
NUMERIC=Path('/data2/neural-shrimp-data/explore/s2-period18-fold/attempt-001')
PINS={PARENT:'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',NUMERIC/'manifest.json':'3a44977630bd3d9f329bad8e6369c4d024fc4a379ee6ebe593c85d9b913cb620',NUMERIC/'outcome.json':'ddd90afa59622d1bc3da59bfc1eb424afcb16570abbcf8dee8f3eefeadd4ec13'}
KERNEL_SHA='e895adb9d0508a95efc85a2b5c13ce32404e5b58e7494b1c55c0105e8e492955'
PRODUCER_FILES=['src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/certify_s2_period18_fold.py','tests/test_rigorous_folds.py','docs/plans/s2-period18-fold-certificate.md']
OWN_FILES=PRODUCER_FILES[:5]+['tools/reviews/review_rigorous_period18_fold.py','tests/reviews/test_rigorous_period18_fold_review.py','docs/plans/rigorous-period18-fold-independent-review.md','docs/reviews/s2-period18-fold-kernel-source-review.json']
CAP=16*1024**2
INTERNAL=120
EXTERNAL=135

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def entry(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def now():return datetime.now(timezone.utc).isoformat()
def exact_pair(pair):
    require(isinstance(pair,list) and len(pair)==2 and all(type(x)is int for x in pair),'dyadic pair')
    return Fraction(pair[0])*Fraction(2)**pair[1]
def interval(value):
    require(isinstance(value,dict) and 'mid'in value and 'rad'in value,'finite serialized ball')
    mid,rad=exact_pair(value['mid']),exact_pair(value['rad']);require(rad>=0,'nonnegative radius')
    return mid-rad,mid+rad
def stored(value):
    interval(value)
    return arb(arb(tuple(value['mid'])),arb(tuple(value['rad'])))
def matrix(value):return arb_mat([[stored(v) for v in row] for row in value])
def exact_norm(value):
    return max(sum(max(abs(lo),abs(hi)) for lo,hi in map(interval,row)) for row in value)
def nested(inner,outer):
    require(len(inner)==len(outer),'nested dimension')
    return all(interval(a)[0]>interval(b)[0] and interval(a)[1]<interval(b)[1] for a,b in zip(inner,outer))
def inventory(root):return {str(p.relative_to(root)):entry(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
def verify(root,pin,nested_manifest=False):
    require(root.is_dir() and not root.is_symlink(),'safe input root')
    require(sha(root/'manifest.json')==pin,'externally pinned manifest')
    manifest=read(root/'manifest.json');files=manifest['files'] if nested_manifest else manifest
    require(files and isinstance(files,dict),'nonempty manifest')
    require(all(not p.is_symlink() for p in root.rglob('*')),'no artifact symlinks')
    require(inventory(root)==files,'exact complete file inventory')
    for name in files:require(not Path(name).is_absolute() and '..' not in Path(name).parts,'safe file path')
    if nested_manifest:require(manifest['total_bytes']==sum(v['bytes'] for v in files.values()) and manifest['within_limit'] is True,'manifest totals')
    return manifest

def source_snapshot():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean review checkout')
    out={}
    for p in OWN_FILES:
        b=(ROOT/p).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed reviewer source')
        out[p]=hashlib.sha256(b).hexdigest()
    require(out[PRODUCER_FILES[0]]==KERNEL_SHA,'reviewed unchanged fold kernel')
    return out

def binding(name,pin):
    root=BASE/name
    verify(root,pin['manifest_sha256'],True)
    verify(root/'run',pin['run_manifest_sha256'],True)
    runtime=read(root/'run/runtime-start.json');launch=read(root/'launcher-result.json');outcome=read(root/'run/outcome.json')
    require(runtime['source_commit']==launch['source_commit']==pin['execution_commit'],'execution source binding')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',pin['execution_commit']+':'+p],cwd=ROOT)).hexdigest() for p in PRODUCER_FILES}
    require(runtime['source_sha256']==launch['source_sha256']==expected,'full producer source binding')
    require(expected[PRODUCER_FILES[0]]==KERNEL_SHA,'same reviewed kernel')
    for p,h in expected.items():require(sha(root/'run/source'/p)==h,'retained source bytes')
    require(runtime['python_flint']==flint.__version__ and runtime['flint']==flint.__FLINT_VERSION__,'matching arithmetic library')
    require(runtime['threads']=={k:'1' for k in THREADS} and runtime['nice']>=10,'CPU/nice contract')
    require(launch['owned_leader_exited'] is True and launch['owned_live_pids_remaining']==[],'producer ownership cleanup')
    for pid in (runtime['pid'],launch['owned_group_pid']):require(not Path('/proc',str(pid)).exists(),'recorded producer PID absent')
    for t,limit in ((outcome['elapsed_seconds'],120),(launch['elapsed_seconds'],142)):
        require(type(t)in(int,float) and math.isfinite(t) and 0<=t<=limit,'recorded elapsed cap')
    require(outcome['ticks']<=160 and outcome['max_rss_kib']<=2*1024**2,'producer tick/RSS cap')
    require(sum(e['bytes'] for e in inventory(root/'run').values())<CAP,'producer raw cap')
    require(sum(e['bytes'] for e in inventory(root).values())<CAP+1024**2,'producer total cap')
    return root,runtime,launch,outcome

def reconstruct(inp):
    for p,h in PINS.items():require(sha(p)==h,'consumed input pin')
    verify(NUMERIC,PINS[NUMERIC/'manifest.json'])
    old=read(PARENT);num=read(NUMERIC/'outcome.json');native=read(NUMERIC/'inputs.json')
    require(num['accepted'] and (num['target'],num['period'],num['ray_index'])==('S2-28011-zoom:587',18,0),'fixed numerical seed')
    require(native['network']==old['network'] and native['row_index']==111 and native['ray_index']==0,'numeric model/selection binding')
    def rat(x):return str(Fraction(float(x)))
    eta=10.0**float(num['parameter_log10'][0]);output=float.fromhex('0x1.a461eed55764cp-1')
    require((10.0**float(num['parameter_log10'][1])).hex()==output.hex(),'fixed output identity')
    expected_model={k:old[k] for k in ('schema','network','x','y')};expected_model.update(lr_hidden=rat(eta),lr_output=rat(output))
    actual=inp['map_inputs'];require({k:actual[k] for k in ('schema','network','x','y')}=={k:expected_model[k] for k in ('schema','network','x','y')},'same consumed exact map/data')
    for key in ('lr_hidden','lr_output'):require(Fraction(actual[key])==Fraction(expected_model[key]),'exact rate '+key)
    proposal=list(num['diagnostics']['state'])+[eta]+list(num['diagnostics']['right_null'])
    require(inp['schema']=='neural-shrimp-fold-input-v1' and inp['period']==18 and len(proposal)==27,'full selected augmented input')
    require([Fraction(x) for x in inp['approximation']]==[Fraction(float(x)) for x in proposal],'exact selected proposal')
    require(inp['provenance']['input_pins']=={str(k):v for k,v in PINS.items()},'serialized provenance pins')
    require(inp['provenance']['output_rate_matches_row111'] is True and inp['provenance']['output_rate_hex']==output.hex(),'serialized fixed output assertion')
    return {'input_files':{str(p):entry(p) for p in PINS},'exact_output_rate':str(Fraction(output)),'period':18,'dimension':13,'selection':'row111, S2-28011-zoom:587, ray0; consumed development only'}

def saved_math(result):
    require(result['passed'] and all(result['gates'].values()),'all full-fold gates')
    require([d['divisor'] for d in result['divisors']]==[1,2,3,6,9],'exact proper divisors')
    require(len(result['box'])==27 and len(result['monodromy'])==13,'full state dimensions')
    require(nested(result['krawczyk_image'],result['box']),'exact Fraction strict Krawczyk inclusion')
    require(exact_norm(result['defect'])<1 and exact_norm(result['left_defect'])<1,'exact saved defect bounds')
    with ctx.workprec(384):
        box=[stored(b) for b in result['box']];center=[stored(b) for b in result['center']]
        n=13;v=box[n+1:];jac=matrix(result['jacobian_box']);pre=matrix(result['preconditioner'])
        require(all(interval(x)[0]==interval(x)[1] for row in result['preconditioner'] for x in row),'point preconditioner')
        defect=eye(27)-pre*jac;contract=norm_inf_upper(defect);require(contract<1,'independent augmented contraction')
        image=arb_mat([[center[i]] for i in range(27)])-pre*arb_mat([[stored(x)] for x in result['g_center']])+defect*arb_mat([[box[i]-center[i]] for i in range(27)])
        require(all(box[i].contains_interior(image[i,0]) for i in range(27)),'independent Krawczyk image')
        require(box[13]>0,'positive actual hidden rate')
        orbit=[[stored(x) for x in row] for row in result['orbit_enclosures']]
        require(len(orbit)==19 and all(len(row)==13 for row in orbit),'full eighteen-step orbit')
        for d in (1,2,3,6,9):require(any(not (orbit[d][i]-box[i]).contains(0) for i in range(13)),'independent divisor exclusion')
        m=matrix(result['monodromy']);a=m-eye(n);w=[stored(x) for x in result['left_vector']]
        require(all(q.contains(0) for q in [sum((a[i,j]*v[j] for j in range(n)),arb(0)) for i in range(n)]),'right null compatibility')
        require(sum((x*x for x in v),arb(0)).contains(1) and sum((x*y for x,y in zip(w,v)),arb(0)).contains(1),'unit and biorthogonal normalization')
        border=arb_mat([[a[j,i] for j in range(n)]+[v[i]] for i in range(n)]+[v+[arb(0)]])
        bd=eye(14)-matrix(result['left_preconditioner'])*border;bc=norm_inf_upper(bd);require(bc<1,'independent border invertibility')
        alpha=stored(result['left_border_multiplier']);left=arb_mat([[x] for x in w+[alpha]])
        resid=border*left-arb_mat([[arb(0)]]*n+[[arb(1)]])
        require(all(resid[i,0].contains(0) for i in range(14)) and alpha.contains(0),'border solution compatibility')
        av=sum((w[i]*stored(result['parameter_derivative'][i]) for i in range(n)),arb(0))
        cv=sum((w[i]*stored(result['second_directional_residual'][i]) for i in range(n)),arb(0))
        require(not av.contains(0) and not cv.contains(0),'independent a/c nonzero')
        require(av.overlaps(stored(result['a'])) and cv.overlaps(stored(result['c'])),'recorded coefficient consistency')
        require((cv/2).overlaps(stored(result['quadratic_normal_form_coefficient_c_over_2'])) and (-cv/av).overlaps(stored(result['hidden_rate_second_derivative'])),'normal-form conventions')
        q=m-arb_mat([[v[i]*w[j] for j in range(n)] for i in range(n)])
        witness=result['complement_witness_power'];require(type(witness)is int and 1<=witness<=4096 and witness&(witness-1)==0,'fixed power witness')
        power=q;k=1
        while k<witness:power=power*power;k*=2
        bound=norm_inf_upper(power);require(bound<1 and exact_norm(result['complement_witness'])<1,'independent complement contraction')
        return {'strict_inclusion':True,'augmented_contraction':ball_json(contract),'left_border_contraction':ball_json(bc),'a':ball_json(av),'c':ball_json(cv),'c_over_2':ball_json(cv/2),'eta_second':ball_json(-cv/av),'complement_power':witness,'complement_bound':ball_json(bound),'proper_divisors':[1,2,3,6,9],'scope':'Saved-ball arithmetic cross-check; the mixed-jet analytic implementation and Arb remain shared primitives.'}

def execute(pins_path,pins_sha,out):
    sources=source_snapshot();require(sha(pins_path)==pins_sha,'external pins JSON hash');pins=read(pins_path)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice>=10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();serial=0;ticks=0;stopped=False;result={'status':'incomplete','stages':[]}
    def save(name,value):
        b=canonical_bytes(value);require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<CAP,'aggregate audit cap')
        with (out/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    def stage(name,value):save(name+'.json',value);result['stages'].append(name)
    def stop(*_):
        nonlocal stopped
        stopped=True
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>320 or stopped or time.monotonic()-start>INTERNAL:raise TimeoutError('review tick/wall cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS above2GiB')
    def observe(kind,value):
        nonlocal serial
        save(f'evaluation-{serial:03d}-{kind}.json',value);serial+=1;tick()
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':sources,'pins_sha256':pins_sha,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'internal_seconds':INTERNAL,'tick_cap':320,'byte_cap':CAP,'cpu_only':True}
    save('runtime-start.json',runtime);save('pins.json',pins)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        require(set(pins)=={'attempt-001-profile','attempt-001-target','attempt-002-target'},'fixed three-attempt inventory')
        bound={}
        for name,pin in pins.items():
            root,runtime0,launch,outcome=binding(name,pin)
            bound[name]={'path':str(root),'pins':pin,'outcome':outcome,'launcher_exit':launch['exit_code'],'source_sha256':runtime0['source_sha256']}
        stage('01-attempt-bindings',bound);tick()
        profile=bound['attempt-001-profile']['outcome'];require(profile['passed'] and profile['scientific_claim'] is False,'completed resource-only profile')
        failed=read(BASE/'attempt-001-target/run/certificate.json')['result'];require(not failed['passed'],'first radius remains failed')
        stage('02-preserved-first-attempt',{'settings':failed['settings'],'status':failed['status'],'gates':failed['gates'],'power_bounds':failed['power_bounds'],'scope':'Preserved adaptive history. No replay or reinterpretation as a full certificate.'})
        target=BASE/'attempt-002-target/run';envelope=read(target/'certificate.json');inp=read(target/'inputs.json');proof=envelope['result']
        require(envelope['inputs_sha256']==sha(target/'inputs.json'),'target input certificate join')
        require(envelope['runtime']==read(target/'runtime-start.json'),'certificate runtime join')
        require(proof['settings']=={'precision':256,'radius_exponent':-100,'newton_steps':5,'max_power':4096},'fixed adapted target settings')
        stage('03-input-binding',reconstruct(inp));tick()
        replayed=check_fold(inp,**proof['settings'],tick=tick,observe=observe)
        save('04-replay-result.json',replayed);require(canonical_bytes(replayed)==canonical_bytes(proof),'byte-identical proof replay')
        require(replayed['passed'],'target full-fold pass');stage('05-replay-summary',{'passed':True,'byte_identical':True,'result_sha256':sha(out/'04-replay-result.json')})
        stage('06-saved-arithmetic',saved_math(proof));tick()
        seed=[Fraction(x) for x in inp['approximation']];den=max([Fraction(1)]+[abs(x) for x in seed]);dist=max(max(abs(a-s),abs(b-s))/den for (a,b),s in zip(map(interval,proof['box']),seed))
        require(dist<=Fraction(1,10**7),'selected whole augmented-box proximity')
        stage('07-whole-box-proximity',{'normalized_distance':str(dist),'display':float(dist),'cap':'1/10000000','scope':'Entire27D box near selected numerical proposal; not seed inclusion or a connection to the interior attracting orbit.'})
        stronger=check_fold(inp,precision=384,radius_exponent=-110,newton_steps=5,max_power=4096,tick=tick,observe=observe)
        save('08-nested-result.json',stronger);require(stronger['passed'],'fixed stronger proof')
        require(nested(stronger['box'],proof['box']),'exact Fraction strict nested proof box')
        stage('09-nested-summary',{'passed':True,'strictly_nested':True,'settings':stronger['settings'],'result_sha256':sha(out/'08-nested-result.json'),'scope':'Same analytic implementation and Arb library at increased precision and smaller box; not a separate AD proof.'})
        result.update(status='passed',scope='Independent source review, byte replay and saved arithmetic; local exact-real primitive18 fold with attracting complement. No whole shrimp geometry, finite-detuning ancestry or utility.')
    except BaseException as exc:result.update(status='failed_or_incomplete',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,evaluations_retained=serial,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=now())
        save('result.json',result);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'status':result['status']})
    print(json.dumps(result),flush=True);return 0 if result['status']=='passed' else 2

def launch(pins_path,pins_sha,out):
    sources=source_snapshot();require(sha(pins_path)==pins_sha,'external pins hash')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    def save(name,value):
        with (out/name).open('xb') as f:f.write(canonical_bytes(value));f.flush();os.fsync(f.fileno())
    command=['timeout','--signal=TERM','--kill-after=5s',str(EXTERNAL)+'s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--pins',str(pins_path),'--pins-sha256',pins_sha,'--out',str(out/'run')]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':sources,'pins_sha256':pins_sha,'argv':command,'launcher_pid':os.getpid(),'started_utc':now(),'load_average':os.getloadavg(),'external_seconds':EXTERNAL,'kill_grace_seconds':5}
    save('launcher-start.json',meta);code=2
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'pid':proc.pid,'command':command});code=proc.wait(timeout=EXTERNAL+7)
        except BaseException as exc:meta['exception']=type(exc).__name__+': '+str(exc)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    remaining=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                s=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(s[2])==proc.pid and s[0]!='Z':remaining.append(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    if remaining or sum(e['bytes'] for e in inventory(out).values())>CAP+1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_leader_exited=proc is not None and proc.poll() is not None,owned_live_pids_remaining=remaining,finished_utc=now());save('launcher-result.json',meta)
    files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':sum(v['bytes'] for v in files.values())<CAP+1024**2})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted((p for p in out.rglob('*') if p.is_dir()),reverse=True):p.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json')}),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pins',type=Path,required=True);p.add_argument('--pins-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((execute if a.execute else launch)(a.pins,a.pins_sha256,a.out))
