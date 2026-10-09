"""One source-bound post-outcome replay and nested check of the selected flip.

No search, orbit continuation, eigensolver or extra target is permitted.
No original proof or numerical attempt is modified.
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
from neural_shrimp.rigorous_flips import check_flip
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,eye,norm_inf_upper
BASE=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-flip-2026-10-09')
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/c1-2026-10-07/attempt-003/inputs.json')
NUMERIC=Path('/data2/neural-shrimp-data/explore/s2-period18-flip/attempt-002-target')
NUMERIC_PIN='bd34d93e8430ed17b0705725ef27c384c7e8ae4853e20f8363042cfcc493123c'
CERTS=Path('/data2/neural-shrimp-data/raw/EXP-028/attempt-002/stage3/certificates.json')
RAYS=Path('/data2/neural-shrimp-data/explore/a2-survey/rays.json')
PINS={PARENT:'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',CERTS:'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8',RAYS:'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4'}
PRODUCER_COMMIT='331d6b617275d45eb7854194f7d645c450fd6a30'
KERNEL_SHA='034543778eca5d08f8ea22e8d30f75e4ab5afa40ac86109408251fea3661e715'
PRODUCER_FILES=['src/neural_shrimp/rigorous_flips.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tools/certify_s2_period18_flip.py','tests/test_rigorous_flips.py','docs/plans/s2-period18-flip-certificate.md']
NUMERIC_FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/numerical_flip.py','tools/explore_s2_period18_flip.py','tests/test_numerical_flip.py','docs/plans/s2-period18-flip-pilot.md']
OWN_FILES=PRODUCER_FILES+['tools/reviews/review_rigorous_period18_flip.py','tests/reviews/test_rigorous_period18_flip_review.py','docs/plans/rigorous-period18-flip-independent-review.md','docs/reviews/s2-period18-flip-review-pins.json','docs/reviews/s2-period18-flip-kernel-source-review.json']
CAP=16*1024**2
INTERNAL=60
EXTERNAL=75

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
    require(out[PRODUCER_FILES[0]]==KERNEL_SHA,'reviewed unchanged flip kernel')
    for p in PRODUCER_FILES:require((ROOT/p).read_bytes()==subprocess.check_output(['git','show',PRODUCER_COMMIT+':'+p],cwd=ROOT),'unchanged full producer source '+p)
    return out

def binding(name,pin):
    root=BASE/name
    verify(root,pin['manifest_sha256'],True)
    verify(root/'run',pin['run_manifest_sha256'],True)
    runtime=read(root/'run/runtime-start.json');launch=read(root/'launcher-result.json');outcome=read(root/'run/outcome.json')
    require(runtime['source_commit']==launch['source_commit']==pin['execution_commit'],'execution source binding')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',pin['execution_commit']+':'+p],cwd=ROOT)).hexdigest() for p in PRODUCER_FILES}
    require(runtime['source_sha256']==launch['source_sha256']==expected,'full producer source binding')
    require(expected=={p:hashlib.sha256(subprocess.check_output(['git','show',PRODUCER_COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in PRODUCER_FILES},'same full reviewed producer source')
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
    verify(NUMERIC,NUMERIC_PIN,True);verify(NUMERIC/'run',sha(NUMERIC/'run/manifest.json'),True)
    raw=NUMERIC/'run';runtime=read(raw/'runtime-start.json');launch=read(NUMERIC/'launcher-result.json')
    require(runtime['source_commit']==launch['source_commit']=='9cc21b7f52b7b2edadb02cd1098d4843c4f1a3aa','fixed numerical execution')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+p],cwd=ROOT)).hexdigest() for p in NUMERIC_FILES}
    require(runtime['source_sha256']==launch['source_sha256']==expected,'complete numerical source inventory')
    for p,h in expected.items():require(sha(raw/'source'/p)==h,'retained numerical source')
    num=read(raw/'inputs.json');point=read(raw/'flip.json');seed=read(raw/'flip-proof-seed.json')
    require(seed['source_commit']==runtime['source_commit'] and seed['inputs_sha256']==sha(raw/'inputs.json'),'seed provenance')
    old=read(PARENT);row=read(CERTS)[111];window=read(RAYS)['S2-28011-zoom:587'];ray=window['rays'][13]
    require((row['map_id'],row['component'],row['period'])==('S2-28011-zoom',587,18),'selected historical row')
    require([i for i,r in enumerate(window['rays']) if r['type']=='flip'][0]==13,'fixed lowest flip ray')
    require(num['row_index']==111 and num['ray_index']==13 and num['period']==18 and num['ray']==ray,'numerical selection')
    require(num['historical_orbit']==row['certificate']['orbit'] and num['input_pins']=={str(p):h for p,h in PINS.items()},'original row and input pins')
    model={k:old[k] for k in ('schema','network','x','y')}
    require(num['consumed_map']==seed['consumed_map']==model,'numerical exact consumed model')
    model.update(lr_hidden='1/1',lr_output='1/1');require(inp['map_inputs']==model,'exact proof model')
    ell=[Fraction(float(x)) for x in window['center']]
    direction=[Fraction(float(a)*float(b)) for a,b in zip(window['scale'],ray['direction'])]
    sf=seed['family'];require(sf==num['family'] and sf['historical_scale']==window['scale'] and sf['historical_unit_direction']==ray['direction'],'original factors')
    for key,expected in [('ell',ell),('direction',direction)]:
        require([Fraction(x) for x in inp['rate_family'][key]]==[Fraction(x) for x in sf[key]]==[Fraction(float(x)) for x in sf[key+'_float']]==expected,'exact family '+key)
    require(inp['schema']=='neural-shrimp-flip-input-v1' and inp['period']==18 and inp['rate_family']['schema']==sf['schema']=='neural-shrimp-exact-log-ray-v1','input schema')
    proposal=list(point['state'])+[point['s']]+list(point['q'])
    require(point['qualified'] is True and len(proposal)==27,'qualified full numerical proposal')
    require([Fraction(x) for x in inp['approximation']]==[Fraction(x) for x in seed['theta']+[seed['parameter']]+seed['right_vector']]==[Fraction(float(x)) for x in proposal],'exact selected proposal')
    p=inp['provenance'];require(p['historical_input_pins']=={str(k):v for k,v in PINS.items()} and p['numeric_root']==str(NUMERIC) and p['numeric_manifest_sha256']==NUMERIC_PIN and p['numeric_run_manifest_sha256']==sha(raw/'manifest.json') and p['numeric_seed_sha256']==sha(raw/'flip-proof-seed.json') and p['numeric_execution_commit']==runtime['source_commit'],'all provenance joins')
    return {'input_files':{str(p):entry(p) for p in PINS},'numerical_manifest':NUMERIC_PIN,'exact_ell':[str(x) for x in ell],'exact_direction':[str(x) for x in direction],'period':18,'dimension':13,'selection':'row111, S2-28011-zoom:587, ray13; consumed development only'}

def saved_math(result,inp):
    require(result['passed'] and len(result['gates'])==8 and all(result['gates'].values()),'all eight flip gates')
    require([d['divisor'] for d in result['divisors']]==[1,2,3,6,9],'exact proper divisors')
    require(len(result['box'])==27 and len(result['monodromy'])==13,'full state dimensions')
    require(nested(result['krawczyk_image'],result['box']),'exact Fraction strict Krawczyk inclusion')
    require(exact_norm(result['defect'])<1,'exact saved augmented defect')
    with ctx.workprec(384):
        box=[stored(b) for b in result['box']];center=[stored(b) for b in result['center']]
        n=13;q=box[n+1:];jac=matrix(result['jacobian_box']);pre=matrix(result['preconditioner'])
        require(all(interval(x)[0]==interval(x)[1] for row in result['preconditioner'] for x in row),'point preconditioner')
        defect=eye(27)-pre*jac;contract=norm_inf_upper(defect);require(contract<1,'independent augmented contraction')
        image=arb_mat([[center[i]] for i in range(27)])-pre*arb_mat([[stored(x)] for x in result['h_center']])+defect*arb_mat([[box[i]-center[i]] for i in range(27)])
        require(all(box[i].contains_interior(image[i,0]) for i in range(27)),'independent Krawczyk image')
        orbit=[[stored(x) for x in row] for row in result['orbit_enclosures']]
        require(len(orbit)==19 and all(len(row)==13 for row in orbit),'complete orbit')
        for d in (1,2,3,6,9):require(any(not (orbit[d][i]-box[i]).contains(0) for i in range(13)),'independent divisor exclusion')
        m=matrix(result['monodromy']);a=m-eye(n);k=m+eye(n)
        border=arb_mat([[k[j,i] for j in range(n)]+[q[i]] for i in range(n)]+[q+[arb(0)]])
        bc=norm_inf_upper(eye(14)-matrix(result['left_inverse_check']['preconditioner'])*border)
        ac=norm_inf_upper(eye(13)-matrix(result['orbit_branch_inverse_check']['preconditioner'])*a)
        require(bc<1 and ac<1,'independent border/branch inverse witnesses')
        left=border.solve(arb_mat([[arb(0)]]*n+[[arb(1)]]));p=[left[i,0] for i in range(n)]
        require(left[n,0].contains(0) and all(p[i].overlaps(stored(result['left_vector'][i])) for i in range(n)),'independent left solve compatibility')
        require(sum((x*x for x in q),arb(0)).contains(1) and sum((x*y for x,y in zip(p,q)),arb(0)).contains(1),'normalizations')
        require(all(sum((k[i,j]*q[j] for j in range(n)),arb(0)).contains(0) for i in range(n)),'minus-one null compatibility')
        mixed=matrix(result['mixed_directional']);r_s=[stored(x) for x in result['parameter_derivative']]
        bqq=[stored(x) for x in result['second_directional']];cqqq=[stored(x) for x in result['third_directional']]
        solved=a.solve(arb_mat([[-r_s[i],bqq[i]] for i in range(n)]));t=[solved[i,0] for i in range(n)];u=[solved[i,1] for i in range(n)]
        require(all(t[i].overlaps(stored(result['orbit_branch_derivative'][i])) and u[i].overlaps(stored(result['quadratic_resolvent'][i])) for i in range(n)),'independent branch/resolvent solve')
        require(all(sum((mixed[i,j]*q[j] for j in range(n)),arb(0)).overlaps(bqq[i]) for i in range(n)),'second-directional tensor contraction consistency')
        total=[mixed[i,n]+sum((mixed[i,j]*t[j] for j in range(n)),arb(0)) for i in range(n)]
        bu=[sum((mixed[i,j]*u[j] for j in range(n)),arb(0)) for i in range(n)]
        alpha=sum((p[i]*total[i] for i in range(n)),arb(0))
        cubic=sum((p[i]*(cqqq[i]/6-bu[i]/2) for i in range(n)),arb(0))
        require(not alpha.contains(0) and not cubic.contains(0),'independent nondegeneracy')
        require(alpha.overlaps(stored(result['alpha'])) and cubic.overlaps(stored(result['cubic_b'])),'stored coefficients consistent')
        require(result['criticality']==('supercritical' if cubic>0 else 'subcritical'),'criticality sign')
        require(result['doubled_side_in_s']==('greater' if -cubic/alpha>0 else 'less'),'detuning side from reduced equation')
        qdef=m+arb_mat([[q[i]*p[j] for j in range(n)] for i in range(n)])
        witness=result['complement_witness_power'];require(type(witness)is int and 1<=witness<=4096 and witness&(witness-1)==0,'fixed power witness')
        power=qdef;kpower=1
        while kpower<witness:power=power*power;kpower*=2
        bound=norm_inf_upper(power);require(bound<1 and exact_norm(result['complement_witness'])<1,'independent complement contraction')
        rates=[]
        for ell,d in zip(inp['rate_family']['ell'],inp['rate_family']['direction']):
            e=Fraction(ell);v=Fraction(d);rate=((arb(e.numerator)/e.denominator+arb(v.numerator)/v.denominator*box[13])*arb(10).log()).exp();require(rate>0 and rate.is_finite(),'positive exact-real rate');rates.append(ball_json(rate))
        return {'strict_inclusion':True,'augmented_contraction':ball_json(contract),'left_border_contraction':ball_json(bc),'orbit_branch_inverse_contraction':ball_json(ac),'alpha':ball_json(alpha),'cubic_b':ball_json(cubic),'detuning_over_amplitude_squared':ball_json(-cubic/alpha),'complement_power':witness,'complement_bound':ball_json(bound),'rates':rates,'proper_divisors':[1,2,3,6,9],'scope':'Independently reduced saved-ball arithmetic with independent linear solves. Stored analytic derivatives and Arb remain shared primitives.'}

def execute(pins_path,pins_sha,out):
    sources=source_snapshot();require(sha(pins_path)==pins_sha,'external pins JSON hash');pins=read(pins_path)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice>=10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();serial=0;ticks=0;stopped=False;result={'status':'incomplete','stages':[]}
    def save(name,value):
        b=canonical_bytes(value);require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<(CAP if name in ('result.json','manifest.json') else CAP-1024**2),'aggregate audit cap')
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
    for p in OWN_FILES:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        require(set(pins)=={'attempt-001-profile','attempt-001-target'},'fixed profile and target inventory')
        bound={}
        for name,pin in pins.items():
            root,runtime0,launch,outcome=binding(name,pin)
            bound[name]={'path':str(root),'pins':pin,'outcome':outcome,'launcher_exit':launch['exit_code'],'source_sha256':runtime0['source_sha256']}
        stage('01-attempt-bindings',bound);tick()
        profile=bound['attempt-001-profile']['outcome'];require(profile['passed'] and profile['scientific_claim'] is False,'completed resource-only profile')
        target=BASE/'attempt-001-target/run';envelope=read(target/'certificate.json');inp=read(target/'inputs.json');proof=envelope['result']
        require(envelope['inputs_sha256']==sha(target/'inputs.json'),'target input certificate join')
        require(envelope['runtime']==read(target/'runtime-start.json'),'certificate runtime join')
        require(proof['settings']=={'precision':256,'radius_exponent':-100,'newton_steps':5,'max_power':4096},'fixed adapted target settings')
        stage('03-input-binding',reconstruct(inp));tick()
        replayed=check_flip(inp,**proof['settings'],tick=tick,observe=observe)
        save('04-replay-result.json',replayed);require(canonical_bytes(replayed)==canonical_bytes(proof),'byte-identical proof replay')
        require(replayed['passed'],'target full-flip pass');stage('05-replay-summary',{'passed':True,'byte_identical':True,'result_sha256':sha(out/'04-replay-result.json')})
        stage('06-saved-arithmetic',saved_math(proof,inp));tick()
        seed=[Fraction(x) for x in inp['approximation']];den=max([Fraction(1)]+[abs(x) for x in seed]);dist=max(max(abs(a-s),abs(b-s))/den for (a,b),s in zip(map(interval,proof['box']),seed))
        require(dist<=Fraction(1,10**7),'selected whole augmented-box proximity')
        require(str(dist)==envelope['whole_box_proximity']['normalized_max_distance'] and envelope['whole_box_proximity']['passed'] is True,'recorded exact proximity');stage('07-whole-box-proximity',{'normalized_distance':str(dist),'display':float(dist),'cap':'1/10000000','scope':'Entire27D box near selected numerical proposal; not seed inclusion or a connection to the interior attracting orbit.'})
        stronger=check_flip(inp,precision=384,radius_exponent=-110,newton_steps=5,max_power=4096,tick=tick,observe=observe)
        save('08-nested-result.json',stronger);require(stronger['passed'],'fixed stronger proof')
        require(nested(stronger['box'],proof['box']),'exact Fraction strict nested proof box')
        stage('09-nested-summary',{'passed':True,'strictly_nested':True,'settings':stronger['settings'],'result_sha256':sha(out/'08-nested-result.json'),'scope':'Same analytic implementation and Arb library at increased precision and smaller box; not a separate AD proof.'})
        result.update(status='passed',scope='Independent source review, byte replay and saved arithmetic; local exact-real primitive18 subcritical flip with attracting complement; qualitative unstable doubled branch only. No whole shrimp geometry, finite-detuning ancestry or utility.')
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
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':sources,'pins_sha256':pins_sha,'argv':command,'launcher_pid':os.getpid(),'started_utc':now(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL,'kill_grace_seconds':5}
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
