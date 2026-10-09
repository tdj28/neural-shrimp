"""Two separate exact-map certificates from the fixed ray51 numerical pilot."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import flint
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import BallTanhMap,check_orbit,canonical_bytes,exact,vector_json,matrix_json
from neural_shrimp.rigorous_flips import FlipTanhMap,augmented,check_flip
RAW=Path('/data2/neural-shrimp-data')
PARENT=RAW/'explore/certificates/c1-2026-10-07/attempt-003/inputs.json'
CERTS=RAW/'raw/EXP-028/attempt-002/stage3/certificates.json'
RAYS=RAW/'explore/a2-survey/rays.json'
PINS={str(PARENT):'a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81',str(CERTS):'414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8',str(RAYS):'a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4'}
NUMERIC=RAW/'explore/s2-period18-other-flips/attempt-002'
NUMERIC_PIN='9a199a50d93954a836ac5b642eb4f035afefcf3628c59a8429702e9e089306db'
NUMERIC_COMMIT='24ef0d36cfe98fe19011def9436340556b47af4a'
NUMERIC_SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/numerical_flip.py','tools/explore_s2_period18_flip.py','tests/test_numerical_flip.py','docs/plans/s2-period18-flip-pilot.md','tools/explore_s2_period18_other_flips.py','tests/test_s2_period18_other_flips.py','docs/plans/s2-period18-other-flips.md']
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_flips.py','tools/certify_s2_ray51.py','tests/test_s2_ray51_certificates.py','docs/plans/s2-ray51-certificates.md']
SETTINGS={'flip':dict(precision=256,radius_exponent=-100,newton_steps=5,max_power=4096),'period36':dict(precision=384,radius_exponent=-160,newton_steps=5,max_power=4096)}
CAP=16*1024**2;INTERNAL=60;EXTERNAL=75
ARTIFACT_ROOT=None

def require(ok,why):
    if not ok:raise ValueError(why)
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entry(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def total(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def save(p,obj):
    b=canonical_bytes(obj);require(len(b)<CAP,'single file cap')
    final=p.name in ('outcome.json','failure.json','manifest.json','launcher-result.json')
    limit=(CAP+1024**2 if (ARTIFACT_ROOT/'run').exists() else CAP) if final else 14*1024**2
    require(total(ARTIFACT_ROOT)+len(b)<limit,'aggregate output cap with finalization reserve');tmp=p.with_name(p.name+'.tmp')
    with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def inventory(p):return {str(f.relative_to(p)):entry(f) for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'manifest.json'}
def verify(p,pin):
    require(p.is_dir() and not p.is_symlink() and sha(p/'manifest.json')==pin,'externally pinned bundle')
    require(not any(f.is_symlink() for f in p.rglob('*')),'no artifact symlinks')
    m=read(p/'manifest.json');require(m['files']==inventory(p) and m['total_bytes']==sum(e['bytes'] for e in m['files'].values()),'complete inventory/totals')
    return m

def source_binding():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed checkout')
    hashes={}
    for p in SOURCES:
        b=(ROOT/p).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed source '+p);hashes[p]=hashlib.sha256(b).hexdigest()
    require(hashes['src/neural_shrimp/rigorous_flips.py']=='034543778eca5d08f8ea22e8d30f75e4ab5afa40ac86109408251fea3661e715','unchanged reviewed flip kernel')
    require(hashes['src/neural_shrimp/rigorous_orbits.py']=='3067fe661fadac0a481c3d33515a26056cef3a5ee0aea018f07f738d7395d8ee','unchanged reviewed C1 kernel')
    return hashes

def assemble(kind,parent,row,window,numeric,point,seed,doubled):
    require((row['map_id'],row['component'],row['period'])==('S2-28011-zoom',587,18),'historical row111')
    require(numeric['row_index']==111 and numeric['ray_index']==51 and numeric['period']==18 and numeric['ray']==window['rays'][51] and numeric['historical_orbit']==row['certificate']['orbit'],'fixed ray51 metadata')
    require(window['rays'][51]['type']=='flip','historical flip classification')
    model={k:parent[k] for k in ('schema','network','x','y')};require(numeric['consumed_map']==seed['consumed_map']==model,'same exact consumed model/data')
    ell=[Fraction(float(x)) for x in window['center']];direction=[Fraction(float(a)*float(b)) for a,b in zip(window['scale'],window['rays'][51]['direction'])]
    family={'schema':'neural-shrimp-exact-log-ray-v1','ell':[str(x) for x in ell],'direction':[str(x) for x in direction]}
    sf=seed['family'];require(sf==numeric['family'] and sf['schema']==family['schema'] and sf['historical_scale']==window['scale'] and sf['historical_unit_direction']==window['rays'][51]['direction'],'same original family factors')
    for key,expected in [('ell',ell),('direction',direction)]:require([Fraction(x) for x in sf[key]]==[Fraction(float(x)) for x in sf[key+'_float']]==expected,'exact family '+key)
    require(seed['schema']=='neural-shrimp-flip-proof-seed-v1' and seed['ray_index']==51 and seed['period']==18 and point['qualified'] is True,'qualified selected flip seed')
    proposal=point['state']+[point['s']]+point['q'];require(len(proposal)==27 and all(math.isfinite(x) for x in proposal),'full finite flip proposal')
    require([Fraction(x) for x in seed['theta']+[seed['parameter']]+seed['right_vector']]==[Fraction(float(x)) for x in proposal],'exact selected flip proposal')
    if kind=='flip':
        model.update(lr_hidden='1/1',lr_output='1/1')
        return {'schema':'neural-shrimp-flip-input-v1','period':18,'map_inputs':model,'rate_family':family,'approximation':[str(Fraction(float(x))) for x in proposal]}, {'kind':kind,'ray_index':51,'map_convention':'Exact-real exponential ray with both rates varying; base rates are unused placeholders.','connection_not_proved':True}
    require(kind=='period36' and len(doubled)==3,'fixed third of three amplitude slots')
    chosen=doubled[2];require(chosen['qualified'] is True and chosen['qualification']['stability']==chosen['opposite_qualification']['stability']=='attracting','selected both-phase attracting36')
    require(chosen['amplitude_requested']==max(x['amplitude_requested'] for x in doubled),'largest amplitude selection')
    data=chosen['data'];require(data['period']==36 and len(data['orbit'])==37 and data['theta']==chosen['state'] and data['orbit'][0]==chosen['state'] and data['s']==chosen['s'],'full selected phase and parameter')
    rates=[10.0**float(float(e)+float(chosen['s'])*float(d)) for e,d in zip(ell,direction)]
    require(all(math.isfinite(x) and x>0 for x in rates),'positive actual numerical rates')
    require([Fraction(float(x)) for x in data['rates']]==[Fraction(x) for x in data['rates_exact']]==[Fraction(x) for x in rates],'exact realized binary64 rate identity')
    require(len(chosen['state'])==13 and all(math.isfinite(x) for x in chosen['state']),'finite full36 phase')
    model.update(lr_hidden=str(Fraction(rates[0])),lr_output=str(Fraction(rates[1])),period=36,approximation=[str(Fraction(float(x))) for x in chosen['state']])
    return model,{'kind':kind,'ray_index':51,'amplitude_index':2,'numeric_s':chosen['s'],'amplitude':chosen['amplitude_requested'],'exact_rate_hex':[x.hex() for x in rates],'exact_rates':[str(Fraction(x)) for x in rates],'map_convention':'Fixed exact dyadic binary64 rates from selected finite36 evaluation; distinct from the exact exponential-family flip.','connection_not_proved':True}

def prepare(kind):
    for p,h in PINS.items():require(sha(Path(p))==h,'consumed input pin '+p)
    verify(NUMERIC,NUMERIC_PIN);verify(NUMERIC/'run',sha(NUMERIC/'run/manifest.json'))
    runtime=read(NUMERIC/'run/runtime-start.json');launcher=read(NUMERIC/'launcher-result.json')
    require(runtime['source_commit']==launcher['source_commit']==NUMERIC_COMMIT and launcher['exit_code']==0 and launcher['owned_live_pids_remaining']==[],'numerical execution join')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',NUMERIC_COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in NUMERIC_SOURCES}
    require(runtime['source_sha256']==launcher['sources']==expected,'complete committed numerical sources')
    for p,h in expected.items():require(sha(NUMERIC/'run/source'/p)==h,'retained numerical source')
    raw=NUMERIC/'run/ray-51';numeric=read(raw/'inputs.json');point=read(raw/'flip.json');seed=read(raw/'flip-proof-seed.json')
    require(numeric['input_pins']==PINS and seed['source_commit']==NUMERIC_COMMIT and seed['inputs_sha256']==sha(raw/'inputs.json'),'seed/model provenance')
    require(read(NUMERIC/'run/selection.json')['selected']==[16,51],'fixed original selection')
    inp,selection=assemble(kind,read(PARENT),read(CERTS)[111],read(RAYS)['S2-28011-zoom:587'],numeric,point,seed,read(raw/'doubled.json'))
    inp['provenance']={'historical_pins':PINS,'numeric_root':str(NUMERIC),'numeric_manifest_sha256':NUMERIC_PIN,'numeric_execution_commit':NUMERIC_COMMIT,'files':{n:entry(raw/n) for n in ['inputs.json','flip.json','flip-proof-seed.json','doubled.json']},'selection':selection}
    return inp

def proximity(inp,result):
    def pair(x):require(len(x)==2 and all(type(i)is int for i in x),'dyadic pair');return Fraction(x[0])*Fraction(2)**x[1]
    seed=[Fraction(x) for x in inp['approximation']];require(len(seed)==len(result['box']),'full proximity dimension');den=max([Fraction(1)]+[abs(x) for x in seed]);distance=Fraction(0)
    for b,x in zip(result['box'],seed):
        mid,rad=pair(b['mid']),pair(b['rad']);require(rad>=0,'nonnegative radius');distance=max(distance,(abs(mid-x)+rad)/den)
    return {'passed':distance<=Fraction(1,10**7),'normalized_whole_box_distance':str(distance),'display':float(distance),'cap':'1/10000000','scope':'Entire proved box near selected numerical proposal, not seed membership or connection between finite36 and flip germ.'}

def synthetic(kind):
    model={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],'y':[str(Fraction((i*3)%17-8,64)) for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
    theta=[str(Fraction(i-6,64)) for i in range(13)]
    if kind=='flip':return {'map_inputs':model,'rate_family':{'schema':'neural-shrimp-exact-log-ray-v1','ell':['-2','-2'],'direction':['1/20','-3/100']},'approximation':theta+['1/7']+[str(Fraction(i+1,64)) for i in range(13)]}
    return dict(model,period=36,approximation=theta)

def profile_gate(profile,pin,kind):
    require(profile is not None and pin is not None,'measured profile required');verify(profile,pin);verify(profile/'run',sha(profile/'run/manifest.json'))
    runtime=read(profile/'run/runtime-start.json');launch=read(profile/'launcher-result.json');result=read(profile/'run/outcome.json')
    require(runtime['kind']==launch['kind']==kind and runtime['mode']==launch['mode']=='profile' and runtime['source_commit']==launch['source_commit'] and launch['exit_code']==0 and launch['owned_live_pids_remaining']==[] and result['status']=='synthetic_profile_complete' and result['passed'] and result['scientific_claim'] is False,'complete same-kind resource profile')
    require(runtime['source_sha256']==launch['source_sha256']==source_binding(),'unchanged full profile source')
    for p,h in runtime['source_sha256'].items():
        require(sha(profile/'run/source'/p)==h and hashlib.sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+p],cwd=ROOT)).hexdigest()==h,'committed/retained profile source')
    require(total(profile/'run')<CAP and total(profile)<CAP+1024**2,'profile aggregate caps')
    require(runtime['threads']=={k:'1' for k in THREADS} and runtime['nice']>=10 and runtime['python_flint']==flint.__version__ and runtime['flint']==flint.__FLINT_VERSION__,'same profile environment')
    require(len(result['timings'])==2 and all(type(t)is float and math.isfinite(t) and 0<t<INTERNAL for t in result['timings']),'two finite timings')
    projection=7*max(result['timings'])+10;require(projection<=45 and result['projected_seconds']==projection,'measured45s projection gate')
    profile_resources(result,launch)
    return {'profile':str(profile),'manifest_sha256':pin,'projected_seconds':projection,'cap_seconds':45,'passed':True}

def profile_resources(result,launch):
    for value,cap in [(result['elapsed_seconds'],INTERNAL),(result['max_rss_kib'],2*1024**2),(launch['elapsed_seconds'],EXTERNAL+7)]:
        require(type(value)in(int,float) and math.isfinite(value) and 0<=value<cap,'finite nonnegative profile resource value')
    require(type(result['max_rss_kib'])is int and result['max_rss_kib']>0,'positive integer RSS')
    require(max(result['timings'])<=result['elapsed_seconds']<=launch['elapsed_seconds'],'consistent profile timing')


def execute(out,kind,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    global ARTIFACT_ROOT
    ARTIFACT_ROOT=out
    sources=source_binding();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();serial=0;ticks=0;stopped=False
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'kind':kind,'mode':mode,'settings':SETTINGS[kind],'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'internal_seconds':INTERNAL,'raw_cap':CAP,'cpu_only':True};save(out/'runtime-start.json',runtime)
    for p in SOURCES:q=out/'source'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
    def stop(*_):
        nonlocal stopped
        stopped=True
        if kind=='period36':raise TimeoutError('unchanged C1 kernel deadline')
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>160 or stopped or time.monotonic()-start>INTERNAL:raise TimeoutError('wall/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored RSS2GiB')
        if total(out)>14*1024**2:raise OSError('ordinary14MiB cap')
    def observe(name,obj):
        nonlocal serial
        save(out/f'evaluation-{serial:03d}-{name}.json',obj);serial+=1;tick()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP));outcome={'passed':False,'status':'started','kind':kind,'mode':mode};code=2
    try:
        if mode=='profile':
            inp=synthetic(kind);save(out/'inputs.json',inp);times=[]
            with ctx.workprec(SETTINGS[kind]['precision']):
                theta=[exact(x) for x in inp['approximation']];radius=arb((1,SETTINGS[kind]['radius_exponent']))
                for tag,z in [('center',theta),('box',[arb(x,radius) for x in theta])]:
                    t=time.monotonic()
                    if kind=='flip':
                        h,j,data=augmented(FlipTanhMap(inp['map_inputs'],inp['rate_family']),z,18,tick);payload={'h':vector_json(h),'jacobian':matrix_json(j)};require(all(x.is_finite() for x in h+data['third_directional']) and all(j[i,k].is_finite() for i in range(j.nrows()) for k in range(j.ncols())),'finite flip profile')
                    else:
                        end,m,orbit,js=BallTanhMap(inp).iterate(z,36);payload={'end':vector_json(end),'monodromy':matrix_json(m),'orbit':[vector_json(x) for x in orbit]};require(all(x.is_finite() for x in end) and all(m[i,k].is_finite() for i in range(13) for k in range(13)),'finite C1 profile')
                    observe('profile-'+tag,payload);times.append(time.monotonic()-t)
            outcome.update(status='synthetic_profile_complete',passed=True,scientific_claim=False,timings=times,projected_seconds=7*max(times)+10);code=0
        else:
            if mode=='target':save(out/'profile-eligibility.json',profile_gate(profile,profile_pin,kind));inp=prepare(kind)
            else:
                require(bundle is not None and bundle_pin is not None,'pinned replay');verify(bundle,bundle_pin);envelope=read(bundle/'certificate.json');inp=read(bundle/'inputs.json')
                require(envelope['kind']==kind and envelope['inputs_sha256']==sha(bundle/'inputs.json'),'replay kind/input join')
                require(canonical_bytes(inp)==canonical_bytes(prepare(kind)),'original exact input reconstruction')
                require(envelope['runtime']['source_sha256']==sources and envelope['runtime']['python_flint']==flint.__version__ and envelope['runtime']['flint']==flint.__FLINT_VERSION__,'replay source/library equality')
                require(envelope['result']['settings']==SETTINGS[kind],'frozen settings')
            save(out/'inputs.json',inp)
            result=check_flip(inp,**SETTINGS[kind],tick=tick,observe=observe) if kind=='flip' else check_orbit(inp,**SETTINGS[kind])
            save(out/'mathematical-result.json',result);tick();near=proximity(inp,result)
            if mode=='replay':require(canonical_bytes(result)==canonical_bytes(envelope['result']) and near==envelope['whole_box_proximity'],'byte-identical proof and proximity replay')
            save(out/'certificate.json',{'schema':'neural-shrimp-ray51-certificate-v1','kind':kind,'runtime':runtime,'inputs_sha256':sha(out/'inputs.json'),'result':result,'whole_box_proximity':near})
            outcome.update(status='certificate_passed' if result['passed'] and near['passed'] else 'certificate_unproved',passed=bool(result['passed'] and near['passed']),mathematical_proof_passed=result['passed'],association_passed=near['passed'],byte_identical_replay=mode=='replay');code=0 if outcome['passed'] else 2
    except BaseException as exc:outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,evaluations_retained=serial,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=now(),exit_code=code);save(out/'outcome.json',outcome);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values()),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,kind,mode,profile=None,profile_pin=None,bundle=None,bundle_pin=None):
    global ARTIFACT_ROOT
    ARTIFACT_ROOT=out
    sources=source_binding();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL}s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--kind',kind,'--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile-root',profile),('--profile-manifest-sha256',profile_pin),('--bundle',bundle),('--manifest-sha256',bundle_pin)]:
        if value is not None:cmd+=[flag,str(value)]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'kind':kind,'mode':mode,'argv':cmd,'launcher_pid':os.getpid(),'started_utc':now(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL,'kill_grace':5};save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=EXTERNAL+7)
        except BaseException as exc:meta['failure']=type(exc).__name__+': '+str(exc)
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
                f=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(f[2])==proc.pid and f[0]!='Z':remaining.append(int(p.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    if remaining or total(out)>CAP+1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_live_pids_remaining=remaining,finished_utc=now());save(out/'launcher-result.json',meta);files=inventory(out);save(out/'manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values()),'within_limit':total(out)<CAP+1024**2})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in out.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--kind',choices=['flip','period36'],required=True);p.add_argument('--mode',choices=['profile','target','replay'],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--profile-root',type=Path);p.add_argument('--profile-manifest-sha256');p.add_argument('--bundle',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args();raise SystemExit((execute if a.execute else launch)(a.out,a.kind,a.mode,a.profile_root,a.profile_manifest_sha256,a.bundle,a.manifest_sha256))
