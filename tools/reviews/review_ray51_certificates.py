"""One bounded, source-bound review of ray51 flip and fixed-dyadic36 certificates.

Shares the previously reviewed proof kernels. Reconstructs adapter inputs and
reduces saved interval witnesses independently of the new producer adapter.
"""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import argparse,hashlib,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import canonical_bytes,check_orbit,ball_json,eye,norm_inf_upper
from neural_shrimp.rigorous_flips import check_flip
import review_rigorous_period18_flip as prior
PRODUCER='04048350fdaff3bba5a5d9e82b059f10761a0dab'
PRODUCER_FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_flips.py','tools/certify_s2_ray51.py','tests/test_s2_ray51_certificates.py','docs/plans/s2-ray51-certificates.md']
OWN_FILES=PRODUCER_FILES+['tools/reviews/review_rigorous_period18_flip.py','tools/reviews/review_ray51_certificates.py','tests/reviews/test_ray51_certificate_review.py','docs/plans/ray51-independent-review.md','docs/reviews/ray51-adapter-source-review.json']
NUMERIC=Path('/data2/neural-shrimp-data/explore/s2-period18-other-flips/attempt-002')
NUMERIC_SHA='9a199a50d93954a836ac5b642eb4f035afefcf3628c59a8429702e9e089306db'
NUMERIC_COMMIT='24ef0d36cfe98fe19011def9436340556b47af4a'
NUMERIC_FILES=prior.NUMERIC_FILES+['tools/explore_s2_period18_other_flips.py','tests/test_s2_period18_other_flips.py','docs/plans/s2-period18-other-flips.md']
ORIGINAL={'flip':dict(precision=256,radius_exponent=-100,newton_steps=5,max_power=4096),'period36':dict(precision=384,radius_exponent=-160,newton_steps=5,max_power=4096)}
STRONGER={'flip':dict(precision=384,radius_exponent=-110,newton_steps=5,max_power=4096),'period36':dict(precision=512,radius_exponent=-170,newton_steps=5,max_power=4096)}
CAP=32*1024**2;INTERNAL=60;EXTERNAL=75
require=prior.require;sha=prior.sha;read=prior.read;entry=prior.entry;interval=prior.interval;stored=prior.stored;matrix=prior.matrix;nested=prior.nested

def now():return datetime.now(timezone.utc).isoformat()
def inventory(root):return {str(p.relative_to(root)):entry(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
def verify(root,pin):
    require(root.is_dir() and not root.is_symlink() and sha(root/'manifest.json')==pin,'external manifest pin')
    require(not any(p.is_symlink() for p in root.rglob('*')),'no artifact symlinks')
    m=read(root/'manifest.json');require(m['files']==inventory(root),'exact complete inventory')
    require(m['total_bytes']==sum(e['bytes'] for e in m['files'].values()),'manifest total')
    if 'within_limit'in m:require(m['within_limit'] is True,'manifest cap')
    return m

def finite_time(value,cap):
    require(type(value)in(int,float) and math.isfinite(value) and 0<=value<=cap,'finite nonnegative resource bound')

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean reviewer checkout')
    result={}
    for p in OWN_FILES:
        b=(ROOT/p).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed reviewer source '+p);result[p]=hashlib.sha256(b).hexdigest()
    for p in PRODUCER_FILES:require((ROOT/p).read_bytes()==subprocess.check_output(['git','show',PRODUCER+':'+p],cwd=ROOT),'unchanged reviewed producer source '+p)
    return result

def bind_bundle(pin,kind,mode):
    root=Path(pin['root']);verify(root,pin['manifest_sha256']);verify(root/'run',pin['run_manifest_sha256'])
    run=root/'run';rt=read(run/'runtime-start.json');launch=read(root/'launcher-result.json');result=read(run/'outcome.json')
    require(rt['source_commit']==launch['source_commit']==pin['execution_commit'],'execution commit join')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',PRODUCER+':'+p],cwd=ROOT)).hexdigest() for p in PRODUCER_FILES}
    require(rt['source_sha256']==launch['source_sha256']==expected,'full reviewed source inventory')
    for p,h in expected.items():require(sha(run/'source'/p)==h,'retained source '+p)
    require(rt['kind']==launch['kind']==result['kind']==kind and rt['mode']==launch['mode']==result['mode']==mode,'same kind/mode')
    require(rt['settings']==ORIGINAL[kind],'fixed producer settings')
    require(rt['threads']=={k:'1' for k in THREADS} and rt['nice']>=10,'CPU1/nice')
    require(rt['python_flint']==flint.__version__ and rt['flint']==flint.__FLINT_VERSION__,'same Arb environment')
    require(launch['exit_code']==result['exit_code']==0 and launch['owned_live_pids_remaining']==[],'successful producer exit')
    for pid in (rt['pid'],launch['launcher_pid'],launch['owned_group_pid']):require(not Path('/proc',str(pid)).exists(),'owned producer PID absent')
    for value,cap in ((result['elapsed_seconds'],60),(launch['elapsed_seconds'],82),(result['max_rss_kib'],2*1024**2),(result['ticks'],160)):finite_time(value,cap)
    require(sum(v['bytes'] for v in inventory(root).values())<17*1024**2,'producer total size cap')
    if mode=='profile':
        require(result['passed'] and result['scientific_claim'] is False and result['status']=='synthetic_profile_complete','resource-only profile')
        require(len(result['timings'])==2,'two profile evaluations')
        for t in result['timings']:finite_time(t,60);require(t>0,'positive evaluation time')
        require(result['projected_seconds']==7*max(result['timings'])+10<=45,'profile projection gate')
    else:require(result['passed'] and result['mathematical_proof_passed'] and result['association_passed'],'producer full target pass')
    return root,rt,result

def exact_family(window):
    ray=window['rays'][51]
    return {'schema':'neural-shrimp-exact-log-ray-v1','ell':[str(Fraction(float(x))) for x in window['center']],'direction':[str(Fraction(float(a)*float(b))) for a,b in zip(window['scale'],ray['direction'])]}

def input_identity(kind,inp,parent,row,window,num,point,seed,branches):
    require((row['map_id'],row['component'],row['period'])==('S2-28011-zoom',587,18),'fixed historical orbit')
    require(num['row_index']==111 and num['ray_index']==51 and num['period']==18 and num['ray']==window['rays'][51] and num['historical_orbit']==row['certificate']['orbit'],'ray51 native metadata')
    model={k:parent[k] for k in ('schema','network','x','y')};require(model==num['consumed_map']==seed['consumed_map'],'same consumed model/data')
    family=exact_family(window);sf=seed['family'];require(sf==num['family'] and sf['schema']==family['schema'],'family schema')
    for k in ('ell','direction'):require([Fraction(x) for x in sf[k]]==[Fraction(float(x)) for x in sf[k+'_float']]==[Fraction(x) for x in family[k]],'family coefficient '+k)
    require(sf['historical_scale']==window['scale'] and sf['historical_unit_direction']==window['rays'][51]['direction'],'family original factors')
    require(seed['schema']=='neural-shrimp-flip-proof-seed-v1' and seed['period']==18 and seed['ray_index']==51 and point['qualified'] is True,'qualified fixed flip proposal')
    proposal=point['state']+[point['s']]+point['q'];require(len(proposal)==27,'full flip proposal')
    require([Fraction(x) for x in seed['theta']+[seed['parameter']]+seed['right_vector']]==[Fraction(float(x)) for x in proposal],'exact numeric flip proposal')
    if kind=='flip':
        model.update(lr_hidden='1/1',lr_output='1/1')
        require(inp['schema']=='neural-shrimp-flip-input-v1' and inp['period']==18 and inp['map_inputs']==model and inp['rate_family']==family,'exact exponential-family proof input')
        expected=[Fraction(float(x)) for x in proposal]
        detail={'rate_family':family,'map_scope':'Exact exponential family, not the fixed dyadic36 map.'}
    else:
        require(kind=='period36' and len(branches)==3,'exact fixed amplitude slots')
        branch=branches[2];require(branch['qualified'] is True and branch['qualification']['stability']==branch['opposite_qualification']['stability']=='attracting','both numerical starts attracting')
        require(branch['amplitude_requested']==max(b['amplitude_requested'] for b in branches),'fixed maximal-amplitude slot2')
        data=branch['data'];require(data['period']==36 and len(data['orbit'])==37 and data['theta']==data['orbit'][0]==branch['state'] and data['s']==branch['s'],'saved full36 state')
        rates=[10.0**float(float(Fraction(e))+float(branch['s'])*float(Fraction(d))) for e,d in zip(family['ell'],family['direction'])]
        exact_rates=[Fraction(float(v)) for v in rates]
        require(exact_rates==[Fraction(float(x)) for x in data['rates']]==[Fraction(x) for x in data['rates_exact']],'original scalar-power actual rates')
        require(inp['schema']==model['schema'] and inp['period']==36,'fixed36 schema/period')
        for k in ('network','x','y'):require(inp[k]==model[k],'exact36 consumed '+k)
        require([Fraction(inp[k]) for k in ('lr_hidden','lr_output')]==exact_rates,'finite36 dyadic rate identity')
        expected=[Fraction(float(x)) for x in branch['state']]
        detail={'amplitude_index':2,'exact_fixed_rates':[str(x) for x in exact_rates],'rate_hex':[x.hex() for x in rates],'map_scope':'Fixed dyadic rates; no equality or continuous connection to the exponential flip family asserted.'}
    require([Fraction(x) for x in inp['approximation']]==expected,'exact selected full-state proposal')
    return detail

def reconstruct(kind,inp):
    for p,h in prior.PINS.items():require(sha(p)==h,'historical hash pin')
    verify(NUMERIC,NUMERIC_SHA);verify(NUMERIC/'run',sha(NUMERIC/'run/manifest.json'))
    raw=NUMERIC/'run';rt=read(raw/'runtime-start.json');launcher=read(NUMERIC/'launcher-result.json')
    require(rt['source_commit']==launcher['source_commit']==NUMERIC_COMMIT and launcher['exit_code']==0 and launcher['owned_live_pids_remaining']==[],'numeric execution')
    expected={p:hashlib.sha256(subprocess.check_output(['git','show',NUMERIC_COMMIT+':'+p],cwd=ROOT)).hexdigest() for p in NUMERIC_FILES}
    require(rt['source_sha256']==launcher['sources']==expected,'complete numeric source inventory')
    for p,h in expected.items():require(sha(raw/'source'/p)==h,'saved numeric source')
    raw=raw/'ray-51';num=read(raw/'inputs.json');point=read(raw/'flip.json');seed=read(raw/'flip-proof-seed.json');branches=read(raw/'doubled.json')
    require(num['input_pins']=={str(p):h for p,h in prior.PINS.items()} and seed['source_commit']==NUMERIC_COMMIT and seed['inputs_sha256']==sha(raw/'inputs.json'),'historical numeric joins')
    require(read(NUMERIC/'run/selection.json')['selected']==[16,51],'fixed numerical selection')
    detail=input_identity(kind,inp,read(prior.PARENT),read(prior.CERTS)[111],read(prior.RAYS)['S2-28011-zoom:587'],num,point,seed,branches)
    provenance=inp['provenance'];require(provenance['historical_pins']=={str(p):h for p,h in prior.PINS.items()} and provenance['numeric_root']==str(NUMERIC) and provenance['numeric_manifest_sha256']==NUMERIC_SHA and provenance['numeric_execution_commit']==NUMERIC_COMMIT,'proof provenance joins')
    require(provenance['files']=={n:entry(raw/n) for n in ('inputs.json','flip.json','flip-proof-seed.json','doubled.json')},'all named selected input files')
    selection=provenance['selection'];require(selection['kind']==kind and selection['ray_index']==51 and selection['connection_not_proved'] is True,'selection scope')
    if kind=='period36':require(selection['amplitude_index']==2 and selection['exact_rates']==detail['exact_fixed_rates'] and selection['exact_rate_hex']==detail['rate_hex'],'fixed36 selection identity')
    return dict(detail,numeric_manifest_sha256=NUMERIC_SHA,period=inp['period'])

def orbit_math(result):
    require(result['passed'] and result['evidence_rung']==3 and result['exact_statement']['primitive_period']==36 and result['exact_statement']['dimension']==13,'full primitive36 proof')
    ds=[1,2,3,4,6,9,12,18];require([d['divisor'] for d in result['divisors']]==ds,'all eight proper divisors36')
    require(nested(result['krawczyk_image'],result['box']) and prior.exact_norm(result['defect'])<1,'exact serialized inclusion/contraction')
    with ctx.workprec(512):
        box=[stored(x) for x in result['box']];center=[stored(x) for x in result['center']]
        m=matrix(result['monodromy']);pre=matrix(result['preconditioner']);require(all(interval(x)[0]==interval(x)[1] for row in result['preconditioner'] for x in row),'point inverse')
        defect=eye(13)-pre*(m-eye(13));bound=norm_inf_upper(defect);require(bound<1,'independent36 contraction')
        image=arb_mat([[x] for x in center])-pre*matrix(result['residual'])+defect*arb_mat([[box[i]-center[i]] for i in range(13)])
        require(all(box[i].contains_interior(image[i,0]) for i in range(13)),'independent36 Krawczyk image')
        orbit=[[stored(x) for x in row] for row in result['orbit_enclosures']];require(len(orbit)==37 and all(len(row)==13 for row in orbit),'full36 phase enclosures')
        for d in ds:require(any(not (orbit[d][j]-box[j]).contains(0) for j in range(13)),'independent36 proper divisor')
        power=m;k=1;w=result['attraction_witness_power'];require(type(w)is int and 1<=w<=4096 and w&(w-1)==0,'bounded36 power')
        while k<w:power=power*power;k*=2
        norm=norm_inf_upper(power);require(norm<1 and prior.exact_norm(result['attraction_witness'])<1,'independent36 attraction')
        return {'passed':True,'divisors':ds,'contraction':ball_json(bound),'power':w,'power_bound':ball_json(norm),'scope':'Saved derivatives and Arb shared; independent interval matrix reductions, no new AD.'}

def proximity(inp,result):
    seed=[Fraction(x) for x in inp['approximation']];require(len(seed)==len(result['box']),'full box dimension');scale=max([Fraction(1)]+[abs(x) for x in seed]);d=max(max(abs(lo-x),abs(hi-x))/scale for x,(lo,hi) in zip(seed,map(interval,result['box'])));require(d<=Fraction(1,10**7),'whole-box proximity cap');return d

def execute(pinpath,pinsha,out):
    bound=sources();require(sha(pinpath)==pinsha,'external review pin JSON');pins=read(pinpath)
    require(set(pins)=={'profiles','targets'} and all(set(pins[k])=={'flip','period36'} for k in pins),'exact four producer bundles')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();ticks=0;serial=0;result={'status':'incomplete','stages':[],'kinds':{}}
    def save(name,value):
        data=canonical_bytes(value);limit=CAP if name in ('result.json','manifest.json') else CAP-2*1024**2
        require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(data)<limit,'aggregate audit size cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    def stage(name,value):save(name+'.json',value);result['stages'].append(name)
    def stop(*_):raise TimeoutError('fixed one-shot audit deadline/termination')
    def tick():
        nonlocal ticks
        ticks+=1
        if ticks>360 or time.monotonic()-start>INTERNAL:raise TimeoutError('audit wall/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored audit RSS2GiB')
    def observe(name,value):
        nonlocal serial
        save(f'evaluation-{serial:03d}-{name}.json',value);serial+=1;tick()
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'internal_seconds':INTERNAL,'tick_cap':360,'raw_cap':CAP,'pins_sha256':pinsha,'shared_kernel_authorship':'Reviewer wrote rigorous_flips.py; root separately reviewed it. New producer adapter independently reviewed here; saved reductions and stronger proof share Arb and analytic derivatives.'}
    save('runtime-start.json',runtime);save('pins.json',pins)
    for p in OWN_FILES:
        dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        for kind in ('flip','period36'):
            profile,prt,pr=bind_bundle(pins['profiles'][kind],kind,'profile')
            target,trt,tr=bind_bundle(pins['targets'][kind],kind,'target');tick()
            eligibility=read(target/'run/profile-eligibility.json')
            require(eligibility['profile']==str(profile) and eligibility['manifest_sha256']==pins['profiles'][kind]['manifest_sha256'] and eligibility['projected_seconds']==pr['projected_seconds'] and eligibility['passed'] is True,'target measured profile join')
            stage(kind+'-01-bundles',{'profile':pins['profiles'][kind],'target':pins['targets'][kind],'profile_outcome':pr,'target_outcome':tr})
            raw=target/'run';envelope=read(raw/'certificate.json');inp=read(raw/'inputs.json');proof=envelope['result']
            require(envelope['schema']=='neural-shrimp-ray51-certificate-v1' and envelope['kind']==kind and envelope['runtime']==trt,'envelope runtime/kind')
            require(envelope['inputs_sha256']==sha(raw/'inputs.json') and proof==read(raw/'mathematical-result.json'),'certificate/result/input identity')
            require(proof['settings']==ORIGINAL[kind],'fixed original settings')
            stage(kind+'-02-inputs',reconstruct(kind,inp));tick()
            replay=check_flip(inp,**ORIGINAL[kind],tick=tick,observe=observe) if kind=='flip' else check_orbit(inp,**ORIGINAL[kind])
            save(kind+'-03-replay.json',replay);tick();require(canonical_bytes(replay)==canonical_bytes(proof),'byte-identical mathematical replay');require(replay['passed'],'replayed gates pass')
            reduction=prior.saved_math(proof,inp) if kind=='flip' else orbit_math(proof)
            stage(kind+'-04-saved-arithmetic',reduction);tick()
            distance=proximity(inp,proof);require(envelope['whole_box_proximity']['passed'] is True and str(distance)==envelope['whole_box_proximity']['normalized_whole_box_distance'],'exact stored proximity')
            stage(kind+'-05-proximity',{'passed':True,'normalized_distance':str(distance),'cap':'1/10000000','scope':'Whole-box proximity only; no continuous connection or seed membership.'})
            stronger=check_flip(inp,**STRONGER[kind],tick=tick,observe=observe) if kind=='flip' else check_orbit(inp,**STRONGER[kind])
            save(kind+'-06-nested.json',stronger);tick();require(stronger['passed'],'fixed stronger gates pass');require(nested(stronger['box'],proof['box']),'strictly nested full box')
            if kind=='period36':require([d['divisor'] for d in stronger['divisors']]==[1,2,3,4,6,9,12,18],'nested primitive36 divisors')
            stage(kind+'-07-summary',{'passed':True,'byte_identical_replay':True,'original_settings':ORIGINAL[kind],'nested_settings':STRONGER[kind],'strictly_nested':True,'nested_sha256':sha(out/(kind+'-06-nested.json')),'original_statement':proof['exact_statement'],'original_criticality':proof.get('criticality'),'original_complement_power':proof.get('complement_witness_power'),'original_attraction_power':proof.get('attraction_witness_power')})
            result['kinds'][kind]={'passed':True,'byte_identical_replay':True,'strictly_nested':True}
        tick();result['status']='passed';result['scope']='Two distinct exact maps: local exponential-ray flip germ and fixed-dyadic attracting primitive36 orbit. No validated connection, whole geometry, finite-precision dynamics or utility proof.'
    except BaseException as error:result.update(status='failed_or_incomplete',error={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,evaluations_retained=serial,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=now());save('result.json',result);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values()),'within_limit':sum(e['bytes'] for e in files.values())<CAP})
    print(json.dumps(result),flush=True);return 0 if result['status']=='passed' else 2

def launch(pinpath,pinsha,out):
    bound=sources();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    args=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL}s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--pins',str(pinpath),'--pins-sha256',pinsha,'--out',str(out/'run')]
    def save(name,obj):
        payload=canonical_bytes(obj)
        with (out/name).open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':bound,'argv':args,'launcher_pid':os.getpid(),'pins_sha256':pinsha,'started_utc':now(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL,'kill_grace':5}
    save('launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);meta['owned_group_pid']=proc.pid;save('owned-process.json',{'pid':proc.pid,'argv':args});code=proc.wait(timeout=EXTERNAL+7)
        except BaseException as error:meta['error']=type(error).__name__+': '+str(error)
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    live=[]
    if proc is not None:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                fields=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(p.name))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    if live or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>CAP+1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_live_pids_remaining=live,owned_leader_exited=proc is not None and proc.poll() is not None,finished_utc=now());save('launcher-result.json',meta);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':sum(v['bytes'] for v in files.values())<CAP+1024**2})
    for f in out.rglob('*'):
        if f.is_file():f.chmod(0o444)
    for d in sorted((p for p in out.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):d.chmod(0o555)
    out.chmod(0o555);print(json.dumps({'exit_code':code,'root':str(out),'manifest':entry(out/'manifest.json')}),flush=True);return code

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pins',type=Path,required=True);parser.add_argument('--pins-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args();raise SystemExit((execute if args.execute else launch)(args.pins,args.pins_sha256,args.out))
