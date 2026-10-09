"""One bounded wider-chart replay and independent joined-branch/stability review."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,importlib.util,json,math as pymath,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import flint,numpy,scipy
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,matrix_json
from neural_shrimp.rigorous_connection import HiddenRateMap
from neural_shrimp.rigorous_connection_frames import two_charts
import review_joined_period18_math as math
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-framed-2026-10-09/attempt-002-target/run')
PARENT_PIN='b7fcf410a9f806f21dfa3d1bd650bb0f93782b0e30294cba0ed4b4a1c2efd270';PARENT_CERT='2df4a3c51362350bb53dce07f50dd5fc0ec796a82bb4de0156e33eea992a1a58'
PARENT_OUTER='8d21dd9921a218d0eb98172a653c83d06bd234e3c760f6d5698d39aec84bce48'
PARENT_COMMIT='5b4ce6d1eed6cc30e000ec4860e8efab61855eef'
RECEIPT='certificates/s2-period18-receipt.json';RECEIPT_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
SETTINGS=dict(precision=256,halfwidth_exponent=-12,radius_exponent=-20,newton_steps=2,max_power=4096)
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py','tools/certify_s2_period18_connection_frames.py',RECEIPT,'tools/reviews/review_joined_period18_math.py','tools/reviews/review_joined_period18_branch.py','tests/reviews/test_joined_period18_math.py','docs/plans/joined-period18-independent-review.md']
PARENT_FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_connection_frames.py','tools/certify_s2_period18_connection_frames.py','tests/test_rigorous_connection.py','tests/test_rigorous_connection_frames.py','docs/plans/s2-period18-connection-framed-pilot.md',RECEIPT]
SLICE_FILES=PARENT_FILES+['src/neural_shrimp/rigorous_periodic_lyapunov.py','tests/test_rigorous_periodic_lyapunov.py','tools/certify_s2_period18_periodic_metric.py','docs/reviews/s2-period18-connection-framed-outcome.json','docs/plans/s2-period18-periodic-metric-pilot.md','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/rigorous_metric_slices.py','tests/test_rigorous_metric_slices.py','tools/certify_s2_period18_sliced_metric.py','docs/plans/s2-period18-sliced-metric-pilot.md']
SLICE_SOURCE='973b8a70554a69bbbed9097b483961153d198c7f'
SLICED_ROOT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-sliced-metric-2026-10-09/attempt-001-target/run')
SLICED_PIN='62dc7bb7eb51510645042cd3eb7ed00907cd37b276374dc5750020508d6a36cb'
SLICED_OUTER='e73ba7a2b4d2946d4a938622eb33b5cb5c70aae09b77518e31738da9c769e2cb'
SLICED_COMMIT='71bcfa97a565d40ca209a68a90128f329ba96966'
CAP=64*1024**2;INTERNAL=120;EXTERNAL=135

def need(v,m):
    if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ent(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb') as f:data=f.read(256*1024**2+1)
        need(len(data)<=256*1024**2,'bounded decompression')
        meta=json.loads(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json').read_bytes());need(len(data)==meta['uncompressed_bytes'] and hashlib.sha256(data).hexdigest()==meta['uncompressed_sha256'],'decompressed identity')
        return json.loads(data)
    return json.loads(p.read_bytes())
def inventory(p):return {str(f.relative_to(p)):ent(f) for f in sorted(p.rglob('*')) if f.is_file() and f!=p/'manifest.json'}
def verify(p,pin):
    need(p.is_dir() and not p.is_symlink() and sha(p/'manifest.json')==pin,'fixed manifest pin')
    need(not any(f.is_symlink() for f in p.rglob('*')),'no artifact symlinks');m=read(p/'manifest.json');need(inventory(p)==m['files'],'complete exact inventory')
    if 'total_bytes'in m:need(m['total_bytes']==sum(x['bytes'] for x in m['files'].values()),'inventory totals')
    return m

def sources():
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed checkout')
    out={}
    for p in FILES:
        data=(ROOT/p).read_bytes();need(data==subprocess.check_output(['git','show','HEAD:'+p],cwd=ROOT),'committed reviewer source');out[p]=hashlib.sha256(data).hexdigest()
    return out

def bind_runtime(folder,commit,expected_files):
    runtime=read(folder/'runtime-start.json');need(runtime['source_commit']==commit,'fixed execution commit')
    need(set(runtime['source_sha256'])==set(expected_files),'full expected source inventory')
    for p,h in runtime['source_sha256'].items():
        need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest()==h and sha(folder/'source'/p)==h,'historical committed/retained source '+p)
    need(runtime['threads']=={k:'1' for k in THREADS} and runtime['nice']>=10,'producer CPU1/nice contract')
    need(runtime['python_flint']==flint.__version__ and runtime['flint']==flint.__FLINT_VERSION__,'same Arb implementation')
    return runtime

def bounded(value,upper,name):
    need(type(value) in (int,float) and pymath.isfinite(value) and 0<=value<=upper,name)

def resource_metadata(folder,runtime,expected_exit,raw_cap):
    outer=folder.parent;start=read(outer/'launcher-start.json');end=read(outer/'launcher-result.json');outcome=read(folder/'outcome.json')
    for obj in (start,end):
        need(obj['source_commit']==runtime['source_commit'] and obj['source_sha256']==runtime['source_sha256'] and obj['mode']=='target' and obj['cpu_only'] is True,'launcher execution/source identity')
        need(obj['external_seconds']==135 and obj['kill_grace_seconds']==5,'producer external limits')
    need(start['argv']==end['argv'] and start['owned_group_pid']==end['owned_group_pid'],'owned command/group identity')
    need(end['exit_code']==outcome['exit_code']==expected_exit and end['owned_live_pids']==[] and 'error' not in end,'producer terminal status')
    need(runtime['cpu_only'] is True and runtime['paid'] is False and runtime['internal_seconds']==120,'producer local resource contract')
    bounded(outcome['elapsed_seconds'],120,'producer internal elapsed');bounded(end['elapsed_seconds'],142,'producer external elapsed');bounded(outcome['max_rss_kib'],2*1024**2,'producer monitored RSS')
    need(sum(x['bytes'] for x in inventory(outer).values())<raw_cap+1024**2,'producer total storage bound')
    # Recorded PID identities can be reused later. Reject only a still-live PID
    # whose cmdline points to this exact exclusive attempt, never kill it.
    for pid in {runtime['pid'],runtime['ppid'],start['launcher_pid'],start['owned_group_pid']}:
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:continue
        need(str(outer).encode() not in cmd,'producer owned attempt still active')
    need(all(not (f.stat().st_mode&0o222) for f in outer.rglob('*')) and not(outer.stat().st_mode&0o222),'immutable producer artifact tree')
    return {'launcher_start':ent(outer/'launcher-start.json'),'launcher_result':ent(outer/'launcher-result.json'),'process_log':ent(outer/'process.log'),'exit_code':expected_exit,'elapsed_seconds':outcome['elapsed_seconds'],'launcher_elapsed_seconds':end['elapsed_seconds'],'max_rss_kib':outcome['max_rss_kib'],'owned_attempt_absent':True}

def expected_coverage(charts):
    result=[]
    for chart in charts:
        left,right=map(Fraction,chart['exact_parameter_endpoints']);step=(right-left)/16
        result.append([[str(left+k*step),str(left+(k+1)*step)] for k in range(16)])
    return result

def original_inputs():
    need(sha(ROOT/RECEIPT)==RECEIPT_SHA,'canonical original anchor')
    receipt=read(ROOT/RECEIPT);base=Path(receipt['external_root']);files={**receipt['files'],'receipt.json':ent(ROOT/RECEIPT)}
    actual={str(p.relative_to(base)):ent(p) for p in base.rglob('*') if p.is_file()};need(actual==files,'complete original anchor inventory')
    original=read(base/'certificate.json')['result'];model=read(base/'inputs.json')
    need(receipt['passed'] and original['passed'] and receipt['statement']==original['exact_statement'],'previously proved anchor')
    need(model['period']==18 and model['lr_hidden']=='1254985102288567/17592186044416' and model['lr_output']=='1848862009810323/2251799813685248','original exact rates/period')
    need(sha(base/'inputs.json')=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','consumed original model/data')
    phases=original['orbit_enclosures'][:18];initial=[[str(math.pair(v['mid'])) for v in row] for row in phases]
    inp=read(PARENT/'inputs.json');need(inp['model_inputs']==model and inp['initial']==initial and inp['rate']==model['lr_hidden'] and inp['anchor']=={'rate':model['lr_hidden'],'phase_boxes':phases},'exact original phase/input identity')
    return inp,{'receipt_sha256':RECEIPT_SHA,'original_certificate':ent(base/'certificate.json'),'original_inputs':ent(base/'inputs.json'),'native_phases':18,'scope':'Bind the already independently reviewed center certificate; do not rerun the earlier initial tube.'}

def actual_slice(model,chart,left,right):
    lo,hi=Fraction(left),Fraction(right);h=Fraction(chart['parameter_center']);center=(lo+hi)/2;width=(hi-lo)/2
    def ar(x):return arb(x.numerator)/x.denominator
    delta=arb(ar(center-h),ar(width));rate=arb(ar(center),ar(width));base=[math.stored(x) for row in chart['center'] for x in row];slope=math.mat(chart['predictor_slope']);image=math.mat(chart['image_correction'])
    boxes=[base[i]+slope[i,0]*delta+image[i,0] for i in range(234)];blocks=[]
    for j in range(18):_,J,_=model.step(boxes[j*13:(j+1)*13],rate);blocks.append(J)
    return boxes,rate,blocks

def run(out,sliced,sliced_pin,sliced_commit):
    need(sliced==SLICED_ROOT and sliced_pin==SLICED_PIN and sliced_commit==SLICED_COMMIT,'fixed target invocation');bound=sources();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();tick_count=0;model_calls=0;stopped=False;stages=[];result={'passed':False,'status':'started'}
    def save(name,obj):
        data=canonical_bytes(obj);need(len(data)<256*1024**2,'decompressed output cap')
        if name.endswith('.gz'):data=gzip.compress(data,compresslevel=6,mtime=0)
        cap=CAP if name in ('outcome.json','manifest.json') else CAP-4*1024**2
        need(sum(f.stat().st_size for f in out.rglob('*') if f.is_file())+len(data)<cap,'aggregate review cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    def tick():
        nonlocal tick_count
        tick_count+=1
        if stopped or time.monotonic()-start>INTERNAL or tick_count>400:raise TimeoutError('review wall/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2*1024**2:raise MemoryError('monitored2GiB RSS')
    def stop(*_):
        nonlocal stopped
        stopped=True
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'started_utc':datetime.now(timezone.utc).isoformat(),'sliced_root':str(sliced),'sliced_manifest':sliced_pin,'sliced_commit':sliced_commit,'internal_seconds':INTERNAL,'byte_cap':CAP,'cpu_only':True}
    save('runtime-start.json',runtime)
    for p in FILES:q=out/'source'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/p).read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,INTERNAL);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    try:
        verify(PARENT.parent,PARENT_OUTER);verify(PARENT,PARENT_PIN);need(sha(PARENT/'certificate.json.gz')==PARENT_CERT,'fixed wider certificate')
        parent=read(PARENT/'certificate.json.gz');pr=bind_runtime(PARENT,PARENT_COMMIT,PARENT_FILES)
        need(parent['runtime']==pr and parent['inputs_sha256']==sha(PARENT/'inputs.json'),'parent envelope binding')
        for p,h in pr['source_sha256'].items():need(sha(ROOT/p)==h,'unchanged wider-chart source '+p)
        for p in ['src/neural_shrimp/models.py','src/neural_shrimp/controls.py']:need((ROOT/p).read_bytes()==subprocess.check_output(['git','show',PARENT_COMMIT+':'+p],cwd=ROOT),'bind implicit original import')
        need(pr['numpy']==numpy.__version__ and pr['scipy']==scipy.__version__,'same numerical frame proposals for exact replay')
        inp,anchor=original_inputs();save('original-anchor-binding.json',anchor);save('inputs.json',inp);tick()
        old=parent['result'];need(old['gates']['connected_primitive_branch'] and len(old['charts'])==2,'parent joined branch')
        save('parent-resource-binding.json',resource_metadata(PARENT,pr,2,64*1024**2))
        oldbytes=canonical_bytes(old);old_identity={'bytes':len(oldbytes),'sha256':hashlib.sha256(oldbytes).hexdigest()};del oldbytes,old,parent;gc.collect()
        # Replaying the same wider charts binds their precise saved Kimage and
        # derivative acquisition; it is not the old narrow initial tube.
        with ctx.workprec(256):replayed=two_charts(HiddenRateMap(inp['model_inputs']),inp,**SETTINGS,tick=tick)
        save('wider-replay.json.gz',replayed);data=canonical_bytes(replayed);need({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==old_identity,'byte-identical canonical wider-chart replay including Kimage');del data
        save('wider-replay-binding.json',{'passed':True,'canonical_result':old_identity,'original_compressed_certificate':PARENT_CERT})
        stages.append('wider_byte_replay');old=replayed;del replayed;gc.collect()
        joined=math.audit_join(old['charts'],inp,SETTINGS);save('independent-join.json.gz',joined);stages.append('independent_root_primitive_anchor_join');tick()
        verify(sliced.parent,SLICED_OUTER);verify(sliced,sliced_pin);sr=bind_runtime(sliced,sliced_commit,SLICE_FILES);
        for p,h in sr['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',SLICE_SOURCE+':'+p],cwd=ROOT)).hexdigest()==h,'frozen slice scientific source '+p)
        save('slice-resource-binding.json',resource_metadata(sliced,sr,0,256*1024**2));need(sr['mode']=='target' and sr['parent_manifest']==PARENT_PIN and sr['parent_certificate']==PARENT_CERT,'slice parent/source identity')
        whole=read(sliced/'result.json');need(whole['passed'] and len(whole['units'])==32 and whole['parent_certificate_sha256']==PARENT_CERT,'all32 complete stability units')
        outcome=read(sliced/'outcome.json');need(outcome['passed'] and outcome['attempted_model_steps']==outcome['returned_model_steps']==576 and outcome['completed_units']==32,'complete original native acquisitions')
        need(whole['coverage']==expected_coverage(old['charts']) and whole['connected_parameter_endpoints']==joined['exact_interval'],'exact declared coverage and connected endpoints')
        summaries=[]
        with ctx.workprec(384):
            model=HiddenRateMap(inp['model_inputs'])
            for number,row in enumerate(whole['units']):
                j,k=divmod(number,16);chart=old['charts'][j];left,right=map(Fraction,chart['exact_parameter_endpoints']);step=(right-left)/16;ends=[str(left+k*step),str(left+(k+1)*step)]
                need(row['chart']==j and row['index']==k and row['kind']=='slice' and row['exact_endpoints']==ends and row['passed'],'exact native slice order/closed coverage')
                need(ent(sliced/row['file'])==row['identity'],'complete slice file identity');record=read(sliced/row['file']);need(record['unit']=={key:row[key] for key in ['chart','kind','index','exact_endpoints']},'saved unit identity')
                content=record['record'];domain=content['domain'];need(content['passed'] and domain['exact_endpoints']==ends and not domain['floor_only'],'closed parameter domain')
                r0,r1=math.interval(domain['parameter_interval']);need(r0<=Fraction(ends[0])<=Fraction(ends[1])<=r1,'saved parameter interval covers exact slice')
                boxes,rate,blocks=actual_slice(model,chart,*ends);model_calls+=18
                # Exact rational domain check does not infer containment merely
                # from overlap or from the producer's positivity flag.
                h=Fraction(chart['parameter_center']);base=[math.point(x) for vals in chart['center'] for x in vals];slopes=[math.point(vals[0]) for vals in chart['predictor_slope']]
                for phase in range(18):
                    for i in range(13):
                        z=phase*13+i;e0,e1=math.interval(chart['image_correction'][z][0]);v0,v1=sorted([slopes[z]*(Fraction(ends[0])-h),slopes[z]*(Fraction(ends[1])-h)]);lo,hi=math.interval(domain['phase_boxes'][phase][i]);need(lo<=base[z]+v0+e0 and base[z]+v1+e1<=hi,'saved slice phase domain contains proved root range')
                need(len(domain['phase_jacobians'])==18 and domain['phase_jacobians']==content['proof']['interval_J'],'acquisition/witness exact derivative join')
                witness=dict(content['proof'],interval_J=[matrix_json(x) for x in blocks]);verified=math.audit_periodic(witness)
                save(f'slice-{number:02d}-independent.json.gz',{'chart':j,'slice':k,'exact_endpoints':ends,'phase_boxes':[[ball_json(x) for x in boxes[a*13:(a+1)*13]] for a in range(18)],'parameter_interval':ball_json(rate),'phase_jacobians':[matrix_json(x) for x in blocks],'positivity':verified})
                summaries.append({'chart':j,'slice':k,'passed':True,'exact_endpoints':ends});del record,content,domain,witness,verified,boxes,blocks;gc.collect();tick()
        save('stability-summary.json',{'passed':True,'native_derivatives_recomputed':model_calls,'units':summaries,'exact_interval':joined['exact_interval']});stages.append('all32_independent_derivative_and_metric_checks');result.update(passed=True,status='joined_attracting_branch_review_passed',exact_interval=joined['exact_interval'],scope='Uniform attracting primitive18 branch on the two original exact hidden-rate charts; no fold/flip or full-window connection.')
    except BaseException as exc:result.update(status='failed_or_incomplete',error={'type':type(exc).__name__,'message':str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(stages=stages,ticks=tick_count,slice_model_steps=model_calls,elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=datetime.now(timezone.utc).isoformat());save('outcome.json',result);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values())})
    print(json.dumps(result),flush=True);return 0 if result['passed'] else 2

def launch(out,sliced,pin,commit):
    sources();need(sliced==SLICED_ROOT and pin==SLICED_PIN and commit==SLICED_COMMIT,'fixed launch invocation');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
    def save(name,obj):
        with (out/name).open('xb') as f:f.write(canonical_bytes(obj));f.flush();os.fsync(f.fileno())
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{EXTERNAL}s','nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--out',str(out/'run'),'--sliced',str(sliced),'--sliced-manifest',pin,'--sliced-commit',commit]
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'argv':cmd,'launcher_pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'load_average':os.getloadavg(),'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':EXTERNAL,'kill_grace_seconds':5};save('launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'pid':proc.pid});code=proc.wait(timeout=EXTERNAL+7)
        except BaseException as exc:meta['error']=type(exc).__name__+': '+str(exc)
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
                f=(p/'stat').read_text().rsplit(')',1)[1].split()
                if int(f[2])==proc.pid and f[0]!='Z':live.append(int(p.name))
            except (OSError,ValueError):pass
    if live or sum(e['bytes'] for e in inventory(out).values())>CAP+1024**2:code=2
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live,finished_utc=datetime.now(timezone.utc).isoformat());save('launcher-result.json',meta);files=inventory(out);save('manifest.json',{'files':files,'total_bytes':sum(e['bytes'] for e in files.values())})
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps({'out':str(out),'exit_code':code,'manifest':ent(out/'manifest.json')}),flush=True);return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sliced',type=Path,required=True);p.add_argument('--sliced-manifest',required=True);p.add_argument('--sliced-commit',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((run if a.execute else launch)(a.out,a.sliced,a.sliced_manifest,a.sliced_commit))
