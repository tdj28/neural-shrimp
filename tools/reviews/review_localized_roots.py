"""Fixed saved-only review of the localization profile, floors and finite probes."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
import gc,gzip,hashlib,json,math,platform,resource,signal,subprocess,sys,time
import flint
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from flint import ctx
from neural_shrimp.rigorous_orbits import canonical_bytes
import review_joined_period18_math as M
import review_localization_math as L
SOURCE='6295b6584735762d0ab0d6b389b5b2c01bacbb1d'
BASE=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-localized-roots-2026-10-09')
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-meanvalue-2026-10-09/attempt-001-target/run')
PARENT_PIN='dbb6ea5e57606788cbf1a7659ebcff2b4e91ebad2d11ca7779b88c32e2dcc9b7'
UNITS=[('profile',SOURCE,'9b0ede50c4991806659ab7364b71197cbb12dfca3c2cd030457165f511fe9115','ea3714349c5f406edac7005ddaf877dfc74ef53181aa62f432f33ea31d26102f'),('floor','4bb62f049454ec06a79262e1c9b30ea3a927fcb5','260cc0fe095563537fa8bc179a09628b124f97ebbea4db14a6b6cea94fd99f15','d93c158266269401dde17e154c75f122fbde0b80b2ce7d186c0efbb4b81c8ce4'),('probe','839c2494ad2ff5e2489e03e584a44e437518cd42','6eb16640e673d76ee7b3c338f0eb7ed99b1fe49e374c828e13925915b9a26cc2','4f3565a939ff47b349badecf56192edd9f45c5a357c52353b01bce9d6b13137b')]
FILES=['tools/reviews/review_localized_roots.py','tools/reviews/review_localization_math.py','tools/reviews/review_joined_period18_math.py','tools/reviews/launch_localized_roots_review.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tests/reviews/test_localization_saved_review.py','docs/plans/localized-roots-independent-review.md','docs/reviews/s2-period18-localized-roots-profile.json','docs/reviews/s2-period18-localized-roots-floor.json','docs/reviews/s2-period18-localized-roots-outcome.json','docs/reviews/s2-period18-meanvalue-independent-review.json']

def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024*1024):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
        M.need(len(b)<=256*1024**2,'decompression cap');meta=json.loads(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json').read_bytes());M.need(meta=={'uncompressed_bytes':len(b),'uncompressed_sha256':hashlib.sha256(b).hexdigest()},'payload hash');return json.loads(b)
    return json.loads(p.read_bytes())

def verify(root,pin):
    M.need(entry(root/'manifest.json')['sha256']==pin,'fixed manifest');inv=read(root/'manifest.json')['files']
    files={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file() and p!=root/'manifest.json'}
    M.need(set(files)==set(inv),'complete exact inventory')
    for n,p in files.items():M.need(not p.is_symlink() and not(p.stat().st_mode&0o222) and entry(p)==inv[n],'immutable hash '+n)
    return {'files':len(inv),'bytes':sum(v['bytes']for v in inv.values()),'manifest':pin}

def validate_payloads(run,tick):
    for p in run.glob('*.json.gz'):
        meta=read(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json'));h=hashlib.sha256();size=0
        with gzip.open(p,'rb')as f:
            while b:=f.read(1024*1024):size+=len(b);M.need(size<=256*1024**2,'payload bound');h.update(b)
        M.need(meta=={'uncompressed_bytes':size,'uncompressed_sha256':h.hexdigest()},'all payload checks');tick()

def main(out):
    M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean review source')
    M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
    source={n:entry(ROOT/n)for n in FILES};out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=False;result={'passed':False,'units':[]}
    def save(name,value):
        b=canonical_bytes(value);M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<8*1024**2,'8MiB cap')
        with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def stop(*_):
        nonlocal stopped
        stopped=True
    def tick():
        M.need(not stopped and time.monotonic()-start<90,'90s budget');M.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'1.5GiB monitored RSS')
    save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':source,'producer_source':SOURCE,'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k]for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'started_unix':time.time()})
    for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
    try:
        profile_runtime=read(ROOT/'docs/reviews/s2-period18-localized-roots-profile.json')['runtime'];expected=profile_runtime['source_sha256']
        M.need(profile_runtime['flint']==flint.__FLINT_VERSION__ and profile_runtime['python_flint']==flint.__version__ and profile_runtime['python']==platform.python_version(),'original saved-arithmetic library versions')
        for n,h in expected.items():M.need(hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==h,'frozen source '+n)
        old=read(ROOT/'docs/reviews/s2-period18-meanvalue-independent-review.json');prior=Path(old['audit']['root']);verify(prior,old['audit']['outer_manifest']['sha256']);M.need(read(prior/'run/outcome.json')['audit_passed'],'existing independent parent root/primitive/join review')
        verify(PARENT,PARENT_PIN);env=read(PARENT/'certificate.json.gz');proof=env['result'];M.need(proof['gates']['connected_primitive_branch'],'known connected parent')
        keys=('gates','statement','center','predictor_slope','preconditioner','image_correction','exact_parameter_endpoints','parameter_center')
        charts=[]
        for k,c in enumerate(proof['charts']):
            binding=L.parent_image(c);save(f'parent-image-{k}.json',binding);charts.append({n:c[n]for n in keys});tick()
        model_inputs=read(PARENT/'inputs.json');del c,proof,env;gc.collect();tick()
        for mode,execution,outerpin,runpin in UNITS:
            root=BASE/('attempt-001-'+mode);run=root/'run';inv=verify(root,outerpin);verify(run,runpin);validate_payloads(run,tick)
            rt=read(run/'runtime-start.json');done=read(run/'outcome.json');launch=read(root/'launcher-result.json');M.need(rt['source_commit']==launch['source_commit']==execution,'execution identity')
            M.need(rt['source_sha256']==launch['source_sha256']==expected,'unchanged scientific sources')
            for key in ('python','python_flint','flint','numpy','scipy'):M.need(rt[key]==profile_runtime[key],'producer profile library binding')
            for n,h in expected.items():M.need(entry(run/'source'/n)['sha256']==h and hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==h,'retained execution source')
            M.need(launch['exit_code']==done['exit_code']==0 and not launch['owned_live_pids'] and 'error'not in launch,'completed processes')
            for pid in (rt['pid'],launch['owned_group_pid'],launch['launcher_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
            internal,external,calls=(20,25,126)if mode=='profile'else(60,75,216 if mode=='floor'else 180)
            for v,cap in ((done['elapsed_seconds'],internal),(launch['elapsed_seconds'],external+7),(done['max_rss_kib'],2*1024**2)):
                M.need(type(v)in(int,float) and math.isfinite(v) and 0<=v<cap,'finite resource gate')
            M.need(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10 and rt['precision']==256 and rt['refinements']==3 and rt['cpu_only'] and not rt['paid'],'runtime scope')
            M.need(done['counts']=={'map_attempted':calls,'map_returned':calls,'unit_attempted':2*calls,'unit_returned':2*calls},'complete bounded acquisition counters')
            extract=read(run/'parent-extract.json');localcharts=extract['charts']
            if mode!='profile':
                M.need(localcharts==charts and extract['parent_manifest_sha256']==PARENT_PIN and read(run/'inputs.json')==model_inputs,'known parent and exact model/native anchor identity')
                M.need(read(run/'cost-gate.json')['manifest_sha256']==UNITS[0][3],'profile dependency')
            if mode=='probe':M.need(read(run/'floor-gate.json')=={'manifest_sha256':UNITS[1][3],'passed':True},'floor dependency')
            complete=sorted(run.glob('stage-*-completed_probe.json.gz'));M.need(len(complete)==len(localcharts)==(1 if mode=='profile'else 2),'complete fixed slots')
            rows=[]
            for k,path in enumerate(complete):
                record=read(path);M.need(record['chart']==k,'native chart index');loc=record['result']['localization'];ends=loc['exact_endpoints']
                if mode in ('profile','floor'):M.need(ends==[localcharts[k]['parameter_center']]*2,'exact fixed floor')
                else:
                    lo,hi=map(Fraction,localcharts[k]['exact_parameter_endpoints']);width=Fraction(1,32768);M.need(ends==([str(lo),str(lo+width)]if k==0 else[str(hi-width),str(hi)]),'fixed outer finite slice')
                check=L.audit_localization(localcharts[k],loc,ends);metric=M.audit_periodic(record['result']['stability']);M.need(record['result']['passed'],'reported qualified metric')
                # Bind duplicated acquisition records and exact native order.
                serial=int(path.name.split('-')[1]);derivative=read(run/f'stage-{serial-2:03d}-localized_derivatives.json.gz')
                M.need(derivative['phase_jacobians']==record['result']['stability']['interval_J'],'same returned native Jacobians')
                for field in ('exact_endpoints','final_correction','initial_correction'):M.need(done['probes'][k][field]==(ends if field=='exact_endpoints'else loc[field]),'terminal summary '+field)
                M.need(done['probes'][k]['localized_passed'] and done['probes'][k]['refinement_count']==3,'terminal qualification')
                M.need(done['probes'][k]['baseline_passed']==(None if record['baseline']is None else record['baseline']['passed']),'baseline label binding')
                row={'chart':k,'containment':check,'metric':metric,'baseline_passed':None if record['baseline']is None else record['baseline']['passed']};save(f'{mode}-chart-{k}.json',row);rows.append({'chart':k,'passed':True,'exact_endpoints':ends,'baseline_passed':row['baseline_passed']});del record,loc,metric,check,derivative;gc.collect();tick()
            result['units'].append({'mode':mode,'execution':execution,'inventory':inv,'probes':rows});save('progress.json',result) if not(out/'progress.json').exists()else None
        result.update(passed=True,scope='known-parent containment and selected cyclic stability only; no full1/256 coverage or boundary connection; analytic derivative acquisitions and Arb remain shared trust')
    except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);inv={str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv,'total_bytes':sum(v['bytes']for v in inv.values())})
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        out.chmod(0o555)
    print(json.dumps(result),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
