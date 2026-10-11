"""One fixed saved384-bit independent native705 reduction; zero native calls."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_unit_pair_pal_math as A
need=A.need
OUT=Path('/data2/neural-shrimp-data/audits/native705-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/attempt003-target')
PINS=dict(source='b03d990f5d5801db5addd6cd2c18e718bd4ba8ad',outer='a9af91fe221aa4f9dfaadd9b1fbb8d29ef50f84832a404d8780655e2e84f62aa',worker='2b5344c9a2f1a699abadceb837ed9c75674f24ac14292c3c2493d7779e49734d')
FILES=['tools/reviews/review_unit_pair_pal_math.py','tools/reviews/review_unit_pair_pal_driver.py','tools/reviews/launch_unit_pair_pal_review.py','tests/test_unit_pair_pal_review.py','docs/plans/native705-independent.md']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
        need(len(b)<=256*1024**2,'bounded256MiB decompression');return json.loads(b)
    return json.loads(p.read_bytes())
def inventory(root,pin):
    need(entry(root/'manifest.json')['sha256']==pin,'literal manifest');m=read(root/'manifest.json')['files']
    need(set(m)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete raw inventory')
    for n,v in m.items():need(entry(root/n)==v,'immutable file '+n)

def main():
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed nice10 source')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;ticks=0;result=dict(passed=False)
    def stop(*_):raise TimeoutError('240s independent705 deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,240)
    def tick():
        nonlocal ticks
        ticks+=1
        if time.monotonic()-start>=240:raise TimeoutError('240s independent705 deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*1024**2:raise MemoryError('4GiB independent705 RSS')
    def save(name,value,closing=False):
        nonlocal used
        data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        need(used+len(data)<(64 if closing else 60)*1024**2,'64MiB raw cap');used+=len(data);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identities={n:entry(ROOT/n)for n in FILES}
        for n,v in identities.items():
            need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed dependency');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
        save('runtime.json',dict(source_commit=source,sources=identities,precision=384,model_calls=0,derivative_calls=0,python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,pid=os.getpid(),threads=dict.fromkeys(THREADS,'1')))
        inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker']);rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');done=read(PARENT/'run/outcome.json')
        need(rt['commit']==PINS['source']and la['commit']==PINS['source']and la['owned_live_pids']==[],'exact completed producer ownership')
        need(done['ticks']==253 and done['events']['direction_return']==144 and done['events']['pal_path_phase_return']==108 and done['strict_inclusions']==705,'complete source acquisition')
        need(done['gates']['uniform_full_native_unit_pair_locus']and not done['gates']['anchor_identified'],'producer uniform proof and retained connection gap')
        for n,v in rt['sources'].items():need(entry(PARENT/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',PINS['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical archived source '+n)
        tick();cf=PARENT/'run/certificate.json.gz';need(entry(cf)['sha256']=='9681483730377a352327474a41d1bb085a11e9470f7217954a8b87f94bdc033f','fixed complete certificate')
        directions=sorted((PARENT/'run/events').glob('*-direction_return.json.gz'));paths=sorted((PARENT/'run/events').glob('*-pal_path_phase_return.json.gz'))
        need(len(directions)==144 and len(paths)==108,'all direction/path files')
        save('input-binding.json',dict(root=str(PARENT),pins=PINS,certificate=entry(cf),events={str(p.relative_to(PARENT/'run')):entry(p)for p in directions+paths},scope='Uniform chart only. Original coarse anchor retained to record failed membership; no connected result inferred.'))
        chart=read(cf);ci=read(PARENT/'run/chart-inputs.json')
        need(chart['weights']==ci['weights']and chart['normal']==ci['normal']and chart['origin']==ci['origin'],'same exact geometry proposal')
        retain={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','normal','origin','center','slope','curvature','weights','coordinate_radii','preconditioner','whole_box'}
        for k in list(chart):
            if k not in retain:del chart[k]
        gc.collect();tick();proof=A.check_chart(chart,[read(p)for p in directions],[read(p)for p in paths],ci['anchor'],384,tick);save('native705-384.json',proof);tick()
        uniform=all(v for k,v in proof['gates'].items()if k!='anchor_identified');need(uniform and proof['inclusions']==705,'independent full uniform705 existence')
        result.update(passed=True,uniform_primitive_positive_unit_pair=True,connected=proof['gates']['anchor_identified'],inclusions=705,contraction=proof['contraction'],anchor=proof['anchor'],scope=proof['scope'])
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks,native_calls=0);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
