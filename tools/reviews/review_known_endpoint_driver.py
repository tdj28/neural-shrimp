"""One fixed saved384-bit independent native705 reduction; zero native calls."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_unit_pair_pal_math as A
import review_known_endpoint_math as E
from fractions import Fraction as F
from flint import ctx
need=A.need
OUT=Path('/data2/neural-shrimp-data/audits/native705-endpoint-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/attempt003-target')
PINS=dict(source='b03d990f5d5801db5addd6cd2c18e718bd4ba8ad',outer='a9af91fe221aa4f9dfaadd9b1fbb8d29ef50f84832a404d8780655e2e84f62aa',worker='2b5344c9a2f1a699abadceb837ed9c75674f24ac14292c3c2493d7779e49734d')
FILES=['tools/reviews/review_known_endpoint_math.py','tools/reviews/review_known_endpoint_driver.py','tools/reviews/launch_known_endpoint_review.py','tests/test_known_endpoint_review.py','docs/plans/native705-endpoint-independent.md']
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
        save('runtime.json',dict(source_commit=source,sources=identities,precision=480,model_calls=0,derivative_calls=0,python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,pid=os.getpid(),threads=dict.fromkeys(THREADS,'1')))
        endpoint=PARENT.parent/'endpoint-localization/attempt001';epins=dict(source='5f6c0a8a',outer='73ea6f9056eece1179b48edca07839121eb8b5b23fe5b8cdd1093b773e9a5e8e',worker='09fb08ac9f7dc7d2b22120f53b0656c52ab5f7975a31efe3f18c3bd18e227cac')
        inventory(endpoint,epins['outer']);inventory(endpoint/'run',epins['worker']);rt=read(endpoint/'run/runtime.json');launch=read(endpoint/'launcher-result.json');out=read(endpoint/'run/outcome.json')
        need(rt['commit'].startswith(epins['source'])and launch['commit']==rt['commit']and launch['exit_code']==0 and launch['owned_live_pids']==[]and out['passed']and out['ticks']==144,'exact completed endpoint source')
        for n,v in rt['sources'].items():need(entry(endpoint/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical endpoint source '+n)
        inp=read(endpoint/'run/inputs.json');binding=inp['binding'];old=Path(binding['producer']['root']);audit=Path(binding['independent']['root'])
        for root,pins in [(old,binding['producer']),(audit,binding['independent'])]:inventory(root,pins['outer']);inventory(root/'run',pins['worker'])
        original=read(old/'run/certificate.json.gz');independent=read(audit/'run/native704-384.json');oldout=read(audit/'run/outcome.json')
        need(oldout['passed']and independent['passed']and independent['inclusions']==independent['anchor_coordinates']==704,'known connected704 exact roots')
        need(entry(audit/'run/native704-384.json')==binding['independent_proof'],'literal prior independent proof')
        known=independent['recomputed_endpoints'][1];need(len(known)==704 and original['parameter_endpoints']==['0','1/32768'],'fixed known endpoint')
        for outer,inner in zip(inp['known_endpoint'][:-1],known,strict=True):A.U.contains(outer,inner,'producer endpoint input covers existing same root')
        need(inp['map_inputs']==read(old/'run/inputs.json')['map_inputs']==read(PARENT/'run/inputs.json')['map_inputs'],'exact physical map throughout')
        ci=read(endpoint/'run/point-2.json.gz');point=[A.M.point(v)for v in ci['center']];need(len(point)==704,'all refined center coordinates')
        dp=sorted((endpoint/'run/events').glob('*-direction_return.json.gz'));wp=sorted((old/'run/events').glob('*-direction_return.json.gz'));need(len(dp)==144 and len(wp)==216,'complete direction inventories')
        tick()
        with ctx.workprec(480):
            t=A.R.exact(F(1,32768));need(A.M.point(inp['parameter'])==A.M.point(ci['t'])==F(1,32768),'same exact endpoint t')
            c=[A.R.exact(v)for v in point];G,_,_=A.U.rebuild(c,[read(p)for p in dp[108:]],13,18,t)
            whole=original['whole_box'];W=[A.R.original_ball(v)for v in whole]
            # Rebuild the original whole-tube derivative with the independently
            # established complete parameter interval [0,1/32768].
            from flint import arb
            ti=arb(A.R.exact(F(1,65536)),A.R.exact(F(1,65536)))
            _,J,_=A.U.rebuild(W,[read(p)for p in wp[180:]],13,18,ti);tick()
            pre=A.R.matrix(original['preconditioner']);weights=[A.R.exact(F(v))for v in original['weights']]
            localized=E.localize(pre,J,G,c,known,whole,weights);save('independent-localization.json',localized);tick()
            del pre,J,original,W;gc.collect()
            # The independent705 uniform proof supplies uniqueness; the new
            # known-root box supplies existence on precisely that same curve.
            independent705=Path('/data2/neural-shrimp-data/audits/native705-independent/attempt001');inventory(independent705,'9177a2a3c17b5886a0c3f08d831dcb22822fe09f3e15ec278cde411367414f76')
            uniform=read(independent705/'run/outcome.json');ub=read(independent705/'run/input-binding.json');need(uniform['passed']and uniform['uniform_primitive_positive_unit_pair']and uniform['inclusions']==705 and ub['pins']==PINS and ub['root']==str(PARENT),'independent identical705 uniform chart')
            inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker']);chart=read(PARENT/'run/certificate.json.gz')
            center,slope,curve=[[A.R.exact(A.M.point(v))for v in chart[k]]for k in ['center','slope','curvature']];normal=[A.R.exact(F(v))for v in chart['normal']];origin=[A.R.exact(F(v))for v in chart['origin']];radii=[A.R.exact(A.M.point(v))for v in chart['coordinate_radii']]
            box=localized['box']+[A.ball_json(t)];join=A.anchor_membership(box,center,slope,curve,normal,origin,A.R.exact(F(chart['parameter_center'])),A.R.exact(F(2)**chart['settings']['halfwidth_exponent']),radii);save('independent-membership.json',join);tick();need(join['passed']and join['coordinates']==705,'strict full705 same-root connection')
            save('input-binding.json',dict(endpoint=str(endpoint),endpoint_pins=epins,prior704=binding,prior705=ub,map_inputs=inp['map_inputs']))
            result.update(passed=True,connected=True,anchor_coordinates=705,weighted_bound=localized['weighted_bound'],parameter_endpoints=chart['parameter_endpoints'],scope='Independent480-bit saved native704 residual/whole-Jacobian reconstruction, known-root Neumann enclosure and full705 strict membership in separately independently proved uniform chart. Shared analytic acquisitions/Arb. Same primitive18 positive-unit-pair branch; complement/criticality/global topology separate.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks,native_calls=0);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
