#!/usr/bin/env python3
"""One synthetic-only470D quadratic cost screen; no target mode or raw input."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import gzip,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import profile_eigenpair_locus as P
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube,path_coefficients
from neural_shrimp.rigorous_eigenpair_locus import phase,require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,matrix_json
SOURCES=P.SOURCES+['src/neural_shrimp/rigorous_eigenpair_path.py','tests/test_rigorous_eigenpair_path.py',
 'tools/profile_eigenpair_path.py','tools/launch_eigenpair_path_profile.py','docs/plans/eigenpair-quadratic-path.md',
 'docs/reviews/eigenpair-locus-weighted-outcome.json']

def projection(charts,mixed,path,peak,raw):
    require(len(charts)==2 and len(mixed)==len(path)==12,'profile count')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in charts+mixed+path),'finite profile durations')
    require(type(peak) is int and peak>0 and type(raw) is int and raw>0,'profile resources')
    seconds=2*(sum(charts)+162*max(mixed)+108*max(path))+10
    # Actual affine two-chart acquisition gives a separate conservative floor.
    measured_floor=2*48.10717263398692+10
    seconds=max(seconds,measured_floor)
    rss=2*peak+128*1024;storage=max(4*raw+8*1024**2,2*132701277)
    return {'two_chart_seconds':seconds,'synthetic_seconds':2*(sum(charts)+162*max(mixed)+108*max(path))+10,
            'affine_actual_time_floor':measured_floor,'rss_kib':rss,'raw_bytes':storage,
            'admitted':seconds<120 and rss<2*1024**2 and storage<256*1024**2,
            'scope':'heuristic resource screen; neither target authorization nor scientific pass'}


def main(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];total=[0];serial=[0];counts={}
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def save(name,value,final=False):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);before=total[0];cap=(256 if final else 248)*1024**2
        try:
            with p.open('xb') as raw:
                h=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
                try:
                    parts=[];size=0
                    for part in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
                        parts.append(part);size+=len(part)
                        if size>=65536:h.write(''.join(parts).encode());parts=[];size=0;require(before+raw.tell()<cap,'profile raw cap')
                    if parts:h.write(''.join(parts).encode())
                finally:
                    if h is not raw:h.close()
                raw.flush();os.fsync(raw.fileno())
            require(before+p.stat().st_size<cap,'profile raw cap')
        finally:
            if p.exists():total[0]=before+p.stat().st_size
    def tick():
        ticks[0]+=1
        if stop[0] or time.monotonic()-start>=120 or ticks[0]>300:raise TimeoutError('fixed profile time/operation cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('profile monitored2GiB RSS')
    def observe(kind,value):
        counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',value);serial[0]+=1
        if stop[0] or time.monotonic()-start>=120:raise TimeoutError('deadline after saved return')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'pid':os.getpid(),'mode':'synthetic_only','sources':{name:P.entry(ROOT/name) for name in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'precision':256,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in THREADS},'time_cap':120,'tick_cap':300,'rss_cap_kib':2*1024**2,'hard_virtual_address_bytes':4*1024**3})
    for name in SOURCES:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());total[0]+=p.stat().st_size
    result={'passed':False,'target_acquired':False,'chart_times':[],'mixed_times':[],'path_times':[]}
    try:
        with ctx.workprec(256):
            initial,settings=P.fixture();save('synthetic-inputs.json',{'initial':initial,'settings':{k:v for k,v in settings.items() if k!='anchor'},'known_root':'repeated primitive2; primitive18 should fail'})
            for slot in range(2):
                before=time.monotonic();r,_=quadratic_tube(P.Synthetic13(),initial,**settings,tick=tick,observe=observe);result['chart_times'].append(time.monotonic()-before)
                save(f'chart-{slot}.json.gz',r);require(r['gates']['uniform_normalized_eigenpair_locus'] and not r['gates']['primitive_period'],'synthetic root/primitive control')
                del r
            inp,x,q,rates=P.gd_fixture();save('synthetic-gd-inputs.json',inp);model=TwoRateCuspMap(inp)
            z=x+q+rates;s=[exact(str(Fraction(i-14,128))) for i in range(28)];t=[exact(str(Fraction(7-i,256))) for i in range(28)];normal=[arb(0)]*27+[arb(1)];d=arb(0,arb((1,-14)))
            for slot in range(12):
                tick();before=time.monotonic();v,b,e=phase(model,x,q,rates);result['mixed_times'].append(time.monotonic()-before)
                save(f'gd-mixed-{slot}.json.gz',{'value':matrix_json(v),'block':matrix_json(b),'rates':matrix_json(e)})
                before=time.monotonic();c=path_coefficients(model,z,s,t,d,1,1,normal,z,arb(0),tick,observe);result['path_times'].append(time.monotonic()-before);save(f'gd-path-{slot}.json.gz',{'coefficients':matrix_json(c)})
            result['projection']=projection(result['chart_times'],result['mixed_times'],result['path_times'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,total[0]);result['passed']=result['projection']['admitted']
    except BaseException as e:result['failure']={'type':type(e).__name__,'message':str(e)}
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=counts,stored_bytes_before_final=total[0])
    if result['elapsed_seconds']>=120 or result['peak_rss_kib']>=2*1024**2:result['passed']=False
    save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):P.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
