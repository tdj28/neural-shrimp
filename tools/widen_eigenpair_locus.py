#!/usr/bin/env python3
"""One fixed doubled-width pilot, reacquiring every proof array and gate."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import diagnose_eigenpair_width as D
M=D.M;T=D.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.eigenpair_wider import widen
DIAG=T.BASE/'attempt-001-width-diagnosis'
DIAG_SOURCE='b71ad685d11202b558091be7c99d628c46534767'
DIAG_OUTER='143b722dcd04504eaa66fece394445047ba5d98cc3d257c52a2c76b444ad970a'
DIAG_RUN='5a266b7d946421860f19f39f61d3d57c8e48ac95bb1c611da6873f642f6bb351'
EXTRA=['src/neural_shrimp/eigenpair_wider.py','tools/widen_eigenpair_locus.py','tools/launch_eigenpair_wider.py','tests/test_eigenpair_wider.py','tests/test_eigenpair_wider_adapter.py','docs/plans/eigenpair-doubled-width.md','docs/reviews/eigenpair-width-diagnosis.json']
SOURCES=list(dict.fromkeys(D.SOURCES+EXTRA));entry=M.entry


def cost_gate(old,diag,reanchor):
    require(old['passed'] is True and old['ticks']==508 and old['status']=='four_joined_charts','completed original-cost unit')
    require(diag['passed'] is True and diag['ticks']==0 and diag['summary']['double_width_proxy_below_one'] is True,'saved diagnostic supports one trial only')
    require(reanchor['passed'] is True and reanchor['ticks']==326,'observed memory reference')
    require(all(M.finite(v,2*1024**2) for v in (old['peak_rss_kib'],reanchor['peak_rss_kib'])) and M.finite(old['elapsed_seconds'],300) and M.finite(old['stored_bytes_before_final'],768*1024**2),'finite cost values')
    seconds=2*old['elapsed_seconds']*254/508+20;storage=old['stored_bytes_before_final']+8*1024**2;rss=max(old['peak_rss_kib'],reanchor['peak_rss_kib'])+128*1024
    require(seconds<150 and storage<376*1024**2 and rss<2*1024**2,'measured widened-pilot cost admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':rss,'scope':'Measured same-kernel operation count with2x timing allowance+20s, full4chart bytes and observed larger reanchorRSS+128MiB. Wider-domain cost remains a heuristic; fixed actual caps mandatory.'}


def bind():
    chart,bindings=D.bind();parent=compact(chart);del chart;gc.collect()
    require(parent['settings']['halfwidth_exponent']==-7 and parent['parameter_endpoints'][1]=='19/128','fixed old endpoint/width')
    bindings['diagnosis_outer']=M.Q.B.inventory(DIAG,DIAG_OUTER);bindings['diagnosis_run']=M.Q.B.inventory(DIAG/'run',DIAG_RUN)
    rt,la,diag=T.source_runtime(DIAG,DIAG_SOURCE,D.SOURCES)
    require(diag['status']=='saved_budget_reduction_completed' and diag['summary']['double_width_proxy_below_one'] is True,'declared heuristic support; not a proof')
    old=M.read(D.PARENT/'run/outcome.json');reanchor=M.read(T.ANCHOR/'run/outcome.json')
    bindings['memory_reference_outer']=M.Q.B.inventory(T.ANCHOR,T.ANCHOR_OUTER)
    bindings['cost']=cost_gate(old,diag,reanchor)
    inp=M.read(D.PARENT/'run/inputs.json')['map_inputs']
    return inp,parent,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=150;rss=2*1024**2;cap=384*1024**2;reserve=8*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'localization_file':None,'proposal_file':None,'chart_files':[]}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    def save(name,value,final=False):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);before=total[0];bound=cap if final else cap-reserve
        try:
            with p.open('xb') as raw:
                h=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
                try:
                    chunks=[];size=0
                    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
                        chunks.append(piece);size+=len(piece)
                        if size>=65536:h.write(''.join(chunks).encode());chunks=[];size=0;require(before+raw.tell()<bound,'aggregate raw cap')
                    if chunks:h.write(''.join(chunks).encode())
                finally:
                    if h is not raw:h.close()
                raw.flush();os.fsync(raw.fileno())
            require(before+p.stat().st_size<bound,'aggregate raw cap')
        finally:
            if p.exists():total[0]=before+p.stat().st_size
        return name
    def check():
        if stop[0] or time.monotonic()-start>=limit:raise TimeoutError('fixed continuation deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=rss:raise MemoryError('monitored RSS')
    def tick():
        ticks[0]+=1;check();require(ticks[0]<=260,'fixed260tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='chart_return':progress['chart_files'].append(name)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':260,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'same full-state tangent gauge; fixed doubled width; no genericity claim'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind();save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-6,'chart_cap':2})
            check();model=TwoRateCuspMap(inp);result=widen(model,parent,tick=tick,observe=observe)
            reduced={'passed':result['passed'],'charts':[compact(c) for c in result['charts'] if c['passed']],'gates':[c['gates'] for c in result['charts']]};save('result.json',reduced)
            outcome.update(passed=result['passed'],status='two_wider_charts' if result['passed'] else 'wider_proof_gate_unresolved',gates=reduced['gates'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
