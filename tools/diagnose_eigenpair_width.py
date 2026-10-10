#!/usr/bin/env python3
"""One fixed saved-array coefficient/defect diagnosis; no target acquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import continue_eigenpair_tangent_chain as T
from neural_shrimp.eigenpair_width_diagnostic import split_width
from neural_shrimp.rigorous_eigenpair_locus import require
M=T.M;entry=M.entry
PARENT=T.BASE/'attempt-002-tangent-march-target'
PARENT_SOURCE='5f32fccb71262f3106fcb2e69dce0f5c5dcfe033'
PARENT_OUTER='60c2b2d52d6a19a9e68d11d62978d9e148e63866818a5f6d00ea1f827740f7ed'
PARENT_RUN='0eb0261857d5b2361c7907a4e66af7e2b25399203d80de461b6ab48823e6dd9d'
EXTRA=['src/neural_shrimp/eigenpair_width_diagnostic.py','tools/diagnose_eigenpair_width.py','tools/launch_eigenpair_width_diagnostic.py','tests/test_eigenpair_width_diagnostic.py','docs/plans/eigenpair-width-diagnostic.md','docs/reviews/eigenpair-tangent-unit002.json']
SOURCES=list(dict.fromkeys(T.SOURCES+EXTRA))


def bind():
    bindings={'outer':M.Q.B.inventory(PARENT,PARENT_OUTER),'run':M.Q.B.inventory(PARENT/'run',PARENT_RUN),'execution':PARENT_SOURCE}
    rt,la,out=T.source_runtime(PARENT,PARENT_SOURCE,T.SOURCES)
    require(out['status']=='four_joined_charts' and len(out['progress']['accepted'])==4 and out['ticks']==508,'fixed complete unit002')
    result=M.read(PARENT/'run/march.json');last=out['progress']['last_accepted']
    with gzip.open(T.C.input_path(PARENT,last['full_chart_file']),'rt') as f:chart=json.load(f)
    with gzip.open(T.C.input_path(PARENT,last['accepted_file']),'rt') as f:accepted=json.load(f)
    T.validate_parent(out,result,accepted,chart,'19/128',result['last_chart'])
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same saved arithmetic environment '+k)
    bindings.update(full_chart=last['full_chart_file'],full_identity=entry(T.C.input_path(PARENT,last['full_chart_file'])))
    return chart,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=45;rss=1536*1024;cap=64*1024**2;reserve=4*1024**2
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
        ticks[0]+=1;check();require(ticks[0]<=360,'fixed360tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':360,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'saved-only width-scaling diagnosis; no native model calls'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            chart,bindings=bind();save('bindings.json',bindings);check()
            result=split_width(chart,observe);save('result.json',result)
            outcome.update(passed=True,status='saved_budget_reduction_completed',summary=result['summary'],scope=result['scope'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
