#!/usr/bin/env python3
"""One saved-array positive-radius inflation diagnosis; no target acquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import widen_eigenpair_locus as W
from neural_shrimp.eigenpair_radius_diagnostic import diagnose
from neural_shrimp.rigorous_eigenpair_locus import require
M=W.M;T=W.T;entry=M.entry
PARENT=T.BASE/'attempt-001-wider-target'
PARENT_SOURCE='3e2681f53c0134f58846cb9a953667d7d2833efa'
PARENT_OUTER='d3a46e841992027d12da3b9495bf93a03a64235da1508870efe4b040cfc0a1d0'
PARENT_RUN='7a01b994bf31e9a9057ee3e6e369a0acc9da0f3a4450924d66031404484e4e90'
EXTRA=['src/neural_shrimp/eigenpair_radius_diagnostic.py','tools/diagnose_eigenpair_inflation.py','tools/launch_eigenpair_inflation_diagnostic.py','tests/test_eigenpair_radius_diagnostic.py','docs/plans/eigenpair-inflation-diagnostic.md','docs/reviews/eigenpair-wider-outcome.json']
SOURCES=list(dict.fromkeys(W.SOURCES+EXTRA))


def bind():
    bindings={'outer':M.Q.B.inventory(PARENT,PARENT_OUTER),'run':M.Q.B.inventory(PARENT/'run',PARENT_RUN),'execution':PARENT_SOURCE}
    rt=M.read(PARENT/'run/runtime.json');la=M.read(PARENT/'launcher-result.json');out=M.read(PARENT/'run/outcome.json')
    require(rt['commit']==la['commit']==PARENT_SOURCE and rt['mode']==la['mode']=='target','exact failed execution')
    require(la['exit_code']==2 and la['stop_reason'] is None and not la['owned_live_pids'],'expected gate failure')
    require(out['status']=='wider_proof_gate_unresolved' and out['passed'] is False and out['ticks']==127 and len(out['progress']['chart_files'])==1,'fixed single first-chart failure')
    require(M.finite(out['elapsed_seconds'],150) and M.finite(out['peak_rss_kib'],2*1024**2) and M.finite(la['elapsed_seconds'],175),'within prior actual caps')
    for n in W.SOURCES:
        v=entry(PARENT/'run/source'/n);require(v==rt['sources'][n]==entry(ROOT/n),'unchanged source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',PARENT_SOURCE+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'git source '+n)
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same saved arithmetic environment '+k)
    name=out['progress']['chart_files'][0]
    with gzip.open(T.C.input_path(PARENT,name),'rt') as f:chart=json.load(f)
    require(chart['parameter_endpoints']==['19/128','23/128'] and chart['period']==18 and chart['sigma']==1 and chart['dimension']==13,'fixed native failed chart')
    require([i for i,v in enumerate(chart['coordinate_inclusions']) if not v]==[387],'fixed failure index')
    bindings.update(full_chart=name,full_identity=entry(T.C.input_path(PARENT,name)))
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
        ticks[0]+=1;check();require(ticks[0]==0,'no native calls')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':0,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'saved-only radius inflation diagnosis; no native model calls'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            chart,bindings=bind();save('bindings.json',bindings);check()
            result=diagnose(chart,observe);save('result.json',result)
            outcome.update(passed=True,status='saved_inflation_proposal_completed',summary=result['summary'],scope=result['scope'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
