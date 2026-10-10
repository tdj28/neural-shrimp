#!/usr/bin/env python3
"""One fresh whole-domain radius repair; last accepted parent remains the anchor."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import diagnose_eigenpair_partial as I
M=I.M;T=I.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.eigenpair_partial_repair import repair
DIAG=T.BASE/'attempt-001-partial-diagnosis'
DIAG_SOURCE='ee4753e71ddad0d9eae6bf58d21a77961979ff88'
DIAG_OUTER='a857de50ada4f0a84ef94e7641d45f4803e19d56d6887549698a29a9f0b66ac2'
DIAG_RUN='627c2da8bdaa47d9f704c42f59c34fcb01099a20972091e2f3b596c12c35af5a'
EXTRA=['src/neural_shrimp/eigenpair_partial_repair.py','tools/repair_eigenpair_partial.py','tools/launch_eigenpair_partial_repair.py','tests/test_eigenpair_partial_repair.py','docs/plans/eigenpair-partial-repair.md','docs/reviews/eigenpair-partial-diagnosis.json']
SOURCES=list(dict.fromkeys(I.SOURCES+EXTRA));entry=M.entry


def cost_gate(old):
    require(old['passed'] is False and old['ticks']==508 and old['calls']==4 and old['status']=='qualified_prefix_then_unresolved','exact partial measured reference')
    require(M.finite(old['peak_rss_kib'],2*1024**2) and M.finite(old['elapsed_seconds'],400) and M.finite(old['stored_bytes_before_final'],512*1024**2),'finite measured costs')
    seconds=2*old['elapsed_seconds']/4+20;storage=2*old['stored_bytes_before_final']/4+8*1024**2;rss=old['peak_rss_kib']+512*1024
    require(seconds<90 and storage<120*1024**2 and rss<2*1024**2,'measured one-repair admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':rss,'scope':'Twice same-unit per-call average +20s; twice per-call bytes+8MiB; observed peak+512MiB. Not a worst-case guarantee; actual caps mandatory.'}


def bind():
    failed,parent,inp,bindings=I.bind_all()
    # Discard the large old matrices after binding. No old derivative enters the new proof.
    failed={k:failed[k] for k in ('passed','gates','period','dimension','sigma','shooting_dimension','normal','origin','settings','weights','coordinate_radii','parameter_endpoints')};gc.collect()
    bindings['diagnosis_outer']=M.Q.B.inventory(DIAG,DIAG_OUTER);bindings['diagnosis_run']=M.Q.B.inventory(DIAG/'run',DIAG_RUN)
    rt,la,out=T.source_runtime(DIAG,DIAG_SOURCE,I.SOURCES)
    require(out['status']=='saved_inflation_proposal_completed' and out['ticks']==0 and M.finite(out['elapsed_seconds'],45) and M.finite(out['peak_rss_kib'],1536*1024),'fixed saved-only proposal')
    proposal=M.read(DIAG/'run/result.json')
    require(proposal['summary']==out['summary'] and proposal['summary']['positive'] is True and proposal['summary']['all_old_radii_retained'] is True,'proposal identity')
    require(math.isfinite(proposal['summary']['proposal_old_domain_max_ratio']) and 0<=proposal['summary']['proposal_old_domain_max_ratio']<1,'proposal support only')
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same libraries/threads '+k)
    bindings['cost']=cost_gate(M.read(I.PARENT/'run/outcome.json'))
    return inp,parent,failed,proposal,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=90;rss=2*1024**2;cap=128*1024**2;reserve=8*1024**2
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
        ticks[0]+=1;check();require(ticks[0]<=130,'fixed130tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='chart_return':progress['chart_files'].append(name)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':130,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'same full-state tangent gauge; one radius repair; no genericity claim'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,failed,proposal,bindings=bind();save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'failed_radius_context':failed,'proposal':proposal,'halfwidth_exponent':-6,'chart_cap':1})
            check();model=TwoRateCuspMap(inp);chart=repair(model,parent,failed,proposal,tick=tick,observe=observe)
            reduced={'passed':chart['passed'],'charts':[compact(chart)] if chart['passed'] else [],'gates':[chart['gates']]};save('result.json',reduced)
            outcome.update(passed=chart['passed'],status='one_repaired_partial_chart' if chart['passed'] else 'repair_proof_gate_unresolved',gates=reduced['gates'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
