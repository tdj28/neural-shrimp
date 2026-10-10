#!/usr/bin/env python3
"""One fixed halved-width pilot, reacquiring every proof array and gate."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import repair_eigenpair_partial as R
M=R.M;T=R.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.eigenpair_narrower import narrow
PARENT=T.BASE/'attempt-001-partial-repair'
PARENT_SOURCE='25e9ac5f824a8d8a0fcbcb6b7252606ef68c5c9c'
PARENT_OUTER='346a8c03213f4819c73f02feefdb3a38cfd578e1ef211b427281c2def263dc47'
PARENT_RUN='d2517ecc93e3a046ac59ddfae9eb64efda686a0b14e212416e51f7afed93f388'
EXTRA=['src/neural_shrimp/eigenpair_narrower.py','tools/narrow_eigenpair_locus.py','tools/launch_eigenpair_narrower.py','tests/test_eigenpair_narrower.py','docs/plans/eigenpair-halved-width.md','docs/reviews/eigenpair-partial-repair-outcome.json']
SOURCES=list(dict.fromkeys(R.SOURCES+EXTRA));entry=M.entry


def cost_gate(old):
    require(old['passed'] is True and old['ticks']==127 and old['status']=='one_repaired_partial_chart','completed same-kernel cost reference')
    require(M.finite(old['elapsed_seconds'],90) and M.finite(old['peak_rss_kib'],2*1024**2) and M.finite(old['stored_bytes_before_final'],128*1024**2),'finite cost reference')
    seconds=2*old['elapsed_seconds']*2+20;storage=2*old['stored_bytes_before_final']+8*1024**2;rss=old['peak_rss_kib']+512*1024
    require(seconds<150 and storage<248*1024**2 and rss<2*1024**2,'measured two-chart admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':rss,'scope':'Two fresh127tick calls,2x timing allowance+20s,2x raw+8MiB,observedRSS+512MiB. Heuristic forecast only; actual fixed caps mandatory.'}


def validate_parent(full,result,old):
    parent=compact(full)
    require(result['passed'] is True and result['charts']==[parent],'complete exact full/compact identity')
    require(parent['parameter_endpoints']==['219/128','223/128'] and parent['settings']['halfwidth_exponent']==-6 and parent['settings']['precision']==256 and parent['settings']['newton_steps']==2,'exact completed domain/settings')
    require(parent['period']==18 and parent['sigma']==1 and parent['dimension']==13 and parent['shooting_dimension']==470,'native dimensions/sign')
    for k in ('normal','origin','period','sigma','dimension'):require(parent[k]==old[k],'same inherited native gauge '+k)
    require(parent['parameter_endpoints'][0]==old['parameter_endpoints'][1],'exact proved parent endpoint')
    require(len(full['coordinate_inclusions'])==470 and all(full['coordinate_inclusions']) and full['anchor_check']['passed'] and len(full['anchor_check']['coordinate_inclusions'])==470 and all(full['anchor_check']['coordinate_inclusions']),'all full native inclusion gates')
    require([v['divisor']for v in full['divisors']]==[1,2,3,6,9] and all(v['excluded']for v in full['divisors']),'primitive18')
    return parent


def bind():
    bindings={'parent_outer':M.Q.B.inventory(PARENT,PARENT_OUTER),'parent_run':M.Q.B.inventory(PARENT/'run',PARENT_RUN)}
    rt,la,out=T.source_runtime(PARENT,PARENT_SOURCE,R.SOURCES)
    require(out['status']=='one_repaired_partial_chart' and out['ticks']==127 and len(out['progress']['chart_files'])==1,'fixed complete repaired source')
    name=out['progress']['chart_files'][0]
    with gzip.open(T.C.input_path(PARENT,name),'rt') as f:full=json.load(f)
    pi=M.read(PARENT/'run/inputs.json');parent=validate_parent(full,M.read(PARENT/'run/result.json'),pi['parent'])
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same libraries/threads '+k)
    bindings['cost']=cost_gate(out);bindings['selected_parent']={'path':str(PARENT),'execution':PARENT_SOURCE,'outer':PARENT_OUTER,'run':PARENT_RUN,'right':'223/128','full_chart':name,'full_identity':entry(T.C.input_path(PARENT,name))}
    del full;gc.collect();return pi['map_inputs'],parent,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=150;rss=2*1024**2;cap=256*1024**2;reserve=8*1024**2
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
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':260,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'same full-state tangent gauge; fixed halved width; no genericity claim'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind();save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-7,'chart_cap':2})
            check();model=TwoRateCuspMap(inp);result=narrow(model,parent,tick=tick,observe=observe)
            reduced={'passed':result['passed'],'charts':[compact(c) for c in result['charts'] if c['passed']],'gates':[c['gates'] for c in result['charts']]};save('result.json',reduced)
            outcome.update(passed=result['passed'],status='two_narrower_charts' if result['passed'] else 'narrower_proof_gate_unresolved',gates=reduced['gates'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
