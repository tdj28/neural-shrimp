#!/usr/bin/env python3
"""One four-chart bounded wider continuation; each chart permits one full repair."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import repair_eigenpair_wider as R
M=R.M;T=R.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_inflation_march import march
PARENT=T.BASE/'attempt-001-wider-repair'
PARENT_SOURCE='534c4cb129d14a0269506a9b58814e9a35ced0f2'
PARENT_OUTER='5fe146921e1d836036054e32f421e8ccd425204db1f57c31d708017c95dd239d'
PARENT_RUN='2035aba9276f0d6aa6a41691fbdef13b9d49f4ba3d478e9832ca3f6962be27d1'
EXTRA=['src/neural_shrimp/eigenpair_boundary_diagnostics.py','src/neural_shrimp/rigorous_eigenpair_inflation_march.py','tools/continue_eigenpair_wider.py','tools/launch_eigenpair_wider_march.py','tests/test_eigenpair_inflation_march.py','tests/test_eigenpair_wider_march_adapter.py','docs/plans/eigenpair-wider-march.md','docs/reviews/eigenpair-wider-repair-outcome.json']
SOURCES=list(dict.fromkeys(R.SOURCES+EXTRA));entry=M.entry


def cost_gate(old,reanchor):
    require(old['passed'] is True and old['ticks']==127 and old['status']=='one_repaired_wider_chart','completed repaired reference')
    require(reanchor['passed'] is True and reanchor['ticks']==326,'larger observed memory reference')
    require(M.finite(old['elapsed_seconds'],90) and M.finite(old['stored_bytes_before_final'],128*1024**2) and all(M.finite(x,2*1024**2) for x in [old['peak_rss_kib'],reanchor['peak_rss_kib']]),'finite measured resources')
    seconds=2*old['elapsed_seconds']*8+20;storage=8*old['stored_bytes_before_final']+8*1024**2;rss=max(old['peak_rss_kib'],reanchor['peak_rss_kib'])+128*1024
    require(seconds<400 and storage<504*1024**2 and rss<2*1024**2,'measured eight-call admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':rss,'scope':'Twice measured127tick repair times eight full calls+20s; eight full byte bundles+8MiB; observed larger reanchorRSS+128MiB. Diagnostics add at most eight13D eigs/eight14D solves/four470D proposal solves. Actual caps decisive.'}


def bind():
    inp,old,proposal,bindings=R.bind();del old,proposal;gc.collect()
    bindings['parent_outer']=M.Q.B.inventory(PARENT,PARENT_OUTER);bindings['parent_run']=M.Q.B.inventory(PARENT/'run',PARENT_RUN)
    rt,la,out=T.source_runtime(PARENT,PARENT_SOURCE,R.SOURCES)
    require(out['status']=='one_repaired_wider_chart' and out['ticks']==127 and len(out['progress']['chart_files'])==1,'fixed repaired accepted chart')
    name=out['progress']['chart_files'][0]
    with gzip.open(T.C.input_path(PARENT,name),'rt') as f:full=json.load(f)
    result=M.read(PARENT/'run/result.json');parent=compact(full)
    require(result['passed'] is True and result['charts']==[parent] and parent['parameter_endpoints']==['19/128','23/128'],'complete full/compact identity')
    require(parent['settings']['halfwidth_exponent']==-6 and parent['settings']['precision']==256 and parent['settings']['newton_steps']==2 and parent['period']==18 and parent['sigma']==1 and parent['dimension']==13,'native settings')
    pi=M.read(PARENT/'run/inputs.json');require(pi['map_inputs']==inp,'same exact consumed neural inputs')
    for k in ('normal','origin','period','sigma','dimension'):require(parent[k]==pi['parent'][k],'inherited native gauge '+k)
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same libraries/threads '+k)
    bindings['cost']=cost_gate(out,M.read(T.ANCHOR/'run/outcome.json'))
    bindings['selected_parent']={'path':str(PARENT),'execution':PARENT_SOURCE,'outer':PARENT_OUTER,'run':PARENT_RUN,'right':'23/128','full_chart':name,'full_identity':entry(T.C.input_path(PARENT,name))}
    del full;gc.collect();return inp,parent,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=400;rss=2*1024**2;cap=512*1024**2;reserve=8*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'active_call':None,'full_chart_file':None,'accepted':[],'last_accepted':None}
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
        ticks[0]+=1;check();require(ticks[0]<=1040,'fixed1040tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='march_call_start':progress['active_call']=value['call'];progress['full_chart_file']=None
        if kind=='chart_return':progress['full_chart_file']=name
        if kind=='march_accepted':
            require(progress['full_chart_file'] is not None,'durable full chart before accepted reference')
            rec={'call':value['call'],'index':value['index'],'full_chart_file':progress['full_chart_file'],'accepted_file':name,'right':value['chart']['parameter_endpoints'][1]}
            progress['accepted'].append(rec);progress['last_accepted']=rec;progress['active_call']=None
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':1040,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'positive full-state tangent gauge, four wider charts, at most one nonshrinking repair each; diagnostics not proof'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind();save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-6,'accepted_cap':4,'call_cap':8,'repair_cap_per_chart':1})
            check();model=TwoRateCuspMap(inp);result=march(model,parent,tick=tick,observe=observe);save('march.json',result)
            outcome.update(passed=result['completed'],status='four_joined_charts' if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],repairs=result['repairs'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
