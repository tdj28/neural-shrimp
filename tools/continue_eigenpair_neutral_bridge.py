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
import re
from fractions import Fraction
import narrow_eigenpair_locus as N
M=N.M;T=N.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_inflation_march import march
from neural_shrimp.eigenpair_boundary_target import BoundaryTargetReached,event_flags
EXTRA=['src/neural_shrimp/eigenpair_boundary_target.py','tools/continue_eigenpair_neutral_bridge.py','tools/launch_eigenpair_neutral_bridge.py','tests/test_eigenpair_neutral_bridge.py','docs/plans/eigenpair-neutral-bridge.md','docs/reviews/eigenpair-narrower-outcome.json','docs/reviews/eigenpair-boundary-trend-outcome.json']
SOURCES=list(dict.fromkeys(N.SOURCES+EXTRA));entry=M.entry
FIRST_PINS={'parent_source':'05c79b257f2246bcd1be69b5e0c194fdc1fcd6ea','parent_manifest':'32cd52e51ea65398e450621d78749688372cb24e2e64a2d2f7836de211ba288a','parent_run_manifest':'a23585330ec14de6a294ae11b39abc2777c715bc3f2f0b629aae5bd6f292625c'}


def install(args):
    require(type(args.unit)is int and 1<=args.unit<=9,'fixed target bridge units')
    require(args.expected_right==str(Fraction(227+8*(args.unit-1),128)),'exact bridge predecessor endpoint')
    expected=T.BASE/('attempt-001-narrower-target' if args.unit==1 else f'attempt-{args.unit-1:03d}-neutral-bridge')
    require(args.parent==expected and not args.parent.is_symlink(),'exact owned predecessor path')
    require(re.fullmatch('[0-9a-f]{40}',args.parent_source or '') and all(re.fullmatch('[0-9a-f]{64}',v or '')for v in (args.parent_manifest,args.parent_run_manifest)),'full exact parent pins')
    if args.unit==1:require(all(getattr(args,k)==v for k,v in FIRST_PINS.items()),'exact completed narrower pilot')


def accepted_check(full,expected):
    c=compact(full)
    require(c['period']==18 and c['dimension']==13 and c['sigma']==1 and c['shooting_dimension']==470,'native primitive18 normalized+1 dimensions')
    require(c['settings']['halfwidth_exponent']==-7 and c['settings']['precision']==256 and c['settings']['newton_steps']==2,'same scientific settings')
    require(c['parameter_endpoints'][1]==expected,'exact complete endpoint')
    require(len(full['coordinate_inclusions'])==470 and all(full['coordinate_inclusions']) and full['anchor_check']['passed'] and len(full['anchor_check']['coordinate_inclusions'])==470 and all(full['anchor_check']['coordinate_inclusions']),'strict native root and anchor')
    require([v['divisor']for v in full['divisors']]==[1,2,3,6,9] and all(v['excluded']for v in full['divisors']),'primitive18')
    return c


def validate_parent(out,result,accepted,full,expected):
    require(out['passed'] is True and out['status']=='four_joined_charts' and result['completed'] is True and len(result['accepted'])==len(out['progress']['accepted'])==4,'complete unflagged four-chart bridge only')
    last=out['progress']['last_accepted'];require(last==out['progress']['accepted'][-1] and out['progress']['active_call']is None,'durable complete predecessor')
    require(0<=result['repairs']<=4 and result['calls']==4+result['repairs']==out['calls'] and result['repairs']==out['repairs'],'complete bounded call count')
    require([v['call']for v in result['attempts']]==list(range(result['calls'])) and sum(v['passed']for v in result['attempts'])==4,'all attempts retained')
    require(last['call']==accepted['call']==result['calls']-1 and last['index']==accepted['index']==3,'final accepted native call')
    c=accepted_check(full,expected);require(c==accepted['chart']==result['last_chart'],'full compact summary identity')
    require(not event_flags(accepted['diagnostics']),'no flagged event can silently advance')
    return c


def cost_gate(pilot):
    require(pilot['passed'] is True and pilot['ticks']==254 and pilot['status']=='two_narrower_charts','completed exact narrower kernel reference')
    require(M.finite(pilot['elapsed_seconds'],150) and M.finite(pilot['stored_bytes_before_final'],256*1024**2) and M.finite(pilot['peak_rss_kib'],2*1024**2),'finite measured resources')
    seconds=2*pilot['elapsed_seconds']*4+20;storage=4*pilot['stored_bytes_before_final']+8*1024**2;rss=pilot['peak_rss_kib']+512*1024
    require(seconds<400 and storage<504*1024**2 and rss<2*1024**2,'measured eight-call admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':rss,'scope':'Four times observed two-chart bundle,2x timing+20s,observed peak+512MiB. Heuristic forecast; actual fixed caps mandatory.'}


def bind(args):
    bindings={'parent_outer':M.Q.B.inventory(args.parent,args.parent_manifest),'parent_run':M.Q.B.inventory(args.parent/'run',args.parent_run_manifest)}
    rt=M.read(args.parent/'run/runtime.json');la=M.read(args.parent/'launcher-result.json');out=M.read(args.parent/'run/outcome.json')
    require(rt['commit']==la['commit']==args.parent_source and rt['mode']==la['mode']=='target' and rt['precision']==256 and rt['nice']>=10,'exact parent execution/settings')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None and M.finite(la['elapsed_seconds'],427),'complete owned parent exit')
    require(out['passed'] and M.finite(out['elapsed_seconds'],400) and M.finite(out['peak_rss_kib'],2*1024**2),'completed bounded parent')
    allowed=N.SOURCES if args.unit==1 else SOURCES
    require(set(rt['sources'])==set(allowed),'exact prior runner inventory')
    for n,v in rt['sources'].items():
        require(entry(ROOT/n)==entry(args.parent/'run/source'/n)==v,'unchanged parent source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',args.parent_source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical parent source '+n)
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same libraries/threads '+k)
    if args.unit==1:
        require(out['status']=='two_narrower_charts' and out['ticks']==254 and len(out['progress']['chart_files'])==2,'complete two-chart pilot')
        name=out['progress']['chart_files'][-1]
        with gzip.open(T.C.input_path(args.parent,name),'rt')as f:full=json.load(f)
        parent=accepted_check(full,args.expected_right);result=M.read(args.parent/'run/result.json')
        require(result['passed'] and len(result['charts'])==2 and result['charts'][-1]==parent,'complete pilot full/compact identity')
        require(result['charts'][0]['parameter_endpoints']==['223/128','225/128'] and parent['parameter_endpoints']==['225/128','227/128'],'exact gapless pilot intervals')
    else:
        require(rt['unit']==args.unit-1,'immediate bridge unit only');last=out['progress']['last_accepted'];name=last['full_chart_file']
        with gzip.open(T.C.input_path(args.parent,name),'rt')as f:full=json.load(f)
        with gzip.open(T.C.input_path(args.parent,last['accepted_file']),'rt')as f:accepted=json.load(f)
        parent=validate_parent(out,M.read(args.parent/'run/march.json'),accepted,full,args.expected_right)
    pi=M.read(args.parent/'run/inputs.json');old=pi['parent']
    for k in ('normal','origin','period','sigma','dimension'):require(parent[k]==old[k],'same exact native gauge '+k)
    start='223/128' if args.unit==1 else str(Fraction(args.expected_right)-Fraction(8,128))
    require(old['parameter_endpoints'][1]==start,'exact accepted lineage advancement')
    reference=T.BASE/'attempt-001-narrower-target'
    bindings['reference_outer']=M.Q.B.inventory(reference,FIRST_PINS['parent_manifest'])
    bindings['reference_run']=M.Q.B.inventory(reference/'run',FIRST_PINS['parent_run_manifest'])
    require(pi['map_inputs']==M.read(reference/'run/inputs.json')['map_inputs'],'same exact original neural inputs')
    bindings['cost']=cost_gate(M.read(reference/'run/outcome.json'))
    bindings['selected_parent']={'path':str(args.parent),'execution':args.parent_source,'outer':args.parent_manifest,'run':args.parent_run_manifest,'right':args.expected_right,'full_chart':name,'full_identity':entry(T.C.input_path(args.parent,name))}
    del full;gc.collect();return pi['map_inputs'],parent,bindings


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
            flags=event_flags(value['diagnostics'])
            if flags:raise BoundaryTargetReached(flags)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'unit':args.unit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':1040,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'native bridge toward second-neutral interaction; fixed accepted-root diagnostic stop; no event certification'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind(args);save('bindings.json',bindings);accepted_cap=2 if args.unit==9 else 4
            save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-7,'accepted_cap':accepted_cap,'call_cap':2*accepted_cap,'repair_cap_per_chart':1,'objective_gauge':'295/128'})
            check();model=TwoRateCuspMap(inp)
            try:
                result=march(model,parent,accepted_cap=accepted_cap,tick=tick,observe=observe);save('march.json',result)
                outcome.update(passed=result['completed'],status=('target_gauge_reached' if args.unit==9 else 'four_joined_charts') if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],repairs=result['repairs'])
            except BoundaryTargetReached as event:
                save('event-stop.json',{'flags':event.flags,'accepted':progress['accepted'],'scope':'Numerical search stop after durable qualified roots; not a certified event.'})
                outcome.update(passed=True,status='qualified_event_stop',event_flags=event.flags,accepted_count=len(progress['accepted']))
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);p.add_argument('--unit',type=int,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-source',required=True);p.add_argument('--parent-manifest',required=True);p.add_argument('--parent-run-manifest',required=True);p.add_argument('--expected-right',required=True);return p
if __name__=='__main__':
    args=parser().parse_args();install(args);raise SystemExit(main(args))
