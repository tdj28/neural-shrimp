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
import continue_eigenpair_wider_chain as C
import continue_eigenpair_radius_forecast as F
W=C.W
M=W.M;T=W.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_inflation_march import march
EXTRA=['tools/continue_eigenpair_repair_chain.py','tools/launch_eigenpair_repair_chain.py','tests/test_eigenpair_repair_chain.py','docs/plans/eigenpair-repair-chain.md','docs/reviews/eigenpair-wider-march-unit004.json']
SOURCES=list(dict.fromkeys(F.SOURCES+EXTRA));entry=M.entry
FIRST_PINS={'parent_source':'af481a1a800465d715ae91eeb4171eb46dd4f0bc','parent_manifest':'f846f878fb5cbcf266c9846a5a9c7135fefd5334df57429e3216bfa595f9f27e','parent_run_manifest':'1c512659bf31255e70393f49f509117f8f8af6a23d9177dcce804d196c6359b2'}

def parent_inventory(unit):
    require(type(unit)is int and 5<=unit<=64,'bounded repair-only unit')
    return F.SOURCES if unit==5 else SOURCES



def install(args):
    require(type(args.unit)is int and 5<=args.unit<=64,'bounded explicit unit')
    require(args.expected_right==str(Fraction(16*(args.unit-1)+23,128)),'exact prospective endpoint')
    require(args.parent==T.BASE/f'attempt-{args.unit-1:03d}-wider-march' and not args.parent.is_symlink(),'exact prior unit path')
    require(re.fullmatch('[0-9a-f]{40}',args.parent_source or '') and all(re.fullmatch('[0-9a-f]{64}',v or '') for v in (args.parent_manifest,args.parent_run_manifest)),'full exact parent pins')

    if args.unit==5:require(all(getattr(args,k)==v for k,v in FIRST_PINS.items()),'exact complete forecast parent')

validate_parent=C.validate_parent


def bind(args):
    inp,anchor,bindings=W.bind()
    bindings['parent_outer']=M.Q.B.inventory(args.parent,args.parent_manifest);bindings['parent_run']=M.Q.B.inventory(args.parent/'run',args.parent_run_manifest)
    rt=M.read(args.parent/'run/runtime.json');la=M.read(args.parent/'launcher-result.json');out=M.read(args.parent/'run/outcome.json')
    require(rt['commit']==la['commit']==args.parent_source and rt['mode']==la['mode']=='target' and rt['precision']==256 and rt['nice']>=10,'parent execution/settings')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and M.finite(la['elapsed_seconds'],427),'owned successful exit')
    require(out['passed'] is True and M.finite(out['elapsed_seconds'],400) and M.finite(out['peak_rss_kib'],2*1024**2),'finite admitted parent')
    allowed=parent_inventory(args.unit)
    require(set(rt['sources'])==set(allowed),'exact parent inventory')
    require(rt['unit']==args.unit-1,'exact immediately prior unit number')
    for n,v in rt['sources'].items():
        require(entry(ROOT/n)==entry(args.parent/'run/source'/n)==v,'unchanged source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',args.parent_source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed parent source '+n)
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same libraries/threads '+k)
    result=M.read(args.parent/'run/march.json');last=out['progress']['last_accepted']
    with gzip.open(T.C.input_path(args.parent,last['full_chart_file']),'rt') as f:full=json.load(f)
    with gzip.open(T.C.input_path(args.parent,last['accepted_file']),'rt') as f:accepted=json.load(f)
    parent=validate_parent(out,result,accepted,full,args.expected_right,anchor)
    pi=M.read(args.parent/'run/inputs.json');require(pi['map_inputs']==inp and pi['parent']['parameter_endpoints'][1]==str(Fraction(args.expected_right)-Fraction(16,128)),'exact map and four-chart advancement')
    bindings['selected_parent']={'path':str(args.parent),'execution':args.parent_source,'outer':args.parent_manifest,'run':args.parent_run_manifest,'right':args.expected_right,'full_chart':last['full_chart_file'],'full_identity':entry(T.C.input_path(args.parent,last['full_chart_file']))}
    del full,accepted;gc.collect();return inp,parent,bindings


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
    save('runtime.json',{'commit':commit,'unit':args.unit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':1040,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'positive full-state tangent gauge, inherited qualified radius floor without forecast; at most one fresh nonshrinking repair each; diagnostics not proof'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind(args);save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-6,'accepted_cap':4,'call_cap':8,'repair_cap_per_chart':1})
            check();model=TwoRateCuspMap(inp);result=march(model,parent,tick=tick,observe=observe);save('march.json',result)
            outcome.update(passed=result['completed'],status='four_joined_charts' if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],repairs=result['repairs'])
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
