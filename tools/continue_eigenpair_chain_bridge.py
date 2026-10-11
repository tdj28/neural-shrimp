#!/usr/bin/env python3
"""Fixed ten-chart maximum native bridge to the authenticated generalized-chain root."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,re,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_eigenpair_chain_lift as L
B=L.C.B;M=B.M;T=B.T;entry=L.entry
from neural_shrimp.rigorous_chain_lift import quadratic_membership
from neural_shrimp.rigorous_eigenpair_march import compact,decode
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_inflation_march import march
BASE=L.BASE/'membership-bridge'
LIFT=L.BASE/'attempt001-target'
LPINS={'source':'bc2f0ad7a435d0d04f859d683223c688f178db9e','outer':'96ab6b8c668f7631805a7a6eb78bf90fa924cce7ca280891c51ca02dd1b4c84f','worker':'4dd882c89745b946e42540dc46a62b704c821057c83d46c4ba78acc913fd9ffb'}
SOURCES=list(dict.fromkeys(L.SOURCES+['tools/continue_eigenpair_chain_bridge.py','tools/launch_eigenpair_chain_bridge.py','tests/test_eigenpair_chain_bridge.py','docs/plans/eigenpair-chain-bridge.md','docs/reviews/eigenpair-chain-lift-target.json']))

class BoundaryMembershipReached(RuntimeError):
    def __init__(self,membership):self.membership=membership;super().__init__('strict full470 native membership proved')

def install(args):
    require(type(args.unit)is int and 1<=args.unit<=3,'fixed4+4+2 chart bridge')
    expected=L.C.PARENT if args.unit==1 else BASE/f'unit{args.unit-1:03d}'
    require(args.parent==expected and args.out==BASE/f'unit{args.unit:03d}/run','fixed exclusive parent/output paths')
    require(args.expected_right==str(Fraction(275+8*(args.unit-1),128)),'exact branch predecessor')
    require(re.fullmatch('[0-9a-f]{40}',args.parent_source or '')and all(re.fullmatch('[0-9a-f]{64}',v or '')for v in (args.parent_manifest,args.parent_run_manifest)),'full parent pins')
    if args.unit==1:require([args.parent_source,args.parent_manifest,args.parent_run_manifest]==[L.C.PINS[k]for k in ('source','outer','worker')],'exact retained eventstop parent')

def complete_parent(out,result,accepted,full,expected):
    require(out['passed']is True and out['status']=='four_joined_charts'and result['completed']is True,'completed nonterminal predecessor')
    require(len(result['accepted'])==len(out['progress']['accepted'])==4 and out['progress']['active_call']is None,'four accepted native charts')
    last=out['progress']['last_accepted'];require(last==out['progress']['accepted'][-1],'last durable accepted predecessor')
    require(result['calls']==4+result['repairs']==out['calls'] and result['repairs']==out['repairs']and 0<=result['repairs']<=4,'complete bounded calls')
    require([r['call']for r in result['attempts']]==list(range(result['calls']))and sum(r['passed']for r in result['attempts'])==4,'all attempts retained')
    require(last['call']==accepted['call']==result['calls']-1 and last['index']==accepted['index']==3,'exact final native chart')
    c=B.accepted_check(full,expected);require(c==accepted['chart']==result['last_chart'],'full470 compact identity');return c

def lift_binding():
    info={'outer':L.inventory(LIFT,LPINS['outer']),'worker':L.inventory(LIFT/'run',LPINS['worker'])}
    rt=L.read(LIFT/'run/runtime.json');la=L.read(LIFT/'launcher-result.json');out=L.read(LIFT/'run/outcome.json');inp=L.read(LIFT/'run/inputs.json')
    require(rt['commit']==la['commit']==LPINS['source']and rt['mode']==la['mode']=='target','exact lift execution')
    require(la['exit_code']==0 and not la['owned_live_pids']and la['stop_reason']is None,'completed owned lift')
    require(out['passed']is True and out['status']=='authenticated_native_lift_completed'and out['ticks']==18,'full native target lift')
    require(rt['settings']==L.SETTINGS and rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'exact lift settings')
    require(set(rt['sources'])==set(L.SOURCES),'exact lift source inventory')
    for n,row in rt['sources'].items():
        require(entry(ROOT/n)==entry(LIFT/'run/source'/n)==row,'unchanged lift source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',LPINS['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical lift source '+n)
    lift=L.read(LIFT/'run/lift.json');require(lift['closure_contains_zero']and lift['parameter']==out['parameter'],'complete conditional lift')
    with ctx.workprec(384):
        s=decode(lift['parameter']);require(s>decode({'mid':[293,-7],'rad':[0,0]})and s<decode({'mid':[295,-7],'rad':[0,0]}),'fixed last-chart strict gauge coverage')
    require(inp['bindings']['pins']==L.PINS,'exact isolated proved root source')
    info.update(root=str(LIFT),pins=LPINS,lift_identity=entry(LIFT/'run/lift.json'),inputs_identity=entry(LIFT/'run/inputs.json'))
    return inp['map_inputs'],lift,info

def bind(args):
    model,lift,info=lift_binding();bindings={'lift':info}
    if args.unit==1:
        inp,_,_,native=L.C.bind_event();parent=L.read(T.C.input_path(args.parent,native['accepted_file']))['chart'];bindings['native_parent']=native
    else:
        bindings.update(parent_outer=L.inventory(args.parent,args.parent_manifest),parent_worker=L.inventory(args.parent/'run',args.parent_run_manifest))
        rt=L.read(args.parent/'run/runtime.json');la=L.read(args.parent/'launcher-result.json');out=L.read(args.parent/'run/outcome.json');pi=L.read(args.parent/'run/inputs.json')
        require(rt['commit']==la['commit']==args.parent_source and rt['unit']==args.unit-1 and rt['mode']==la['mode']=='target','immediate completed source/unit')
        require(la['exit_code']==0 and not la['owned_live_pids']and la['stop_reason']is None,'owned parent exit')
        require(rt['precision']==256 and rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10 and set(rt['sources'])==set(SOURCES),'exact native parent source/settings')
        for n,row in rt['sources'].items():
            require(entry(ROOT/n)==entry(args.parent/'run/source'/n)==row,'unchanged preceding bridge source '+n)
            require(hashlib.sha256(subprocess.check_output(['git','show',args.parent_source+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical bridge source '+n)
        last=out['progress']['last_accepted'];full=L.read(T.C.input_path(args.parent,last['full_chart_file']));accepted=L.read(T.C.input_path(args.parent,last['accepted_file']))
        parent=complete_parent(out,L.read(args.parent/'run/march.json'),accepted,full,args.expected_right);inp=pi['map_inputs']
        require(pi['parent']['parameter_endpoints'][1]==str(Fraction(args.expected_right)-Fraction(8,128)),'exact predecessor advancement')
        for k in ('normal','origin','period','sigma','dimension'):require(parent[k]==pi['parent'][k],'same predecessor gauge '+k)
        del full;gc.collect()
    require(inp==model,'same exact physical map as proved root')
    for k in ('normal','origin','period','sigma','dimension','shooting_dimension'):require(parent[k]==lift[k],'same exact native lift convention '+k)
    require(parent['parameter_endpoints'][1]==args.expected_right,'exact accepted right endpoint')
    reference=T.BASE/'attempt-001-narrower-target';L.inventory(reference,B.FIRST_PINS['parent_manifest']);L.inventory(reference/'run',B.FIRST_PINS['parent_run_manifest'])
    bindings['cost']=B.cost_gate(L.read(reference/'run/outcome.json'))
    bindings['parent']={'root':str(args.parent),'source':args.parent_source,'outer':args.parent_manifest,'worker':args.parent_run_manifest,'endpoint':args.expected_right}
    return inp,parent,lift,bindings

def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=400;rss=2*1024**2;cap=512*1024**2;reserve=8*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};lift=[None];progress={'active_call':None,'full_chart_file':None,'accepted':[],'last_accepted':None}
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
            with ctx.workprec(384):member=quadratic_membership(lift[0],value['chart'])
            save(f'membership-{value["index"]:03d}.json',member)
            if member['passed']:raise BoundaryMembershipReached(member)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'unit':args.unit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':1040,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'fixed native bridge to authenticated Jordan-chain root; strict full470 membership, no genericity or stability claim'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Native primitive18+1 locus joined to an authenticated isolated generalized-chain root only if full470 uniqueness membership passes; no generic unfolding or attraction claim.'}
    try:
        with ctx.workprec(256):
            inp,parent,lift[0],bindings=bind(args);save('bindings.json',bindings);accepted_cap=2 if args.unit==3 else 4
            save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-7,'accepted_cap':accepted_cap,'call_cap':2*accepted_cap,'repair_cap_per_chart':1,'objective_gauge':'295/128'})
            check();model=TwoRateCuspMap(inp)
            try:
                result=march(model,parent,accepted_cap=accepted_cap,tick=tick,observe=observe);save('march.json',result)
                outcome.update(passed=result['completed'] and args.unit<3,status=('target_gauge_reached_without_membership' if args.unit==3 else 'four_joined_charts') if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],repairs=result['repairs'])
            except BoundaryMembershipReached as event:
                save('membership-stop.json',{'membership':event.membership,'accepted':progress['accepted'],'scope':'The source-authenticated lifted root lies strictly in every native uniqueness coordinate and the chart gauge interior.'})
                outcome.update(passed=True,status='rigorous_chain_membership',membership=event.membership,accepted_count=len(progress['accepted']))
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
