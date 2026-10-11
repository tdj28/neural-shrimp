#!/usr/bin/env python3
"""Exact endpoint-pinned bounded march in the reviewed full-state tangent gauge."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,re,subprocess,sys,math,resource,signal,time
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import reanchor_eigenpair_locus as R
M=R.M;C=R.C;BASE=C.BASE
from flint import ctx
import numpy as np,flint
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.eigenpair_gauge_diagnostics import saved_predictor_diagnostics
ANCHOR=BASE/'attempt-001-reanchor-target'
ANCHOR_SOURCE='2cd9e2dc43c48a920d8156d56f9e74a38fc65f04'
ANCHOR_OUTER='5a344e0c0b891d6051c62f54e3138360e9a385f2c7f9133f40d8f133386ea7c8'
ANCHOR_RUN='2147813200b2c5cc90132baccafcee13fd4bce0bb9bd6a93d972d036d0af2171'
EXTRA=['src/neural_shrimp/eigenpair_gauge_diagnostics.py','tools/continue_eigenpair_tangent_chain.py','tools/launch_eigenpair_tangent_chain.py','tests/test_eigenpair_tangent_chain.py','docs/plans/eigenpair-tangent-march.md','docs/reviews/eigenpair-reanchor-outcome.json']
SOURCES=list(dict.fromkeys(R.SOURCES+EXTRA))


def source_runtime(path,commit,allowed):
    rt=M.read(path/'run/runtime.json');la=M.read(path/'launcher-result.json');out=M.read(path/'run/outcome.json')
    require(rt['commit']==la['commit']==commit and rt['mode']==la['mode']=='target' and rt['precision']==256 and rt['nice']>=10,'source/runtime settings')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and M.finite(la['elapsed_seconds'],322),'owned successful exit')
    require(out['passed'] is True and M.finite(out['elapsed_seconds'],300) and M.finite(out['peak_rss_kib'],2*1024**2),'passed finite resource outcome')
    require(set(rt['sources'])==set(allowed),'exact parent source inventory')
    for name,v in rt['sources'].items():
        require(M.entry(ROOT/name)==M.entry(path/'run/source'/name)==v,'unchanged parent source '+name)
        require(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'committed parent source '+name)
    return rt,la,out


def profile_and_cost(args,parent_out,parent_rt,bindings):
    bindings['profile']=M.Q.B.inventory(args.profile,args.profile_manifest);bindings['profile_outer']=M.Q.B.inventory(args.profile.parent,args.profile_outer_manifest)
    rt=M.read(args.profile/'runtime.json');la=M.read(args.profile.parent/'launcher-result.json');out=M.read(args.profile/'outcome.json')
    require(rt['mode']==la['mode']=='profile' and rt['commit']==la['commit'] and rt['precision']==256 and rt['nice']>=10 and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and M.finite(la['elapsed_seconds'],32),'profile successful settings/exit')
    require(out['passed'] is True and out['native_calls']==0 and M.finite(out['elapsed_seconds'],20) and M.finite(out['peak_rss_kib'],512*1024),'synthetic profile only')
    expected={'python':sys.version,'numpy':M.np.__version__,'flint':M.flint.__version__,'flint_library':M.flint.__FLINT_VERSION__,'threads':dict.fromkeys(M.THREADS,'1')}
    for k,v in expected.items():require(rt[k]==parent_rt[k]==v,'exact libraries/threads '+k)
    require(set(rt['sources'])==set(SOURCES),'exact profile inventory')
    for name,v in rt['sources'].items():
        require(M.entry(ROOT/name)==M.entry(args.profile/'source'/name)==v,'profile/current source '+name)
        require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'committed profile '+name)
    gate=cost_gate(parent_out,out);bindings['cost_gate']=gate


def cost_gate(anchor_out,profile):
    require(anchor_out['passed'] is True and anchor_out['ticks']==326 and anchor_out['status']=='two_reanchored_charts','exact reference reanchor cost')
    require(M.finite(anchor_out['elapsed_seconds'],180) and M.finite(anchor_out['peak_rss_kib'],2*1024**2) and M.finite(anchor_out['stored_bytes_before_final'],512*1024**2),'finite measured reanchor resources')
    require(profile['passed'] is True and profile['native_calls']==0 and M.finite(profile['elapsed_seconds'],20) and M.finite(profile['peak_rss_kib'],512*1024),'synthetic overhead profile')
    seconds=2*anchor_out['elapsed_seconds']*635/326+max(20,4*profile['elapsed_seconds'])
    storage=5*anchor_out['stored_bytes_before_final']+8*1024**2;peak=anchor_out['peak_rss_kib']+128*1024
    require(seconds<300 and storage<760*1024**2 and peak<2*1024**2,'measured tangent march admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':peak,'formula':'2x measured326tick reanchor scaled to635maximum marchticks + max20s/4profile;5complete reanchor byte bundles+8MiB;observed reanchorRSS+128MiB. Heuristic only, actual fixed caps remain.'}


def validate_parent(out,result,accepted,full,expected_right,anchor):
    require(out['passed'] is True and out['status']=='four_joined_charts' and result['completed'] is True and len(result['accepted'])==len(out['progress']['accepted'])==4,'complete four-chart unit only')
    last=out['progress']['last_accepted'];require(last==out['progress']['accepted'][-1] and out['progress']['active_call'] is None,'durable complete parent')
    require(4<=result['calls']<=5 and result['calls']==out['calls'] and accepted['call']==last['call'] and accepted['index']==last['index']==3,'call/index join')
    cc=compact(full);require(cc==accepted['chart']==result['last_chart'],'full/compact/summary identity')
    for key in ('period','sigma','dimension','shooting_dimension','normal','origin'):
        require(cc[key]==anchor[key],'same native gauge '+key)
    require(cc['settings']['halfwidth_exponent']==-7 and cc['settings']['precision']==256 and cc['settings']['newton_steps']==2,'same width/precision')
    require(cc['parameter_endpoints'][1]==last['right']==expected_right,'exact pinned endpoint')
    return cc


def bind(args):
    # Bind original exact neural map and original certificate/profile ancestry.
    inputs,_,bindings,_=M.Q.bind_inputs();inp=inputs['map_inputs']
    bindings['tangent_anchor_outer']=M.Q.B.inventory(ANCHOR,ANCHOR_OUTER);bindings['tangent_anchor_run']=M.Q.B.inventory(ANCHOR/'run',ANCHOR_RUN)
    art,_,ao=source_runtime(ANCHOR,ANCHOR_SOURCE,R.SOURCES)
    ar=M.read(ANCHOR/'run/reanchor.json');require(ar['passed'] is True and len(ar['charts'])==2 and len(ao['progress']['chart_files'])==2,'two passed reanchor charts')
    with gzip.open(C.input_path(ANCHOR,ao['progress']['chart_files'][-1]),'rt') as f:full=json.load(f)
    anchor=compact(full);require(anchor==ar['charts'][-1] and anchor['parameter_endpoints'][1]=='3/128','exact full reanchor endpoint')
    require(anchor['period']==18 and anchor['sigma']==1 and anchor['dimension']==13 and anchor['normal']!=['0']*469+['1'],'native full-state anchor')
    require(M.read(ANCHOR/'run/inputs.json')['map_inputs']==inp,'same exact original model/data')
    profile_and_cost(args,ao,art,bindings)
    if args.unit==1:
        require(args.parent==ANCHOR and args.parent_source==ANCHOR_SOURCE and args.parent_manifest==ANCHOR_OUTER and args.parent_run_manifest==ANCHOR_RUN and args.expected_right=='3/128','fixed first anchor pins')
        previous=anchor
    else:
        require(args.parent==BASE/f'attempt-{args.unit-1:03d}-tangent-march-target','exact preceding unit')
        bindings['parent_outer']=M.Q.B.inventory(args.parent,args.parent_manifest);bindings['parent_run']=M.Q.B.inventory(args.parent/'run',args.parent_run_manifest)
        rt,_,out=source_runtime(args.parent,args.parent_source,SOURCES)
        for k in ('python','numpy','flint','flint_library','threads'):require(rt[k]==art[k],'same parent environment '+k)
        result=M.read(args.parent/'run/march.json');last=out['progress']['last_accepted']
        with gzip.open(C.input_path(args.parent,last['accepted_file']),'rt') as f:accepted=json.load(f)
        with gzip.open(C.input_path(args.parent,last['full_chart_file']),'rt') as f:full=json.load(f)
        previous=validate_parent(out,result,accepted,full,args.expected_right,anchor)
        pi=M.read(args.parent/'run/inputs.json');require(pi['map_inputs']==inp and pi['parent']['parameter_endpoints'][1]==str(Fraction(args.expected_right)-Fraction(8,128)),'exact prior4chart input advance')
    bindings['selected_parent']={'path':str(args.parent),'execution':args.parent_source,'outer':args.parent_manifest,'run':args.parent_run_manifest,'right':args.expected_right}
    del full,ar;gc.collect();return inp,previous,bindings


def install(args):
    if args.mode=='target':
        require(type(args.unit)is int and 1<=args.unit<=64,'bounded explicit unit')
        require(args.expected_right==str(Fraction(8*(args.unit-1)+3,128)),'prospective exact tangent endpoint')
        require(args.parent is not None and not args.parent.is_symlink() and re.fullmatch('[0-9a-f]{40}',args.parent_source or '') and all(re.fullmatch('[0-9a-f]{64}',v or '') for v in (args.parent_manifest,args.parent_run_manifest)),'explicit parent pins')


def marched(model,parent,**kw):
    observer=kw['observe']
    def observed(kind,value):
        observer(kind,value)
        if kind=='chart_return':
            try:diag=saved_predictor_diagnostics(value)
            except (ValueError,OverflowError,ZeroDivisionError) as exc:diag={'unresolved':str(exc),'scope':'saved predictor diagnostic only'}
            observer('tangent_projection_return',diag)
    return M.march(model,parent,**dict(kw,observe=observed))


THREADS=M.THREADS;entry=M.entry;synthetic_profile=M.synthetic_profile


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=args.mode=='target';limit=300 if target else 20;rss=(2*1024**2 if target else 512*1024);cap=(768 if target else 8)*1024**2;reserve=(8 if target else 1)*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'accepted':[],'last_accepted':None,'active_call':None,'full_chart_file':None}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,((4 if target else 1)*1024**3,)*2)
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
        ticks[0]+=1;check();require(ticks[0]<=650,'fixed650tick cap')
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
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':650,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'positive fixed full-state tangent gauge from authenticated reanchor; no assumed cusp association'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            if not target:outcome.update(synthetic_profile(observe),status='synthetic_profile_completed')
            else:
                require(args.profile is not None and args.profile_manifest is not None and args.profile_outer_manifest is not None,'profile pin required')
                inp,parent,bindings=bind(args);save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'accepted_cap':4,'kernel_call_cap':5,'rebox_cap':1})
                check();model=TwoRateCuspMap(inp);result=marched(model,parent,tick=tick,observe=observe);save('march.json',result)
                outcome.update(passed=result['completed'],status='four_joined_charts' if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],rebox_used=result['rebox_used'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2


def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--unit',type=int);p.add_argument('--parent',type=Path);p.add_argument('--parent-source');p.add_argument('--parent-manifest');p.add_argument('--parent-run-manifest');p.add_argument('--expected-right');return p
if __name__=='__main__':
    args=parser().parse_args();install(args);raise SystemExit(main(args))
