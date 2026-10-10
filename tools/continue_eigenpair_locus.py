#!/usr/bin/env python3
"""One bounded four-chart continuation, with a synthetic proposal cost screen."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np,flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import certify_eigenpair_locus_quadratic as Q
from neural_shrimp.rigorous_eigenpair_march import compact,march,proposed_radii,midpoint_spectrum
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import ball_json
PARENT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-003-quadratic-target')
PARENT_COMMIT='8bb19e4abc2b02aaff91fc11f13c26853fbb3071'
PARENT_PIN='cd181db39a626724dc6307974f97a99efdb311358523101778c58ba3e5a0059c'
PARENT_RUN='cbb8445ec92f4d91369fa66827ef6ea81ced600ac4c33519d9dafd8aec0b6db9'
PARENT_RESULT='214e5c42a17d95d0d62dd8baf284beaf453f76797c549914322027fc170e936d'
SOURCES=list(dict.fromkeys(Q.SOURCES+['src/neural_shrimp/rigorous_eigenpair_march.py','tools/continue_eigenpair_locus.py','tools/launch_eigenpair_march.py','tests/test_eigenpair_march.py','docs/plans/eigenpair-quadratic-march.md','docs/reviews/eigenpair-quadratic-outcome.json']))
entry=Q.entry;read=Q.read


def finite(value,cap):return type(value) in (int,float) and math.isfinite(value) and 0<=value<cap

def cost_gate(outcome,runtime,launcher,profile):
    require(outcome['passed'] is True and len(outcome['charts'])==2 and outcome['ticks']==272,'two passed quadratic charts')
    require(runtime['commit']==launcher['commit']==PARENT_COMMIT and launcher['exit_code']==0 and not launcher['owned_live_pids'] and launcher['stop_reason'] is None,'parent source/exit')
    require(runtime['settings']==Q.SETTINGS and runtime['mode']=='target' and runtime['nice']>=10 and runtime['threads']==dict.fromkeys(THREADS,'1'),'parent exact settings')
    require(finite(outcome['elapsed_seconds'],120) and finite(outcome['peak_rss_kib'],2*1024**2) and finite(launcher['elapsed_seconds'],142),'parent finite resources')
    require(profile['passed'] is True and profile['native_calls']==0 and finite(profile['elapsed_seconds'],20) and finite(profile['peak_rss_kib'],512*1024),'synthetic proposal cost profile')
    seconds=5*outcome['elapsed_seconds']+max(20.,4*profile['elapsed_seconds'])
    storage=5*outcome['stored_bytes_before_final']+8*1024**2
    peak=outcome['peak_rss_kib']+128*1024
    require(seconds<300 and storage<760*1024**2 and peak<2*1024**2,'measured continuation admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':peak,'formula':'2x five chart calls from measured two-chart unit; max20s/four profile elapsed overhead;2x five per-chart raw plus8MiB','scope':'Heuristic cost screen; actual resource limits remain mandatory.'}


def bind_parent(profile_root,profile_pin,profile_outer_pin):
    inputs,_,bindings,_=Q.bind_inputs()
    bindings['parent']=Q.B.inventory(PARENT,PARENT_PIN);bindings['parent_run']=Q.B.inventory(PARENT/'run',PARENT_RUN)
    rt=read(PARENT/'run/runtime.json');launch=read(PARENT/'launcher-result.json');outcome=read(PARENT/'run/outcome.json')
    require(entry(PARENT/'run/result.json.gz')['sha256']==PARENT_RESULT,'fixed parent result')
    for n in Q.SOURCES:
        require(entry(ROOT/n)==entry(PARENT/'run/source'/n)==rt['sources'][n],'unchanged parent mathematical/source file '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',PARENT_COMMIT+':'+n],cwd=ROOT)).hexdigest()==rt['sources'][n]['sha256'],'parent source commit '+n)
    bindings['profile']=Q.B.inventory(profile_root,profile_pin)
    bindings['profile_outer']=Q.B.inventory(profile_root.parent,profile_outer_pin)
    pr=read(profile_root/'runtime.json');po=read(profile_root/'outcome.json')
    pl=read(profile_root.parent/'launcher-result.json')
    require(pr['mode']==pl['mode']=='profile' and pr['precision']==256 and pr['nice']>=10 and pl['commit']==pr['commit'] and pl['exit_code']==0 and not pl['owned_live_pids'] and pl['stop_reason'] is None and finite(pl['elapsed_seconds'],32),'profile mode/nice/owned-exit')
    for k in ('python','numpy','flint','flint_library','threads'):
        expected={'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}[k]
        require(pr[k]==rt[k]==expected,'library/thread equality '+k)
    for n in SOURCES:
        require(entry(ROOT/n)==entry(profile_root/'source'/n)==pr['sources'][n],'unchanged profile source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',pr['commit']+':'+n],cwd=ROOT)).hexdigest()==pr['sources'][n]['sha256'],'profile committed source '+n)
    bindings['measured_cost']=cost_gate(outcome,rt,launch,po)
    with gzip.open(PARENT/'run/result.json.gz','rt') as f:result=json.load(f)
    require(result['passed'] is True and len(result['charts'])==2,'parent result')
    first,last=result['charts'];require(first['parameter_endpoints'][1]==last['parameter_endpoints'][0] and all(r['passed'] and all(r['gates'].values()) for r in result['charts']),'joined passed parent')
    require(last['period']==18 and last['sigma']==1 and last['dimension']==13 and last['settings']=={k:Q.SETTINGS[k] for k in ('precision','newton_steps','halfwidth_exponent','radius_exponent')},'parent native settings')
    require(last['normal']==['0']*469+['1'] and last['parameter_endpoints'][1]=='3/32768','fixed endpoint/gauge')
    previous=compact(last);del result,first,last;gc.collect()
    require(read(PARENT/'run/exact-map-inputs.json')==inputs['map_inputs'],'same exact model')
    return inputs['map_inputs'],previous,bindings


def synthetic_profile(observe):
    # Fixed 470D nonnegative contraction: diagonal1/4, superdiagonal1/8.
    z=ball_json(arb(0));one=ball_json(arb(1));b=ball_json(arb('0.25'));c=ball_json(arb('0.125'));f=ball_json(arb((1,-25)));size=470
    row={'gates':{'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'preconditioner':[[one if i==j else z for j in range(size)] for i in range(size)],'defect':[[b if i==j else c if j==i+1 else z for j in range(size)] for i in range(size)],'predictor_residual':[[f] for _ in range(size)]}
    observe('synthetic_recipe',{'size':470,'C':'identity','D':'diagonal1/4,superdiagonal1/8','R':'constant2^-25','native_calls':0})
    proposal=proposed_radii(row,observe);require(len(proposal['weights'])==470 and max(proposal['predicted_old_domain_ratios'])<1,'known positive resolvent control')
    n=13;z=ball_json(arb(0));j=[[z for _ in range(28)] for _ in range(28)]
    # period1 state block is A-I; two +1 modes must be flagged descriptively.
    for i in range(n):j[i][i]=ball_json(arb(0 if i<2 else -.5))
    chart={'dimension':n,'period':1,'sigma':1,'center_jacobian':j,'center':[z]*n+[one]+[z]*(n-1)+[one,one]}
    spectrum=midpoint_spectrum(chart,observe);observe('synthetic_spectrum',spectrum);require(spectrum['complementary_near_critical_flag'],'known complementary+1 control')
    return {'passed':True,'native_calls':0,'proposal_dimension':470,'eig_dimension':13,'scope':'Synthetic proposal/diagnostic cost only; no target acquisition.'}


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
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':650,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'+ output-rate gauge toward retained cusp-a region; no assumed distant association'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            if not target:outcome.update(synthetic_profile(observe),status='synthetic_profile_completed')
            else:
                require(args.profile is not None and args.profile_manifest is not None and args.profile_outer_manifest is not None,'profile pin required')
                inp,parent,bindings=bind_parent(args.profile,args.profile_manifest,args.profile_outer_manifest);save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'accepted_cap':4,'kernel_call_cap':5,'rebox_cap':1})
                check();model=TwoRateCuspMap(inp);result=march(model,parent,tick=tick,observe=observe);save('march.json',result)
                outcome.update(passed=result['completed'],status='four_joined_charts' if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],rebox_used=result['rebox_used'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');raise SystemExit(main(p.parse_args()))
