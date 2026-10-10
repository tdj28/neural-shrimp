#!/usr/bin/env python3
"""One fixed saved-only radius diagnosis after the explicit unit013 bound stop."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import continue_eigenpair_repair_chain as R
from neural_shrimp.eigenpair_radius_diagnostic import diagnose
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
M=R.M;T=R.T;entry=M.entry
PARENT=T.BASE/'attempt-013-wider-march'
PARENT_SOURCE='a52c4c99857cdac3562da3858dac1ec20c5d1259'
PARENT_OUTER='e8483b59300401f36bfd71d33942f652d2e2c94480bebe84c1965e50875421c0'
PARENT_RUN='9b6ae04c6157cf17cbc0c7213657417e25072716074a051a058ad35c9391bb7e'
QUALIFIED_FILE='events/0276-chart_return.json.gz'
ACCEPTED_FILE='events/0280-march_accepted.json.gz'
FAILED_FILE='events/0557-chart_return.json.gz'
EXTRA=['tools/diagnose_eigenpair_partial.py','tools/launch_eigenpair_partial_diagnosis.py','tests/test_eigenpair_partial.py','docs/plans/eigenpair-partial-diagnosis.md','docs/reviews/eigenpair-wider-march-unit013.json']
SOURCES=list(dict.fromkeys(R.SOURCES+EXTRA))


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def validate_prefix(out,result,accepted,qualified,failed,anchor):
    require(out['passed'] is False and out['status']=='qualified_prefix_then_unresolved' and result['completed'] is False,'exact bounded partial status')
    require(out['calls']==result['calls']==4 and out['repairs']==result['repairs']==2,'four calls two repairs')
    progress=out['progress'];last=progress['last_accepted']
    require(len(progress['accepted'])==len(result['accepted'])==1 and last==progress['accepted'][0],'one durable accepted prefix')
    require(last['call']==accepted['call']==1 and last['index']==accepted['index']==0 and last['right']=='219/128','accepted call1 only')
    require(last['full_chart_file']==QUALIFIED_FILE and last['accepted_file']==ACCEPTED_FILE,'fixed durable accepted artifacts')
    require(progress['active_call']==3 and progress['full_chart_file']==FAILED_FILE,'durable failed tail record, not a live operation')
    require([v['call']for v in result['attempts']]==[0,1,2,3] and [v['passed']for v in result['attempts']]==[False,True,False,False],'exact ordered complete attempt records')
    require(result['accepted'][0]==result['attempts'][1],'accepted summary is proved call1')
    require(qualified['passed'] and all(qualified['gates'].values()) and qualified==accepted['chart']==result['last_chart'],'qualified full/compact identity')
    require(qualified['parameter_endpoints']==['215/128','219/128'] and anchor['parameter_endpoints'][1]=='215/128','proved prefix endpoints')
    require(failed['passed'] is False and failed['gates']=={'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'failed root only, never parent')
    require(failed['parameter_endpoints']==['219/128','223/128'] and len(failed['coordinate_inclusions'])==470 and [i for i,v in enumerate(failed['coordinate_inclusions'])if not v]==[364],'fixed final failed coordinate')
    for key,val in [('period',18),('dimension',13),('sigma',1),('shooting_dimension',470)]:
        require(qualified[key]==failed[key]==anchor[key]==val,'native identity '+key)
    for key in ('normal','origin'):require(qualified[key]==failed[key]==anchor[key],'same native gauge '+key)
    for obj in (qualified,failed):
        require(obj['settings']['precision']==256 and obj['settings']['newton_steps']==2 and obj['settings']['halfwidth_exponent']==-6,'unchanged scientific settings')
    return qualified


def bind_all():
    bindings={'outer':M.Q.B.inventory(PARENT,PARENT_OUTER),'run':M.Q.B.inventory(PARENT/'run',PARENT_RUN),'execution':PARENT_SOURCE}
    rt=M.read(PARENT/'run/runtime.json');la=M.read(PARENT/'launcher-result.json');out=M.read(PARENT/'run/outcome.json')
    require(rt['commit']==la['commit']==PARENT_SOURCE and rt['mode']==la['mode']=='target' and rt['unit']==13 and rt['precision']==256 and rt['nice']>=10,'exact partial execution/settings')
    require(la['exit_code']==2 and la['stop_reason'] is None and not la['owned_live_pids'],'retained scientific gate failure')
    require(out['ticks']==508 and M.finite(out['elapsed_seconds'],400) and M.finite(out['peak_rss_kib'],2*1024**2) and M.finite(la['elapsed_seconds'],427),'prior finite resource success')
    require(set(rt['sources'])==set(R.SOURCES),'complete original source inventory')
    for n,v in rt['sources'].items():
        require(entry(ROOT/n)==entry(PARENT/'run/source'/n)==v,'unchanged source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',PARENT_SOURCE+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'git source '+n)
    for k,v in {'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same environment '+k)
    pi=M.read(PARENT/'run/inputs.json');result=M.read(PARENT/'run/march.json');accepted=read_gzip(PARENT/'run'/ACCEPTED_FILE)
    full=read_gzip(PARENT/'run'/QUALIFIED_FILE)
    require(len(full['coordinate_inclusions'])==470 and all(full['coordinate_inclusions']) and full['anchor_check']['passed'],'all strict prefix gates')
    require([v['divisor']for v in full['divisors']]==[1,2,3,6,9] and all(v['excluded']for v in full['divisors']),'primitive prefix')
    qualified=compact(full);del full;gc.collect()
    failed=read_gzip(PARENT/'run'/FAILED_FILE)
    qualified=validate_prefix(out,result,accepted,qualified,failed,pi['parent'])
    bindings.update(qualified_file=QUALIFIED_FILE,qualified_identity=entry(PARENT/'run'/QUALIFIED_FILE),failed_file=FAILED_FILE,failed_identity=entry(PARENT/'run'/FAILED_FILE),qualified_right='219/128',failed_interval=['219/128','223/128'],scope='Failed chart used only for radius proposal. Accepted native prefix remains the sole future anchor.')
    return failed,qualified,pi['map_inputs'],bindings


def bind():
    failed,qualified,inputs,bindings=bind_all()
    return failed,bindings


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
