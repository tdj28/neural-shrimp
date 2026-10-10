"""Fixed source-bound saved native704 unit-pair audit; no acquisition calls."""
import gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_unit_pair_math as U
from neural_shrimp.rigorous_orbits import canonical_bytes
need=U.need
OUT=Path('/data2/neural-shrimp-data/audits/native-unit-pair-independent/attempt001/run')
CARD='docs/plans/native-unit-pair-independent-inputs.json'
FILES=[CARD,'docs/plans/native-unit-pair-independent.md','tools/reviews/review_unit_pair_math.py','tools/reviews/review_unit_pair_driver.py','tools/reviews/launch_unit_pair_review.py','tests/reviews/test_unit_pair_review.py','tests/reviews/test_unit_pair_driver.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb') as f:data=f.read(256*1024**2+1)
        need(len(data)<=256*1024**2,'bounded256MiB decompression');return json.loads(data)
    return json.loads(p.read_bytes())

def archived(run,source,runtime,folder):
    rt=read(run/runtime);need(rt.get('commit',rt.get('source_commit'))==source,'exact historical execution')
    for p in (run/folder).rglob('*'):
        if not p.is_file():continue
        name=str(p.relative_to(run/folder));digest=hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest();need(O.entry(p)['sha256']==digest,'historical archived source '+name)
        if 'sources'in rt:need(O.entry(p)==rt['sources'][name],'runtime archive identity '+name)
    if 'sources'in rt:need(set(rt['sources'])=={str(p.relative_to(run/folder))for p in(run/folder).rglob('*')if p.is_file()},'entire runtime source inventory')
    need(not Path('/proc',str(rt['pid'])).exists(),'prior worker absent');return rt

def check_native_contract(done,rt,launch,card):
    need(rt['commit']==launch['commit']==card['producer']['source'] and launch['exit_code']==0 and launch['owned_live_pids']==[]and launch['stop_reason']is None,'complete owned original execution')
    need(rt['mode']=='target' and rt['settings']==card['settings'] and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1'),'same original arithmetic/native settings')
    need(done['passed']and done['status']=='qualified_unit_pair_tube'and done['strict_inclusions']==704 and done['parameter_endpoints']==card['exact_parameter_endpoints'],'complete producer native704 tube')
    need(done['events']=={'chart_return':1,'direction_return':216,'newton_proposal':2,'preconditioner_return':1}and done['ticks']==217,'complete original bounded acquisition inventory')

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10')
    card=read(ROOT/CARD);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False}
    def save(name,value):
        b=canonical_bytes(value);need(len(b)+sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())<24*1024**2,'24MiB output cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():need(not stop[0]and time.monotonic()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s/4GiB cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,300)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'model_calls':0,'derivative_calls':0})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        producer=card['producer'];pr=Path(producer['root']);O.verify(pr,producer['outer']);O.verify(pr/'run',producer['worker']);rt=archived(pr/'run',producer['source'],'runtime.json','source');done=read(pr/'run/outcome.json');launch=read(pr/'launcher-result.json');check_native_contract(done,rt,launch,card)
        chain=card['chain'];cr=Path(chain['root']);O.verify(cr,chain['outer']);O.verify(cr/'run',chain['worker']);archived(cr/'run',chain['source'],'runtime.json','audit-source');co=read(cr/'run/outcome.json');cla=read(cr/'launch-end.json')
        need(cla['source_commit']==chain['source']and cla['exit_code']==0 and cla['owned_group_empty']and cla['child_pid_absent'],'complete independent exact root launch')
        need(co['passed']and co['root_passed']and co['primitive_period']==18 and co['strict_inclusions']==41 and co['positive_actual_rates']and co['exact_algebraic_multiplicity_two'],'independently proved normalized exact Jordan root')
        root_box=read(cr/'run/fresh-box.json')['box'];cb=read(cr/'run/input-binding.json')
        con=card['connection'];xr=Path(con['root']);O.verify(xr,con['outer']);archived(xr/'run',con['source'],'runtime.json','source');xo=read(xr/'run/outcome.json');xb=read(xr/'run/input-bindings.json');xla=read(xr/'launcher-result.json')
        need(xo['passed']and xo['total_connected_charts']==107 and xo['native_coordinate_inclusions']==470 and xb['chain']==[chain['root'],chain['outer'],chain['worker'],chain['source']],'same root independently on107 native locus')
        need(xla['source_commit']==con['source']and xla['exit_code']==0 and xla['owned_live_pids']==[],'complete independent native connection launch')
        inp=read(pr/'run/inputs.json');lift=read(pr/'run/native-lift.json');ci=read(pr/'run/chart-inputs.json');repair=read(pr/'run/repair-binding.json');bind=inp['bindings']['audited_root']
        need(bind['root']==chain['root']and bind['pins']=={k:chain[k]for k in('source','worker','outer')}and bind['fresh_identity']==O.entry(cr/'run/fresh-box.json'),'exact prior independently enlarged root identity')
        need(inp['map_inputs']==cb['map_inputs'],'identical actual physical network and data')
        need(len(inp['original_41d_box'])==len(root_box)==41,'complete41D prior box')
        for outer,inner in zip(inp['original_41d_box'],root_box,strict=True):U.contains(outer,inner,'original lift input covers independent41D root')
        need(ci['settings']==rt['settings']==repair['settings']==card['settings'] and ci['weights']==repair['exact_weights'] and ci['anchor']==lift,'same exact fixed704 anchor/radii/settings')
        cert=pr/'run/certificate.json.gz';need(O.entry(cert)['sha256']==producer['certificate']==O.entry(pr/'run/events/0219-chart_return.json.gz')['sha256'],'exact single complete chart payload')
        paths=sorted((pr/'run/events').glob('*-direction_return.json.gz'));need(len(paths)==216,'all216 native directions')
        events=[read(p)for p in paths];domain=U.check_lift(root_box,lift,events[:36]);save('native-lift-domain.json',domain);save('input-bindings.json',{'card':card,'certificate':O.entry(cert),'native_direction_files':{str(p.relative_to(pr/'run')):O.entry(p)for p in paths}});tick()
        c=read(cert);need(c['weights']==ci['weights']and c['settings']=={k:card['settings'][k]for k in('precision','newton_steps','halfwidth_exponent','radius_exponent')},'exact repair weights and chart geometry')
        # Fresh assembly makes these producer reductions unnecessary; retain
        # their immutable source, not a second192MB certificate in memory.
        retain={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','weights','coordinate_radii','preconditioner','whole_box','divisors'}
        for key in list(c):
            if key not in retain:del c[key]
        gc.collect();tick();proof=U.check_chart(c,lift['box'],events,384,tick=tick);save('native704-384.json',proof);tick()
        need(proof['passed']and proof['inclusions']==proof['anchor_coordinates']==704,'independent all704 strict proof and anchor')
        result.update(passed=True,primitive_period=18,parameter_endpoints=card['exact_parameter_endpoints'],strict_inclusions=proof['inclusions'],anchor_coordinates=proof['anchor_coordinates'],contraction=proof['contraction'],same_independently_connected_root=True,scope=proof['scope'])
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
