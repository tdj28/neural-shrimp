"""Independent one-piece audit of the three retained historical refinement stages."""
import gc,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_s2_attraction_tail as Q
from neural_shrimp.rigorous_orbits import canonical_bytes
O,L,T,N=Q.O,Q.L,Q.T,Q.N
read,need=Q.read,Q.need
OUT=Path('/data2/neural-shrimp-data/audits/s2-attraction-reused/attempt001/run')
CONFIG='docs/plans/s2-attraction-reused-inputs.json'
FILES=[CONFIG,'docs/plans/s2-attraction-reused.md','tools/reviews/review_s2_attraction_reused.py','tools/reviews/launch_s2_attraction_reused.py','tests/reviews/test_attraction_reused_review.py']
FILES=sorted(set(FILES+Q.FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def contracts(meta,card,inputs):
    b,t,r=meta['base'],meta['tangent'],meta['refined'];spec=card['stages'];ends=card['exact_endpoints'];bi,ri=inputs
    need(ends==['359/1024','45/128'] and card['halfwidth']=='1/2048' and bi['exact_endpoints']==ri['exact_endpoints']==ends and ri['halfwidth']==card['halfwidth'],'fixed exact reused interval')
    need(b['runtime']['mode']=='target'and b['outcome']['status']=='fixed_slice_bounds_completed'and b['outcome']['result_status']=='complete_slice_bounds'and b['outcome']['passed']is False,'preserve completed insufficient original bound')
    need(b['outcome']['stages']==85 and b['outcome']['completed_intersections']==6 and b['outcome']['completed_physical_phases']==b['outcome']['completed_products']==18,'all original native stages')
    need(b['outcome']['adapter_counts']=={'map_attempted':162,'map_cap':162,'map_returned':162,'unit_attempted':324,'unit_cap':324,'unit_returned':324}and b['outcome']['mixed_counts']=={'attempted':18,'returned':18,'cap':18},'full original native acquisition counters')
    need(t['runtime']['parent_manifest']==spec['base']['worker_manifest']['sha256']and t['outcome']['status']=='saved_tangent_variation_completed'and t['outcome']['conditional_bound_passed']and t['runtime']['model_calls']==t['runtime']['derivative_calls']==0,'exact saved-only second-derivative dependency')
    need(r['runtime']['mode']=='target'and r['runtime']['tangent_manifest']==spec['tangent']['worker_manifest']['sha256']and r['outcome']['status']=='fixed_refined_cocycle_completed'and r['outcome']['passed']and r['outcome']['stages']==75 and r['outcome']['mixed_counts']=={'attempted':18,'returned':18,'cap':18}and r['runtime']['model_C1_calls']==0,'exact18 new mixed acquisitions')
    need(ri['bindings']=={'original_manifest':spec['base']['worker_manifest']['sha256'],'tangent_manifest':spec['tangent']['worker_manifest']['sha256']}and ri['model_inputs']==bi['model_inputs'],'unchanged exact original model and refinement identities')

def authenticate(spec):
    root=Path(spec['root']);run=root/'run';O.verify(root,spec['outer_manifest']['sha256']);O.verify(run,spec['worker_manifest']['sha256']);rt=read(run/'runtime-start.json');done=read(run/'outcome.json');launch=read(root/'launcher-result.json');source=spec['source_commit']
    need(rt['source_commit']==launch['source_commit']==source and launch['exit_code']==done['exit_code']==0 and launch['owned_live_pids']==[],'complete historical source and owned exit')
    need(rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1'),'original arithmetic and bounded threads')
    for name,pin in rt['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest()==pin,'exact historical committed source '+name)
    for pid in(rt['pid'],launch['launcher_pid'],launch['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'original owned processes absent')
    return {'runtime':rt,'outcome':done},run

def main():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False}
    def save(name,value):
        b=canonical_bytes(value);need(len(b)+sum(p.stat().st_size for p in OUT.rglob('*')if p.is_file())<8*1024**2,'8MiB worker output cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():need(not stop[0]and time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'60s/2GiB worker cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,60)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'precision':[256,384],'model_calls':0,'derivative_calls':0,'sources':{name:O.entry(ROOT/name)for name in FILES}})
    for name in FILES:p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    try:
        card=read(ROOT/CONFIG);meta={};runs={}
        for key,spec in card['stages'].items():meta[key],runs[key]=authenticate(spec)
        b,t,r=runs['base'],runs['tangent'],runs['refined'];inp=read(b/'inputs.json.gz');ri=read(r/'inputs.json.gz');contracts(meta,card,[inp,ri]);c=Q.bind_parent(inp,card['native_parent'])
        orig=read(ROOT/'certificates/s2-period18-receipt.json');mp=Path(orig['external_root'])/'inputs.json';need(O.entry(mp)['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f'and inp['model_inputs']==read(mp),'original exact central training map');save('input-binding.json',card)
        def stages(run,kind):return[read(p)for p in sorted(run.glob(f'stage-*-{kind}.json.gz'))]
        def one(run,kind):
            rows=stages(run,kind);need(len(rows)==1,'single complete stage '+kind);return rows[0]
        base=read(b/'slice-result.json.gz');need(base['status']=='complete_slice_bounds'and base['exact_endpoints']==card['exact_endpoints']and not base['passed'],'immutable original insufficient bound')
        ws=one(b,'whole_slice_start');wr=stages(b,'whole_slice_intersection');cs=one(b,'center_slice_start');cr=stages(b,'center_slice_intersection');actual=one(b,'physical_inputs')
        whole=L.localization(c,ws,wr,base['whole_localization']);center=L.localization(c,cs,cr,base['center_localization'])
        need(actual['exact_endpoints']==card['exact_endpoints']and actual['center_parameter']==str(sum(map(Fraction,card['exact_endpoints']))/2),'same exact native center')
        for key,known in [('actual_whole_box',whole['final_state_box']),('actual_center_box',center['final_state_box'])]:
            need(len(actual[key])==235,'full physical state and hidden rate')
            for a,z in zip(actual[key],known,strict=True):L.contains(a,z,'actual domains contain entire known root')
        tangent=L.tangent(ws['preconditioner'],wr[-1]['jacobian'],c['exact_weights'],base['uniform_tangent'],actual['actual_uniform_tangent']);saved=one(b,'uniform_tangent')
        need(saved['bound']==base['uniform_tangent']and saved['root_derivative_domain']==wr[-1]['derivative_domain']and saved['root_final_state_box']==whole['final_state_box'],'original whole first tangent binding')
        whole.pop('final_state_box');center.pop('final_state_box');gc.collect();tick()
        phases=stages(b,'phase_derivatives');refine=read(t/'tangent-result.json.gz');refactual=one(r,'actual_inputs');refphases=stages(r,'refined_phase')
        need(ri['whole_box']==actual['actual_whole_box']and ri['center_J']==[v['center_J']for v in phases],'same original whole domain and complete native center derivatives')
        for a,z in zip(refactual['actual_uniform_tangent'],ri['whole_tangent'],strict=True):L.contains(a,z,'actual restored tangent includes complete saved input')
        refinement=T.audit(c,ws,wr[-1]['jacobian'],cr[-1]['jacobian'],base['uniform_tangent'],actual,phases,refine,refactual)
        need(len(phases)==len(refphases)==18 and [v['phase']for v in refphases]==list(range(18))and all(x['center_J']==y['center_J']for x,y in zip(phases,refphases,strict=True)),'all18 same native center phases')
        frames=one(r,'frames');need(frames==one(b,'frames'),'identical certified cyclic frames');normphases=stages(r,'norm_phase');metric=one(r,'norm_metric');answer=read(r/'result.json.gz')
        need(answer['passed']and answer['metric']==metric and answer['exact_endpoints']==card['exact_endpoints'],'same passed final metric and native interval')
        with ctx.workprec(384):norm=N.audit(refphases,frames,normphases,metric,Fraction(card['halfwidth']))
        need(norm['passed'],'independent strict native attraction');save('slice-384.json',{'passed':True,'exact_endpoints':card['exact_endpoints'],'whole_localization':whole,'center_localization':center,'uniform_tangent':tangent,'refinement':refinement,'norm':norm});tick()
        result.update(passed=True,completed_slices=1,coverage=card['exact_endpoints'],primitive_period=18,original_base_passed=False,original_refined_passed=True,scope='Independent256 known-root/U′/U″/refined-tangent and384 full native norm arithmetic on one retained historical closed piece. Shared source-authenticated analytic acquisitions and Arb; no whole-window or neutral-fold attraction claim.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
