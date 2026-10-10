"""Independent arithmetic audit of eight closed slices of one fixed intervening primitive18 chart."""
import gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_correlated_slice_math as L
import review_correlated_norm_math as N
import review_tangent_variation_math as T
from neural_shrimp.rigorous_orbits import canonical_bytes
need=L.need
BASE=Path('/data2/neural-shrimp-data')
TARGET=BASE/'explore/certificates/s2-period18-quarter-chart-stability-2026-10-10/attempt001-target'
PIN='b44b231b4a31a02581c7e438896f548d1a419a350a1619ef9e5872d1dad8fb91'
WORKER='4c1e73612a1b78d86dee337ffb6f861ffee7c6122ceb4d98f07c81893f54b3de'
EXEC='b8ab6ba630f87304488aef8cae17c886164e6ab1'
PARENT=BASE/'audits/s2-period18-small-march/attempt-001';PARENTPIN='d6888862e0cc416e7aa7beb5301b76ec8188735d0cf2de09a1870a6e6009cd16'
FILES=O.FILES+['tools/reviews/review_localization_math.py','tools/reviews/review_eigenpair_path_math.py','tools/reviews/review_endpoint_attraction.py','tools/reviews/review_correlated_slice_math.py','tools/reviews/review_correlated_norm_math.py','tools/reviews/review_s2_stability_cover.py','tools/reviews/launch_s2_stability_cover.py','tests/reviews/test_correlated_slice_review.py','tests/reviews/test_correlated_norm_review.py','docs/plans/s2-stability-cover-independent.md','tools/reviews/review_tangent_variation_math.py','tools/reviews/review_s2_refined_stability.py','tools/reviews/launch_s2_refined_stability.py','tests/reviews/test_tangent_variation_review.py','docs/plans/s2-refined-stability-independent.md','tools/reviews/review_s2_stability_gap.py','tools/reviews/launch_s2_stability_gap.py','docs/plans/s2-refined-stability-gap-inputs.json','docs/plans/s2-refined-stability-gap.md']

# Bind every loaded local arithmetic dependency, including shared serializers.
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb') as f:data=f.read(256*1024**2+1)
        need(len(data)<=256*1024**2,'bounded decompression')
        side=p.with_name(p.name.removesuffix('.json.gz')+'-payload.json')
        if side.exists():need(O.read(side)=={'uncompressed_bytes':len(data),'uncompressed_sha256':hashlib.sha256(data).hexdigest()},'exact payload binding')
        return json.loads(data)
    return O.read(p)

def worker(out,chart):
    global TARGET,PIN,WORKER,EXEC
    need(type(chart)is int and 147<=chart<=151,'fixed intervening chart')
    specs=read(ROOT/'docs/plans/s2-refined-stability-gap-inputs.json');spec=next(v for v in specs['units'] if v['chart']==chart)
    TARGET=Path(spec['root']);PIN=None if spec['outer_manifest'] is None else spec['outer_manifest']['sha256'];WORKER=spec['worker_manifest']['sha256'];EXEC=spec['execution'];expected=spec['coverage_endpoints']
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[False];result={'passed':False,'slices':[]};resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
    def save(n,v):
        b=canonical_bytes(v);need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<16*1024**2,'16MiB output cap')
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():need(not stop[0] and time.monotonic()-started<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'240s/2GiB audit cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':[256,384],'model_calls':0})
    for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        O.verify(TARGET/'run',WORKER);O.verify(PARENT,PARENTPIN)
        need(read(PARENT/'run/outcome.json')['passed'],'independently audited primitive parent charts')
        run=TARGET/'run';rt=read(run/'runtime-start.json');done=read(run/'outcome.json');inp=read(run/'inputs.json.gz')
        if PIN is not None:
            O.verify(TARGET,PIN);launch=read(TARGET/'launcher-result.json')
            need(rt['source_commit']==EXEC and launch['source_commit']==EXEC and launch['exit_code']==done['exit_code']==0 and launch['owned_live_pids']==[],'complete exact producer execution')
        else:
            need(chart==151,'only previously recovered worker lacks original launcher')
            rr=specs['recovered151'];recovery=Path(rr['root']);O.verify(recovery,rr['outer_manifest']['sha256']);rec=read(recovery/'recovery.json')
            need(rec['execution']==rt['source_commit']==EXEC and rec['original_root']==str(TARGET) and rec['worker_outcome']==done and done['exit_code']==0,'exact complete recovered worker')
            need(rec['launcher_exit_code'] is rec['launcher_final_elapsed_seconds'] is rec['launcher_peak_group_rss_kib'] is None,'missing launcher history remains missing')
            files={str(p.relative_to(TARGET)):p for p in TARGET.rglob('*') if p.is_file()};need(set(files)==set(rec['original_inventory']),'complete recovered original inventory')
            for n,p in files.items():need(not p.is_symlink() and not p.stat().st_mode&0o222 and O.entry(p)==rec['original_inventory'][n],'recovered original immutable bytes '+n)
            need(not Path('/proc',str(rt['pid'])).exists(),'original worker absent')
            result['historical_execution_limit']='Original launcher exit, duration and peak-group RSS remain unavailable; this fresh independent audit supplies its own execution records.'
        need(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1'),'native arithmetic and CPU1')
        for n,h in rt['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==h,'all historical source bytes '+n)
        need(done['passed'] and done['completed_slices']==8 and done['stages']==1288 and len(done['rows'])==8,'complete original inventory')
        need(done['counts']==spec['counts'],'complete native derivative calls')
        selected=inp['selected_artifact'];need(selected==spec['selected_artifact'] and selected['index']==chart-147 and selected['source_commit']=='e987c8669efb5fb9275046f401cd4cfa172cfa9b','exact certified branch chart')
        parent=Path(selected['root']);need(parent==BASE/'explore/certificates/s2-period18-fold-small-march-2026-10-10/unit-001/run' and selected['manifest']=='a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2','same independently audited native parent');O.verify(parent,selected['manifest']);pp=parent/f'chart-{chart-147:03d}.json.gz';need(O.entry(pp)==selected['identity'],'exact original chart bytes')
        need(read(pp)==inp['chart'],'same full235D primitive chart');c=inp['chart'];need(c['statement']['period']==18 and c['statement']['phase_dimension']==13,'native model dimensions')
        prior=read(PARENT/f'run/march-{chart-147}-384.json');need(prior['passed'],'independent strict native chart proof')
        audited_source=read(PARENT/'run/runtime.json')['source_commit'];bound_script=subprocess.check_output(['git','show',audited_source+':tools/reviews/review_s2_small_march.py'],cwd=ROOT,text=True)
        need(selected['manifest'] in bound_script and selected['source_commit'] in bound_script,'earlier independent review pins this exact native chart')
        orig=read(ROOT/'certificates/s2-period18-receipt.json');need(inp['model_inputs']==read(Path(orig['external_root'])/'inputs.json'),'original exact central training map')
        need(c['exact_parameter_endpoints']==inp['coverage_endpoints']==done['complete_coverage']==expected,'fixed covered interval')
        ends=sorted([list(map(Fraction,v)) for v in inp['slices']]);need(len(ends)==8 and ends[0][0]==Fraction(expected[0]) and ends[-1][1]==Fraction(expected[1]) and all(b-a==Fraction(1,1024) for a,b in ends) and all(ends[i][1]==ends[i+1][0] for i in range(7)),'closed exact gapless partition')
        del pp;gc.collect();tick()
        for index,interval in enumerate(inp['slices']):
            def stages(kind):return [read(p) for p in sorted(run.glob(f'slice-{index:02d}-stage-*-{kind}.json.gz'))]
            def one(kind):
                v=stages(kind);need(len(v)==1,'single stage '+kind);return v[0]
            answer=read(run/f'slice-{index:02d}-result.json.gz');base=answer['base'];need(base['status']=='complete_slice_bounds' and base['exact_endpoints']==interval and answer['passed'] and answer['status']=='complete_refined_slice_bounds','original base and successful refinement retained')
            ws=one('base_whole_slice_start');wr=stages('base_whole_slice_intersection');cs=one('base_center_slice_start');cr=stages('base_center_slice_intersection');actual=one('base_physical_inputs')
            whole=L.localization(c,ws,wr,base['whole_localization']);center=L.localization(c,cs,cr,base['center_localization'])
            need(actual['exact_endpoints']==interval and actual['center_parameter']==str(sum(map(Fraction,interval))/2),'same exact physical parameter')
            for key,known in [('actual_whole_box',whole['final_state_box']),('actual_center_box',center['final_state_box'])]:
                need(len(actual[key])==235,'complete physical state and rate')
                for a,b in zip(actual[key],known,strict=True):L.contains(a,b,'actual model domain retains entire known root')
            tangent=L.tangent(ws['preconditioner'],wr[-1]['jacobian'],c['exact_weights'],base['uniform_tangent'],actual['actual_uniform_tangent'])
            saved=one('base_uniform_tangent');need(saved['bound']==base['uniform_tangent'] and saved['root_derivative_domain']==wr[-1]['derivative_domain'] and saved['root_final_state_box']==whole['final_state_box'],'whole-domain implicit tangent input identity')
            for key in ('final_state_box',):whole.pop(key);center.pop(key)
            gc.collect();tick()
            phases=stages('base_phase_derivatives');refine=one('tangent_refinement');refactual=one('refined_inputs');refphases=stages('refined_phase')
            need(refine==answer['tangent_refinement'],'same complete refinement result')
            refinement=T.audit(c,ws,wr[-1]['jacobian'],cr[-1]['jacobian'],base['uniform_tangent'],actual,phases,refine,refactual)
            need(len(phases)==len(refphases)==18 and [v['phase'] for v in refphases]==list(range(18)),'complete native refined phase order')
            need(all(x['center_J']==y['center_J'] for x,y in zip(phases,refphases,strict=True)),'same original center Jacobian')
            frames=one('refined_frames');normphases=stages('refined_norm_phase');metric=one('refined_norm_metric')
            need(frames==one('base_frames'),'same certified cyclic frames')
            need(answer['norm_cocycle']['metric']==metric and metric['passed'],'same saved final native metric')
            # Source-bound C1/MixedJet acquisitions are shared; independent
            # forcing, inverse, integration and native norm reductions here.
            half=(Fraction(interval[1])-Fraction(interval[0]))/2
            with ctx.workprec(384):norm=N.audit(refphases,frames,normphases,metric,half)
            need(norm['passed'],'independent strict native attraction')
            row={'base_passed':base['passed'],'index':index,'exact_endpoints':interval,'whole_localization':whole,'center_localization':center,'uniform_tangent':tangent,'refinement':refinement,'norm':norm,'passed':True};save(f'slice-{index:02d}.json',row);result['slices'].append({'index':index,'exact_endpoints':interval,'passed':True});del answer,base,phases,refphases,frames,normphases,metric,ws,wr,cs,cr,refine,refactual;gc.collect();tick()
        result.update(passed=True,coverage=inp['coverage_endpoints'],native_parameters=13,primitive_period=18,scope='Independent256 localization/implicit first and second derivative/integrated-tangent arithmetic and384 native framed polynomial/error/Lyapunov reduction on eight adjacent closed slices of the already independently certified branch. Actual physical state/rate/tangent domains and complete source/phase identity bound. Shared analytic C1/MixedJet acquisitions and Arb; no new derivative implementation, neutral-fold attraction, whole branch or full shrimp claim.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--chart',type=int,choices=list(range(147,152)),required=True);a=p.parse_args();raise SystemExit(worker(a.out,a.chart))
