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
CONFIG='docs/plans/s2-attraction-remaining-inputs.json'
FILES=[CONFIG,'docs/plans/s2-attraction-remaining.md','docs/reviews/s2-attraction-remaining-helper-provenance.json','tools/reviews/review_s2_attraction_remaining.py','tools/reviews/launch_s2_attraction_remaining.py','tests/reviews/test_attraction_remaining_review.py','tests/reviews/test_correlated_slice_review.py','tests/reviews/test_correlated_norm_review.py','tests/reviews/test_tangent_variation_review.py','certificates/s2-period18-receipt.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def validate_partition(inp,done,spec):
    expected=spec['coverage_endpoints'];count=4 if spec['chart']==145 else 8
    need(inp['chart']['exact_parameter_endpoints']==inp['coverage_endpoints']==done['complete_coverage']==expected,'fixed complete parent interval')
    need(inp['slices']==spec['slices'],'exact fixed slice ordering')
    ends=sorted([list(map(Fraction,v)) for v in inp['slices']])
    need(len(ends)==count and ends[0][0]==Fraction(expected[0]) and ends[-1][1]==Fraction(expected[1]) and all(b-a==Fraction(1,512) for a,b in ends) and all(ends[i][1]==ends[i+1][0] for i in range(count-1)),'closed exact gapless partition')
    need(done['passed'] and done['completed_slices']==count and done['stages']==161*count and len(done['rows'])==count,'complete original inventory')
    need(done['counts']==spec['counts']=={'map_attempted':162*count,'map_returned':162*count,'unit_attempted':324*count,'unit_returned':324*count,'mixed_attempted':36*count,'mixed_returned':36*count},'complete original derivative counts')
    return count

def bind_parent(inp,spec):
    chart=spec['chart'];selected=inp['selected_artifact'];need(selected==spec['selected_artifact'],'exact certified chart selection')
    parent=Path(selected['root']);O.verify(parent,selected['manifest']);pp=parent/spec['parent_file'];need(O.entry(pp)==selected['identity'],'exact original chart bytes')
    original=read(pp);original=original if spec['parent_chart_index'] is None else original['charts'][spec['parent_chart_index']]
    need(original==inp['chart'],'same full235D primitive chart');c=inp['chart'];need(c['passed'] and all(c['gates'].values()) and c['statement']['period']==18 and c['statement']['phase_dimension']==13,'native proved dimensions')
    prior=spec['prior_audit'];ar=Path(prior['root']);O.verify(ar,prior['outer']);adone=read(ar/'run/outcome.json');art=read(ar/'run'/prior['runtime'])
    need(adone['passed'] and art['source_commit']==prior['source'],'passed exact independent parent audit')
    proof=read(ar/'run'/prior['proof']);proof=proof if prior['proof_selector'] is None else proof[prior['proof_selector']]
    need(proof['passed'] and proof['precision']==384 and proof['coordinate_count']==235 and all(proof['gates'].values()),'independently proved strict full native chart')
    script=subprocess.check_output(['git','show',prior['source']+':'+prior['script']],cwd=ROOT,text=True)
    if chart<145:
        need(selected['global_chart']==chart and selected['index']==chart-130 and spec['parent_file']==f'chart-{chart-130:03d}.json.gz' and prior['proof']==f'chart-{chart-130:03d}.json','literal original family/index')
        p=adone['producer'];need(p['root']==str(parent.parent) and p['worker']==selected['manifest'] and p['execution']==selected['source_commit'] and p['qualified']==15 and adone['total_qualified_charts']==144,'same exact independently audited original fifteen-chart prefix')
    else:
        need(selected['global_chart']==145 and selected['certificate_chart_index']==spec['parent_chart_index']==0 and prior['proof']=='rechart-first-384.json','only independently qualified first rechart')
        need(selected['manifest'] in script and selected['source_commit'] in script and adone['total_connected_charts']==156,'same independently bound rechart source')
    return c

def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb') as f:data=f.read(256*1024**2+1)
        need(len(data)<=256*1024**2,'bounded decompression')
        side=p.with_name(p.name.removesuffix('.json.gz')+'-payload.json')
        if side.exists():need(O.read(side)=={'uncompressed_bytes':len(data),'uncompressed_sha256':hashlib.sha256(data).hexdigest()},'exact payload binding')
        return json.loads(data)
    return O.read(p)

def worker(out,chart):
    need(type(chart)is int and chart in (142,143,144,145),'fixed remaining chart')
    specs=read(ROOT/CONFIG);spec=next(v for v in specs['units'] if v['chart']==chart)
    TARGET=Path(spec['root']);PIN=spec['outer_manifest']['sha256'];WORKER=spec['worker_manifest']['sha256'];EXEC=spec['execution'];expected=spec['coverage_endpoints']
    # Set the already-approved worker envelope before any Git preflight subprocess.
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stop=[False];result={'passed':False,'chart':chart,'slices':[]};resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
    def save(n,v):
        b=canonical_bytes(v);need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<16*1024**2,'16MiB output cap')
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():need(not stop[0] and time.monotonic()-started<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'240s/2GiB audit cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':[256,384],'model_calls':0})
    for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        O.verify(TARGET,PIN);O.verify(TARGET/'run',WORKER)
        run=TARGET/'run';rt=read(run/'runtime-start.json');done=read(run/'outcome.json');inp=read(run/'inputs.json.gz');launch=read(TARGET/'launcher-result.json')
        need(rt['source_commit']==EXEC and launch['source_commit']==EXEC and launch['exit_code']==done['exit_code']==0 and launch['owned_live_pids']==[],'complete exact producer execution')
        need(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1'),'native arithmetic and CPU1')
        for n,h in rt['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==h,'all historical committed source bytes '+n)
        for pid in (rt['pid'],launch['launcher_pid'],launch['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'completed original owned processes')
        c=bind_parent(inp,spec);count=validate_partition(inp,done,spec)
        orig=read(ROOT/'certificates/s2-period18-receipt.json');mp=Path(orig['external_root'])/'inputs.json'
        need(O.entry(mp)['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f' and inp['model_inputs']==read(mp),'original exact central training map')
        save('input-binding.json',spec);gc.collect();tick()
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
        result.update(passed=True,coverage=inp['coverage_endpoints'],completed_slices=count,native_parameters=13,primitive_period=18,scope='Independent256 localization/implicit first and second derivative/integrated-tangent arithmetic and384 native framed polynomial/error/Lyapunov reduction on the fixed adjacent closed slices of the already independently certified branch. Actual physical state/rate/tangent domains and complete source/phase identity bound. Shared analytic C1/MixedJet acquisitions and Arb; no new derivative implementation, neutral-fold attraction, whole branch or full shrimp claim.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--chart',type=int,choices=[142,143,144,145],required=True);a=p.parse_args();raise SystemExit(worker(a.out,a.chart))
