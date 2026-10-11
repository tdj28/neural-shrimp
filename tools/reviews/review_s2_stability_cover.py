"""Independent arithmetic audit of eight closed native period18 stability slices."""
import gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_correlated_slice_math as L
import review_correlated_norm_math as N
from neural_shrimp.rigorous_orbits import canonical_bytes
need=L.need
BASE=Path('/data2/neural-shrimp-data')
TARGET=BASE/'explore/certificates/s2-period18-stability-march-2026-10-10/unit001'
PIN='72c9fde4132b29f918aadcaf274fa4326f88d04ff748334b8c719b0d44fe6c4c'
WORKER='6842aea9f88dfb07b9c78f793e8b7a4c14701ef77c644ea19d118dbaab85e49a'
EXEC='564734eaa30a3d8cdd647b8d3619f46970515983'
PARENT=BASE/'audits/s2-period18-small-march/attempt-001';PARENTPIN='d6888862e0cc416e7aa7beb5301b76ec8188735d0cf2de09a1870a6e6009cd16'
FILES=O.FILES+['tools/reviews/review_localization_math.py','tools/reviews/review_eigenpair_path_math.py','tools/reviews/review_endpoint_attraction.py','tools/reviews/review_correlated_slice_math.py','tools/reviews/review_correlated_norm_math.py','tools/reviews/review_s2_stability_cover.py','tools/reviews/launch_s2_stability_cover.py','tests/reviews/test_correlated_slice_review.py','tests/reviews/test_correlated_norm_review.py','docs/plans/s2-stability-cover-independent.md']

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

SPECS={
 1: ('564734eaa30a3d8cdd647b8d3619f46970515983', '72c9fde4132b29f918aadcaf274fa4326f88d04ff748334b8c719b0d44fe6c4c', '6842aea9f88dfb07b9c78f793e8b7a4c14701ef77c644ea19d118dbaab85e49a', ['21/64', '43/128']),
 2: ('636d0df868707aa78fdc02fee7c8a7e8e62a4dd2', '1180509c0f3d59ca253267446e2125585a2866d56705b057e02b3286a07b1b73', 'a736b79f79b0ad00cc2f4bead28654b9e0656b17c208b15873dd99748e6092bf', ['41/128', '21/64']),
 3: ('770418547513ff59b892ef8ea8abdf169f12b5a9', '3d8dc132d07a73e5a393a770c285312a35da7affd0cfade21e7efaca3e0ec08c', 'a4afd73e8598ef5ebe1cba817b5e9bcb3a268d74c593eb3d6d792098b2859e07', ['5/16', '41/128'])}

def worker(out,unit=1):
    global TARGET,PIN,WORKER,EXEC
    need(type(unit)is int and unit in SPECS,'fixed audited unit choice')
    EXEC,PIN,WORKER,expected=SPECS[unit];TARGET=BASE/f'explore/certificates/s2-period18-stability-march-2026-10-10/unit{unit:03d}'
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
        O.verify(TARGET,PIN);O.verify(TARGET/'run',WORKER);O.verify(PARENT,PARENTPIN)
        need(read(PARENT/'run/outcome.json')['passed'],'independently audited primitive parent charts')
        run=TARGET/'run';rt=read(run/'runtime-start.json');done=read(run/'outcome.json');launch=read(TARGET/'launcher-result.json');inp=read(run/'inputs.json.gz')
        need(rt['source_commit']==EXEC and launch['source_commit']==EXEC and launch['exit_code']==done['exit_code']==0 and launch['owned_live_pids']==[],'complete exact producer execution')
        need(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1'),'native arithmetic and CPU1')
        for n,h in rt['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==h,'all historical source bytes '+n)
        need(done['passed'] and done['completed_slices']==8 and done['stages']==1288 and len(done['rows'])==8,'complete original inventory')
        need(done['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete native derivative calls')
        selected=inp['selected_artifact'];need(selected['index']==8-unit and selected['source_commit']=='e987c8669efb5fb9275046f401cd4cfa172cfa9b','exact certified branch chart')
        parent=Path(selected['root']);need(parent==BASE/'explore/certificates/s2-period18-fold-small-march-2026-10-10/unit-001/run' and selected['manifest']=='a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2','same independently audited raw parent');O.verify(parent,selected['manifest']);pp=parent/f'chart-{8-unit:03d}.json.gz';need(O.entry(pp)==selected['identity'],'exact original chart bytes')
        need(read(pp)==inp['chart'],'same full235D primitive chart');c=inp['chart'];need(c['statement']['period']==18 and c['statement']['phase_dimension']==13,'native model dimensions')
        orig=read(ROOT/'certificates/s2-period18-receipt.json');need(inp['model_inputs']==read(Path(orig['external_root'])/'inputs.json'),'original exact central training map')
        need(c['exact_parameter_endpoints']==inp['coverage_endpoints']==done['complete_coverage']==expected,'fixed covered interval')
        ends=sorted([list(map(Fraction,v)) for v in inp['slices']]);need(len(ends)==8 and ends[0][0]==Fraction(expected[0]) and ends[-1][1]==Fraction(expected[1]) and all(b-a==Fraction(1,1024) for a,b in ends) and all(ends[i][1]==ends[i+1][0] for i in range(7)),'closed exact gapless partition')
        del pp;gc.collect();tick()
        for index,interval in enumerate(inp['slices']):
            def stages(kind):return [read(p) for p in sorted(run.glob(f'slice-{index:02d}-stage-*-{kind}.json.gz'))]
            def one(kind):
                v=stages(kind);need(len(v)==1,'single stage '+kind);return v[0]
            answer=read(run/f'slice-{index:02d}-result.json.gz');base=answer['base'];need(base['passed'] and base['status']=='complete_slice_bounds' and base['exact_endpoints']==interval,'first independent norm proof exists')
            ws=one('base_whole_slice_start');wr=stages('base_whole_slice_intersection');cs=one('base_center_slice_start');cr=stages('base_center_slice_intersection');actual=one('base_physical_inputs')
            whole=L.localization(c,ws,wr,base['whole_localization']);center=L.localization(c,cs,cr,base['center_localization'])
            need(actual['exact_endpoints']==interval and actual['center_parameter']==str(sum(map(Fraction,interval))/2),'same exact physical parameter')
            for key,known in [('actual_whole_box',whole['final_state_box']),('actual_center_box',center['final_state_box'])]:
                need(len(actual[key])==235,'complete physical state and rate')
                for a,b in zip(actual[key],known,strict=True):L.contains(a,b,'actual model domain retains entire known root')
            tangent=L.tangent(ws['preconditioner'],wr[-1]['jacobian'],c['exact_weights'],base['uniform_tangent'],actual['actual_uniform_tangent'])
            saved=one('base_uniform_tangent');need(saved['bound']==base['uniform_tangent'] and saved['root_derivative_domain']==wr[-1]['derivative_domain'] and saved['root_final_state_box']==whole['final_state_box'],'whole-domain implicit tangent input identity')
            for key in ('final_state_box',):whole.pop(key);center.pop(key)
            del ws,wr,cs,cr;gc.collect();tick()
            phases=stages('base_phase_derivatives');frames=one('base_frames');normphases=stages('base_norm_phase');metric=one('base_norm_metric')
            need(len(phases)==len(normphases)==18 and [v['phase'] for v in phases]==list(range(18)),'complete native phase order')
            need(base['norm_cocycle']['metric']==metric and metric['passed'],'same saved native product metric')
            # Source-bound MixedJet acquisitions used exactly these physical
            # boxes and this total tangent; both kernels and ordering are bound
            # above. This independent audit does not acquire new derivatives.
            half=(Fraction(interval[1])-Fraction(interval[0]))/2
            with ctx.workprec(384):norm=N.audit(phases,frames,normphases,metric,half)
            need(norm['passed'],'independent strict native attraction')
            row={'index':index,'exact_endpoints':interval,'whole_localization':whole,'center_localization':center,'uniform_tangent':tangent,'norm':norm,'passed':True};save(f'slice-{index:02d}.json',row);result['slices'].append({'index':index,'exact_endpoints':interval,'passed':True});del answer,base,phases,frames,normphases,metric;gc.collect();tick()
        result.update(passed=True,coverage=inp['coverage_endpoints'],native_parameters=13,primitive_period=18,scope='Independent256 localization/implicit-tangent arithmetic and384 native framed polynomial/error/Lyapunov reduction on eight adjacent closed slices of the already independently certified branch. Actual physical state/rate/tangent domains and complete source/phase identity bound. Shared analytic C1/MixedJet acquisitions and Arb; no new derivative implementation, neutral-fold attraction, whole branch or full shrimp claim.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--unit',type=int,choices=[1,2,3],default=1);a=p.parse_args();raise SystemExit(worker(a.out,a.unit))
