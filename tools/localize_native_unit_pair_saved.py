#!/usr/bin/env python3
"""One saved full704 root/parameter-tangent localization, no acquisitions."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_floquet_complement as D
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_parameter_localization import localize_parameter_root
O=D.O;U=D.U;need=D.need;read=D.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/attempt006-localization/run')
CARD='docs/plans/unit-pair-saved-localization-inputs.json'
FILES=sorted(set(D.FILES+[CARD,'docs/plans/unit-pair-correlated-complement.md','docs/plans/unit-pair-saved-localization.md','tools/localize_native_unit_pair_saved.py','tools/launch_native_unit_pair_localization.py','src/neural_shrimp/rigorous_parameter_localization.py','tests/test_rigorous_parameter_localization.py','tests/test_unit_pair_saved_localization.py']))

def synthetic_localization(n,tick):
    need(type(n)is int and 1<=n<=704,'bounded full synthetic profile');small=arb((1,-16))
    C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)])
    slope=arb_mat([[arb('1/8')]for _ in range(n)]);ht=-(J*slope);g0=arb_mat([[arb((1,-80))]for _ in range(n)])
    return localize_parameter_root(C,J,g0,J,ht,ht,[arb(1)]*n,slope,[arb(1)]*n,arb((1,-16)),arb((1,-11)),tick=tick)

def native_inputs(chart,groups,tick):
    need(chart['schema']=='neural-shrimp-native-unit-pair-tube-v1'and (chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,704),'exact native704 convention')
    need(chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-16,'radius_exponent':-11}and U.F(chart['parameter_center'])==U.F(1,65536)and chart['parameter_endpoints']==['0','1/32768'],'fixed original finite geometry')
    need(len(groups)==3 and all(len(g)==36 for g in groups),'all three exact source-acquisition groups')
    c=[U.M.point(v)for v in chart['center']];s=[U.M.point(v[0])for v in chart['slope']];w=list(map(U.F,chart['weights']));need(len(c)==len(s)==len(w)==704 and all(0<x<=1 for x in w),'full exact predictor and weights')
    rho=U.F(1,2048);half=U.F(1,65536);need([U.M.point(v)for v in chart['coordinate_radii']]==[rho*x for x in w],'literal original correction radius')
    for row in chart['preconditioner']:
        for value in row:U.M.point(value)
    center=[U.R.exact(x)for x in c];slope=arb_mat([[U.R.exact(x)]for x in s]);delta=arb(0,U.R.exact(half));t=U.R.exact(half)+delta
    predictor=[center[i]+slope[i,0]*delta for i in range(704)]
    whole=[predictor[i]+arb(0,U.R.exact(rho*w[i]))for i in range(704)]
    need(len(chart['whole_box'])==704,'all original whole coordinates')
    for old,new in zip(chart['whole_box'],whole,strict=True):U.contains(old,D.ball_json(new),'original native derivative box covers every predictor-root segment')
    g0,_,_=U.rebuild(center,groups[0],13,18,U.R.exact(half));tick()
    _,jp,htp=U.rebuild(predictor,groups[1],13,18,t);tick()
    _,j,htw=U.rebuild(whole,groups[2],13,18,t);tick()
    return dict(C=U.R.matrix(chart['preconditioner']),J=j,G0=g0,Jp=jp,Htp=htp,Htw=htw,center=center,slope=slope,weights=[U.R.exact(x)for x in w],halfwidth=U.R.exact(half),original_radius=U.R.exact(rho))

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];result={'passed':False,'status':'incomplete','native_calls':0,'complementary_stability_evaluated':False}
    def save(name,value):
        raw=D.canonical_bytes(value);need(used[0]+len(raw)<64*1024**2,'64MiB raw cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():need(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
    signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'python':D.platform.python_version(),'numpy':D.np.__version__,'scipy':D.scipy.__version__,'python_flint':D.flint.__version__,'flint':D.flint.__FLINT_VERSION__,'native_calls':0})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/n).read_bytes();need(used[0]+len(b)<64*1024**2,'source cap');p.write_bytes(b);used[0]+=len(b)
    try:
        with ctx.workprec(384):
            before=time.monotonic();synthetic=synthetic_localization(704,tick);seconds=time.monotonic()-before;forecast=4*seconds+4*15.713+20
            save('profile.json',{'passed':synthetic['passed'],'dimension':704,'seconds':seconds,'forecast_seconds':forecast,'reference_independent704_seconds':15.713,'target_arrays_unopened':True});need(synthetic['passed']and forecast<240,'measured full704 admission');del synthetic;gc.collect()
            card=read(ROOT/CARD);flat,blocks,binding=D.bind(card,tick);save('bindings.json',binding);del flat,blocks;gc.collect()
            failed=card['failed_orthogonal'];fr=Path(failed['root']);O.verify(fr,failed['outer']);O.verify(fr/'run',failed['worker']);D.B.archived(fr/'run',failed['source'],'runtime.json','source');fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json');fb=read(fr/'run/bindings.json')
            need(fo['passed']is False and fo['status']=='completed_sufficient_bound_unresolved'and fo['native_calls']==0 and fo['complement_dimension']==11 and fl['exit_code']==2 and not fl['owned_live_pids'],'retained completed insufficient independent-box bound')
            need(fb['card']['producer']==card['producer']and fb['card']['independent']==card['independent'],'same independently proved parent for amendment')
            parent=Path(card['producer']['root']);files=sorted((parent/'run/events').glob('*-direction_return.json.gz'));need(len(files)==216,'complete unchanged original acquisition inventory')
            selected=files[108:];groups=[[read(p)for p in selected[k:k+36]]for k in range(0,108,36)];chart=read(parent/'run/certificate.json.gz')
            keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','weights','coordinate_radii','preconditioner','whole_box'}
            for key in list(chart):
                if key not in keep:del chart[key]
            gc.collect();args=native_inputs(chart,groups,tick);save('selected-source-inputs.json',{'card':card,'certificate':O.entry(parent/'run/certificate.json.gz'),'groups':'center,predictor,whole: original groups4,5,6','direction_files':{str(p.relative_to(parent/'run')):O.entry(p)for p in selected},'shared_native_acquisitions':108,'new_native_calls':0});del chart,groups;gc.collect();tick()
            proof=localize_parameter_root(**args,tick=tick);save('localization.json',proof);tick()
            center=[U.R.original_ball(v)for v in proof['center_box']];whole=[U.R.original_ball(v)for v in proof['whole_box']]
            need(len(center)==len(whole)==len(proof['tangent_box'])==704 and all(x>0 for x in center[-2:]+whole[-2:]),'all actual coordinates and both positive rates')
            result.update(passed=True,status='conditional_full704_localization_completed',weighted_neumann_bound=proof['weighted_neumann_bound'],center_correction_radius=proof['center_refinement']['weighted_uniform_radius'],whole_correction_radius=proof['whole_refinement']['weighted_uniform_radius'],tangent_correction_radius=proof['tangent_refinement']['weighted_uniform_radius'],parameter_endpoints=['0','1/32768'],shared_native_acquisitions=108,scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
