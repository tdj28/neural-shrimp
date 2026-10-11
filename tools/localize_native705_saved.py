#!/usr/bin/env python3
"""One saved-only localization/tangent of the independently joined705 chart."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_unit_pair_saved as L
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
D=L.D;U=L.U;O=L.O;need=L.need;read=L.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/attempt001-localization/run')
CARD='docs/plans/native705-saved-localization-inputs.json'
FILES=sorted(set(L.FILES+[CARD,'docs/plans/native705-correlated-complement.md','docs/plans/native705-saved-localization.md','tools/localize_native705_saved.py','tools/launch_native705_localization.py','src/neural_shrimp/rigorous_quadratic_localization.py','tests/test_rigorous_quadratic_localization.py','tests/test_native705_saved_localization.py']))

def synthetic(tick,n=705):
    need(type(n)is int and 1<=n<=705,'bounded synthetic size');e=arb((1,-16));C=arb_mat([[arb(int(i==j))+e for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-e for j in range(n)]for i in range(n)])
    s=[arb('1/8')]*n;partial=-(J*arb_mat([[v]for v in s]));g0=arb_mat([[arb((1,-80))]for _ in range(n)])
    return localize_quadratic_root(C,J,g0,g0,partial,[arb(1)]*n,s,[arb(0)]*n,[arb(1)]*n,arb((1,-7)),arb((1,-10)),tick=tick)

def chart_inputs(chart):
    need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,705),'exact full705 native chart')
    need(chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10}and U.F(chart['parameter_center'])==U.F(31,4096)and chart['parameter_endpoints']==['-1/4096','63/4096'],'literal connected first quadratic interval')
    need(chart['gates']==dict(uniform_full_native_unit_pair_locus=True,primitive_period=True,realified_frame_independent=True,anchor_identified=False),'unchanged original unjoined producer gates')
    names=['center','slope','curvature'];c,s,a=[[U.R.exact(U.M.point(v))for v in chart[k]]for k in names];w=[U.R.exact(U.F(v))for v in chart['weights']]
    need(len(c)==len(s)==len(a)==len(w)==705 and all(0<x<=1 for x in w),'full exact quadratic predictor and weights')
    rho=arb((1,-10));h=arb((1,-7));d=arb(0,h);r=[rho*x for x in w]
    need([U.M.point(v)for v in chart['coordinate_radii']]==[U.M.point(D.ball_json(v))for v in r],'exact original coordinate radii')
    known=[c[i]+s[i]*d+a[i]*d*d/2+arb(0,r[i])for i in range(705)]
    for old,new in zip(chart['whole_box'],known,strict=True):U.contains(old,D.ball_json(new),'literal original mean-value domain covers actual quadratic root box')
    need(known[-3]>0 and known[-2]>0 and known[-1]>0,'both positive actual rates and positive entire angle')
    C=U.R.matrix(chart['preconditioner']);J=U.R.matrix(chart['whole_jacobian']);G0=U.R.matrix(chart['center_residual'])
    coefficients=U.R.matrix(chart['quadratic_center_coefficients']);third=U.R.matrix(chart['quadratic_whole_coefficients'])
    need(coefficients.nrows()==third.nrows()==705 and coefficients.ncols()==third.ncols()==4,'complete source-authenticated cubic Taylor predictor enclosures')
    residual=arb_mat([[coefficients[i,0]+coefficients[i,1]*d+coefficients[i,2]*d*d+third[i,3]*d*d*d]for i in range(705)])
    for i in range(705):U.contains(chart['predictor_residual'][i][0],D.ball_json(residual[i,0]),'same or narrower authenticated Taylor residual')
    partial=arb_mat(705,1);partial[704,0]=-1
    return dict(C=C,J=J,G0=G0,path_residual=residual,parameter_partial=partial,center=c,slope=s,curvature=a,weights=w,halfwidth=h,radius=rho)

def bind(card,tick):
    checked={}
    for label in ['producer','independent','connection']:
        rec=card[label];root=Path(rec['root']);O.verify(root,rec['outer']);O.verify(root/'run',rec['worker']);rt=read(root/'run/runtime.json');source=rt.get('commit',rt.get('source_commit'));D.B.archived(root/'run',source,'runtime.json','source');checked[label]={'source':source,'root':root,'out':read(root/'run/outcome.json'),'launch':read(root/'launcher-result.json')}
    p,u,e=[checked[k]for k in ['producer','independent','connection']]
    need(p['source']==card['producer']['source']and p['launch']['exit_code']==2 and not p['launch']['owned_live_pids'],'preserved complete unjoined producer ownership')
    need(u['out']['passed']and u['out']['uniform_primitive_positive_unit_pair']and u['out']['inclusions']==705 and u['out']['native_calls']==0 and u['launch']['exit_code']==0 and not u['launch']['owned_live_pids'],'independently proved entire positive primitive705 curve')
    ub=read(u['root']/'run/input-binding.json');pp={k:card['producer'][k]for k in ['source','outer','worker']}
    need(ub['root']==str(p['root'])and ub['pins']==pp,'literal independent proof of original chart')
    need(e['out']['passed']and e['out']['connected']and e['out']['anchor_coordinates']==705 and e['out']['native_calls']==0 and e['launch']['exit_code']==0 and not e['launch']['owned_live_pids'],'independently connected same primitive root')
    eb=read(e['root']/'run/input-binding.json');need(eb['prior705']==ub,'literal independent endpoint membership in same chart')
    chart=read(p['root']/'run/certificate.json.gz');args=chart_inputs(chart);tick()
    return args,{'card':card,'certificate':O.entry(p['root']/'run/certificate.json.gz'),'independent_uniform':O.entry(u['root']/'run/native705-384.json'),'independent_connection':O.entry(e['root']/'run/independent-membership.json'),'source_dependencies':{k:v['source']for k,v in checked.items()},'shared_inputs':'Original source-authenticated full705 C/J/G0 and complete cubic Taylor predictor residual; parent independent384 curve and480 same-root endpoint proof. No new or independent AD acquisition in this localization.'}

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];result={'passed':False,'status':'incomplete','native_calls':0,'complementary_stability_evaluated':False}
    def save(name,value,closing=False):
        raw=D.canonical_bytes(value);need(used[0]+len(raw)<(64 if closing else 62)*1024**2,'64MiB raw/final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():need(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
    signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'python':D.platform.python_version(),'numpy':D.np.__version__,'scipy':D.scipy.__version__,'python_flint':D.flint.__version__,'flint':D.flint.__FLINT_VERSION__,'native_calls':0})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/n).read_bytes();need(used[0]+len(b)<62*1024**2,'source cap');p.write_bytes(b);used[0]+=len(b)
    try:
        with ctx.workprec(384):
            before=time.monotonic();trial=synthetic(tick);seconds=time.monotonic()-before;forecast=4*seconds+4*20.116+20
            save('profile.json',{'passed':trial['passed'],'dimension':705,'seconds':seconds,'forecast_seconds':forecast,'target_arrays_unopened':True});need(trial['passed']and forecast<240,'measured full705 source admission');del trial;gc.collect()
            args,binding=bind(read(ROOT/CARD),tick);save('bindings.json',binding);proof=localize_quadratic_root(**args,tick=tick);save('localization.json',proof);tick()
            whole=[U.R.original_ball(v)for v in proof['whole_box']];need(len(whole)==705 and all(v>0 for v in whole[-3:]),'both actual rates and angle positive')
            result.update(passed=True,status='conditional_full705_localization_completed',weighted_neumann_bound=proof['weighted_neumann_bound'],parameter_endpoints=['-1/4096','63/4096'],scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
