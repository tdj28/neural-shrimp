#!/usr/bin/env python3
"""One next wider independently rooted705 chart; unchanged correlated complement arithmetic."""
import argparse,gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_refined_complement as B
import refine_native705_tangent as T
import localize_native705_saved as L
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
from neural_shrimp.rigorous_quadratic_tangent_forcing import path_residual_derivative,mixed_hessian_from_second_coefficients,refine_tangent
from neural_shrimp.rigorous_unit_pair_pal import path_coefficients
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import vector_json,matrix_json
D,U,O,need,read=B.D,B.U,B.O,B.need,B.read
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/wide001')
CARD='docs/plans/native705-complement-wide-inputs.json'
FILES=sorted(set(B.FILES+[CARD,'docs/plans/native705-complement-wide.md','tools/probe_native705_complement_wide.py','tools/launch_native705_complement_wide.py','tests/test_native705_complement_wide.py']))


def geometry(unit):
    need(type(unit)is int and unit==1,'only the next fixed independently proved wide chart')
    return '383/4096',['319/4096','447/4096']


def chart_inputs(chart,unit):
    sc,endpoints=geometry(unit)
    need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,705),'full native705 chart')
    need(chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-6,'radius_exponent':-10}and chart['parameter_center']==sc and chart['parameter_endpoints']==endpoints,'fixed exact independently proved chart interval')
    need(all(chart['gates'][k]for k in ['uniform_full_native_unit_pair_locus','primitive_period','realified_frame_independent']),'whole original root/period/frame')
    c,s,a=[[U.R.exact(U.M.point(v))for v in chart[k]]for k in ['center','slope','curvature']];w=[U.R.exact(F(v))for v in chart['weights']]
    need(len(c)==len(s)==len(a)==len(w)==705 and all(0<x<=1 for x in w),'all exact705 predictor/weights')
    rho=arb((1,-10));h=arb((1,-6));d=arb(0,h);r=[rho*x for x in w]
    need([U.M.point(v)for v in chart['coordinate_radii']]==[F(1,1024)*F(v)for v in chart['weights']],'exact original weighted coordinate radius')
    whole=[c[i]+s[i]*d+a[i]*d*d/2+arb(0,r[i])for i in range(705)]
    for old,new in zip(chart['whole_box'],whole,strict=True):U.contains(old,D.ball_json(new),'actual predictor-root segment inside exact original derivative domain')
    need(all(v>0 for v in whole[-3:])and abs(whole[-1]).upper()<=arb('1/4'),'positive entire rates and bounded angle')
    C=U.R.matrix(chart['preconditioner']);J=U.R.matrix(chart['whole_jacobian']);G0=U.R.matrix(chart['center_residual']);cc=U.R.matrix(chart['quadratic_center_coefficients']);wc=U.R.matrix(chart['quadratic_whole_coefficients'])
    need(cc.nrows()==wc.nrows()==705 and cc.ncols()==wc.ncols()==4,'full authenticated Taylor residual')
    residual=arb_mat([[cc[i,0]+cc[i,1]*d+cc[i,2]*d*d+wc[i,3]*d*d*d]for i in range(705)])
    for i in range(705):U.contains(chart['predictor_residual'][i][0],D.ball_json(residual[i,0]),'same or narrower original predictor residual')
    partial=arb_mat(705,1);partial[704,0]=-1
    return dict(C=C,J=J,G0=G0,path_residual=residual,parameter_partial=partial,center=c,slope=s,curvature=a,weights=w,halfwidth=h,radius=rho)


def validate_prefix(card,producer_outcome,independent_outcome,independent_binding):
    need(card['units']==[1],'one fixed next chart only')
    pr=card['producer'];po=producer_outcome;io=independent_outcome;ib=independent_binding
    need(po['commit']==pr['source']and not po['passed']and po['connected_count']==1 and not po['owned_live_pids'],'retain original partial batch and only its successful prefix')
    need(len(po['workers'])==4 and [v['unit']for v in po['workers']]==[1,2,3,4]and [v['exit_code']for v in po['workers']]==[0,2,2,2]and all(not v['owned_live_pids']for v in po['workers']),'complete original four-worker outcomes with three failures retained')
    need(io['passed']and io['connected']and io['connected_count']==1 and io['native_calls']==0 and io['connected_parameter_endpoints']==geometry(1)[1],'independent same-root wide chart and exact interval')
    need(ib['card']==card['independent_input_card']and ib['card']['root']==pr['root']and ib['card']['outer']==pr['outer']and ib['card']['source']==pr['source'],'literal authenticated independent input card')
    need(ib['prior']==ib['card']['prior']['root']and ib['prior_manifest']==ib['card']['prior']['outer']and len(ib['charts'])==1,'exact independent connected predecessor')
    need(ib['card']['charts']==[dict(path='chart001/run',unit=1,worker=pr['worker'])]and ib['card']['qualified_prefix']==1 and ib['card']['expected_workers']==4,'only one qualified original chart')


def bind(card,unit,tick):
    geometry(unit);sc,ends=geometry(unit);producer=card['producer'];pr=Path(producer['root']);O.verify(pr,producer['outer'])
    po=read(pr/'outcome.json');run=pr/'chart001/run';O.verify(run,producer['worker']);rt=D.B.archived(run,producer['source'],'runtime.json','source');done=read(run/'outcome.json')
    need(done['passed']and done['unit']==unit and done['ticks']==253 and done['strict_inclusions']==705 and done['events']['direction_return']==144 and done['events']['pal_path_phase_return']==108,'exact complete selected full705 original acquisition inventory')
    independent=card['independent'];ir=Path(independent['root']);O.verify(ir,independent['outer']);O.verify(ir/'run',independent['worker']);D.B.archived(ir/'run',independent['source'],'runtime.json','source')
    io=read(ir/'run/outcome.json');il=read(ir/'launcher-result.json');ib=read(ir/'run/input-binding.json');validate_prefix(card,po,io,ib)
    need(il['exit_code']==0 and not il['owned_live_pids'],'independent complete ownership')
    selected=ib['charts'][0];need(selected['root']==str(run)and selected['manifest']==O.entry(run/'manifest.json')and selected['certificate']==O.entry(run/'certificate.json.gz'),'same literal independently proved full chart')
    uniform=read(ir/'run/chart-1/uniform.json');join=read(ir/'run/chart-1/join.json');need(uniform['inclusions']==705 and all(v for k,v in uniform['gates'].items()if k!='anchor_identified')and join['passed']and join['coordinates']==705 and join['parameter']==ends[0],'strict full uniform native proof and known-root join at the exact shared parameter')
    chart=read(run/'certificate.json.gz');ci=read(run/'chart-inputs.json');need(all(chart[k]==ci[k]for k in ['normal','origin','weights']),'same original gauge and scales')
    args=chart_inputs(chart,unit);inp=read(run/'inputs.json')['map_inputs'];need(inp['network']['d']==2 and inp['network']['width']==3 and inp['network']['bias']is True and 1<=len(inp['x'])<=24,'same profile-sized13D model')
    for name in B.NATIVE+T.PATH_FILES:need(O.entry(ROOT/name)==rt['sources'][name]==O.entry(run/'source'/name),'unchanged analytic native/path source '+name)
    frozen=card['frozen_basis'];fr=Path(frozen['root']);O.verify(fr,frozen['outer']);O.verify(fr/'run',frozen['worker']);D.B.archived(fr/'run',frozen['source'],'runtime.json','source')
    fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json');fb=read(fr/'run/bindings.json');prop=read(fr/'run/events/001-proposal.json')
    need(not fo['passed']and fo['status']=='completed_sufficient_bound_unresolved'and fo['native_calls']==0 and fl['exit_code']==2 and not fl['owned_live_pids']and prop['strategy']=='periodic_lyapunov_balanced_orthogonal','retained original basis proposal, no inference from failed bound')
    origin=card['physical_origin'];rr=Path(origin['root']);need(fb['card']['producer']==origin,'same exact source of frozen proposals');O.verify(rr,origin['outer']);O.verify(rr/'run',origin['worker']);D.B.archived(rr/'run',origin['source'],'runtime.json','source')
    need(inp==read(rr/'run/inputs.json')['map_inputs'],'identical original native model/data/rates across exact family')
    E=[U.R.matrix(v)for v in prop['phase_complements']];metrics=[D.exact_float_matrix(D.np.asarray(v))for v in prop['upper_cholesky_factors']]
    need(len(E)==len(metrics)==18 and all(v.is_exact()and v.is_finite()for m in E+metrics for row in m.tolist()for v in row),'all18 unchanged exact E/B proposals')
    model=B.TwoRateCuspMap(inp);need(model.n==13,'exact native dimension');normal=[U.R.exact(F(v))for v in chart['normal']];originv=[U.R.exact(F(v))for v in chart['origin']]
    need(len(normal)==len(originv)==705 and any(normal),'full exact affine gauge');rp=path_residual_derivative(U.R.matrix(chart['quadratic_center_coefficients']),U.R.matrix(chart['quadratic_whole_coefficients']),args['halfwidth'])
    oldwhole=chart['whole_box'];del chart;gc.collect();tick()
    return model,args,oldwhole,rp,normal,originv,U.R.exact(F(sc)),E,metrics,dict(card=card,unit=unit,parameter_endpoints=ends,map_inputs=inp,original_run=str(run),original_certificate=selected['certificate'],independent_uniform=O.entry(ir/f'run/chart-{unit}/uniform.json'),independent_join=O.entry(ir/f'run/chart-{unit}/join.json'),shared_native_acquisitions=252,scope='Same independently connected primitive18 quadratic root; all705 variables retained. Source-authenticated original C/J/Taylor data, new correlated tangent and complementary arithmetic; no new roots or later charts.')


def save_bound(out,name,value,used,closing=False):
    raw=D.canonical_bytes(value);need(used[0]+len(raw)<(128 if closing else 124)*1024**2,'128MiB cap with4MiB closing reserve');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    used[0]+=len(raw)


def main(unit):
    geometry(unit);resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');out=BASE/f'chart{unit:03d}/run';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();used=[0];stopped=[False];calls=[0];result=dict(passed=False,status='incomplete',unit=unit)
    def save(name,v,closing=False):save_bound(out,name,v,used,closing)
    def tick():need(not stopped[0]and time.monotonic()-start<400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'400s/4GiB owned worker cap')
    def paths(model,flat,v,normal,origin,sc,stage):
        begin=calls[0];need(begin in(0,36,108,144),'exact four36-phase path groups')
        class Recorded:
            n=13
            def cusp_step(self,x,a,b):
                tick();need(calls[0]<216 and calls[0]-begin<36,'216 total/36 path group native cap');calls[0]+=1;phase=(calls[0]-begin-1)//2;direction='q'if(calls[0]-begin)%2 else'r'
                save(f'native/{calls[0]:03d}-input.json',dict(kind='path',stage=stage,phase=phase,direction=direction,inputs=[z.serialized()for z in x+[a,b]]));value=model.cusp_step(x,a,b);save(f'native/{calls[0]:03d}-return.json',dict(kind='path',stage=stage,returns=[z.serialized()for z in value]));tick();return value
        def observe(kind,row):
            if kind=='pal_path_assembly_return':save('coefficients/'+stage+'.json',row)
        answer=path_coefficients(Recorded(),flat,v,[arb(0)]*705,arb(0),18,normal,origin,sc,tick,observe);need(calls[0]-begin==36,'complete retained path group');return answer
    def jac(model,x,rates,v,vp,kind,phase):
        tick();need(72<=calls[0]<108 or 180<=calls[0]<216,'two exact36 native total-derivative groups');calls[0]+=1
        save(f'native/{calls[0]:03d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));a,ap=total_two_rate_jacobian(model,x,rates,v,vp);save(f'native/{calls[0]:03d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(a),Aprime=matrix_json(ap)));tick();return a,ap
    signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,400)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},unit=unit,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),precision=384,python=D.platform.python_version(),numpy=D.np.__version__,scipy=D.scipy.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,native_call_cap=216))
        for n in FILES:
            p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);raw=(ROOT/n).read_bytes();need(used[0]+len(raw)<124*1024**2,'source archive cap');p.write_bytes(raw);used[0]+=len(raw)
        with ctx.workprec(384):
            before=time.monotonic();trial=L.synthetic(tick);need(trial['passed'],'unrelated full705 localization');del trial;gc.collect()
            m,flat,plus,minus,pprime,normal,origin,sc,inp=T.synthetic_inputs();save('profile-path-inputs.json',dict(map_inputs=inp,segment=vector_json(flat),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=D.ball_json(sc)))
            cp=paths(m,flat,plus,normal,origin,sc,'profile_plus');cm=paths(m,flat,minus,normal,origin,sc,'profile_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);small=arb((1,-16));n=705;C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)])
            trial=refine_tangent(C,J,arb_mat(n,1),mixed,pprime,[arb(1)]*n,tick=tick);need(trial['passed'],'full705 correlated tangent profile');del C,J,cp,cm,mixed,trial;gc.collect()
            inp,x,rates,v,vp=B.gd_fixture();m=B.TwoRateCuspMap(inp);save('profile-jac-inputs.json',dict(map_inputs=inp))
            for j in range(36):jac(m,x,rates,v,vp,'synthetic',j)
            trial=B.synthetic_cocycle(tick);need(trial['passed'],'full18phase11-mode profile');seconds=time.monotonic()-before;forecast=4*seconds+4*(15.419+21.120+14.366)+20
            save('profile.json',dict(passed=True,seconds=seconds,forecast_seconds=forecast,native_calls=calls[0],target_arrays_unopened=True));need(calls[0]==108 and forecast<400,'measured whole-pipeline admission before target arrays');del trial,m;gc.collect()
            card=read(ROOT/CARD);model,args,oldwhole,rp,normal,origin,sc,E,metrics,binding=bind(card,unit,tick);save('bindings.json',binding)
            localization=localize_quadratic_root(**args,tick=tick);save('localization.json',localization);segment,error,pprime,plus,minus=T.segment_inputs(args,localization,oldwhole)
            save('tangent-domain-inputs.json',dict(segment=vector_json(segment),error=vector_json(error),predictor_tangent=vector_json(pprime),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=D.ball_json(sc),scale='1/256',parameter_endpoints=geometry(unit)[1],residual_derivative=matrix_json(rp)))
            cp=paths(model,segment,plus,normal,origin,sc,'target_plus');cm=paths(model,segment,minus,normal,origin,sc,'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);tangent=refine_tangent(args['C'],args['J'],rp,mixed,pprime,args['weights'],tick=tick);save('tangent.json',tangent)
            localization['tangent_box']=tangent['tangent_box'];center,whole,v,terminal,full=B.native_domains(localization);save('quotient-domain-inputs.json',dict(center=vector_json(center),whole=vector_json(whole),tangent=vector_json(v),full705=full,terminal=matrix_json(terminal),halfwidth='1/64',parameter_endpoints=geometry(unit)[1],complements=[matrix_json(e)for e in E],metrics=[matrix_json(e)for e in metrics]))
            rows=[]
            for j in range(18):
                off=39*j;A0,_=jac(model,center[off:off+13],center[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);Aw,Ap=jac(model,whole[off:off+13],whole[-2:],v[off:off+13],v[-2:],'whole_tangent',j);rows.append(dict(phase=j,center_A=A0,whole_A=Aw,whole_Aprime=Ap))
            need(calls[0]==216,'all108 profile and108 target native calls');seq=[0]
            def observe(kind,row):seq[0]+=1;save(f'events/{seq[0]:03d}-{kind}.json',row);tick()
            proof=correlated_complement(rows,center,whole,v,E,metrics,args['halfwidth'],terminal,tick=tick,observe=observe);save('complement.json',proof)
            result.update(passed=proof['passed'],status='whole_chart_complement_attracting'if proof['passed']else'completed_correlated_bound_unresolved',parameter_endpoints=geometry(unit)[1],remainder=proof['remainder_norm_upper'],tangent_correction=tangent['tangent_error']['weighted_uniform_radius'],metric_passed=proof['metric']['passed'],complement_dimension=11,scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=calls[0],elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',type=int,choices=[1],required=True);a=p.parse_args();raise SystemExit(main(a.unit))
