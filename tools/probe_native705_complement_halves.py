#!/usr/bin/env python3
"""Two fixed closed restrictions of one known705 root; fresh local J/C."""
import argparse,gc,gzip,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_complement_wide as W
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_quadratic_restriction import restrict_quadratic
from neural_shrimp.rigorous_unit_pair_pal import assemble,proposal,path_coefficients
from neural_shrimp.rigorous_quadratic_localization import localize_quadratic_root
from neural_shrimp.rigorous_quadratic_tangent_forcing import mixed_hessian_from_second_coefficients,refine_tangent
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import vector_json,matrix_json
B,T,L,D,U,O,need,read=W.B,W.T,W.L,W.D,W.U,W.O,W.need,W.read
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/wide-halves001')
CARD='docs/plans/native705-complement-halves-inputs.json'
FILES=sorted(set(W.FILES+[CARD,'docs/plans/native705-complement-halves.md','docs/plans/native705-restricted-complement.md','docs/reviews/native705-complement-wide001.json','src/neural_shrimp/rigorous_quadratic_restriction.py','tests/test_rigorous_quadratic_restriction.py','tools/probe_native705_complement_halves.py','tools/launch_native705_complement_halves.py','tests/test_native705_complement_halves.py']))


def geometry(unit):
    need(type(unit)is int and unit in(1,2),'exact two closed halves only')
    return str(F(351+64*(unit-1),4096)),[str(F(319+64*(unit-1),4096)),str(F(383+64*(unit-1),4096))]


def bind(card,unit,tick):
    need(card['units']==[1,2],'fixed complete two-half panel');sc,ends=geometry(unit)
    failure=card['failed_wide'];fr=Path(failure['root']);O.verify(fr,failure['outer']);O.verify(fr/'chart001/run',failure['worker']);D.B.archived(fr/'chart001/run',failure['source'],'runtime.json','source')
    fo=read(fr/'chart001/run/outcome.json');fl=read(fr/'launcher-result.json');fb=read(fr/'chart001/run/bindings.json')
    need(not fo['passed']and fo['status']=='completed_correlated_bound_unresolved'and fo['native_calls']==216 and not fl['passed']and not fl['owned_live_pids']and all(v['exit_code']==2 and not v['owned_live_pids']for v in fl['workers']),'complete retained original inverse-wrapping failure')
    need(fb['card']==card['original']and fo['parameter_endpoints']==W.geometry(1)[1],'same original failed chart and map')
    model,a,_,_,normal,origin,oldsc,E,metrics,parent=W.bind(card['original'],1,tick)
    original=read(Path(parent['original_run'])/'certificate.json.gz')
    cc=U.R.matrix(original['quadratic_center_coefficients']);wc=U.R.matrix(original['quadratic_whole_coefficients'])
    newsc=U.R.exact(F(sc));shift=newsc-oldsc;h=arb((1,-7))
    restricted=restrict_quadratic(a['center'],a['slope'],a['curvature'],a['weights'],a['halfwidth'],a['radius'],shift,h,cc,wc)
    need(shift==arb((-1 if unit==1 else 1,-7))and all(v>0 for v in restricted['fresh_domain'][-3:]),'exact half shift and positive fresh rates/angle')
    rp=restricted.pop('residual_derivative');tick()
    binding=dict(card=card,unit=unit,parameter_endpoints=ends,original=parent,shift=D.ball_json(shift),original_parameter_center=D.ball_json(oldsc),scope='Exact same known primitive18 root restricted to one closed half. Fresh full705 derivatives and C; original Taylor residual authority retained only on the original parameter domain. No new root or Newton iteration.')
    del original,a,cc,wc;gc.collect()
    return model,restricted,rp,normal,origin,newsc,E,metrics,binding


def jet_json(z):
    return dict(f=D.ball_json(z.f),g=vector_json(z.g),d=D.ball_json(z.d),m=vector_json(z.m))

def save_bound(out,name,value,used,closing=False):
    raw=D.canonical_bytes(value);raw=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw;need(used[0]+len(raw)<(256 if closing else 252)*1024**2,'256MiB cap with4MiB closing reserve');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
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
        begin=calls[0];need(begin in(72,108,252,288),'exact four36-phase path groups after local-J acquisitions')
        class Recorded:
            n=13
            def cusp_step(self,x,a,b):
                tick();need(calls[0]<360 and calls[0]-begin<36,'360 total/36 path group native cap');calls[0]+=1;phase=(calls[0]-begin-1)//2;direction='q'if(calls[0]-begin)%2 else'r'
                save(f'native/{calls[0]:03d}-input.json',dict(kind='path',stage=stage,phase=phase,direction=direction,inputs=[z.serialized()for z in x+[a,b]]));value=model.cusp_step(x,a,b);save(f'native/{calls[0]:03d}-return.json',dict(kind='path',stage=stage,returns=[z.serialized()for z in value]));tick();return value
        def observe(kind,row):
            if kind=='pal_path_assembly_return':save('coefficients/'+stage+'.json',row)
        answer=path_coefficients(Recorded(),flat,v,[arb(0)]*705,arb(0),18,normal,origin,sc,tick,observe);need(calls[0]-begin==36,'complete retained path group');return answer
    def jac(model,x,rates,v,vp,kind,phase):
        tick();need(144<=calls[0]<180 or 324<=calls[0]<360,'two exact36 native total-derivative groups');calls[0]+=1
        save(f'native/{calls[0]:03d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));a,ap=total_two_rate_jacobian(model,x,rates,v,vp);save(f'native/{calls[0]:03d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(a),Aprime=matrix_json(ap)));tick();return a,ap
    def full_assembly(model,flat,normal,origin,sc,stage):
        begin=calls[0];need(begin in(0,36,180,216),'fixed four36-call full705 assembly groups')
        save('assembly/'+stage+'-domain.json',dict(flat=vector_json(flat),normal=vector_json(normal),origin=vector_json(origin),parameter=D.ball_json(sc)))
        class Recorded:
            n=13
            def cusp_step(self,x,a,b):
                tick();need(calls[0]<360 and calls[0]-begin<36,'fixed360/36 full-J call cap');calls[0]+=1
                save(f'native/{calls[0]:03d}-input.json',dict(kind='full_jacobian',stage=stage,phase=(calls[0]-begin-1)//2,direction='q'if(calls[0]-begin)%2 else'r',inputs=[jet_json(v)for v in x+[a,b]]))
                values=model.cusp_step(x,a,b);save(f'native/{calls[0]:03d}-return.json',dict(kind='full_jacobian',stage=stage,returns=[jet_json(v)for v in values]));tick();return values
        def observe(kind,row):
            if kind=='direction_return':save(f'assembly/{stage}-{row["phase"]:02d}-{row["direction"]}.json',row)
        g,j,local=assemble(Recorded(),flat,18,normal,origin,sc,tick,observe)
        need(calls[0]-begin==36,'complete fresh full705 Jacobian group');save('assembly/'+stage+'-full.json.gz',dict(value=matrix_json(g),jacobian=matrix_json(j)));return g,j,local
    signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,400)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},unit=unit,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),precision=384,python=D.platform.python_version(),numpy=D.np.__version__,scipy=D.scipy.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,native_call_cap=360))
        for n in FILES:
            p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);raw=(ROOT/n).read_bytes();need(used[0]+len(raw)<252*1024**2,'source archive cap');p.write_bytes(raw);used[0]+=len(raw)
        with ctx.workprec(384):
            before=time.monotonic();trial=L.synthetic(tick);need(trial['passed'],'unrelated full705 localization');del trial;gc.collect()
            m,flat,plus,minus,pprime,normal,origin,sc,inp=T.synthetic_inputs();save('profile-path-inputs.json',dict(map_inputs=inp,segment=vector_json(flat),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=D.ball_json(sc)))
            profile_center=[v.mid()for v in flat];pg,pj,pl=full_assembly(m,profile_center,normal,origin,sc,'profile_center');pc,small=proposal(pj,pl,13,profile_center[-1]);save('assembly/profile-preconditioner.json.gz',dict(C=matrix_json(pc),border=matrix_json(small)));del pg,pj,pl,pc,small;gc.collect()
            pg,pj,pl=full_assembly(m,flat,normal,origin,sc,'profile_whole');del pg,pj,pl;gc.collect()
            cp=paths(m,flat,plus,normal,origin,sc,'profile_plus');cm=paths(m,flat,minus,normal,origin,sc,'profile_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);small=arb((1,-16));n=705;C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)])
            trial=refine_tangent(C,J,arb_mat(n,1),mixed,pprime,[arb(1)]*n,tick=tick);need(trial['passed'],'full705 correlated tangent profile');del C,J,cp,cm,mixed,trial;gc.collect()
            inp,x,rates,v,vp=B.gd_fixture();m=B.TwoRateCuspMap(inp);save('profile-jac-inputs.json',dict(map_inputs=inp))
            for j in range(36):jac(m,x,rates,v,vp,'synthetic',j)
            trial=B.synthetic_cocycle(tick);need(trial['passed'],'full18phase11-mode profile');seconds=time.monotonic()-before;forecast=4*seconds+4*(15.419+21.120+14.366)+20
            storage_forecast=3*used[0]+48*1024**2;save('profile.json',dict(passed=True,seconds=seconds,forecast_seconds=forecast,forecast_bytes=storage_forecast,native_calls=calls[0],target_arrays_unopened=True));need(calls[0]==180 and forecast<400 and storage_forecast<256*1024**2,'measured full-pipeline time/storage admission before target arrays');del trial,m;gc.collect()
            card=read(ROOT/CARD);model,restricted,rp,normal,origin,sc,E,metrics,binding=bind(card,unit,tick);save('bindings.json',binding)
            save('restriction.json',dict(center=vector_json(restricted['center']),slope=vector_json(restricted['slope']),curvature=vector_json(restricted['curvature']),weights=vector_json(restricted['weights']),radius=D.ball_json(restricted['radius']),halfwidth=D.ball_json(restricted['halfwidth']),original_delta=D.ball_json(restricted['original_delta']),fresh_domain=vector_json(restricted['fresh_domain']),path_residual=matrix_json(restricted['path_residual']),residual_derivative=matrix_json(rp)))
            g0,j0,local=full_assembly(model,restricted['center'],normal,origin,sc,'target_center');C,small=proposal(j0,local,13,restricted['center'][-1]);save('assembly/target-preconditioner.json.gz',dict(C=matrix_json(C),border=matrix_json(small)));del j0,local,small;gc.collect()
            gw,J,local=full_assembly(model,restricted['fresh_domain'],normal,origin,sc,'target_whole');del gw,local;gc.collect()
            partial=arb_mat(705,1);partial[704,0]=-1;oldwhole=vector_json(restricted['fresh_domain']);args={k:restricted[k]for k in ['center','slope','curvature','weights','halfwidth','radius','path_residual']};args.update(C=C,J=J,G0=g0,parameter_partial=partial);del restricted;gc.collect()
            localization=localize_quadratic_root(**args,tick=tick);save('localization.json',localization);segment,error,pprime,plus,minus=T.segment_inputs(args,localization,oldwhole)
            save('tangent-domain-inputs.json',dict(segment=vector_json(segment),error=vector_json(error),predictor_tangent=vector_json(pprime),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=D.ball_json(sc),scale='1/256',parameter_endpoints=geometry(unit)[1],residual_derivative=matrix_json(rp)))
            cp=paths(model,segment,plus,normal,origin,sc,'target_plus');cm=paths(model,segment,minus,normal,origin,sc,'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);tangent=refine_tangent(args['C'],args['J'],rp,mixed,pprime,args['weights'],tick=tick);save('tangent.json',tangent)
            localization['tangent_box']=tangent['tangent_box'];center,whole,v,terminal,full=B.native_domains(localization);save('quotient-domain-inputs.json',dict(center=vector_json(center),whole=vector_json(whole),tangent=vector_json(v),full705=full,terminal=matrix_json(terminal),halfwidth='1/128',parameter_endpoints=geometry(unit)[1],complements=[matrix_json(e)for e in E],metrics=[matrix_json(e)for e in metrics]))
            rows=[]
            for j in range(18):
                off=39*j;A0,_=jac(model,center[off:off+13],center[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);Aw,Ap=jac(model,whole[off:off+13],whole[-2:],v[off:off+13],v[-2:],'whole_tangent',j);rows.append(dict(phase=j,center_A=A0,whole_A=Aw,whole_Aprime=Ap))
            need(calls[0]==360,'all180 profile and180 target native calls');seq=[0]
            def observe(kind,row):seq[0]+=1;save(f'events/{seq[0]:03d}-{kind}.json',row);tick()
            proof=correlated_complement(rows,center,whole,v,E,metrics,args['halfwidth'],terminal,tick=tick,observe=observe);save('complement.json',proof)
            result.update(passed=proof['passed'],status='whole_chart_complement_attracting'if proof['passed']else'completed_correlated_bound_unresolved',parameter_endpoints=geometry(unit)[1],remainder=proof['remainder_norm_upper'],tangent_correction=tangent['tangent_error']['weighted_uniform_radius'],metric_passed=proof['metric']['passed'],complement_dimension=11,scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=calls[0],elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',type=int,choices=[1,2],required=True);a=p.parse_args();raise SystemExit(main(a.unit))
