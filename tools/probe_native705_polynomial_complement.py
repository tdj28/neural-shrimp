#!/usr/bin/env python3
"""One original chart31 quarter, retaining the certified polynomial inverse."""
import gc,gzip,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_later_complement as P
import diagnose_native705_inverse_correlation as V
from neural_shrimp.rigorous_polynomial_localization import localize_polynomial_root,refine_polynomial_tangent
from neural_shrimp.rigorous_unit_pair_pal import path_coefficients
from neural_shrimp.rigorous_quadratic_tangent_forcing import balanced_directions,mixed_hessian_from_second_coefficients
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json
from flint import arb,arb_mat,ctx
M,H,B,T,L,D,U,O,need,read=P.M,P.H,P.B,P.T,P.L,P.D,P.U,P.O,P.need,P.read
OUT=P.BASE.with_name('polynomial001')/'run';LEFT=F(-49,2048);RIGHT=F(-17,2048);SHIFT=F(-3,128);HALF=F(1,128)
DIAG=P.BASE.with_name('later001-inverse-diagnosis');DIAG_OUTER='3c64c71a014c3193b9b246fd21efd220d54bb99452a0de288e57e38671acf0a0'
FILES=sorted(set(V.FILES+['src/neural_shrimp/rigorous_polynomial_localization.py','tests/test_polynomial_localization.py','tools/probe_native705_polynomial_complement.py','tools/launch_native705_polynomial_complement.py','tests/test_native705_polynomial_complement.py','docs/plans/native705-polynomial-localization.md','docs/plans/native705-polynomial-complement.md','docs/reviews/native705-inverse-correlation-diagnosis.json']))


def small_matrix(path,cols):
    raw=read(path);need(raw['encoding']=='mid-rad-dyadic-pairs'and len(raw['rows'])==705 and all(len(row)==cols for row in raw['rows']),'literal full705 independent coefficient')
    rows=[]
    for row in raw['rows']:
        rr=[]
        for m,r in row:
            need(len(m)==len(r)==2 and all(type(x)is int for x in m+r)and r[0]>=0,'literal dyadic coefficient encoding');z=arb(arb(tuple(m)),arb(tuple(r)));need(z.is_finite(),'finite original coefficient');rr.append(z)
        rows.append(rr)
    return arb_mat(rows)


def exact_coverage(old,new,message):
    for a,b in zip(old,new,strict=True):
        lo,hi=U.M.interval(a);aa,bb=U.M.interval(ball_json(b));need(lo<=aa<=bb<=hi,message)


def bind(w):
    O.verify(P.BASE,V.FAILED);old=read(P.BASE/'launcher-result.json');need(not old['passed']and old['launched']==4 and old['complete_charts']==0 and not old['owned_live_pids'],'retained four failed quarter attempts')
    O.verify(DIAG,DIAG_OUTER);diagnosis=read(DIAG/'run/outcome.json');need(diagnosis['passed']and diagnosis['native_calls']==0 and len(diagnosis['rows'])==4,'retained saved diagnosis');D.B.archived(DIAG/'run','34dddac79a68565d025cc6ca635f89f7072cbc14','runtime.json','source')
    card=read(ROOT/P.CARD);row,run,ir,rt,domain,inp,proof,ib=M.metadata(card,31,w.tick);need(read(P.BASE/'fixed-index.json')==card,'same original literal root index')
    need(row['parameter_endpoints']==['-49/2048','79/2048']and next(v for v in old['attempts']if v['assignment']['attempt']==1)['assignment']['cell']['endpoints']==[str(LEFT),str(RIGHT)],'fixed first failed quarter only')
    cfg=row['producer']['settings'];sc=U.R.exact(F(cfg['parameter']));h=F(1,32);need(F(cfg['parameter'])+SHIFT-HALF==LEFT and F(cfg['parameter'])+SHIFT+HALF==RIGHT,'exact original-offset quarter')
    normal=[U.R.exact(F(x))for x in proof['chart']['normal']];origin=[U.R.exact(F(x))for x in proof['chart']['origin']];c,s,a=[[U.R.exact(U.M.point(v))for v in domain[k]]for k in ('center','slope','curvature')]
    radii=[U.R.exact(U.M.point(v))for v in proof['root']['radii']];weights=[v/arb((1,-10))for v in radii];original=arb_mat([[arb(0,r)]for r in radii]);image=arb_mat([[U.R.original_ball(v)]for v in proof['root']['image_correction']]);exact_coverage(domain['error'],[original[i,0]for i in range(705)],'original acquired E covers restored initial correction')
    # The archived quadratic_tube source emits this exact pre variable, then
    # serializes the same unmodified variable as certificate.preconditioner.
    prior=ib['card']['proposal'];pr=Path(prior['root'])/'run';O.verify(pr,prior['worker']);prt=D.B.archived(pr,ib['card']['packet']['source'],'runtime.json','source');need(prt['sources']['src/neural_shrimp/rigorous_unit_pair_pal.py']==rt['sources']['src/neural_shrimp/rigorous_unit_pair_pal.py'],'same immutable preconditioner emission/certificate source')
    candidates=sorted((pr/'events').glob('*-pal_preconditioner_return.json.gz'));need(len(candidates)==1,'single literal old C0 proposal emission');C0=U.R.matrix(read(candidates[0])['matrix']);C1=small_matrix(ir/'target/inverse_linear_proposal.json.gz',705);need(all(x.is_exact()for z in [C0,C1]for rr in z.tolist()for x in rr),'literal exact old C0/C1 at480bits')
    defect=[small_matrix(ir/f'target/defect_coefficient_{k}.json.gz',705)for k in range(4)];forcing=[small_matrix(ir/f'target/forcing_coefficient_{k}.json.gz',1)for k in range(5)];res=[small_matrix(ir/f'target/residual_coefficient_{k}.json.gz',1)for k in range(4)];rp=[res[1],2*res[2],3*res[3]]
    # R' has its own Taylor theorem with the original whole third derivative.
    E,metrics=M.OLD.fixed_frames(inp['map_inputs'],card);model=B.TwoRateCuspMap(inp['map_inputs']);w.tick()
    payload=dict(index=row,prior_card=ib['card'],old_preconditioner=O.entry(candidates[0]),old_preconditioner_path=str(candidates[0]),coefficient_artifacts={str(q.relative_to(ir)):O.entry(q)for q in sorted((ir/'target').glob('*coefficient_*.json.gz'))},inverse_linear=O.entry(ir/'target/inverse_linear_proposal.json.gz'),original_failure=V.FAILED,diagnosis=DIAG_OUTER,parameter_endpoints=[str(LEFT),str(RIGHT)],original_offset=[str(SHIFT-HALF),str(SHIFT+HALF)],scope='Same independently proved native705 root and original affine C/defect/residual family; no new root or fixed-C Jacobian approximation. All four failed prior quarters remain retained; only first quarter is tested.')
    w.save('input-binding.json',payload);return model,c,s,a,normal,origin,sc,weights,original,image,[C0,C1],defect,forcing,rp,E,metrics,domain


class Work:
    def __init__(self):self.start=time.monotonic();self.used=0;self.calls=0;self.stopped=False
    def tick(self):need(not self.stopped and time.monotonic()-self.start<400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'400s4GiB owned polynomial worker')
    def save(self,name,v,closing=False):
        raw=D.canonical_bytes(v);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw;need(self.used+len(raw)<(384 if closing else 380)*1024**2,'384MiB with4MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.used+=len(raw)
    def paths(self,model,flat,direction,normal,origin,sc,stage):
        start=self.calls;need(start in (0,36,108,144),'fixed four full36-phase polarized groups');owner=self
        class Recorded:
            n=13
            def cusp_step(self,x,a,b):
                owner.tick();need(owner.calls<216 and owner.calls-start<36,'complete bounded216/36 calls');owner.calls+=1;phase=(owner.calls-start-1)//2;which='q'if(owner.calls-start)%2 else'r'
                owner.save(f'native/{owner.calls:03d}-input.json',dict(kind='path',stage=stage,phase=phase,direction=which,inputs=[z.serialized()for z in x+[a,b]]));values=model.cusp_step(x,a,b);owner.save(f'native/{owner.calls:03d}-return.json',dict(kind='path',stage=stage,returns=[z.serialized()for z in values]));owner.tick();return values
        coeff=path_coefficients(Recorded(),flat,direction,[arb(0)]*705,arb(0),18,normal,origin,sc,self.tick,lambda kind,row:None);need(self.calls-start==36,'all actual polarized phases');self.save('coefficients/'+stage+'.json',matrix_json(coeff));return coeff
    def jac(self,model,x,rates,v,vp,kind,phase):
        self.tick();need(72<=self.calls<108 or 180<=self.calls<216,'exact36 actual total-derivative group');self.calls+=1;self.save(f'native/{self.calls:03d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));A,Ap=total_two_rate_jacobian(model,x,rates,v,vp);self.save(f'native/{self.calls:03d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(A),Aprime=matrix_json(Ap)));self.tick();return A,Ap


def synthetic(w):
    before=time.monotonic();n=705;h=F(1,32);I=arb_mat([[int(i==j)for j in range(n)]for i in range(n)]);small=arb((1,-24));dense=arb_mat([[small for _ in range(n)]for _ in range(n)]);ds=[dense,dense/2,dense/4,dense/8];C=[I+dense,dense];zero=arb_mat(n,1);ff=[arb_mat([[arb((1,-60))]for _ in range(n)])]+[zero]*4;old=arb_mat([[arb(0,arb((1,-10)))]for _ in range(n)])
    _,loc=localize_polynomial_root(ds,ff,[arb(1)]*n,old,-h,0,h,tick=w.tick);_,point=localize_polynomial_root(ds,ff,[arb(1)]*n,old,-h/2,-h/2,h,tick=w.tick);need(loc['passed']and point['passed'],'full705 family/center synthetic')
    model,flat,plus,minus,pprime,normal,origin,sc,inp=T.synthetic_inputs();w.save('profile-native-inputs.json',dict(map_inputs=inp,segment=vector_json(flat),plus=vector_json(plus),minus=vector_json(minus),predictor_tangent=vector_json(pprime),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(sc)))
    cp=w.paths(model,flat,plus,normal,origin,sc,'profile_plus');cm=w.paths(model,flat,minus,normal,origin,sc,'profile_minus');mix=mixed_hessian_from_second_coefficients(cp,cm);_,tangent=refine_polynomial_tangent(C,ds,[zero]*3,mix,arb_mat([[v]for v in pprime]),[arb(1)]*n,-h,0,h,tick=w.tick);need(tangent['passed'],'full705 polynomial tangent synthetic');del C,ds,I,dense,cp,cm,mix,tangent,loc,point;gc.collect()
    inp,x,rates,v,vp=B.gd_fixture();model=B.TwoRateCuspMap(inp);w.save('profile-jac-inputs.json',dict(map_inputs=inp))
    for j in range(36):w.jac(model,x,rates,v,vp,'synthetic',j)
    cocycle=B.synthetic_cocycle(w.tick);need(cocycle['passed'],'full18phase11mode synthetic');seconds=time.monotonic()-before;forecast=4*seconds+2*61.466+30;raw=3*w.used+96*1024**2
    report=dict(passed=w.calls==108 and forecast<400 and raw<384*1024**2,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,historical_saved_decode_seconds=61.466,native_calls=w.calls,target_arrays_unopened=True);w.save('profile.json',report);need(report['passed'],'complete measured profile and original saved decode cost admission')


def target(w):
    model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=bind(w)
    lo,hi=SHIFT-HALF,SHIFT+HALF;error,wholeproof=localize_polynomial_root(ds,ff,weights,original,lo,hi,F(1,32),known_image=image,tick=w.tick);ce,centerproof=localize_polynomial_root(ds,ff,weights,original,SHIFT,SHIFT,F(1,32),known_image=image,tick=w.tick)
    dd=arb(U.R.exact(SHIFT),U.R.exact(HALF));shift=U.R.exact(SHIFT);predictor=[c[i]+s[i]*dd+a[i]*dd*dd/2 for i in range(705)];pprime=[s[i]+a[i]*dd for i in range(705)];center=[c[i]+s[i]*shift+a[i]*shift*shift/2+ce[i,0]for i in range(705)];whole=[predictor[i]+error[i,0]for i in range(705)];segment=[predictor[i]+arb(0,abs(error[i,0]).upper())for i in range(705)]
    exact_coverage(domain['whole_box'],center,'original native domain contains every center root coordinate');exact_coverage(domain['whole_box'],whole,'original native domain contains every root coordinate');exact_coverage(domain['whole_box'],segment,'original native domain contains complete predictor/root segment')
    plus,minus=balanced_directions([error[i,0]for i in range(705)],pprime,arb('1/256'));w.save('localization.json',dict(passed=True,dimension=705,center_refinement=centerproof,whole_refinement=wholeproof,center_box=vector_json(center),whole_box=vector_json(whole)))
    w.save('tangent-domain-inputs.json',dict(segment=vector_json(segment),error=matrix_json(error),predictor_tangent=vector_json(pprime),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(sc+shift),scale='1/256',parameter_endpoints=[str(LEFT),str(RIGHT)]))
    cp=w.paths(model,segment,plus,normal,origin,sc+shift,'target_plus');cm=w.paths(model,segment,minus,normal,origin,sc+shift,'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);tangent,proof=refine_polynomial_tangent(C,ds,rp,mixed,arb_mat([[v]for v in pprime]),weights,lo,hi,F(1,32),tick=w.tick);w.save('tangent.json.gz',proof);del C,ds,ff,cp,cm,mixed;gc.collect()
    loc=dict(passed=True,dimension=705,weighted_neumann_bound=proof['weighted_neumann_bound'],center_box=vector_json(center),whole_box=vector_json(whole),tangent_box=matrix_json(tangent));center,whole,v,terminal,full=B.native_domains(loc);w.save('quotient-domain-inputs.json',dict(center=vector_json(center),whole=vector_json(whole),tangent=vector_json(v),full705=full,terminal=matrix_json(terminal),halfwidth=str(HALF),parameter_endpoints=[str(LEFT),str(RIGHT)],complements=[matrix_json(e)for e in E],metrics=[matrix_json(e)for e in metrics]))
    rows=[]
    for j in range(18):
        off=39*j;A0,_=w.jac(model,center[off:off+13],center[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);Aw,Ap=w.jac(model,whole[off:off+13],whole[-2:],v[off:off+13],v[-2:],'whole_tangent',j);rows.append(dict(phase=j,center_A=A0,whole_A=Aw,whole_Aprime=Ap))
    need(w.calls==216,'complete108 profile and108 target phases');events=[0]
    def observe(kind,row):events[0]+=1;w.save(f'events/{events[0]:03d}-{kind}.json',row);w.tick()
    proof=correlated_complement(rows,center,whole,v,E,metrics,U.R.exact(HALF),terminal,tick=w.tick,observe=observe);w.save('complement.json',proof)
    return dict(passed=proof['passed'],status='first_quarter_complement_attracting'if proof['passed']else'completed_correlated_bound_unresolved',parameter_endpoints=[str(LEFT),str(RIGHT)],remainder=proof['remainder_norm_upper'],metric_passed=proof['metric']['passed'],complement_dimension=11,scope=proof['scope'])


def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,400)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,native_call_cap=216))
        for n in FILES:
            raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<380*1024**2,'bounded source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
        with ctx.workprec(480):synthetic(w);result=target(w)
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,108),target_native_calls=max(0,w.calls-108),elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
