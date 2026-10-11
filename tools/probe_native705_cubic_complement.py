"""Generic cubic known-root complement mechanism; fixed first chart32 portion."""
import gc,gzip,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_second_quarter as S
import bind_native705_cubic_complement as G
from neural_shrimp import cubic_complement_geometry as H
from neural_shrimp.rigorous_polynomial_localization import localize_polynomial_root,refine_polynomial_tangent
from neural_shrimp.rigorous_polynomial_point import finite_point_correction,localize_fresh_polynomial_point
from neural_shrimp.rigorous_quadratic_tangent_forcing import balanced_directions,mixed_hessian_from_second_coefficients
from neural_shrimp.rigorous_correlated_complement import correlated_complement
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json
from flint import arb,arb_mat,ctx
OLD,P,O,D,U,B,need,read=S.OLD,S.P,S.O,S.D,S.U,S.B,S.need,S.read
OUT=S.OUT.parent.with_name('cubic32-first-quarter001')/'run'
CARD='docs/plans/native705-cubic-complement-inputs.json'
CAPS=dict(seconds=3000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=6*1024**3,profile_calls=144,maximum_target_calls=3960,maximum_total_calls=4104)
FILES=sorted(set(S.FILES+['tools/reviews/check_native705_chart31_boundary_hypotheses.py','docs/reviews/native705-chart31-boundary-hypotheses.json','src/neural_shrimp/cubic_complement_geometry.py','tests/test_cubic_complement_geometry.py','tools/bind_native705_cubic_complement.py','tools/probe_native705_cubic_complement.py','tools/launch_native705_cubic_complement.py','tests/test_native705_cubic_complement.py','docs/plans/native705-cubic-complement.md',CARD]))


def card_check(card):
 need(card['status']=='reviewed'and card['ordinal']==32 and card['parameter_endpoints']==['39/1024','55/1024'],'fixed reviewed first cubic32 portion')
 need(card['domains']==H.domains(*card['parameter_endpoints'],2)and card['maximum_depth']==3 and card['maximum_centers']==30 and card['caps']==CAPS,'exact two domains/depth3/caps')
 need(card['root_index']==G.M.select(read(ROOT/G.M.CARD),32),'literal authenticated original cubic root index')


class Work(S.Work):
 def __init__(self):super().__init__();self.output_dir=OUT


def synthetic(w):
 # Retain the full144-phase prior native profile, then exercise the complete
 # cubic coefficient counts and exact signed parameter ranges at full705 size.
 OLD.synthetic(w);n=705;zero=arb_mat(n,1);I=arb_mat([[int(i==j)for j in range(n)]for i in range(n)]);small=arb((1,-24));dense=arb_mat([[small for _ in range(n)]for _ in range(n)]);ds=[dense,dense/2,dense/4,dense/8];C=[I+dense,dense];ff=[arb_mat([[arb((1,-60))]for _ in range(n)])]+[zero]*5;weights=[arb(1)]*n;error=arb_mat([[arb(0,arb((1,-10)))]for _ in range(n)]);h=F(1,32)
 _,loc=localize_polynomial_root(ds,ff,weights,error,-h,-h/2,h,tick=w.tick)
 _,tan=refine_polynomial_tangent(C,ds,[zero]*4,zero,zero,weights,-h,-h/2,h,tick=w.tick)
 need(loc['passed']and tan['passed'],'full705 cubic six-forcing/four-derivative profile');w.save('cubic-profile.json',dict(passed=True,dimension=n,forcing_coefficients=6,residual_derivative_coefficients=4,native_calls=w.calls,scope='Synthetic algebra only; complete native144 profile retained.'))
 del I,dense,ds,C,ff,error,loc,tan;gc.collect()


def derivative_domain(w,row,context,domains):
 start=w.calls;model,c,s,a,b,normal,origin,sc,h,weights,original,image,C,ds,ff,rp,E,metrics,domain=[context[k]for k in("model","c","s","a","b","normal","origin","sc","halfwidth","weights","original","image","C","ds","ff","rp","E","metrics","domain")]
 left,right=map(F,row['endpoints']);center=(left+right)/2;half=(right-left)/2;shift=center-U.M.point(ball_json(sc));lo,hi=shift-half,shift+half
 need(row in domains and -h<=lo<hi<=h,'fixed derivative domain inside exact original chart')
 error,wp=localize_polynomial_root(ds,ff,weights,original,lo,hi,h,known_image=image,tick=w.tick);ce,cp=localize_polynomial_root(ds,ff,weights,original,shift,shift,h,known_image=image,tick=w.tick)
 dd=arb(U.R.exact(shift),U.R.exact(half));d=U.R.exact(shift);predictor=H.predictor(c,s,a,b,dd);pp=H.tangent(s,a,b,dd);pc=H.predictor(c,s,a,b,d);xc=[pc[i]+ce[i,0]for i in range(705)];whole=[predictor[i]+error[i,0]for i in range(705)];segment=[predictor[i]+arb(0,abs(error[i,0]).upper())for i in range(705)]
 for values,name in[(xc,'center'),(whole,'whole'),(segment,'segment')]:P.exact_coverage(domain['whole_box'],values,'literal original full705 '+name+' domain')
 plus,minus=balanced_directions([error[i,0]for i in range(705)],pp,arb('1/256'));w.save('localization.json',dict(passed=True,dimension=705,center_refinement=cp,whole_refinement=wp,center_box=vector_json(xc),whole_box=vector_json(whole)));w.save('tangent-domain-inputs.json',dict(domain=row,segment=vector_json(segment),error=matrix_json(error),predictor_tangent=vector_json(pp),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(U.R.exact(center)),scale='1/256',original_offset=[str(lo),str(hi)]))
 cp=w.paths(model,segment,plus,normal,origin,U.R.exact(center),'target_plus');cm=w.paths(model,segment,minus,normal,origin,U.R.exact(center),'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);tangent,proof=refine_polynomial_tangent(C,ds,rp,mixed,arb_mat([[v]for v in pp]),weights,lo,hi,h,tick=w.tick);w.save('tangent.json.gz',proof)
 loc=dict(passed=True,dimension=705,weighted_neumann_bound=proof['weighted_neumann_bound'],center_box=vector_json(xc),whole_box=vector_json(whole),tangent_box=matrix_json(tangent));xc,whole,v,terminal,full=B.native_domains(loc);qi=dict(domain=row,full705=full,terminal=matrix_json(terminal),complements=[matrix_json(e)for e in E],metrics=[matrix_json(e)for e in metrics]);w.save('quotient-domain-inputs.json',qi)
 derivatives=[]
 for j in range(18):
  off=39*j;Aw,Ap=w.jac(model,whole[off:off+13],whole[-2:],v[off:off+13],v[-2:],'whole_tangent',j);derivatives.append(dict(phase=j,A=Aw,Aprime=Ap,call=w.calls,input=read(w.output_dir/w.prefix/f'native/{w.calls:04d}-input.json')))
 need(w.calls-start==90,'72polarized+18whole derivative phases');w.save('domain-outcome.json',dict(passed=True,domain=row,dimension=705,native_calls=90,first_call=start+1,last_call=w.calls));return qi,proof,derivatives


def fresh_cell(w,row,context,qi,tan,derivatives,domains):
 start=w.calls;model,c,s,a,b,normal,origin,sc,h,weights,original,image,C,ds,ff,rp,E,metrics,domain=[context[k]for k in("model","c","s","a","b","normal","origin","sc","halfwidth","weights","original","image","C","ds","ff","rp","E","metrics","domain")]
 need(row==H.cell(domains,row['domain'],row['depth'],row['index'])and qi['domain']==domains[row['domain']-1],'literal cell and actual derivative domain');center=F(row['center']);half=F(row['halfwidth']);shift=center-U.M.point(ball_json(sc))
 error,initial=localize_polynomial_root(ds,ff,weights,original,shift,shift,h,known_image=image,tick=w.tick);d=U.R.exact(shift);predictor=H.predictor(c,s,a,b,d);known=[predictor[i]+error[i,0]for i in range(705)];OLD.original_contains(domain['error'],predictor,known);point=[x.mid()for x in known];w.save('center-authority.json',dict(cell=row,known=vector_json(known),predictor=vector_json(predictor),original_error=domain['error'],refinement=initial,offset=str(shift)))
 for k in range(3):
  OLD.original_contains(domain['error'],predictor,point);G=w.residual(model,point,normal,origin,U.R.exact(center),f'target{k}')
  if k<2:point=finite_point_correction(C,G,point,shift,h);OLD.original_contains(domain['error'],predictor,point);w.save(f'finite-proposal-{k}.json',dict(point=vector_json(point)))
  else:localized,proof=localize_fresh_polynomial_point(C,ds,G,point,known,weights,shift,h,tick=w.tick)
 w.save('center-refinement.json',proof);OLD.original_contains(domain['error'],predictor,localized);P.exact_coverage(qi['full705']['whole'],localized,'actual derivative domain contains new same-root center')
 loc=dict(passed=True,dimension=705,weighted_neumann_bound=tan['weighted_neumann_bound'],center_box=vector_json(localized),whole_box=qi['full705']['whole'],tangent_box=[[v]for v in qi['full705']['tangent']]);xc,whole,v,terminal,full=B.native_domains(loc);w.save('quotient-domain-inputs.json',dict(cell=row,full705=full,halfwidth=str(half),parameter_endpoints=row['endpoints'],terminal=matrix_json(terminal),complements=qi['complements'],metrics=qi['metrics']))
 rows=[]
 for j,retained in enumerate(derivatives):
  need(retained['phase']==j,'literal ordered actual derivative phase');off=39*j
  for key,values in [('state',whole[off:off+13]),('rates',whole[-2:]),('state_tangent',v[off:off+13]),('rate_tangent',v[-2:])]:P.exact_coverage(retained['input'][key],values,'actual reused derivative '+key+' coverage')
  A0,_=w.jac(model,xc[off:off+13],xc[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);rows.append(dict(phase=j,center_A=A0,whole_A=retained['A'],whole_Aprime=retained['Aprime']))
 need(w.calls-start==126,'108fresh residual+18center exact');events=[0]
 def observe(kind,value):events[0]+=1;w.save(f'events/{events[0]:03d}-{kind}.json',value);w.tick()
 proof=correlated_complement(rows,xc,whole,v,E,metrics,U.R.exact(half),terminal,tick=w.tick,observe=observe);w.save('complement.json',proof);w.save('retained-derivatives.json',dict(domain=qi['domain'],calls=[v['call']for v in derivatives],scope='Actual unchanged whole state/rate/tangent derivatives of this half-quarter; same-root cell centers fresh.'))
 return dict(cell=row,passed=proof['passed'],completed=True,native_calls=126,dimension=705,complement_dimension=11,remainder=proof['remainder_norm_upper'],first_call=start+1,last_call=w.calls)


def target(w,card):
 w.prefix='context';context=G.bind(w,card);w.prefix='';domains=card['domains']
 for drow in domains:
  w.prefix=f'domains/{drow["index"]:03d}';qi,tan,derivatives=derivative_domain(w,drow,context,domains);queue=[H.cell(domains,drow['index'])]
  while queue:
   need(len(w.attempts)<30,'fixed30 center ceiling');row=queue.pop(0);ordinal=len(w.attempts)+1;w.prefix=f'cells/{ordinal:03d}';w.save('cell-input.json',row);out=fresh_cell(w,row,context,qi,tan,derivatives,domains);w.save('cell-outcome.json',out);w.attempts.append(out);w.prefix='';w.save(f'completed-attempt-{ordinal:03d}.json',out)
   if out['passed']:w.accepted.append(out)
   else:queue=H.children(domains,row,out)+queue
   gc.collect()
 union=H.complete(domains,w.accepted);need(w.calls==144+180+126*len(w.attempts)<=4104,'complete inclusive declared phase accounting');w.save('union.json',union)
 return dict(passed=True,status='cubic32_first_portion_complement_covered',union=union,scope='Producer full11 complementary contraction on exact first quarter of independently connected cubic32. Entire prior stable endpoint projection is included. New cells require independent replay; critical pair remains neutral, no complete chart32/window claim.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,3000)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<6*1024**3-8*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();synthetic(w);elapsed=time.monotonic()-begin;pm=read(S.Q.PILOT/'run/manifest.json');need(O.entry(S.Q.PILOT/'manifest.json')['sha256']==S.Q.PILOT_PINS[0]and O.entry(S.Q.PILOT/'run/manifest.json')['sha256']==S.Q.PILOT_PINS[1]and O.entry(S.Q.PILOT/'run/outcome.json')==pm['files']['outcome.json'],'literal measured pilot cost authority');pilot=read(S.Q.PILOT/'run/outcome.json');report=S.H.forecast(elapsed,w.used,pilot['elapsed_seconds'],sum(v['bytes']for v in pm['files'].values()),time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('batch-profile.json',report);need(report['passed'],'all30 centers+2domains measured admission before arrays');result=target(w,card);w.tick()
 except BaseException as e:result.update(passed=False,status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),completed_attempts=w.attempts,accepted_cells=w.accepted,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
