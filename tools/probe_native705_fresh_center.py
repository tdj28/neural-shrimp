"""One centered subinterval using a freshly localized identical root center."""
import gc,gzip,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_polynomial_complement as P
from neural_shrimp.rigorous_polynomial_point import finite_point_correction,localize_fresh_polynomial_point
from neural_shrimp.rigorous_unit_pair_pal import assemble
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from flint import arb,arb_mat,ctx
O,D,U,B,need,read=P.O,P.D,P.U,P.B,P.need,P.read
PARENT=P.OUT.parent;PARENT_PIN='fd297b95b22708c5c33b0284fca27f55c095a2fd2d3d0cfc223c42a9e840e031';PARENT_EXEC='2d932da23fddf602a90ee747da88dce9085a5ba9'
DIAG=PARENT.with_name('polynomial001-remainder-diagnosis');DIAG_PIN='1f770b3f829f900cd5b03da3351fd715e3072a7c341ed4dd7db29baf64994628'
OUT=PARENT.with_name('fresh-center001')/'run';HALF=F(1,2048);ENDS=[F(-34,2048),F(-32,2048)]
FILES=sorted(set(P.FILES+['src/neural_shrimp/rigorous_polynomial_point.py','tests/test_polynomial_point.py','tools/probe_native705_fresh_center.py','tools/launch_native705_fresh_center.py','tests/test_native705_fresh_center.py','docs/plans/native705-fresh-center.md','docs/reviews/native705-remainder-diagnosis.json']))


def original_contains(error,predictor,box):
 need(len(error)==len(predictor)==len(box)==705,'complete original root correction domain')
 for old,p,x in zip(error,predictor,box,strict=True):
  lo,hi=U.M.interval(old);pl,ph=U.M.interval(ball_json(p));xl,xh=U.M.interval(ball_json(x));need(pl==ph and pl+lo<=xl<=xh<=pl+hi,'entire actual point/known box inside literal original p+E')


def retained_whole(a,v,j,whole,tangent):
 need(a['kind']==v['kind']=='whole_tangent'and a['phase']==v['phase']==j,'same retained actual phase')
 k=39*j
 for field,value in [('state',whole[k:k+13]),('rates',whole[-2:]),('state_tangent',tangent[k:k+13]),('rate_tangent',tangent[-2:])]:P.exact_coverage(a[field],value,'retained native actual '+field+' domain')
 A=U.R.matrix(v['A']);Ap=U.R.matrix(v['Aprime']);need(A.nrows()==A.ncols()==Ap.nrows()==Ap.ncols()==13 and all(x.is_finite()for m in(A,Ap)for row in m.tolist()for x in row),'complete finite retained total derivative');return A,Ap


class Work(P.Work):
 def save(self,name,v,closing=False):
  if name=='profile.json':name='base-profile.json'
  raw=D.canonical_bytes(v);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw;need(self.used+len(raw)<(384 if closing else 380)*1024**2,'384MiB4MiB closing reserve');q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def residual(self,model,point,normal,origin,parameter,stage):
  start=self.calls;need(start in(108,144,180,216),'fixed four36-phase residual groups');owner=self
  def jet(z):return dict(f=ball_json(z.f),g=vector_json(z.g),d=ball_json(z.d),m=vector_json(z.m))
  class Recorded:
   n=13
   def cusp_step(self,x,a,b):
    owner.tick();need(owner.calls-start<36 and owner.calls<252,'fixed residual cap');owner.calls+=1;k=owner.calls;owner.save(f'native/{k:03d}-input.json',dict(kind='residual',stage=stage,inputs=[jet(z)for z in x+[a,b]]));ret=model.cusp_step(x,a,b);owner.save(f'native/{k:03d}-return.json',dict(kind='residual',stage=stage,returns=[jet(z)for z in ret]));owner.tick();return ret
  self.save('residual-inputs/'+stage+'.json',dict(point=vector_json(point),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(parameter)))
  out,J,_=assemble(Recorded(),point,18,normal,origin,parameter,self.tick,lambda kind,row:None);del J;need(self.calls-start==36,'complete full705 residual');self.save('residuals/'+stage+'.json',matrix_json(out));return out
 def jac(self,model,x,rates,v,vp,kind,phase):
  self.tick();need(72<=self.calls<108 or 252<=self.calls<270,'fixed36profile18target Jacobians');self.calls+=1;self.save(f'native/{self.calls:03d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));A,Ap=total_two_rate_jacobian(model,x,rates,v,vp);self.save(f'native/{self.calls:03d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(A),Aprime=matrix_json(Ap)));self.tick();return A,Ap


def synthetic(w):
 begin=time.monotonic();P.synthetic(w);model,flat,plus,minus,pprime,normal,origin,sc,inp=P.T.synthetic_inputs();w.save('profile-residual-map.json',inp);w.residual(model,[v.mid()for v in flat],normal,origin,sc,'profile');elapsed=time.monotonic()-begin
 report=dict(passed=w.calls==144 and 4*elapsed+2*61.466+30<400 and 4*w.used+96*1024**2<384*1024**2,seconds=elapsed,forecast_seconds=4*elapsed+2*61.466+30,forecast_bytes=4*w.used+96*1024**2,profile_calls=w.calls,target_call_cap=126,target_arrays_unopened=True);w.save('profile-admission.json',report);need(report['passed'],'measured full705/144phase profile admission')


def target(w):
 O.verify(PARENT,PARENT_PIN);rt=D.B.archived(PARENT/'run',PARENT_EXEC,'runtime.json','source');old=read(PARENT/'run/outcome.json');la=read(PARENT/'launcher-result.json');need(not old['passed']and old['status']=='completed_correlated_bound_unresolved'and old['native_calls']==216 and old['target_native_calls']==108 and la['exit_code']==2 and not la['owned_live_pids'],'complete original metric failure')
 for n,h in rt['sources'].items():need(O.entry(ROOT/n)==h,'unchanged complete original acquisition/proof source '+n)
 O.verify(DIAG,DIAG_PIN);do=read(DIAG/'run/outcome.json');need(do['passed']and do['native_calls']==0 and len(do['rows'])==7 and not next(r for r in do['rows']if r['name']=='center_error_floor')['metric_passed'],'complete measured center-floor diagnosis')
 model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=P.bind(w);del original,image,ff,rp;gc.collect()
 oldloc=read(PARENT/'run/localization.json');tan=read(PARENT/'run/tangent.json.gz');qi=read(PARENT/'run/quotient-domain-inputs.json');need(oldloc['passed']and oldloc['dimension']==705 and tan['passed']and tan['dimension']==705 and qi['parameter_endpoints']==[str(P.LEFT),str(P.RIGHT)]and qi['halfwidth']==str(P.HALF),'same completed known-root/tangent quarter')
 need(qi['complements']==[matrix_json(e)for e in E]and qi['metrics']==[matrix_json(e)for e in metrics],'literal fixed original quotient frames')
 known=[U.R.original_ball(x)for x in oldloc['center_box']];shift=U.R.exact(P.SHIFT);predictor=[c[i]+s[i]*shift+a[i]*shift*shift/2 for i in range(705)];need(all(x.is_exact()for x in predictor),'exact original center predictor');original_contains(domain['error'],predictor,known)
 point=[x.mid()for x in known];parameter=sc+shift;need(U.M.point(ball_json(parameter))==F(-33,2048)and ENDS==[F(-33,2048)-HALF,F(-33,2048)+HALF]and HALF==P.HALF/16,'fixed centered one16th subinterval')
 w.save('fresh-center-inputs.json',dict(map_inputs=read(Path(P.M.select(read(ROOT/P.P.CARD),31)['producer']['run'])/'inputs.json')['map_inputs'],known=vector_json(known),predictor=vector_json(predictor),original_error=domain['error'],offset=str(P.SHIFT),parameter=str(F(-33,2048)),halfwidth=str(HALF),parameter_endpoints=[str(x)for x in ENDS],original_outer=PARENT_PIN,diagnosis_outer=DIAG_PIN))
 for j in range(3):
  original_contains(domain['error'],predictor,point);G=w.residual(model,point,normal,origin,parameter,f'target{j}')
  if j<2:
   point=finite_point_correction(C,G,point,P.SHIFT,F(1,32));original_contains(domain['error'],predictor,point);w.save(f'finite-proposal-{j}.json',dict(point=vector_json(point),scope='Finite old-C proposal only; same original domain checked.'))
  else:localized,proof=localize_fresh_polynomial_point(C,ds,G,point,known,weights,P.SHIFT,F(1,32),tick=w.tick)
 w.save('center-refinement.json',proof);del C,ds;gc.collect();original_contains(domain['error'],predictor,localized)
 loc=dict(passed=True,dimension=705,weighted_neumann_bound=tan['weighted_neumann_bound'],center_box=vector_json(localized),whole_box=qi['full705']['whole'],tangent_box=[[v]for v in qi['full705']['tangent']]);center,whole,v,terminal,full=B.native_domains(loc)
 P.exact_coverage(qi['full705']['whole'],localized,'same inherited whole root domain covers new center');w.save('quotient-domain-inputs.json',dict(full705=full,whole_parent=PARENT_PIN,halfwidth=str(HALF),parameter_endpoints=[str(x)for x in ENDS],complements=qi['complements'],metrics=qi['metrics'],terminal=matrix_json(terminal)))
 rows=[];retained={}
 for j in range(18):
  k=182+2*j;ip=PARENT/f'run/native/{k:03d}-input.json';op=PARENT/f'run/native/{k:03d}-return.json';Aw,Ap=retained_whole(read(ip),read(op),j,whole,v);retained[str(k)]=dict(input=O.entry(ip),output=O.entry(op));off=39*j;A0,_=w.jac(model,center[off:off+13],center[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);rows.append(dict(phase=j,center_A=A0,whole_A=Aw,whole_Aprime=Ap))
 need(w.calls==270,'144profile+126target complete');events=[0]
 def observe(kind,row):events[0]+=1;w.save(f'events/{events[0]:03d}-{kind}.json',row);w.tick()
 proof=correlated_complement(rows,center,whole,v,E,metrics,U.R.exact(HALF),terminal,tick=w.tick,observe=observe);w.save('complement.json',proof);w.save('retained-derivatives.json',dict(parent=PARENT_PIN,rows=retained,scope='Entire original quarter true derivative is reused only on a strict centered subinterval; no new whole-domain derivatives or old center reused.'))
 return dict(passed=proof['passed'],status='centered_subinterval_complement_attracting'if proof['passed']else'completed_centered_bound_unresolved',parameter_endpoints=[str(x)for x in ENDS],original_quarter=[str(P.LEFT),str(P.RIGHT)],subinterval_only=True,remainder=proof['remainder_norm_upper'],metric_passed=proof['metric']['passed'],scope=proof['scope'])


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,400)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,native_call_cap=270))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact own committed source');need(w.used+len(b)<380*1024**2,'bounded source archive');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  with ctx.workprec(480):synthetic(w);result=target(w)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(q.relative_to(OUT)):O.entry(q)for q in OUT.rglob('*')if q.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
