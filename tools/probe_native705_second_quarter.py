"""Same chart31 root, two fresh derivative domains and bounded center coverage."""
import gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_fresh_center_quarter as Q
import check_native705_quarter_overlap as OVERLAP
from neural_shrimp import fresh_center_followon as H
from neural_shrimp.rigorous_polynomial_localization import localize_polynomial_root,refine_polynomial_tangent
from neural_shrimp.rigorous_polynomial_point import finite_point_correction,localize_fresh_polynomial_point
from neural_shrimp.rigorous_unit_pair_pal import assemble,path_coefficients
from neural_shrimp.rigorous_quadratic_tangent_forcing import balanced_directions,mixed_hessian_from_second_coefficients
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
from flint import arb,arb_mat,ctx
OLD,P,O,D,U,B,need,read=Q.OLD,Q.P,Q.O,Q.D,Q.U,Q.B,Q.need,Q.read
OUT=Q.OUT.parent.with_name('second-quarter001')/'run';CARD='docs/plans/native705-second-quarter-inputs.json'
CAPS=dict(seconds=3000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=6*1024**3,profile_calls=144,maximum_target_calls=3960,maximum_total_calls=4104)
FILES=sorted(set(Q.FILES+['tools/reviews/check_native705_quarter_overlap.py','docs/reviews/native705-quarter-stable-overlap.json','src/neural_shrimp/fresh_center_followon.py','tests/test_fresh_center_followon.py','tools/probe_native705_second_quarter.py','tools/launch_native705_second_quarter.py','tests/test_native705_second_quarter.py','docs/plans/native705-second-quarter.md',CARD,'docs/reviews/native705-fresh-widths002.json']))
FIXED_FIRST=dict(root='/data2/neural-shrimp-data/audits/native705-fresh-quarter-independent/attempt001',source='9b218f7cfe5c8c6227d4fc20c78400a86daed50f',outer='acefd8944c1cc88f38f60532f1142672d19e6ff11a325eebd1eca059108b4809',worker='1efe71c70410fc3ab1436cc6777edcc2e112ea179af05f98c50fa99384879f13')
FIXED_WIDTH=dict(root=str(Q.OUT.parent.with_name('fresh-widths002')),source='be12d70a9d862aaf16c5774b95fdf4aabbcb4993',outer='a07d2444559898a1e09f76f6adaa230e2c2197023266bd175cae44975a2be07b',worker='2fd6677e54e2788cd6c0fb1e78f07cc2b1c3eef6b23e5eb1773c967a13e26586')


def card_check(card):
 need(card['status']=='reviewed'and card['domains']==H.domains()and card['parameter_endpoints']==['-17/2048','15/2048']and card['maximum_depth']==3 and card['maximum_centers']==30,'literal reviewed two-domain bounded coverage')
 need(card['caps']==CAPS and card['independent_first_quarter']==FIXED_FIRST and card['width_diagnosis']==FIXED_WIDTH,'fixed caps and completed prior authorities')


def authority(spec):
 root=Path(spec['root']);O.verify(root,spec['outer']);O.verify(root/'run',spec['worker']);rt=D.B.archived(root/'run',spec['source'],'runtime.json','source');own=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 need(own['source_commit']==spec['source']and own['exit_code']==0 and own['failure']is None and not own['owned_live_pids']and own['worker_pid_absent'],'complete prior owned execution')
 for pid in[rt['pid'],own['launcher_pid'],own['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual prior owner absent')
 need(out['passed']and out['native_calls']==0,'completed zero-native independent/diagnostic authority');return out


def bind_meta(w,card):
 first=authority(card['independent_first_quarter']);width=authority(card['width_diagnosis']);qcard=read(ROOT/Q.CARD)
 need(first['parameter_endpoints']==qcard['original_quarter']and first['total_cells']==17 and first['complement_dimension']==11,'entire original first-quarter independent proof')
 ib=read(Path(FIXED_FIRST['root'])/'run/authority-binding.json');need(ib['pilot']==qcard['pilot']and ib['prior']==qcard['independent_pilot']and ib['original_whole']['outer']==OLD.PARENT_PIN,'same native-root prior genealogy')
 need([r['halfwidth']for r in width['results']]==['1/2048','1/1024','1/512','1/256','1/128']and[r['passed']for r in width['results']]==[True,True,False,False,False],'all fixed width outcomes retained');overlap=OVERLAP.check();need(overlap==read(ROOT/'docs/reviews/native705-quarter-stable-overlap.json'),'literal source-bound exact stable-overlap check')
 need(overlap['second_quarter_contains_previous_endpoint']and overlap['second_quarter']==card['parameter_endpoints']and overlap['first_quarter']==first['parameter_endpoints']and overlap['quarters_exactly_adjacent'],'entire prior endpoint and first-quarter adjacency')
 w.save('authority-binding.json',dict(first=FIXED_FIRST,first_outcome=first,width=FIXED_WIDTH,width_outcome=width,overlap=overlap,prior_first_authority=ib));return overlap


class Work(Q.Work):
 output_dir=None;maximum_calls=4104;raw_cap=6*1024**3
 def __init__(self):super().__init__();self.output_dir=OUT if self.output_dir is None else self.output_dir;self.attempts=[];self.accepted=[]
 def tick(self):need(not self.stopped and time.monotonic()-self.start<3000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'3000s4GiB owned worker')
 def save(self,name,value,closing=False):
  if name=='profile.json':name='base-profile.json'
  if name=='profile-admission.json':name='single-profile-admission.json'
  if self.prefix:name=self.prefix+'/'+name
  raw=D.canonical_bytes(value);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw;need(self.used+len(raw)<self.raw_cap-(0 if closing else 8*1024**2),'declared raw cap with8MiB final reserve');p=self.output_dir/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def paths(self,model,flat,direction,normal,origin,sc,stage):
  start=self.calls;owner=self
  class Recorded:
   n=13
   def cusp_step(self,x,a,b):
    owner.tick();need(owner.calls<owner.maximum_calls and owner.calls-start<36,'bounded complete path group');owner.calls+=1;k=owner.calls;owner.save(f'native/{k:04d}-input.json',dict(kind='path',stage=stage,phase=(k-start-1)//2,direction='q'if(k-start)%2 else'r',inputs=[z.serialized()for z in x+[a,b]]));values=model.cusp_step(x,a,b);owner.save(f'native/{k:04d}-return.json',dict(kind='path',stage=stage,returns=[z.serialized()for z in values]));owner.tick();return values
  coeff=path_coefficients(Recorded(),flat,direction,[arb(0)]*705,arb(0),18,normal,origin,sc,self.tick,lambda *_:None);need(self.calls-start==36,'all actual polarized phases');self.save('coefficients/'+stage+'.json',matrix_json(coeff));return coeff
 def residual(self,model,point,normal,origin,parameter,stage):
  start=self.calls;owner=self
  def jet(z):return dict(f=ball_json(z.f),g=vector_json(z.g),d=ball_json(z.d),m=vector_json(z.m))
  class Recorded:
   n=13
   def cusp_step(self,x,a,b):
    owner.tick();need(owner.calls-start<36 and owner.calls<owner.maximum_calls,'bounded complete residual group');owner.calls+=1;k=owner.calls;owner.save(f'native/{k:04d}-input.json',dict(kind='residual',stage=stage,inputs=[jet(z)for z in x+[a,b]]));values=model.cusp_step(x,a,b);owner.save(f'native/{k:04d}-return.json',dict(kind='residual',stage=stage,returns=[jet(z)for z in values]));owner.tick();return values
  self.save('residual-inputs/'+stage+'.json',dict(point=vector_json(point),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(parameter)));G,J,_=assemble(Recorded(),point,18,normal,origin,parameter,self.tick,lambda *_:None);del J;need(self.calls-start==36,'complete full705 residual');self.save('residuals/'+stage+'.json',matrix_json(G));return G
 def jac(self,model,x,rates,v,vp,kind,phase):
  self.tick();need(self.calls<self.maximum_calls,'inclusive native cap');self.calls+=1;k=self.calls;self.save(f'native/{k:04d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));A,Ap=total_two_rate_jacobian(model,x,rates,v,vp);self.save(f'native/{k:04d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(A),Aprime=matrix_json(Ap)));self.tick();return A,Ap


def derivative_domain(w,row,context,quarter=2):
 start=w.calls;model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=context
 left,right=map(F,row['endpoints']);center=(left+right)/2;half=(right-left)/2;shift=center-U.M.point(ball_json(sc));lo,hi=shift-half,shift+half
 need(row in H.domains(quarter)and F(-1,32)<=lo<hi<=F(1,32),'fixed derivative domain inside exact original chart')
 error,wp=localize_polynomial_root(ds,ff,weights,original,lo,hi,F(1,32),known_image=image,tick=w.tick);ce,cp=localize_polynomial_root(ds,ff,weights,original,shift,shift,F(1,32),known_image=image,tick=w.tick)
 dd=arb(U.R.exact(shift),U.R.exact(half));d=U.R.exact(shift);predictor=[c[i]+s[i]*dd+a[i]*dd*dd/2 for i in range(705)];pp=[s[i]+a[i]*dd for i in range(705)];xc=[c[i]+s[i]*d+a[i]*d*d/2+ce[i,0]for i in range(705)];whole=[predictor[i]+error[i,0]for i in range(705)];segment=[predictor[i]+arb(0,abs(error[i,0]).upper())for i in range(705)]
 for values,name in[(xc,'center'),(whole,'whole'),(segment,'segment')]:P.exact_coverage(domain['whole_box'],values,'literal original full705 '+name+' domain')
 plus,minus=balanced_directions([error[i,0]for i in range(705)],pp,arb('1/256'));w.save('localization.json',dict(passed=True,dimension=705,center_refinement=cp,whole_refinement=wp,center_box=vector_json(xc),whole_box=vector_json(whole)));w.save('tangent-domain-inputs.json',dict(domain=row,segment=vector_json(segment),error=matrix_json(error),predictor_tangent=vector_json(pp),plus=vector_json(plus),minus=vector_json(minus),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(U.R.exact(center)),scale='1/256',original_offset=[str(lo),str(hi)]))
 cp=w.paths(model,segment,plus,normal,origin,U.R.exact(center),'target_plus');cm=w.paths(model,segment,minus,normal,origin,U.R.exact(center),'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);tangent,proof=refine_polynomial_tangent(C,ds,rp,mixed,arb_mat([[v]for v in pp]),weights,lo,hi,F(1,32),tick=w.tick);w.save('tangent.json.gz',proof)
 loc=dict(passed=True,dimension=705,weighted_neumann_bound=proof['weighted_neumann_bound'],center_box=vector_json(xc),whole_box=vector_json(whole),tangent_box=matrix_json(tangent));xc,whole,v,terminal,full=B.native_domains(loc);qi=dict(domain=row,full705=full,terminal=matrix_json(terminal),complements=[matrix_json(e)for e in E],metrics=[matrix_json(e)for e in metrics]);w.save('quotient-domain-inputs.json',qi)
 derivatives=[]
 for j in range(18):
  off=39*j;Aw,Ap=w.jac(model,whole[off:off+13],whole[-2:],v[off:off+13],v[-2:],'whole_tangent',j);derivatives.append(dict(phase=j,A=Aw,Aprime=Ap,call=w.calls,input=read(w.output_dir/w.prefix/f'native/{w.calls:04d}-input.json')))
 need(w.calls-start==90,'72polarized+18whole derivative phases');w.save('domain-outcome.json',dict(passed=True,domain=row,dimension=705,native_calls=90,first_call=start+1,last_call=w.calls));return qi,proof,derivatives


def fresh_cell(w,row,context,qi,tan,derivatives,quarter=2):
 start=w.calls;model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=context
 need(row==H.cell(row['domain'],row['depth'],row['index'],quarter)and qi['domain']==H.domains(quarter)[row['domain']-1],'literal cell and actual derivative domain');center=F(row['center']);half=F(row['halfwidth']);shift=center-U.M.point(ball_json(sc))
 error,initial=localize_polynomial_root(ds,ff,weights,original,shift,shift,F(1,32),known_image=image,tick=w.tick);d=U.R.exact(shift);predictor=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(705)];known=[predictor[i]+error[i,0]for i in range(705)];OLD.original_contains(domain['error'],predictor,known);point=[x.mid()for x in known];w.save('center-authority.json',dict(cell=row,known=vector_json(known),predictor=vector_json(predictor),original_error=domain['error'],refinement=initial,offset=str(shift)))
 for k in range(3):
  OLD.original_contains(domain['error'],predictor,point);G=w.residual(model,point,normal,origin,U.R.exact(center),f'target{k}')
  if k<2:point=finite_point_correction(C,G,point,shift,F(1,32));OLD.original_contains(domain['error'],predictor,point);w.save(f'finite-proposal-{k}.json',dict(point=vector_json(point)))
  else:localized,proof=localize_fresh_polynomial_point(C,ds,G,point,known,weights,shift,F(1,32),tick=w.tick)
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
 overlap=bind_meta(w,card);w.prefix='context';context=P.bind(w);w.prefix='';w.save('scope-binding.json',dict(domains=card['domains'],parameter_endpoints=card['parameter_endpoints'],ancestry='context/input-binding.json is unchanged original fullchart authority; its first-quarter scope describes its historical caller only.',overlap=overlap))
 for drow in card['domains']:
  w.prefix=f'domains/{drow["index"]:03d}';qi,tan,derivatives=derivative_domain(w,drow,context);queue=[H.cell(drow['index'])]
  while queue:
   need(len(w.attempts)<30,'fixed30 center ceiling');row=queue.pop(0);ordinal=len(w.attempts)+1;w.prefix=f'cells/{ordinal:03d}';w.save('cell-input.json',row);out=fresh_cell(w,row,context,qi,tan,derivatives);w.save('cell-outcome.json',out);w.attempts.append(out);w.prefix='';w.save(f'completed-attempt-{ordinal:03d}.json',out)
   if out['passed']:w.accepted.append(out)
   else:queue=H.children(row,out)+queue
   gc.collect()
 union=H.complete(w.accepted);need(w.calls==144+180+126*len(w.attempts)<=4104,'complete inclusive declared phase accounting');w.save('union.json',union);return dict(passed=True,status='second_quarter_complement_covered',union=union,scope='Producer full11 complementary contraction on exact chart31 second quarter, touching prior independent firstquarter and containing whole chart30 known-root projection. New cells require independent replay. Neutral critical pair retained; no wholechart/window claim.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,3000)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<6*1024**3-8*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();OLD.synthetic(w);elapsed=time.monotonic()-begin;pm=read(Q.PILOT/'run/manifest.json');need(O.entry(Q.PILOT/'manifest.json')['sha256']==Q.PILOT_PINS[0]and O.entry(Q.PILOT/'run/manifest.json')['sha256']==Q.PILOT_PINS[1]and O.entry(Q.PILOT/'run/outcome.json')==pm['files']['outcome.json'],'literal measured pilot cost authority');pilot=read(Q.PILOT/'run/outcome.json');report=H.forecast(elapsed,w.used,pilot['elapsed_seconds'],sum(v['bytes']for v in pm['files'].values()),time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('batch-profile.json',report);need(report['passed'],'all30 centers+2domains measured admission before arrays');result=target(w,card)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),completed_attempts=w.attempts,accepted_cells=w.accepted,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
