"""Fixed sixteen new centers plus one prior cell; exact first-quarter cover."""
import gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_fresh_center as OLD
from neural_shrimp import fresh_center_quarter as Q
from neural_shrimp.rigorous_polynomial_localization import localize_polynomial_root
from neural_shrimp.rigorous_polynomial_point import finite_point_correction,localize_fresh_polynomial_point
from neural_shrimp.rigorous_unit_pair_pal import assemble
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json
from flint import arb,arb_mat,ctx
P,O,D,U,B,need,read=OLD.P,OLD.O,OLD.D,OLD.U,OLD.B,OLD.need,OLD.read
PILOT=OLD.OUT.parent;PILOT_EXEC='f7db2cf7d9cac864b5f62e1522befc49fe50946f';PILOT_PINS=['08b6319ba099074535ac60f071dc62013cb76fbb7781a78c75632f867d91edc1','c4ffa9c7b92f170c0ce77b2e820455034c505163a7f3653de1bf98f0138e00ce']
OUT=PILOT.with_name('fresh-quarter001')/'run';CARD='docs/plans/native705-fresh-center-quarter-inputs.json'
FILES=sorted(set(OLD.FILES+['src/neural_shrimp/fresh_center_quarter.py','tests/test_fresh_center_quarter.py','tools/probe_native705_fresh_center_quarter.py','tools/launch_native705_fresh_center_quarter.py','tests/test_native705_fresh_center_quarter.py','docs/plans/native705-fresh-center-quarter.md',CARD,'docs/reviews/native705-fresh-center-attempt001.json','docs/reviews/native705-fresh-center-audit-guide.md']))


def card_check(card):
 need(card['status']=='reviewed'and card['cells']==Q.partition()and card['original_quarter']==['-49/2048','-17/2048']and card['reused_cell']==['-17/1024','-1/64'],'fixed exact reviewed17-cell cover')
 need(card['pilot']==dict(root=str(PILOT),source=PILOT_EXEC,outer=PILOT_PINS[0],worker=PILOT_PINS[1]),'literal same method pilot')
 a=card['independent_pilot'];need(isinstance(a,dict)and set(a)=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in a.values()),'completed independently replayed pilot required')
 need(card['caps']==dict(seconds=1500,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=3*1024**3,profile_calls=144,target_calls=2016,total_calls=2160),'unchanged complete batch caps')


def independent(w,card):
 a=card['independent_pilot'];root=Path(a['root']);need(root.is_absolute()and str(root).startswith('/data2/neural-shrimp-data/audits/'),'literal local independent evidence root')
 O.verify(root,a['outer']);O.verify(root/'run',a['worker']);rt=read(root/'run/runtime.json');own=read(root/'launcher-result.json');out=read(root/'run/outcome.json');ib=read(root/'run/source-binding.json')
 need(rt['source_commit']==own['source_commit']==a['source']and own['exit_code']==0 and own['failure']is None and not own['owned_live_pids']and own['worker_pid_absent'],'complete independent owned execution')
 for pid in[rt['pid'],own['launcher_pid'],own['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual independent owner absent')
 actual={str(p.relative_to(root/'run/source'))for p in(root/'run/source').rglob('*')if p.is_file()};need(actual==set(rt['sources']),'complete independent source archive')
 for n,pin in rt['sources'].items():need(O.entry(root/'run/source'/n)==pin and hashlib.sha256(subprocess.check_output(['git','show',a['source']+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'exact independent committed source')
 need(out['passed']and out['native_calls']==0 and out['parameter_endpoints']==card['reused_cell']and out['complement_dimension']==11,'full11 independent central-cell proof')
 need(ib['pilot']==card['pilot'],'exact independently replayed original pilot')
 w.save('independent-pilot-binding.json',dict(spec=a,outcome=out,input_binding=ib));return dict(passed=True,endpoints=card['reused_cell'])


class Work(OLD.Work):
 def __init__(self):super().__init__();self.prefix='';self.cell_start=None;self.completed=[]
 def tick(self):need(not self.stopped and time.monotonic()-self.start<1500 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'1500s4GiB batch cap')
 def save(self,name,value,closing=False):
  if name=='profile.json':name='base-profile.json'
  if name=='profile-admission.json':name='single-profile-admission.json'
  if self.prefix:name=self.prefix+'/'+name
  raw=D.canonical_bytes(value);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw;need(self.used+len(raw)<3*1024**3-(0 if closing else 8*1024**2),'3GiB8MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def residual(self,model,point,normal,origin,parameter,stage):
  start=self.calls;allowed=(108,)if self.cell_start is None else tuple(self.cell_start+36*k for k in range(3));need(start in allowed,'exact three residual groups per fresh cell');owner=self
  def jet(z):return dict(f=ball_json(z.f),g=vector_json(z.g),d=ball_json(z.d),m=vector_json(z.m))
  class Recorded:
   n=13
   def cusp_step(self,x,a,b):
    owner.tick();need(owner.calls-start<36 and owner.calls<2160,'complete bounded36-phase group');owner.calls+=1;k=owner.calls;owner.save(f'native/{k:04d}-input.json',dict(kind='residual',stage=stage,inputs=[jet(z)for z in x+[a,b]]));out=model.cusp_step(x,a,b);owner.save(f'native/{k:04d}-return.json',dict(kind='residual',stage=stage,returns=[jet(z)for z in out]));owner.tick();return out
  self.save('residual-inputs/'+stage+'.json',dict(point=vector_json(point),normal=vector_json(normal),origin=vector_json(origin),parameter=ball_json(parameter)));res,J,_=assemble(Recorded(),point,18,normal,origin,parameter,self.tick,lambda *_:None);del J;need(self.calls-start==36,'entire original full705 residual');self.save('residuals/'+stage+'.json',matrix_json(res));return res
 def jac(self,model,x,rates,v,vp,kind,phase):
  allowed=72<=self.calls<108 if self.cell_start is None else self.cell_start+108<=self.calls<self.cell_start+126
  need(allowed and self.calls<2160,'complete center group and inclusive cap');self.tick();self.calls+=1;k=self.calls;self.save(f'native/{k:04d}-input.json',dict(kind=kind,phase=phase,state=vector_json(x),rates=vector_json(rates),state_tangent=vector_json(v),rate_tangent=vector_json(vp)));A,Ap=total_two_rate_jacobian(model,x,rates,v,vp);self.save(f'native/{k:04d}-return.json',dict(kind=kind,phase=phase,A=matrix_json(A),Aprime=matrix_json(Ap)));self.tick();return A,Ap


def bind(w,card):
 O.verify(PILOT,PILOT_PINS[0]);O.verify(PILOT/'run',PILOT_PINS[1]);rt=D.B.archived(PILOT/'run',PILOT_EXEC,'runtime.json','source');pilot=read(PILOT/'run/outcome.json');need(pilot['passed']and pilot['native_calls']==270 and pilot['parameter_endpoints']==card['reused_cell'],'complete actual original method pilot')
 for n,h in rt['sources'].items():need(O.entry(ROOT/n)==h,'byte-identical original pilot dependency '+n)
 reused=independent(w,card)
 # Original failed whole quarter remains the literal full tangent authority.
 O.verify(OLD.PARENT,OLD.PARENT_PIN);prt=D.B.archived(OLD.PARENT/'run',OLD.PARENT_EXEC,'runtime.json','source');po=read(OLD.PARENT/'run/outcome.json');need(not po['passed']and po['status']=='completed_correlated_bound_unresolved'and po['native_calls']==216,'retain original completed negative metric')
 for n,h in prt['sources'].items():need(O.entry(ROOT/n)==h,'same original whole-quarter derivative source '+n)
 context=P.bind(w);qi=read(OLD.PARENT/'run/quotient-domain-inputs.json');tan=read(OLD.PARENT/'run/tangent.json.gz');need(qi['parameter_endpoints']==card['original_quarter']and tan['passed']and tan['dimension']==705,'complete original same-root true tangent')
 E,metrics=context[-3:-1];need(qi['complements']==[matrix_json(e)for e in E]and qi['metrics']==[matrix_json(e)for e in metrics],'unchanged exact fixed full11 frames')
 return context,qi,tan,reused,pilot


def cell(w,row,context,qi,tan):
 model,c,s,a,normal,origin,sc,weights,original,image,C,ds,ff,rp,E,metrics,domain=context
 center=F(row['center']);half=F(row['halfwidth']);shift=center-U.M.point(ball_json(sc));ends=[F(x)for x in row['endpoints']]
 need(ends==[center-half,center+half]and P.LEFT<=ends[0]<ends[1]<=P.RIGHT and half<=F(1,2048),'fixed native first-quarter subset')
 error,initial=localize_polynomial_root(ds,ff,weights,original,shift,shift,F(1,32),known_image=image,tick=w.tick);d=U.R.exact(shift);predictor=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(705)];known=[predictor[i]+error[i,0]for i in range(705)]
 OLD.original_contains(domain['error'],predictor,known);point=[v.mid()for v in known];w.save('center-authority.json',dict(cell=row,known=vector_json(known),predictor=vector_json(predictor),original_error=domain['error'],refinement=initial,offset=str(shift)))
 for k in range(3):
  OLD.original_contains(domain['error'],predictor,point);G=w.residual(model,point,normal,origin,U.R.exact(center),f'target{k}')
  if k<2:point=finite_point_correction(C,G,point,shift,F(1,32));OLD.original_contains(domain['error'],predictor,point);w.save(f'finite-proposal-{k}.json',dict(point=vector_json(point)))
  else:localized,proof=localize_fresh_polynomial_point(C,ds,G,point,known,weights,shift,F(1,32),tick=w.tick)
 w.save('center-refinement.json',proof);OLD.original_contains(domain['error'],predictor,localized);P.exact_coverage(qi['full705']['whole'],localized,'original whole root domain covers every new center')
 loc=dict(passed=True,dimension=705,weighted_neumann_bound=tan['weighted_neumann_bound'],center_box=vector_json(localized),whole_box=qi['full705']['whole'],tangent_box=[[v]for v in qi['full705']['tangent']]);xc,whole,v,terminal,full=B.native_domains(loc)
 w.save('quotient-domain-inputs.json',dict(cell=row,full705=full,halfwidth=str(half),parameter_endpoints=row['endpoints'],terminal=matrix_json(terminal),complements=qi['complements'],metrics=qi['metrics']))
 rows=[];retained={}
 for j in range(18):
  k=182+2*j;ip=OLD.PARENT/f'run/native/{k:03d}-input.json';op=OLD.PARENT/f'run/native/{k:03d}-return.json';Aw,Ap=OLD.retained_whole(read(ip),read(op),j,whole,v);retained[str(k)]=dict(input=O.entry(ip),output=O.entry(op));off=39*j;A0,_=w.jac(model,xc[off:off+13],xc[-2:],[arb(0)]*13,[arb(0)]*2,'center',j);rows.append(dict(phase=j,center_A=A0,whole_A=Aw,whole_Aprime=Ap))
 need(w.calls-w.cell_start==126,'all126 native phases per selected cell');count=[0]
 def observe(kind,value):count[0]+=1;w.save(f'events/{count[0]:03d}-{kind}.json',value);w.tick()
 proof=correlated_complement(rows,xc,whole,v,E,metrics,U.R.exact(half),terminal,tick=w.tick,observe=observe);w.save('complement.json',proof);w.save('retained-derivatives.json',dict(parent=OLD.PARENT_PIN,rows=retained))
 return dict(index=row['index'],endpoints=row['endpoints'],passed=proof['passed'],dimension=705,complement_dimension=11,native_calls=126,remainder=proof['remainder_norm_upper'])


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,1500)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),caps=card['caps']))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(b)<3*1024**3-8*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();OLD.synthetic(w);elapsed=time.monotonic()-before;pm=read(PILOT/'run/manifest.json');need(O.entry(PILOT/'manifest.json')['sha256']==PILOT_PINS[0]and O.entry(PILOT/'run/manifest.json')['sha256']==PILOT_PINS[1]and O.entry(PILOT/'run/outcome.json')==pm['files']['outcome.json'],'literal full method cost manifests/outcome');pilot=read(PILOT/'run/outcome.json');oldbytes=sum(v['bytes']for v in pm['files'].values());profile=Q.forecast(elapsed,w.used,pilot['elapsed_seconds'],oldbytes,time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('batch-profile.json',profile);need(profile['passed'],'complete16-cell measured admission before target arrays')
   context,qi,tan,reused,_=bind(w,card)
   for row in card['cells']:
    w.prefix=f'cells/{row["index"]:03d}';w.cell_start=w.calls;w.save('cell-input.json',row);out=cell(w,row,context,qi,tan);w.save('cell-outcome.json',out);w.completed.append(out);w.prefix='';w.save(f'completed-prefix-{len(w.completed):02d}.json',dict(rows=w.completed));need(out['passed'],'fixed cell metric unresolved; no automatic subdivision');gc.collect()
   union=Q.complete_union(w.completed,reused);need(w.calls==2160,'144profile+2016target exact');w.save('union.json',union);result.update(passed=True,status='first_quarter_complement_covered',union=union,scope='Producer full11 complementary contraction on exact original chart31 first-quarter closed union; central prior cell independently proved, sixteen new cells need independent replay. Neutral critical pair retained; no wholechart/window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),completed_cells=w.completed,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
