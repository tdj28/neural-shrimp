"""One output-rate rechart with fresh conditional refinement of a known root."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import march_native470_correlated as M
C=M.C
from neural_shrimp.rigorous_signed_endpoint_refinement import refine,fixed_parameter_domain
from neural_shrimp.rigorous_eigenpair_locus import assemble,exact_mid
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,ball_json,canonical_bytes
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
need,read,entry=C.need,C.read,C.entry
OUT=M.OUT.parents[2]/'signed-flip-output-gauge001/attempt001/run';CARD='docs/plans/native470-output-gauge-inputs.json';RECEIPT='docs/reviews/native470-correlated-march001.json'
SETTINGS=dict(period=18,sigma=-1,precision=256,newton_steps=2,halfwidth_exponent=-15,radius_exponent=-12,refinement_evaluations=3,old_index=22,old_parameter='-2001/256')
LIMITS=dict(seconds=300,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=1024**3,native_calls=432,ticks=450)
FILES=sorted(set(M.FILES+[CARD,RECEIPT,'docs/plans/native470-output-gauge.md','tools/probe_native470_output_gauge.py','tools/launch_native470_output_gauge.py','tests/test_native470_output_gauge.py','src/neural_shrimp/rigorous_signed_endpoint_refinement.py','tests/test_signed_endpoint_refinement.py']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.ticks=0;self.native=0;self.serial=0;self.stage='profile'
 def check(self):need(time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s/4GiB complete output-gauge packet')
 def tick(self):self.ticks+=1;need(self.ticks<=450,'243 profile+207 target inclusive ticks');self.check()
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);data=gzip.compress(raw,mtime=0)if n.endswith('.gz')else raw;need(self.used+len(data)<1024**3-(0 if closing else 8*1024**2),'1GiB with8MiB closing reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  self.used+=len(data)
  if not closing:self.check()
 def observe(self,k,v):
  if k=='native_input':self.native+=1;need(self.native<=432,'234 profile+198 target native phases')
  if k=='signed_correlated_chart_return':self.save(self.stage+'-certificate.json.gz',v)
  else:self.save(f'events/{self.serial:04d}-{self.stage}-{k}.json.gz',v);self.serial+=1

def old_polynomial(chart,parameter):
 rows=chart['inverse_polynomial_exact'];need(len(rows)==2 and all(len(a)==470 and all(len(row)==470 for row in a)for a in rows),'complete actual certified inverse polynomial')
 matrices=[arb_mat([[arb(tuple(v))for v in row]for row in a])for a in rows];d=exact(str(F(parameter)-F(chart['parameter_center'])));return matrices[0]+matrices[1]*d

def bind(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS and card['producer_receipt']==entry(ROOT/RECEIPT),'fixed reviewed output-gauge packet')
 spec=read(ROOT/RECEIPT);p,rt=C.history(w,spec,2);o=read(p/'run/outcome.json');ib=read(p/'run/input-binding.json');mc=read(ROOT/M.CARD)
 need(spec['source']=='a34129fdf71b15aac2b539d8e55ba6a8896c5dfb'and o==spec['outcome']and not o['passed']and o['connected']and o['reason']=='association_failed'and o['native_calls']==3492 and o['ticks']==3636,'literal retained complete23attempt march')
 need(ib['card']==mc and rt['policy']==M.POLICY and rt['limits']==M.LIMITS,'complete original execution policy');M.binding(M.Prefix(w,'march-ancestry-'),mc,True)
 for n,e in ib['native_source'].items():need(rt['sources'][n]==entry(ROOT/n)==e,'same actual native model '+n)
 for n in ['src/neural_shrimp/rigorous_signed_correlated.py','src/neural_shrimp/rigorous_signed_gradient_path.py','src/neural_shrimp/rigorous_signed_proposal.py','src/neural_shrimp/rigorous_polynomial_inverse.py']:need(rt['sources'][n]==entry(ROOT/n),'unchanged certified kernel '+n)
 dirs=sorted((p/'run').glob('attempt*'));need([d.name for d in dirs]==[f'attempt{i:03d}'for i in range(1,24)],'all23 attempts retained');known=ib['known'];accepted=[]
 for d,row in zip(dirs,o['attempts'],strict=True):
  inp=read(d/'inputs.json');actual=read(d/'outcome.json');need(actual==row and inp['known']==known and inp['source_card']==rt['sources'][M.CARD]and inp['model']==ib['model']and inp['weights']==ib['weights']and inp['normal']==known['normal']and inp['origin']==known['origin'],'literal whole predecessor/source/model/gauge for every attempt')
  need(entry(d/'target-certificate.json.gz')==row['certificate']and row['native_calls']==144 and row['ticks']==150,'all complete attempted acquired proofs')
  if row['decision']['action']=='advance':
   need(row['passed']and all(row['gates'].values())and row['strict_inclusions']==row['anchor_inclusions']==470,'only proved joined root advances');known=read(d/'connected-left-endpoint.json');accepted.append(row['index'])
 need(accepted==o['accepted_indices']==[1,2,3,4,6,7,9,10,12,13,15,16,18,19,21,22]and known==read(p/'run/connected-left-endpoint.json')and entry(p/'run/connected-left-endpoint.json')==spec['endpoint'],'complete earned16prefix and exact last root')
 need(known['exact_parameter']==SETTINGS['old_parameter']and known['normal']==['0']*468+['1','0']and known['dimension']==13 and known['period']==18 and known['sigma']==-1 and len(known['box'])==470,'literal last same-phase signed root')
 w.save('history-binding.json',dict(card=card,producer=spec,model=ib['model'],native_source=ib['native_source'],accepted=accepted,scope='Existing producer root authority; independent march replay is separate. Entire failed-width history retained.'))
 if metadata_only:return p
 d=p/'run/attempt022';chart=read(d/'target-certificate.json.gz');domain_paths=list((d/'events').glob('*-signed_correlated_domain.json.gz'));need(len(domain_paths)==1,'one actual full correction domain');domain=read(domain_paths[0]);inp=read(d/'inputs.json')
 need(chart['passed']and all(chart['gates'].values())and chart['endpoint_root_boxes'][0]==known['box']and chart['parameter_endpoints'][0]==known['exact_parameter']and chart['normal']==known['normal']and chart['origin']==known['origin']and chart['weights']==ib['weights'],'exact accepted chart22 and endpoint')
 need(chart['center']==domain['center']and chart['slope']==domain['slope']and chart['curvature']==domain['curvature']and chart['whole_box']==domain['whole_box']and chart['parameter_center']==domain['parameter']==inp['parameter']and domain['sigma']==-1 and domain['halfwidth_exponent']==inp['settings']['halfwidth_exponent']and domain['radius_exponent']==-12,'actual original full-error domain and predictor')
 need(len(chart['bound']['coordinate_inclusions'])==470 and all(chart['bound']['coordinate_inclusions'])and original_ball(chart['bound']['contraction'])<1,'original entire strict weighted theorem')
 w.save('input-binding.json',dict(card=card,known=known,model=ib['model'],native_source=ib['native_source'],chart=entry(d/'target-certificate.json.gz'),domain=dict(path=str(domain_paths[0]),**entry(domain_paths[0])),scope='Same original weighted C(s)/q/error domain; no old hidden scalar is used in the new output-gauge anchor.'))
 return known,chart,domain,ib['model']

def endpoint_profile(w):
 old=read(OUT/'profile-certificate.json.gz');pc=old_polynomial(old,'0');inp,z=C.C.L.K.fixture();z=[v.mid()for v in z];x=z[:13];rates=z[13:15];q=z[15:28];flat=(x+q)*18+rates;normal=[arb(0)]*468+[arb(1),arb(0)];times=[];used=w.used
 w.save('profile-endpoint-inputs.json',dict(map_inputs=inp,point=vector_json(flat),normal=vector_json(normal),origin=vector_json(flat),scope='Unrelated full470 residual, polynomial-matrix action and weighted arithmetic timing only; no root assertion.'))
 model=C.Retained(C.C.TwoRateCuspMap(inp),w.observe)
 for i in range(3):
  start=time.monotonic();r,j,local=assemble(model,flat,18,-1,normal,flat,arb(0),w.tick,w.observe);f=-pc*r;norm=max(abs(f[k,0]).upper()for k in range(470));w.tick();w.save(f'profile-endpoint-{i}.json.gz',dict(residual=matrix_json(r),forcing=matrix_json(f),weighted_bound=ball_json((norm/(1-arb(1)/2)).upper())));times.append(time.monotonic()-start);del r,j,local,f
 w.save('endpoint-profile.json',dict(times=times,added_bytes=w.used-used,native_calls=54,ticks=57,target_arrays_unopened=True));need(w.native==234 and w.ticks==243,'complete full234 native/243tick unrelated profile')

def admission(w,card):
 pr=read(OUT/'profile.json');ep=read(OUT/'endpoint-profile.json');spec=read(ROOT/RECEIPT);rows=spec['outcome']['attempts'];old=read(ROOT/'docs/reviews/native470-correlated003.json')['outcome'];cost=max(old['seconds'],max(r['seconds']for r in rows));secs=time.monotonic()-w.start+max(pr['forecast_seconds'],2*cost+15)+6*max(ep['times']);raw=max(pr['forecast_bytes'],2*max(r['certificate']['bytes']for r in rows)+32*1024**2)+4*ep['added_bytes']+w.used+8*1024**2;rss=max(pr['forecast_rss_kib'],2*spec['outcome']['peak_rss_kib']+256*1024,2*old['peak_rss_kib']+256*1024)
 ok=secs<300 and raw<1024**3 and rss<4*1024**2;w.save('measured-admission.json',dict(passed=ok,seconds=secs,raw_bytes=raw,rss_kib=rss,retained_max_actual_unit_seconds=cost,native_calls=432,ticks=450,target_arrays_unopened=True));need(ok,'complete cold-profile+retained actual-cost admission')

def sharpen(w,known,chart,domain,model):
 p=exact(known['exact_parameter']);normal=[exact(v)for v in known['normal']];origin=[exact(v)for v in known['origin']];box=[original_ball(v)for v in known['box']];point=[v.mid()for v in box];weights=[exact(v)for v in chart['weights']];Cpoint=old_polynomial(chart,known['exact_parameter']);q=original_ball(chart['bound']['contraction']);native=C.Retained(C.C.TwoRateCuspMap(model),w.observe)
 for k in range(3):
  fixed_parameter_domain(chart,domain,vector_json(box),vector_json(point),known['exact_parameter']);w.save(f'endpoint-refinement-{k}-input.json',dict(point=vector_json(point),known=vector_json(box),exact_parameter=known['exact_parameter']))
  residual,j,local=assemble(native,point,18,-1,normal,origin,p,w.tick,w.observe);w.save(f'endpoint-refinement-{k}-residual.json',matrix_json(residual));vals=[residual[i,0]for i in range(470)];box,record=refine(Cpoint,q,weights,point,vals,box);fixed_parameter_domain(chart,domain,vector_json(box),vector_json(point),known['exact_parameter']);w.save(f'endpoint-refinement-{k}-return.json',record);w.tick()
  if k<2:
   force=Cpoint*residual;point=[(point[i]-force[i,0]).mid()for i in range(470)];fixed_parameter_domain(chart,domain,vector_json(box),vector_json(point),known['exact_parameter']);w.save(f'endpoint-refinement-{k}-proposal.json',vector_json(point))
  del residual,j,local
 result=dict(known,box=vector_json(box));w.save('refined-known470.json',result);return result

def output_proposal(known,old):
 box=[original_ball(v)for v in known['box']];h=arb((1,-15));origin=old['origin'];mu=(box[-1]-exact(origin[-1])).mid();parameter=mu+h/2;d=exact(str(F(known['exact_parameter'])-F(old['parameter_center'])));velocity=[original_ball(s)+original_ball(a)*d for s,a in zip(old['slope'],old['curvature'],strict=True)];gamma=velocity[-1].mid();need(gamma.is_finite()and not gamma.contains(0),'nonzero finite output tangent proposal');initial=[exact_mid(box[i].mid()+velocity[i].mid()/gamma*(h/2))for i in range(470)];normal=['0']*469+['1'];anchor=dict(box=box,period=18,sigma=-1,dimension=13)
 need('exact_parameter'not in anchor,'changed gauge requires entire known-root projection');return initial,normal,origin,exact_mid(parameter),anchor

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False,endpoint_refined=False)
 def stop(*_):raise TimeoutError('300s owned output-gauge packet')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');pp=OUT/'source'/n;pp.parent.mkdir(parents=True,exist_ok=True);pp.write_bytes(data);w.used+=len(data)
  with ctx.workprec(256):
   card=read(ROOT/CARD);need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'source admitted before profile');C.profile(w);endpoint_profile(w);bind(M.Prefix(w,'metadata-'),card,True);admission(w,card);known,old,domain,model=bind(w,card);w.stage='endpoint';ans['target_acquired']=True;known=sharpen(w,known,old,domain,model);ans['endpoint_refined']=True
   initial,normal,origin,parameter,anchor=output_proposal(known,old);w.stage='target';w.save('target-inputs.json',dict(map_inputs=model,initial=initial,known=known,normal=normal,origin=origin,parameter=parameter,settings=SETTINGS,weights=old['weights'],anchor_exact_parameter_used=False));r=C.chart(C.Retained(C.C.TwoRateCuspMap(model),w.observe),initial,normal,origin,parameter,old['weights'],anchor,SETTINGS,w)
   need(w.native==432 and w.ticks==450,'complete fixed432phase/450tick packet');ans.update(passed=r['passed'],gates=r['gates'],strict_inclusions=sum(r['bound']['coordinate_inclusions']),anchor_inclusions=sum(r['anchor_check']['coordinate_inclusions']),contraction=r['bound']['contraction'],parameter_endpoints=r['parameter_endpoints'],physical_endpoints=[v[-2:]for v in r['endpoint_root_boxes']],scope='Fresh output-gauge primitive18 negative-one chart with complete470 known-root projection; no critical simplicity/complement or distant-junction proof.')
   if r['passed']:w.save('connected-right-endpoint.json',dict(box=r['endpoint_root_boxes'][1],normal=normal,origin=origin,exact_parameter=r['parameter_endpoints'][1],dimension=13,period=18,sigma=-1))
 except BaseException as e:ans.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and elapsed<300 and peak<4*1024**2);ans.update(seconds=elapsed,peak_rss_kib=peak,native_calls=w.native,ticks=w.ticks,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
