"""One freshly acquired scaled output-gauge chart from the same known endpoint."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
from pathlib import Path
import sys,json,gzip,time,signal,subprocess,platform,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native470_output_gauge as D
import diagnose_native470_output_gauge as G
from flint import arb,ctx
import flint
from fractions import Fraction as F
C=D.C;need,read,entry=D.need,D.read,D.entry
from neural_shrimp.rigorous_orbits import canonical_bytes,vector_json,exact
from neural_shrimp.rigorous_eigenpair_locus import exact_mid
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
OUT=D.OUT.parents[1]/'attempt002/run';CARD='docs/plans/native470-output-weighted-inputs.json'
RECEIPT='docs/reviews/native470-output-diagnosis001.json'
SETTINGS=dict(period=18,sigma=-1,precision=256,newton_steps=2,halfwidth_exponent=-18,radius_exponent=-12)
LIMITS=dict(seconds=240,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=768*1024**2,native_calls=324,ticks=336)
FILES=sorted(set(G.FILES+[CARD,RECEIPT,'tools/probe_native470_output_weighted.py','tools/launch_native470_output_weighted.py','tests/test_native470_output_weighted.py','docs/plans/native470-output-weighted.md']))
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.ticks=0;self.native=0;self.serial=0;self.stage='profile'
 def check(self):need(time.monotonic()-self.start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB complete scaled output chart')
 def tick(self):self.ticks+=1;need(self.ticks<=336,'186 profile plus150 target ticks');self.check()
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);data=gzip.compress(raw,mtime=0)if n.endswith('.gz')else raw;need(self.used+len(data)<768*1024**2-(0 if closing else 4*1024**2),'768MiB with4MiB closing reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  self.used+=len(data)
  if not closing:self.check()
 def observe(self,k,v):
  if k=='native_input':self.native+=1;need(self.native<=324,'180 profile plus144 target phases')
  if k=='signed_correlated_chart_return':self.save(self.stage+'-certificate.json.gz',v)
  else:self.save(f'events/{self.serial:04d}-{self.stage}-{k}.json.gz',v);self.serial+=1

def check_authority(ao,ai,known,prior,oldib):
 need(ao['passed']and ao['native_calls']==0 and ao['accepted_indices']==prior['outcome']['accepted_indices']and ao['exact_parameter']=='-2001/256'and len(ao['rows'])==16 and all(r['passed']and r['root_coordinates']==r['anchor_coordinates']==470 for r in ao['rows']),'entire independent16 root and join authority')
 need(ai['card']['producer']=={k:prior[k]for k in ('root','source','outer','worker','outcome','exit_code')} and ai['all_attempts']==23 and ai['accepted_indices']==ao['accepted_indices'],'literal independent same producer genealogy')
 need(ai['model']==oldib['model']and ai['native_source']==oldib['native_source'],'same entire actual model/native equations')
 need(known['exact_parameter']=='-2001/256'and known['normal']==['0']*468+['1','0']and len(known['box'])==470 and known['dimension']==13 and known['period']==18 and known['sigma']==-1,'literal independent470 phase/gauge/scalar')

def bind(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS and card['diagnosis_receipt']==entry(ROOT/RECEIPT),'fixed source-admitted scaled output chart')
 ds=read(ROOT/RECEIPT);dp,dr=C.history(w,ds);do=read(dp/'run/outcome.json');need(do==ds['outcome']and do['passed']and do['native_calls']==0 and entry(dp/'run/diagnostic.json')==ds['diagnostic'],'exact saved proposal only')
 failed=read(ROOT/G.RECEIPT);fp,fr=C.history(w,failed,2);fo=read(fp/'run/outcome.json');need(fo==failed['outcome']and not fo['passed']and fo['native_calls']==432 and fo['endpoint_refined'],'complete failed output chart remains failed')
 G.bind(D.M.Prefix(w,'diagnosis-'),read(ROOT/G.CARD));prior=read(ROOT/D.RECEIPT);mp=D.bind(D.M.Prefix(w,'old-'),read(ROOT/D.CARD),True)
 ap,ar=C.history(w,card['independent']);ao=read(ap/'run/outcome.json');ai=read(ap/'run/input-binding.json');known=read(ap/'run/connected-left-endpoint.json');oldib=read(mp/'run/input-binding.json')
 need(entry(ap/'run/connected-left-endpoint.json')==card['independent']['endpoint'],'literal independent reconstructed endpoint');check_authority(ao,ai,known,prior,oldib)
 for n,e in ai['native_source'].items():need(entry(ROOT/n)==fr['sources'][n]==e,'same current native map '+n)
 for n in ['src/neural_shrimp/rigorous_signed_correlated.py','src/neural_shrimp/rigorous_signed_gradient_path.py','src/neural_shrimp/rigorous_signed_proposal.py','src/neural_shrimp/rigorous_polynomial_inverse.py']:need(entry(ROOT/n)==fr['sources'][n],'unchanged proof kernel '+n)
 w.save('history-binding.json',dict(diagnosis=ds,failed_output=failed,independent=card['independent'],model=ai['model'],native_source=ai['native_source'],scope='Independent endpoint and prior refined box identify the same root through the original chart22 uniqueness domain. Diagnostic weights remain proposals only.'))
 if metadata_only:return fp,mp,ap
 dg=read(dp/'run/diagnostic.json');weights=dg['dyadic_weight_proposal'];need(len(weights)==470 and all(F(v)>0 for v in weights)and F(weights[-2])==1 and F(weights[-1])==F(1,1048576),'actual frozen positive dyadic scaling proposal')
 refined=read(fp/'run/refined-known470.json');need(entry(fp/'run/refined-known470.json')==failed['files']['refined-known470.json'],'literal passed refinement')
 old=read(mp/'run/attempt022/target-certificate.json.gz');paths=list((mp/'run/attempt022/events').glob('*-signed_correlated_domain.json.gz'));need(len(paths)==1,'one actual original error domain');domain=read(paths[0])
 need(old['passed']and all(old['gates'].values())and len(old['bound']['coordinate_inclusions'])==470 and all(old['bound']['coordinate_inclusions'])and old['parameter_endpoints'][0]==known['exact_parameter'],'original strict uniqueness theorem at same exact scalar')
 need(all(refined[k]==known[k]for k in ('dimension','period','sigma','exact_parameter','normal','origin'))and old['normal']==known['normal']and old['origin']==known['origin'],'same full signed phase and old gauge')
 with ctx.workprec(512):
  a=[original_ball(v)for v in known['box']];b=[original_ball(v)for v in refined['box']];box=[]
  for values in (a,b):D.fixed_parameter_domain(old,domain,vector_json(values),vector_json([v.mid()for v in values]),known['exact_parameter'])
  for x,y in zip(a,b,strict=True):
   lo=max(x.lower(),y.lower());hi=min(x.upper(),y.upper());need(lo<=hi,'same proved-root boxes intersect');box.append(arb((lo+hi)/2,(hi-lo)/2))
  D.fixed_parameter_domain(old,domain,vector_json(box),vector_json([v.mid()for v in box]),known['exact_parameter'])
  joined=dict(known,box=vector_json(box))
 w.save('known-root-identity.json',dict(passed=True,coordinates=470,known=known,refined=refined,intersection=joined,original_chart=entry(mp/'run/attempt022/target-certificate.json.gz'),original_domain=entry(paths[0]),scope='Conditional uniqueness in the same original closed error box at the exact old hidden parameter; no new derivative evaluation.'))
 return box,old,ai['model'],weights,joined

def proposal(box,old,halfwidth=-18):
 h=arb((1,halfwidth));mu=(box[-1]-exact(old['origin'][-1])).mid();parameter=mu+h/2;d=exact(str(F('-2001/256')-F(old['parameter_center'])));v=[original_ball(s)+original_ball(a)*d for s,a in zip(old['slope'],old['curvature'],strict=True)];gamma=v[-1].mid();need(gamma.is_finite()and not gamma.contains(0),'finite nonzero output tangent proposal')
 initial=[exact_mid(box[i].mid()+v[i].mid()/gamma*h/2)for i in range(470)];anchor=dict(box=box,period=18,sigma=-1,dimension=13)
 return initial,['0']*469+['1'],old['origin'],exact_mid(parameter),anchor

def admission(w):
 pr=read(OUT/'profile.json');old=read(ROOT/G.RECEIPT)['outcome'];diag=read(ROOT/RECEIPT)['outcome'];forecast=dict(seconds=time.monotonic()-w.start+max(pr['forecast_seconds'],2*old['seconds']+15)+2*diag['seconds'],rss_kib=max(pr['forecast_rss_kib'],2*old['peak_rss_kib']+256*1024),bytes=w.used+pr['forecast_bytes']+32*1024**2)
 ok=forecast['seconds']<240 and forecast['rss_kib']<4*1024**2 and forecast['bytes']<768*1024**2;w.save('measured-admission.json',dict(passed=ok,forecast=forecast,retained_failed_cost=old['seconds'],retained_diagnosis_cost=diag['seconds'],target_arrays_unopened=True));need(ok,'full fresh profile plus actual retained cold costs')

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('240s owned weighted output chart')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(D.THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed complete source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);w.used+=len(data)
  with ctx.workprec(256):
   card=read(ROOT/CARD);need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'source admitted before profile');C.profile(w);bind(D.M.Prefix(w,'metadata-'),card,True);admission(w);box,old,model,weights,known=bind(w,card);initial,normal,origin,parameter,anchor=proposal(box,old);w.stage='target';ans['target_acquired']=True
   w.save('target-inputs.json',dict(initial=initial,normal=normal,origin=origin,parameter=parameter,weights=weights,known=known,map_inputs=model,settings=SETTINGS,anchor_exact_parameter_used=False));r=C.chart(C.Retained(C.C.TwoRateCuspMap(model),w.observe),initial,normal,origin,parameter,weights,anchor,SETTINGS,w)
   need(w.native==324 and w.ticks==336,'complete fixed acquisition');ans.update(passed=r['passed'],gates=r['gates'],strict_inclusions=sum(r['bound']['coordinate_inclusions']),anchor_inclusions=sum(r['anchor_check']['coordinate_inclusions']),contraction=r['bound']['contraction'],parameter_endpoints=r['parameter_endpoints'],physical_endpoints=[v[-2:]for v in r['endpoint_root_boxes']],scope='Fresh scaled output-gauge primitive18 negative-one curve with full470 projected known-root join; no critical simplicity/complement/distant-junction claim.')
   if r['passed']:w.save('connected-right-endpoint.json',dict(box=r['endpoint_root_boxes'][1],normal=normal,origin=origin,exact_parameter=r['parameter_endpoints'][1],dimension=13,period=18,sigma=-1))
 except BaseException as e:ans.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);secs=time.monotonic()-w.start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ans['passed']=bool(ans['passed']and secs<240 and rss<4*1024**2);ans.update(seconds=secs,peak_rss_kib=rss,native_calls=w.native,ticks=w.ticks,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
