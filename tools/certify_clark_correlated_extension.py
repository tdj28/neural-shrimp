#!/usr/bin/env python3
"""One bounded two-segment extension from the independently known Clark51 endpoint."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_clark_correlated_fold as C
from flint import arb,ctx
import flint
B,OLD,CT,T=C.B,C.OLD,C.CT,C.T
need,entry,decode=C.need,C.entry,C.decode
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:raw=f.read(64*1024**2+1)
 else:raw=p.read_bytes()
 need(len(raw)<=64*1024**2,'bounded saved prefix read');return json.loads(raw)
ball_json,vector_json,matrix_json,midpoint_matrix=C.ball_json,C.vector_json,C.matrix_json,C.midpoint_matrix
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extension/attempt001/run')
CARD='docs/plans/clark-correlated-extension-inputs.json';THREADS=C.THREADS
SETTINGS=dict(C.SETTINGS,seconds=90,raw_bytes=256*1024**2,attempts=4,accepted_charts=2,initial_halfwidth_exponent=-15,initial_amplitude_squared='3/8192')
CAPS=dict(explicit=18,path=20,implicit=1,anchor=1,context=3,algebra=12,native_phase_returns=138,cache_reuses=8,events=1000)
FILES=sorted(set(C.FILES+[CARD,'docs/plans/clark-correlated-extension.md','docs/reviews/clark-correlated-prefix-independent.json','tools/certify_clark_correlated_extension.py','tools/launch_clark_correlated_extension.py','tests/test_clark_correlated_extension.py']))

class Work(C.Work):
 def __init__(self):super().__init__();self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'90s/2GiB cap')
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed extension '+key+' cap');self.calls[key]+=1
 def save(self,name,value,closing=False):
  raw=C.canonical_bytes(value)
  if name.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(256 if closing else 252)*1024**2,'256MiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

def forecast(profile):
 C.forecast(profile) # also validates all exact original counts/extrema
 counts=dict(path=20,mixed=20,context=3*8,algebra=12*4)
 seconds=2*sum(counts[k]*profile['max_seconds'][k]for k in counts)+20
 raw=2*sum(counts[k]*profile['max_bytes'][k]for k in counts)+16*1024**2
 return dict(passed=bool(seconds<90 and raw<252*1024**2),seconds=seconds,bytes=raw,charged_categories=counts)

def admission(w,card):
 need(card['schema']=='clark-correlated-extension-v1'and card['settings']==SETTINGS and card['caps']==CAPS and card['original_card']==entry(ROOT/C.CARD),'literal fixed extension card')
 oldcard=read(ROOT/C.CARD);C.bind_profile(w,oldcard);p=Path(oldcard['profile']['root']);f=forecast(read(p/'run/profile.json'));need(f['passed'],'same-source extension measured admission');w.save('extension-profile.json',dict(forecast=f,target_arrays_unopened=True));return oldcard

def prefix(w,card):
 spec=card['independent_prefix'];p=Path(spec['root']);invs=[B.B.inventory(p,spec['outer'],w.tick),B.B.inventory(p/'run',spec['worker'],w.tick)];rt=OLD.historical(p,spec['source'],0);out=read(p/'run/outcome.json');ib=read(p/'run/input-binding.json');ep=read(p/'run/connected-endpoint.json')
 need(out['passed']and out['native_calls']==0 and out['range']==['0','3/8192']and len(out['accepted'])==6 and [x['attempt']for x in out['accepted']]==list(range(3,9))and all(x['native_contexts']==4 for x in out['accepted']),'exact independently qualified six-prefix authority')
 need(entry(p/'run/connected-endpoint.json')==spec['endpoint']and ep['dimension']==51 and ep['actual_native_period']==6 and decode(ep['parameter'])==arb(3)/8192 and len(ep['box'])==51,'literal full51 known endpoint')
 parent=card['producer_prefix'];pp=Path(parent['root']);need(ib['root']==parent['root']and ib['source']==parent['source']and [ib['outer'],ib['worker']]==[parent['outer'],parent['worker']],'same audited producer and immutable genealogy')
 invs +=[B.B.inventory(pp,parent['outer'],w.tick),B.B.inventory(pp/'run',parent['worker'],w.tick)];OLD.historical(pp,parent['source'],2)
 need(read(pp/'run/outcome.json')['error']=='ValueError: bounded correlated critical range incomplete','retained incomplete original range')
 # Native kernel and inherited caller remain byte-identical to reviewed prefix.
 original=read(pp/'run/runtime.json')
 for n in original['sources']:
  if n.startswith('src/')or n=='tools/certify_clark_correlated_fold.py':need(entry(ROOT/n)==original['sources'][n],'same original map/equations/caller '+n)
 charts=[]
 for k in range(3,9):
  root=read(p/f'run/charts/{k:02d}/root.json.gz');join=read(p/f'run/charts/{k:02d}/join.json');native=read(p/f'run/charts/{k:02d}/native.json.gz');need(root['passed']and join['passed']and join['coordinates']==51 and native['passed']and len(native['contexts'])==4,'whole independent prior root/native/join')
  t=read(pp/f'run/charts/{k:02d}/tube.json.gz');charts.append(dict(attempt='prefix-'+str(k),**{n:[decode(x)for x in t[n]]for n in('center','slope','curvature','radii')},parameter=decode(t['parameter']),halfwidth=decode(t['halfwidth'])))
 w.save('prefix-binding.json',dict(independent=spec,producer=parent,inventories=invs,endpoint=spec['endpoint'],scope='Exact native phase/all51 endpoint inherited from independent512 raw-assembly/root/native replay; original failed wider bounds and incomplete range remain immutable.'))
 return [decode(x)for x in ep['box']],charts,ib

prepare=C.prepare
def march(w,j,known):
 left=arb(3)/8192;limit=arb(1)/2048;h=arb((1,-15));minimum=arb((1,-17));attempts=[];accepted=[];_,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in CT.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known]
 for attempt in range(1,5):
  if left==limit or len(accepted)>=2:break
  h=min(h,(limit-left)/2);need(h>=minimum,'fixed minimum step reached');b=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(51)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record)
  try:
   c,s,a,r,cache=prepare(w,j,candidate,known,left,b,h);w.charge('algebra');whole,ends,data,proof=CT.tube(j,c,s,a,b,h,r,acquire=cache.acquire,observe=w.emit,tick=w.tick);need(len(cache.used)==2,'both exact saved residuals reused once');w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=CT.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 curved exact predecessor root');native=w.context(j,whole[:26],b+arb(0,h),CT.col(whole[26:]),support=[left,right]);w.save(tag+'/native.json.gz',native);oriented=all(decode(x)>0 for x in native['partner_orientation'])
    if not native['passed']or not oriented or(left>0 and not native['positive_amplitude']):record.update(status='complete_native_or_orientation_bound_unresolved')
    else:
     record.update(passed=True,status='qualified_connected_native6_correlated_critical_tube',contraction=proof['contraction'],coordinates=51,primitive_period=6,complement_dimension=11,includes_degenerate_zero=bool(left==0),generic_fold_uniformly_proved=False);accepted.append(dict(attempt=attempt,center=c,slope=s,curvature=a,parameter=b,halfwidth=h,radii=r,right=right,endpoint=ends[1]));left=right;known=ends[1];seed=[x.mid()for x in known];slope=[(s[i]+a[i]*h).mid()for i in range(51)]
  except ValueError as error:
   if str(error)!='fixed coordinate radius ceiling':raise
   record.update(status='retained_radius_proposal_bound',error=type(error).__name__+': '+str(error))
  w.save(tag+'/decision.json',record);attempts.append(record)
  if not record['passed']:
   h/=2
   if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(3)/8192),target_parameter=ball_json(limit),scope='Only strict full51 curved roots with exact predecessor and literal native6 partner/phase/mode/primitive/complement gates advance. b0 is degenerate; uniform H_U/genericity remain separate.');w.save('continuation.json',result);need(result['passed'],'bounded correlated critical range incomplete');return accepted

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);C.OUT=OUT;B.OUT=OUT;w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,90);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=C.np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS,settings=SETTINGS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<252*1024**2,'source storage cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   card=read(ROOT/CARD);oldcard=admission(w,card);known,prior,ib=prefix(w,card);j,organizer,gain,arrays=C.bind(w,oldcard)
   need(read(OUT/'input-binding.json')['map']==ib['map'],'same actual independently proved map');charts=march(w,j,known);root,native,curvature,joins=C.endpoint(w,j,gain,arrays,organizer,prior+charts)
   result.update(passed=True,status='exact_organizer_connected_native6_critical_curve_with_generic_endpoint',new_charts=len(charts),prior_independent_charts=6,squared_amplitude_range=['0','1/2048'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Six independent prior tubes plus new full51 joined critical tubes; selected positive endpoint separately proves H_U inverse/physical curvature. No uniform genericity/native12/24/full-window claim. New outcome needs independent replay.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
