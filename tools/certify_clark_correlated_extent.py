#!/usr/bin/env python3
"""Bounded adaptive finite extent of the independently connected native6 fold."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_clark_correlated_extension as X
import diagnose_clark_fold_extent as G
from flint import arb,ctx
import flint
C,B,OLD,CT,T=X.C,X.B,X.OLD,X.CT,X.T
need,entry,decode,read=X.need,X.entry,X.decode,X.read
ball_json,vector_json,matrix_json,midpoint_matrix=X.ball_json,X.vector_json,X.matrix_json,X.midpoint_matrix
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extent/attempt001/run')
CARD='docs/plans/clark-correlated-extent-inputs.json';THREADS=C.THREADS
SETTINGS=dict(C.SETTINGS,seconds=600,raw_bytes=2*1024**3,attempts=32,accepted_charts=32,initial_halfwidth_exponent=-15,maximum_halfwidth_exponent=-9,minimum_halfwidth_exponent=-17,initial_amplitude_squared='1/2048',last_amplitude_squared='1/16',growth_after_consecutive_low_contraction=2,growth_contraction_threshold='1/4')
CAPS=dict(explicit=102,path=160,implicit=1,anchor=1,context=33,algebra=96,native_phase_returns=993,cache_reuses=64,events=7000)
FILES=sorted(set(X.FILES+G.FILES+[CARD,'docs/plans/clark-correlated-extent.md','docs/reviews/clark-fold-extent-geometry.json','docs/reviews/clark-correlated-extension-authority.json','tools/certify_clark_correlated_extent.py','tools/launch_clark_correlated_extent.py','tests/test_clark_correlated_extent.py']))

class Work(C.Work):
 def __init__(self):super().__init__();self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'600s2GiB finite extent')
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed extent '+key+' cap');self.calls[key]+=1
 def save(self,name,value,closing=False):
  raw=C.canonical_bytes(value);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw
  need(self.used+len(raw)<2*1024**3-(0 if closing else 4*1024**2),'2GiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

def forecast(profile,actual_seconds,actual_bytes):
 C.forecast(profile)
 counts=dict(path=160,mixed=105,context=33*8,algebra=96*4)
 seconds=max(2*sum(counts[k]*profile['max_seconds'][k]for k in counts)+20,2*16*actual_seconds+20)
 raw=max(2*sum(counts[k]*profile['max_bytes'][k]for k in counts)+16*1024**2,2*16*actual_bytes+16*1024**2)
 return dict(passed=bool(seconds<600 and raw<2*1024**3-4*1024**2),seconds=seconds,bytes=raw,charged_categories=counts,complete_actual_two_chart_floor=dict(seconds=actual_seconds,bytes=actual_bytes,scale=16))

def card_check(card):
 need(card['schema']=='clark-correlated-extent-v1'and card['status']=='reviewed'and card['settings']==SETTINGS and card['caps']==CAPS and card['original_card']==entry(ROOT/C.CARD),'literal reviewed finite extent design')
 need(card['endpoint_numeric_parent']==G.SPECS[0]and card['endpoint_numeric_tag']=='fold47','fixed next saved native fold point')


def admission(w,card):
 card_check(card);oldcard=read(ROOT/C.CARD);C.bind_profile(w,oldcard)
 spec=card['producer_extension'];p=Path(spec['root']);B.B.inventory(p,spec['outer'],w.tick);B.B.inventory(p/'run',spec['worker'],w.tick);rt=OLD.historical(p,spec['source'],0);out=read(p/'run/outcome.json')
 need(out['passed']and out['new_charts']==2 and out['calls']['native_phase_returns']==93,'complete actual two-chart cost and source floor')
 for n,e in rt['sources'].items():
  if n.startswith('src/')or n=='tools/certify_clark_correlated_fold.py':need(entry(ROOT/n)==e,'same native and correlated kernel '+n)
 profile=read(Path(oldcard['profile']['root'])/'run/profile.json');m=read(p/'run/manifest.json');f=forecast(profile,out['elapsed_seconds'],sum(e['bytes']for e in m['files'].values()));need(f['passed'],'complete32attempt same-source measured admission');w.save('extent-profile.json',dict(forecast=f,target_arrays_unopened=True));return oldcard


def bind_endpoint(w,card):
 spec=card['independent_extension'];p=Path(spec['root']);B.B.inventory(p,spec['outer'],w.tick);B.B.inventory(p/'run',spec['worker'],w.tick);OLD.historical(p,spec['source'],0);o=read(p/'run/outcome.json');ib=read(p/'run/input-binding.json');ep=read(p/'run/connected-endpoint.json');producer=card['producer_extension']
 need(o['passed']and o['native_calls']==0 and o['range']==['0','1/2048']and o['new_independent_tubes']==2 and o['prior_independent_tubes']==6 and o['new_native_contexts']==12 and o['endpoint_coordinate_inclusions']==51 and o['endpoint_memberships'],'exact independently connected generic native6 authority')
 need(ib['parent']==producer['root']and ib['source']==producer['source']and ib['pins']==[producer['outer'],producer['worker']]and entry(p/'run/connected-endpoint.json')==spec['endpoint'],'literal independently replayed producer and endpoint')
 need(len(ep['box'])==51 and decode(ep['parameter'])==arb(1)/2048,'complete oriented51 known starting root');w.save('known-endpoint-binding.json',dict(independent=spec,producer=producer,range=o['range'],map=ib['map'],endpoint=ep));return [decode(v)for v in ep['box']],ib['map']


def numeric_endpoint(w,card):
 spec=card['geometry'];p=Path(spec['root']);B.B.inventory(p,spec['outer'],w.tick);B.B.inventory(p/'run',spec['worker'],w.tick);OLD.historical(p,spec['source'],0);o=read(p/'run/outcome.json');need(o['passed']and o['native_calls']==0 and o['points']==66 and o['unique_points']==65 and o['b_strictly_increases_with_gain']and o['matrix_input_sha256']==G.MATRIX,'exact saved coordinate diagnosis')
 rows=read(p/'run/points.json');row=next(r for r in rows if r['parent']==0 and r['tag']==card['endpoint_numeric_tag']);need(row['gain']==4.109326262981831 and 1/2048<row['b']<1/16,'preselected point inside intended finite amplitude range')
 run,result=G.bind(w,card['endpoint_numeric_parent']);point=result['points'][47];need(point['fold']['tag']==row['tag']and point['gain']==row['gain'],'literal indexed previously qualified fold')
 records=[read(q)for q in sorted(run.glob('[0-9]*.json'))];matching=[r for r in records if r['kind']=='fold_curve_point'and r['tag']==row['tag']];need(len(matching)==1,'unique full native point');d=matching[0];need(all(d[k]==v for k,v in point['fold'].items())and d['passed'],'literal final qualified point record');f=run/d['array_file'];need(entry(f)==row['array']and entry(f)['sha256']==d['array_sha256'],'exact chosen full native arrays')
 with C.np.load(f,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
 value=G.coordinates(arrays['paired_physical'],arrays['cycle_x'],arrays['cycle_y'],arrays['projection']);need(value['b']==row['b']and value['q']==row['q'],'same normalized numerical phase orientation');w.save('endpoint-numeric-binding.json',dict(parent=card['endpoint_numeric_parent'],geometry=spec,row=row,point=point,array=entry(f),scope='Numerical proposal and all-phase comparison only; exact endpoint root and full51 tube membership remain separate.'));return B.B.exact_float(row['gain']),arrays


def next_width(h,passed,contraction,streak):
 if not passed:return h/2,0
 streak=streak+1 if contraction<arb(1)/4 else 0
 return (min(2*h,arb((1,-9))),0)if streak==2 else(h,streak)


def march(w,j,known):
 left=arb(1)/2048;limit=arb(1)/16;h=arb((1,-15));minimum=arb((1,-17));attempts=[];accepted=[];streak=0;_,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in CT.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known]
 for attempt in range(1,33):
  if left==limit:break
  h=min(h,(limit-left)/2)
  if h<minimum:break
  b=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(51)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record);contraction=arb(1)
  try:
   c,s,a,r,cache=C.prepare(w,j,candidate,known,left,b,h);w.charge('algebra');whole,ends,data,proof=CT.tube(j,c,s,a,b,h,r,acquire=cache.acquire,observe=w.emit,tick=w.tick);need(len(cache.used)==2,'both exact residual acquisitions reused once');w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=CT.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 literal predecessor uniqueness join');native=w.context(j,whole[:26],b+arb(0,h),CT.col(whole[26:]),support=[left,right]);w.save(tag+'/native.json.gz',native);oriented=all(decode(x)>0 for x in native['partner_orientation'])
    if not native['passed']or not oriented or not native['positive_amplitude']:record.update(status='complete_native_or_orientation_bound_unresolved')
    else:
     contraction=decode(proof['contraction']);record.update(passed=True,status='qualified_connected_native6_critical_tube',contraction=proof['contraction'],coordinates=51,primitive_period=6,complement_dimension=11,generic_fold_uniformly_proved=False);accepted.append(dict(attempt=attempt,center=c,slope=s,curvature=a,parameter=b,halfwidth=h,radii=r,right=right,endpoint=ends[1]));left=right;known=ends[1];seed=[x.mid()for x in known];slope=[(s[i]+a[i]*h).mid()for i in range(51)];w.save(tag+'/connected-endpoint.json',dict(parameter=ball_json(left),box=vector_json(known),new_qualified_charts=len(accepted)))
  except ValueError as e:
   if str(e)!='fixed coordinate radius ceiling':raise
   record.update(status='retained_radius_proposal_bound',error=type(e).__name__+': '+str(e))
  h,streak=next_width(h,record['passed'],contraction,streak);record.update(next_halfwidth=ball_json(h),growth_streak=streak);w.save(tag+'/decision.json',record);attempts.append(record)
  if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(1)/2048),target_parameter=ball_json(limit),scope='Full51/native qualified prefix retained even if finite target remains incomplete; no step/radius quota expansion.');w.save('continuation.json',result);w.save('connected-endpoint.json',dict(parameter=ball_json(left),box=vector_json(known),new_qualified_charts=len(accepted)));need(result['passed'],'bounded finite extent incomplete');return accepted


def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);C.OUT=OUT;B.OUT=OUT;w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,600);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=C.np.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS,settings=SETTINGS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<2*1024**3-4*1024**2,'complete source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   card=read(ROOT/CARD);oldcard=admission(w,card);known,mapdata=bind_endpoint(w,card);gain,arrays=numeric_endpoint(w,card);j,organizer,_,_=C.bind(w,oldcard);need(read(OUT/'input-binding.json')['map']==mapdata,'same exact independently proved map');charts=march(w,j,known);root,native,curvature,joins=C.endpoint(w,j,gain,arrays,organizer,charts);w.tick()
   result.update(passed=True,status='finite_organizer_connected_native6_critical_extent_with_generic_endpoint',new_charts=len(charts),prior_independent_charts=8,squared_amplitude_range=['0','1/16'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Finite joined exact native6 critical extent; selected positive fixed-gain point separately has H_U inverse/physical curvature. No uniform genericity, full distant boundary, native12/24 or whole-window claim; independent replay required.')
 except BaseException as e:result.update(passed=False,status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);result['passed']=bool(result['passed']and result['elapsed_seconds']<600 and result['peak_rss_kib']<2*1024**2);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
