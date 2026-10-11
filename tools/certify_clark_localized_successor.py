#!/usr/bin/env python3
"""Finite native6 continuation with same-domain subset and point endpoint bounds."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import certify_clark_correlated_extension as X
import certify_clark_correlated_extent as PREVIOUS
import clark_subset_continuation_math as S
import scipy
import certify_clark_subset_successor as PRECEDENT
import clark_endpoint_localization as EL
import diagnose_clark_fold_extent as G
from flint import arb,ctx
import flint
C,B,OLD,CT,T=X.C,X.B,X.OLD,X.CT,X.T
need,entry,decode,read=X.need,X.entry,X.decode,X.read
ball_json,vector_json,matrix_json,midpoint_matrix=X.ball_json,X.vector_json,X.matrix_json,X.midpoint_matrix
OUT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-localized-successor/attempt001/run')
CARD='docs/plans/clark-localized-successor-inputs.json';THREADS=C.THREADS
SETTINGS=dict(C.SETTINGS,seconds=600,raw_bytes=2*1024**3,attempts=24,accepted_charts=24,initial_halfwidth_exponent=-14,maximum_halfwidth_exponent=-7,minimum_halfwidth_exponent=-17,initial_amplitude_squared='1/512',endpoint_localization_iterations=16,endpoint_localization_precision=512,last_amplitude_squared='1/16',growth_after_consecutive_low_contraction=2,growth_contraction_threshold='9/10',subset_lp_seconds_each=1,subset_fallbacks=24)
CAPS=dict(explicit=78,path=120,implicit=1,anchor=1,context=25,algebra=98,subset=25,native_phase_returns=753,cache_reuses=48,events=7000,localization=25)
PRIOR=dict(root='/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extent/attempt001',source='b1fc38c14906d6c2a6ef16fab7e82bbbe2f62676',outer='71e8f0e318d1427950e63aba0466efe213808c4336788277843919fbc48875dd',worker='2812d71cd15953a3cc19772fdd5b488536ec885f8819bf50267adbb1b4cea4ab')
NATIVE_PRIOR=dict(root='/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-subset-successor/attempt001',source='cf1cd08e7a868e0dd948c6bd4b188c99f9a53767',outer='8443f4e777c5ebfa5c4a0659ba90dbc8ebb19b84d44cc0932ed06946dbbd552a',worker='53bbddefb9ec52767b8a48a17de98b0e59326a9859bb874473d8a408a16d0fca')
KNOWN=dict(root='/data2/neural-shrimp-data/audits/clark-subset-endpoint-independent/attempt004',source='93bb96e3acfe58fd4129516db544a6619e0f9a0a',outer='fdd1473c7380d1be8aae0d7eafa084da7bb08cbc0556adfb0ffb13eddcf15949',worker='a90e64bda1536235c4276a268f000a0abbb6304f22d1cd45a9e9e37d162902e9',endpoint=dict(bytes=16253,sha256='b2d46d265de3457bf25346dc1c318cd900bc9f95239b82408eae33ff733b76ac'))
SCREEN=dict(root='/data2/neural-shrimp-data/audits/clark-subset-radius-screen/attempt002',source='f3d9db678a7feecb65b0fd5a7613060cecd456c5',outer='c559253a14829a35a8be9893ffe5344acbce37ea4c88aeed3084951a6c74074b',worker='d58f1a9701323cffd6e42edddbd0dc00ab0eded0b3958315d40d466e5a3061ae')
FILES=sorted(set(PRECEDENT.FILES+[CARD,'docs/plans/clark-localized-successor.md','docs/reviews/clark-subset-endpoint-independent-attempt004.json','tools/clark_endpoint_localization.py','tools/certify_clark_localized_successor.py','tools/launch_clark_localized_successor.py','tests/test_clark_endpoint_localization.py','tests/test_clark_localized_successor.py']))


class Work(C.Work):
 def __init__(self):super().__init__();self.calls=dict.fromkeys(CAPS,0)
 def tick(self):need(not self.stopped and time.monotonic()-self.start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'600s2GiB finite extent')
 def charge(self,key):self.tick();need(self.calls[key]<CAPS[key],'fixed extent '+key+' cap');self.calls[key]+=1
 def save(self,name,value,closing=False):
  raw=C.canonical_bytes(value);raw=gzip.compress(raw,compresslevel=1,mtime=0)if name.endswith('.gz')else raw
  need(self.used+len(raw)<2*1024**3-(0 if closing else 4*1024**2),'2GiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)

def forecast(profile,prior,screen,actual_bytes,screen_bytes,synthetic_seconds,synthetic_bytes,elapsed=0,localization_seconds=0,localization_bytes=0):
 C.forecast(profile)
 counts=dict(path=120,mixed=81,context=25*8,algebra=72*4)
 # The subset unit floor already includes its second reduction and LP; it is
 # disjoint from the three original algebra stages per fresh chart above.
 lp_unit=max(1.,screen['seconds']/10,synthetic_seconds)
 seconds=elapsed+max(2*sum(counts[k]*profile['max_seconds'][k]for k in counts),2*prior['elapsed_seconds'])+2*24*lp_unit+2*24*max(.1,localization_seconds)+20
 raw=max(2*sum(counts[k]*profile['max_bytes'][k]for k in counts),2*actual_bytes)+2*24*max(screen_bytes/10,synthetic_bytes)+2*24*localization_bytes+16*1024**2
 peak=max(2*prior['peak_rss_kib'],2*screen['peak_rss_kib'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024+256*1024**2
 return dict(passed=bool(seconds<600 and raw<2*1024**3-4*1024**2 and peak<2*1024**3),seconds=seconds,bytes=raw,peak_bytes=peak,charged_categories=counts,subset_unit_seconds=lp_unit,complete_prior_seconds=prior['elapsed_seconds'],complete_screen_seconds=screen['seconds'],elapsed_before_target=elapsed,localization_unit_seconds=max(.1,localization_seconds),localization_unit_bytes=localization_bytes)

def card_check(card):
 need(card['schema']=='clark-localized-successor-v1'and card['status']=='reviewed'and card['settings']==SETTINGS and card['caps']==CAPS and card['original_card']==entry(ROOT/C.CARD)and card['prior_extent_card']==entry(ROOT/PREVIOUS.CARD)and card['precedent_card']==entry(ROOT/PRECEDENT.CARD),'literal reviewed subset successor design')
 need(card['prior']==PRIOR and card['known']==KNOWN and card['screen']==SCREEN and card['native_prior']==NATIVE_PRIOR,'literal failed native extent/independent prefix/subset-screen pins')
 need(card['endpoint_numeric_parent']==G.SPECS[0]and card['endpoint_numeric_tag']=='fold47','fixed next saved native fold point')

def history(w,spec,exitcode):
 p=Path(spec['root']);B.B.inventory(p,spec['outer'],w.tick);B.B.inventory(p/'run',spec['worker'],w.tick);rt=OLD.historical(p,spec['source'],exitcode);return p,rt,read(p/'run/outcome.json')

def synthetic_profile(w):
 start=time.monotonic();before=w.used;n=51;I=CT.eye(n);Z=S.arb_mat(n,n);J0=CT.eye(n);J0[0,1]=-2;R=[CT.col([arb(1)/1000]*n)]+[S.arb_mat(n,1)]*3;zero=[arb(0)]*n;r=[arb(1)]*n;h=arb(1)/16;w.charge('algebra');data=CT.reduce([I,Z],[J0,Z,Z],R,r,h,tick=w.tick);w.charge('subset');w.charge('algebra')
 answer=S.restrict([I,Z],[J0,Z,Z],R,data,r,zero,arb(0),zero,zero,zero,arb(0),h,observe=lambda k,v:w.save('profile-subset/'+k+'.json.gz',v),tick=w.tick)
 need(answer is not None and answer[3]['passed'],'fresh full51 synthetic subset/inclusion/join')
 return time.monotonic()-start,w.used-before

def localization_profile(w):
 start=time.monotonic();before=w.used;n=51;I=CT.eye(n);Z=S.arb_mat(n,n);Ds=[I/4,Z,Z,Z];fs=[CT.col([arb(1)/8]*n)]+[S.arb_mat(n,1)]*4;image=fs[0]+Ds[0]*CT.col([arb(0,1)]*n);data=dict(passed=True,dimension=n,defect_coefficients=Ds,forcing_coefficients=fs,image=image);zero=[arb(0)]*n
 w.charge('localization')
 with ctx.workprec(512):result=EL.localize(data,zero,zero,zero,[arb(1)]*n,arb(1)/16,CT.vec(image),tick=w.tick)
 need(result['restricted'],'fresh full51 point localization profile');w.save('profile-endpoint-localization.json.gz',serialize_localization(result));return time.monotonic()-start,w.used-before

def serialize_localization(v):
 return dict(passed=v['passed'],restricted=v['restricted'],iterations=v['iterations'],endpoint=vector_json(v['endpoint']),proposal=vector_json(v['proposal']),serialized_subset_checks=v['serialized_subset_checks'],point_defect=matrix_json(v['point_defect']),point_forcing=matrix_json(v['point_forcing']),history=[{k:vector_json(x)for k,x in row.items()}for row in v['history']],scope=v['scope'])

def admission(w,card):
 card_check(card);oldcard=read(ROOT/C.CARD);C.bind_profile(w,oldcard);p,rt,prior=history(w,PRIOR,2);sp,srt,screen=history(w,SCREEN,0)
 need(not prior['passed']and prior['error']=='ValueError: bounded finite extent incomplete'and prior['calls']['native_phase_returns']==906 and len(read(p/'run/continuation.json')['attempts'])==32,'entire retained32-attempt native extent')
 need(screen['passed']and screen['native_calls']==0 and screen['verified']==10 and len(screen['attempts'])==10,'complete allten subset screen floor')
 for n,e in rt['sources'].items():
  if n.startswith('src/')or n=='tools/certify_clark_correlated_fold.py':need(entry(ROOT/n)==e,'same native/path/correlated kernel '+n)
 for n in('tools/clark_subset_radius_math.py',):need(entry(ROOT/n)==srt['sources'][n],'same exact LP/verifier implementation')
 need(srt['numpy']==C.np.__version__ and srt['scipy']==scipy.__version__ and srt['python_flint']==flint.__version__ and srt['flint']==flint.__FLINT_VERSION__,'same actual LP/interval runtime')
 unit,raw=synthetic_profile(w);lu,lb=localization_profile(w);profile=read(Path(oldcard['profile']['root'])/'run/profile.json');size=lambda q:sum(e['bytes']for e in read(q/'run/manifest.json')['files'].values());f=forecast(profile,prior,screen,size(p),size(sp),unit,raw,time.monotonic()-w.start,lu,lb);w.save('successor-profile.json',dict(forecast=f,synthetic_seconds=unit,synthetic_bytes=raw,target_arrays_unopened=True,prior=PRIOR,screen=SCREEN));need(f['passed'],'complete native+subset measured admission before arrays');return oldcard

def bind_endpoint(w,card):
 p,rt,o=history(w,KNOWN,0);ib=read(p/'run/input-binding.json');ep=read(p/'run/connected-endpoint.json');np,nrt,no=history(w,NATIVE_PRIOR,2)
 need(o['passed']and o['native_calls']==0 and o['range']==['15/8192','1/512']and o['new_independent_tubes']==1 and o['prior_independent_tubes']==30 and o['new_native_contexts']==4,'literal independent31 native6 endpoint authority')
 need(ib['parent']==NATIVE_PRIOR['root']and ib['source']==NATIVE_PRIOR['source']and ib['pins']==[NATIVE_PRIOR['outer'],NATIVE_PRIOR['worker']]and entry(p/'run/connected-endpoint.json')==KNOWN['endpoint'],'literal independently localized producer and all51 endpoint')
 need(not no['passed']and no['calls']['native_phase_returns']==132 and no['error']=='ValueError: bounded finite extent incomplete','entire retained native successor failure');rows=read(np/'run/continuation.json');need(rows['accepted']==1 and len(rows['attempts'])==5,'one qualified plus four unresolved predecessors')
 for n,e in nrt['sources'].items():
  if n.startswith('src/')or n in('tools/certify_clark_correlated_fold.py','tools/clark_subset_continuation_math.py','tools/clark_subset_radius_math.py'):need(entry(ROOT/n)==e,'unchanged native/fullE/subset kernel '+n)
 need(len(ep['box'])==51 and decode(ep['parameter'])==arb(1)/512,'complete oriented51 starting root');w.save('known-endpoint-binding.json',dict(independent=KNOWN,producer=NATIVE_PRIOR,range=o['range'],map=ib['map'],endpoint=ep,prior_failures_preserved=True));return [decode(v)for v in ep['box']],ib['map']


def numeric_endpoint(w,card):
 spec=card['geometry'];p=Path(spec['root']);B.B.inventory(p,spec['outer'],w.tick);B.B.inventory(p/'run',spec['worker'],w.tick);OLD.historical(p,spec['source'],0);o=read(p/'run/outcome.json');need(o['passed']and o['native_calls']==0 and o['points']==66 and o['unique_points']==65 and o['b_strictly_increases_with_gain']and o['matrix_input_sha256']==G.MATRIX,'exact saved coordinate diagnosis')
 rows=read(p/'run/points.json');row=next(r for r in rows if r['parent']==0 and r['tag']==card['endpoint_numeric_tag']);need(row['gain']==4.109326262981831 and 1/2048<row['b']<1/16,'preselected point inside intended finite amplitude range')
 run,result=G.bind(w,card['endpoint_numeric_parent']);point=result['points'][47];need(point['fold']['tag']==row['tag']and point['gain']==row['gain'],'literal indexed previously qualified fold')
 records=[read(q)for q in sorted(run.glob('[0-9]*.json'))];matching=[r for r in records if r['kind']=='fold_curve_point'and r['tag']==row['tag']];need(len(matching)==1,'unique full native point');d=matching[0];need(all(d[k]==v for k,v in point['fold'].items())and d['passed'],'literal final qualified point record');f=run/d['array_file'];need(entry(f)==row['array']and entry(f)['sha256']==d['array_sha256'],'exact chosen full native arrays')
 with C.np.load(f,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
 value=G.coordinates(arrays['paired_physical'],arrays['cycle_x'],arrays['cycle_y'],arrays['projection']);need(value['b']==row['b']and value['q']==row['q'],'same normalized numerical phase orientation');w.save('endpoint-numeric-binding.json',dict(parent=card['endpoint_numeric_parent'],geometry=spec,row=row,point=point,array=entry(f),scope='Numerical proposal and all-phase comparison only; exact endpoint root and full51 tube membership remain separate.'));return B.B.exact_float(row['gain']),arrays


def next_width(h,passed,contraction,streak):
 if not passed:return h/2,0
 streak=streak+1 if contraction<arb(9)/10 else 0
 return (min(2*h,arb((1,-7))),0)if streak==2 else(h,streak)


def march(w,j,known):
 left=arb(1)/512;limit=arb(1)/16;h=arb((1,-14));minimum=arb((1,-17));attempts=[];accepted=[];streak=0;_,J,fb,_=w.family(j,[x.mid()for x in known],left);slope=[x.mid()for x in CT.vec(-midpoint_matrix(J).inv()*fb)];seed=[x.mid()for x in known]
 for attempt in range(1,25):
  if left==limit:break
  h=min(h,(limit-left)/2)
  if h<minimum:break
  b=left+h;right=left+2*h;candidate=[(seed[i]+slope[i]*h).mid()for i in range(51)];tag=f'charts/{attempt:02d}';record=dict(attempt=attempt,left=ball_json(left),right=ball_json(right),halfwidth=ball_json(h),passed=False,accepted_prefix=len(accepted));w.save(tag+'/attempt.json',record);contraction=arb(1)
  try:
   c,s,a,r,cache=C.prepare(w,j,candidate,known,left,b,h);w.charge('algebra');whole,ends,data,proof=CT.tube(j,c,s,a,b,h,r,acquire=cache.acquire,observe=w.emit,tick=w.tick);need(len(cache.used)==2,'both exact residual acquisitions reused once');w.save(tag+'/tube-original.json.gz',proof)
   record['subset_fallback']=False
   if not proof['passed']:
    w.charge('subset');w.charge('algebra');matrix=lambda rows:S.arb_mat([[decode(v)for v in row]for row in rows]);Cs=[matrix(x)for x in proof['inverse_coefficients']];Js=[matrix(x)for x in proof['jacobian_coefficients']];Rs=[matrix(x)for x in proof['residual_coefficients']]
    repaired=S.restrict(Cs,Js,Rs,data,r,known,left,c,s,a,b,h,observe=lambda k,v:w.save(tag+'/subset-'+k+'.json.gz',v),tick=w.tick)
    if repaired is not None:whole,ends,data,proof,r=repaired;record['subset_fallback']=True
   w.save(tag+'/tube.json.gz',proof)
   if not proof['passed']:record.update(status='complete_uniform_bound_unresolved')
   else:
    join=CT.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 literal predecessor uniqueness join');native=w.context(j,whole[:26],b+arb(0,h),CT.col(whole[26:]),support=[left,right]);w.save(tag+'/native.json.gz',native);oriented=all(decode(x)>0 for x in native['partner_orientation'])
    if not native['passed']or not oriented or not native['positive_amplitude']:record.update(status='complete_native_or_orientation_bound_unresolved')
    else:
     contraction=decode(proof['contraction']);record.update(passed=True,status='qualified_connected_native6_critical_tube',contraction=proof['contraction'],coordinates=51,primitive_period=6,complement_dimension=11,generic_fold_uniformly_proved=False);w.charge('localization')
     with ctx.workprec(512):localized=EL.localize(data,c,s,a,r,h,ends[1],iterations=16,tick=w.tick)
     w.save(tag+'/endpoint-localization.json.gz',serialize_localization(localized));record['endpoint_restricted']=localized['restricted'];accepted.append(dict(attempt=attempt,center=c,slope=s,curvature=a,parameter=b,halfwidth=h,radii=r,right=right,endpoint=localized['endpoint']));left=right;known=localized['endpoint'];seed=[x.mid()for x in known];slope=[(s[i]+a[i]*h).mid()for i in range(51)];w.save(tag+'/connected-endpoint.json',dict(parameter=ball_json(left),box=vector_json(known),new_qualified_charts=len(accepted)))
  except ValueError as e:
   if str(e)!='fixed coordinate radius ceiling':raise
   record.update(status='retained_radius_proposal_bound',error=type(e).__name__+': '+str(e))
  h,streak=next_width(h,record['passed'],contraction,streak);record.update(next_halfwidth=ball_json(h),growth_streak=streak);w.save(tag+'/decision.json',record);attempts.append(record)
  if h<minimum:break
 result=dict(passed=bool(left==limit),attempts=attempts,accepted=len(accepted),last_parameter=ball_json(left),initial_parameter=ball_json(arb(1)/512),target_parameter=ball_json(limit),scope='Full51/native qualified prefix retained even if finite target remains incomplete; no step/radius quota expansion.');w.save('continuation.json',result);w.save('connected-endpoint.json',dict(parameter=ball_json(left),box=vector_json(known),new_qualified_charts=len(accepted)));need(result['passed'],'bounded finite extent incomplete');return accepted


def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);C.OUT=OUT;B.OUT=OUT;w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,600);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=384,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1'),python=platform.python_version(),numpy=C.np.__version__,scipy=scipy.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,caps=CAPS,settings=SETTINGS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<2*1024**3-4*1024**2,'complete source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(384):
   card=read(ROOT/CARD);oldcard=admission(w,card);known,mapdata=bind_endpoint(w,card);gain,arrays=numeric_endpoint(w,card);j,organizer,_,_=C.bind(w,oldcard);need(read(OUT/'input-binding.json')['map']==mapdata,'same exact independently proved map');charts=march(w,j,known);root,native,curvature,joins=C.endpoint(w,j,gain,arrays,organizer,charts);w.tick()
   result.update(passed=True,status='finite_organizer_connected_native6_localized_successor_with_generic_endpoint',new_charts=len(charts),prior_independent_charts=31,squared_amplitude_range=['0','1/16'],fixed_gain=ball_json(gain),endpoint_box=root['box'],endpoint_join_charts=[x['attempt']for x in joins if x['join']['passed']],primitive_period=6,critical_native_multiplier=1,complement_dimension=11,endpoint_leak_bb=curvature['leak_bb'],scope='Finite joined exact native6 critical extent; selected positive fixed-gain point separately has H_U inverse/physical curvature. No uniform genericity, full distant boundary, native12/24 or whole-window claim; independent replay required.')
 except BaseException as e:result.update(passed=False,status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=w.calls);result['passed']=bool(result['passed']and result['elapsed_seconds']<600 and result['peak_rss_kib']<2*1024**2);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
