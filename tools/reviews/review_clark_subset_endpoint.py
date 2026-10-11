"""Independent saved successor replay and known-root endpoint localization."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
import review_clark_correlated as D
import review_clark_endpoint_math as E
A,H,M,need=D.A,D.H,D.M,D.need
ROOT=D.ROOT
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-subset-successor/attempt001')
SOURCE='cf1cd08e7a868e0dd948c6bd4b188c99f9a53767'
PINS=['8443f4e777c5ebfa5c4a0659ba90dbc8ebb19b84d44cc0932ed06946dbbd552a','53bbddefb9ec52767b8a48a17de98b0e59326a9859bb874473d8a408a16d0fca']
OUT=Path('/data2/neural-shrimp-data/audits/clark-subset-endpoint-independent/attempt003/run')
THREADS=D.THREADS;read,entry,serial,covered=D.read,D.entry,D.serial,D.covered
EXPECTED={'native_even_odd_phase': 51, 'explicit_organizer_anchor': 1, 'explicit_family_attempt': 16, 'explicit_amplitude_fold_return': 16, 'correlated_center_newton_proposal': 10, 'path_attempt': 25, 'clark_gradient_path_input': 75, 'clark_gradient_path_return': 75, 'clark_correlated_path_return': 25, 'quadratic_predictor_proposal': 5, 'correlated_radius_proposal': 5, 'cached_residual_reuse': 10, 'correlated_explicit_tube_return': 5, 'context_attempt': 1, 'paired_native_phase': 6, 'validated_inverse': 4, 'paired_native_result': 1}
KNOWN={'endpoint': {'bytes': 14972, 'sha256': '149c9884e79635252b11c796c74396af777d948715baabb68d549a8cc0a2401f'}, 'outer': '09aa30ca1399a667217f0be1c09ede0b2a5097f875fd0ebb3acad61699bcba81', 'root': '/data2/neural-shrimp-data/audits/clark-correlated-extent-independent/attempt001', 'source': 'f79b5ad059d08b88cd2aa8c360b538fe7b5c284c', 'worker': '9ecd7edbe7bc802042bbee4f625621a82fa4b64c57a9fae3cd5f7e9705c2a5ea'}
FAILED=dict(root='/data2/neural-shrimp-data/audits/clark-subset-endpoint-independent/attempt001',source='c753b2d04926b2d5a61b66f8c07d3e756e82c4a4',outer='91aaa6a4822a9b44f0cdce6a37c6b77e2b6cd43fba8b2e72a46443bfaeea7503',worker='79dc5b8d9147350306164effb0756efdf6943865ad03e8068d5c6df204669b0c')
FAILED_SERIAL=dict(root='/data2/neural-shrimp-data/audits/clark-subset-endpoint-independent/attempt002',source='86d7816f566f93183ff968f4b01bafaa159fdb46',outer='67bec499f4b4dcff31e153c863eefc89c68ec655ae99b6792bb4f18816a125d7',worker='e0265a7d4cb7ea58fb113bbba2b4efb7eafd244516c19435233716c158f3691d')
CARD='docs/plans/clark-subset-endpoint-independent-inputs.json'
FILES=sorted(set(D.FILES+[CARD,'tools/reviews/review_clark_subset_endpoint.py','tools/reviews/review_clark_endpoint_math.py','tools/launch_clark_subset_endpoint_review.py','tests/reviews/test_clark_subset_endpoint_driver.py','tests/reviews/test_clark_endpoint_math.py','docs/plans/clark-subset-endpoint-independent.md','docs/reviews/clark-subset-endpoint-independent-attempt002.json','docs/reviews/clark-subset-endpoint-independent-attempt001.json','docs/reviews/clark-subset-successor-attempt001.json','docs/reviews/clark-correlated-extent-independent.json']))
class Work(D.Work):
 def tick(self):need(not self.stopped and time.monotonic()-self.start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'120s/2GiB independent extent')
 def save(self,n,v,closing=False):
  raw=(json.dumps(serial(v),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
  if n.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(64 if closing else 60)*1024**2,'64MiB4MiB final reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.tick()

def metadata(w):
 card=read(ROOT/CARD);need(card['status']=='reviewed'and card['caps']==dict(seconds=120,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=64*1024**2,native_calls=0,precision=512,intersections=16)and card['parent']==dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1])and card['known']==KNOWN and card['failed']==FAILED and card['failed_serialization']==FAILED_SERIAL,'fixed reviewed literal source/domain authority')
 fp=Path(FAILED['root']);D.inventory(fp,FAILED['outer'],w);D.inventory(fp/'run',FAILED['worker'],w);D.archive(fp,FAILED['source'],2,w);fo=read(fp/'run/outcome.json');need(not fo['passed']and fo['native_calls']==0 and fo['error']=='ValueError: unchanged actual coefficient/domain residual_coefficients','literal retained001 binding failure')
 sp=Path(FAILED_SERIAL['root']);D.inventory(sp,FAILED_SERIAL['outer'],w);D.inventory(sp/'run',FAILED_SERIAL['worker'],w);D.archive(sp,FAILED_SERIAL['source'],2,w);so=read(sp/'run/outcome.json');need(not so['passed']and so['native_calls']==0 and so['error'].startswith('ValueError: Exceeds the limit (4300 digits)'),'literal retained002 serialization failure')
 for p,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):D.inventory(p,pin,w)
 D.archive(PARENT,SOURCE,2,w);run=PARENT/'run';out=read(run/'outcome.json');cont=read(run/'continuation.json')
 need(not out['passed']and out['error']=='ValueError: bounded finite extent incomplete'and out['calls']==dict(algebra=22,anchor=1,cache_reuses=10,context=1,events=331,explicit=16,implicit=0,native_phase_returns=132,path=25,subset=6),'complete exact bounded successor')
 need(not cont['passed']and len(cont['attempts'])==5 and cont['accepted']==1 and [x['attempt']for x in cont['attempts']if x['passed']]==[1]and A.point(cont['last_parameter'])==F(1,512),'one selected/four preserved failed attempts')
 paths=sorted((run/'events').iterdir());kinds=[p.name.split('-',1)[1].split('.json')[0]for p in paths];need(len(paths)==331 and [int(p.name[:4])for p in paths]==list(range(1,332))and Counter(kinds)==EXPECTED,'complete all331 events/132 native phases')
 original=read(run/'source/docs/plans/clark-subset-successor-inputs.json');need(original['known']==KNOWN,'literal full51 independent endpoint ancestry')
 p=Path(KNOWN['root'])
 for q,pin in zip((p,p/'run'),(KNOWN['outer'],KNOWN['worker']),strict=True):D.inventory(q,pin,w)
 D.archive(p,KNOWN['source'],0,w);prior=read(p/'run/outcome.json');ep=read(p/'run/connected-endpoint.json');ib=read(run/'input-binding.json');auditib=read(p/'run/input-binding.json')
 need(prior['passed']and prior['native_calls']==0 and prior['range']==['1/2048','15/8192']and prior['new_independent_tubes']==22 and prior['prior_independent_tubes']==8 and prior['new_native_contexts']==88 and entry(p/'run/connected-endpoint.json')==KNOWN['endpoint']and A.point(ep['parameter'])==F(15,8192)and len(ep['box'])==51,'entire independently known phase/root box')
 need(auditib['map']==ib['map']and auditib['parent']==original['prior']['root']and auditib['source']==original['prior']['source']and auditib['pins']==[original['prior']['outer'],original['prior']['worker']],'independent exact original map genealogy')
 for key,exitcode in [('prior',2),('screen',0)]:
  spec=original[key];q=Path(spec['root']);D.inventory(q,spec['outer'],w);D.inventory(q/'run',spec['worker'],w);D.archive(q,spec['source'],exitcode,w)
 need(read(run/'known-endpoint-binding.json')['endpoint']==ep and read(run/'known-endpoint-binding.json')['map']==ib['map']and entry(Path(ib['card']['matrix']['path']))==ib['card']['matrix']['entry'],'literal all51 initial box and unchanged fixed matrix')
 w.save('input-binding.json',dict(parent=str(PARENT),source=SOURCE,pins=PINS,known=KNOWN,map=ib['map'],events=EXPECTED,scope='Independent saved coefficient/root/native reduction and exact endpoint restriction; all original histories retained. Native analytic derivatives and Arb shared.'))
 return paths,kinds,[M.decode(x)for x in ep['box']],ib

def selected_domain(original,selected,reencoded=True):
 for key in('center','slope','curvature','parameter','halfwidth','inverse_coefficients','actual_error'):need(selected[key]==original[key],'unchanged actual coefficient/domain '+key)
 for key in('residual_coefficients','jacobian_coefficients'):
  need(len(selected[key])==len(original[key])==(3 if key=='jacobian_coefficients'else 4),'complete coefficient degrees')
  with ctx.workprec(384):
   for oldmat,newmat in zip(original[key],selected[key],strict=True):
    need(len(oldmat)==len(newmat)==51,'all coefficient rows')
    for oldrow,newrow in zip(oldmat,newmat,strict=True):
     need(len(oldrow)==len(newrow)==(51 if key=='jacobian_coefficients'else 1),'all coefficient columns')
     for x,y in zip(oldrow,newrow,strict=True):
      expected=serial(M.decode(x))if reencoded else x
      need(A.interval(expected)==A.interval(y),'exact384 decode-reserialize '+key)
 old=[A.point(x)for x in original['radii']];new=[A.point(x)for x in selected['radii']];need(len(old)==len(new)==51 and all(0<v<=r for v,r in zip(new,old,strict=True)),'positive componentwise subset only')
 return new

def target(w):
 paths,kinds,known,ib=metadata(w);matrix=read(paths[kinds.index('path_attempt')])['matrix'];need([[A.point(x)for x in row]for row in matrix]==[[F(x)for x in row]for row in ib['map']['j0']],'same exact matrix')
 recent=[];left=arb(15)/8192;accepted=[];attempts=[]
 for i,kind in enumerate(kinds):
  if kind=='path_attempt':recent.append(i)
  if kind!='correlated_explicit_tube_return':continue
  attempt=len(attempts)+1;tag=f'charts/{attempt:02d}';t=read(paths[i]);need(t==read(PARENT/f'run/charts/{attempt:02d}/tube-original.json.gz')and len(recent)>=5,'literal complete retained tube');c,s,a,r=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature','radii')];b=M.decode(t['parameter']);h=M.decode(t['halfwidth']);need(all(x.is_exact()for x in c+s+a+r+[b,h])and len(c)==len(s)==len(a)==len(r)==51,'complete exact curved geometry');found={}
  for k in recent[-5:]:
   req=read(paths[k])
   if any(req[x]!=t[y]for x,y in [('center','center'),('slope','slope'),('curvature','curvature'),('parameter','parameter')]):continue
   zero=all(A.point(v)==0 for v in req['error'])if all(A.interval(v)[0]==A.interval(v)[1]for v in req['error'])else False;center=A.interval(req['d'])==(F(0),F(0));key=(zero,center);need(key not in found,'unique actual proof acquisition')
   if not zero:need(all(A.covers(v,(-A.point(z),A.point(z)))for v,z in zip(req['error'],t['radii'],strict=True)),'same full actual error domain')
   if not center:need(A.covers(req['d'],(-A.point(t['halfwidth']),A.point(t['halfwidth']))),'whole acquired parameter covers original interval')
   found[key]=D.path_group(paths,k,matrix);w.tick()
  need(set(found)=={(True,True),(True,False),(False,True),(False,False)},'four complete residual/fullE acquisitions');need(found[False,True][0]['error']==found[False,False][0]['error']==t['actual_error'],'same literal fullE in center/whole Jacobian acquisitions');fc=found[True,True][1];fw=found[True,False][1];J=[found[False,True][2][0],found[False,True][2][1],found[False,False][2][2]];residual_coefficients=[M.col([fc[k,j]if j<3 else fw[k,3]for k in range(51)])for j in range(4)];C=[M.matrix(v)for v in t['inverse_coefficients']];original_data=H.reduction(C,J,residual_coefficients,r,h);w.save(tag+'/original-root.json.gz',original_data);selected=read(PARENT/f'run/charts/{attempt:02d}/tube.json.gz');decision=read(PARENT/f'run/charts/{attempt:02d}/decision.json')
  selected_domain(t,selected,decision['subset_fallback'])
  sr=[M.decode(x)for x in selected['radii']];need(all(v.is_exact()and 0<v<=old for v,old in zip(sr,r,strict=True)),'positive componentwise subset only');data=H.reduction(C,J,residual_coefficients,sr,h);w.save(tag+'/root.json.gz',data);record=dict(attempt=attempt,root_passed=data['passed'],contraction=data['contraction'],original_accepted=decision['passed'],original_contraction=original_data['contraction']);r=sr

  if decision['passed']:
   need(attempt==1 and selected['passed']and data['passed']and left==b-h,'single independently passed original successor')
   join=H.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 exact preceding phase/root identity');need(kinds[i+1]=='context_attempt','fresh native contexts after root');native=D.qualify_native(paths,i+1,selected,data,w,tag)
   ep=read(PARENT/'run/connected-endpoint.json');need(ep['box']==selected['endpoints'][1]and A.point(ep['parameter'])==F(1,512),'literal whole original endpoint')
   localized=E.localize(data,c,s,a,[M.decode(x)for x in ep['box']],iterations=16);w.save('endpoint-localization.json.gz',exact_serial(localized));known=localized['endpoint'];left=b+h;w.save('connected-endpoint.json',dict(parameter=left,box=known,scope='Same independently proved root, fixed-parameter contraction on original E.'));accepted.append(dict(attempt=attempt,contraction=data['contraction'],native_contexts=len(native)));record['localization_rows']=exact_serial(localized['rows'])
  else:
   need(attempt in(2,3,4,5)and not selected['passed']and selected==t,'all four unresolved original bounds retained')
   # These are anchor requirements only; no failed chart is promoted and no LP is rerun.
   oldknown=read(PARENT/'run/connected-endpoint.json')['box'];delta=left-b;old_error=[];new_error=[]
   for k in range(51):
    p=c[k]+s[k]*delta+a[k]*delta*delta/2
    old_error.append(abs(M.decode(oldknown[k])-p).upper()/r[k]);new_error.append(abs(known[k]-p).upper()/r[k])
   record['endpoint_anchor_diagnosis']=dict(old_normalized_errors=old_error,new_normalized_errors=new_error,max_old=max(old_error),max_new=max(new_error),scope='Only fixed known endpoint changed; actual old fullE/J envelope is unchanged. No new tube, LP infeasibility, or smaller-domain Jacobian claim.')
  attempts.append(record);w.save(tag+'/result.json',record)
 need(len(attempts)==5 and len(accepted)==1 and left==arb(1)/512,'complete fixed five-history audit')
 return dict(passed=True,status='independent_one_native6_tube_and_exact_endpoint_localization',range=['15/8192','1/512'],prior_independent_tubes=30,new_independent_tubes=1,new_native_contexts=4,attempts=attempts,accepted=accepted,full_requested_range_completed=False,scope='Shared retained native analytic derivative acquisitions and Arb; separate full51 assembly/root/join/native reductions. Four failed successor bounds stay failed; endpoint restriction adds no new existence theorem.')

def exact_serial(v):
 if isinstance(v,F):
  need(v.denominator&(v.denominator-1)==0 and max(abs(v.numerator).bit_length(),v.denominator.bit_length())<=131072,'bounded exact dyadic serialization')
  return dict(mantissa_hex=format(v.numerator,'x'),exponent=-(v.denominator.bit_length()-1))
 if isinstance(v,dict):return {k:exact_serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [exact_serial(x)for x in v]
 return v

def synthetic(w):
 begin=time.monotonic();n=51;C=[M.eye(n),arb_mat(n,n)];J=[M.eye(n),arb_mat(n,n),arb_mat(n,n)];J[0][0,0]=arb(3)/4;R=[M.col([arb(1)/1024]*n),M.col([arb(1)/8]*n),arb_mat(n,1),arb_mat(n,1)];data=H.reduction(C,J,R,[arb(1)]*n,arb(1)/8);need(data['passed'],'full51 synthetic reduction');z=[arb(0)]*n;loc=E.localize(data,z,z,z,[data['image'][i,0]for i in range(n)]);w.save('profile-root.json.gz',data);w.save('profile-localization.json.gz',exact_serial(loc));root_seconds=time.monotonic()-begin
 begin=time.monotonic();n=12;cycle=[[arb(i+k)for i in range(n)]for k in range(6)];block=arb_mat([[arb(int(i==k))*(arb(1)if i==0 else arb(1)/2)for k in range(n)]for i in range(n)]);out=H.native_context(cycle,[block]*6,[arb(int(i==0))for i in range(n)]);need(out['passed'],'full12 native synthetic reduction');w.save('profile-native.json.gz',out);native_seconds=time.monotonic()-begin
 p=Path(KNOWN['root']);D.inventory(p,KNOWN['outer'],w);D.inventory(p/'run',KNOWN['worker'],w);D.archive(p,KNOWN['source'],0,w);prior=read(p/'run/outcome.json');size=sum(v['bytes']for v in read(p/'run/manifest.json')['files'].values());cost=max(2*(10*root_seconds+4*native_seconds)+30,2*prior['elapsed_seconds']+20);raw=max(16*w.used+4*1024**2,size+4*1024**2);need(cost<120 and raw<60*1024**2,'full synthetic and complete prior actual cost floor')
 w.save('profile.json',dict(passed=True,root_and_localization_seconds=root_seconds,native_seconds=native_seconds,forecast_seconds=cost,forecast_bytes=raw,target_arrays_unopened=True,native_calls=0))

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean CPU1nice10 source');OUT.mkdir(parents=True,exist_ok=False);D.OUT=OUT;w=Work();out=dict(passed=False,status='incomplete');signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,120)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=512,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,native_call_cap=0,seconds=120,rss_bytes=2*1024**3,raw_bytes=64*1024**2))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed own source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):synthetic(w);out=target(w)
 except BaseException as e:out.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(native_calls=0,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out['passed']=bool(out['passed']and out['elapsed_seconds']<120 and out['peak_rss_kib']<2*1024**2);w.stopped=False;w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(serial(out)));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
