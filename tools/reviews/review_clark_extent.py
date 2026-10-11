"""Independent saved replay of the fixed32-attempt finite Clark51 extent."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
import review_clark_correlated as D
A,H,M,need=D.A,D.H,D.M,D.need
ROOT=D.ROOT
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extent/attempt001')
SOURCE='b1fc38c14906d6c2a6ef16fab7e82bbbe2f62676'
PINS=['71e8f0e318d1427950e63aba0466efe213808c4336788277843919fbc48875dd','2812d71cd15953a3cc19772fdd5b488536ec885f8819bf50267adbb1b4cea4ab']
OUT=Path('/data2/neural-shrimp-data/audits/clark-correlated-extent-independent/attempt001/run')
THREADS=D.THREADS;read,entry,serial,covered=D.read,D.entry,D.serial,D.covered
EXPECTED={'clark_gradient_path_input':480,'path_attempt':160,'clark_correlated_path_return':160,'native_even_odd_phase':294,'explicit_amplitude_fold_return':97,'clark_gradient_path_return':480,'explicit_family_attempt':97,'paired_native_result':22,'clark_correlated_path_return':160,'validated_inverse':88,'cached_residual_reuse':64,'correlated_center_newton_proposal':64,'context_attempt':22,'paired_native_phase':132,'correlated_radius_proposal':32,'quadratic_predictor_proposal':32,'correlated_explicit_tube_return':32,'explicit_organizer_anchor':1}
CARD='docs/plans/clark-extent-independent-inputs.json'
FILES=sorted(set(D.FILES+[CARD,'tools/reviews/review_clark_extent.py','tools/launch_clark_extent_review.py','tests/reviews/test_clark_extent_driver.py','docs/plans/clark-extent-independent.md','docs/reviews/clark-correlated-extent-outcome.json','docs/reviews/clark-correlated-extension-authority.json']))
class Work(D.Work):
 def tick(self):need(not self.stopped and time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'300s/2GiB independent extent')
 def save(self,n,v,closing=False):
  raw=(json.dumps(serial(v),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
  if n.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(256 if closing else 252)*1024**2,'256MiB4MiB final reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.tick()

def metadata(w):
 card=read(ROOT/CARD);need(card['status']=='reviewed'and card['caps']==dict(seconds=300,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=256*1024**2,native_calls=0,precision=512)and card['parent']==dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1]),'fixed reviewed extent authority')
 for p,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):D.inventory(p,pin,w)
 D.archive(PARENT,SOURCE,2,w);run=PARENT/'run';out=read(run/'outcome.json');cont=read(run/'continuation.json')
 need(not out['passed']and out['error']=='ValueError: bounded finite extent incomplete'and out['calls']==dict(algebra=96,anchor=1,cache_reuses=64,context=22,events=2257,explicit=97,implicit=0,native_phase_returns=906,path=160),'complete original bounded native unit')
 accepted=[x['attempt']for x in cont['attempts']if x['passed']];need(not cont['passed']and len(cont['attempts'])==32 and cont['accepted']==22 and accepted==[i for i in range(1,33)if i%3!=0]and A.point(cont['last_parameter'])==F(15,8192),'all22 selected prefixes/ten retained wider failures')
 paths=sorted((run/'events').iterdir());kinds=[p.name.split('-',1)[1].split('.json')[0]for p in paths];need(len(paths)==2257 and [int(p.name[:4])for p in paths]==list(range(1,2258))and Counter(kinds)==EXPECTED,'all2257 events/full906 native phases')
 original=read(run/'source/docs/plans/clark-correlated-extent-inputs.json');spec=original['independent_extension'];p=Path(spec['root'])
 need(spec==card['independent_extension'],'literal independent starting authority')
 for q,pin in zip((p,p/'run'),(spec['outer'],spec['worker']),strict=True):D.inventory(q,pin,w)
 D.archive(p,spec['source'],0,w);prior=read(p/'run/outcome.json');ep=read(p/'run/connected-endpoint.json');ib=read(run/'input-binding.json');auditib=read(p/'run/input-binding.json');producer=original['producer_extension']
 need(prior['passed']and prior['native_calls']==0 and prior['range']==['0','1/2048']and prior['new_independent_tubes']==2 and prior['prior_independent_tubes']==6 and prior['endpoint_coordinate_inclusions']==51 and prior['endpoint_memberships']and entry(p/'run/connected-endpoint.json')==spec['endpoint']and A.point(ep['parameter'])==F(1,2048)and len(ep['box'])==51,'whole exact independently joined full51 initial endpoint')
 need(auditib['parent']==producer['root']and auditib['source']==producer['source']and auditib['pins']==[producer['outer'],producer['worker']]and auditib['map']==ib['map'],'exact prior same-map root genealogy')
 pp=Path(producer['root']);D.inventory(pp,producer['outer'],w);D.inventory(pp/'run',producer['worker'],w);D.archive(pp,producer['source'],0,w);oldib=read(pp/'run/input-binding.json');need(ib['map']==oldib['map']and ib['organizer_box']==oldib['organizer_box'],'same original normalized phase/organizer')
 need(read(run/'known-endpoint-binding.json')['endpoint']==ep and entry(Path(ib['card']['matrix']['path']))==ib['card']['matrix']['entry'],'actual caller starting box and fixed matrix')
 w.save('input-binding.json',dict(parent=str(PARENT),source=SOURCE,pins=PINS,independent_extension=spec,producer_extension=producer,map=ib['map'],events=EXPECTED,scope='Same exact independently joined51 endpoint; separate raw-native coefficient/inequality/native reductions. Native analytic derivatives and Arb shared.'))
 return paths,kinds,[M.decode(x)for x in ep['box']],ib

def target(w):
 paths,kinds,known,ib=metadata(w);matrix=read(paths[kinds.index('path_attempt')])['matrix'];need([[A.point(x)for x in row]for row in matrix]==[[F(x)for x in row]for row in ib['map']['j0']],'same exact matrix')
 recent=[];left=arb(1)/2048;accepted=[];attempts=[]
 for i,kind in enumerate(kinds):
  if kind=='path_attempt':recent.append(i)
  if kind!='correlated_explicit_tube_return':continue
  attempt=len(attempts)+1;tag=f'charts/{attempt:02d}';t=read(paths[i]);need(t==read(PARENT/f'run/charts/{attempt:02d}/tube.json.gz')and len(recent)>=5,'literal complete retained tube');c,s,a,r=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature','radii')];b=M.decode(t['parameter']);h=M.decode(t['halfwidth']);need(all(x.is_exact()for x in c+s+a+r+[b,h])and len(c)==len(s)==len(a)==len(r)==51,'complete exact curved geometry');found={}
  for k in recent[-5:]:
   req=read(paths[k])
   if any(req[x]!=t[y]for x,y in [('center','center'),('slope','slope'),('curvature','curvature'),('parameter','parameter')]):continue
   zero=all(A.point(v)==0 for v in req['error'])if all(A.interval(v)[0]==A.interval(v)[1]for v in req['error'])else False;center=A.interval(req['d'])==(F(0),F(0));key=(zero,center);need(key not in found,'unique actual proof acquisition')
   if not zero:need(all(A.covers(v,(-A.point(z),A.point(z)))for v,z in zip(req['error'],t['radii'],strict=True)),'same full actual error domain')
   if not center:need(A.covers(req['d'],(-A.point(t['halfwidth']),A.point(t['halfwidth']))),'whole acquired parameter covers original interval')
   found[key]=D.path_group(paths,k,matrix);w.tick()
  need(set(found)=={(True,True),(True,False),(False,True),(False,False)},'four complete residual/fullE acquisitions');need(found[False,True][0]['error']==found[False,False][0]['error']==t['actual_error'],'same literal fullE in center/whole Jacobian acquisitions');fc=found[True,True][1];fw=found[True,False][1];J=[found[False,True][2][0],found[False,True][2][1],found[False,False][2][2]];residual_coefficients=[M.col([fc[k,j]if j<3 else fw[k,3]for k in range(51)])for j in range(4)];C=[M.matrix(v)for v in t['inverse_coefficients']];data=H.reduction(C,J,residual_coefficients,r,h);w.save(tag+'/root.json.gz',data);decision=read(PARENT/f'run/charts/{attempt:02d}/decision.json');record=dict(attempt=attempt,root_passed=data['passed'],contraction=data['contraction'],original_accepted=decision['passed'])
  if decision['passed']:
   need(attempt in [i for i in range(1,33)if i%3!=0]and data['passed']and left==b-h,'ordered original accepted closed support');join=H.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 exact known phase/root join');need(kinds[i+1]=='context_attempt','native context immediately after accepted root');native=D.qualify_native(paths,i+1,t,data,w,tag);known=H.endpoint(data,c,s,a,1);left=b+h;w.save(tag+'/endpoint.json',dict(parameter=left,box=known));accepted.append(dict(attempt=attempt,parameter=left,contraction=data['contraction'],native_contexts=len(native)))
  attempts.append(record)

 need(len(attempts)==32 and len(accepted)==22 and left==arb(15)/8192,'complete fixed22 successor prefix')
 w.save('connected-endpoint.json',dict(parameter=left,box=known))
 return dict(passed=True,status='independent22_connected_native6_extent_tubes',range=['1/2048','15/8192'],prior_independent_tubes=8,new_independent_tubes=22,new_native_contexts=88,attempts=attempts,accepted=accepted,full_requested_range_completed=False,endpoint_generic_fold_proved=False,scope='All32 acquired histories retained;22 independently strict known-root/native6 critical tubes. Native analytical derivative acquisitions and Arb shared; no new generic endpoint or global window claim.')

def synthetic(w):
 begin=time.monotonic();n=51;C=[M.eye(n),arb_mat(n,n)];J=[M.eye(n),arb_mat(n,n),arb_mat(n,n)];R=[arb_mat(n,1)for _ in range(4)];data=H.reduction(C,J,R,[arb(1)/128]*n,arb(1)/64);need(data['passed'],'full51 strict synthetic reduction');w.save('profile-root.json.gz',data);root_seconds=time.monotonic()-begin
 begin=time.monotonic();n=12;cycle=[[arb(i+k)for i in range(n)]for k in range(6)];block=arb_mat([[arb(int(i==k))*(arb(1)if i==0 else arb(1)/2)for k in range(n)]for i in range(n)]);out=H.native_context(cycle,[block]*6,[arb(int(i==0))for i in range(n)]);need(out['passed'],'full12 native synthetic reduction');w.save('profile-native.json.gz',out);native_seconds=time.monotonic()-begin
 spec=read(ROOT/CARD)['independent_extension'];p=Path(spec['root']);D.inventory(p,spec['outer'],w);D.inventory(p/'run',spec['worker'],w);D.archive(p,spec['source'],0,w);old=read(p/'run/outcome.json');size=read(p/'run/manifest.json')['files'];cost=max(2*(32*root_seconds+88*native_seconds)+80,2*16*old['elapsed_seconds']+30);raw=max(64*w.used+8*1024**2,2*16*sum(x['bytes']for x in size.values())+8*1024**2);need(cost<300 and raw<252*1024**2,'full synthetic and entire actual extension floor')
 w.save('profile.json',dict(passed=True,root_seconds=root_seconds,native_seconds=native_seconds,forecast_seconds=cost,forecast_bytes=raw,native_calls=0,target_arrays_unopened=True))

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean CPU1nice10 source');OUT.mkdir(parents=True,exist_ok=False);D.OUT=OUT;w=Work();out=dict(passed=False,status='incomplete');signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=512,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,native_call_cap=0,seconds=300,rss_bytes=2*1024**3,raw_bytes=256*1024**2))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed own source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):synthetic(w);out=target(w)
 except BaseException as e:out.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(native_calls=0,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out['passed']=bool(out['passed']and out['elapsed_seconds']<300 and out['peak_rss_kib']<2*1024**2);w.stopped=False;w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(serial(out)));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
