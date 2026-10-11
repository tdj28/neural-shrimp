"""Independent saved reversal arithmetic and native-call identity; no model imports."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data2/neural-shrimp-data/audits/tiny-lm-ordering-independent/attempt001/run')
TARGET=Path('/data2/neural-shrimp-data/explore/tiny-lm-ordering/attempt001-target');PROFILE=TARGET.with_name('attempt001-profile')
RESPONSE=Path('/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-target')
EXEC='358a93ca3cc6434d5b8ea5c4989ad391fb471a0f'
PROFILE_EXEC='0e4aea5379fc37b7d9ab57e4fe9f07ad3bd49e98'
PINS={'target':'eb3f0c4503a73ac21443fe56b88a3834ec4cf13ae98fc98d53d3e40b4e8fc686','worker':'ce69af6c687cc66f6cc5118af7744734a8266cc233a6862e34e3d05ea60870ec','profile':'df8ac1cff4741812df9a6579d033be00f4000b6cf130673199371b2a699e9d50','profile_worker':'057ce1c4bd367fb018defa268b24b7b770810e17c2419ce45c526e3ae3d8ee03','response':'fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5','response_worker':'cb57135497afa72437651b7eaab807e58387a304fd0f0d2609113ffcff073aad'}
LEGACY=[(Path('/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-001-profile'),'ce56de837203182187912654b9ac0200096366a15d850ed7e917fb903837d2cd'),(Path('/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-002-collection'),'2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d')]
AUDITS=[(Path('/data2/neural-shrimp-data/audits/tiny-lm-response-independent/attempt001'),'296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948'),(Path('/data2/neural-shrimp-data/audits/tiny-lm-two-snapshot-independent/attempt001'),'7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d')]
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/review_tiny_lm_ordering.py','tools/launch_tiny_lm_ordering_review.py','tests/test_tiny_lm_ordering_review.py','docs/plans/tiny-lm-ordering-independent.md']
def need(v,msg):
 if not v:raise ValueError(msg)
def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def norm(v):return math.sqrt(dot(v,v))
def exact(a,b,message):need(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)
def array(p,shape):
 a=np.load(p,allow_pickle=False);need(a.dtype==np.float64 and a.shape==shape and np.isfinite(a).all(),'complete finite binary64 '+str(p));return a

def reduce(u,p,r,f,g):
 need(all(x.ndim==1 and x.shape==u.shape and np.isfinite(x).all()for x in(u,p,r,f,g)),'same finite vectors')
 odd=np.asarray([(float(x)-float(y))/2 for x,y in zip(p,r,strict=True)]);even=np.asarray([(float(x)+float(y))/2-float(z)for x,y,z in zip(p,r,f,strict=True)]);err=p-f
 aa,bb,ee=dot(odd,odd),dot(even,even),dot(err,err);ab=dot(odd,even);na,nb,ne=math.sqrt(aa),math.sqrt(bb),math.sqrt(ee);base=max(norm(p),norm(r),norm(f),1e-12)
 residue=norm(err-odd-even);squared=abs(ee-(aa+bb+2*ab))/max(ee,aa+bb,base*base);skew=abs(dot(u,odd))/max(norm(u)*max(norm(p),norm(r)),1e-12)
 out=dict(odd_norm=na,even_remainder_norm=nb,frozen_error_norm=ne,odd_to_error=na/max(ne,1e-12),even_to_error=nb/max(ne,1e-12),odd_even_inner_product=ab,squared_error=ee,squared_odd=aa,squared_even=bb,twice_cross_term=2*ab,identity_scale=base,vector_identity_residual=residue,squared_identity_scaled_error=squared,skew_quadratic_relative_error=skew,odd_even_cosine=None if na*nb==0 else ab/(na*nb),algebra_qualified=bool(residue/base<=1e-12 and squared<=1e-12 and skew<=1e-8),ce_projections={k:dot(g,v)for k,v in [('ordered',p),('reversed',r),('frozen',f),('odd',odd),('even_remainder',even),('error',err)]})
 return out,odd,even

def reciprocity(u,p,r):
 need(u.shape==p.shape==r.shape and u.ndim==2 and u.shape[0]==2,'two exact directions')
 rows=[]
 for i in range(2):
  for j in range(2):
   a,b=dot(u[i],p[j]),dot(r[i],u[j]);scale=max(norm(u[i])*norm(p[j]),norm(r[i])*norm(u[j]),1e-12);rows.append(dict(i=i,j=j,lhs=a,rhs=b,relative_error=abs(a-b)/scale))
 return dict(passed=all(x['relative_error']<=1e-8 for x in rows),rows=rows)

def compare(a,b,key=''):
 if isinstance(a,dict):
  need(set(a)<=set(b),'all scalar reduction fields')
  for k,v in a.items():compare(v,b[k],k)
 elif isinstance(a,list):
  need(len(a)==len(b),'all fixed rows')
  for x,y in zip(a,b,strict=True):compare(x,y,key)
 elif isinstance(a,bool)or a is None or isinstance(a,str):need(a==b,'exact reported gate '+key)
 else:
  tol=5e-15 if key in ('vector_identity_residual','squared_identity_scaled_error','skew_quadratic_relative_error','relative_error')else 1e-20
  need(math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,rel_tol=2e-11,abs_tol=tol),'independent fsum scalar '+key)

def inventory_rows(value):
 if isinstance(value,list):
  need(all(isinstance(v,dict)and set(v)=={'path','bytes','sha256'}for v in value),'exact historical manifest schema')
  rows={v['path']:{k:v[k]for k in ('bytes','sha256')}for v in value};need(len(rows)==len(value),'no duplicate historical entries');return rows
 need(isinstance(value,dict),'current manifest schema');return value

def inventory(root,pin,tick):
 need(root.is_dir()and not root.is_symlink()and((root,pin)in LEGACY or not root.stat().st_mode&0o222),'immutable or exact qualified historical container');need(entry(root/'manifest.json')['sha256']==pin and not(root/'manifest.json').stat().st_mode&0o222,'literal immutable manifest')
 rows=inventory_rows(read(root/'manifest.json')['files']);actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(rows)==actual,'complete saved inventory')
 for n,v in rows.items():
  tick();p=root/n;need(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==v,'immutable retained file '+n)
 return dict(root=str(root),manifest=pin,files=len(rows))

def source(run,execution):
 need(execution in (EXEC,PROFILE_EXEC),'one exact original execution')
 rt=read(run/'runtime.json');need(rt['source_commit']==execution,'fixed original execution')
 for n,v in rt['sources'].items():need(entry(run/'source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'complete original executed source '+n)
 return rt

def audit(tick,save):
 pairs=[(TARGET,PINS['target']),(TARGET/'run',PINS['worker']),(PROFILE,PINS['profile']),(PROFILE/'run',PINS['profile_worker']),(RESPONSE,PINS['response']),(RESPONSE/'run',PINS['response_worker'])]+LEGACY+AUDITS
 inv=[inventory(p,h,tick)for p,h in pairs];run=TARGET/'run';rt=source(run,EXEC);pr=source(PROFILE/'run',PROFILE_EXEC);out=read(run/'result.json');pro=read(PROFILE/'run/result.json');events=read(run/'events.json')
 need(out['passed']and out['status']=='ordering_complete'and out['protected_outcomes_evaluated']is False and out['mode']=='target'and out['operation_caps']==out['operation_attempted']==out['operation_returned']==dict(hvp=160),'exact target status/counts')
 for root,runtime,secs,execution in ((TARGET,rt,200,EXEC),(PROFILE,pr,80,PROFILE_EXEC)):
  la=read(root/'launcher-result.json');need(la['source_commit']==execution and la['exit_code']==0 and la['failure']is None and not la['owned_live_pids']and la['worker_pid_absent']and la['elapsed_seconds']<secs+5 and la['maximum_group_rss_kib']<=4*1024**2,'complete bounded owned source')
  for pid in (la['launcher_pid'],la['owned_group_pid'],runtime['pid']):need(not Path('/proc',str(pid)).exists(),'actual owned producer PIDs absent')
 need(pr['sources']==rt['sources']and all(pr[k]==rt[k]for k in ('python','numpy','torch','dtype','device','threads')),'identical measured source/runtime')
 need(pro['passed']and pro['status']=='profile_complete'and pro['operation_caps']==pro['operation_attempted']==pro['operation_returned']==dict(hvp=4),'complete profile actions')
 costs=pro['profile_costs'];need([r['arm']for r in costs]==[8,256]and all(len(r['times'])==2 and all(math.isfinite(t)and t>0 for t in r['times'])for r in costs),'both profile arms')
 forecast=20+sum(160*max(r['times'])for r in costs);compare(forecast,pro['forecast_seconds']);need(forecast<180 and read(run/'profile-binding.json')==dict(root=str(PROFILE),outer=PINS['profile'],worker=PINS['profile_worker'],forecast_seconds=pro['forecast_seconds']),'actual resource admission')
 pe=read(PROFILE/'run/events.json');need(len(pe)==4 and [r['index']for r in pe]==list(range(1,5))and all(r['operation']=='hvp'and r['returned']for r in pe),'complete profile event order');need([r['seconds']for r in pe]==sum((r['times']for r in costs),[]),'actual profile operation timings')
 for k in range(1,5):
  prefix=PROFILE/'run/operations'/f'hvp-{k:03d}';need(read(Path(str(prefix)+'-attempt.json'))==dict(operation='hvp',index=k,returned=False)and read(Path(str(prefix)+'-result.json'))==pe[k-1],'actual profile attempts/returns')
  for tag in ('theta','vector','return'):array(Path(str(prefix)+'-'+tag+'.npy'),(15776,))
 for root,pin in AUDITS:need(read(root/'run/result.json')['passed'],'prior independent original response authority')
 original=read(RESPONSE/'run/result.json');need(original['passed']and original['status']=='response_complete'and original['operation_attempted']==original['operation_returned']==dict(gradient=580,hvp=256,ce=96),'complete original response')
 need(all(rt['sources'][n]==v for n,v in original['source_identity'].items()),'same exact source-qualified native original');need(original['numpy']==rt['numpy']and original['torch']==rt['torch'],'same original native libraries')
 for root,(parent,pin)in ((PROFILE,LEGACY[0]),(TARGET,LEGACY[1])):
  b=read(root/'run/parent-binding.json');need(b['root']==str(parent)and b['manifest']==pin,'exact original data parent')
 count=0;rows=[];checks=[]
 for seed in (28161,28162):
  for arm in (8,256):
   label=f'seed-{seed}-arm-{arm}';old=RESPONSE/'run'/label;md=read(old/'inputs.json');need(read(run/label/'inputs.json')==md and(md['seed'],md['arm'],md['step'])==(seed,arm,600),'same actual state/data/rate identity');eta=md['eta'];need(math.isfinite(eta)and eta>0,'finite positive physical learning rate')
   ref=array(old/'reference.npy',(17,15776));grads=array(old/'reference-gradients.npy',(17,15776));dirs=array(old/'directions.npy',(2,15776));exact(ref[1:],ref[:-1]-eta*grads[:-1],'entire original physical GD recurrence')
   state=Path(md['state_chunk']);need(entry(state)==md['state_chunk_identity'],'original stored state identity');s=np.load(state,allow_pickle=False);exact(ref[0],s[0].astype(np.float64),'literal promotion of original600 state');need(hashlib.sha256(s[0].tobytes()).hexdigest()==md['state_float32_sha256'],'same original float32 state')
   need(abs(norm(dirs[0])-1)<1e-12 and abs(norm(dirs[1])-1)<1e-12,'both original unit directions');exact(dirs[0],grads[0]/np.linalg.norm(grads[0]),'actual inherited gradient direction')
   ordered=[array(old/f'direction-{i}-transported.npy',(17,15776))for i in range(2)];frozen=[array(old/f'direction-{i}-frozen.npy',(17,15776))for i in range(2)];need(all(np.array_equal(ordered[i][0],dirs[i])and np.array_equal(frozen[i][0],dirs[i])for i in range(2)),'same original ordered/frozen initial directions');oldout=read(old/'outcome.json');q=read(run/label/'qualification.json');rev={}
   for h in (4,16):
    for i in range(2):
     where=label+f'/direction-{i}-horizon-{h}';folder=run/where;trace=array(folder/'reverse-trace.npy',(h+1,15776));exact(trace[0],dirs[i],'same reverse initial vector');first=count+1
     for j,t in enumerate(range(h-1,-1,-1)):
      count+=1;prefix=run/'operations'/f'hvp-{count:03d}';theta=array(Path(str(prefix)+'-theta.npy'),(15776,));v=array(Path(str(prefix)+'-vector.npy'),(15776,));a=array(Path(str(prefix)+'-return.npy'),(15776,));exact(theta,ref[t],'literal current-state reversed Hessian');exact(v,trace[j],'literal current reversed vector');exact(trace[j+1],v-eta*a,'entire reversed physical action recurrence')
      need(read(Path(str(prefix)+'-attempt.json'))==dict(operation='hvp',index=count,returned=False),'pre-call input authority');e=read(Path(str(prefix)+'-result.json'));need(e==events[count-1]and e['index']==count and e['operation']=='hvp'and e['returned']and math.isfinite(e['seconds'])and e['seconds']>0,'actual ordered returned operation');tick()
     need(read(folder/'state-order.json')==dict(indices=list(range(h-1,-1,-1)),operation_first=first,operation_last=count),'exact reverse state order');rev[i,h]=trace[-1]
     g=grads[h]-.001*ref[h];exact(array(folder/'ce-gradient.npy',(15776,)),g,'CE-only terminal gradient, ridge removed');answer,a,b=reduce(dirs[i],ordered[i][h],trace[-1],frozen[i][h],g);exact(array(folder/'odd.npy',(15776,)),a,'saved odd vector');exact(array(folder/'even-remainder.npy',(15776,)),b,'saved even vector')
     row=read(folder/'outcome.json');need(row==out['rows'][len(rows)]and(row['seed'],row['arm'],row['direction'],row['horizon'],row['label'],row['operation_first'],row['operation_last'])==(seed,arm,i,h,where,first,count),'all literal output identities and ordering');compare(answer,row['decomposition'])
     qualification=next(v for v in oldout['rows']if v['direction']==('gradient'if i==0 else'random')and v['horizon']==h);need(row['response']==q[f'{i}:{h}']==qualification,'unchanged previously independently audited response qualification')
     need(answer['algebra_qualified'],'independent decomposition/skew gates');fresh=dict(seed=seed,arm=arm,direction=i,horizon=h,qualified=qualification['mechanism_qualified'],decomposition=answer);rows.append(fresh);save(f'row-{len(rows):02d}.json',fresh)
    chk=reciprocity(dirs,np.asarray([ordered[i][h]for i in range(2)]),np.asarray([rev[i,h]for i in range(2)]));compare(chk,read(run/label/f'reciprocity-{h}.json'));compare(dict(seed=seed,arm=arm,horizon=h,**chk),out['reciprocity'][len(checks)]);need(chk['passed'],'independent reciprocal products');checks.append(dict(seed=seed,arm=arm,horizon=h,**chk))
 need(count==len(events)==160 and len(rows)==len(out['rows'])==16 and len(checks)==len(out['reciprocity'])==8,'complete native/case exhaustion');need(len(list((run/'operations').glob('*-attempt.json')))==len(list((run/'operations').glob('*-result.json')))==160,'no hidden extra native operation')
 selected=[r for r in rows if r['arm']==256 and r['direction']==0 and r['horizon']==16];summary=dict(material_threshold=.5,material_reversal_odd_in_both_expanded_gradients=all(r['qualified']and r['decomposition']['odd_to_error']>=.5 for r in selected),selected_rows=[dict(seed=r['seed'],qualified=r['qualified'],odd_to_error=r['decomposition']['odd_to_error'])for r in selected]);compare(summary,out['summary']);save('input-binding.json',dict(inventories=inv,execution=EXEC,profile_execution=PROFILE_EXEC,pins=PINS));save('reciprocity.json',checks)
 return dict(passed=True,native_calls=0,rows=16,retained_hvp_checked=160,summary=summary,maximum_reciprocal_error=max(v['relative_error']for c in checks for v in c['rows']),scope='Independent fsum reduction and exact retained reverse-state/action audit; native Hessians and prior qualified response derivatives remain shared. Consumed four checkpoints only. Even remainder can depend on order; ratios are not percentages; no training, forecasting or operator amplification claim.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB saved audit')
 def stop(*_):raise TimeoutError('owned saved audit deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def write(name,data,closing=False):
  nonlocal used
  need(used+len(data)<(4 if closing else 3)*1024**2,'4MiB with1MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used+=len(data)
 def save(name,value,closing=False):write(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source_commit,pid=os.getpid(),python=sys.version,numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),sources={n:entry(ROOT/n)for n in FILES}))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source_commit+':'+n],cwd=ROOT),'complete committed own source');write('source/'+n,data)
  t=time.monotonic();u=np.arange(15776,dtype=float)+1;u=u/norm(u);a,_,_=reduce(u,u*.8,u*.8,u*.7,u);q=reciprocity(np.array([u,u]),np.array([u*.8,u*.8]),np.array([u*.8,u*.8]));seconds=time.monotonic()-t;forecast=32*seconds+20;save('profile.json',dict(passed=a['algebra_qualified']and q['passed']and forecast<60,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(a['algebra_qualified']and q['passed']and forecast<60,'full-size measured scalar audit gate');result=audit(tick,save)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
