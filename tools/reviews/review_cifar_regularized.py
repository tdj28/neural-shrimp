"""Independent saved-vector physical-metric regularization and actual merit audit."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
import gc,hashlib,json,math,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,svd
import review_cifar_current512 as A
M=A.M;need=M.need;ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-regularized-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/regularized001');EXEC='8254a7a9'
PINS=['9d271c9b4dcefff3d12f58a4eedbdf934622ff86ab87afda28391a50b2e3d3d6','ecff8945abbbba1613eda075eca6f34b84462cea3150841c82b43d4bb93935f7']
FILES=sorted(set(['tools/reviews/review_cifar_regularized.py','tools/reviews/launch_cifar_regularized_review.py','tests/reviews/test_cifar_regularized_review.py','docs/plans/cifar-regularized-independent.md','docs/reviews/cifar-regularized001.json']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def proposal(V,W,g,tick=lambda:None):
 need(V.ndim==2 and W.shape==V.shape and g.shape==(V.shape[0],)and all(np.isfinite(v).all()for v in[V,W,g]),'complete finite same-state inputs')
 gram=V.T@V;d,U=eigh(gram);need(d[0]>0 and d[-1]/d[0]<1e12,'positive physical basis metric');tick()
 # Symmetric spectral whitening, distinct from producer triangular Cholesky.
 S=(U/np.sqrt(d))@U.T;Z=W@S;L,s,Rt=svd(Z,full_matrices=False,lapack_driver='gesvd');tick();need(np.all(s>s[0]*1e-12),'complete action rank')
 z=L.T@g;r=g-L@z;ng=M.norm(g);perp=M.dot(r,r);ss=s*s
 def error(mu):return math.sqrt(perp+math.fsum(float(mu*v/(a+mu))**2 for a,v in zip(ss,z,strict=True)))/ng
 need(error(0)<.49,'attainable fixed target');lo,hi=0.,float(ss[0])
 for _ in range(64):
  if error(hi)>.49:break
  hi*=2
 else:raise ValueError('bounded independent bracket failed')
 for _ in range(80):
  mid=(lo+hi)/2
  if error(mid)<=.49:lo=mid
  else:hi=mid
 c=S@(-Rt.T@((s/(ss+lo))*z));p=V@c;hp=W@c;forcing=M.norm(g+hp)/ng;tick()
 return p,hp,dict(ridge=lo,forcing=forcing,direction_norm=M.norm(p),rank=len(s),gram_error=float(np.max(abs(gram-np.eye(len(d))))),minimum_forcing=error(0))

def events(es,inv):
 counts=dict(hvp=0,direct_gradient=0);calls={};trials=[];gate=None;pos=0
 def item(v):need(v['identity']==inv[v['file']]and Path(v['file']).name==v['file'],'literal retained native file');return v['file']
 while pos<len(es):
  e=es[pos]
  if e['kind']=='current_direction_gate':need(gate is None and counts==dict(hvp=1,direct_gradient=1),'gate after actual initial gradient and action');gate={k:v for k,v in e.items()if k!='kind'};pos+=1;continue
  if e['kind']=='current_merit_trial':
   need(gate is not None and e['index']==len(trials)and not any(t['accepted']for t in trials),'ordered first-acceptance trials');trials.append({k:v for k,v in e.items()if k!='kind'});need(counts['direct_gradient']==1+sum(t['within_trust']for t in trials),'one native gradient per eligible trial');pos+=1;continue
  b=es[pos:pos+4];op=e.get('operation');need(op in counts and len(b)==4 and [v['kind']for v in b]==['native_input','operation_started','operation_returned','native_output'],'full native input/start/return/output quartet');counts[op]+=1;i=counts[op];need(all(v['operation']==op and v['index']==i for v in b),'native ordered indices');need(math.isfinite(b[2]['seconds'])and b[2]['seconds']>=0,'finite native cost')
  if op=='hvp':need(gate is None and i==1 and counts['direct_gradient']==1,'one same-state actual candidate action')
  elif i==1:need(gate is None and counts['hvp']==0,'initial direct gradient before action')
  else:need(gate is not None and i==2+sum(t['within_trust']for t in trials),'new trial gradient before its decision')
  row=dict(state=item(e['state']),result=item(b[3]['result']))
  if op=='hvp':row['direction']=item(e['direction'])
  calls[op,i]=row;pos+=4
 need(gate is not None and counts['hvp']==1 and counts['direct_gradient']==1+sum(t['within_trust']for t in trials),'complete actual stream');return counts,calls,gate,trials

def bind(tick):
 prior=A.bind(tick);M.inventory(PARENT,PINS[0],tick);inv=M.inventory(PARENT/'run',PINS[1],tick);r=M.read(PARENT/'run/result.json');need(r['source_commit'].startswith(EXEC),'literal producer execution');M.source(PARENT,r,r['source_commit']);A.N.owner(PARENT,r['source_commit'])
 need(r['status']=='regularized_correction_complete'and r['profile']['passed']and not r['accepted_orbit']and not r['rigorous_certificate'],'completed bounded numerical scope')
 for n,h in prior[0]['source_sha256'].items():need(r['source_sha256'][n]==h,'unchanged actual native source')
 for k in('data_identity','eta','lam','model'):need(r['input_identity'][k]==prior[0]['input_identity'][k],'same exact physical objective')
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(r['runtime'][k]==prior[0]['runtime'][k],'same native runtime')
 ib=M.read(PARENT/'run/input-binding.json');need(ib['all512_actions_same_original022']and ib['comparison_same_start'],'same state declared');audit=ib['card']['independent'];ao=A.audit_source(Path(audit['root']),audit['source'],audit['inventories'][0]['manifest']['sha256'],audit['inventories'][1]['manifest']['sha256'],tick);need(ao['passed']and ao['recurrence_rows']==511,'independent full512 basis authority')
 counts,calls,gate,trials=events([json.loads(v)for v in(PARENT/'run/events.jsonl').read_text().splitlines()],inv);need(r['operation_attempted']==r['operation_returned']==dict(action=0,j2vp=0,**counts),'no hidden native calls');return prior,r,calls,gate,trials,counts

def audit(binding,tick,save):
 prior,r,calls,gate,trials,counts=binding;old,ocalls,*_=prior;load=lambda n:M.vector(PARENT/'run'/n);oldload=lambda n:M.vector(A.PARENT/'run'/n)
 x,g,origin=[load(n+'.npy')for n in ['initial-point','initial-gradient','global-original-midpoint']]
 need(M.same(x,oldload('state-000.npy'))and M.same(g,oldload('initial-gradient.npy'))and M.same(origin,oldload('global-original-midpoint.npy')),'literal original022 start/gradient/globalorigin')
 need(M.same(load(calls['direct_gradient',1]['state']),x)and M.same(load(calls['direct_gradient',1]['result']),g),'actual byte-exact initial replay')
 V=np.empty((98666,512),order='F');W=np.empty_like(V)
 for j in range(512):
  c=ocalls['hvp',1 if j==0 else j+2];need(c['state']=='state-000.npy','same-state action');V[:,j]=oldload(c['direction']);W[:,j]=oldload(c['result']);tick()
 p,hp,report=proposal(V,W,g,tick);stored=load('candidate-direction.npy');pred=load('predicted-action.npy');delta=M.norm(p-stored)/M.norm(stored);derr=M.norm(hp-pred)/M.norm(g);need(delta<1e-8 and derr<1e-8,'independently whitened direction/action agreement');del V,W;gc.collect()
 actual=calls['hvp',1];need(M.same(load(actual['state']),x)and M.same(load(actual['direction']),stored),'literal candidate physical HVP');pairs=[];k=1
 for t in trials:
  if t['within_trust']:k+=1;c=calls['direct_gradient',k];pairs.append((load(c['state']),load(c['result'])))
  else:pairs.append(None)
 answer,y,gy=A.acceptance(origin,x,g,stored,pred,load(actual['result']),pairs,True);M.compare(gate,answer['direction_gate']);M.compare(trials,answer['trials'])
 for key,value in answer.items():M.compare(value,r['correction'][key])
 need(M.same(y,load('final-point.npy'))and M.same(gy,load('final-gradient.npy')),'literal final accepted state')
 baseline=oldload('final-gradient.npy');ratio=M.dot(gy,gy)/M.dot(g,g);base=M.dot(baseline,baseline)/M.dot(g,g);need(ratio==r['comparison']['new_merit_ratio']and base==r['comparison']['baseline_merit_ratio'],'independent fsum same-start comparison')
 result=dict(passed=True,native_calls=0,producer_counts=counts,independent_proposal=report,direction_relative_difference=delta,action_relative_difference=derr,correction=answer,baseline_merit_ratio=base,regularized_merit_ratio=ratio,improves_same_start_merit=bool(answer['direction_applied']and ratio<base),scope='Independent symmetric spectral whitening/SVD/secular solve and fsum physical merit/trust/first acceptance. Shared acquired native derivatives and NumPy/SciPy libraries; no stationary root, cycle or benefit proof.');save('arithmetic.json',result);return result

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;stopped=False;result=dict(passed=False,native_calls=0)
 def tick():need(not stopped and time.monotonic()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'owned300s4GiBcap')
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 def save(n,v,final=False):
  nonlocal used
  data=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();need(used+len(data)<(8 if final else 6)*1024**2,'8MiB with2reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used+=len(data)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),files={n:M.entry(ROOT/n)for n in FILES},threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal own source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);used+=len(data)
  binding=bind(tick);rng=np.random.default_rng(52525);V=np.asfortranarray(rng.standard_normal((98666,512))/np.sqrt(98666));W=np.asfortranarray(V*np.linspace(.5,2,512));g=V[:,0].copy();t=time.monotonic();z=proposal(V,W,g,tick);duration=time.monotonic()-t;forecast=time.monotonic()-start+2*duration+30;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024;profile=dict(passed=forecast<300 and peak<4*1024**2,seconds=duration,forecast_seconds=forecast,forecast_rss_kib=peak,target_arrays_unopened=True);save('profile.json',profile);need(profile['passed'],'measured complete unrelated algebra');del V,W,g,z;gc.collect();result=audit(binding,tick,save)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
