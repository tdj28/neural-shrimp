"""Independent retained-call replay and fsum temporal-mean/response reductions."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_tiny_lm_ordering as B
need,read,entry,exact,array,dot,norm=B.need,B.read,B.entry,B.exact,B.array,B.dot,B.norm
ROOT=Path(__file__).resolve().parents[1]; DATA=Path('/data2/neural-shrimp-data')
OUT=DATA/'audits/tiny-lm-temporal-mean-independent/attempt001/run'
TARGET=DATA/'explore/tiny-lm-temporal-mean/attempt001-target'; PROFILE=TARGET.with_name('attempt002-profile')
SOURCE='5ce30146fe923456926ecb2c717c8db2e8c2749f'; PSOURCE='adc6bbe153a05f8e47dcd10d2a03ed38972cd377'
PRODUCER=dict(root=str(TARGET),source=SOURCE,outer='bfd68a18be37a672a637da02b1e3dcbb32435ae2e7f5055be5657bef356ccf65',worker='44e8f6a03d7edba850c55e969f8a93d55cbb3277793276a0def5e3f6c24437c2')
PPINS=('7b46a95ca74d33c4b88ab2f279a5882aea1ec09ed67ff41de3a03ac96242e587','199c1d3f794ca630d4e2e369400da44a5ad9b7aae138cea38a73c6120b07eb57')
RESPONSE=B.RESPONSE; SNAP=DATA/'explore/tiny-lm-two-snapshot/attempt001';PARENT=B.LEGACY[1][0]
HISTORY=[(RESPONSE,B.PINS['response'],'081c5dcc779c3b2f60d0a3102275f0fa7609efbf'),(SNAP,'c4ca66df6fdc7d848b98e3d232b0e1b99055a86130e2aae8fe52548ca5f1a91b','a86bb1e3a3332814f3db0040eda9ff1841891a89'),(*B.AUDITS[0],'5a117145c0a0e7d91cb7a9bd96b8f8e49bc514d2'),(*B.AUDITS[1],'14296f7d67e4f10ee0d04d625543bc44e02abac0')]
COUNTS=dict(hvp=1088,gradient=162,ce=58);H=(1,4,16);THREADS=B.THREADS
FILES=['tools/review_tiny_lm_temporal_mean.py','tools/launch_tiny_lm_temporal_mean_review.py','tools/review_tiny_lm_ordering.py','tests/test_tiny_lm_temporal_mean_review.py','docs/plans/tiny-lm-temporal-mean-independent.md']

def close(a,b,key=''):
 if isinstance(a,dict):
  need(set(a)<=set(b),'complete reported fields '+key)
  for k,v in a.items():close(v,b[k],k)
 elif isinstance(a,list):
  need(len(a)==len(b),'complete rows '+key)
  for x,y in zip(a,b,strict=True):close(x,y,key)
 elif a is None or isinstance(a,(bool,str)):need(a==b,'literal gate '+key)
 else:need(math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,rel_tol=2e-11,abs_tol=1e-12),'independent scalar '+key)

def source(run,commit):
 rt=read(run/'runtime.json');need(rt['source_commit']==commit,'literal executed source')
 rows=rt.get('sources',rt.get('source_identity',rt.get('source_sha256')));need(rows,'complete source inventory')
 for n,v in rows.items():
  raw=subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT);sha=v['sha256']if isinstance(v,dict)else v
  need(hashlib.sha256(raw).hexdigest()==sha,'original committed source '+n)
  if(run/'source').is_dir():need((run/'source'/n).read_bytes()==raw,'archived original source '+n)
 return rt

def metadata(tick):
 pairs=[(TARGET,PRODUCER['outer']),(TARGET/'run',PRODUCER['worker']),(PROFILE,PPINS[0]),(PROFILE/'run',PPINS[1]),B.LEGACY[1]]+[(p,h)for p,h,s in HISTORY]
 inv=[B.inventory(p,h,tick)for p,h in pairs];rt=source(TARGET/'run',SOURCE);pr=source(PROFILE/'run',PSOURCE)
 for p,h,s in HISTORY:
  source(p/'run',s);need(read(p/'run/result.json')['passed'],'prior qualified native/response authority')
 old=read(RESPONSE/'run/runtime.json');need(old['source_identity']['src/neural_shrimp/tiny_lm.py']==rt['sources']['src/neural_shrimp/tiny_lm.py'],'same original native model')
 for k in ('torch','numpy'):need(old[k]==rt[k]==pr[k],'same native runtime')
 for n in ('src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_temporal_mean.py'):need(rt['sources'][n]==pr['sources'][n],'same profiled math/model')
 for p,r,s,code in ((TARGET,rt,SOURCE,0),(PROFILE,pr,PSOURCE,2)):
  la=read(p/'launcher-result.json');need(la['source_commit']==s and la['exit_code']==code and la['failure']is None and not la['owned_live_pids']and la['worker_pid_absent'],'owned execution completed')
  for pid in (r['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'exact owner absent')
 p=read(PROFILE/'run/result.json');need(p['attempted']==p['returned']==dict(hvp=2,gradient=2,ce=2)and p['event_pairs']==6 and not p['passed']and 'error'not in p,'retained complete time-rejected profile')
 sec=2*math.fsum(COUNTS[k]*max(p['times'][k])for k in COUNTS)+30
 size=2*(sum(COUNTS[k]*max(p['return_bytes'][k])for k in COUNTS)+352*(15776*8+128))+64*1024**2
 close(dict(seconds=sec,bytes=size),p['forecast']);need(600<sec<900 and size<508*1024**2,'unchanged cold-cost900s amendment')
 b=read(TARGET/'run/input-binding.json');need(b['parent']==dict(root=str(PARENT),manifest=B.LEGACY[1][1])and b['history']==[dict(root=str(p),manifest=h,source=s)for p,h,s in HISTORY],'complete exact input genealogy')
 out=read(TARGET/'run/result.json');need(out['passed']and out['mode']=='target'and out['attempted']==out['returned']==COUNTS and out['event_pairs']==1308 and len(out['cases'])==4,'complete original target')
 return inv,out

class Events:
 def __init__(self,run,tick):self.run=run;self.tick=tick;self.n=0;self.counts=dict.fromkeys(COUNTS,0);self.cache={}
 def obj(self,p):
  need(set(p)=={'path','bytes','sha256','dtype','shape'}and p['dtype']=='float64'and p['path']=='objects/'+p['sha256']+'.npy','literal safe object pointer')
  path=self.run/p['path'];need(entry(path)=={k:p[k]for k in ('bytes','sha256')},'exact pointed bytes')
  if p['path']not in self.cache:self.cache[p['path']]=array(path,tuple(p['shape']))
  return self.cache[p['path']]
 def alias(self,label,name,shape):
  p=read(self.run/'aliases'/label/(name+'.json'))
  a=np.asarray([self.obj(v)for v in p['rows']])if 'rows'in p else self.obj(p['array'])
  need(a.shape==shape and('rows'not in p or p['shape']==list(shape)),'complete alias shape');return a
 def call(self,label,op,state,direction=None):
  self.tick();self.n+=1;self.counts[op]+=1
  a=read(self.run/'events'/f'{self.n:05d}-input.json');b=read(self.run/'events'/f'{self.n:05d}-return.json')
  need({k:b[k]for k in a}==a and set(b)==set(a)|{'returned','seconds'}and a['case']==label and a['operation']==op and a['sequence']==self.n and a['index']==self.counts[op]and math.isfinite(b['seconds'])and b['seconds']>0,'literal ordered complete input/return')
  exact(self.obj(a['state']),state,'native call actual state')
  if direction is None:need('direction'not in a,'no spurious direction')
  else:exact(self.obj(a['direction']),direction,'native call actual direction')
  ret=self.obj(b['returned']);need(ret.shape==(()if op=='ce'else(15776,)),'complete native return');return ret

def mean_step(v,actions,eta,saved):
 need(actions.shape==(16,len(v)),'sixteen same-vector actions')
 total=np.asarray([math.fsum(float(x)for x in row)for row in actions.T]);ind=v-(eta/16)*total
 # Conservative coordinatewise gamma bound for16 additions, multiplication/subtraction.
 bound=64*np.finfo(float).eps*(np.abs(v)+(abs(eta)/16)*np.sum(np.abs(actions),axis=0))+np.finfo(float).tiny
 ratio=float(np.max(np.abs(saved-ind)/bound));need(ratio<=1,'independent mean recurrence rounding envelope');return ratio

def metrics(case,predictors):
 rows=[]
 for i,h in enumerate(H):
  responses=[(r['plus'][h]-r['minus'][h])/(2*e)for r,e in zip(case['branches'],case['eps'],strict=True)];small=responses[1];gain=norm(small);scale=max(gain,1e-12)
  agree=norm(responses[0]-small)/scale;oe=norm(predictors['ordered'][h]-small)/scale
  slopes=[(r['ce_plus'][h]-r['ce_minus'][h])/(2*e)for r,e in zip(case['branches'],case['eps'],strict=True)];cs=max(abs(slopes[1]),1e-8);comp={}
  for name,values in predictors.items():
   v=values[h];vn=norm(v);comp[name]=dict(vector_error=norm(v-small)/scale,gain=vn,signed_alignment=None if vn*gain==0 else dot(v,small)/(vn*gain),training_ce_slope=dot(case['cegrad'][i],v))
  rows.append(dict(horizon=h,response_gain=gain,finite_difference_relative_disagreement=agree,transported_vector_error=oe,resolved=agree<=.01 and oe<=.01,training_ce_slopes=slopes,training_ce_slope_disagreement=abs(slopes[0]-slopes[1])/cs,training_ce_slope_prediction_error=abs(comp['ordered']['training_ce_slope']-slopes[1])/cs,comparisons=comp))
 for row in rows:row.update(mechanism_qualified=row['resolved']and rows[0]['resolved'],training_ce_slope_resolved=row['training_ce_slope_disagreement']<=.01 and row['training_ce_slope_prediction_error']<=.01)
 return rows

def audit(tick,save):
 inv,out=metadata(tick);ev=Events(TARGET/'run',tick);cases=[];rounding=0
 data=read(PARENT/'data.json');need(data['data_identity_sha256']=='a65b88c04f7de18031005672a345af8eaf929ff75cf67e1d0c401b02c7f6f6aa'and len(data['vocabulary'])==66,'actual training data')
 with np.load(PARENT/'data-tokens.npz',allow_pickle=False)as z:
  toks=np.asarray(data['roles']['expanded']['window_token_ids'],dtype=np.int64)
  need(z['expanded_X'].shape==z['expanded_Y'].shape==(256,32)and np.array_equal(z['expanded_X'],toks[:,:-1])and np.array_equal(z['expanded_Y'],toks[:,1:]),'exact expanded training tokens')
 for seed in (28161,28162):
  old=RESPONSE/'run'/f'seed-{seed}-arm-256';md=read(old/'inputs.json');eta=md['eta'];oldout=read(old/'outcome.json')
  ref=array(old/'reference.npy',(17,15776));grad=array(old/'reference-gradients.npy',(17,15776));u=array(old/'directions.npy',(2,15776))[0]
  exact(u,grad[0]/np.linalg.norm(grad[0]),'literal inherited unit gradient');exact(ref[1:],ref[:-1]-eta*grad[:-1],'original GD recurrence')
  state=PARENT/f'seed-{seed}-arm-256/states-000600-000649.npy';raw=np.load(state,allow_pickle=False);need(raw.dtype==np.float32 and raw.shape==(50,15776),'full original state chunk');exact(ref[0],raw[0].astype(np.float64),'exact consumed step600')
  for role,rate in (('original',eta),('quarter_step',eta/4)):
   label=f'seed-{seed}-{role}';binding=read(TARGET/'run'/label/'binding.json')
   expected=dict(seed=seed,arm=256,step=600,eta=eta,chunk=entry(state),state_float32_sha256=hashlib.sha256(raw[0].tobytes()).hexdigest(),data_identity=data['data_identity_sha256'],rate_role=role,actual_eta=rate,direction_sha256=hashlib.sha256(u.tobytes()).hexdigest(),original_direction_preserved=True);need(binding==expected,'literal state/data/rate/direction association')
   load=lambda n,s:ev.alias(label,n,s)
   if role=='original':r,g=ref,grad;f=array(old/'direction-0-frozen.npy',(17,15776));v=array(old/'direction-0-transported.npy',(17,15776));eps=oldout['epsilon_values'];cg=array(old/'readout-gradients.npy',(3,15776));increments=None
   else:
    r=load('reference',(17,15776));g=load('reference-gradients',(17,15776));ce=load('reference-ce',(17,));f=load('frozen',(17,15776));v=load('ordered',(17,15776));cg=load('readout-gradients',(3,15776));eps=[e*max(1.,float(np.linalg.norm(ref[0])))for e in (1e-4,5e-5)]
    exact(r[0],ref[0],'same control origin');exact(g[0],grad[0],'same deterministic initial gradient')
    for h in range(17):exact(ev.call(label,'gradient',r[h]),g[h],'all reference gradients');exact(ev.call(label,'ce',r[h]),ce[h],'all reference CE')
    exact(r[1:],r[:-1]-rate*g[:-1],'all control reference GD updates')
    for h in range(16):
     for pred,theta in ((f,r[0]),(v,r[h])):exact(pred[h+1],pred[h]-rate*ev.call(label,'hvp',theta,pred[h]),'all frozen/ordered physical updates')
    obj=np.asarray([float(c)+.0005*dot(t,t)for c,t in zip(ce,r,strict=True)]);increments=np.diff(obj).tolist();close(increments,load('objective-increments',(16,)).tolist())
   exact(f[0],u,'frozen actual starting direction');exact(v[0],u,'ordered actual starting direction');exact(cg,np.asarray([g[h]-.001*r[h]for h in H]),'CE gradient subtracts ridge')
   close(eps,[e*max(1.,norm(ref[0]))for e in (1e-4,5e-5)]);branches=[]
   for k,e in enumerate(eps):
    branch={}
    for sign,side in ((1,'plus'),(-1,'minus')):
     states=array(old/f'direction-0-epsilon-{k}-{side}.npy',(17,15776))if role=='original'else load(f'epsilon-{k}-{side}',(17,15776))
     cevals=array(old/f'direction-0-epsilon-{k}-{side}-ce.npy',(3,2))if role=='original'else load(f'epsilon-{k}-{side}-ce',(3,2));need(np.array_equal(cevals[:,0],H),'exact branch readouts');ces=dict((int(h),float(c))for h,c in cevals)
     exact(states[0],r[0]+sign*e*u,'literal signed perturbation')
     if role!='original':
      for h in range(1,17):
       exact(states[h],states[h-1]-rate*ev.call(label,'gradient',states[h-1]),'all perturbed physical updates')
       if h in H:need(float(ev.call(label,'ce',states[h]))==ces[h],'actual perturbed CE')
     branch[side]=states;branch['ce_'+side]=ces
    branches.append(branch)
   mean=load('mean-powers',(17,15776));exact(mean[0],u,'same mean starting direction')
   for h in range(16):
    acts=np.asarray([ev.call(label,'hvp',r[k],mean[h])for k in range(16)]);rounding=max(rounding,mean_step(mean[h],acts,rate,mean[h+1]))
   predictors=dict(mean=mean,frozen=f,ordered=v,persistence=np.repeat(u[None,:],17,axis=0))
   if role=='original':
    for name,tag in (('two_snapshot','alternating'),('latest_J1','latest')):
     a=array(SNAP/'run'/f'seed-{seed}-arm-256'/f'direction-0-{tag}.npy',(16,15776));exact(a[0],v[1],'snapshot exact h1 alignment');predictors[name]=np.concatenate((u[None,:],a))
   rows=metrics(dict(branches=branches,eps=eps,cegrad=cg),predictors);case=dict(seed=seed,rate_role=role,eta=rate,rows=rows,small_step_monotone=None if increments is None else all(x<=0 for x in increments),objective_increments=increments)
   original=read(TARGET/'run'/label/'result.json');need(original==out['cases'][len(cases)],'complete exact case ordering');close(case,original);cases.append(case);save(label+'.json',case)
 need(ev.n==1308 and ev.counts==COUNTS and len(list((ev.run/'events').glob('*-input.json')))==len(list((ev.run/'events').glob('*-return.json')))==1308,'all events exhausted')
 gate=all(c['rows'][-1]['mechanism_qualified']and c['rows'][-1]['comparisons']['mean']['vector_error']<.2 and 10*c['rows'][-1]['comparisons']['mean']['vector_error']<c['rows'][-1]['comparisons']['frozen']['vector_error']for c in cases if c['rate_role']=='original');need(gate==out['descriptive_gate'],'fixed descriptive outcome')
 save('input-binding.json',dict(producer=PRODUCER,inventories=inv))
 return dict(passed=True,native_calls=0,counts=ev.counts,rows=12,descriptive_gate=gate,cases=cases,maximum_mean_rounding_ratio=rounding,scope='Independently replayed all1308 retained calls and fsum response/CE/mean reductions on four consumed development cases. Shared native derivatives and prior independent response authorities. Mean failure does not isolate ordering, imply utility or concern held-out outcomes.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'120s1GiB audit cap')
 def stop(*_):raise TimeoutError('owned120s deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
 def write(name,raw,closing=False):
  nonlocal used
  need(used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
 def save(name,value,closing=False):write(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=commit,pid=os.getpid(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),sources={n:entry(ROOT/n)for n in FILES}))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'complete committed own source');write('source/'+n,raw)
  t=time.monotonic();v=np.linspace(.1,1,15776);acts=np.repeat(v[None,:],16,axis=0);mean_step(v,acts,.3,v-.3*v);norm(v);dot(v,v);seconds=time.monotonic()-t;forecast=200*seconds+30;save('profile.json',dict(passed=forecast<120,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<120,'full-size measured reduction admission');result=audit(tick,save)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in ('passed','seconds','error','descriptive_gate')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
