"""Independent saved fsum audit of one numerical stationary-merit step; no model imports."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import hashlib,json,math,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-stationary-saved-merit/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/saved-merit001')
OLD=PARENT.with_name('attempt001')
EXEC='86ab814cbcfbc8465b8a313e711a69314c7406e6';OLD_EXEC='ebc18b48e0c42c793e66b4048379a6740313e10e'
PINS={'parent':'3d77085cf744c2bb60e1e5d905765e20e6bf2e1cb10abd28415fa20139593300','worker':'6782082fbc0be0a39bf1c267c2e0fa5d8bf06abaa0cf460ecd7bd36a9911c0fb','old':'36fb7d130b05c4392b32b583567aa7090da70223b79b5463f82e77e1cf4879be','old_worker':'a91a3d7551a707ec673d9e8f2f03cf609d4956cba25f00dd7b3b521d0da62893'}
SETTINGS=dict(forcing_limit=.9,armijo=1e-4,trust_relative=.05,trials=13,stationarity=1e-10)
FILES=['tools/reviews/review_cifar_stationary_merit.py','tools/reviews/launch_cifar_stationary_merit_review.py','tests/reviews/test_cifar_stationary_merit_review.py','docs/plans/cifar-stationary-merit-independent.md']
def need(v,msg):
 if not v:raise ValueError(msg)
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_bytes())
def close(a,b,label):need(math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-20),'scalar '+label)
def norm(v):return math.sqrt(math.fsum(float(x)*float(x)for x in v))
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b))
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def vector(p,n=98666):
 a=np.load(p,allow_pickle=False);need(a.dtype==np.float64 and a.shape==(n,)and np.isfinite(a).all(),'finite full binary64 '+str(p));return a

def arithmetic(x,g,p,hp,trials):
 need(x.shape==g.shape==p.shape==hp.shape and all(np.isfinite(v).all()for v in(x,g,p,hp)),'finite equal vectors')
 ng=norm(g);need(ng>0,'positive original gradient');forcing=norm(g+hp)/ng;slope=dot(g,hp);phi=.5*dot(g,g)
 gate=dict(forcing=forcing,slope=slope,initial_merit=phi,cauchy_descent_upper=-(1-forcing)*dot(g,g),old_forcing_pass=forcing<=.5,passed=forcing<=.9 and slope<0)
 rows=[];chosen=None
 for i,(y,newg)in enumerate(trials):
  need(chosen is None and i<13,'no trial after acceptance');alpha=2.**(-i);need(same(y,x+alpha*p),'exact candidate coordinates');need(y.shape==newg.shape==x.shape and np.isfinite(newg).all(),'full finite trial gradient')
  distance=norm(y-x)/max(1.,norm(x));merit=.5*dot(newg,newg);bound=phi+1e-4*alpha*slope
  row=dict(index=i,alpha=alpha,relative_origin_distance=distance,within_trust=distance<=.05,merit=merit,armijo_bound=bound,relative_gradient=norm(newg)/max(1.,norm(y)),accepted=gate['passed']and distance<=.05 and merit<=bound)
  rows.append(row)
  if row['accepted']:chosen=i
 need(chosen is not None or len(rows)==13,'complete stopping rule')
 y,newg=(x,g)if chosen is None else trials[chosen]
 return dict(direction_gate=gate,trials=rows,accepted_index=chosen,initial_relative_gradient=ng/max(1.,norm(x)),final_relative_gradient=norm(newg)/max(1.,norm(y)),merit_ratio=dot(newg,newg)/dot(g,g),stationarity_qualified=norm(newg)/max(1.,norm(y))<=1e-10)

def inventory(root,pin,tick):
 need(root.is_dir()and not root.is_symlink()and not root.stat().st_mode&0o222,'immutable root')
 need(entry(root/'manifest.json')['sha256']==pin,'literal manifest pin')
 rows=read(root/'manifest.json')['files'];actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(rows)==actual,'complete inventory')
 for name,v in rows.items():
  tick();p=root/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==v,'immutable bytes '+name)
 return rows

def source(root,row,execution):
 need(row['source_commit']==execution,'literal execution')
 with tarfile.open(root/'run/source.tar.gz','r:gz')as tar:
  members=tar.getmembers();need(len(members)==len(row['source_sha256'])and {m.name for m in members}==set(row['source_sha256'])and all(m.isfile()and m.size<2*1024**2 for m in members),'complete source archive')
  for m in members:
   b=tar.extractfile(m).read();sha=hashlib.sha256(b).hexdigest();need(sha==row['source_sha256'][m.name]==hashlib.sha256(subprocess.check_output(['git','show',execution+':'+m.name],cwd=ROOT)).hexdigest(),'historical source '+m.name)

def event_block(events,pos,operation,index,folder,state,output,direction=None):
 block=events[pos:pos+4];need(len(block)==4 and [v['kind']for v in block]==['native_input','operation_started','operation_returned','native_output'],'complete native event order')
 for e in block:need(e['operation']==operation and e.get('index',index)==index,'native operation/index')
 need(math.isfinite(block[2]['seconds'])and block[2]['seconds']>=0,'operation duration')
 def item(v,expected):
  name=v['file'];need(Path(name).name==name and v['identity']==entry(folder/name),'actual event file binding');need(same(vector(folder/name),expected),'actual native array')
 item(block[0]['state'],state);item(block[3]['result'],output)
 if direction is not None:item(block[0]['direction'],direction)
 else:need('direction'not in block[0],'gradient has no direction')
 return pos+4

def compare(actual,saved):
 need(type(actual)==type(saved)or isinstance(actual,(int,float))and not isinstance(actual,bool)and isinstance(saved,(int,float))and not isinstance(saved,bool),'matching value types')
 if isinstance(actual,dict):
  need(set(actual)<=set(saved),'complete reported keys')
  for k,v in actual.items():compare(v,saved[k])
 elif isinstance(actual,list):
  need(len(actual)==len(saved),'complete reported rows')
  for a,b in zip(actual,saved):compare(a,b)
 elif isinstance(actual,(int,float))and not isinstance(actual,bool):close(actual,saved,'reported arithmetic')
 else:need(actual==saved,'reported flag')

def audit(tick,save):
 inv=inventory(PARENT,PINS['parent'],tick);inventory(PARENT/'run',PINS['worker'],tick);oldinv=inventory(OLD,PINS['old'],tick);inventory(OLD/'run',PINS['old_worker'],tick)
 row=read(PARENT/'run/result.json');old=read(OLD/'run/result.json');outer=read(PARENT/'launcher-result.json')
 source(PARENT,row,EXEC);source(OLD,old,OLD_EXEC)
 need(row['settings']==SETTINGS and row['review_commit']==EXEC and row['status']=='stationary_saved_merit_complete'and row['comparison_qualified']and not row['accepted_orbit']and not row['rigorous_certificate'],'exact completed candidate scope')
 need(outer['source_commit']==EXEC and outer['exit_code']==0 and outer['scientific_complete']and outer['failure']is None and not outer['owned_live_pids']and outer['worker_pid_absent'],'complete owned producer')
 for pid in(outer['launcher_pid'],outer['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'actual producer ownership ended')
 need(outer['maximum_group_rss_kib']<=2*1024**2 and outer['elapsed_seconds']<=110 and row['elapsed_seconds']<=90,'observed producer caps')
 oldstep=old['stationary']['steps'];need(old['status']=='stationary_seed_complete'and old['qualification']['passed']and not old['seed_qualified']and old['mode']is None and old['stationary']['status']=='true_forcing_failed'and len(oldstep)==1 and oldstep[0]['trials']==[]and oldstep[0]['accepted_index']is None and oldstep[0]['iterations']==256,'preserved original negative')
 need(old['operation_attempted']==old['operation_returned']==dict(action=0,j2vp=0,hvp=259,direct_gradient=7),'original operation counts')
 for name,sha in old['source_sha256'].items():need(row['source_sha256'][name]==sha,'unchanged original source')
 for key in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(row['runtime'][key]==old['runtime'][key],'same source runtime')
 for key in('data_identity','eta','lam','model'):need(row['input_identity'][key]==old['input_identity'][key],'same objective '+key)
 need(row['input_identity']['parent']==str(OLD)and row['input_identity']['parent_manifest']==PINS['old']and row['input_identity']['parent_source']==OLD_EXEC,'exact direct predecessor')
 need(read(PARENT/'run/input-identity.json')==row['input_identity'],'identity mirror')
 folder=PARENT/'run';load=lambda name:vector(folder/(name+'.npy'));loadold=lambda name:vector(OLD/'run'/(name+'.npy'))
 x,g,p,hp=[load(n)for n in('initial-point','initial-gradient','saved-direction','saved-action')]
 for a,n in((x,'initial-midpoint'),(x,'stationary-point'),(g,'stationary-gradient'),(g,'native-direct_gradient-0001-return'),(p,'native-hvp-0259-vector'),(hp,'native-hvp-0259-return')):need(same(a,loadold(n)),'exact old endpoint/direction '+n)
 oldevents=[json.loads(v)for v in(OLD/'run/events.jsonl').read_text().splitlines()]
 for operation,index,state,output,direction in [('direct_gradient',1,x,g,None),('hvp',259,x,hp,p)]:
  positions=[i for i,e in enumerate(oldevents)if e['kind']=='native_input'and e['operation']==operation and e['index']==index];need(len(positions)==1,'unique original native call');event_block(oldevents,positions[0],operation,index,OLD/'run',state,output,direction)
 trials=[(load(f'trial-{i:02d}-point'),load(f'trial-{i:02d}-gradient'))for i in range(len(row['trials']))]
 answer=arithmetic(x,g,p,hp,trials);compare(answer,row);close(answer['direction_gate']['forcing'],oldstep[0]['forcing'],'old forcing');need(not answer['direction_gate']['old_forcing_pass']and answer['direction_gate']['passed'],'disclosed relaxed gate')
 chosen=answer['accepted_index'];need(chosen==3 and len(trials)==4,'fixed observed four-trial outcome');need(same(load('final-point'),trials[chosen][0])and same(load('final-gradient'),trials[chosen][1]),'exact final accepted state')
 events=[json.loads(v)for v in(folder/'events.jsonl').read_text().splitlines()];pos=event_block(events,0,'direct_gradient',1,folder,x,g);pos=event_block(events,pos,'hvp',1,folder,x,hp,p)
 for i,(y,newg)in enumerate(trials):
  pos=event_block(events,pos,'direct_gradient',i+2,folder,y,newg);need(events[pos]==dict(kind='stationary_saved_merit_trial',**row['trials'][i]),'literal trial after its gradient');pos+=1
 need(pos==len(events),'complete no-extra event stream');need(row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=1,direct_gradient=5),'actual six native calls')
 need(row['operation_caps']==dict(action=0,j2vp=0,hvp=1,direct_gradient=14),'unchanged cap')
 # The old timing record governs the disclosed producer cost gate; recompute it.
 mx={k:max(e['seconds']for e in oldevents if e['kind']=='operation_returned'and e['operation']==k)for k in('hvp','direct_gradient')};compare(dict(maximum_seconds=mx,forecast_seconds=2*(mx['hvp']+14*mx['direct_gradient'])+20),row['profile']);need(row['profile']['forecast_seconds']<90,'producer measured gate')
 save('input-binding.json',dict(parent=str(PARENT),old=str(OLD),pins=PINS,producer_source=EXEC,old_source=OLD_EXEC,parent_files=len(inv),old_files=len(oldinv),native_event_count=len(events),all_arrays_source_bound=True))
 save('arithmetic.json',answer);return dict(passed=True,native_calls=0,analysis=answer,scope='Independent saved binary64 scalar acceptance audit. Shared native gradient/Hessian outputs explicitly retained; no independent derivative calculation, stationary point, eigenpair or training orbit proof.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'30s1GiB budget')
 def stop(*_):raise TimeoutError('saved audit deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<4*1024**2-(0 if closing else 1024**2),'4MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,python=sys.version,numpy=np.__version__,pid=os.getpid(),threads={k:os.environ[k]for k in THREADS},files={n:entry(ROOT/n)for n in FILES}))
  for name in FILES:
   b=(ROOT/name).read_bytes();need(b==subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT),'committed own source');p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
  # Full-size unrelated fsum reducer admission before research arrays.
  t=time.monotonic();x=np.ones(98666);g=x*.01;p=-x*.01;hp=-g;sample=arithmetic(x,g,p,hp,[(x+p,g*.5)]);seconds=time.monotonic()-t;save('profile.json',dict(seconds=seconds,forecast_seconds=8*seconds+10,target_arrays_unopened=True));need(8*seconds+10<30,'measured full-size reducer gate');del x,g,p,hp,sample
  result=audit(tick,save)
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
