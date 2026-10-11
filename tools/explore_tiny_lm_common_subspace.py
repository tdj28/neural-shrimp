"""Fixed saved-only common-subspace LM pilot; no native/model imports."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import hashlib,io,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp import tiny_lm_common_subspace as H
need=H.need
DATA=Path('/data2/neural-shrimp-data')
OUT=DATA/'explore/tiny-lm-common-subspace/attempt001/run'
CARD='docs/plans/tiny-lm-common-subspace-inputs.json'
PRODUCER=dict(root=str(DATA/'explore/tiny-lm-temporal-mean/attempt001-target'),source='5ce30146fe923456926ecb2c717c8db2e8c2749f',outer='bfd68a18be37a672a637da02b1e3dcbb32435ae2e7f5055be5657bef356ccf65',worker='44e8f6a03d7edba850c55e969f8a93d55cbb3277793276a0def5e3f6c24437c2')
RESPONSE=DATA/'explore/tiny-lm-response/attempt001-target'
HISTORY=[
 (RESPONSE,'fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5','081c5dcc779c3b2f60d0a3102275f0fa7609efbf'),
 (DATA/'explore/tiny-lm-two-snapshot/attempt001','c4ca66df6fdc7d848b98e3d232b0e1b99055a86130e2aae8fe52548ca5f1a91b','a86bb1e3a3332814f3db0040eda9ff1841891a89'),
 (DATA/'audits/tiny-lm-response-independent/attempt001','296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948','5a117145c0a0e7d91cb7a9bd96b8f8e49bc514d2'),
 (DATA/'audits/tiny-lm-two-snapshot-independent/attempt001','7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d','14296f7d67e4f10ee0d04d625543bc44e02abac0')]
NATIVE='27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8'
CAPS=dict(seconds=60,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=64*1024**2,reserve_bytes=4*1024**2,native_calls=0)
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/tiny_lm_common_subspace.py','tools/explore_tiny_lm_common_subspace.py','tools/launch_tiny_lm_common_subspace.py','tests/test_tiny_lm_common_subspace.py','tests/test_tiny_lm_common_subspace_driver.py','docs/plans/tiny-lm-common-subspace.md',CARD]


def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def git(source,name):return subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)


def inventory(root,pin,tick):
 need(not root.is_symlink()and entry(root/'manifest.json')['sha256']==pin,'literal immutable inventory')
 rows=read(root/'manifest.json')['files']
 if isinstance(rows,list):rows={x['path']:{k:x[k]for k in('bytes','sha256')}for x in rows}
 actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 need(actual==set(rows),'complete file inventory')
 for name,value in rows.items():
  tick();p=root/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink(),'safe exact path')
  need(entry(p)=={k:value[k]for k in('bytes','sha256')}and not p.stat().st_mode&0o222,'immutable exact file '+name)
 need(not(root/'manifest.json').stat().st_mode&0o222,'immutable manifest');return rows


def historical(root,source,exit_code=0):
 rt=read(root/'run/runtime.json');need(rt['source_commit']==source,'historical execution source')
 sources=rt.get('sources',rt.get('source_identity',rt.get('source_sha256')));need(isinstance(sources,dict)and sources,'full historical source map')
 for name,value in sources.items():
  raw=git(source,name);pin=value['sha256']if isinstance(value,dict)else value
  need(hashlib.sha256(raw).hexdigest()==pin,'exact historical source')
  if(root/'run/source').exists():need((root/'run/source'/name).read_bytes()==raw,'full historical archive')
 ownerfile=root/('launcher-result.json'if(root/'launcher-result.json').exists()else'launch-end.json');owner=read(ownerfile)
 need(owner.get('exit_code',owner.get('code'))==exit_code,'literal historical exit')
 if ownerfile.name=='launcher-result.json':need(not owner['owned_live_pids']and owner.get('worker_pid_absent',True),'owned producer group absent')
 else:need(owner['owned_group_empty']and owner['child_pid_absent'],'owned audit group absent')
 for pid in [rt.get('pid'),owner.get('launcher_pid'),owner.get('owned_group_pid'),owner.get('child_pid')]:
  if pid is not None:need(not Path('/proc',str(pid)).exists(),'historical owner absent')
 return rt


def card_check(card):
 need(card['status']=='reviewed'and card['producer']==PRODUCER and card['gates']==H.GATES and card['caps']==CAPS,'literal reviewed fixed design')
 a=card['independent'];need(isinstance(a,dict)and set(a)=={'root','source','outer','worker'}and all(isinstance(v,str)and v for v in a.values()),'sealed independent temporal-mean authority required')


def bind(w,card):
 run=Path(PRODUCER['root'])/'run';rows=inventory(run.parent,PRODUCER['outer'],w.tick);files=inventory(run,PRODUCER['worker'],w.tick);rt=historical(run.parent,PRODUCER['source'])
 need(rt['sources']['src/neural_shrimp/tiny_lm.py']['sha256']==NATIVE,'same fixed native model')
 old=read(run/'result.json');need(old['passed']and old['descriptive_gate']is False and old['attempted']==old['returned']==dict(hvp=1088,gradient=162,ce=58)and old['event_pairs']==1308,'complete retained failed mean sufficiency pilot')
 a=card['independent'];ar=Path(a['root']);inventory(ar,a['outer'],w.tick);inventory(ar/'run',a['worker'],w.tick);historical(ar,a['source']);audit=read(ar/'run/result.json');binding=read(ar/'run/input-binding.json')
 need(audit['passed']and audit['native_calls']==0 and audit['counts']==dict(hvp=1088,gradient=162,ce=58)and audit['rows']==12 and audit['descriptive_gate']is False and binding['producer']==PRODUCER,'complete independent prior arithmetic and literal inputs')
 for root,pin,source in HISTORY:inventory(root,pin,w.tick);historical(root,source)
 # The original failed profiles remain immutable prerequisites, including the
 # zero-native failure that has no worker directory.
 failure=DATA/'explore/tiny-lm-temporal-mean/attempt001-profile';inventory(failure,'4760072e1ecaf3487b215656a98b80dc54a37cad2b87fbdcc7270173b7af8c0b',w.tick);own=read(failure/'launcher-result.json')
 need(own['source_commit']=='a717415c7852e2e4303b089e581acad52ed3d0b4'and own['exit_code']==1 and not own['owned_live_pids']and own['worker_pid_absent']and not(failure/'run').exists(),'retained exact zero-native failure')
 need((failure/'launcher.py').read_bytes()==git(own['source_commit'],'tools/launch_tiny_lm_temporal_mean.py'),'original failed launcher source')
 for pid in (own['launcher_pid'],own['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'failed profile owners absent')
 profile=DATA/'explore/tiny-lm-temporal-mean/attempt002-profile';inventory(profile,'7b46a95ca74d33c4b88ab2f279a5882aea1ec09ed67ff41de3a03ac96242e587',w.tick);inventory(profile/'run','199c1d3f794ca630d4e2e369400da44a5ad9b7aae138cea38a73c6120b07eb57',w.tick)
 historical(profile,'adc6bbe153a05f8e47dcd10d2a03ed38972cd377',2)
 pr=read(profile/'run/result.json');need(not pr['passed']and pr['attempted']==pr['returned']==dict(hvp=2,gradient=2,ce=2)and pr['event_pairs']==6,'all cold profile acquisitions retained')
 w.save('input-binding.json',dict(producer=PRODUCER,independent=a,history=[dict(root=str(r),outer=p,source=s)for r,p,s in HISTORY],native_sha=NATIVE,event_pairs=1308,scope='Shared authenticated native actions; new reduced-model arithmetic only.'))
 return run,files,old


def load_object(run,pointer,files):
 need(set(pointer)=={'path','bytes','sha256','shape','dtype'}and pointer['dtype']=='float64','full literal float64 object pointer')
 name=pointer['path'];need(name=='objects/'+pointer['sha256']+'.npy'and files.get(name)=={k:pointer[k]for k in('bytes','sha256')},'content-addressed exact retained object')
 a=np.load(run/name,allow_pickle=False);need(a.dtype==np.float64 and list(a.shape)==pointer['shape']and np.isfinite(a).all(),'finite original full object');return a


def alias(run,name,files):
 row=read(run/'aliases'/(name+'.json'))
 if set(row)=={'array'}:return load_object(run,row['array'],files)
 need(set(row)=={'rows','shape'},'lossless per-row alias');out=np.asarray([load_object(run,x,files)for x in row['rows']]);need(list(out.shape)==row['shape'],'complete retained alias shape');return out


def events(run,files):
 expected={f'events/{i:05d}-{kind}.json'for i in range(1,1309)for kind in('input','return')};need({n for n in files if n.startswith('events/')}==expected,'complete1308 paired events')
 counts=dict(hvp=0,gradient=0,ce=0);result=[]
 for i in range(1,1309):
  a=read(run/f'events/{i:05d}-input.json');b=read(run/f'events/{i:05d}-return.json');op=a['operation'];need(op in counts,'declared operation');counts[op]+=1
  need(a['sequence']==i and a['index']==counts[op]and set(b)==set(a)|{'returned','seconds'}and all(b[k]==v for k,v in a.items())and np.isfinite(b['seconds'])and b['seconds']>0,'complete exact attempted/returned identity');result.append(b)
 need(counts==dict(hvp=1088,gradient=162,ce=58),'all original operation counts');return result


def acquire_case(run,files,records,seed,role):
 label=f'seed-{seed}-{role}';selected=[x for x in records if x['case']==label];mean=alias(run,label+'/mean-powers',files);b=read(run/label/'binding.json');need(b['seed']==seed and b['arm']==256 and b['step']==600 and b['rate_role']==role and b['original_direction_preserved'],'literal original case')
 folder=RESPONSE/'run'/f'seed-{seed}-arm-256'
 u=np.load(folder/'directions.npy',allow_pickle=False)[0];eta=read(folder/'inputs.json')['eta'];need(b['actual_eta']==eta/(1 if role=='original'else 4)and hashlib.sha256(u.tobytes()).hexdigest()==b['direction_sha256'],'exact physical rate and original direction')
 if role=='original':
  reference=np.load(folder/'reference.npy',allow_pickle=False);ordered=np.load(folder/'direction-0-transported.npy',allow_pickle=False);need(len(selected)==256 and all(x['operation']=='hvp'for x in selected),'only256 original mean actions')
 else:
  reference=alias(run,label+'/reference',files);ordered=alias(run,label+'/ordered',files);need(len(selected)==398 and {k:sum(x['operation']==k for x in selected)for k in('hvp','gradient','ce')}==dict(hvp=288,gradient=81,ce=29),'complete matched-control operations')
 need(mean.shape==reference.shape==ordered.shape==(17,15776)and all(np.isfinite(v).all()for v in(mean,reference,ordered))and np.array_equal(mean[0],u)and np.array_equal(ordered[0],u),'complete same literal direction and fixed references')
 selected=selected[-256:];need(all(x['operation']=='hvp'for x in selected),'last256 literal mean acquisitions')
 actions=np.empty((16,16,15776));indices=[]
 for h in range(16):
  for t in range(16):
   event=selected[16*h+t];state=load_object(run,event['state'],files);direction=load_object(run,event['direction'],files);value=load_object(run,event['returned'],files)
   need(np.array_equal(state,reference[t])and np.array_equal(direction,mean[h])and value.shape==(15776,),'all16 times at same exact mean direction');actions[h,t]=value;indices.append(event['sequence'])
 rows=read(run/label/'result.json')['rows'];qualified=bool(len(rows)==3 and [x['horizon']for x in rows]==[1,4,16]and all(x['mechanism_qualified']and x['training_ce_slope_resolved']for x in rows))
 return mean,actions,ordered,b['actual_eta'],dict(seed=seed,rate_role=role,operation_sequences=indices,original_qualification=qualified)


def synthetic(n,save):
 need(16<=n<=15776,'fixed full-size synthetic dimension')
 rng=np.random.default_rng(28169);basis,_=np.linalg.qr(rng.standard_normal((n,16)),mode='reduced');u=np.ones(16)/4;eta=.2
 matrices=np.asarray([np.diag(np.linspace(-1.1,1.1,16)+.015*np.sin(t))for t in range(16)])
 mean=H.product_response(np.repeat(matrices.mean(0)[None],16,axis=0),u);ordered=H.product_response(matrices,u)
 actions=np.asarray([[basis@((np.eye(16)-m)@v/eta)for m in matrices]for v in mean[:16]])
 result=H.analyze(mean@basis.T,actions,ordered@basis.T,eta,save);need(result['passed']and result['rank']==16,'complete full16 synthetic reconstruction');return result


def final_pass(passed, seconds, peak_rss_kib):
 return bool(passed and np.isfinite(seconds)and 0<=seconds<60 and 0<=peak_rss_kib<1024**2)


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.rows=[]
 def tick(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB saved-only cap')
 def write(self,name,data,closing=False):
  if not closing:self.tick()
  need(self.used+len(data)<(64 if closing else 60)*1024**2,'64MiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  self.used+=len(data)
 def save(self,name,value,closing=False):self.write(name,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 def array(self,name,value):
  b=io.BytesIO();np.save(b,np.asarray(value),allow_pickle=False);self.write(name+'.npy',b.getvalue())


def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ[k]=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source')
 card=read(ROOT/CARD);card_check(card);OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0,cases=[])
 def stop(*_):raise TimeoutError('owned60s saved pilot')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,threads=dict.fromkeys(THREADS,'1'),native_calls=0))
  for name in FILES:
   raw=(ROOT/name).read_bytes();need(raw==git(source,name),'complete committed source');w.write('source/'+name,raw)
  start=time.monotonic();before=w.used;trial=synthetic(15776,lambda k,v:w.array('profile/'+k,v));seconds=time.monotonic()-start;size=w.used-before;forecast=dict(seconds=8*seconds+15,bytes=w.used+8*size+8*1024**2)
  admission=bool(trial['passed']and forecast['seconds']<60 and forecast['bytes']<60*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024<1024**2);w.save('profile.json',dict(passed=admission,seconds=seconds,bytes=size,forecast=forecast,target_arrays_unopened=True));need(admission,'full-size measured saved-arithmetic admission')
  run,files,old=bind(w,card);records=events(run,files)
  for seed in(28161,28162):
   for role in('original','quarter_step'):
    w.tick();mean,actions,ordered,eta,binding=acquire_case(run,files,records,seed,role);label=f'seed-{seed}-{role}';w.save(label+'/binding.json',binding)
    if binding['original_qualification']:row=H.analyze(mean,actions,ordered,eta,lambda k,v:w.array(label+'/'+k,v))
    else:row=dict(passed=False,status='abstained',reasons=['original1% FD/transport/readout qualifications'],gates=H.GATES)
    row.update(seed=seed,rate_role=role);w.rows.append(row);w.save(label+'/result.json',row);del mean,actions,ordered
  result.update(passed=True,cases=w.rows,qualified_cases=sum(x['passed']for x in w.rows),abstained_cases=sum(not x['passed']for x in w.rows),scope='Consumed development, shared saved HVPs, conditional reduced-model sufficiency only. All abstentions retained; no new native calls, forecast, geometry or utility claim.')
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(cases=w.rows,seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);within=final_pass(result['passed'],result['seconds'],result['peak_rss_kib']);result.update(passed=within);w.save('result.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in('passed','qualified_cases','abstained_cases','error')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
