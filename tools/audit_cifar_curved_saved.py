"""Separately authored saved-only full-vector audit; no model or producer math imports."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
import hashlib,json,math,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
from review_cifar_curved_math import arithmetic,path,need,norm,dot,close,vec
OUT=Path('/data2/neural-shrimp-data/audits/cifar-curved-path-independent/attempt001')
CARD='docs/plans/cifar-curved-independent-inputs.json'
PRODUCER=dict(root='/data2/neural-shrimp-data/explore/cifar/stationary-seed/curved-path001',source='e470dded40446ca5532526dc313b6fbea4889a02',outer='109a48d9a8807c36c22a9ed3bfcf597cae9609f57b3e1517f1ad1f605202a489',worker='8dc30f591fe18f04d07271be13275814a71396ea4602ad6f57c24f600d17b198')
PARENT=dict(root='/data2/neural-shrimp-data/explore/cifar/stationary-seed/fresh512002',source='4de18a1d43e0265980dbfe85372bffaa9407214d',outer='6fec47a99f296f98d2098c9e4ca7f55a99433fbdee43be3343ef1fe79fe4c193',worker='31463434567088f82976557de6d22a480aaae7679590b35736c4362b94362969')
AUDIT=dict(root='/data2/neural-shrimp-data/audits/cifar-current512-independent/attempt001',source='eabcd27ca4a6c1c1007ef67f79fe41140c2a0a39',outer='10a4294cc1f88f921bb2d027ddf6b084afef0bf0b17430f264c3302678d16d40',worker='2471e751e812c0dfc1bd2961c3246f98992b00cfd49e7cbe0df78f2bb4d5f7d1')
REMAINDER=dict(root='/data2/neural-shrimp-data/explore/cifar/stationary-seed/remainder-span001',source='345ad394f11c66a708c0dc257ed77bb8584e0b53',outer='4d2e7fdfd748a185a3bb669441aa2762adedaac793823f0efd2a55471d333d8f',worker='e9c76f9169a5e5c63266fccfec7e5a05e66b1213cbbf9a63c56fab8933bd103b')
REGULARIZED=dict(root='/data2/neural-shrimp-data/explore/cifar/stationary-seed/regularized001',source='8254a7a9ad9e57e4b1af09e7cf4a4cdbf440f986',outer='9d271c9b4dcefff3d12f58a4eedbdf934622ff86ab87afda28391a50b2e3d3d6',worker='ecff8945abbbba1613eda075eca6f34b84462cea3150841c82b43d4bb93935f7')
SPECS=[PRODUCER,PARENT,AUDIT,REMAINDER,REGULARIZED]
CAPS=dict(seconds=60,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=8*1024**2,reserve_bytes=2*1024**2,native_calls=0)
FILES=[CARD,'docs/plans/cifar-curved-independent.md','tools/audit_cifar_curved_saved.py','tools/launch_cifar_curved_saved_audit.py','tools/reviews/review_cifar_curved_math.py','tests/test_cifar_curved_saved.py']
def read(p):return json.loads(p.read_text())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def git(source,n):return subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)
class Work:
 def __init__(self):self.start=time.monotonic();self.stop=False;self.used=0;self.root=OUT/'run';self.root.mkdir(parents=True,exist_ok=False)
 def tick(self):need(not self.stop and time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'saved60s1GiB audit')
 def write(self,n,raw,closing=False):
  need(self.used+len(raw)<CAPS['raw_bytes']-(0 if closing else CAPS['reserve_bytes']),'saved8MiB/reserved2MiB');p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def save(self,n,v,closing=False):self.write(n,(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal inventory');inv=read(root/'manifest.json')['files'];actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}-{'manifest.json'};need(set(inv)==actual,'complete inventory')
 for n,e in inv.items():
  tick();p=root/n;need(p.is_relative_to(root)and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==e,'immutable bytes '+n)
 return inv

def history(spec,w):
 root=Path(spec['root']);outer=inventory(root,spec['outer'],w.tick);inv=inventory(root/'run',spec['worker'],w.tick);owner=read(root/'launcher-result.json');need(owner['source_commit']==spec['source']and owner['exit_code']==0 and not owner['owned_live_pids']and owner['worker_pid_absent']and owner['failure']is None,'sealed historical success')
 for p in(owner['launcher_pid'],owner['owned_group_pid']):need(not Path('/proc',str(p)).exists(),'historical owner absent')
 run=root/'run';rp=run/'result.json';rt=read(rp if rp.exists()else run/'runtime.json');source=rt.get('source_commit',rt.get('source'));need(source==spec['source'],'exact historical source');sources=rt.get('source_sha256',rt.get('files',rt.get('sources')));need(isinstance(sources,dict)and sources,'historical source closure')
 archive=run/'source.tar.gz'
 if archive.exists():
  with tarfile.open(archive,'r:gz')as tar:
   members={m.name:m for m in tar.getmembers()};need(set(members)==set(sources)and all(m.isfile()for m in members.values()),'exact archive closure')
   for n,v in sources.items():
    w.tick();raw=git(spec['source'],n);pin=v['sha256']if isinstance(v,dict)else v;need(hashlib.sha256(raw).hexdigest()==pin and tar.extractfile(members[n]).read()==raw,'historical archived source '+n)
 else:
  for n,v in sources.items():
   w.tick();raw=git(spec['source'],n);pin=v['sha256']if isinstance(v,dict)else v;need(hashlib.sha256(raw).hexdigest()==pin and (run/'source'/n).read_bytes()==raw,'historical source '+n)
 return run,inv,rt

def events(run,inv):
 rows=[json.loads(s)for s in(run/'events.jsonl').read_text().splitlines()];ops={};pending=None;counts={}
 for e in rows:
  kind=e['kind']
  if kind=='native_input':need(pending is None,'nonoverlapping actual call');key=(e['operation'],e['index']);need(key not in ops,'unique call');pending=[key,e]
  elif kind=='operation_started':need(pending and len(pending)==2 and (e['operation'],e['index'])==pending[0],'attempt matches actual input');pending.append(e)
  elif kind=='operation_returned':need(pending and len(pending)==3 and (e['operation'],e['index'])==pending[0]and e['seconds']>0,'actual return matches attempt');pending.append(e)
  elif kind=='native_output':
   need(pending and len(pending)==4 and (e['operation'],e['index'])==pending[0],'full native return');key,inp,_,elapsed=pending
   for z in[ inp['state'],e['result']]+([inp['direction']]if 'direction'in inp else[]):need(inv[z['file']]==z['identity'],'actual acquisition byte pointer')
   counts[key[0]]=counts.get(key[0],0)+1;need(key[1]==counts[key[0]],'sequential actual native index');ops[key]=dict(state=inp['state']['file'],direction=inp.get('direction',{}).get('file'),result=e['result']['file'],seconds=elapsed['seconds']);pending=None
  else:need(pending is None,'no unfinished native event')
 need(pending is None,'complete native calls');return ops,rows

def metadata(w):
 bound=[history(s,w)for s in SPECS];run,inv,o=bound[0];pr,pinv,old=bound[1];ar,_,_=bound[2];dr,_,_=bound[3];rr,_,reg=bound[4]
 card=read(run/'curved-binding.json')['card'];need(card['status']=='reviewed'and card['operation_caps']==dict(action=0,j2vp=0,hvp=1,direct_gradient=2),'fixed producer design');need(read(run/'input-binding.json')['card']['producer']==PARENT,'exact original current source')
 au=read(ar/'outcome.json');ab=read(ar/'input-binding.json');need(au['passed']and au['native_calls']==0 and au['recurrence_rows']==511 and au['numerical_action_rank']==512 and au['correction']['accepted_index']==6 and ab['producer']==PARENT['root']and ab['source']==PARENT['source']and ab['pins']==[PARENT['outer'],PARENT['worker']],'independent tangent authority')
 basis=old['basis'];need(basis['completed_fixed_extension']and basis['total_columns']==512 and basis['status']=='maximum_columns'and basis['orthogonality_qualified'],'qualified original basis')
 need(o['source_commit']==PRODUCER['source']and o['operation_attempted']==o['operation_returned']==dict(action=0,j2vp=0,hvp=1,direct_gradient=2)and o['accepted_orbit']is False and o['rigorous_certificate']is False,'actual complete fixed target')
 need(read(run/'correction.json')==o['correction']and read(run/'comparison.json')==o['comparison'],'literal stored result copies')
 need(o['input_identity']==old['input_identity']and read(run/'input-identity.json')==o['input_identity'],'same actual objective and original ancestry')
 for n in ['src/neural_shrimp/cifar.py']:
  need(o['source_sha256'][n]==old['source_sha256'][n]==reg['source_sha256'][n],'same native source')
 for k in('python','numpy','scipy','torch','device','gpu','torch_threads','torch_interop_threads','thread_env'):need(o['runtime'][k]==old['runtime'][k]==reg['runtime'][k],'same native runtime')
 ops,rows=events(run,inv);pops,_=events(pr,pinv);need(set(ops)=={('hvp',1),('direct_gradient',1),('direct_gradient',2)}and len(pops)==523,'all actual operation identities')
 need(read(dr/'outcome.json')['diagnosis']['fixed_reference_alpha']==1/64,'fixed saved acceleration proposal')
 w.save('input-binding.json',dict(producer=PRODUCER,histories=SPECS,actual_counts=dict(hvp=1,gradient=2),scope='Shared native acquisition/source; separately authored full-vector arithmetic. No SVD or model reevaluation.'))
 return bound,ops,pops,rows

def array(p,w):w.tick();return vec(np.load(p,allow_pickle=False),98666)
def equal(a,b,message):need(np.array_equal(a,b),message)
def check_row(fresh,saved):
 for k,v in fresh.items():need(saved[k]is v if isinstance(v,bool)else close(saved[k],v),'independent scalar '+k)
def audit(w):
 # Full-size unrelated arithmetic before target arrays; source/inventory metadata may follow.
 n=98666;x=np.ones(n);g=np.full(n,1e-4);p=-g;a=p/10;states=[path(x,p,a,t)for t in(1/32,1/64)];grads=[g+y-x for y in states];start=time.monotonic();arithmetic(x,g,x,p,p,p,a,a,a,states,grads);dt=time.monotonic()-start
 need(8*dt+15<60,'full-size measured scalar admission');w.save('profile.json',dict(seconds=dt,forecast_seconds=8*dt+15,n=98666,native_calls=0,target_arrays_unopened=True));del x,g,p,a,states,grads
 bound,ops,pops,rows=metadata(w);run,_,o=bound[0];pr,_,old=bound[1];dr=bound[3][0];rr=bound[4][0]
 load=lambda n:array(run/n,w);parent=lambda n:array(pr/n,w)
 x,g,origin,p,hp,pred,a,predha=[load(n+'.npy')for n in['initial-point','initial-gradient','global-original-midpoint','candidate-direction','original-action','original-predicted-action','acceleration','predicted-acceleration-action']]
 for value,name in[(x,'state-000.npy'),(g,pops['direct_gradient',1]['result']),(origin,'global-original-midpoint.npy'),(p,'candidate-direction.npy'),(hp,pops['hvp',514]['result']),(pred,'predicted-action.npy')]:equal(value,parent(name),'literal original current/tangent bytes '+name)
 equal(p,parent(pops['hvp',514]['direction']),'old tangent actual direction');equal(x,parent(pops['hvp',514]['state']),'old tangent actual state')
 equal(a,array(dr/'fits/03-acceleration.npy',w),'literal selected acceleration');equal(predha,array(dr/'fits/03-predicted_action.npy',w),'literal saved predicted acceleration action')
 hv=ops['hvp',1];equal(x,load(hv['state']),'actual H a at exact current');equal(a,load(hv['direction']),'actual H a direction');ha=load(hv['result']);states=[load(ops['direct_gradient',i]['state'])for i in(1,2)];grads=[load(ops['direct_gradient',i]['result'])for i in(1,2)]
 fresh=arithmetic(x,g,origin,p,hp,pred,a,ha,predha,states,grads);c=o['correction'];check_row(fresh['tangent'],c['tangent_gate']);need(c['tangent_gate']['passed']and c['tangent_gate']['full_basis_qualified']and c['acceleration_action_qualified']and close(fresh['acceleration_action_relative_error'],c['acceleration_action_relative_error']),'strict actual action qualifications')
 for f,s,e in zip(fresh['trials'],c['trials'],[v for v in rows if v['kind']=='curved_path_trial'],strict=True):check_row(f,s);check_row(f,e)
 chosen=fresh['accepted_index'];need(chosen==c['accepted_index']and o['direction_applied']==(chosen is not None),'fixed first accepted point')
 finalx,finalg=(x,g)if chosen is None else(states[chosen],grads[chosen]);equal(load('final-point.npy'),finalx,'selected final point');equal(load('final-gradient.npy'),finalg,'selected final gradient')
 equal(parent('final-gradient.npy'),parent(pops['direct_gradient',9]['result']),'audited baseline gradient');equal(parent('final-point.npy'),x+(1/64)*p,'audited straight baseline point');baseline=dot(parent('final-gradient.npy'),parent('final-gradient.npy'))/dot(g,g);chosenratio=dot(finalg,finalg)/dot(g,g);cmp=o['comparison'];need(close(cmp['baseline_merit_ratio'],baseline)and close(cmp['chosen_merit_ratio'],chosenratio)and cmp['baseline_alpha']==1/64 and cmp['chosen_alpha']==(None if chosen is None else[1/32,1/64][chosen]),'same-start baseline and fixed selected alpha')
 need(cmp['improves_same_start_merit']==bool(chosen is not None and chosenratio<baseline),'independent comparison')
 for a,b in zip(cmp['trial_merit_ratios'],[v['merit_ratio']for v in fresh['trials']],strict=True):need(close(a,b),'all unselected results retained')
 profile=read(run/'profile.json');actual=ops['hvp',1]['seconds'];upstream=read(rr/'profile.json')['maximum_native_seconds']['hvp']
 return dict(passed=True,native_calls=0,counts=dict(hvp=1,gradient=2),arithmetic=fresh,baseline_merit_ratio=baseline,chosen_merit_ratio=chosenratio,improves_same_start=chosenratio<baseline,cost_observation=dict(actual_hvp_seconds=actual,selected_profile_hvp_seconds=profile['maximum_native_seconds']['hvp'],retained_upstream_hvp_seconds=upstream,exceeded_selected_component=actual>profile['maximum_native_seconds']['hvp'],within_total_actual_cap=o['elapsed_seconds']<120),scope='Shared retained native actions/gradients; independent fsum scalar and literal vector/source/event replay. Fixed first passing point; no SVD, Hessian or gradient re-evaluation, stationarity, periodic-orbit or utility claim.')
def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');card=read(ROOT/CARD);need(card['status']=='reviewed'and card['producer']==PRODUCER and card['histories']==SPECS and card['caps']==CAPS,'fixed reviewed card');w=Work();result=dict(passed=False,native_calls=0)
 signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stop',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stop',True));signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in FILES};w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources=identity,numpy=np.__version__))
  for n in FILES:raw=(ROOT/n).read_bytes();need(raw==git(source,n),'exact own committed source');w.write('source/'+n,raw)
  result=audit(w);w.tick()
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=bool(result['passed']and result['seconds']<60 and result['peak_rss_kib']<1024**2);w.save('result.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(w.root)):entry(p)for p in w.root.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in('passed','seconds','error','chosen_merit_ratio')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
