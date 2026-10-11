"""Saved first-order temporal-response replay with complete inherited authority."""
import json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_tiny_lm_block_mean as M
import review_tiny_lm_block_first_order_core as C
need,read,entry,close=M.need,M.read,M.entry,M.close
ROOT=Path(__file__).resolve().parents[1];THREADS=M.THREADS
OUT=Path('/data2/neural-shrimp-data/audits/tiny-lm-block-first-order-independent/attempt001/run')
SPEC=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-block-first-order/attempt001-target',source='4ad497e52e14b99ec3c1e52a0bd335e30a70a7d1',outer='90390d6ac37e126e0618640cf26f3efab6eb6b01554ecb38a2c38842da26055c',worker='f3b79cc92ef1c7824c137ffc260b3960d47eccfc002c30bc7d4885b54afec40c')
AUTH=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-block-mean-independent/attempt001',source='e4c5117a65774e98cf58ae31ab5f3c494247272c',outer='e8e3a32649844788439d1173d98cf5fd55abe04f5b920d0736e36297bda3c181',worker='e7233fcadb628e0a7b3afaf8fe4695c6e6ce2ac7779a4a4ebd5a5457e8351b45')
PROFILE=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-block-first-order/attempt001-profile',source='ae4db39cd795c19449430e6d3c5f5a0a5a9df5bf',outer='c013f2a61ed2ec031668179bd61a8be1f95e1ab6fae48d95dd7b10b6e9829aa3',worker='5ebfadf981070ae53cfdfeddea0bb3e57b7cd91f66904f7757d35d9bb83a348b')
CARD='docs/plans/tiny-lm-block-first-order-independent-inputs.json'
CAPS=dict(seconds=180,rss_bytes=1024**3,as_bytes=2*1024**3,raw_bytes=8*1024**2,native_calls=0)
FILES=sorted(set(M.FILES+[CARD,'tools/review_tiny_lm_block_first_order.py','tools/review_tiny_lm_block_first_order_core.py','tools/launch_tiny_lm_block_first_order_review.py','tests/test_tiny_lm_block_first_order_core.py','tests/test_tiny_lm_block_first_order_review.py','docs/plans/tiny-lm-block-first-order-independent.md']))

def card_check(card):
 need(card==dict(status='reviewed',producer=SPEC,profile=PROFILE,mean_independent=AUTH,caps=CAPS),'literal committed reviewed first-order audit')

def metadata(tick):
 card_check(read(ROOT/CARD));inv,mean,_,_=M.metadata(tick)
 ar,art,ao,iv=M.P.bundle(AUTH,tick);inv+=iv;close(ao['cases'],mean['cases'])
 need(ao['passed']and ao['native_calls']==0 and ao['retained_hvp_checked']==512 and ao['mean_powers_checked']==32 and len(ao['cases'])==4 and all(v['qualified']for v in ao['cases']),'entire independent block-mean authority')
 need(read(ar/'run/input-binding.json')['producer']==M.SPEC and ao['descriptive']==mean['descriptive'],'same independently audited producer and retained failed accuracy gate')
 target,rt,out,iv=M.P.bundle(SPEC,tick);inv+=iv;pr,prt,po,iv=M.P.bundle(PROFILE,tick);inv+=iv
 card=read(target/'run/run-card.json');cc='docs/plans/tiny-lm-block-first-order-inputs.json'
 settings=dict(profile_counts=dict(hvp=2),target_counts=dict(hvp=448),gates=C.GATES,blocks=8,powers=8,seeds=[28161,28162],roles=['original','quarter_step'],worker_seconds=dict(profile=60,target=1200),rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=512*1024**2)
 need(card==read(target/'run/source'/cc)and card['schema']=='tiny-lm-block-first-order-v1'and card['status']=='target_admitted'and card['source_review']and card['settings']==settings and card['mean_producer']==M.SPEC and card['mean_authority']==AUTH and card['profile']==PROFILE,'entire executed fixed first-order design')
 old=read(pr/'run/run-card.json');need(old==read(pr/'run/source'/cc)and old==dict(card,status='profile_admitted',profile=None),'literal separately admitted profile and target design')
 need(set(rt['sources'])==set(prt['sources']),'entire profiled source closure')
 for n,pin in rt['sources'].items():
  if n!=cc:need(prt['sources'][n]==pin,'exact same profiled native/caller source '+n)
 oldrt=read(Path(M.SPEC['root'])/'run/runtime.json')
 for k in('torch','numpy'):need(rt[k]==prt[k]==oldrt[k],'identical native libraries')
 need(rt['sources']['src/neural_shrimp/tiny_lm.py']==oldrt['sources']['src/neural_shrimp/tiny_lm.py'],'identical underlying full15776 model')
 need(po['passed']and 'error'not in po and po['mode']=='profile'and po['attempted']==po['returned']==dict(hvp=2)and po['event_pairs']==2 and po['target_arrays_unopened']and po['seconds']<60 and po['peak_rss_kib']<4*1024**2,'entire completed cold profile')
 binding=read(target/'run/block-mean-binding.json');need(binding==read(pr/'run/block-mean-binding.json')and binding['producer']==M.SPEC and binding['independent']==AUTH and binding['profile']==M.PROFILE,'same literal complete prior measured costs')
 meanroot=Path(M.SPEC['root']);costs=[]
 for i in range(1,513):
  a=read(meanroot/'run/events'/f'{i:05d}-input.json');b=read(meanroot/'run/events'/f'{i:05d}-return.json');need(a['operation']=='hvp'and a['sequence']==a['index']==i and set(b)==set(a)|{'returned','seconds'}and all(b[k]==v for k,v in a.items()),'all512 exact inherited action cost associations');costs.append(b['seconds'])
 costs+=read(Path(M.PROFILE['root'])/'run/result.json')['times']['hvp'];need(costs==binding['all514_costs']and all(math.isfinite(v)and v>0 for v in costs),'all514 target/cold costs')
 previous=read(meanroot/'run/profile-reference.json')['original_forecast']['effective_max_hvp_seconds'];floor=max(previous,*costs);need(floor==binding['effective_max_hvp_seconds'],'retain all earlier cold maxima')
 need(len(po['times']['hvp'])==len(po['return_bytes']['hvp'])==2 and all(math.isfinite(v)and v>0 for v in po['times']['hvp'])and all(type(v)is int and v>0 for v in po['return_bytes']['hvp']),'complete finite profile cost measurements')
 maximum=max(floor,*po['times']['hvp']);forecast=dict(seconds=2*448*maximum+30,bytes=2*1600*max(po['return_bytes']['hvp'])+64*1024**2+po['preforecast_bytes'],effective_max_hvp_seconds=maximum,old_max_hvp_seconds=floor,new_max_hvp_seconds=max(po['times']['hvp']),vectors=1600)
 pref=read(target/'run/profile-reference.json');need(pref['profile']==PROFILE,'same profile authority');close(forecast,po['forecast']);close(forecast,pref['original_forecast']);close({k:v for k,v in forecast.items()if k!='bytes'},{k:v for k,v in pref['current_forecast'].items()if k!='bytes'});need(forecast['seconds']<1200 and forecast['bytes']<512*1024**2 and pref['current_forecast']['bytes']<512*1024**2,'full inclusive measured admission')
 need(out['passed']and out['mode']=='target'and out['attempted']==out['returned']==dict(hvp=448)and out['event_pairs']==448 and len(out['cases'])==4 and out['seconds']<1200 and out['peak_rss_kib']<4*1024**2,'entire bounded448-action target')
 return inv,out,ao


def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');card_check(read(ROOT/CARD));OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'180s1GiB saved replay')
 def stop(*_):raise TimeoutError('owned180s deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,180)
 def write(name,raw,closing=False):
  nonlocal used
  need(used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
 def save(name,row,closing=False):write(name,(json.dumps(row,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=commit,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'entire exact committed archive');write('source/'+n,raw)
  inv,out,mean=metadata(tick)
  u=np.linspace(.1,1,15776);ends=np.asarray([u*(k+1)for k in range(8)]);force=u/16;t=time.monotonic();C.correction_sum(ends,force,np.sum(ends,axis=0)/8+force);C.C.comparison(u,dict(ordered=u,mean=2*u,frozen=3*u,direction=u,ce_gradient=u),[u,u],[C.dot(u,u)]*2);sec=time.monotonic()-t
  forecast=time.monotonic()-start+64*sec+16*mean['seconds']+30;save('profile.json',dict(passed=forecast<180,seconds=sec,forecast_seconds=forecast,retained_prior_audit_seconds=mean['seconds'],full_vector_dimension=15776,target_arrays_unopened=True));need(forecast<180,'fullsize correction/response and complete prior replay cost before target arrays')
  result=C.replay(Path(SPEC['root']),out,Path(M.SPEC['root']),mean['cases'],tick,save);save('input-binding.json',dict(producer=SPEC,profile=PROFILE,mean_independent=AUTH,inventories=inv));tick()
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=bool(result['passed']and result['seconds']<180 and result['peak_rss_kib']<1024**2);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in('passed','seconds','descriptive','error')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
