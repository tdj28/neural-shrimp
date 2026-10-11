"""Independently authored full512-action block-mean replay and authority binding."""
import ast,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_tiny_lm_block_mean_core as C
B,P,M=C.B,C.P,C.M
need,read,entry,close=C.need,C.read,B.entry,C.close
ROOT=Path(__file__).resolve().parents[1];THREADS=B.THREADS
OUT=Path('/data2/neural-shrimp-data/audits/tiny-lm-block-mean-independent/attempt001/run')
SPEC=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-block-mean/attempt001-target',source='858fec5feaabd4c6eee31449445330d390500a01',outer='7f4fedc79c4b3ae16980e406cb768f9e32353eb48e1762d41fd7d08390c60fe5',worker='833baaa3d853a03b71576dd3f590bdc321c12efb42571dd78676cbc1377c40fe')
AUTH=dict(root='/data2/neural-shrimp-data/audits/tiny-lm-pair-blocks-independent/attempt001',source='99e4efa111c97bc09c562dbbd30f5c90aecef1e8',outer='906fc65ff04aee2d86aedf7bc1d63cd62a0016c7076412b4c8015f9b754896c5',worker='e6a22991e3bcddf5eabe74ecec5d3d8f3fd1b03083dddd27297a38a0f91b429f')
PROFILE=dict(root='/data2/neural-shrimp-data/explore/tiny-lm-block-mean/attempt001-profile',source='58d08555eab4032dee87843bbe2b280aa9da4a03',outer='02dcb4f349c1589ad0e556de97c5e8e44412c9f8159f550de88dc52349bf8c34',worker='87c4d5876d6a2c2d2135c982f5eca0cb2ebded325b8239f11bc30bbf91e6084a')
CARD='docs/plans/tiny-lm-block-mean-independent-inputs.json'
CAPS=dict(seconds=180,rss_bytes=1024**3,as_bytes=2*1024**3,raw_bytes=8*1024**2,native_calls=0)
FILES=sorted(set(B.FILES+[CARD,'tools/review_tiny_lm_block_mean.py','tools/review_tiny_lm_block_mean_core.py','tools/launch_tiny_lm_block_mean_review.py','tests/test_tiny_lm_block_mean_core.py','tests/test_tiny_lm_block_mean_review.py','docs/plans/tiny-lm-block-mean-independent.md']))

def card_check(card):
 need(card==dict(status='reviewed',producer=SPEC,profile=PROFILE,pair_independent=AUTH,caps=CAPS),'literal committed reviewed audit admission')

def normalized(raw):
 tree=ast.parse(raw);tree.body=[n for n in tree.body if not(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name=='profile_binding')]
 class TimeOnly(ast.NodeTransformer):
  def visit_Constant(self,n):return ast.copy_location(ast.Constant(1200),n)if type(n.value)is int and n.value==900 else n
 return ast.dump(TimeOnly().visit(tree),include_attributes=False)

def metadata(tick):
 card_check(read(ROOT/CARD));inv,pair,parity,mean=B.metadata(tick)
 ar,art,ao,iv=P.bundle(AUTH,tick);inv+=iv;close(ao['cases'],pair['cases'])
 need(ao['passed']and ao['native_calls']==0 and ao['retained_hvp_checked']==128 and all(v['qualified']for v in ao['cases'])and read(ar/'run/input-binding.json')['producer']==B.SPEC,'complete independent pair authority')
 target,rt,out,iv=P.bundle(SPEC,tick);inv+=iv;pr,prt,po,iv=P.bundle(PROFILE,tick,2);inv+=iv
 card=read(target/'run/run-card.json');need(card==read(target/'run/source/docs/plans/tiny-lm-block-mean-inputs.json')and card['status']=='target_admitted'and card['pair_producer']==B.SPEC and card['pair_authority']==AUTH and card['profile']==PROFILE,'literal executed block-mean design')
 settings=dict(profile_counts=dict(hvp=2),target_counts=dict(hvp=512),gates=C.GATES,blocks=8,powers=8,seeds=[28161,28162],roles=['original','quarter_step'],worker_seconds=dict(profile=60,target=1200),rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=512*1024**2)
 need(card['settings']==settings and set(rt['sources'])==set(prt['sources']),'complete fixed design and profile closure')
 wc='tools/explore_tiny_lm_block_mean.py';lc='tools/launch_tiny_lm_block_mean.py';tc='tests/test_tiny_lm_block_mean_driver.py';pc='docs/plans/tiny-lm-block-mean.md';cc='docs/plans/tiny-lm-block-mean-inputs.json'
 for n,pin in rt['sources'].items():
  if n not in(wc,lc,tc,pc,cc):need(prt['sources'][n]==pin,'exact unchanged profiled source '+n)
 get=lambda p,n:(p/'run/source'/n).read_text()
 need(normalized(get(target,wc))==normalized(get(pr,wc)),'same entire caller outside reviewed time/admission-only amendment')
 need(get(pr,lc).replace('else 915','else 1215')==get(target,lc)and get(pr,tc).replace('co_consts.count(900)','co_consts.count(1200)').replace("'else 915'","'else 1215'")==get(target,tc)and get(target,pc).startswith(get(pr,pc)),'literal time-only launcher/test and appended plan')
 oldcard=read(pr/'run/source'/cc);need(oldcard['settings']==dict(settings,worker_seconds=dict(profile=60,target=900))and oldcard['status']=='profile_admitted'and oldcard['profile']is None,'original900s card retained')
 oldrt=read(B.TARGET/'run/runtime.json')
 for k in('torch','numpy'):need(rt[k]==prt[k]==oldrt[k],'same full native runtime')
 need(rt['sources']['src/neural_shrimp/tiny_lm.py']==oldrt['sources']['src/neural_shrimp/tiny_lm.py'],'same native equations')
 need(not po['passed']and 'error'not in po and po['mode']=='profile'and po['attempted']==po['returned']==dict(hvp=2)and po['event_pairs']==2 and po['target_arrays_unopened']and po['seconds']<60 and po['peak_rss_kib']<4*1024**2,'entire time-only negative fresh profile')
 pb=read(target/'run/pair-binding.json');need(pb['producer']==B.SPEC and pb['independent']==AUTH,'same pair native cost authority')
 paircard=read(B.TARGET/'run/run-card.json');need(pb['profile']==paircard['profile'],'same pair cold profile');costs=[]
 for i in range(1,129):
  a=read(B.TARGET/'run/events'/f'{i:05d}-input.json');b=read(B.TARGET/'run/events'/f'{i:05d}-return.json');need(a['operation']=='hvp'and a['sequence']==a['index']==i and set(b)==set(a)|{'returned','seconds'}and all(b[k]==v for k,v in a.items()),'all original pair cost associations');costs.append(b['seconds'])
 costs+=read(Path(pb['profile']['root'])/'run/result.json')['times']['hvp'];need(costs==pb['all130_costs']and all(math.isfinite(v)and v>0 for v in costs),'complete130 actual/cold costs')
 upstream=read(B.TARGET/'run/parity-binding.json')['effective_max_hvp_seconds'];old=max(upstream,*costs);need(old==pb['effective_max_hvp_seconds'],'entire cold-cost floor')
 maximum=max(old,*po['times']['hvp']);f=dict(seconds=2*512*maximum+30,bytes=2*1280*max(po['return_bytes']['hvp'])+64*1024**2+po['preforecast_bytes'],effective_max_hvp_seconds=maximum,old_max_hvp_seconds=old,new_max_hvp_seconds=max(po['times']['hvp']),vectors=1280)
 pref=read(target/'run/profile-reference.json');need(pref['profile']==PROFILE,'same actual measured admission');close(f,po['forecast']);close(f,pref['original_forecast']);close(f['seconds'],pref['current_forecast']['seconds']);need(900<f['seconds']<1200 and f['bytes']<512*1024**2 and pref['current_forecast']['bytes']<512*1024**2,'preserved900s rejection and1200s measured gate')
 need(out['passed']and out['mode']=='target'and out['attempted']==out['returned']==dict(hvp=512)and out['event_pairs']==512 and len(out['cases'])==4 and out['seconds']<1200 and out['peak_rss_kib']<4*1024**2,'entire bounded512 target')
 return inv,out,mean,ao

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
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'complete literal committed archive');write('source/'+n,raw)
  inv,out,mean,pair=metadata(tick)
  u=np.linspace(.1,1,15776);ends=np.asarray([u*(k+1)for k in range(8)]);t=time.monotonic();C.mean_of_eight(ends,np.sum(ends,axis=0)/8);C.comparison(u,dict(ordered=u,mean=2*u,frozen=3*u,direction=u,ce_gradient=u),[u,u],[C.dot(u,u)]*2);sec=time.monotonic()-t
  forecast=time.monotonic()-start+128*sec+16*pair['seconds']+30;save('profile.json',dict(passed=forecast<180,seconds=sec,forecast_seconds=forecast,full_vector_dimension=15776,target_arrays_unopened=True));need(forecast<180,'complete fullsize measured saved-replay admission')
  result=C.replay(Path(SPEC['root']),out,mean['cases'],pair['cases'],tick,save);save('input-binding.json',dict(producer=SPEC,profile=PROFILE,pair_independent=AUTH,inventories=inv));tick()
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=bool(result['passed']and result['seconds']<180 and result['peak_rss_kib']<1024**2);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in('passed','seconds','descriptive','error')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
