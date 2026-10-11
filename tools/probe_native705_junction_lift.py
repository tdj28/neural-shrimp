#!/usr/bin/env python3
"""One measured conditional55-to705 lift of the independently proved junction."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.rigorous_junction_lift import sharpen_known_root,lift_known_junction,original_ball,contained
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,eye,vector_json,ball_json,canonical_bytes
from neural_shrimp.rigorous_eigenpair_locus import require as need
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/junction-lift001/attempt003/run')
PARENT=Path('/data2/neural-shrimp-data/audits/native-unit-pair-flip-independent/attempt002')
SOURCE='1a991e0205cf6ddf00adb5b678a14d0f44857a49'
PINS=['d14c91bdfc3c754a6654a8c30ffbd4b69ac59150aeea717d1426fc2b1480e952','7f0e39d8b206859914b5aabb4da063427ad81c0b9f26f836bbb6b9899d47db49']
ORIGINAL_SOURCE='714813fa28110cbeb5fac57833c833ea19a3ed0e'
NATIVE=['src/neural_shrimp/'+n+'.py'for n in('rigorous_cusps','rigorous_folds','rigorous_orbits','rigorous_eigenpair_locus','rigorous_unit_pair')]
FAILED=OUT.parent.with_name('attempt001')
FAILED_SCHEMA=OUT.parent.with_name('attempt002')
FAILED_SCHEMA_OUTER='bdcbe018611a72f07d1d8ebc10d74e7afd95dd7fb0f8902926f53d48dd365857'
FAILED_OUTER='8ea83a85ac8286ddc7b14b5d3d2d705476fbf8ef9422fe64215af782d78082a1'
FILES=['docs/reviews/native705-junction-lift-schema-failure.json','docs/reviews/native705-junction-lift-profile-failure.json','tools/probe_native705_junction_lift.py','tools/launch_native705_junction_lift.py','tests/test_rigorous_junction_lift.py','tests/test_native705_junction_lift_driver.py','docs/plans/native705-junction-lift.md']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def mat(rows):return arb_mat([[original_ball(v)for v in row]for row in rows])
def settings():return dict(precision=512,period=18,profile_calls=72,target_calls=36,refinement_intersections=3,seconds=60,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=80*1024**2)
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.calls=0;self.serial=0;self.stage='profile'
 def check(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'60s/2GiB measured worker cap')
 def tick(self):self.calls+=1;need(self.calls<=108,'72profile plus36target complete native ceiling');self.check()
 def save(self,name,value,closing=False):
  raw=canonical_bytes(value);need(self.used+len(raw)<(80 if closing else 78)*1024**2,'80MiB with2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def observe(self,k,v):self.save(f'events/{self.serial:03d}-{self.stage}-{k}.json',v);self.serial+=1;self.check()

def inventory(root,pin,w):
 need(entry(root/'manifest.json')['sha256']==pin,'exact historical manifest');m=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(actual)==set(m),'complete historical inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==m[n],'immutable historical bytes '+n);w.check()

def archive(root,source,w):
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');owned=read(root/'owned-process.json');need(rt['commit']==la['commit']==source and la['exit_code']==0 and not la['owned_live_pids'],'complete historical execution')
 if source==SOURCE:
  need(set(owned)=={'pid'}and 'owned_group_pid'not in la,'literal old independent ownership schema');ids=[owned['pid'],rt['pid']]
 else:
  need(set(owned)=={'group_pid'}and owned['group_pid']==la['owned_group_pid'],'literal old producer ownership');ids=[owned['group_pid'],rt['pid'],la['launcher_pid']]
 need(all(type(v)is int and v>0 and not Path('/proc',str(v)).exists()for v in ids),'historical owned groups exited')
 files={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};need(set(files)==set(rt['sources']),'entire executed historical archive')
 for n,row in rt['sources'].items():need(entry(files[n])==row and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'exact historical source '+n);w.check()
 for n in NATIVE:need(entry(ROOT/n)==rt['sources'][n],'identical current actual native map '+n)
 return rt

def qualify(root,out,identity,old,center,whole):
 need(out['passed']and out['same_unique_root']and out['coordinates']==55 and out['native_calls']==162,'independent full55 completed theorem')
 need(root['passed']and root['dimension']==13 and root['period']==18 and root['augmented_dimension']==55 and root['coordinate_inclusions']==[True]*55 and all(root['gates'].values())and root['settings']==dict(newton_steps=0,radius_exponent=-84),'exact original independent root gates')
 need(identity['fresh_box']==root['box']and identity['original_box']==old['box']and identity['fresh_inside_original']==[True]*55,'same unique full55 root authority')
 need(all(contained(a,b)for a,b in zip(root['box'],old['box'],strict=True)),'literal nested domain')
 need([d['divisor']for d in root['divisors']]==[1,2,3,6,9]and all(d['excluded']for d in root['divisors']),'primitive18 root')
 need(center['center']==root['center']and center['value']==root['value']and whole['box']==root['box']and whole['jacobian']==root['whole_jacobian']and whole['monodromy']==root['monodromy'],'exact center/whole acquired derivative association')
 need(len(out['fresh']['complement_bounds'])==10 and all(original_ball(v)<1 for v in out['fresh']['complement_bounds'])and not original_ball(out['fresh']['crossing']).contains(0),'inherited simultaneous critical pair/simpleflip stable complement and physical crossing')

def bind(w):
 for p,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):inventory(p,pin,w)
 rt=archive(PARENT,SOURCE,w);need(rt['precision']==480,'original480 derivative acquisition');b=read(PARENT/'run/binding.json');need(b['source']==ORIGINAL_SOURCE,'fixed original local junction')
 orig=Path(b['root']);
 for p,k in((orig,'outer'),(orig/'run','worker')):inventory(p,b[k],w)
 archive(orig,ORIGINAL_SOURCE,w)
 root=read(PARENT/'run/fresh-root.json');old=read(orig/'run/root.json');identity=read(PARENT/'run/root-identity.json');ce=read(PARENT/'run/fresh-center_evaluation.json');wh=read(PARENT/'run/fresh-box_evaluation.json');qualify(root,read(PARENT/'run/outcome.json'),identity,old,ce,wh)
 need(b['map_inputs']==read(orig/'run/inputs.json')['map_inputs']and TwoRateCuspMap(b['map_inputs']).n==13,'same actual13D two-rate native map')
 w.save('input-binding.json',dict(independent_root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],root=entry(PARENT/'run/fresh-root.json'),original=b,root_identity=entry(PARENT/'run/root-identity.json'),map_inputs=b['map_inputs'],native_source={n:rt['sources'][n]for n in NATIVE}));return root,b['map_inputs']

def fixture():
 inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[[str(F(i-12,48)),str(F((i*7)%23-11,48))]for i in range(24)],'y':[str(F((i*3)%17-8,64))for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
 x=[arb(exact(str(F(i-6,64))),arb((1,-100)))for i in range(13)];q=[arb(exact(str(F(i+1,32))),arb((1,-100)))for i in range(13)];r=[arb(exact(str(F(i-3,64))),arb((1,-100)))for i in range(13)];rates=[arb(exact('1/100'),arb((1,-100)))]*2
 return inp,x+rates+q+r+[exact('1/16')]+[arb(0)]*13

def forecast(times,sizes,peak):
 need(len(times)==len(sizes)==2 and all(v>0 for v in times+sizes)and peak>0,'two full13/18 native profile units')
 seconds=4*max(times)+15;raw=4*max(sizes)+16*1024**2;rss=2*peak+128*1024
 return dict(seconds=seconds,bytes=raw,rss_kib=rss,admitted=seconds<60 and raw<80*1024**2 and rss<2*1024**2)

def profile(w):
 inp,z=fixture();w.save('synthetic-inputs.json',dict(map_inputs=inp,box=vector_json(z)));times=[];sizes=[]
 for i in range(2):
  begin=time.monotonic();used=w.used;w.stage=f'profile{i}';flat,p=lift_known_junction(TwoRateCuspMap(inp),z,18,tick=w.tick,observe=w.observe);need(len(flat)==705 and p['native_calls']==36 and not p['normalization_contains_zero'],'unrelated full13/18 profile never claims known root');w.save(f'profile{i}-lift.json',p)
  c=[arb(j)/64 for j in range(55)];B=vector_json([arb(v,arb(1)/1024)for v in c]);J=eye(55);J[0,54]=arb(1)/4
  _,p=sharpen_known_root(c,B,eye(55),arb_mat(55,1),J,tick=w.check,observe=w.observe);w.save(f'profile{i}-refinement.json',p);times.append(time.monotonic()-begin);sizes.append(w.used-used)
 need(w.calls==72,'complete measured72 native profile');pr=forecast(times,sizes,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('profile.json',dict(times=times,sizes=sizes,forecast=pr,target_arrays_unopened=True));need(pr['admitted'],'measured full-size time/RSS/bytes admission')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete',target_acquired=False)
 def stop(*_):raise TimeoutError('60s fixed junction lift cap')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=settings(),python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,sources={n:entry(ROOT/n)for n in FILES},threads=dict.fromkeys(THREADS,'1'),nice=10))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):
   inventory(FAILED_SCHEMA,FAILED_SCHEMA_OUTER,w);schema_failure=read(FAILED_SCHEMA/'run/outcome.json');need(schema_failure['native_calls']==72 and not schema_failure['target_acquired']and schema_failure['error']=="KeyError: 'J'"and not read(FAILED_SCHEMA/'launcher-result.json')['owned_live_pids'],'retained exact historical-schema failure');w.save('prior-schema-failure-binding.json',dict(root=str(FAILED_SCHEMA),outer=FAILED_SCHEMA_OUTER,outcome=schema_failure));inventory(FAILED,FAILED_OUTER,w);failure=read(FAILED/'run/outcome.json');need(not failure['passed']and failure['native_calls']==72 and failure['target_acquired']is False and not read(FAILED/'launcher-result.json')['owned_live_pids'],'retained complete72-call profile failure with no target');w.save('prior-failure-binding.json',dict(root=str(FAILED),outer=FAILED_OUTER,outcome=failure));profile(w);root,inp=bind(w);w.stage='target';result['target_acquired']=True
   box,refinement=sharpen_known_root([original_ball(v)for v in root['center']],root['box'],mat(root['preconditioner']),mat(root['value']),mat(root['whole_jacobian']),tick=w.check,observe=w.observe);w.save('known-root-refinement.json',refinement)
   flat,lift=lift_known_junction(TwoRateCuspMap(inp),box,18,tick=w.tick,observe=w.observe);w.save('native705-lift.json',lift);need(w.calls==108 and len(flat)==705 and lift['closure_contains_zero']and lift['normalization_contains_zero'],'complete conditional same-junction normalized lift')
   result.update(passed=True,status='same_independent_junction_lifted_to_native705',physical_rates_angle=vector_json(flat[-3:]),scope='Same independently certified local55 unit-pair/simpleflip junction, conditionally sharpened then freshly lifted in literal18 native phases to705 coordinates. No connection to the continued curve or extended complementary spectrum follows until full705 membership is separately proved.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.calls,events=w.serial);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
