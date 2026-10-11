"""Saved-only interval physical stability corner at the exact native18 junction."""
import hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from neural_shrimp import junction_audit as A
from neural_shrimp.unit_flip_corner import gradients
from test_unit_flip_corner import fixture
PARENT=Path('/data2/neural-shrimp-data/audits/native-unit-pair-flip-independent/attempt002')
SOURCE='1a991e0205cf6ddf00adb5b678a14d0f44857a49'
PINS=['d14c91bdfc3c754a6654a8c30ffbd4b69ac59150aeea717d1426fc2b1480e952','7f0e39d8b206859914b5aabb4da063427ad81c0b9f26f836bbb6b9899d47db49']
OUT=Path('/data2/neural-shrimp-data/explore/certificates/native-unit-flip-stability-corner/attempt001/run')
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/junction_audit.py','src/neural_shrimp/unit_flip_corner.py','tools/certify_unit_flip_corner.py','tools/launch_unit_flip_corner.py','tests/test_unit_flip_corner.py','tests/test_unit_flip_corner_driver.py','docs/plans/native-unit-flip-stability-corner.md','docs/reviews/native-unit-pair-flip-independent.json']
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def pack(v):
 if isinstance(v,arb):
  A.need(v.is_finite(),'finite serialized enclosure');return dict(mid=[int(x)for x in v.mid().man_exp()],rad=[int(x)for x in v.rad().man_exp()],display=str(v))
 if isinstance(v,arb_mat):return [[pack(v[i,j])for j in range(v.ncols())]for i in range(v.nrows())]
 if isinstance(v,dict):return {k:pack(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [pack(x)for x in v]
 return v
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):A.need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'30s512MiB saved algebra cap')
 def save(self,name,value):
  raw=(json.dumps(pack(value),sort_keys=True,indent=2,allow_nan=False)+'\n').encode();A.need(self.used+len(raw)<16*1024**2,'16MiB output cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
def inventory(root,pin,w):
 A.need(entry(root/'manifest.json')['sha256']==pin,'literal immutable outer pin');m=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};A.need(set(actual)==set(m),'complete immutable inventory')
 for n,p in actual.items():A.need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==m[n],'full original bytes '+n);w.tick()
def archive(root,source,w):
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');A.need(rt['commit']==la['commit']==source and la['exit_code']==0 and not la['owned_live_pids'],'exact completed historical source')
 for pid in [rt['pid'],la['launcher_pid'],la['owned_group_pid']]:A.need(not Path('/proc',str(pid)).exists(),'historical owned process absent')
 actual={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};A.need(set(actual)==set(rt['sources']),'complete executed source archive')
 for n,v in rt['sources'].items():A.need(entry(actual[n])==v and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical source '+n);w.tick()
 return rt

def history_gate(out,identity,root,old):
 A.need(out['passed']and out['same_unique_root']and out['coordinates']==55,'independently proved same55 root')
 A.need(root['passed']and all(root['gates'].values())and root['dimension']==13 and root['period']==18 and root['settings']==dict(newton_steps=0,radius_exponent=-84),'literal fresh480 root geometry')
 A.need(identity['original_box']==old['box']and identity['fresh_box']==root['box']and all(identity['fresh_inside_original']),'literal all55 root identity')
 A.strict_nested(old['box'],root['box'])

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);A.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact execution source');A.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0,status='incomplete')
 def stop(*_):raise TimeoutError('30s saved physical corner cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),precision=512,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,sources={n:entry(ROOT/n)for n in FILES},threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();A.need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(b).hexdigest(),'exact current source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):
   begin=time.monotonic();probe=gradients(*fixture(13));C=A.eye(55);C[0,54]=arb(1)/3;control=A.validated_solve(C,A.eye(55));w.save('profile-algebra.json',dict(gradient=probe,inverse55=control));elapsed=time.monotonic()-begin;forecast=20*elapsed+6;raw=4*w.used+2*1024**2
   w.save('profile.json',dict(passed=bool(probe['passed']and forecast<30 and raw<16*1024**2),seconds=elapsed,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));A.need(probe['passed']and forecast<30 and raw<16*1024**2,'full13/55 measured source/serialization admission')
   for r,pin in zip([PARENT,PARENT/'run'],PINS):inventory(r,pin,w)
   rt=archive(PARENT,SOURCE,w);A.need(rt['precision']==480,'literal independent derivative precision')
   binding=read(PARENT/'run/binding.json');orig=Path(binding['root']);A.need(binding['source']=='714813fa28110cbeb5fac57833c833ea19a3ed0e','literal original junction authority')
   for r,key in [(orig,'outer'),(orig/'run','worker')]:inventory(r,binding[key],w)
   archive(orig,binding['source'],w)
   root=read(PARENT/'run/fresh-root.json');old=read(orig/'run/root.json');identity=read(PARENT/'run/root-identity.json');history_gate(read(PARENT/'run/outcome.json'),identity,root,old)
   proposal=read(orig/'run/same-root-complement.json');rootcheck=A.root_arithmetic(root);prior=A.spectrum_crossing(root,proposal)
   A.need(rootcheck['inclusions']==55 and len(prior['complement_bounds'])==10 and all(z<1 for z in prior['complement_bounds']),'same exact primitive18 root and10 stable complementary modes')
   w.save('input-binding.json',dict(independent_root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],original=binding,root_proof=entry(PARENT/'run/fresh-root.json'),root_identity=entry(PARENT/'run/root-identity.json'),root_check=rootcheck,complement_and_crossing=prior,shared='Arb and authenticated mixed-derivative acquisitions; no native reevaluation'))
   z=gradients(A.mat(root['monodromy']),A.mat(root['whole_jacobian']),[A.decode(v)for v in root['box']],proposal['coordinate_columns']);w.save('physical-corner.json',z)
   A.need(z['passed'],'strict full physical stability-gradient determinant');A.need(z['flip_derivative_along_unit_hidden']is not None and z['flip_derivative_along_unit_hidden'].overlaps(prior['crossing']),'agreement with independent fixed-unit hidden-rate flip crossing')
   result.update(passed=True,status='exact_local_physical_stability_corner',physical_gradient=z['physical_gradient'],determinant=z['determinant'],flip_derivative_along_unit_hidden=z['flip_derivative_along_unit_hidden'],primitive_period=18,stable_complement_dimension=10,complement_bounds=prior['complement_bounds'],scope=z['scope'])
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc(),status='failed_or_interrupted')
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}))
 print(json.dumps(pack(result)));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
