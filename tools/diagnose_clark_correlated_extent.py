"""Saved coefficient/radius diagnosis of fixed32 extent attempts; no native calls."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
import gzip,hashlib,json,math,platform,signal,subprocess,sys,time,traceback
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/data2/neural-shrimp-data/audits/clark-correlated-extent-diagnosis/attempt001')
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-extent/attempt001')
SOURCE='b1fc38c14906d6c2a6ef16fab7e82bbbe2f62676';PINS=['71e8f0e318d1427950e63aba0466efe213808c4336788277843919fbc48875dd','2812d71cd15953a3cc19772fdd5b488536ec885f8819bf50267adbb1b4cea4ab']
CARD='docs/plans/clark-correlated-extent-diagnosis-inputs.json'
CAPS=dict(seconds=30,rss_bytes=512*1024**2,as_bytes=2*1024**3,raw_bytes=8*1024**2,native_calls=0)
FILES=[CARD,'docs/plans/clark-correlated-extent-diagnosis.md','tools/diagnose_clark_correlated_extent.py','tools/launch_clark_correlated_extent_diagnosis.py','tests/test_clark_correlated_extent_diagnosis.py','docs/reviews/clark-correlated-extent-outcome.json']
def need(v,m):
 if not v:raise ValueError(m)
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):
 if p.name.endswith('.gz'):
  with gzip.open(p,'rb')as f:raw=f.read(32*1024**2+1)
  need(len(raw)<=32*1024**2,'32MiB decoded proof cap');return json.loads(raw)
 return json.loads(p.read_bytes())
def dyad(v):
 need(isinstance(v,list)and len(v)==2 and all(type(x)is int for x in v)and abs(v[1])<100000,'finite exact dyadic')
 return Fraction(v[0])*Fraction(2)**v[1]
def ball(v):
 m,r=dyad(v['mid']),dyad(v['rad']);need(r>=0,'nonnegative saved radius');return m,r
def bound(v):m,r=ball(v);return abs(m)+r
def scalar(v):m,r=ball(v);need(r==0,'exact saved scalar');return m
def group(i):return 'state_rate_mode'if i<26 else'paired_tangent'
def diagnose(p):
 n=len(p['radii']);need(n==51 and len(p['image_correction'])==len(p['preconditioned_residual'])==n and len(p['whole_defect'])==n,'all51 saved bound coordinates')
 radii=[scalar(v)for v in p['radii']];h=scalar(p['halfwidth']);need(h>0 and all(r>0 for r in radii),'positive exact original geometry')
 D=[[bound(x)for x in row]for row in p['whole_defect']];need(all(len(row)==n for row in D),'full51 defect rows')
 residual=[bound(row[0])/radii[i]for i,row in enumerate(p['preconditioned_residual'])];image=[bound(row[0])/radii[i]for i,row in enumerate(p['image_correction'])];defect=[sum(D[i][j]*radii[j]for j in range(n))/radii[i]for i in range(n)]
 includes=[x<1 for x in image];need(includes==p['coordinate_inclusions']and bool(all(includes)and bound(p['contraction'])<1)==p['passed'],'literal saved finite gates')
 q=bound(p['contraction']);need(max(defect)<=q*(1+Fraction(1,2**24)),'saved contraction agrees with decoded exact row sum')
 # Individual powers are triangle upper magnitudes, a diagnostic decomposition;
 # the saved whole polynomial bound retains more sign/parameter correlation.
 fp=[[bound(m[i][0])*h**k/radii[i]for i in range(n)]for k,m in enumerate(p['forcing_coefficients'])]
 dp=[[sum(bound(m[i][j])*radii[j]for j in range(n))*h**k/radii[i]for i in range(n)]for k,m in enumerate(p['defect_coefficients'])]
 need(len(fp)==5 and len(dp)==4,'complete degree4 forcing/degree3 defect')
 rows=[dict(index=i,group=group(i),radius=float(radii[i]),residual_over_radius=float(residual[i]),defect_row=float(defect[i]),image_over_radius=float(image[i]),included=includes[i],forcing_term_magnitudes=[float(x[i])for x in fp],defect_term_magnitudes=[float(x[i])for x in dp])for i in range(n)]
 need(all(math.isfinite(x)for row in rows for x in[row['radius'],row['residual_over_radius'],row['defect_row'],row['image_over_radius']]+row['forcing_term_magnitudes']+row['defect_term_magnitudes']),'finite diagnostic floats')
 return dict(passed_saved=p['passed'],halfwidth=float(h),parameter=float(scalar(p['parameter'])),saved_contraction=float(q),exact_decoded_max_defect=float(max(defect)),inclusions=sum(includes),max_residual_over_radius=float(max(residual)),max_image_over_radius=float(max(image)),largest=sorted(rows,key=lambda x:x['image_over_radius'],reverse=True)[:8],all_coordinates=rows,scope='Saved exact dyadic gate consistency and floating contribution diagnostics only. No new derivative, root, expanded-domain or optimized-radius proof.')
class Work:
 def __init__(self):self.root=OUT/'run';self.root.mkdir(parents=True,exist_ok=False);self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'30s512MiB')
 def save(self,n,v,closing=False):
  raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();need(self.used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB reserve');p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
def inventory(root,pin,w):
 need(entry(root/'manifest.json')['sha256']==pin,'literal full inventory');files=read(root/'manifest.json')['files'];need(set(files)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete inventory')
 for n,e in files.items():
  w.tick();p=root/n;need(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==e,'immutable exact '+n)
 return files
def bind(w):
 inventory(PARENT,PINS[0],w);files=inventory(PARENT/'run',PINS[1],w);rt=read(PARENT/'run/runtime.json');lr=read(PARENT/'launcher-result.json');o=read(PARENT/'run/outcome.json');co=read(PARENT/'run/continuation.json')
 need(rt['source_commit']==lr['source_commit']==SOURCE and lr['exit_code']==2 and lr['failure']is None and not lr['owned_live_pids']and lr['worker_pid_absent']and not o['passed']and o['error']=='ValueError: bounded finite extent incomplete'and o['calls']['native_phase_returns']==906 and co['accepted']==22 and len(co['attempts'])==32,'complete actual fixed32 partial prefix')
 for pid in(rt['pid'],lr['launcher_pid'],lr['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'old owners absent')
 for n,e in rt['sources'].items():w.tick();need(entry(PARENT/'run/source'/n)==e and hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'historical full source')
 events={n for n in files if n.startswith('events/')};need(len(events)==2257 and sorted(int(Path(n).name.split('-')[0])for n in events)==list(range(1,2258)),'all2257 original event inventories');w.save('input-binding.json',dict(root=str(PARENT),source=SOURCE,pins=PINS,events=2257,native_phases=906,attempts=32,accepted=22));return co

def synthetic():
 b=lambda x:dict(mid=[x,0],rad=[0,0]);z=b(0);n=51;matrix=lambda d:[[b(d if i==j else 0)for j in range(n)]for i in range(n)];column=lambda x:[[b(x)]for _ in range(n)]
 small=dict(mid=[2**383+123456789,-405],rad=[7,-430]);dense=[[small for _ in range(n)]for _ in range(n)]
 return dict(radii=[b(16)]*n,halfwidth=b(1),parameter=b(2),whole_defect=dense,preconditioned_residual=column(1),image_correction=column(1),contraction=dict(mid=[1,-10],rad=[0,0]),coordinate_inclusions=[True]*n,passed=True,forcing_coefficients=[column(1)]+[column(0)]*4,defect_coefficients=[dense]*4)
def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');need(read(ROOT/CARD)==dict(status='reviewed',parent=str(PARENT),source=SOURCE,pins=PINS,caps=CAPS),'literal reviewed fixed diagnosis');w=Work();out=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('saved30s deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');p=w.root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  start=time.monotonic();diagnose(synthetic());seconds=time.monotonic()-start;forecast=2*32*seconds+10;w.save('profile.json',dict(seconds=seconds,forecast_seconds=forecast,passed=forecast<30,target_arrays_unopened=True));need(forecast<30,'full51 exact-ratio synthetic admission')
  co=bind(w);rows=[]
  for d in co['attempts']:
   w.tick();p=PARENT/'run/charts'/f"{d['attempt']:02d}"/'tube.json.gz';proof=read(p);need(proof['halfwidth']==d['halfwidth']and proof['passed']==d['passed'],'literal decision geometry/pass');answer=diagnose(proof);w.save(f"attempt-{d['attempt']:02d}.json",dict(attempt=d['attempt'],file=entry(p),diagnosis=answer));rows.append(dict(attempt=d['attempt'],**{k:v for k,v in answer.items()if k not in('all_coordinates','largest','scope')}))
  out.update(passed=True,attempts=32,qualified=22,whole_bound_failures=10,rows=rows,scope='Saved bounds diagnosis only; independent prefix proof replay and every new-domain acquisition remain separate.');w.tick()
 except BaseException as e:out.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out['passed']=bool(out['passed']and out['seconds']<30 and out['peak_rss_kib']<512*1024);w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(w.root)):entry(p)for p in w.root.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:out.get(k)for k in('passed','seconds','error')}));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
