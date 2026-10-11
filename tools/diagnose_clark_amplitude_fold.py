#!/usr/bin/env python3
"""Saved-only reconstruction of failed paired inverse bounds; no model calls."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,vector_json,matrix_json,canonical_bytes
from neural_shrimp.rigorous_clark_amplitude_fold import require as need
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/amplitude-fold-bridge/attempt001')
SOURCE='8c61057063a9415b6b0656d0bb07d404f62cd20d';PINS=['0a991b99d5069b30def61f09b274605d0ed13d29df87e7bd6a7febe4258698e3','7688e55eb82486e57dccbd6d9a153423e06165e5dee992d321f07c56f1bd5544']
OUT=Path('/data2/neural-shrimp-data/audits/clark-amplitude-inverse/attempt001/run')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/diagnose_clark_amplitude_fold.py','tools/launch_clark_amplitude_diagnosis.py','tests/test_clark_amplitude_diagnosis.py','docs/plans/clark-amplitude-inverse-diagnosis.md','docs/reviews/clark-amplitude-fold-attempt001.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 import hashlib
 return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(gzip.decompress(p.read_bytes())if p.name.endswith('.gz')else p.read_bytes())
def decode(v):return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def matrix(v):return arb_mat([[decode(x)for x in row]for row in v])

def reconstruct(attempt,phases):
 need(len(phases)==3 and [x['phase']for x in phases]==[0,1,2]and len(attempt['input'])==26,'full failed first pass only');z=[decode(x)for x in attempt['input']];p=phases[-1]
 for phase in phases:
  need(len(phase['even'])==len(phase['odd'])==12 and all(set(x)=={'f','g','d','m'}and len(x['g'])==len(x['m'])==27 for x in phase['even']+phase['odd']),'actual full native mixed jet fields')
  need(all(decode(x['d']).is_zero()and all(decode(v).is_zero()for v in x['m'])for x in phase['even']+phase['odd']),'original first pass direction exactly zero')
 even=[[decode(v)for v in x['g']]for x in p['even']];odd=[[decode(v)for v in x['g']]for x in p['odd']]
 A=arb_mat([[-even[i][k]-int(i==k)for k in range(25)]for i in range(12)]+[[-odd[i][k]+int(k==13+i)for k in range(25)]for i in range(12)]+[[z[k]if 13<=k<25 else arb(0)for k in range(25)]])
 hb=arb_mat([[-even[i][26]]for i in range(12)]+[[-odd[i][26]]for i in range(12)]+[[arb(0)]])
 need(all(v.is_finite()for row in A.tolist()+hb.tolist()for v in row),'finite full paired reconstruction');return A,hb


def diagnose(A):
 n=A.nrows();C=midpoint_matrix(midpoint_matrix(A).inv());D=eye(n)-C*A;R=np.array([[float(abs(D[i,j]).upper())for j in range(n)]for i in range(n)]);rho=float(max(abs(np.linalg.eigvals(R))));rows=np.sum(R,axis=1);norm=norm_inf_upper(D);proposal=None;weighted=None
 try:
  u=np.linalg.solve(np.eye(n)-R,np.ones(n))
  if np.isfinite(u).all()and np.all(u>0):
   from fractions import Fraction
   weights=[arb(str(Fraction.from_float(float(x))))for x in u];weighted=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));proposal=vector_json(weights)
 except np.linalg.LinAlgError:pass
 return dict(unweighted_bound=ball_json(norm),floating_absolute_spectral_radius=rho,largest_rows=[dict(index=int(i),sum=float(rows[i]))for i in np.argsort(-rows)[:6]],finite_weight_proposal=proposal,rigorous_weighted_bound=None if weighted is None else ball_json(weighted),weighted_inverse_only=bool(weighted is not None and weighted<1),preconditioner=matrix_json(C),defect=matrix_json(D),scope='Saved actual first-pass Jacobian only. Floating spectral radius/resolvent are diagnosis/proposals; if weighted_bound<1 it certifies only this internal matrix inverse, not unacquired mixed derivatives, fold root or branch.')


def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal inventory');m=read(root/'manifest.json')['files'];files={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(m),'complete immutable files')
 for n,p in files.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==m[n],'retained immutable '+n);tick()

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean committed CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'30s512MiB cap')
 def stop(*_):raise TimeoutError('saved diagnosis deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 def save(name,value,closing=False):
  nonlocal used
  raw=canonical_bytes(value);need(used+len(raw)<(4 if closing else 3)*1024**2,'4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,pid=os.getpid(),precision=384,native_calls=0,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,flint=flint.__FLINT_VERSION__,python_flint=flint.__version__))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT),'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);used+=len(raw)
  inventory(PARENT,PINS[0],tick);inventory(PARENT/'run',PINS[1],tick);rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');continuation=read(PARENT/'run/continuation.json')
  need(rt['source_commit']==la['source_commit']==SOURCE and la['exit_code']==2 and not la['owned_live_pids']and not out['passed']and out['calls']==dict(context=1,events=214,family=24,native_phase_returns=135),'literal retained execution')
  for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'old owned process gone')
  for n,pin in rt['sources'].items():need(entry(PARENT/'run/source'/n)==pin and __import__('hashlib').sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'complete historic source '+n)
  events=sorted((PARENT/'run/events').glob('*.json.gz'));need(len(events)==214 and [int(p.name.split('-')[0])for p in events]==list(range(1,215)),'complete original event stream');groups=[];current=None
  for p in events:
   kind=p.name.split('-',1)[1].removesuffix('.json.gz');v=read(p)
   if kind=='family_attempt':
    current=dict(input=v,phases=[],complete=False,files=[dict(path=str(p.relative_to(PARENT)),identity=entry(p))]);groups.append(current)
   elif kind=='native_even_odd_phase':need(current is not None,'associated native phase');current['phases'].append(v);current['files'].append(dict(path=str(p.relative_to(PARENT)),identity=entry(p)))
   elif kind=='amplitude_fold_return':need(current is not None and len(current['phases'])==6,'complete double pass');current['complete']=True
  failures=[x for x in groups if not x['complete']];need(len(groups)==24 and len(failures)==5 and continuation['accepted']==0 and len(continuation['attempts'])==5,'all five exact failed first-pass acquisitions')
  answers=[]
  with ctx.workprec(384):
   for i,(group,decision)in enumerate(zip(failures,continuation['attempts']),1):
    need(decision['error']=='ValueError: strict full paired inverse bound','literal failed internal bound');A,hb=reconstruct(group['input'],group['phases']);report=diagnose(A);need(decode(report['unweighted_bound'])>=1,'reproduced original inverse rejection');save(f'failure-{i}.json',dict(input=group['input'],files=group['files'],matrix=matrix_json(A),amplitude_derivative=matrix_json(hb),diagnosis=report));answers.append(dict(attempt=i,halfwidth=decision['halfwidth'],**{k:v for k,v in report.items()if k not in('preconditioner','defect','finite_weight_proposal')}));tick()
  save('input-binding.json',dict(root=str(PARENT),source=SOURCE,pins=PINS,events=214,family_attempts=24,failed_first_passes=5));result.update(passed=True,status='saved_internal_inverse_diagnosis_complete',answers=answers,scope='No root/nonexistence or branch claim; saved first-pass matrices only. Fresh complete native derivative acquisitions are required for any new proof.')
 except BaseException as error:result.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
