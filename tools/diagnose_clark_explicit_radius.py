#!/usr/bin/env python3
"""Saved-only51 residual versus known-anchor radius diagnosis; no native calls."""
import gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import flint
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,vector_json,matrix_json,canonical_bytes
from neural_shrimp.rigorous_clark_amplitude_fold import require as need
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/explicit-fold-bridge/attempt002')
SOURCE='9d0c4a1c6094e7ee897b1944e9e169b1ef65d7ec';PINS=['370c9c06e8a238a0bffd54ccb5e29f32af68355c5a8bacfab3ad8a962fcde86c','736c170618362558570b2c232058b944ef45dc8d243593ab63e083935b8cc817']
OUT=Path('/data2/neural-shrimp-data/audits/clark-explicit-radius/attempt001/run')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/diagnose_clark_explicit_radius.py','tools/launch_clark_explicit_radius.py','tests/test_clark_explicit_radius.py','docs/plans/clark-explicit-radius-diagnosis.md','docs/reviews/clark-explicit-fold-attempt002.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 import hashlib
 return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(gzip.decompress(p.read_bytes())if p.name.endswith('.gz')else p.read_bytes())
def decode(v):return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))
def matrix(v):return arb_mat([[decode(x)for x in row]for row in v])

def col(v):return arb_mat([[x]for x in v])
def vec(v):return [v[i,0]for i in range(v.nrows())]
def diagnose(center,predictor,known,decision):
 c=[decode(x)for x in center['input']];p=[decode(x)for x in predictor['input']];bc=decode(center['squared_amplitude']);h=decode(decision['halfwidth']);left=decode(decision['left']);d=arb(0,h);N=len(c)
 need(N==51 and len(p)==len(known)==51,'all51 saved coordinates');J=matrix(center['jacobian']);f=matrix(center['value']);hb=matrix(center['parameter_derivative']);Jp=matrix(predictor['jacobian']);bp=matrix(predictor['parameter_derivative'])
 need(J.nrows()==J.ncols()==Jp.nrows()==Jp.ncols()==51 and f.nrows()==hb.nrows()==bp.nrows()==51 and bc==left+h,'actual complete51 center/predictor derivatives')
 C=midpoint_matrix(midpoint_matrix(J).inv());slope=[x.mid()for x in vec(-C*hb)]
 need(all((p[i]-c[i]-slope[i]*d).contains(0)for i in range(51))and (decode(predictor['squared_amplitude'])-bc-d).contains(0),'associated literal finite predictor consistency')
 constant=-C*f;variation=-C*(Jp*col(slope)+bp)*d;residual=constant+variation;anchor=[known[i]-c[i]-slope[i]*(left-bc)for i in range(51)];needed=[max((8*abs(residual[i,0])).upper(),(4*abs(anchor[i])).upper())for i in range(51)];ceil=[arb((1,-8 if i<26 else 2))for i in range(51)]
 need(all(x.is_finite()for x in needed),'finite complete saved radius quantities');ratios=[float((needed[i]/ceil[i]).upper())for i in range(51)];violations=[i for i in range(51)if needed[i]>=ceil[i]]
 defect=eye(51)-C*Jp;absolute=np.array([[float(abs(defect[i,k]).upper())for k in range(51)]for i in range(51)]);rho=float(max(abs(np.linalg.eigvals(absolute))));top=sorted(range(51),key=lambda i:ratios[i],reverse=True)[:8]
 return dict(halfwidth=decision['halfwidth'],reproduced_radius_rejection=bool(violations),violating_coordinates=violations,center_residual=matrix_json(constant),parameter_variation=matrix_json(variation),total_preconditioned_residual=matrix_json(residual),known_root_projection=vector_json(anchor),minimum_radius_proposal=vector_json(needed),declared_ceilings=vector_json(ceil),predictor_only_defect_norm=ball_json(norm_inf_upper(defect)),predictor_only_floating_absolute_radius=rho,largest=[dict(index=i,group='state_mode_rate'if i<26 else'paired_tangent',ratio=ratios[i],residual_requirement=ball_json((8*abs(residual[i,0])).upper()),anchor_requirement=ball_json((4*abs(anchor[i])).upper()),ceiling=ball_json(ceil[i]))for i in top],scope='Saved center and predictor only; no complete expanded error-box Jacobian exists. Minimum radii and floating predictor defect diagnose proposal failure, never root/contraction or nonexistence authority.')

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
  need(rt['source_commit']==la['source_commit']==SOURCE and la['exit_code']==2 and not la['owned_live_pids']and not out['passed']and out['calls']==dict(anchor=2,context=1,events=157,explicit=24,implicit=1,native_phase_returns=87),'literal retained execution')
  for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'old owned process gone')
  for n,pin in rt['sources'].items():need(entry(PARENT/'run/source'/n)==pin and __import__('hashlib').sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'complete historic source '+n)
  events=sorted((PARENT/'run/events').glob('*.json.gz'));need(len(events)==157 and [int(p.name.split('-')[0])for p in events]==list(range(1,158)),'complete original157 event stream');returns=[(p,read(p))for p in events if p.name.endswith('-explicit_amplitude_fold_return.json.gz')];need(len(returns)==24 and continuation['accepted']==0 and len(continuation['attempts'])==5,'all24 returns and5 rejected proposals')
  anchor=read(PARENT/'run/explicit-anchor.json');answers=[]
  with ctx.workprec(384):
   known=[decode(x)for x in anchor['known']]
   for i,decision in enumerate(continuation['attempts']):
    need(decision['error']=='ValueError: fixed coordinate radius ceiling'and not decision['passed'],'literal proposal ceiling failure');group=returns[4+4*i:8+4*i];need(len(group)==4,'twoNewton pluscenter andpredictor')
    center,predictor=group[2][1],group[3][1];report=diagnose(center,predictor,known,decision);need(report['reproduced_radius_rejection'],'reproduced actual saved radius rejection');save(f'failure-{i+1}.json',dict(files=[dict(path=str(p.relative_to(PARENT)),identity=entry(p))for p,_ in group],diagnosis=report));answers.append({k:v for k,v in report.items()if k not in('center_residual','parameter_variation','total_preconditioned_residual','known_root_projection','minimum_radius_proposal','declared_ceilings')});tick()
  save('input-binding.json',dict(root=str(PARENT),source=SOURCE,pins=PINS,events=157,explicit_returns=24,original_anchor=entry(PARENT/'run/explicit-anchor.json'),failed_proposals=5));result.update(passed=True,status='saved_explicit_radius_diagnosis_complete',answers=answers,scope='Zero native calls; saved51 center/predictor residual and exact known-root projection only. Any new proof requires fresh complete wholeE acquisitions and unchanged identity gates.')
 except BaseException as error:result.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
