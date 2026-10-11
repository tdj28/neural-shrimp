"""Saved-only enclosure-width diagnosis of the failed fixed72D period58 certificate."""
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/census-period58/attempt001-target')
PIN='c0f7488ef7836f9c3f631e1636b9f7ca28a85191e7f1834d3060451fb9f45fc8';SOURCE='3488be385d98b6c38a2280adca7e509ea8ff72da'
FILES=['tools/diagnose_census_long_period.py','tools/launch_census_long_period_diagnosis.py','tests/test_census_long_period_diagnosis.py','docs/plans/census-long-period-diagnosis.md']
def need(v,m):
 if not v:raise ValueError(m)
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_bytes())
def exponent(pair):
 need(isinstance(pair,list)and len(pair)==2 and all(type(v)is int for v in pair),'dyadic pair');n,e=pair
 return None if n==0 else abs(n).bit_length()-1+e

def summary(balls):
 finite=[v for v in balls if v.get('finite',True)]
 need(all('mid'in v and 'rad'in v and v['rad'][0]>=0 for v in finite),'finite ball encoding')
 mids=[exponent(v['mid'])for v in finite];rads=[exponent(v['rad'])for v in finite]
 return dict(count=len(balls),finite_count=len(finite),nonfinite_count=len(balls)-len(finite),maximum_midpoint_floor_log2=max((v for v in mids if v is not None),default=None),maximum_radius_floor_log2=max((v for v in rads if v is not None),default=None))

def diagnose(r):
 need(r['settings']==dict(precision=512,radius_exponent=-160,newton_steps=5,max_power=4096),'exact failed settings')
 need(not r['passed']and len(r['center'])==len(r['box'])==72 and len(r['orbit_enclosures'])==59 and len(r['jacobians'])==58,'complete failed native72/58 proof')
 rows=[dict(phase=i,state=summary(row),jacobian=None if i==58 else summary([v for line in r['jacobians'][i]for v in line]))for i,row in enumerate(r['orbit_enclosures'])]
 first_state=next((v['phase']for v in rows if v['state']['nonfinite_count']),None);first_jac=next((v['phase']for v in rows if v['jacobian']and v['jacobian']['nonfinite_count']),None)
 wide=next((v['phase']for v in rows if v['state']['maximum_radius_floor_log2']is not None and v['state']['maximum_radius_floor_log2']>=-20),None)
 initial=rows[0]['state']['maximum_radius_floor_log2'];finite=[v for v in rows if v['state']['nonfinite_count']==0]
 return dict(phases=rows,center=summary(r['center']),center_residual=summary([v for line in r['residual']for v in line]),preconditioner=summary([v for line in r['preconditioner']for v in line]),first_nonfinite_state=first_state,first_nonfinite_jacobian=first_jac,first_radius_at_least_2_to_minus20=wide,initial_radius_floor_log2=initial,last_fully_finite_phase=finite[-1]['phase']if finite else None,divisor_records=r['divisors'],power_bounds=r['power_bounds'],contraction=r['contraction_norm_upper'],newton_history_available=False,newton_history_note='The unchanged original C1 kernel saved the refined center and its final residual, but no intermediate Newton residual history. No missing history is reconstructed or inferred.',scope='Exact saved dyadic floor-log2 radius/midpoint summaries and retained failure diagnosis only. No new native calls or proof verdict.')

def main(out):
 need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0
 def stop(*_):raise TimeoutError('30s saved diagnosis')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 def save(name,row):
  nonlocal used
  b=(json.dumps(row,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<4*1024**2,'4MiB raw cap');used+=len(b)
  with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 rt=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),model_calls=0,derivative_calls=0,sources={n:entry(ROOT/n)for n in FILES});save('runtime.json',rt);result={'passed':False}
 try:
  need(entry(PARENT/'manifest.json')['sha256']==PIN,'exact failed outer inventory');inv=read(PARENT/'manifest.json')['files'];need(set(inv)=={str(p.relative_to(PARENT))for p in PARENT.rglob('*')if p.is_file()and p!=PARENT/'manifest.json'},'complete closed inventory')
  for name,item in inv.items():
   p=PARENT/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and entry(p)==item and not p.stat().st_mode&0o222,'immutable original file')
  la=read(PARENT/'launcher-result.json');rr=read(PARENT/'run/runtime-start.json');done=read(PARENT/'run/result.json');need(la['source_commit']==rr['source_commit']==SOURCE and la['exit_code']==2 and la['owned_live_pids']==[]and not done['passed'],'preserve failed completed execution')
  for name,digest in rr['source_sha256'].items():need(entry(PARENT/'run/source'/name)['sha256']==digest==hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)).hexdigest(),'historical source')
  c=read(PARENT/'run/certificate/certificate.json');need(c['runtime']['source_commit']==SOURCE,'exact certificate');answer=diagnose(c['result']);save('diagnosis.json',answer);save('binding.json',dict(root=str(PARENT),outer=PIN,source=SOURCE,certificate=entry(PARENT/'run/certificate/certificate.json')));result.update(passed=True,first_nonfinite_state=answer['first_nonfinite_state'],first_nonfinite_jacobian=answer['first_nonfinite_jacobian'],center_residual=answer['center_residual'])
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=result['passed']and result['elapsed_seconds']<30 and result['peak_rss_kib']<2*1024**2;save('outcome.json',result);save('manifest.json',dict(files={p.name:entry(p)for p in out.iterdir()if p.is_file()}))
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
