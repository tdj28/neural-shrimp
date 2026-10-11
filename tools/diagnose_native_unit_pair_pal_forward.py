#!/usr/bin/env python3
"""Fixed immutable saved-domain conditioning diagnostic; zero native calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_pal as C
from neural_shrimp.unit_pair_pal_forward_diagnosis import diagnose,project_geometry
import probe_native_unit_pair_pal_batch as B
from neural_shrimp.unit_pair_conditioning import middle_radius
BASE=C.BASE;PARENT=B.BASE;OUT=BASE/'batch001-diagnosis/run'
PIN='2c116e332c8c8e85edda2d0c14329f9ee97ec0f8c1ab006f78cda3e99deb8cff'
SOURCES=list(dict.fromkeys(B.SOURCES+['src/neural_shrimp/unit_pair_pal_forward_diagnosis.py','tools/diagnose_native_unit_pair_pal_forward.py','tools/launch_native_unit_pair_pal_forward_diagnosis.py','tests/test_unit_pair_pal_forward_diagnosis.py','docs/plans/native-unit-pair-pal-forward-diagnosis.md']))
def main():
 C.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');C.require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 def stop(*_):raise TimeoutError('60s saved diagnostic')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 def save(name,row):
  raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode();C.require(used[0]+len(raw)<16*1024**2,'16MiB diagnostic cap')
  with(OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:C.entry(ROOT/n)for n in SOURCES},'threads':dict.fromkeys(C.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'model_calls':0})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'native_calls':0}
 try:
  C.inventory(PARENT,PIN);batch=C.read(PARENT/'outcome.json')
  C.require(batch['commit']=='cb83dcc7677e5e67fabe7e2f86fd8e7255395265'and batch['passed']and batch['connected_count']==4 and batch['connected_right']=='319/4096'and not batch['owned_live_pids'],'complete connected exact four-chart batch')
  for k,(worker,join)in enumerate(zip(batch['workers'],batch['joins'],strict=True),1):
   C.require(worker['unit']==join['unit']==k and worker['exit_code']==0 and not worker['owned_live_pids']and worker['stop_reason']is None and join['passed']and join['coordinates']==705,'all complete native joins')
   child=PARENT/f'chart{k:03d}';C.inventory(child/'run',join['worker_manifest']['sha256']);C.archive_bound(child,batch['commit'])
   rt=C.read(child/'run/runtime.json');out=C.read(child/'run/outcome.json')
   C.require(rt['settings']==B.settings(k)and out['passed']and out['ticks']==253 and out['strict_inclusions']==705,'full indexed source-bound acquisition')
  chart=C.read(PARENT/'chart004/run/certificate.json.gz');C.require(chart['parameter_center']=='287/4096'and chart['parameter_endpoints']==['255/4096','319/4096'],'fixed last connected chart')
  answer=diagnose(chart);save('diagnostic.json',answer)
  # Numerical anatomy of the already retained complete native scout/section.
  scout=BASE.parent/'scout/attempt002-target';section=BASE.parent/'section/attempt001-target'
  pins=[(scout,'4941c57ff6a37a7574f7300b17b5980aafad35bd27e6091ba0f99bdf724403f0','dde339494bc7b5ff77eda74cfde8d38cc9a250db24d15ffe57187a327582dddd','3c456e977f3b9ea3a34e05efcee3b8e141c76fad',20),(section,'2cde595d4b05e4e65b7f5ca4c2a394609bf98493219b2d1486bddb5223500d40','26ce0c94f3c27aa4f05d2f23eabb3c05449a9c2f05e4107f877c6659414187d2','f70f3e6a9696c89f4aed0c2398dbfd3b78ec1020',33)]
  surveys=[]
  for root,outer,worker,source,count in pins:
   C.inventory(root,outer);C.inventory(root/'run',worker);C.archive_bound(root,source);la=C.read(root/'launcher-result.json');out=C.read(root/'run/outcome.json');survey=C.read(root/'run/survey.json')
   C.require(la['exit_code']==0 and not la['owned_live_pids']and out['passed']and out['accepted_points']==len(survey['points'])==count,'complete original native numerical survey')
   C.require(all(p['diagnostics']['passed']and all(p['diagnostics']['gates'].values())for p in survey['points']),'qualified saved numerical contexts');surveys.append(survey)
  inp=C.read(PARENT/'chart004/run/inputs.json')['map_inputs'];old=C.read(scout/'run/inputs.json');sec=C.read(section/'run/inputs.json')
  C.require(inp==old['map_inputs']and sec['original_finite_proof_binding']==old,'same actual map and original native endpoint')
  C.require(len(surveys[1]['reverse_prefix'])==19 and all(p['association']['passed']for p in surveys[1]['reverse_prefix']),'retained full reverse old prefix')
  # Both surveys include their same join. Retain it twice rather than drop evidence.
  geometry=project_geometry(surveys[0]['points']+surveys[1]['points'],[float(v)for v in map(Fraction,chart['normal'])],[float(v)for v in map(Fraction,chart['origin'])]);save('geometry.json',geometry)
  result.update(passed=True,scaling={k:v for k,v in answer['scaling'].items()if k!='weights'},groups=answer['groups'],resolvent_available='resolvent_proposal'in answer,geometry_points=len(geometry['points']),projection_reversal_proposals=geometry['projection_reversal_proposals'],scope='Saved conditioning and geometry proposals only; no new acquisition or proof.')
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if result['seconds']>=60 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):C.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
