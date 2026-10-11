#!/usr/bin/env python3
"""Fixed immutable saved-domain conditioning diagnostic; zero native calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,json,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_pal as C
from neural_shrimp.unit_pair_pal_route_diagnosis import diagnose,route_lengths
import probe_native705_rotated as W
import probe_native_unit_pair_pal_batch as B
from neural_shrimp.unit_pair_conditioning import middle_radius
BASE=C.BASE;PARENT=W.BASE;OUT=BASE/'rotated001-diagnosis/run'
PIN='1071d42d45a686b4d30869d13931a29fa4609db685bce24d17fb4070029e2643'
SOURCES=list(dict.fromkeys(W.SOURCES+['src/neural_shrimp/unit_pair_pal_route_diagnosis.py','tools/diagnose_native705_route.py','tools/launch_native705_route_diagnosis.py','tests/test_unit_pair_pal_route_diagnosis.py','docs/plans/native705-route-diagnosis.md']))
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
  C.inventory(PARENT,PIN);packet=C.read(PARENT/'outcome.json')
  C.require(packet['source']=='2c9fc56dd9e8fd75434970487d4a0fafb21fbe6d'and not packet['passed']and packet['connected_charts']==24 and len(packet['batches'])==4 and not packet['owned_live_pids'],'exact complete retained rotated prefix24/failure8')
  for i,batch in enumerate(packet['batches'],1):
   C.require(batch['batch']==i and batch['connected_count']==(8 if i<=3 else 0)and len(batch['workers'])==8 and not batch['owned_live_pids'],'complete ordered fixed packet')
   for unit in range(1,9):
    run=C.read(PARENT/f'batch{i:03d}/chart{unit:03d}/run/outcome.json')
    C.require(run['batch']==i and run['unit']==unit and run['ticks']==253 and run['events']['direction_return']==144 and run['events']['pal_path_phase_return']==108 and run['passed']==(i<=3),'all32 actual complete native attempts')
  chosen=PARENT/'batch004/chart001';C.archive_bound(chosen,packet['source']);chart=C.read(chosen/'run/certificate.json.gz')
  C.require(chart['parameter_center']=='15/2048'and chart['parameter_endpoints']==['-1/2048','31/2048'],'fixed firstfailed rotated chart')
  answer=diagnose(chart);save('diagnostic.json',answer);del chart
  # Numerical anatomy of the already retained complete native scout/section.
  scout=BASE.parent/'scout/attempt002-target';section=BASE.parent/'section/attempt001-target'
  pins=[(scout,'4941c57ff6a37a7574f7300b17b5980aafad35bd27e6091ba0f99bdf724403f0','dde339494bc7b5ff77eda74cfde8d38cc9a250db24d15ffe57187a327582dddd','3c456e977f3b9ea3a34e05efcee3b8e141c76fad',20),(section,'2cde595d4b05e4e65b7f5ca4c2a394609bf98493219b2d1486bddb5223500d40','26ce0c94f3c27aa4f05d2f23eabb3c05449a9c2f05e4107f877c6659414187d2','f70f3e6a9696c89f4aed0c2398dbfd3b78ec1020',33)]
  surveys=[]
  for root,outer,worker,source,count in pins:
   C.inventory(root,outer);C.inventory(root/'run',worker);C.archive_bound(root,source);la=C.read(root/'launcher-result.json');out=C.read(root/'run/outcome.json');survey=C.read(root/'run/survey.json')
   C.require(la['exit_code']==0 and not la['owned_live_pids']and out['passed']and out['accepted_points']==len(survey['points'])==count,'complete original native numerical survey')
   C.require(all(p['diagnostics']['passed']and all(p['diagnostics']['gates'].values())for p in survey['points']),'qualified saved numerical contexts');surveys.append(survey)
  inp=C.read(chosen/'run/inputs.json')['map_inputs'];old=C.read(scout/'run/inputs.json');sec=C.read(section/'run/inputs.json')
  C.require(inp==old['map_inputs']and sec['original_finite_proof_binding']==old,'same actual map and original native endpoint')
  C.require(len(surveys[1]['reverse_prefix'])==19 and all(p['association']['passed']for p in surveys[1]['reverse_prefix']),'retained full reverse old prefix')
  # Both surveys include their same join. Retain it twice rather than drop evidence.
  geometry=route_lengths(surveys[0]['points']+surveys[1]['points']);save('geometry.json',geometry)
  result.update(passed=True,scaling={k:v for k,v in answer['scaling'].items()if k!='weights'},groups=answer['groups'],resolvent_available='resolvent_proposal'in answer,geometry_points=len(geometry['points']),flip_bracket_polygonal_lengths=[geometry['points'][i]['polygonal_arclength']for i in (46,47)],scope='Saved conditioning and full-native polygonal geometry proposals only; no new acquisition or proof.')

 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  if result['seconds']>=60 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
  save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):C.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
  for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  OUT.chmod(0o555)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
