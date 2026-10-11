"""Fixed consumed census period58/72D pilot using unchanged Arb orbit kernel."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import certify_c1 as C
from neural_shrimp.rigorous_orbits import BallTanhMap,canonical_bytes,exact,sha256,matrix_json,vector_json
CARD='docs/plans/census-long-period-inputs.json';PLAN='docs/plans/census-long-period.md'
SOURCES=C.SOURCES+['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py','tools/certify_census_long_period.py','tools/launch_census_long_period.py','tests/test_census_long_period.py',CARD,PLAN]
LIMIT=512*1024**2

def load(p):return json.loads(Path(p).read_text())
def require(v,m):
 if not v:raise ValueError(m)
def save(p,v):
 with p.open('xb')as f:f.write(canonical_bytes(v));f.flush();os.fsync(f.fileno())
def entry(p):return dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()))
def located(s):return ROOT/s[5:] if s.startswith('repo:')else Path(s)
def selected_indices(summary,audit,certs):
 keys={(r['map_id'],r['component'])for r in summary['gate']['qualifying']} & {(r['map_id'],r['component'])for r in audit['audit_gate']['qualifying']}
 eligible=[(i,r)for i,r in enumerate(certs)if(r['map_id'],r['component'])in keys]
 require(len(keys)==len(eligible)==61,'complete61agreedrepresentatives')
 largest=max(r['period']for i,r in eligible)
 return [i for i,r in eligible if r['period']==largest]
def prepare(card):
 paths={k:located(v['path'])for k,v in card['pins'].items()}
 for k,p in paths.items():require(entry(p)['sha256']==card['pins'][k]['sha256'],'inputpin '+k)
 rows=load(paths['certificates']);require(selected_indices(load(paths['summary']),load(paths['audit']),rows)==[card['row_index']],'unique fixed longestperiod selection')
 row=rows[card['row_index']];require(all(row[k]==card[k]for k in ['map_id','component','period']),'exact row identity')
 c=row['certificate'];require(c['verified']and c['attracting']and c['primitive_period']==58,'historical native qualification')
 orbit=np.asarray(c['orbit'],dtype=np.float64);require(orbit.shape==(58,72)and np.isfinite(orbit).all(),'finite native phase sequence')
 inputs=C.prepare(paths['certificates'],paths['plan'],card['row_index'],card['selection']);require(inputs['period']==58 and len(inputs['approximation'])==72,'expected exact map')
 return inputs,orbit

def source_binding():
 d={}
 for name in SOURCES:
  b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committedsource '+name);d[name]=sha256(b)
 return d

def profile_forecast(times,period):
 require(len(times)==3 and all(np.isfinite(v)and v>0 for v in times),'threepositivephasecosts')
 return 2*7*period*max(times)+30

def stop(*args):raise TimeoutError('bounded census pilot deadline')
def run(mode,out,profile=None):
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(LIMIT,LIMIT));signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,180 if mode=='profile'else 600)
 sources=source_binding();card=load(ROOT/CARD);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();result={'status':'started','mode':mode,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'settings':card['settings'],'passed':False,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},'gpu':False,'paid':False}
 save(out/'runtime-start.json',result)
 try:
  require(result['nice']>=10,'nice10');inputs,orbit=prepare(card);save(out/'inputs.json',inputs)
  for name in SOURCES:
   p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
  if mode=='profile':
   times=[]
   with ctx.workprec(card['settings']['precision']):
    model=BallTanhMap(inputs)
    for phase in [0,19,38]:
     state=[exact(C.dyadic(v))for v in orbit[phase]];t=time.monotonic();v,j=model.step(state);times.append(time.monotonic()-t);require(all(x.is_finite()for x in v)and all(j[i,k].is_finite()for i in range(72)for k in range(72)),'finite profile')
     save(out/f'phase-{phase}.json',{'state':vector_json(v),'jacobian':matrix_json(j)})
   forecast=profile_forecast(times,58);result.update(status='profile_complete',passed=bool(forecast<600),map_seconds=times,forecast_seconds=forecast,scope='Only native map/Jacobian cost at3saved phases; no orbit proof verdict.')
  else:
   require(profile is not None,'prior measured profile');m=load(profile/'manifest.json')['files'];require(set(m)=={str(p.relative_to(profile))for p in profile.rglob('*')if p.is_file()and p!=profile/'manifest.json'},'profilecompleteinventory')
   require(all(entry(profile/n)==v for n,v in m.items()),'profilefullhash');prior=load(profile/'result.json');require(prior['mode']=='profile'and prior['passed']and prior['source_sha256']==sources and prior['settings']==card['settings'],'same completedprofile')
   require((profile/'inputs.json').read_bytes()==canonical_bytes(inputs),'profile exactinputs');require(prior['forecast_seconds']==profile_forecast(prior['map_seconds'],58)<600,'measured target gate')
   code=C.run(inputs,out/'certificate',card['settings']);receipt=load(out/'certificate/receipt.json');result.update(status='certificate_complete',passed=bool(code==0),certificate_receipt=receipt,profile_manifest=entry(profile/'manifest.json'))
 except BaseException as e:result.update(status='failed',error=type(e).__name__+': '+str(e));raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save(out/'result.json',result);files={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()};require(sum(v['bytes']for v in files.values())<=LIMIT,'raw512MiBcap');save(out/'manifest.json',{'files':files})
 print(json.dumps({k:result[k]for k in ['status','passed','elapsed_seconds','max_rss_kib']}));return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);a=p.parse_args();raise SystemExit(run(a.mode,a.out,a.profile))
