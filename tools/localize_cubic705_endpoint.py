#!/usr/bin/env python3
"""Fixed saved-only known cubic endpoint localization and retained-tube join."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint
from flint import arb,arb_mat,ctx
import probe_native705_adaptive as W
from neural_shrimp.rigorous_cubic_endpoint import localize_cubic_endpoint
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json
BASE=W.P.BASE/'endpoint-localization001';TARGET=BASE/'attempt001/run';CAP=64*1024**2
AUDIT=Path('/data2/neural-shrimp-data/audits/native705-cubic-prefix-independent/attempt001')
AUDIT_SOURCE='f2f0d768542d3f2bafc93f445a1c928f3d019f81';AUDIT_OUTER='21c4c93586dfe0f71386eafb1dc39fa8de6925baf4c22837c40784125cdc27d2';AUDIT_WORKER='1623ae469c37d35ffc3957aa48b5961bd317317400735be9f413401282928b21';ENDPOINT='c853109173b17c21b33d15dc47f2d448913b262464e0aceb9ccb3aa20003c371'
PARENT=W.P.BASE/'narrow001/batch001';PARENT_SOURCE='6ae48eb09c199b7b2ce1a816b0355d35ce2265ec';PARENT_OUTER='4fe4f57aa561dec7230cc8586ca0f95ec4d4c1ebc015acad4d080d391740a289'
NEW=W.P.BASE/'adaptive002';NEW_SOURCE='c73978cb1bb7f747972cdf669a7f687a807b5b24';NEW_OUTER='58f6ee7fa434bb9df81383203e532ba7c3048b97edfc849d4937c43dbdace585';NEW_WORKER='83aace9dde32e7683e22767c99209361a867fc12f2e65aa26de010bfef79f641'
SOURCES=list(dict.fromkeys(W.SOURCES+['src/neural_shrimp/rigorous_cubic_endpoint.py','src/neural_shrimp/rigorous_parameter_localization.py','tools/localize_cubic705_endpoint.py','tools/launch_cubic705_endpoint.py','tests/test_rigorous_cubic_endpoint.py','tests/test_cubic705_endpoint_driver.py','docs/plans/native705-cubic-endpoint-localization.md','docs/reviews/native705-adaptive002-unjoined.json']))
need=W.require;read=W.read;entry=W.entry;decode=W.decode

def matrix_file(path,n,cols):
 data=read(path);need(data['encoding']=='mid-rad-dyadic-pairs'and len(data['rows'])==n and all(len(r)==cols for r in data['rows']),'full saved matrix dimensions')
 rows=data['rows'];out=[]
 for row in rows:
  values=[]
  for v in row:
   need(type(v)is list and len(v)==2 and all(type(x)is list and len(x)==2 and all(type(z)is int for z in x)for x in v)and v[1][0]>=0,'exact finite dyadic pair schema')
   values.append(arb(tuple(v[0]),arb(tuple(v[1]))))
  out.append(values)
 return arb_mat(out)

def admission(out,proof,endpoint):
 need(out['passed']and out['connected']and out['connected_count']==3 and out['connected_right']=='115/512'and out['native_calls']==0,'independently proved complete prefix')
 need(len(out['charts'])==3 and all(v['passed']and v['strict_inclusions']==v['anchor_coordinates']==705 for v in out['charts']),'three exact full roots and joins')
 need(proof['passed']and all(proof['gates'].values())and len(proof['root']['coordinate_inclusions'])==705 and all(proof['root']['coordinate_inclusions']),'last independently passed complete root')
 need(endpoint['chart']==proof['chart']and endpoint['endpoint']==proof['endpoints'][1]and endpoint['chart']['parameter_endpoints']==['83/512','115/512'],'literal independently recomputed right endpoint')
 return True

def bind(identity,save,check):
 for root,pin in ((AUDIT,AUDIT_OUTER),(AUDIT/'run',AUDIT_WORKER),(PARENT,PARENT_OUTER),(NEW,NEW_OUTER),(NEW/'attempt002/run',NEW_WORKER)):
  W.inventory(root,pin);check()
 rt=read(AUDIT/'run/runtime.json');need(rt['source_commit']==AUDIT_SOURCE and rt['precision']==480 and rt['native_calls']==0,'exact independent arithmetic runtime')
 for n,row in rt['sources'].items():need(entry(AUDIT/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',AUDIT_SOURCE+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'literal independent source '+n)
 outcome=read(AUDIT/'run/outcome.json');proof=read(AUDIT/'run/chart003/independent-proof.json');ep=read(AUDIT/'run/connected-endpoint.json');need(entry(AUDIT/'run/connected-endpoint.json')['sha256']==ENDPOINT,'exact independent endpoint pin');admission(outcome,proof,ep)
 ib=read(AUDIT/'run/input-binding.json');need(ib['card']['producer']['root']==str(PARENT)and ib['card']['producer']['source']==PARENT_SOURCE and ib['card']['producer']['outer']==PARENT_OUTER,'same exact cubic prefix producer')
 pc=read(PARENT/'input-card.json');oldrt=read(PARENT/'chart003/run/runtime.json');newrt=read(NEW/'attempt002/run/runtime.json');nc=read(NEW/'attempt002/input-card.json');no=read(NEW/'outcome.json');nd=read(NEW/'attempt002/run/outcome.json')
 need(pc['source']==oldrt['commit']==PARENT_SOURCE and newrt['commit']==nc['source']==NEW_SOURCE and nc['map_inputs']==pc['map_inputs'],'same physical native model and exact source genealogies')
 for n in W.C.NATIVE_SOURCES:need(identity[n]==oldrt['sources'][n]==newrt['sources'][n]==ib['source_native'][n],'same actual native source '+n)
 for root,rt,source in ((PARENT/'chart003/run',oldrt,PARENT_SOURCE),(NEW/'attempt002/run',newrt,NEW_SOURCE)):
  for n,row in rt['sources'].items():need(entry(root/'source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'literal proof source '+n)
 need(no['status']=='mathematical_stop'and no['connected_count']==0 and len(no['attempts'])==2 and not no['owned_live_pids']and not no['attempts'][0]['root_passed']and no['attempts'][1]['root_passed']and not no['attempts'][1]['joined'],'retained completed failure plus successful unjoined tube')
 need(nd['passed']and not nd['connected']and nd['status']=='uniform705_unjoined'and nd['ticks']==331 and nd['events']==W.EXPECTED and nd['strict_inclusions']==705,'complete retained new full native root')
 chart=read(NEW/'attempt002/run/certificate.json.gz');need(W.uniform(chart)and chart['settings']['halfwidth_exponent']==-6 and chart['parameter_endpoints']==['-1/2048','63/2048']and len(chart['bound']['coordinate_inclusions'])==705,'same preserved successful h1/64 tube')
 domain_paths=list((PARENT/'chart003/run/events').glob('*-cubic_correlated_domain.json.gz'));need(len(domain_paths)==1,'one exact original whole-error domain');domain=read(domain_paths[0])
 need(domain['parameter']=='99/512'and domain['halfwidth_exponent']==-5 and domain['radius_exponent']==-10 and ep['chart']['normal']==pc['normal']and ep['chart']['origin']==pc['origin'],'exact old cubic endpoint/gauge')
 save('input-binding.json',dict(independent_source=AUDIT_SOURCE,independent_outer=AUDIT_OUTER,independent_worker=AUDIT_WORKER,independent_endpoint=ENDPOINT,producer_source=PARENT_SOURCE,producer_outer=PARENT_OUTER,new_source=NEW_SOURCE,new_outer=NEW_OUTER,new_worker=NEW_WORKER,domain=entry(domain_paths[0]),model=pc['map_inputs'],source_native=ib['source_native'],old_join=read(NEW/'attempt002/join.json')))
 return domain,ep,chart,pc

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');TARGET.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
 start=time.monotonic();used=[0];stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 def check():need(not stop[0]and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'120s/6GiB saved endpoint cap')
 def save(name,value,closing=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();raw=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n';need(used[0]+len(raw)<CAP-(0 if closing else 4*1024**2),'64MiB with closing reserve');p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 result=dict(passed=False,native_calls=0,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
  for n,row in identity.items():need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed own source');p=TARGET/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
  save('runtime.json',dict(commit=source,sources=identity,precision=480,pid=os.getpid(),python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0),native_calls=0))
  with ctx.workprec(480):
   # Actual dense705 dyadic serialization/decode and all endpoint reductions,
   # using an unrelated strictly contractive linear synthetic fixture.
   ps=time.monotonic();n=705;D=arb_mat([[arb(((i+1)*6364136223846793005+(j+1)*1442695040888963407)**7%2**440)/(arb((1,460))*n)for j in range(n)]for i in range(n)])
   rows=[[[list(map(int,D[i,j].mid().man_exp())),list(map(int,D[i,j].rad().man_exp()))]for j in range(n)]for i in range(n)]
   save('profile-matrix.json.gz',dict(encoding='mid-rad-dyadic-pairs',rows=rows));del rows;gc.collect();Q=matrix_file(TARGET/'profile-matrix.json.gz',705,705)
   zero=arb_mat(n,1);known=[arb(0,arb((1,-10)))]*n;_,syn=localize_cubic_endpoint([Q,arb_mat(n,n),arb_mat(n,n),arb_mat(n,n)],[zero]*6,arb(1)/32,arb(1)/32,[arb(0)]*n,known,[arb(1)]*n,known,tick=check)
   elapsed=time.monotonic()-ps;forecast=6*elapsed+30;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+3*1024**3;save('profile.json',dict(passed=bool(syn['passed']and forecast<120 and peak<6*1024**3),seconds=elapsed,forecast_seconds=forecast,forecast_peak_bytes=peak,dimension=705,target_arrays_unopened=True));need(forecast<120 and peak<6*1024**3,'fresh full705 measured saved-array admission');del D,Q,known,syn;gc.collect();check()
   domain,ep,chart,pc=bind(identity,save,check);independent_join=W.projected_join(dict(box=ep['endpoint']),chart);save('independent-endpoint-membership.json',independent_join)
   d=arb(1)/32;c,s,a,b=[[decode(v)for v in domain[k]]for k in ('center','slope','curvature','cubic')];point=[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d**3 for i in range(705)]
   defect=[matrix_file(AUDIT/f'run/chart003/defect_coefficient_{k}.json.gz',705,705)for k in range(4)];check();forcing=[matrix_file(AUDIT/f'run/chart003/forcing_coefficient_{k}.json.gz',705,1)for k in range(6)]
   box,proof=localize_cubic_endpoint(defect,forcing,d,d,point,[decode(v)for v in ep['endpoint']],[exact(v)for v in pc['weights']],[decode(v)for v in domain['error']],tick=check);save('localization.json',proof);del defect,forcing;gc.collect()
   membership=W.projected_join(dict(box=vector_json(box)),chart);save('localized-endpoint-membership.json',membership);passed=bool(membership['passed']and len(membership['coordinate_inclusions'])==705 and all(membership['coordinate_inclusions']));save('known-endpoint.json',dict(chart=ep['chart'],endpoint=vector_json(box)))
   if passed:save('connected-endpoint.json',dict(chart={k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=chart['endpoint_root_boxes'][1]))
   result.update(passed=passed,status='same_known_endpoint_joined'if passed else'completed_membership_unresolved',independent_join_inclusions=sum(independent_join['coordinate_inclusions']),localized_join_inclusions=sum(membership['coordinate_inclusions']),weighted_neumann_bound=proof['weighted_neumann_bound'],scope='Same independently known cubic endpoint, saved-only conditional localization and full705 projected membership into retained uniform tube. No new native evaluation, root rerun, complement or global-window claim.')
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(TARGET)):entry(p)for p in sorted(TARGET.rglob('*'))if p.is_file()}),True);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
