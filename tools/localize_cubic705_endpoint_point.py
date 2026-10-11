#!/usr/bin/env python3
"""One fixed36-phase residual at the inherited endpoint; no root reacquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint
from flint import arb,arb_mat,ctx
import localize_cubic705_endpoint as C
import profile_native_unit_pair_pal as P
from neural_shrimp.rigorous_cubic_endpoint import localize_fresh_endpoint
from neural_shrimp.rigorous_unit_pair_pal import assemble
from neural_shrimp.rigorous_unit_pair import phase
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,ball_json,vector_json,matrix_json
W=C.W;need=C.need;read=C.read;entry=C.entry;decode=C.decode
BASE=W.P.BASE/'endpoint-point001';TARGET=BASE/'attempt001/run';CAP=64*1024**2
OLD=C.TARGET.parent;OLD_OUTER='c42e7d3777dba303c8209b6b93388642d0bf2b79bdfad29786b6e3bc4320eab5';OLD_WORKER='0d2de921b936d101808a75119fceb14ca25e37ebb2df26ee65fbc2ce474a60a6';OLD_SOURCE='24b748947862c754ae7358885f3905f3415ef2a6'
SOURCES=list(dict.fromkeys(C.SOURCES+['tools/localize_cubic705_endpoint_point.py','tools/launch_cubic705_endpoint_point.py','tests/test_cubic705_endpoint_point.py','docs/plans/native705-cubic-endpoint-point.md','docs/reviews/native705-cubic-endpoint-localization001.json']))

def endpoints(v):
 need(set(v)>={'mid','rad'},'serialized domain');m,e=v['mid'];r,f=v['rad'];need(all(type(z)is int for z in [m,e,r,f])and r>=0,'exact dyadic domain');a=F(m)*F(2)**e;b=F(r)*F(2)**f;return a-b,a+b

def old_domain_contains(error,predictor,boxes):
 need(len(error)==len(predictor)==len(boxes),'entire original domain')
 for e,p,x in zip(error,predictor,boxes,strict=True):
  lo,hi=endpoints(e);pl,ph=endpoints(ball_json(p));a,b=endpoints(ball_json(x));need(pl==ph and pl+lo<=a<=b<=ph+hi,'exact original fixed-parameter correction-domain containment')

def prior(save,check):
 W.inventory(OLD,OLD_OUTER);W.inventory(OLD/'run',OLD_WORKER);rt=read(OLD/'run/runtime.json');o=read(OLD/'run/outcome.json');owner=read(OLD/'launcher-result.json')
 need(rt['commit']==OLD_SOURCE and o['status']=='completed_membership_unresolved'and not o['passed']and o['native_calls']==0 and o['localized_join_inclusions']==686 and owner['exit_code']==2 and not owner['owned_live_pids'],'exact complete retained saved-only outcome')
 for n,row in rt['sources'].items():need(entry(OLD/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',OLD_SOURCE+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'literal old localization source '+n)
 p=read(OLD/'run/localization.json');ep=read(OLD/'run/known-endpoint.json');need(p['passed']and p['dimension']==705 and ep['endpoint']==p['box'],'identical localized known endpoint')
 save('prior-localization.json',dict(root=str(OLD),outer=OLD_OUTER,worker=OLD_WORKER,source=OLD_SOURCE,outcome=o));check();return ep,o

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');TARGET.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
 start=time.monotonic();used=[0];calls=[0];counts={};seq=[0];stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 def check():need(not stop[0]and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'120s/6GiB fresh endpoint cap')
 def save(name,value,closing=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();raw=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n';need(used[0]+len(raw)<CAP-(0 if closing else 4*1024**2),'64MiB with closing reserve');p=TARGET/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 def tick():calls[0]+=1;need(calls[0]<=72,'36 profile plus36 target native phases');check()
 def observe(kind,value):counts[kind]=counts.get(kind,0)+1;save(f'events/{seq[0]:03d}-{kind}.json.gz',value);seq[0]+=1;check()
 result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
  for n,row in identity.items():need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed own source');p=TARGET/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
  save('runtime.json',dict(commit=source,sources=identity,precision=480,pid=os.getpid(),python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0),native_cap=72))
  with ctx.workprec(480):
   # Original complete saved-array unit supplies actual measured parsing/math
   # cost; unrelated full13/18 native phase profile supplies new residual cost.
   W.inventory(OLD,OLD_OUTER);previous=read(OLD/'run/outcome.json');need(previous['seconds']>0 and previous['seconds']<120 and previous['native_calls']==0,'complete positive historical cost floor');ps=time.monotonic();inp,x,q,r,rates=P.gd_fixture();model=TwoRateCuspMap(inp);save('profile-input.json',dict(model=inp,x=vector_json(x),q=vector_json(q),r=vector_json(r),rates=vector_json(rates)))
   for k in range(18):phase(model,x,q,r,rates,tick,observe,k)
   elapsed=time.monotonic()-ps;forecast=3*previous['seconds']+3*elapsed+10;peak=max(previous['peak_rss_kib'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024+3*1024**3
   save('profile.json',dict(passed=bool(calls[0]==36 and forecast<120 and peak<6*1024**3),seconds=elapsed,actual_saved_seconds=previous['seconds'],forecast_seconds=forecast,forecast_peak_bytes=peak,native_profile_calls=calls[0],target_arrays_unopened=True));need(calls[0]==36 and forecast<120 and peak<6*1024**3,'measured complete old-math plus new native cost admission')
   domain,independent,chart,pc=C.bind(identity,save,check);old,previous=prior(save,check);need(old['chart']==independent['chart'],'same exact independently proved endpoint scalar/gauge')
   d=arb(1)/32;c,s,a,b=[[decode(v)for v in domain[k]]for k in ('center','slope','curvature','cubic')];predictor=[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d**3 for i in range(705)]
   known=[decode(v)for v in old['endpoint']];point=[v.mid()for v in known];old_domain_contains(domain['error'],predictor,known);old_domain_contains(domain['error'],predictor,point)
   normal=[exact(v)for v in pc['normal']];origin=[exact(v)for v in pc['origin']];parameter=exact('115/512');save('point-input.json',dict(point=vector_json(point),known=old['endpoint'],original_predictor=vector_json(predictor),original_error=domain['error'],normal=pc['normal'],origin=pc['origin'],parameter='115/512',model=pc['map_inputs']))
   G,J,blocks=assemble(TwoRateCuspMap(pc['map_inputs']),point,18,normal,origin,parameter,tick,observe);save('point-residual.json',matrix_json(G));del J,blocks;need(calls[0]==72 and counts=={'direction_return':72},'complete actual native profile/point returns')
   propaths=list((C.PARENT/'chart003/run/events').glob('*-cubic_predictor_return.json.gz'));need(len(propaths)==1,'unique finite C0 source');proposal=read(propaths[0]);raw=proposal['preconditioner_exact'];need(len(raw)==705 and all(len(row)==705 for row in raw),'complete original exact C0');C0=arb_mat([[arb(tuple(v))for v in row]for row in raw]);del proposal,raw;gc.collect()
   C1=C.matrix_file(C.AUDIT/'run/chart003/inverse_linear_proposal.json.gz',705,705);defect=[C.matrix_file(C.AUDIT/f'run/chart003/defect_coefficient_{k}.json.gz',705,705)for k in range(4)];check()
   box,proof=localize_fresh_endpoint(defect,C0,C1,G,d,d,point,known,[exact(v)for v in pc['weights']],tick=check);save('localization.json',proof);del defect,C0,C1;gc.collect();membership=W.projected_join(dict(box=vector_json(box)),chart);save('localized-endpoint-membership.json',membership)
   passed=bool(membership['passed']and len(membership['coordinate_inclusions'])==705 and all(membership['coordinate_inclusions']));save('known-endpoint.json',dict(chart=independent['chart'],endpoint=vector_json(box)))
   if passed:save('connected-endpoint.json',dict(chart={k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=chart['endpoint_root_boxes'][1]))
   result.update(passed=passed,status='same_known_endpoint_joined'if passed else'completed_membership_unresolved',localized_join_inclusions=sum(membership['coordinate_inclusions']),weighted_neumann_bound=proof['weighted_neumann_bound'],scope='Same independent known cubic endpoint localized with36 fresh native point phases; saved full original correction-domain derivative and retained705 tube reused. No root rerun, complement or global-window claim.')
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=calls[0],events=counts);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(TARGET)):entry(p)for p in sorted(TARGET.rglob('*'))if p.is_file()}),True);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
