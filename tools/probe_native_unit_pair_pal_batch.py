#!/usr/bin/env python3
"""One independent full705 curve proposed in a fixed four-chart batch."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import profile_native_unit_pair_pal as P
import probe_native_unit_pair as OLD
import probe_native_unit_pair_pal as C
import diagnose_native_unit_pair_pal as D
import probe_native_unit_pair_pal_repair as R
import localize_native_unit_pair_endpoint as E
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE/'batch001';CAP=768*1024**2;PARENT=P.BASE/'attempt003-target'
PINS=E.PINS
JOIN=E.TARGET.parent
JPINS={'source':'5f6c0a8aaf1f354fd9a7df5f948ddea08113160d','outer':'73ea6f9056eece1179b48edca07839121eb8b5b23fe5b8cdd1093b773e9a5e8e','worker':'09fb08ac9f7dc7d2b22120f53b0656c52ab5f7975a31efe3f18c3bd18e227cac'}
SOURCES=list(dict.fromkeys(R.SOURCES+E.SOURCES+['tools/probe_native_unit_pair_pal_batch.py','tools/launch_native_unit_pair_pal_batch.py','tests/test_native_unit_pair_pal_batch.py','docs/plans/native-unit-pair-pal-batch.md']))
read=C.read;entry=C.entry;inventory=C.inventory

AUDIT=Path('/data2/neural-shrimp-data/audits/native705-independent/attempt001')
APINS={'source':'b53ddc9ab3801b5102c66ec4a587876986d1c04c','outer':'9177a2a3c17b5886a0c3f08d831dcb22822fe09f3e15ec278cde411367414f76','worker':'49f0e1682e0b18ec3999c6ad205f3a803fbba883c46b120fc07c5b29c5f0e2ef'}
JAUDIT=Path('/data2/neural-shrimp-data/audits/native705-endpoint-independent/attempt001')
JAPINS={'source':'ff0372a9fc9bf658f345c02f33e61d336f87ce79','outer':'93a0029c3d9f61899b0dd7c4ef574b1c7ef546fa52477bb426007e3ef6eda4e8','worker':'20b05a8f29b2f73a5d3bf66fddad04ef2c1cdd0ebbf44ceaac9dd87f2dd41819'}

def independent_parent(inp):
 for root,pins in ((AUDIT,APINS),(JAUDIT,JAPINS)):
  inventory(root,pins['outer']);inventory(root/'run',pins['worker']);C.archive_bound(root,pins['source'])
  launch=read(root/'launcher-result.json');require(launch['exit_code']==0 and not launch['owned_live_pids'],'complete independent705 owned audit')
 outcome=read(AUDIT/'run/outcome.json');proof=read(AUDIT/'run/native705-384.json');binding=read(AUDIT/'run/input-binding.json')
 require(outcome['passed']is True and outcome['uniform_primitive_positive_unit_pair']is True and outcome['inclusions']==705 and proof['precision']==384,'independent full705 existence')
 require(binding['root']==str(PARENT)and binding['pins']==PINS,'same original705 producer')
 join=read(JAUDIT/'run/outcome.json');jb=read(JAUDIT/'run/input-binding.json')
 require(join['passed']is True and join['connected']is True and join['anchor_coordinates']==705 and join['parameter_endpoints']==['-1/4096','63/4096'],'independent same-root full705 join')
 require(jb['map_inputs']==inp and jb['endpoint']==str(JOIN)and jb['endpoint_pins']['outer']==JPINS['outer']and jb['endpoint_pins']['worker']==JPINS['worker']and jb['prior705']['root']==str(PARENT)and jb['prior705']['pins']==PINS,'exact independent endpoint/curve/map binding')
 endpoints=proof['recomputed_endpoints'];require(len(endpoints)==2 and all(len(v)==705 for v in endpoints),'all independently recomputed705 endpoints')
 return endpoints,{'uniform':dict(root=str(AUDIT),**APINS),'join':dict(root=str(JAUDIT),**JAPINS)}

def settings(unit):
 require(type(unit)is int and 1<=unit<=4,'fixed four proposed charts')
 return dict(period=18,precision=256,parameter=str(Fraction(31+64*unit,4096)),halfwidth_exponent=-7,radius_exponent=-10,newton_steps=2)

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def bind_parent():
 inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker']);C.archive_bound(PARENT,PINS['source'])
 done=read(PARENT/'run/outcome.json');launch=read(PARENT/'launcher-result.json');inp=read(PARENT/'run/inputs.json')
 require(launch['exit_code']==2 and not launch['owned_live_pids']and done['ticks']==253 and done['strict_inclusions']==705 and done['gates']['uniform_full_native_unit_pair_locus']and done['gates']['whole_angle_positive'],'complete fixed original705 uniform proof')
 inventory(JOIN,JPINS['outer']);inventory(JOIN/'run',JPINS['worker']);C.archive_bound(JOIN,JPINS['source'])
 jd=read(JOIN/'run/outcome.json');jl=read(JOIN/'launcher-result.json');ji=read(JOIN/'run/inputs.json');member=read(JOIN/'run/membership.json')
 require(jd['passed']is True and jd['status']=='known_endpoint_joined705'and jd['ticks']==144 and jd['anchor_coordinates']==705 and jd['parent']==dict(root=str(PARENT),**PINS),'full same-root original705 join')
 require(jl['exit_code']==0 and not jl['owned_live_pids']and member['passed']is True and member['coordinate_inclusions']==[True]*705,'complete705 membership and owner')
 require(ji['map_inputs']==inp['map_inputs']and ji['binding']==inp['bindings'],'same exact root/map/model')
 forecast={'seconds':2*done['elapsed_seconds']+20,'rss_kib':2*done['peak_rss_kib']+128*1024,'bytes':2*done['stored_bytes_before_final']+8*1024**2}
 require(forecast['seconds']<240 and forecast['rss_kib']<6*1024**2 and forecast['bytes']<CAP,'actual complete253call measured admission')
 endpoints,audit=independent_parent(inp['map_inputs']);chart=read(PARENT/'run/certificate.json.gz')
 require(chart['gates']['uniform_full_native_unit_pair_locus']and all(chart['coordinate_inclusions'])and chart['parameter_endpoints']==['-1/4096','63/4096'],'unchanged first705 interval')
 chart['endpoint_root_boxes']=endpoints
 return inp['map_inputs'],chart,{'independent':audit,'parent':dict(root=str(PARENT),**PINS),'join':dict(root=str(JOIN),**JPINS),'measured_forecast_per_worker':forecast,'scope':'Known connected original705 chart, four independent new uniform root proofs, ordered exact-gauge joins evaluated only after complete workers.'}

def seed(parent,unit):
 cfg=settings(unit);delta=exact(cfg['parameter'])-exact(parent['parameter_center'])
 initial=[exact_mid(decode(c)+decode(s)*delta+decode(k)*delta*delta/2)for c,s,k in zip(parent['center'],parent['slope'],parent['curvature'],strict=True)]
 weights=[str(2*Fraction(w))for w in parent['weights']]
 require(len(initial)==len(weights)==705 and all(0<Fraction(w)<=1 for w in weights),'bounded full705 enlarged coordinate weights')
 return initial,list(parent['normal']),list(parent['origin']),weights

def uniform_admission(proof,positive):
 require(proof['shooting_dimension']==705 and proof['dimension']==13 and proof['period']==18,'complete native705 proof')
 expected={'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'anchor_identified':False}
 return bool(proof['gates']==expected and all(proof['coordinate_inclusions'])and positive)

def ordered_join(previous,current):
 # Exact parameter can replace projection only for byte-identical fixed gauge.
 require(previous['normal']==current['normal']and previous['origin']==current['origin'],'identical authenticated gauge')
 require(previous['period']==current['period']==18 and previous['dimension']==current['dimension']==13,'same full native family')
 s=previous['parameter_endpoints'][1];require(Fraction(s)==Fraction(current['parameter_endpoints'][0]),'exact adjoining closed parameter boundary')
 a={'box':[decode(v)for v in previous['endpoint_root_boxes'][1]],'dimension':13,'period':18,'exact_parameter':s}
 from neural_shrimp.rigorous_unit_pair_pal import anchor_test
 return anchor_test(a,[decode(v)for v in current['center']],[decode(v)for v in current['slope']],[decode(v)for v in current['curvature']],[exact(v)for v in current['normal']],[exact(v)for v in current['origin']],exact(current['parameter_center']),arb((1,-7)),[decode(v)for v in current['coordinate_radii']],13,18)

def main(args):
 cfg=settings(args.unit);target=BASE/f'chart{args.unit:03d}/run';identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(args.out==target,'exclusive target path');out=args.out;out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();stop=[False];ticks=[0];serial=[0];used=[0];counts={}
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
 def check():
  if stop[0]or time.monotonic()-start>=240:raise TimeoutError('240s target cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=6*1024**2:raise MemoryError('6GiB sampled target RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n'
  require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'aggregate target raw cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=253,'252 native passes and1preconditioner maximum');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':'target','settings':cfg,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':10*1024**3,'tick_cap':253})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'unit':args.unit,'status':'unresolved','connected':False,'scope':'One independent uniform705 curve proof only; ordered strict native endpoint joins are a separate completed-batch operation.'}
 try:
  profile=C.bind_profile(args,identity);save('profile-binding.json',profile);check()
  with ctx.workprec(256):
   inp,parent,binding=bind_parent();initial,normal,origin,weights=seed(parent,args.unit);proposal={'parent_parameter':parent['parameter_center'],'delta':str(Fraction(cfg['parameter'])-Fraction(parent['parameter_center'])),'scope':'independent fixed-gauge quadratic seed; no predecessor root or join assumed'};del parent;gc.collect()
   save('inputs.json',{'map_inputs':inp,'bindings':binding,'settings':cfg});save('chart-inputs.json',{'initial':initial,'normal':normal,'origin':origin,'weights':weights,'anchor':None,'proposal':proposal,'settings':cfg});check()
   proof,_=quadratic_tube(TwoRateCuspMap(inp),initial,**cfg,normal=normal,origin=origin,weights=weights,anchor=None,tick=tick,observe=observe);save('certificate.json.gz',proof);check()
   require(ticks[0]==253 and counts['direction_return']==144 and counts['pal_path_phase_return']==108,'complete fixed full705 acquisition')
   positive=bool(decode(proof['whole_box'][-1])>0)
   result.update(passed=uniform_admission(proof,positive),status='uniform705_unjoined'if uniform_admission(proof,positive)else'completed_bounds_failed',gates=dict(proof['gates'],whole_angle_positive=positive),contraction=proof['contraction'],strict_inclusions=sum(proof['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],endpoint_physical=[v[-3:]for v in proof['endpoint_root_boxes']])
 except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--unit',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-manifest',required=True);p.add_argument('--profile-outer-manifest',required=True);return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
