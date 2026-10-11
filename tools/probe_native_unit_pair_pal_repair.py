#!/usr/bin/env python3
"""One freshly recomputed half-width reweighted705 connection chart."""
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
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE;TARGET=BASE/'attempt003-target/run';CAP=768*1024**2
SETTINGS={'period':18,'precision':256,'parameter':'31/4096','halfwidth_exponent':-7,'radius_exponent':-10,'newton_steps':2}
SOURCES=list(dict.fromkeys(C.SOURCES+D.SOURCES+['tools/probe_native_unit_pair_pal_repair.py','tools/launch_native_unit_pair_pal_repair.py','tests/test_native_unit_pair_pal_repair.py','docs/plans/native-unit-pair-pal-repair.md']))
read=C.read;entry=C.entry;inventory=C.inventory
DIAG=D.OUT.parent
DPINS={'source': '61e72cf15947b63a586f5a94b17bf4fb0a5a9e6e', 'outer': 'e8b443f98e2f0b21c2f66e1de18694b2e81510c1a78ccebefb278ca21f30df90', 'worker': 'd2fb96c658f10dcc2402dfd19bf444bf545b5f557b15d1dcf153d125bdc62195'}

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def repair_weights(diagnostic):
 require(diagnostic['scaling']['iterations']==512 and diagnostic['scaling']['collatz_max']<.95,'completed fixed numerical scaling proposal')
 rp=diagnostic['resolvent_proposal'];radii=rp['radii']
 require(rp['nonnegative']is True and len(radii)==705,'full positive resolvent proposal')
 require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in radii),'finite strictly positive proposal radii')
 # base radius2^-10 * weight4096r = 4r. Every derivative is freshly acquired.
 weights=[Fraction.from_float(float(v))*4096 for v in radii]
 require(all(0<v<=1 for v in weights),'bounded exact dyadic proposed weights')
 return [str(v)for v in weights]

def bind_repair():
 inventory(DIAG,DPINS['outer']);inventory(DIAG/'run',DPINS['worker']);C.archive_bound(DIAG,DPINS['source'])
 rt=read(DIAG/'run/runtime.json');la=read(DIAG/'launcher-result.json');out=read(DIAG/'run/outcome.json');diag=read(DIAG/'run/diagnostic.json')
 require(rt['commit']==la['commit']==DPINS['source']and la['exit_code']==0 and not la['owned_live_pids'],'complete saved diagnosis ownership')
 require(out['passed']is True and out['native_calls']==rt['model_calls']==0 and out['resolvent_available']is True,'proposal-only complete diagnosis')
 for n in D.SOURCES:require(entry(ROOT/n)==rt['sources'][n],'same reviewed diagnosis '+n)
 weights=repair_weights(diag)
 inventory(D.PARENT,'9fe4a84dddb1e1d2f2782d357795c5d45ae2ec61258996772c6498378ca73bba');inventory(D.PARENT/'run','c1d6754f456b10dd22450737d0525e910125f18d8a1ade390d3bc98bf5b0eb21')
 cost=read(D.PARENT/'run/outcome.json');require(cost['status']=='completed_bounds_failed'and cost['ticks']==253,'full measured same-size operation')
 measured={'seconds':2*cost['elapsed_seconds']+20,'rss_kib':2*cost['peak_rss_kib']+128*1024,'bytes':2*cost['stored_bytes_before_final']+8*1024**2}
 require(measured['seconds']<240 and measured['rss_kib']<6*1024**2 and measured['bytes']<CAP,'actual complete acquired cost admission')
 return weights,{'root':str(DIAG),'pins':DPINS,'measured_target_forecast':measured,'diagnostic':entry(DIAG/'run/diagnostic.json'),'settings':SETTINGS,'exact_weights':weights,'amendment':'Half the arc width while retaining exact left bound−1/4096 and full original independent704 anchor; radii4 times saved positive resolvent proposal, all253 native/tick acquisitions fresh. No old failed domain reclassified.'}

def chart_seed(anchor,proof,weights):
 _,normal,origin,_,proposal=C.chart_seed(anchor,proof)
 sc=exact(SETTINGS['parameter']);slope=[decode(v)for v in proposal['initial_arc_slope']]
 initial=[exact_mid(anchor[i].mid()+slope[i]*sc)for i in range(705)]
 proposal['weights_origin']='exact dyadic4096 times saved positive resolvent; actual radii4 times proposal'
 return initial,normal,origin,weights,proposal

def main(args):
 identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(args.out==TARGET,'exclusive target path');out=args.out;out.mkdir(parents=True,exist_ok=False)
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
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':'target','settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':10*1024**3,'tick_cap':253})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'status':'unresolved','scope':'One connected full705 native primitive18 positive unit-pair chart from the independent finite704 endpoint. Complement, signed criticality and junction connection remain separate.'}
 try:
  profile=C.bind_profile(args,identity);save('profile-binding.json',profile);weights,repair=bind_repair();save('repair-binding.json',repair);check()
  with ctx.workprec(256):
   inp,anchor,old,binding=C.bind_root();initial,normal,origin,weights,proposal=chart_seed(anchor,old,weights);del old;gc.collect()
   save('inputs.json',{'map_inputs':inp,'bindings':binding,'settings':SETTINGS});save('chart-inputs.json',{'initial':initial,'normal':normal,'origin':origin,'weights':weights,'anchor':vector_json(anchor),'proposal':proposal,'settings':SETTINGS});check()
   proof,_=quadratic_tube(TwoRateCuspMap(inp),initial,**SETTINGS,normal=normal,origin=origin,weights=weights,anchor={'box':anchor,'period':18,'dimension':13},tick=tick,observe=observe);save('certificate.json.gz',proof);check()
   require(ticks[0]==253 and counts['direction_return']==144 and counts['pal_path_phase_return']==108,'complete fixed full705 acquisition')
   positive=bool(decode(proof['whole_box'][-1])>0)
   result.update(passed=proof['passed']and positive,status='qualified_positive_unit_pair_pal_tube'if proof['passed']and positive else'completed_bounds_failed',gates=dict(proof['gates'],whole_angle_positive=positive),contraction=proof['contraction'],strict_inclusions=sum(proof['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],endpoint_physical=[v[-3:]for v in proof['endpoint_root_boxes']])
 except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-manifest',required=True);p.add_argument('--profile-outer-manifest',required=True);return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
