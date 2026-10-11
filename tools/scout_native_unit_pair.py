#!/usr/bin/env python3
"""Source-bound synthetic profile and one separately carded native NS scout."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native_unit_pair_spectrum as S
import profile_native_unit_pair as P
from neural_shrimp.unit_pair_scout import TanhMap,evaluate,continue_branch,need
from neural_shrimp.rigorous_eigenpair_march import decode
BASE=P.BASE/'scout';SETTINGS={'max_points':64,'max_evaluations':360,'t_max':math.pi**2/4}
SOURCES=list(dict.fromkeys(S.SOURCES+['src/neural_shrimp/unit_pair_scout.py','tools/scout_native_unit_pair.py','tools/launch_native_unit_pair_scout.py','tests/test_unit_pair_scout.py','tests/test_native_unit_pair_scout_driver.py','docs/plans/native-unit-pair-scout.md']))
entry=S.entry;read=S.read;inventory=S.inventory

def forecast(times,peak,sizes):
 need(len(times)==12 and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in times),'complete positive profile timings')
 need(len(sizes)==12 and all(type(v)is int and v>0 for v in sizes),'complete positive stored return sizes')
 raw=2*SETTINGS['max_evaluations']*max(sizes)+8*1024**2
 seconds=2*SETTINGS['max_evaluations']*max(times)+20
 return {'seconds':seconds,'rss_kib':max(2*peak,1400000),'raw_bytes':raw,'admitted':seconds<240 and max(2*peak,1400000)<2*1024**2 and raw<64*1024**2,'native_phase_call_cap':36*SETTINGS['max_evaluations']}

def endpoint(seed,rows):
 need(len(seed)==len(rows)==704,'full704 numerical endpoint association')
 def rational(v):
  mid=Fraction(v['mid'][0])*Fraction(2)**v['mid'][1];rad=Fraction(v['rad'][0])*Fraction(2)**v['rad'][1]
  return mid,rad
 checks=[]
 for x,row in zip(seed,rows):
  need(math.isfinite(x),'finite native endpoint');m,r=rational(row);checks.append(abs(Fraction.from_float(float(x))-m)<r)
 return {'passed':all(checks),'coordinate_inclusions':checks,'scope':'Exact rational comparison of binary64 corrected coordinates with saved endpoint balls; numerical association only, not existence/uniqueness of this floating point.'}

def bind_target():
 # Reuses full producer/source/model and all native whole-domain admission.
 flat,blocks,bindings=S.bind();del flat,blocks;gc.collect()
 proof=read(S.PARENT/'run/certificate.json.gz');right=proof['endpoint_root_boxes'][1];need(proof['parameter_endpoints']==['0','1/32768'],'exact right endpoint')
 seed=np.array([float(decode(v).mid())for v in right]);need(len(seed)==704,'704 native seed')
 z=np.r_[seed[:13],seed[-2:],seed[13:26],seed[26:39]];del proof;gc.collect()
 inputs=read(S.PARENT/'run/inputs.json')['map_inputs']
 return TanhMap(inputs),z,right,{'producer':bindings,'right_endpoint':right,'map_inputs':inputs,'parameter':'1/32768'}

def bind_profile(args,identity):
 root=BASE/'attempt002-profile';inventory(root,args.profile_outer);inventory(root/'run',args.profile_worker)
 rt=read(root/'run/runtime.json');out=read(root/'run/outcome.json');la=read(root/'launcher-result.json')
 need(rt['mode']=='profile'and rt['settings']==SETTINGS and rt['commit']==la['commit'],'exact profile mode/settings/source')
 need(la['exit_code']==0 and la['owned_live_pids']==[] and out['passed']and not out['target_acquired']and out['native_calls']==432,'completed synthetic-only profile')
 need(rt['sources']==identity and rt['numpy']==np.__version__ and rt['python']==sys.version,'same source/library profile')
 for n,row in identity.items():need(entry(root/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profiled source')
 projection=forecast(out['times'],out['peak_rss_kib'],out['return_bytes']);need(projection==out['forecast']and projection['admitted'],'measured full-return forecast')
 return {'root':str(root),'outer':args.profile_outer,'worker':args.profile_worker,'source':rt['commit'],'forecast':projection}

def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out=BASE/f'attempt002-{args.mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;serial=0;used=0;stopped=False
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
 def halt(*_):raise TimeoutError('240s scout worker cap')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,240)
 def check():
  need(time.monotonic()-start<240,'240s cap');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'2GiB sampled RSS')
 def save(name,row):
  nonlocal used
  raw=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(used+len(raw)<64*1024**2,'64MiB output cap')
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
  return len(raw)
 def tick():
  nonlocal calls
  calls+=1;need(calls<=(432 if args.mode=='profile'else 36*SETTINGS['max_evaluations']),'all native phase call cap');check()
 def observe(kind,row):
  nonlocal serial
  save(f'events/{serial:04d}-{kind}.json',row);serial+=1;check()
 identity={n:entry(ROOT/n)for n in SOURCES};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 for n,row in identity.items():need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source')
 save('runtime.json',{'mode':args.mode,'commit':commit,'pid':os.getpid(),'sources':identity,'settings':SETTINGS,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 answer={'passed':False,'target_acquired':False,'scope':'Numerical proposals only. No additional certified NS extent, stable side or completed shrimp.'}
 try:
  if args.mode=='profile':
   inp,x,q,r,rates=P.gd_fixture();model=TanhMap(inp);z=np.r_[[float(v.mid())for v in x],[float(v.mid())for v in rates],[float(v.mid())for v in q],[float(v.mid())for v in r]]
   save('synthetic-inputs.json',{'map_inputs':inp,'z':z.tolist(),'period':18,'t':1/32768});times=[];sizes=[]
   for i in range(12):
    before=time.monotonic();v,J,ht,data=evaluate(model,z,18,1/32768,tick);times.append(time.monotonic()-before)
    sizes.append(save(f'profile-{i:02d}.json',{'value':v.tolist(),'jacobian':J.tolist(),'parameter_derivative':ht.tolist(),'native':{k:a.tolist()for k,a in data.items()}}))
   projection=forecast(times,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sizes);answer.update(passed=projection['admitted'],times=times,return_bytes=sizes,forecast=projection,status='completed_synthetic_profile')
  else:
   save('profile-binding.json',bind_profile(args,identity));model,z,right,binding=bind_target();save('inputs.json',binding);check();answer['target_acquired']=True
   def observed(kind,row):
    if kind=='seed':
     association=endpoint(row['native_flat'],right);save('seed-association.json',association);need(association['passed'],'all704 original endpoint numerical association')
    observe(kind,row)
   result=continue_branch(model,z,1/32768,**SETTINGS,tick=tick,observe=observed);save('survey.json',result)
   answer.update(passed=True,status=result['status'],accepted_points=len(result['points']),evaluations=result['evaluations'],last=result['points'][-1]['diagnostics'],last_t=result['points'][-1]['t'])
 except BaseException as exc:answer.update(error=type(exc).__name__+': '+str(exc),status='failed_or_interrupted')
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);answer.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=calls)
  if answer['seconds']>=240 or answer['peak_rss_kib']>=2*1024**2:answer['passed']=False
  save('outcome.json',answer);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
