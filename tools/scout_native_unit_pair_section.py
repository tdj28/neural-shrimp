#!/usr/bin/env python3
"""Source-bound synthetic profile and one reverse-associated native NS section survey."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import scout_native_unit_pair as C
import probe_native_unit_pair_spectrum as S
import profile_native_unit_pair as P
from neural_shrimp.unit_pair_scout import TanhMap,need
from neural_shrimp.unit_pair_section import section_evaluation,survey
from neural_shrimp.rigorous_eigenpair_march import decode
BASE=P.BASE/'section';SETTINGS={'max_points':48,'max_evaluations':360,'hidden_step':.25}
SOURCES=list(dict.fromkeys(C.SOURCES+['src/neural_shrimp/unit_pair_section.py','tools/scout_native_unit_pair_section.py','tools/launch_native_unit_pair_section.py','tests/test_unit_pair_section.py','tests/test_native_unit_pair_section_driver.py','docs/plans/native-unit-pair-section.md']))
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
 root=C.BASE/'attempt002-target';outer='4941c57ff6a37a7574f7300b17b5980aafad35bd27e6091ba0f99bdf724403f0';worker='dde339494bc7b5ff77eda74cfde8d38cc9a250db24d15ffe57187a327582dddd'
 inventory(root,outer);inventory(root/'run',worker)
 rt=read(root/'run/runtime.json');out=read(root/'run/outcome.json');la=read(root/'launcher-result.json');old=read(root/'run/survey.json');inp=read(root/'run/inputs.json');anchor=read(root/'run/seed-association.json')
 need(rt['commit']==la['commit']and la['exit_code']==0 and la['owned_live_pids']==[]and out['passed'],'complete exact earlier numerical scout')
 need(out['accepted_points']==len(old['points'])==20 and out['evaluations']==84 and out['native_calls']==3024 and out['status']=='corrector_limit','exact prior prefix')
 need(anchor['passed']and anchor['coordinate_inclusions']==[True]*704 and inp['parameter']=='1/32768','original finite704 endpoint association retained')
 for n,row in rt['sources'].items():need(entry(root/'run/source'/n)==entry(ROOT/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'unchanged entire old scout source')
 need(all(p['diagnostics']['passed']and all(p['diagnostics']['gates'].values())for p in old['points']),'all exact prior native qualifications')
 return TanhMap(inp['map_inputs']),old['points'],{'root':str(root),'outer':outer,'worker':worker,'source':rt['commit'],'original_finite_proof_binding':inp,'last':old['points'][-1]}


def bind_profile(args,identity):
 root=BASE/'attempt001-profile';inventory(root,args.profile_outer);inventory(root/'run',args.profile_worker)
 rt=read(root/'run/runtime.json');out=read(root/'run/outcome.json');la=read(root/'launcher-result.json')
 need(rt['mode']=='profile'and rt['settings']==SETTINGS and rt['commit']==la['commit'],'exact profile mode/settings/source')
 need(la['exit_code']==0 and la['owned_live_pids']==[] and out['passed']and not out['target_acquired']and out['native_calls']==432,'completed synthetic-only profile')
 need(rt['sources']==identity and rt['numpy']==np.__version__ and rt['python']==sys.version,'same source/library profile')
 for n,row in identity.items():need(entry(root/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profiled source')
 projection=forecast(out['times'],out['peak_rss_kib'],out['return_bytes']);need(projection==out['forecast']and projection['admitted'],'measured full-return forecast')
 return {'root':str(root),'outer':args.profile_outer,'worker':args.profile_worker,'source':rt['commit'],'forecast':projection}

def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out=BASE/f'attempt001-{args.mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;serial=0;used=0;stopped=False
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
    before=time.monotonic();u=z.copy();u[13]=1/32768;v,J,ht,data,_,_=section_evaluation(model,u,float(z[13]),18,tick);times.append(time.monotonic()-before)
    sizes.append(save(f'profile-{i:02d}.json',{'value':v.tolist(),'jacobian':J.tolist(),'parameter_derivative':ht.tolist(),'native':{k:a.tolist()for k,a in data.items()}}))
   projection=forecast(times,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sizes);answer.update(passed=projection['admitted'],times=times,return_bytes=sizes,forecast=projection,status='completed_synthetic_profile')
  else:
   save('profile-binding.json',bind_profile(args,identity));model,points,binding=bind_target();save('inputs.json',binding);check();answer['target_acquired']=True
   result=survey(model,points,**SETTINGS,tick=tick,observe=observe);save('survey.json',result)
   answer.update(passed=True,status=result['status'],accepted_points=len(result['points']),reverse_prefix_points=len(result['reverse_prefix']),evaluations=result['evaluations'],last=result['points'][-1]['diagnostics'],last_t=result['points'][-1]['t'])
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
