#!/usr/bin/env python3
"""One bounded numerical native NS/flip localization and bracket re-association."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import scout_native_unit_pair_section as S
from neural_shrimp.unit_pair_scout import TanhMap,evaluate,qualify,native_flat,need
from neural_shrimp.unit_pair_section import section_evaluation,native_association
from neural_shrimp.unit_pair_flip import augmented,native_direction,spectrum_report,crossing_derivative
BASE=S.P.BASE/'flip';PARENT=S.BASE/'attempt001-target'
PINS={'outer':'2cde595d4b05e4e65b7f5ca4c2a394609bf98493219b2d1486bddb5223500d40','worker':'26ce0c94f3c27aa4f05d2f23eabb3c05449a9c2f05e4107f877c6659414187d2','execution':'f70f3e6a9696c89f4aed0c2398dbfd3b78ec1020'}
CAPS={'augmented':72,'section':36,'adjacent':4}
SOURCES=list(dict.fromkeys(S.SOURCES+['src/neural_shrimp/unit_pair_flip.py','tools/localize_native_unit_pair_flip.py','tools/launch_native_unit_pair_flip.py','tests/test_unit_pair_flip.py','tests/test_native_unit_pair_flip_driver.py','docs/plans/native-unit-pair-flip.md']))
entry,read,inventory=S.entry,S.read,S.inventory


def forecast(times,sizes,peak):
 need(set(times)==set(sizes)==set(CAPS)and all(len(times[k])==len(sizes[k])==6 for k in CAPS),'complete six representative returns each')
 need(all(type(v)in(int,float)and math.isfinite(v)and v>0 for values in times.values()for v in values),'finite positive timings')
 need(all(type(v)is int and v>0 for values in sizes.values()for v in values),'positive stored return sizes')
 seconds=2*sum(CAPS[k]*max(times[k])for k in CAPS)+20
 raw=2*sum(CAPS[k]*max(sizes[k])for k in CAPS)+8*1024**2
 rss=max(2*peak,256*1024)
 return {'seconds':seconds,'raw_bytes':raw,'rss_kib':rss,'native_phase_calls':54*72+36*36+18*4,'admitted':seconds<120 and raw<64*1024**2 and rss<1024**2}


def bind():
 inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker'])
 rt=read(PARENT/'run/runtime.json');out=read(PARENT/'run/outcome.json');la=read(PARENT/'launcher-result.json');survey=read(PARENT/'run/survey.json')
 need(rt['commit']==la['commit']==PINS['execution']and out['passed']and la['exit_code']==0 and la['owned_live_pids']==[],'exact completed native section')
 need(out['accepted_points']==len(survey['points'])==33 and out['evaluations']==351 and out['native_calls']==12636 and out['reverse_prefix_points']==len(survey['reverse_prefix'])==19,'complete native section counts')
 need(all(v['association']['passed']for v in survey['reverse_prefix'])and all(v['diagnostics']['passed']for v in survey['points']),'all full native associations and qualifications')
 for n,row in rt['sources'].items():need(entry(PARENT/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical native section source')
 points=[survey['points'][i]for i in (26,27)];need(points[0]['diagnostics']['complement_max_modulus']<1<points[1]['diagnostics']['complement_max_modulus'],'exact competing stability crossing')
 returns=[read(p)for p in sorted((PARENT/'run/events').glob('*-evaluation_return.json'))];need(len(returns)==351,'complete section return inventory')
 matches=[v for v in returns if v['z']==points[1]['z']and v['t']==points[1]['t']];need(len(matches)==1,'unique actual unstable bracket return')
 M=np.array(matches[0]['native']['monodromy']);vals,vec=np.linalg.eig(M);idx=int(np.argmax(abs(vals)));w=vec[:,idx]
 need(abs(vals[idx].imag)<1e-8 and vals[idx].real<-1 and max(abs(w.imag))<1e-8,'actual negative unstable leading mode')
 w=w.real/np.linalg.norm(w.real);need(np.linalg.norm(M@w-vals[idx].real*w)<1e-8,'actual seed eigen residual')
 old=read(PARENT/'run/inputs.json');model=TanhMap(old['original_finite_proof_binding']['map_inputs'])
 need(model.n==13 and len(model.x)==24,'exact13D24sample target map')
 seed=np.r_[points[1]['z'],points[1]['t'],w]
 return model,points,seed,{'parent':str(PARENT),'pins':PINS,'endpoints':points,'seed':seed.tolist(),'eigenvalues':[[float(v.real),float(v.imag)]for v in vals],'leading_index':idx,'original':old}


def localize(model,points,seed,call,emit):
 n=model.n;k=3*n+2;y=seed.copy();left,right=[v['z'][n]for v in points];width=right-left
 need(n==13 and 0<width<=.25,'fixed native bracket width')
 for iteration in range(12):
  need(left-.1*width<=y[n]<=right+.1*width and np.all(y[n:n+2]>0)and 0<y[k]<3 and y[k+1:]@seed[k+1:]>0,'same actual rates/angle/flip orientation')
  v,J,data=call('augmented',y)
  if max(abs(v))<1e-11:break
  need(iteration<11,'twelve augmented returns');step=np.linalg.solve(J,v);emit('newton_step',{'y':y.tolist(),'step':step.tolist(),'iteration':iteration})
  need(max(abs(step)/np.maximum(1.,abs(y)))<.1,'all native coordinates local Newton guard');y-=step
 z,t,w=y[:k],float(y[k]),y[k+1:]
 need(left<z[n]<right and min(p['t']for p in points)<t<max(p['t']for p in points),'junction strictly inside saved rate/angle bracket')
 basic=qualify(z,t,v[:k],data);need(basic['passed'],'native primitive unit-pair root')
 phase0=spectrum_report(data['monodromy'],w,math.sqrt(t));need(phase0['passed'],'phase0 simple flip/stable remaining spectrum')
 wa=data['w_phases'][1];need(np.linalg.norm(wa)>1e-8,'nonzero adjacent native flip vector');wa=wa/np.linalg.norm(wa)
 f,g,mw,mixed,adj=call('adjacent',np.r_[data['orbit'][1],z[n:n+2],wa])
 need(max(abs(f-data['orbit'][1]))<1e-9,'adjacent native periodic return')
 need(all(max(abs(adj['orbit'][d]-adj['orbit'][0]))>1e-5 for d in (1,2,3,6,9)),'adjacent native primitive18')
 phase1=spectrum_report(g[:,:n],wa,math.sqrt(t));need(phase1['passed'],'adjacent simple flip/stable remaining spectrum')
 derivative=crossing_derivative(J,w,data['monodromy'],n);need(derivative['passed'],'transverse flip crossing along native section')
 joined=[];section=J[:k,:k].copy();section[:,n]=J[:k,k];tan=np.linalg.solve(section,-J[:k,n])
 for point in points:
  hidden=point['z'][n];step=hidden-z[n];u=z+tan*step;u[n]=t+tan[n]*step
  for iteration in range(9):
   value,jac,fh,dd,zz,tt=call('section',(u,hidden))
   if max(abs(value))<1e-11:break
   need(iteration<8,'eight reverse native section updates');delta=np.linalg.solve(jac,value)
   emit('reverse_newton_step',{'u':u.tolist(),'hidden':hidden,'step':delta.tolist(),'iteration':iteration})
   need(max(abs(delta)/np.maximum(1.,abs(u)))<.1,'local full native reverse guard');u-=delta
  assoc=native_association(native_flat(zz,dd),point['native_flat']);diag=qualify(zz,tt,value,dd,data)
  need(assoc['passed']and abs(tt-point['t'])<1e-7 and diag['passed'],'both exact bracket full native reverse associations')
  row={'saved_index':point['index'],'z':zz.tolist(),'t':tt,'native_flat':native_flat(zz,dd).tolist(),'association':assoc,'diagnostics':diag};emit('bracket_reverse',row);joined.append(row)
 return {'passed':True,'y':y.tolist(),'native_flat':native_flat(z,data).tolist(),'native_diagnostics':basic,'phase0_spectrum':phase0,'phase1_spectrum':phase1,'crossing_derivative':derivative,'bracket_reverse':joined,'scope':'Floating full55D native NS-flip candidate with simple critical modes/stable other10 and reverse associations. No rigorous junction, complement theorem, NS criticality or connected interval extent.'}


def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out=BASE/f'attempt001-{args.mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();native=0;serial=0;used=0;counts=dict.fromkeys(CAPS,0)
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 def halt(*_):raise TimeoutError('120s native junction cap')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,120)
 def save(name,row):
  nonlocal used
  b=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(used+len(b)<64*1024**2,'64MiB output cap');used+=len(b)
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  return len(b)
 def check():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'120s/1GiB sampled cap')
 def emit(kind,row):
  nonlocal serial
  size=save(f'events/{serial:04d}-{kind}.json',row);serial+=1;check();return size
 def tick():
  nonlocal native
  native+=1;need(native<=(648 if args.mode=='profile'else 5256),'inclusive native phase cap');check()
 identity={n:entry(ROOT/n)for n in SOURCES};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 save('runtime.json',{'mode':args.mode,'commit':commit,'pid':os.getpid(),'sources':identity,'caps':CAPS,'numpy':np.__version__,'python':sys.version,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1')})
 for n,row in identity.items():
  need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 answer={'passed':False,'target_acquired':False}
 try:
  if args.mode=='profile':
   inp,x,q,r,rates=S.P.gd_fixture();model=TanhMap(inp);z=np.r_[[float(v.mid())for v in x],[float(v.mid())for v in rates],[float(v.mid())for v in q],[float(v.mid())for v in r]];w=np.arange(1,14,dtype=float);w/=np.linalg.norm(w);seed=np.r_[z,1/32768,w];points=None
   save('synthetic-inputs.json',{'map_inputs':inp,'seed':seed.tolist()})
  else:
   profile=BASE/'attempt001-profile';inventory(profile,args.profile_outer);inventory(profile/'run',args.profile_worker);rt=read(profile/'run/runtime.json');po=read(profile/'run/outcome.json');la=read(profile/'launcher-result.json')
   need(rt['mode']=='profile'and rt['caps']==CAPS and rt['sources']==identity and rt['python']==sys.version and rt['numpy']==np.__version__ and rt['commit']==la['commit'],'same exact source/settings/library profile')
   need(po['passed']and not po['target_acquired']and po['native_calls']==648 and la['exit_code']==0 and la['owned_live_pids']==[],'completed synthetic profile')
   for n,row in identity.items():need(entry(profile/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profile source')
   projection=forecast(po['times'],po['sizes'],po['peak_rss_kib']);need(projection==po['forecast']and projection['admitted'],'measured target admission')
   save('profile-binding.json',{'root':str(profile),'outer':args.profile_outer,'worker':args.profile_worker,'forecast':projection})
   model,points,seed,binding=bind();save('inputs.json',binding);answer['target_acquired']=True
  def call(kind,arg):
   need(counts[kind]<(6 if args.mode=='profile'else CAPS[kind]),'complete return quota '+kind);counts[kind]+=1
   desc={'kind':kind,'input':arg.tolist()if isinstance(arg,np.ndarray)else [arg[0].tolist(),arg[1]]};emit('evaluation_input',desc)
   before=time.monotonic()
   if kind=='augmented':
    v,J,data=augmented(model,arg,18,tick);result=(v,J,data);row={'value':v.tolist(),'jacobian':J.tolist(),'native':{k:a.tolist()for k,a in data.items()}}
   elif kind=='section':
    result=section_evaluation(model,arg[0],arg[1],18,tick);v,J,fh,data,z,t=result;row={'value':v.tolist(),'jacobian':J.tolist(),'parameter_derivative':fh.tolist(),'native':{k:a.tolist()for k,a in data.items()},'z':z.tolist(),'t':t}
   else:
    n=model.n;result=native_direction(model,arg[:n],arg[n:n+2],arg[n+2:],18,tick);f,g,mw,mixed,data=result;row={'value':f.tolist(),'first':g.tolist(),'directional':mw.tolist(),'mixed':mixed.tolist(),'native':{k:a.tolist()for k,a in data.items()}}
   elapsed=time.monotonic()-before;size=emit('evaluation_return',dict(desc,**row));timings[kind].append(elapsed);sizes[kind].append(size);return result
  timings={k:[]for k in CAPS};sizes={k:[]for k in CAPS}
  if args.mode=='profile':
   for _ in range(6):
    call('augmented',seed);u=seed[:41].copy();u[13]=seed[41];call('section',(u,float(seed[13])));call('adjacent',np.r_[seed[:15],seed[42:]])
   projection=forecast(timings,sizes,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);answer.update(passed=projection['admitted'],status='completed_synthetic_profile',forecast=projection,times=timings,sizes=sizes)
  else:
   result=localize(model,points,seed,call,emit);save('candidate.json',result);answer.update(passed=result['passed'],status='numerical_native_ns_flip_localized',candidate={k:v for k,v in result.items()if k not in ('native_flat','bracket_reverse')})
 except BaseException as exc:answer.update(error=type(exc).__name__+': '+str(exc),status='failed_or_interrupted')
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);answer.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=native,counts=counts)
  if answer['seconds']>=120 or answer['peak_rss_kib']>=1024**2:answer['passed']=False
  save('outcome.json',answer);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
