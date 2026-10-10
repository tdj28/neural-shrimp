#!/usr/bin/env python3
"""One profile-bound384-bit full55 native NS/flip root qualification."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import localize_native_unit_pair_flip as N
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_unit_pair_flip import augmented,root_proof,complement,transverse_crossing
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json,matrix_json,eye,midpoint_matrix
need=N.need;entry=N.entry;read=N.read;inventory=N.inventory
BASE=N.BASE/'rigorous';PARENT=N.BASE/'attempt001-target'
PINS={'execution':'f5021f3aa08bfbc45b9a1e8515315b32c2271b94','outer':'c9bbd5cd3b3baba3a7db6b4d53ddb1cf1a9ae1b09f364baf87311a84ba9dd8de','worker':'f2f357709fb600286b8ef3a8bb4a41575bd651c74fc20423ed0136bf6ba55d19'}
SETTINGS={'precision':384,'period':18,'newton_steps':4,'radius_exponent':-80}
SOURCES=list(dict.fromkeys(N.SOURCES+['src/neural_shrimp/rigorous_unit_pair_flip.py','tools/certify_native_unit_pair_flip.py','tools/launch_rigorous_native_unit_pair_flip.py','tests/test_rigorous_unit_pair_flip.py','tests/test_rigorous_native_unit_pair_flip_driver.py','docs/plans/native-unit-pair-flip-rigorous.md','docs/plans/native-unit-pair-flip-certificate.md']))


def forecast(times,sizes,peak):
 need(len(times)==len(sizes)==2 and all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in times),'two positive full55 profiles')
 need(all(type(v)is int and v>0 for v in sizes),'complete profile sizes')
 seconds=12*max(times)+15;raw=12*max(sizes)+8*1024**2;rss=max(2*peak,256*1024)
 return {'seconds':seconds,'raw_bytes':raw,'rss_kib':rss,'native_calls':324,'admitted':seconds<60 and raw<32*1024**2 and rss<1024**2}


def bind_candidate():
 inventory(PARENT,PINS['outer']);inventory(PARENT/'run',PINS['worker'])
 rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');c=read(PARENT/'run/candidate.json');inp=read(PARENT/'run/inputs.json')
 need(rt['commit']==la['commit']==PINS['execution']and la['exit_code']==0 and la['owned_live_pids']==[]and out['passed']and out['status']=='numerical_native_ns_flip_localized','exact complete numerical candidate')
 need(out['counts']=={'augmented':4,'adjacent':1,'section':6}and out['native_calls']==450,'full candidate acquired inventory')
 for n,row in rt['sources'].items():need(entry(PARENT/'run/source'/n)==entry(ROOT/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'same numerical source/archive/history')
 need(c['passed']and c['native_diagnostics']['passed']and c['phase0_spectrum']['passed']and c['phase1_spectrum']['passed']and c['crossing_derivative']['passed'],'all numerical qualifications retained')
 need([v['saved_index']for v in c['bracket_reverse']]==[26,27]and all(v['association']['passed']for v in c['bracket_reverse']),'both full native reverse joins')
 need(inp['pins']==N.PINS and c['y']==out['candidate']['y']and len(c['y'])==55,'exact source bracket and full candidate')
 seed=[exact(dyadic(v))for v in c['y']];model=inp['original']['original_finite_proof_binding']['map_inputs']
 need(TwoRateCuspMap(model).n==13 and len(model['x'])==24,'same13D24sample map')
 return TwoRateCuspMap(model),seed,{'root':str(PARENT),'pins':PINS,'candidate':entry(PARENT/'run/candidate.json'),'map_inputs':model,'seed':vector_json(seed),'numerical_reverse_association':c['bracket_reverse'],'bracket':inp['endpoints']}


def qualify_same_root(root):
 need(root['passed']and root['augmented_dimension']==55 and root['dimension']==13 and root['period']==18,'same proved full55 root')
 need(root['coordinate_inclusions']==[True]*55 and decode(root['contraction'])<1 and all(root['gates'].values()),'strict root gates')
 need([v['divisor']for v in root['divisors']]==[1,2,3,6,9]and all(v['excluded']for v in root['divisors']),'proper primitive18 exclusions')
 box=[decode(v)for v in root['box']];M=arb_mat([[decode(v)for v in row]for row in root['monodromy']]);J=arb_mat([[decode(v)for v in row]for row in root['whole_jacobian']])
 need(len(box)==55 and M.nrows()==M.ncols()==13 and J.nrows()==J.ncols()==55,'complete same-box monodromy and derivative')
 q,r,w,t=box[15:28],box[28:41],box[42:55],box[41]
 # Conservative restore can enlarge each enclosure, never narrow it. Every
 # argument here comes only from this root's acquired box and whole derivative.
 spectral=complement(M,q,r,w,t);crossing=transverse_crossing(J,M,w,13)
 return spectral,crossing


def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 out=BASE/f'attempt002-{args.mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;serial=0;used=0
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 def halt(*_):raise TimeoutError('60s native interval junction cap')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,60)
 def save(name,row):
  nonlocal used
  b=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();need(used+len(b)<32*1024**2,'32MiB cap');used+=len(b);p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  return len(b)
 def check():need(time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s/1GiB sampled cap')
 def tick():
  nonlocal calls
  calls+=1;need(calls<=(108 if args.mode=='profile'else 324),'fixed native call cap');check()
 identity={n:entry(ROOT/n)for n in SOURCES};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 save('runtime.json',{'mode':args.mode,'commit':commit,'pid':os.getpid(),'sources':identity,'settings':SETTINGS,'python':sys.version,'arb_version':{'python_flint':flint.__version__,'flint':str(getattr(flint,'__FLINT_VERSION__','unavailable'))},'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1')})
 for n,row in identity.items():
  need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 answer={'passed':False,'target_acquired':False,'scope':'Local full55 native NS/flip proof and same-root spectrum/crossing only. No certified NS connection from704 anchor, normal-form criticality or completed shrimp.'}
 try:
  with ctx.workprec(SETTINGS['precision']):
   if args.mode=='profile':
    inp,x,q,r,rates=N.S.P.gd_fixture();model=TwoRateCuspMap(inp);w=[exact(str(i+1))for i in range(13)];scale=sum((v*v for v in w),arb(0)).sqrt();w=[v/scale for v in w];seed=x+rates+q+r+[exact('1/100')]+w
    save('synthetic-inputs.json',{'map_inputs':inp,'seed':vector_json(seed)});times=[];sizes=[]
    for i in range(2):
     box=[arb(v.mid(),arb((1,SETTINGS['radius_exponent'])))for v in seed];begin=time.monotonic();v,J,data=augmented(model,box,18,tick);pre=midpoint_matrix(J.inv());defect=eye(55)-pre*J;correction=pre*v;times.append(time.monotonic()-begin)
     sizes.append(save(f'profile-{i}.json',{'box':vector_json(box),'value':matrix_json(v),'jacobian':matrix_json(J),'preconditioner':matrix_json(pre),'defect':matrix_json(defect),'correction':matrix_json(correction)}))
    projection=forecast(times,sizes,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);answer.update(passed=projection['admitted'],status='completed_synthetic_profile',forecast=projection,times=times,sizes=sizes)
   else:
    profile=BASE/'attempt002-profile';inventory(profile,args.profile_outer);inventory(profile/'run',args.profile_worker);rt=read(profile/'run/runtime.json');po=read(profile/'run/outcome.json');la=read(profile/'launcher-result.json')
    need(rt['mode']=='profile'and rt['settings']==SETTINGS and rt['sources']==identity and rt['python']==sys.version and rt['arb_version']=={'python_flint':flint.__version__,'flint':str(getattr(flint,'__FLINT_VERSION__','unavailable'))}and rt['commit']==la['commit'],'same exact arithmetic/source/profile settings')
    need(po['passed']and not po['target_acquired']and po['native_calls']==108 and la['exit_code']==0 and la['owned_live_pids']==[],'completed synthetic profile')
    for n,row in identity.items():need(entry(profile/'run/source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profile source')
    projection=forecast(po['times'],po['sizes'],po['peak_rss_kib']);need(projection==po['forecast']and projection['admitted'],'measured target admission');save('profile-binding.json',{'root':str(profile),'outer':args.profile_outer,'worker':args.profile_worker,'forecast':projection})
    model,seed,binding=bind_candidate();save('inputs.json',binding);answer['target_acquired']=True
    def observe(kind,row):
     nonlocal serial
     save(f'events/{serial:03d}-{kind}.json',row);serial+=1;check()
     if 'center'in row:
      center=[decode(v)for v in row['center']];need(all(abs(a-b)<arb((1,-20))*max(arb(1),abs(b))for a,b in zip(center,seed,strict=True)),'all55 local candidate proximity')
    proof,data=root_proof(model,seed,18,radius_exponent=SETTINGS['radius_exponent'],newton_steps=SETTINGS['newton_steps'],tick=tick,observe=observe);save('root.json',proof)
    need(proof['passed'],'full55 strict interval root')
    box=[decode(v)for v in proof['box']];ends=binding['bracket'];need(exact(dyadic(ends[0]['z'][13]))<box[13]and box[13]<exact(dyadic(ends[1]['z'][13]))and exact(dyadic(ends[1]['t']))<box[41]and box[41]<exact(dyadic(ends[0]['t'])),'whole proved root strictly inside exact numerical bracket')
    spectral,crossing=qualify_same_root(proof);save('same-root-complement.json',spectral);save('same-root-crossing.json',crossing)
    answer.update(passed=spectral['passed']and crossing['passed'],status='qualified_local_native_ns_flip'if spectral['passed']and crossing['passed']else'completed_local_qualification_unresolved',root_passed=True,complement_passed=spectral['passed'],crossing_passed=crossing['passed'],actual_rates=proof['box'][13:15],angle_squared=proof['box'][41],crossing_derivative=crossing['multiplier_hidden_rate_derivative'])
 except BaseException as exc:answer.update(error=type(exc).__name__+': '+str(exc),status='failed_or_interrupted')
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);answer.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=calls)
  if answer['seconds']>=60 or answer['peak_rss_kib']>=1024**2:answer['passed']=False
  save('outcome.json',answer);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
