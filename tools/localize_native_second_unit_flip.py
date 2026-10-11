"""One measured numerical second complex-unit/flip bracket localization."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_output_wide as S
C=S.C;need,entry=C.need,C.entry
def read(p):
 with (gzip.open(p,'rb')if p.suffix=='.gz'else p.open('rb'))as f:raw=f.read(256*1024**2+1)
 need(len(raw)<=256*1024**2,'bounded256MiB JSON');return json.loads(raw)
from neural_shrimp.second_unit_flip import seed_from_signed,localize,augmented,native_direction,evaluate
OUT=S.OUT.parents[2]/'second-unit-flip-localization001/attempt001/run';CARD='docs/plans/second-unit-flip-inputs.json'
CAPS=dict(augmented=12,section=19,adjacent=1)
SETTINGS=dict(period=18,profile_each=6,operation_caps=CAPS,native_calls=1548,seconds=240,rss_bytes=3*1024**3,as_bytes=8*1024**3,raw_bytes=256*1024**2)
FILES=sorted(set(S.FILES+[CARD,'docs/plans/second-unit-flip.md','docs/reviews/native470-output-scout003.json','src/neural_shrimp/second_unit_flip.py','src/neural_shrimp/unit_pair_flip.py','src/neural_shrimp/unit_pair_section.py','tools/localize_native_second_unit_flip.py','tools/launch_native_second_unit_flip.py','tests/test_second_unit_flip.py','tests/test_second_unit_flip_driver.py']))

def jet(v):return dict(f=v.f,g=v.g,d=v.d,m=v.m)
class Retained:
 def __init__(self,m,w):self.model=m;self.n=m.n;self.w=w;self.index=0
 def cusp_step(self,z,h,o):
  i=self.index;self.index+=1;self.w.observe('native_input',dict(index=i,inputs=[jet(v)for v in list(z)+[h,o]]));out=self.model.cusp_step(z,h,o);self.w.observe('native_return',dict(index=i,returns=[jet(v)for v in out]));return out

def call(model,w,counts,kind,arg,profile=False):
 cap=6 if profile else CAPS[kind]
 if counts[kind]>=cap:raise RuntimeError('fixed '+kind+' evaluation cap')
 counts[kind]+=1;w.observe('evaluation_input',dict(kind=kind,index=counts[kind],argument=arg));t=time.monotonic();used=w.used
 if kind=='augmented':
  result=augmented(model,arg,18,w.tick);v,J,d=result;row=dict(value=v,jacobian=J,native=d)
 elif kind=='section':
  result=evaluate(model,arg[0],arg[1],18,w.tick);v,J,do,d=result;row=dict(value=v,jacobian=J,output_derivative=do,native=d)
 else:
  n=model.n;result=native_direction(model,arg[:n],arg[n:n+2],arg[n+2:],18,w.tick);f,g,mw,mixed,d=result;row=dict(value=f,first=g,direction=mw,mixed=mixed,native=d)
 w.observe('evaluation_return',dict(kind=kind,index=counts[kind],**row));return result,time.monotonic()-t,w.used-used

def profile(w):
 inp,x=C.C.L.K.fixture();z=np.array([float(v.mid())for v in x]);n=13;y=np.r_[z[:41],.014,np.eye(n)[0]];model=Retained(C.TanhMap(inp),w);counts=dict.fromkeys(CAPS,0);times={k:[]for k in CAPS};sizes={k:[]for k in CAPS};w.save('profile-inputs.json',dict(map_inputs=inp,y=y))
 for i in range(6):
  for kind,arg in [('augmented',y),('section',(np.r_[y[:n],y[n],y[42:]],float(y[n+1]))),('adjacent',np.r_[y[:15],y[42:]])]:
   _,dt,size=call(model,w,counts,kind,arg,True);times[kind].append(dt);sizes[kind].append(size)
 seconds=time.monotonic()-w.start+2*sum(CAPS[k]*max(times[k])for k in CAPS)+30;raw=w.used+2*sum(CAPS[k]*max(sizes[k])for k in CAPS)+32*1024**2;rss=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024;ok=seconds<240 and raw<256*1024**2 and rss<3*1024**2
 w.save('profile.json',dict(passed=ok,times=times,sizes=sizes,counts=counts,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True,native_calls=w.calls));need(ok and w.calls==540,'complete unrelated55/section/adjacent measured profile')

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS,'fixed second-junction design');S.binding(w,card['previous_card'],metadata_only=True);r=card['previous'];p=Path(r['root']);need(p==S.OUT.parent,'literal last wider numerical route')
 o=S.D.F.owned_source(w,p,r,0);need(o['passed']and o['status']=='noncritical_unit_crossing_proposal'and o['points']==24 and o['evaluations']==152 and o['native_calls']==2952,'complete exact sampled complex crossing')
 need(entry(p/'run/survey.json')==r['survey']and entry(p/'run/initial-tube-association.json')==r['initial_association']and read(p/'run/source'/S.CARD)==card['previous_card'],'literal complete survey and reviewed ancestry')
 if metadata_only:return p
 survey=read(p/'run/survey.json');points=survey['points'];need(len(points)==24 and all(v['diagnostics']['passed']for v in points),'all native points qualified');targets=points[-2:];need(targets[0]['diagnostics']['complement_max_modulus']<1<targets[1]['diagnostics']['complement_max_modulus'],'actual bracketing unit modulus')
 def rows(name):return [(f,read(f))for f in sorted((p/'run/events').glob('*-target-'+name+'.json.gz'))]
 ins=rows('evaluation_input');outs=rows('evaluation_return');joins=[v for _,v in rows('continuation_attempt')if v['passed']];accepted=rows('accepted_point')
 need(len(ins)==len(outs)==152 and [v['index']for _,v in ins]==[v['index']for _,v in outs]==list(range(1,153))and [v for _,v in accepted]==points,'all acquisition and accepted records')
 need([v['index']for v in joins]==list(range(23))and all(v['reverse_coordinates']==470 and v['reverse_error']<1e-7 for v in joins),'all23 literal native reverse joins')
 chosen=targets[1];matches=[i for i,(_,v)in enumerate(ins)if v['z']==chosen['z']and v['output']==chosen['output']and int(outs[i][0].name.split('-')[0])<int(accepted[-1][0].name.split('-')[0])];need(matches,'retained accepted unstable return');idx=matches[-1];data=outs[idx][1]['native'];M=np.asarray(data['monodromy']);z=np.asarray(chosen['z']);flat=np.asarray(chosen['native_flat'])
 need(z.shape==(27,)and flat.shape==(470,)and np.array_equal(z[:13],flat[:13])and np.array_equal(z[14:],flat[13:26])and z[13]==flat[-2]==chosen['hidden']and chosen['output']==flat[-1]and np.array_equal(data['orbit'][0],z[:13])and np.array_equal(data['modes'][0],z[14:]),'literal same state/negative-mode/physical-rate seed')
 seed,diagnostic=seed_from_signed(chosen,M);inp=read(p/'run/input-binding.json')['map_inputs'];w.save('input-binding.json',dict(previous=r,previous_card=card['previous_card'],map_inputs=inp,targets=targets,accepted_return=entry(outs[idx][0]),accepted_evaluation=ins[idx][1]['index'],scope='Second junction candidate on a numerical negative-one route, distinct from first independently proved55.'))
 w.save('seed-proposal.json',dict(y=seed,diagnostics=diagnostic,targets=targets));return C.TanhMap(inp),seed,targets

def main():
 C.OUT=OUT;C.SETTINGS=SETTINGS;resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=C.Work();answer=dict(passed=False,target_acquired=False);counts=dict.fromkeys(CAPS,0)
 def stop(*_):raise TimeoutError('240s owned second junction')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,sources={n:entry(ROOT/n)for n in FILES},numpy=np.__version__,python=platform.python_version(),threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  profile(w);model,seed,targets=binding(w,read(ROOT/CARD));w.stage='target';answer['target_acquired']=True;m=Retained(model,w);result=localize(m,targets,seed,lambda kind,arg:call(m,w,counts,kind,arg)[0],w.observe);w.save('candidate.json',result);answer.update(passed=True,status='second_numerical_unitpair_flip_localized',rates=result['y'][13:15],angle_squared=result['y'][41],phase0_spectrum=result['phase0_spectrum'],phase1_spectrum=result['phase1_spectrum'],crossing_derivative=result['crossing_derivative'],reverse=[{k:v for k,v in r.items()if k!='native_flat'}for r in result['bracket_reverse']])
 except BaseException as e:answer.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;answer['passed']=bool(answer['passed']and elapsed<240 and peak<3*1024**2);answer.update(seconds=elapsed,peak_rss_kib=peak,native_calls=w.calls,counts=counts,events=w.serial);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(C.safe(answer)));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
