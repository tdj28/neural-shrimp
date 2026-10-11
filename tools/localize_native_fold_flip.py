"""Fixed numerical41D plus/minus-one localization from retained native route."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import scout_native470_flip_extension as D
C=D.C;need,read,entry=C.need,C.read,C.entry
from neural_shrimp.fold_flip_scout import augmented,qualify
from neural_shrimp.signed_flip_scout import evaluate,flat,qualify as signed_qualify
OUT=D.OUT.parents[2]/'fold-flip-localization001/attempt001/run';CARD='docs/plans/native-fold-flip-localization-inputs.json'
SETTINGS=dict(period=18,newton_evaluations=12,adjacent_evaluations=1,reverse_evaluations=18,profile_evaluations=12,native_calls=1224,seconds=240,rss_bytes=3*1024**3,as_bytes=8*1024**3,raw_bytes=256*1024**2)
FILES=sorted(set(D.FILES+[CARD,'docs/plans/native-fold-flip-localization.md','docs/reviews/native470-flip-extension002.json','src/neural_shrimp/fold_flip_scout.py','tools/localize_native_fold_flip.py','tools/launch_native_fold_flip.py','tests/test_native_fold_flip_driver.py','tests/test_fold_flip_scout.py']))

def owned_source(w,p,spec,exitcode):
 for b,k in ((p,'outer'),(p/'run','worker')):C.C.L.K.inventory(b,spec[k]['sha256'],w)
 rt=read(p/'run/runtime.json');la=read(p/'launcher-result.json');o=read(p/'run/outcome.json')
 need(rt['source_commit']==la['source_commit']==spec['source']and o==spec['outcome']and la['exit_code']==exitcode and la['failure']is None and not la['owned_live_pids']and all(not Path('/proc',str(la[k])).exists()for k in ('launcher_pid','owned_group_pid')),'complete immutable failed extension/source/ownership')
 for n,v in rt['sources'].items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'literal failed extension source');w.check()
 for n in ('src/neural_shrimp/signed_flip_scout.py','src/neural_shrimp/unit_pair_scout.py'):need(entry(ROOT/n)==rt['sources'][n],'identical actual native map equations')
 return o

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS,'fixed reviewed41 packet');old=C.SETTINGS
 try:C.SETTINGS=D.SETTINGS;D.binding(w,card['preceding'],metadata_only=True)
 finally:C.SETTINGS=old
 spec=card['failed'];p=Path(spec['root']);need(p==D.OUT.parent,'literal failed extension');o=owned_source(w,p,spec,2)
 need(o['error']=='RuntimeError: fixed full return cap'and o['native_calls']==3096 and spec['accepted_points']==17 and spec['full470_reverse_joins']==16,'retained capped attempt')
 need(entry(p/spec['last_point_file'])==spec['last_point_pin'],'literal accepted final point')
 if metadata_only:return p
 def events(name):return [(f,json.loads(gzip.decompress(f.read_bytes())))for f in sorted((p/'run/events').glob('*-target-'+name+'.json.gz'))]
 points=events('accepted_point');attempts=events('continuation_attempt');ei=events('evaluation_input');er=events('evaluation_return');joins=[v for _,v in attempts if v['passed']]
 need(len(points)==17 and [v['index']for _,v in points]==list(range(17))and all(v['diagnostics']['passed']for _,v in points)and [v['index']for v in joins]==list(range(16))and all(v['reverse_coordinates']==470 and v['reverse_error']<1e-7 for v in joins),'entire accepted prefix and full470 reverse joins')
 need([v['index']for _,v in ei]==[v['index']for _,v in er]==list(range(1,161)),'complete retained160 evaluations')
 last=points[-1][1];matches=[i for i,(_,v)in enumerate(ei)if v['hidden']==last['hidden']and v['z']==last['z']and int(er[i][0].name.split('-')[0])<int(points[-1][0].name.split('-')[0])];need(matches,'accepted native root return association');i=matches[-1];data=er[i][1]['native'];M=np.asarray(data['monodromy']);values,vectors=np.linalg.eig(M);j=int(np.argmin(abs(values-1)));q=vectors[:,j]
 need(abs(values[j].imag)<1e-10 and max(abs(q.imag))<1e-10 and abs(values[j]-1)<.03,'literal near-positive real mode');q=q.real/np.linalg.norm(q.real)
 if q[np.argmax(abs(q))]<0:q=-q
 z=np.asarray(last['z']);seed=np.r_[z[:13],last['hidden'],z[13],q,z[14:]];inp=read(Path(card['preceding']['proof']['root'])/'run/target-inputs.json')
 w.save('input-binding.json',dict(failed=spec,preceding=card['preceding'],map_inputs=inp['map_inputs'],accepted_evaluation=ei[i][1]['index'],accepted_return=entry(er[i][0]),scope='Exact retained numerical route, not a proved branch connection.'));w.save('seed-proposal.json',dict(seed=seed,near_positive_eigenvalue=[float(values[j].real),float(values[j].imag)],targets=[v for _,v in points[-2:]]));return C.TanhMap(inp['map_inputs']),seed,[v for _,v in points[-2:]]

def profile(w):
 inp,x=C.C.L.K.fixture();n=13;y=np.r_[[float(v.mid())for v in x[:n+2]],np.ones(n)/np.sqrt(n),np.arange(1,n+1)/np.linalg.norm(np.arange(1,n+1))];model=C.TanhMap(inp);times=[];sizes=[];w.save('profile-inputs.json',dict(map_inputs=inp,y=y))
 for i in range(12):
  start=time.monotonic();size=w.used;w.observe('evaluation_input',dict(index=i,y=y));v,J,d=augmented(model,y,18,w.tick,w.observe);w.observe('evaluation_return',dict(index=i,value=v,jacobian=J,native=d));times.append(time.monotonic()-start);sizes.append(w.used-size)
 forecast=time.monotonic()-w.start+44*max(times)+30;raw=w.used+44*max(sizes)+32*1024**2;rss=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024
 a=dict(passed=forecast<240 and raw<256*1024**2 and rss<3*1024**2,seconds=times,bytes=sizes,forecast_seconds=forecast,forecast_bytes=raw,forecast_rss_kib=rss,native_calls=w.calls,target_arrays_unopened=True);w.save('profile.json',a);need(a['passed']and w.calls==432,'full unrelated41 measured profile')

def localize(model,seed,targets,w):
 y=seed.copy();last=None
 for i in range(12):
  w.observe('augmented_input',dict(index=i,y=y));v,J,d=augmented(model,y,18,w.tick,w.observe);w.observe('augmented_return',dict(index=i,value=v,jacobian=J,native=d));last=(v,J,d)
  if max(abs(v))<1e-10:break
  need(i<11,'fixed twelve augmented evaluations');step=np.linalg.solve(J,-v);w.observe('augmented_step',dict(index=i,step=step));need(max(abs(step)/np.maximum(1.,abs(y)))<.1,'local full41 step');y+=step
 else:raise AssertionError('unreachable')
 augmented_count=i+1;v,J,d=last;report=qualify(y,v,d);need(report['passed']and abs(y[13]-seed[13])<.3 and abs(y[14]-seed[14])<.02 and max(abs(y[:13]-seed[:13]))<.1,'complete nearby native plus/minus proposal')
 qp=d['q_phases'][1];wp=d['w_phases'][1];adj=np.r_[d['orbit'][1],y[13:15],qp/np.linalg.norm(qp),wp/np.linalg.norm(wp)];w.observe('adjacent_input',dict(y=adj));av,aJ,ad=augmented(model,adj,18,w.tick,w.observe);w.observe('adjacent_return',dict(value=av,jacobian=aJ,native=ad));ar=qualify(adj,av,ad);need(ar['passed'],'literal adjacent phase primitive/simple/complement')
 joins=[]
 for target in targets:
  z=np.r_[y[:13],y[14],y[28:]];h=target['hidden'];z[13]=target['z'][13]
  for i in range(9):
   w.observe('reverse_input',dict(target=target['index'],iteration=i,z=z,hidden=h));rv,rJ,rh,rd=evaluate(model,z,h,18,w.tick,w.observe);w.observe('reverse_return',dict(target=target['index'],iteration=i,value=rv,jacobian=rJ,hidden_derivative=rh,native=rd))
   if max(abs(rv))<1e-10:break
   need(i<8,'fixed reverse corrector');step=np.linalg.solve(rJ,-rv);w.observe('reverse_step',dict(target=target['index'],iteration=i,step=step));need(max(abs(step)/np.maximum(1.,abs(z)))<.1,'local reverse fullstate step');z+=step
  actual=flat(z,h,rd);old=np.asarray(target['native_flat']);error=float(max(abs(actual-old)/np.maximum(1.,abs(old))));sr=signed_qualify(z,h,rv,rd);join=dict(target=target['index'],passed=bool(error<1e-7 and sr['passed']),coordinates=470,relative_error=error,qualification=sr);w.observe('reverse_association',join);need(join['passed'],'full470 reverse association to saved nearby points');joins.append(join)
 return dict(passed=True,root=y,diagnostics=report,adjacent=ar,reverse=joins,augmented_evaluations=augmented_count,scope='Numerical native plus/minus-one root with adjacent/reverse association only. No rigorous root, unfolding, criticality or new interval connection.')

def main():
 C.OUT=OUT;C.SETTINGS=SETTINGS;resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=C.Work();ans=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('240s owned')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,sources={n:entry(ROOT/n)for n in FILES},numpy=np.__version__,python=platform.python_version(),threads=dict.fromkeys(C.THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  profile(w);model,seed,targets=binding(w,read(ROOT/CARD));w.stage='target';ans['target_acquired']=True;result=localize(model,seed,targets,w);w.save('localization.json',result);ans.update(passed=True,status='numerical_plus_minus_localized',rates=result['root'][13:15],diagnostics=result['diagnostics'],reverse=result['reverse'])
 except BaseException as e:ans.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);ans.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.calls,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(C.safe(ans)));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
