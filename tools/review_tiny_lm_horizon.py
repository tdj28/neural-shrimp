"""Independent saved causal-horizon arithmetic; no model or producer reducer imports."""
import argparse,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_response as A
need=A.need
TARGET=Path('/data2/neural-shrimp-data/explore/tiny-lm-horizon/attempt002-target')
PIN='0e4a96fd89e322951d8735945d964a5a59b526139ec97baef2da2b8f8a08391c'
WORKER='bdd4d7e1580943232231d0b1f8f68d415969c67d8cbab22f202491533c25c0c8'
EXECUTION='c59a319e0d6e2451c9d4d8ce2494cec11ae6041a'
PROFILE=TARGET.with_name('attempt002-profile')
PROFILE_PIN='cd6bcd0f2b6d13301a621bbe1613f9a8d08e122b60756c4adb27bc50e5dd8bd7'
PROFILE_WORKER='8a5f2595df633dafb2b356384d83260a5a97bb6fb7877d5ea2c64595d2fad452'
FILES=A.FILES+['tools/review_tiny_lm_horizon.py','tools/launch_tiny_lm_horizon_review.py','tests/test_tiny_lm_horizon_review.py','docs/plans/tiny-lm-horizon-independent.md']
METHODS=('drift','persistent_defect','gradient_change','loss_change','static_curvature','baseline_composite')

def same(a,b):
 if isinstance(b,dict):
  need(isinstance(a,dict)and set(a)==set(b),'complete independent columns')
  for k,v in b.items():same(a[k],v)
 elif isinstance(b,list):
  need(isinstance(a,list)and len(a)==len(b),'complete independent rows')
  for x,y in zip(a,b):same(x,y)
 else:A.same(a,b)

def crossing(values):return next((i+1 for i,v in enumerate(values)if v>.2),None)

def prediction(early,losses):
 u,v1,a2,f2,g0,g1=early
 need(abs(A.norm(u)-1)<1e-10,'unit early direction')
 defect=A.rel(f2,a2);gradient=A.rel(g1,g0);loss=abs(losses[1]-losses[0])/max(abs(losses[0]),1e-12);curvature=A.rel(u,v1)
 scores=dict(drift=[0.,defect,3*defect,6*defect],persistent_defect=[defect]*4,
  gradient_change=[h*gradient for h in range(1,5)],loss_change=[h*loss for h in range(1,5)],static_curvature=[h*curvature for h in range(1,5)])
 scores['baseline_composite']=[max(scores[k][i]for k in METHODS[1:5])for i in range(4)]
 return dict(scores=scores,predicted_crossing={k:crossing(v)for k,v in scores.items()},early_defect=defect,gradient_change=gradient,relative_loss_change=loss,directional_curvature=curvature,threshold=.2,scope='Triangular drift is a fixed growth heuristic of early defect, not an error bound or additional information.')

def evaluation(pred,u,frozen,ordered,large,small):
 qualified=A.rel(large[0],small[0])<=.01 and A.rel(ordered[0],small[0])<=.01;rows=[]
 for h in range(1,5):
  fd=A.rel(large[h],small[h]);transport=A.rel(ordered[h],small[h]);gain=A.norm(small[h]);scalar=A.norm(frozen[1])**h
  rows.append(dict(horizon=h,finite_difference_disagreement=fd,transported_error=transport,
   qualified=bool(qualified and fd<=.01 and transport<=.01),frozen_error=A.rel(frozen[h],small[h]),persistence_error=A.rel(u,small[h]),actual_gain=gain,scalar_gain=scalar,scalar_gain_error=abs(scalar-gain)/max(gain,1e-12)))
 resolved=all(v['qualified']for v in rows);errors=[v['frozen_error']for v in rows];observed=crossing(errors)if resolved else None
 eligible=bool(resolved and max(errors[:2])<=.2)
 return dict(rows=rows,origin_qualified=bool(qualified),resolved=resolved,observed_crossing=observed,right_censored=bool(resolved and observed is None),future_warning_eligible=eligible,warnings={k:bool(max(v[2:])>.2)for k,v in pred['scores'].items()},future_failure=bool(eligible and max(errors[2:])>.2))

def summary(rows):
 need(len(rows)==32,'exact32 fixed rows')
 eligible=[r['evaluation']for r in rows if r['evaluation']['future_warning_eligible']];counts={}
 for name in METHODS:
  c=dict(true_warning=0,false_warning=0,missed_failure=0,correct_quiet=0)
  for e in eligible:
   w,f=e['warnings'][name],e['future_failure'];key={ (True,True):'true_warning',(True,False):'false_warning',(False,True):'missed_failure',(False,False):'correct_quiet'}[w,f];c[key]+=1
  counts[name]=c
 return dict(total_rows=32,resolved_rows=sum(r['evaluation']['resolved']for r in rows),future_eligible_rows=len(eligible),already_failing_rows=sum(r['evaluation']['resolved']and not r['evaluation']['future_warning_eligible']for r in rows),future_warning_counts=counts,drift_only_caught_failures=sum(e['future_failure']and e['warnings']['drift']and not e['warnings']['baseline_composite']for e in eligible),scope='Correlated consumed development prefixes; no population inference, fitting, intervention or claimed generalization.')

def exact(actual,expected,message):need(np.array_equal(actual,expected),message)

def main(out):
 need(out==Path("/data2/neural-shrimp-data/audits/tiny-lm-horizon-independent/attempt001/run"),"fixed exclusive audit output")
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in A.THREADS),'CPU1nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();written=0;result=dict(passed=False,native_calls=0,rows=[])
 def check(*args):
  if args or time.monotonic()-start>=60:raise TimeoutError('60s saved independent audit')
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'1GiB RSS')
 def write(name,data):
  nonlocal written
  need(written+len(data)<4*1024**2,'4MiB worker cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data)
  written+=len(data)
 def save(name,value):write(name,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
 def arr(folder,name,shape):
  check();v=np.load(folder/(name+'.npy'),allow_pickle=False);need(v.dtype==np.float64 and v.shape==shape and np.isfinite(v).all(),'finite exact retained array '+name);return v
 signal.signal(signal.SIGTERM,check);signal.signal(signal.SIGALRM,check);signal.alarm(60)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();sources={n:A.sha(ROOT/n)for n in FILES}
  for n,h in sources.items():
   data=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==h,'committed audit source');write('source/'+n,data)
  save('runtime.json',dict(source_commit=commit,sources=sources,pid=os.getpid(),numpy=np.__version__,threads=dict.fromkeys(A.THREADS,'1'),native_calls=0))
  run=TARGET/'run';inventory_inputs=[(TARGET,PIN),(run,WORKER),(PROFILE,PROFILE_PIN),(PROFILE/'run',PROFILE_WORKER),(A.TARGET,A.PIN),(A.TARGET/'run','cb57135497afa72437651b7eaab807e58387a304fd0f0d2609113ffcff073aad'),(A.PARENT,A.PARENT_PIN),
   (Path('/data2/neural-shrimp-data/audits/tiny-lm-response-independent/attempt001'),'296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948'),
   (Path('/data2/neural-shrimp-data/audits/tiny-lm-two-snapshot-independent/attempt001'),'7378d61f7e7b50789156f0195407ea302be9093d71612ece6514ead2673ab74d'),
   (TARGET.with_name('attempt001-profile'),'b5475b36a5e22d5381473fe490bedfbfcf422e01ab8880bccf40d6e1891925a7')]
  result['inventories']=[A.inventory(p,h,True)for p,h in inventory_inputs]
  for p,h in inventory_inputs:need(not(p/'manifest.json').stat().st_mode&0o222,'readonly input manifest')
  for p,h in inventory_inputs[7:9]:need(A.load(p/'run/result.json')['passed'],'prior independent audit passed')
  failed=A.load(inventory_inputs[-1][0]/'run/result.json');need(not failed['passed']and failed['operation_attempted']==failed['operation_returned']==dict(hvp=0,ce=0),'preserved zero-native failed profile')
  source=A.load(run/'result.json');runtime=A.load(run/'runtime.json');launch=A.load(TARGET/'launcher-result.json');profile=A.load(PROFILE/'run/result.json');pr=A.load(PROFILE/'run/runtime.json');pl=A.load(PROFILE/'launcher-result.json')
  need(runtime['source_commit']==launch['source_commit']==EXECUTION and source['passed']and source['status']=='causal_horizon_complete'and source['protected_outcomes_evaluated']is False,'exact complete target')
  for la in (launch,pl):need(la['exit_code']==0 and not la['owned_live_pids']and la['worker_pid_absent'],'owned completed worker')
  for rt in (runtime,pr):
   need(not Path('/proc',str(rt['pid'])).exists(),'owned native worker absent')
   for n,row in rt['sources'].items():
    folder=run if rt is runtime else PROFILE/'run'
    need(A.sha(folder/'source'/n)==row['sha256']and hashlib.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'all archived native source bytes')
  need(pr['sources']==runtime['sources']and all(pr[k]==runtime[k]for k in ('python','numpy','torch','dtype','device','threads')),'same measured profile source/runtime')
  need(profile['passed']and profile['operation_caps']==profile['operation_attempted']==profile['operation_returned']==dict(hvp=4,ce=4),'complete profile')
  costs=profile['profile_costs'];need([r['arm']for r in costs]==[8,256],'both cost arms');forecast=20.
  for r in costs:
   need(set(r['times'])=={'hvp','ce'},'both native cost types')
   for kind,count in [('hvp',36),('ce',16)]:
    v=r['times'][kind];need(len(v)==2 and all(math.isfinite(t)and t>0 for t in v),'two positive measurements');forecast+=2*count*max(v)
  binding=A.load(run/'profile-binding.json');need(binding==dict(root=str(PROFILE),outer=PROFILE_PIN,worker=PROFILE_WORKER,forecast_seconds=forecast)and forecast==profile['forecast_seconds']<180,'same measured admission')
  need(source['operation_caps']==source['operation_attempted']==source['operation_returned']==dict(hvp=72,ce=32),'exact fixed native counts')
  seal=A.load(run/'predictions-closed.json');need(seal['count']==32 and seal['evaluation_arrays_unopened']is True and len(seal['files'])==32,'all32 sealed before outcomes')
  events=A.load(run/'events.json');counts=dict(hvp=0,ce=0);event_index=0;labels=[]
  def operation(kind,theta,vector=None):
   nonlocal event_index
   counts[kind]+=1;i=counts[kind];prefix=f'{kind}-{i:03d}';folder=run/'operations'
   exact(arr(folder,prefix+'-theta',(15776,)),theta,'exact native local state')
   if vector is not None:exact(arr(folder,prefix+'-vector',(15776,)),vector,'exact native frozen direction')
   attempt=A.load(folder/(prefix+'-attempt.json'));need(attempt==dict(operation=kind,index=i,returned=False),'saved pre-call attempt')
   end=A.load(folder/(prefix+'-result.json'));need(end==events[event_index]and end['operation']==kind and end['index']==i and end['returned']is True and math.isfinite(end['seconds'])and end['seconds']>0,'exact ordered native event');event_index+=1
   return arr(folder,prefix+'-return',(15776,)if kind=='hvp'else ())
  for seed in (28161,28162):
   for arm in (8,256):
    label=f'seed-{seed}-arm-{arm}';old=A.TARGET/'run'/label;folder=run/label;md=A.load(old/'inputs.json');need(A.load(folder/'inputs.json')==md and md['seed']==seed and md['arm']==arm and md['step']==600,'same exact original map/data');eta=md['eta'];need(eta>0,'positive learning rate')
    ref=arr(old,'reference',(17,15776));grads=arr(old,'reference-gradients',(17,15776));exact(ref[1:],ref[:-1]-eta*grads[:-1],'complete actual reference GD recurrence')
    tangents=[arr(old,f'direction-{d}-transported',(17,15776))for d in (0,1)];originalf=[arr(old,f'direction-{d}-frozen',(17,15776))for d in (0,1)]
    for t in (0,4,8,12):
     losses=[float(operation('ce',v))+.0005*A.dot(v,v)for v in ref[t:t+2]]
     for d in (0,1):
      rowlabel=label+f'/origin-{t:02d}-direction-{d}';labels.append(rowlabel);rowfolder=run/rowlabel;record=A.load(rowfolder/'prediction.json');scale=record['normalization'];need(math.isfinite(scale)and scale>0 and math.isclose(scale,A.norm(tangents[d][t]),rel_tol=2e-15,abs_tol=0),'independent original tangent norm')
      for n,e in seal['files'][rowlabel].items():need(set(seal['files'][rowlabel])=={'prediction.json','early-vectors.npy','early-states.npy','frozen.npy'}and e==dict(bytes=(rowfolder/n).stat().st_size,sha256=A.sha(rowfolder/n)),'exact predictor seal')
      frozen=arr(rowfolder,'frozen',(5,15776));early=arr(rowfolder,'early-vectors',(6,15776));exact(arr(rowfolder,'early-states',(2,15776)),ref[t:t+2],'two local states only')
      exact(early,np.array([tangents[d][t]/scale,tangents[d][t+1]/scale,tangents[d][t+2]/scale,frozen[2],grads[t],grads[t+1]]),'six exact early vector inputs');exact(frozen[:2],early[:2],'same first action')
      if t==0:exact(frozen,originalf[d][:5]/scale,'exact inherited origin0 frozen forecast')
      else:
       for j in (1,2,3):exact(frozen[j+1],frozen[j]-eta*operation('hvp',ref[t],frozen[j]),'exact later frozen action recurrence')
      same(record['losses'],losses);pred=prediction(early,losses);same(record['prediction'],pred)
      need((record['seed'],record['arm'],record['origin'],record['direction'],record['label'])==(seed,arm,t,d,rowlabel),'fixed32 case identity')
      saved=A.load(rowfolder/'evaluation.json');need({k:saved[k]for k in record}==record,'evaluation uses sealed prediction record')
      eps=A.load(old/'outcome.json')['epsilon_values'];need(len(eps)==2 and eps[0]==2*eps[1]and all(math.isfinite(e)and e>0 for e in eps),'two original positive amplitudes');responses=[]
      for k,name in [(0,'response-large'),(1,'response-small')]:
       plus=arr(old,f'direction-{d}-epsilon-{k}-plus',(17,15776));minus=arr(old,f'direction-{d}-epsilon-{k}-minus',(17,15776));r=(plus[t:t+5]-minus[t:t+5])/(2*eps[k]*scale);exact(arr(rowfolder,name,(5,15776)),r,'exact reconstructed finite difference');responses.append(r)
      orderedpath=old/f'direction-{d}-transported.npy';need(saved['ordered_source']==dict(path=str(orderedpath),identity=dict(bytes=orderedpath.stat().st_size,sha256=A.sha(orderedpath)),slice=[t,t+5],normalization=scale),'exact actual ordered source slice')
      answer=evaluation(pred,early[0],frozen,tangents[d][t:t+5]/scale,*responses);same(saved['evaluation'],answer)
      index=len(result['rows']);need(source['rows'][index]==saved,'complete source row order');fresh=dict(seed=seed,arm=arm,origin=t,direction=d,prediction=pred,evaluation=answer);result['rows'].append(fresh);save(f'row-{index:02d}.json',fresh)
  need(set(labels)==set(seal['files'])and len(events)==event_index==104 and counts==dict(hvp=72,ce=32)and len(source['rows'])==32,'all cases and operations exhausted')
  measured=summary(result['rows']);same(source['summary'],measured);check();result.update(passed=True,summary=measured,forecast_seconds=forecast,retained_native_operations_checked=counts,rows_recomputed=32)
 except BaseException as exc:
  import traceback;result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.alarm(0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent saved32-row arithmetic and exact native input/output recurrences; acquired HVP/CE and original ordered transport remain shared. Causality is source/seal separation, not an independent reacquisition. Consumed correlated development only; no utility/generalization claim.');save('result.json',result);save('manifest.json',dict(files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=A.sha(p))for p in out.rglob('*')if p.is_file()}))
 return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
