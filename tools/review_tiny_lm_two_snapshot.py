"""Independent saved240-action two-snapshot forecast outcome; zero native calls."""
import argparse,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_tiny_lm_response as A
TARGET=Path('/data2/neural-shrimp-data/explore/tiny-lm-two-snapshot/attempt001')
PIN='c4ca66df6fdc7d848b98e3d232b0e1b99055a86130e2aae8fe52548ca5f1a91b'
WORKER='1b734e6179b41c0e7561d290d8313e82434b15819252553badfd45e8b5483591'
EXECUTION='a86bb1e3a3332814f3db0040eda9ff1841891a89'
FILES=A.FILES+['tools/review_tiny_lm_two_snapshot.py','tools/launch_tiny_lm_two_snapshot_review.py','tests/test_tiny_lm_two_snapshot_review.py','docs/plans/tiny-lm-two-snapshot-independent.md']
need=A.need

def recurrence(prediction,initial,eta,actions):
 need(np.array_equal(prediction[0],initial)and len(prediction)==16 and len(actions)==15,'exact v1 and15actions')
 for k,action in enumerate(actions,1):need(np.isfinite(action).all()and np.array_equal(prediction[k],prediction[k-1]-eta*action),'exact saved action recurrence')

def fixed_gate(rows):
 selected=[r for r in rows if r['arm']==256 and r['direction']=='gradient'and r['horizon']==16]
 need(len(selected)==2 and {r['seed']for r in selected}=={28161,28162},'two fixed expanded gradient cases')
 return bool(all(r['mechanism_qualified']and r['alternating_vector_error']<.2 and 10*r['alternating_vector_error']<r['latest_vector_error']and 10*r['alternating_vector_error']<r['persistence_vector_error']for r in selected))

def main(out):
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in A.THREADS),'CPU1nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();written=0;result=dict(passed=False,rows=[],native_calls=0)
 def check(*args):
  if args or time.monotonic()-start>=30:raise TimeoutError('30s saved audit')
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'1GiB peak')
 def save(name,value):
  nonlocal written
  data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();need(written+len(data)<2*1024**2,'2MiB worker cap');written+=len(data)
  with(out/name).open('xb')as f:f.write(data)
 signal.signal(signal.SIGTERM,check);signal.signal(signal.SIGALRM,check);signal.alarm(30)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();sources={n:A.sha(ROOT/n)for n in FILES}
  for n,h in sources.items():need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==h,'committed source')
  save('runtime.json',dict(source_commit=commit,sources=sources,pid=os.getpid(),numpy=np.__version__,threads=dict.fromkeys(A.THREADS,'1'),native_calls=0))
  run=TARGET/'run';result['inventories']=[A.inventory(TARGET,PIN,True),A.inventory(run,WORKER,True),A.inventory(A.TARGET,A.PIN,True),A.inventory(A.PROFILE,A.PROFILE_PIN,True)]
  prior_audit=Path('/data2/neural-shrimp-data/audits/tiny-lm-response-independent/attempt001');result['inventories'].append(A.inventory(prior_audit,'296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948',True));need(A.load(prior_audit/'run/result.json')['passed'],'prior independent native response audit')
  source=A.load(run/'result.json');runtime=A.load(run/'runtime.json');launch=A.load(TARGET/'launcher-result.json');need(runtime['source_commit']==launch['source_commit']==EXECUTION and source['passed']and launch['exit_code']==0 and not launch['owned_live_pids'],'complete fixed producer')
  need(source['hvp_attempted']==source['hvp_returned']==240 and len(source['rows'])==16,'complete acquired240/16')
  for name,row in runtime['sources'].items():need(A.sha(run/'source'/name)==row['sha256']and hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==row['sha256'],'archived historical source')
  old=A.load(A.TARGET/'run/result.json');profile=A.load(A.PROFILE/'result.json');need(profile['passed']and profile['operation_returned']==dict(gradient=4,hvp=4,ce=4),'original operation profile')
  for record in [old,profile]:need(record['torch']==runtime['torch']and record['numpy']==runtime['numpy']and all(runtime['sources'][n]==v for n,v in record['source_identity'].items()),'exact original source/library costs')
  costs=profile['profile_costs'];need([v['arm']for v in costs]==[8,256],'both fixed cost arms');need(all(len(v['times']['hvp'])==2 and all(math.isfinite(t)and t>0 for t in v['times']['hvp'])for v in costs),'two measured HVP costs')
  forecast=240*sum(max(v['times']['hvp'])for v in costs)+20;need(forecast<180 and forecast==source['forecast_seconds']==A.load(run/'bindings.json')['forecast_seconds'],'exact measured forecast')
  events=A.load(run/'events.json');expected=[]
  def array(folder,name,shape):
   check();v=np.load(folder/(name+'.npy'),allow_pickle=False);need(v.dtype==np.float64 and v.shape==shape and np.isfinite(v).all(),'finite full retained array');return v
  for seed in [28161,28162]:
   for arm in [8,256]:
    label=f'seed-{seed}-arm-{arm}';oldfolder=A.TARGET/'run'/label;folder=run/label;md=A.load(oldfolder/'inputs.json');eta=md['eta'];case=A.load(oldfolder/'outcome.json');directions=array(oldfolder,'directions',(2,15776));readout=array(oldfolder,'readout-gradients',(3,15776))
    need(eta>0 and md['seed']==seed and md['arm']==arm and md['step']==600,'exact inherited case')
    for d,name in enumerate(['gradient','random']):
     transported=array(oldfolder,f'direction-{d}-transported',(17,15776));frozen=array(oldfolder,f'direction-{d}-frozen',(17,15776));responses=[];slopes=[]
     for ep,epsilon in enumerate(case['epsilon_values']):
      plus=array(oldfolder,f'direction-{d}-epsilon-{ep}-plus',(17,15776));minus=array(oldfolder,f'direction-{d}-epsilon-{ep}-minus',(17,15776));responses.append((plus-minus)/(2*epsilon))
      cp=array(oldfolder,f'direction-{d}-epsilon-{ep}-plus-ce',(3,2));cm=array(oldfolder,f'direction-{d}-epsilon-{ep}-minus-ce',(3,2));need(np.array_equal(cp[:,0],[1,4,16])and np.array_equal(cm[:,0],[1,4,16]),'exact auxiliary readout horizons');slopes.append((cp[:,1]-cm[:,1])/(2*epsilon))
     one=bool(A.rel(responses[0][1],responses[1][1])<=.01 and A.rel(transported[1],responses[1][1])<=.01)
     predictions={}
     for predictor in ['alternating','latest']:
      v=array(folder,f'direction-{d}-'+predictor,(16,15776));actions=[array(folder,f'direction-{d}-{predictor}-hvp-{k:02d}',(15776,))for k in range(1,16)];recurrence(v,transported[1],eta,actions);predictions[predictor]=v
      expected.extend([(label,1 if predictor=='latest'or k%2 else 0)for k in range(1,16)])
     measured=[]
     for h in [4,16]:
      target=responses[1][h];slope=float(slopes[1][[1,4,16].index(h)]);r=dict(seed=seed,arm=arm,direction=name,horizon=h,mechanism_qualified=bool(one and A.rel(responses[0][h],target)<=.01 and A.rel(transported[h],target)<=.01),persistence_vector_error=A.rel(directions[d],target),frozen_vector_error=A.rel(frozen[h],target))
      for predictor,v in predictions.items():
       value=v[h-1];pred=A.dot(readout[[1,4,16].index(h)],value);r.update({predictor+'_vector_error':A.rel(value,target),predictor+'_gain':A.norm(value),predictor+'_ce_slope':pred,predictor+'_ce_slope_relative_error':abs(pred-slope)/max(abs(slope),1e-8)})
      measured.append(r)
     saved=A.load(folder/f'direction-{d}-outcome.json');need(len(saved)==2,'two horizon rows')
     for fresh,retained in zip(measured,saved):
      need(set(fresh)==set(retained),'complete comparison columns')
      for k,v in fresh.items():A.same(retained[k],v)
     result['rows'].extend(measured);save(label+f'-direction-{d}.json',measured)
  need(len(events)==len(expected)==240,'exact action event count')
  for i,(event,(case,snapshot))in enumerate(zip(events,expected)):need(event['case']==case and event['index']==i and event['snapshot']==snapshot and event['returned']is True and math.isfinite(event['seconds'])and event['seconds']>0,'exact causal snapshot/case action order')
  need(len(result['rows'])==16,'all16 rows')
  for fresh,saved in zip(result['rows'],source['rows']):
   for k,v in fresh.items():A.same(saved[k],v)
  gate=fixed_gate(result['rows']);need(gate is source['candidate_gate'],'independent fixed candidate gate');check();result.update(passed=True,candidate_gate=gate,forecast_seconds=forecast,actions_recomputed=240,rows_recomputed=16)
 except BaseException as exc:
  import traceback;result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.alarm(0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent saved240HVP recurrence, math.fsum errors/gains/readout slopes and fixed16-row gate. Shared acquired HVP and earlier finite-difference/CE arrays not reevaluated. Consumed-development causal input restriction, no heldout confirmation or demonstrated utility.');save('result.json',result);save('manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=A.sha(p))for p in out.iterdir()if p.is_file()}))
 return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
