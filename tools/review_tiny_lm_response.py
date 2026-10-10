#!/usr/bin/env python3
"""Independent saved response arithmetic; no model/gradient/HVP imports or calls."""
import argparse,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TARGET=Path('/data2/neural-shrimp-data/explore/tiny-lm-response/attempt001-target')
PIN='fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5'
EXECUTION='081c5dcc779c3b2f60d0a3102275f0fa7609efbf'
PARENT=Path('/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-002-collection')
PARENT_PIN='2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d'
PROFILE=TARGET.with_name('attempt001-profile')/'run'
PROFILE_PIN='246cbba006bd85ade50a5060587e9ba5f0b56da5f8f35bfb86960273c738a84f'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
FILES=['tools/review_tiny_lm_response.py','tools/launch_tiny_lm_response_review.py','tests/test_tiny_lm_response_review.py','docs/plans/tiny-lm-response-independent.md']
HORIZONS=(1,4,16)
def need(x,m):
 if not x:raise ValueError(m)
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def inventory(p,pin=None,immutable=False):
 m=p/'manifest.json';need(pin is None or sha(m)==pin,'exact manifest '+str(p));v=load(m)['files']
 if isinstance(v,list):v={x['path']:x for x in v}
 need(set(v)=={str(f.relative_to(p))for f in p.rglob('*')if f.is_file()and f!=m},'complete inventory')
 for n,d in v.items():
  f=p/n;need(not Path(n).is_absolute()and '..'not in Path(n).parts and not f.is_symlink()and f.stat().st_size==d['bytes']and sha(f)==d['sha256'],'consumed file identity')
  need(not immutable or not f.stat().st_mode&0o222,'immutable response evidence')
 return dict(root=str(p),manifest_sha256=sha(m),files_including_manifest=len(v)+1,bytes_including_manifest=m.stat().st_size+sum(x['bytes']for x in v.values()))
def norm(v):
 a=np.asarray(v,dtype=float);need(np.isfinite(a).all(),'finite norm input');return math.sqrt(math.fsum(float(x)*float(x)for x in a.ravel()))
def rel(a,b):return norm(a-b)/max(norm(b),1e-12)
def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b))
def same(a,b):
 if type(b)is bool:need(type(a)is bool and a==b,'same boolean gate')
 elif isinstance(b,(int,float)):need(math.isfinite(float(a))and math.isclose(a,b,rel_tol=2e-10,abs_tol=2e-9),'saved scalar matches independent arithmetic')
 elif isinstance(b,list):
  need(len(a)==len(b),'same list length')
  for x,y in zip(a,b):same(x,y)
 else:need(a==b,'same label')
def response_row(direction,h,small,large,frozen,transported,initial,one_step_gain,ce_slopes,readout):
 gain=norm(small);fd=rel(large,small);terr=rel(transported,small);pred=dot(readout,transported);den=max(abs(ce_slopes[1]),1e-8);scalar=one_step_gain**h
 return dict(direction=direction,horizon=h,response_gain=gain,finite_difference_relative_disagreement=fd,transported_vector_error=terr,frozen_vector_error=rel(frozen,small),persistence_vector_error=rel(initial,small),scalar_gain=scalar,scalar_gain_relative_error=abs(scalar-gain)/max(gain,1e-12),resolved=bool(fd<=.01 and terr<=.01),training_ce_slopes=ce_slopes,transported_training_ce_slope=pred,training_ce_slope_disagreement=abs(ce_slopes[0]-ce_slopes[1])/den,training_ce_slope_prediction_error=abs(pred-ce_slopes[1])/den,training_ce_slope_resolved=bool(abs(ce_slopes[0]-ce_slopes[1])/den<=.01 and abs(pred-ce_slopes[1])/den<=.01))
def expected_operations():
 case=['gradient']*17+['hvp']*64
 for _ in range(8):
  for h in range(1,17):
   case.append('gradient')
   if h in HORIZONS:case.append('ce')
 return case*4
def main(out):
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'status':'incomplete','passed':False,'cases':[]};written=[0]
 def save(name,v):
  data=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();need(written[0]+len(data)<=2*1024**2,'2MiB audit raw cap');(out/name).write_bytes(data);written[0]+=len(data)
 def check(*_):
  if _ or time.monotonic()-start>30:raise TimeoutError('30s saved audit')
  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=1024*1024,'1GiB RSS')
 signal.signal(signal.SIGTERM,check);signal.signal(signal.SIGALRM,check);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  own={p:sha(ROOT/p)for p in FILES};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  for p,h in own.items():need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest()==h,'committed audit source')
  save('runtime.json',dict(source_commit=commit,source_sha256=own,pid=os.getpid(),numpy=np.__version__,threads={k:os.environ[k]for k in THREADS},nice=os.getpriority(os.PRIO_PROCESS,0)))
  result['inventories']=[inventory(TARGET,PIN,True),inventory(TARGET/'run',immutable=True),inventory(PARENT,PARENT_PIN),inventory(PROFILE,PROFILE_PIN,True)]
  run=TARGET/'run';producer=load(run/'result.json');runtime=load(run/'runtime.json');launch=load(TARGET/'launcher-result.json')
  need(launch['source_commit']==producer['source_commit']==runtime['source_commit']==EXECUTION and launch['exit_code']==0 and not launch['owned_live_pids'],'completed exact producer')
  need(producer['status']=='response_complete'and producer['dtype']=='float64'and producer['device']=='cpu'and producer['model_parameters']==15776 and producer['protected_outcomes_evaluated']is False,'fixed native interpretation')
  need(runtime['source_identity']==producer['source_identity'],'runtime source identity')
  for p,d in producer['source_identity'].items():
   need(sha(run/'source'/p)==d['sha256']and hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+p],cwd=ROOT)).hexdigest()==d['sha256'],'exact archived and historical native source')
  need(producer['source_identity']['src/neural_shrimp/tiny_lm.py']['sha256']=='27508573066c416322fe697471ada13ebd1c99618a82ae0e1115672a3c94f7e8','native map pin')
  binding=load(run/'input-binding.json');need(binding['parent_root']==str(PARENT)and binding['parent_manifest']==PARENT_PIN,'same parent')
  pr=load(run/'profile-reference.json');profile=load(PROFILE/'result.json');need(pr['root']==str(PROFILE)and pr['manifest']==PROFILE_PIN and profile['mode']=='profile'and profile['source_identity']==producer['source_identity']and all(profile[k]==producer[k]for k in ['torch','numpy','dtype','device']),'exact same-source measured profile')
  cost=0.;need(sorted(x['arm']for x in profile['profile_costs'])==[8,256],'both measured arms')
  for row in profile['profile_costs']:
   need(set(row['times'])=={'gradient','hvp','ce'},'all profile operations')
   for name,count in [('gradient',145),('hvp',64),('ce',24)]:
    ts=row['times'][name];need(len(ts)==2 and all(math.isfinite(t)and t>0 for t in ts),'two measured costs');cost+=count*max(ts)
  forecast=4*cost+20;need(forecast<300 and forecast==profile['forecast_seconds']==pr['forecast_seconds'],'independent profile admission')
  caps=dict(gradient=580,hvp=256,ce=96);need(producer['operation_caps']==producer['operation_attempted']==producer['operation_returned']==caps,'complete actual operation counts')
  events=load(run/'events.json');need([e['operation']for e in events]==expected_operations(),'all actual operation order');counts={k:0 for k in caps}
  for e in events:counts[e['operation']]+=1;need(e['index']==counts[e['operation']]and math.isfinite(e['seconds'])and e['seconds']>0,'operation index and cost')
  need(counts==caps,'actual event counts');need(profile['operation_attempted']==profile['operation_returned']==dict(gradient=4,hvp=4,ce=4),'complete profile counts')
  cases=producer['cases'];need([(c['seed'],c['arm'],c['step'])for c in cases]==[(s,a,600)for s in [28161,28162]for a in [8,256]],'exact four-case selection')
  def arr(folder,name,shape):
   a=np.load(folder/(name+'.npy'),allow_pickle=False);need(a.dtype==np.float64 and a.shape==shape and np.isfinite(a).all(),'finite exact retained array shape');return a
  for case in cases:
   check();seed,arm=case['seed'],case['arm'];folder=run/f'seed-{seed}-arm-{arm}';md=load(folder/'inputs.json');old=PARENT/f'seed-{seed}-arm-{arm}';oldr=load(old/'result.json');chunk=old/'states-000600-000649.npy'
   need(md['seed']==seed and md['arm']==arm and md['step']==600 and md['eta']==oldr['eta']and md['eta']>0 and md['state_chunk']==str(chunk)and md['state_chunk_identity']==dict(bytes=chunk.stat().st_size,sha256=sha(chunk)),'same original map/state/positive rate')
   state=np.load(chunk,allow_pickle=False);need(state.dtype==np.float32 and state.shape==(50,15776),'exact original float32 state');initial=arr(folder,'initial',(15776,));need(np.array_equal(initial,state[0].astype(float))and hashlib.sha256(state[0].tobytes()).hexdigest()==md['state_float32_sha256'],'exact promotion of step600')
   ref=arr(folder,'reference',(17,15776));grad=arr(folder,'reference-gradients',(17,15776));u=arr(folder,'directions',(2,15776));readout=arr(folder,'readout-gradients',(3,15776));need(np.array_equal(ref[0],initial)and np.array_equal(ref[1:],ref[:-1]-md['eta']*grad[:-1]),'reference update arithmetic')
   need(np.array_equal(readout,grad[list(HORIZONS)]-.001*ref[list(HORIZONS)]),'training CE gradient subtraction')
   rnd=np.random.default_rng(28166).standard_normal(15776)
   for i,v in enumerate([grad[0],rnd]):need(np.max(abs(u[i]-v/norm(v)))<2e-15,'fixed normalized gradient/random directions')
   eps=[v*max(1,norm(initial))for v in [1e-4,5e-5]];same(eps,case['epsilon_values']);rows=[]
   for i,label in enumerate(['gradient','random']):
    f=arr(folder,f'direction-{i}-frozen',(17,15776));t=arr(folder,f'direction-{i}-transported',(17,15776));need(np.array_equal(f[0],u[i])and np.array_equal(t[0],u[i])and np.array_equal(f[1],t[1]),'same initial tangent and one-step action')
    pairs=[];ces=[]
    for k,e in enumerate(case['epsilon_values']):
     plus=arr(folder,f'direction-{i}-epsilon-{k}-plus',(17,15776));minus=arr(folder,f'direction-{i}-epsilon-{k}-minus',(17,15776));need(np.array_equal(plus[0],initial+e*u[i])and np.array_equal(minus[0],initial-e*u[i]),'exact perturbation starts')
     cp=arr(folder,f'direction-{i}-epsilon-{k}-plus-ce',(3,2));cm=arr(folder,f'direction-{i}-epsilon-{k}-minus-ce',(3,2));need(np.array_equal(cp[:,0],HORIZONS)and np.array_equal(cm[:,0],HORIZONS),'exact retained CE horizons')
     for lab,cs in [('ce_plus',cp),('ce_minus',cm)]:need(case['branch_training_ce'][f'{i}:{k}'][lab]=={str(h):float(cs[j,1])for j,h in enumerate(HORIZONS)},'all retained readouts match case')
     pairs.append((plus-minus)/(2*e));ces.append((cp[:,1]-cm[:,1])/(2*e))
    for j,h in enumerate(HORIZONS):rows.append(response_row(label,h,pairs[1][h],pairs[0][h],f[h],t[h],u[i],norm(f[1]),[float(c[j])for c in ces],readout[j]))
   one={r['direction']:r['resolved']for r in rows if r['horizon']==1}
   need(len(case['rows'])==6,'six exact rows per case')
   for r,saved in zip(rows,case['rows']):
    r['mechanism_qualified']=bool(r['resolved']and one[r['direction']]);need(set(r)==set(saved),'complete comparator fields')
    for key,value in r.items():same(saved[key],value)
   need(load(folder/'outcome.json')==case,'case result binding');result['cases'].append(dict(seed=seed,arm=arm,rows=rows));save(f'case-{seed}-{arm}.json',result['cases'][-1])
  selected=[r for c in result['cases']if c['arm']==256 for r in c['rows']if r['direction']=='gradient'and r['horizon']==16];gate=all(r['mechanism_qualified']and r['frozen_vector_error']>.2 for r in selected)
  need(len(selected)==2 and gate==producer['narrow_frozen_curvature_inadequacy'],'independent fixed narrow case gate');check();result.update(status='independent_saved_arithmetic_passed',passed=True,narrow_frozen_curvature_inadequacy=gate,operation_counts=counts,forecast_seconds=forecast,rows_recomputed=24)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent saved norms, dot products, all central responses/comparators/CE slopes/gates, reference update arithmetic and exact source/input/count bindings. Shared acquired native gradients/HVP/CE values are not recomputed. No native calls, causal forecast, utility, chaos or shrimp claim.');save('result.json',result);save('manifest.json',{'files':{p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.iterdir()if p.is_file()}})
 return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
