"""Independent saved nested Krylov audit. No producer recurrence/native imports."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,math,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_cifar_reorthogonalized as A
M=A.M;need=M.need;ROOT=Path(__file__).resolve().parents[2]
OUT=Path('/data2/neural-shrimp-data/audits/cifar-nested512-independent/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/nested512001')
EXEC='545c1d422af09af8829580db91e9dbeceecccdda';PINS=['685f103da15a2fe5c275908119fce16d9b190b96e9985577dbc6b671c0ac9229','95ea3a44398d4df7f0bead0c633898cbe6d88ad021536015a210823bd700b989']
PRIOR=Path('/data2/neural-shrimp-data/audits/cifar-reorthogonalized-independent/attempt001')
PRIOR_SOURCE='1550c628882e7474425513ed4a486e313b5c74f6';PRIOR_PINS=['d3b140937df127b89765d7901b6926bb4bf2094c5e7edb195cbda3c593850f84','d72991b66572a43ff7bb0e8bc560ad3eb6b8cb3a904aaa88b860cbf95daea7b1']
FILES=['tools/reviews/review_cifar_stationary_merit.py','tools/reviews/review_cifar_stationary_continuation.py','tools/reviews/review_cifar_reorthogonalized.py','tools/reviews/review_cifar_nested512.py','tools/reviews/launch_cifar_nested512_review.py','tests/reviews/test_cifar_nested512_review.py','docs/plans/cifar-nested512-independent.md']


def pair_project(V,w):
 """Separate implementation of two classical projection passes."""
 q=w.copy();coeff=[]
 for _ in range(2):
  a=np.einsum('ij,i->j',V,q,optimize=True);q=q-np.einsum('ij,j->i',V,a,optimize=True);coeff.append(a)
 return q,coeff


def recurrence(V,W,g,rows,tick=lambda:None):
 need(V.shape==W.shape and V.ndim==2 and len(rows)==V.shape[1]-1,'complete recurrence geometry')
 need(M.norm(V[:,0]+g/M.norm(g))<1e-12,'initial negative normalized gradient')
 answers=[]
 for j,row in enumerate(rows):
  q,cs=pair_project(V[:,:j+1],W[:,j]);nq=M.norm(q);nw=M.norm(W[:,j]);need(nq>32*np.finfo(float).eps*max(1,nw),'no hidden breakdown')
  old=np.asarray(row['coefficients'],dtype=np.float64);need(old.shape==(2,j+1),'both full coefficient rows')
  err=float(np.max(np.abs(np.asarray(cs)-old)))/max(1,nw);need(err<1e-11,'independent projection coefficients')
  ne=abs(nq-float(row['residual_norm']))/max(nq,1e-300);ve=M.norm(q/nq-V[:,j+1]);need(ne<1e-9 and ve<1e-9,'full next basis recurrence')
  if 'raw_action_norm'in row:need(abs(nw-row['raw_action_norm'])<1e-11*max(1,nw),'raw action norm')
  alignment=float(np.max(np.abs(V[:,:j+1].T@V[:,j+1])));need(alignment<=1e-10,'actual new-vector alignment')
  answers.append(dict(column=j+1,coefficient_error=err,residual_relative_error=ne,next_vector_error=ve,alignment=alignment));tick()
 return answers


def events_bound(events,manifest):
 counts=0;calls={};rows=[];pos=0
 def item(v):
  name=v['file'];need(Path(name).name==name and manifest[name]==v['identity'],'native file binding');return name
 while pos<len(events):
  e=events[pos]
  if e['kind']=='nested_reorthogonalization':
   need(counts==len(rows)+1 and e['from_column']==255+len(rows)and e['status']=='next_vector_qualified','nested row immediately before its action');rows.append(e);pos+=1;continue
  b=events[pos:pos+4];counts+=1
  need(len(b)==4 and [v['kind']for v in b]==['native_input','operation_started','operation_returned','native_output']and all(v['operation']=='hvp'and v['index']==counts for v in b),'exact new ordered HVP quartet')
  need(counts==1 or counts==len(rows)+1 or counts==258 and len(rows)==256,'one action for each new vector plus identity/final')
  need(math.isfinite(b[2]['seconds'])and b[2]['seconds']>=0,'finite operation duration');calls[counts]=dict(state=item(e['state']),direction=item(e['direction']),result=item(b[3]['result']));pos+=4
 need(counts==258 and len(rows)==256,'exact258 new actions/256 new vectors');return calls,rows


def comparison(g,oldhp,actual,pred,orth,complete):
 ng=M.norm(g);need(ng>0,'positive native gradient');base=M.norm(g+oldhp)/ng;force=M.norm(g+actual)/ng;err=M.norm(actual-pred)/max(ng,M.norm(actual),1e-12)
 qualified=bool(complete and orth<=1e-8 and err<=1e-8)
 return dict(qualified=qualified,baseline_forcing=base,true_forcing=force,forcing_ratio=force/base,physical_action_relative_error=err,material_improvement=bool(qualified and force<=.8*base),strict_forcing_pass=bool(qualified and force<=.5))


def owner(root,execution):
 q=M.read(root/'launcher-result.json');need(q['source_commit']==execution and q['exit_code']==0 and q['failure']is None and not q['owned_live_pids']and q['worker_pid_absent'],'complete owned input')
 for k in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(q[k])).exists(),'actual original owner absent')
 return q


def bind(tick):
 inv=M.inventory(PARENT,PINS[0],tick);wi=M.inventory(PARENT/'run',PINS[1],tick);M.inventory(A.PARENT,A.PINS[0],tick);oi=M.inventory(A.PARENT/'run',A.PINS[1],tick)
 M.inventory(PRIOR,PRIOR_PINS[0],tick);M.inventory(PRIOR/'run',PRIOR_PINS[1],tick)
 r=M.read(PARENT/'run/result.json');old=M.read(A.PARENT/'run/result.json');need(r['source_commit']==EXEC,'fixed nested execution');execution=r['source_commit'];M.source(PARENT,r,execution);M.source(A.PARENT,old,A.EXEC)
 o=owner(PARENT,execution);owner(A.PARENT,A.EXEC);owner(PRIOR,PRIOR_SOURCE)
 need(o['elapsed_seconds']<315 and r['elapsed_seconds']<300 and r['peak_rss_kib']<4*1024**2,'original owned resource bounds')
 prior=M.read(PRIOR/'run/outcome.json');ib=M.read(PRIOR/'run/input-binding.json');runtime=M.read(PRIOR/'run/runtime.json')
 need(runtime['source_commit']==PRIOR_SOURCE and prior['passed']and prior['native_calls']==0 and prior['numerical_action_rank']==256 and prior['comparison']['qualified'],'independent prefix authority')
 need(ib['producer']==str(A.PARENT)and ib['producer_source']==A.EXEC and ib['producer_pins']==A.PINS and ib['all_at_historical_state018'],'same exact independently audited prefix')
 for n,v in runtime['files'].items():
  need(M.entry(PRIOR/'run/source'/n)==v,'independent original source bytes')
  need(M.entry(PRIOR/'run/source'/n)['sha256']==__import__('hashlib').sha256(subprocess.check_output(['git','show',PRIOR_SOURCE+':'+n],cwd=ROOT)).hexdigest(),'exact committed prior source')
  if n in FILES[:3]:need(M.entry(ROOT/n)==v,'exact reused independent helper')
 for n,h in old['source_sha256'].items():need(r['source_sha256'][n]==h,'unchanged prefix/native source '+n)
 for k in('data_identity','eta','lam','model'):need(r['input_identity'][k]==old['input_identity'][k],'same objective '+k)
 for k in old['runtime']:
  if k!='pid':need(r['runtime'][k]==old['runtime'][k],'same runtime '+k)
 need(r['status']=='nested512_comparison_complete'and not r['direction_applied']and not r['accepted_orbit']and not r['rigorous_certificate'],'unchanged comparison scope')
 need(r['operation_caps']==r['operation_attempted']==r['operation_returned']==dict(action=0,j2vp=0,hvp=258,direct_gradient=0),'inclusive new operation count')
 b=M.read(PARENT/'run/nested-binding.json');need(b['prefix']==dict(root=str(A.PARENT),source=A.EXEC,outer=A.PINS[0],worker=A.PINS[1])and b['independent']==dict(root=str(PRIOR),source=PRIOR_SOURCE,outer=PRIOR_PINS[0],worker=PRIOR_PINS[1])and b['columns']==dict(old=256,new=256,total=512)and b['old_state']=='state-018.npy'and b['no_direction_applied'],'exact nested genealogy')
 events=[json.loads(x)for x in(PARENT/'run/events.jsonl').read_text().splitlines()];calls,rows=events_bound(events,wi)
 oe=[json.loads(x)for x in(A.PARENT/'run/events.jsonl').read_text().splitlines()];counts,ocalls,orows=A.events_bound(oe,oi);need(counts==dict(hvp=257,direct_gradient=1),'complete prefix operations')
 need(all(v['state']=='state-000.npy'for v in list(calls.values())+list(ocalls.values())),'all actions at same literal state')
 return r,old,prior,calls,rows,ocalls,orows,execution


def audit(binding,tick,save):
 r,old,prior,calls,rows,ocalls,orows,execution=binding;folder=PARENT/'run';load=lambda n:M.vector(folder/n);oldload=lambda n:M.vector(A.PARENT/'run'/n)
 need(M.same(load('state-000.npy'),oldload('state-000.npy')),'same historical018 state')
 g=oldload(ocalls['direct_gradient',1]['result']);need(M.same(g,load('historical-gradient.npy')),'literal old physical gradient')
 need(M.same(load(calls[1]['direction']),oldload(ocalls['hvp',1]['direction']))and M.same(load(calls[1]['result']),oldload(ocalls['hvp',1]['result'])),'literal first-action identity replay')
 V=np.empty((98666,512),order='F');W=np.empty_like(V)
 for j in range(512):
  c=ocalls['hvp',j+1]if j<256 else calls[j-254];f=oldload if j<256 else load;V[:,j]=f(c['direction']);W[:,j]=f(c['result']);tick()
 need(orows[-1]['coefficients']==rows[0]['coefficients']and orows[-1]['residual_norm']==rows[0]['residual_norm'],'literal terminal prefix continuation')
 replay=M.read(folder/'prefix-terminal-replay.json');need(replay==dict(coefficients=orows[-1]['coefficients'],residual_norm=orows[-1]['residual_norm'],passed=True),'saved terminal identity')
 rec=recurrence(V,W,g,orows[:-1]+rows,tick);save('recurrence.json',dict(rows=rec,all512_columns=True))
 report=M.read(folder/'basis-report.json');need(report==r['basis']and report['prefix_columns']==256 and report['total_columns']==512 and report['new_basis_actions']==256 and report['status']=='maximum_columns'and report['completed_fixed_extension'],'literal complete basis')
 coef=np.asarray(report['coefficients']);p,hp,gram,orth,rank,singular=A.assembly(V,W,coef);tick()
 own,_,ownrank,sv=np.linalg.lstsq(W,-g,rcond=1e-12);tick();need(ownrank==rank==report['rank']==512,'full numerical action rank')
 ce=float(np.linalg.norm(own-coef))/max(1,float(np.linalg.norm(own)));need(ce<1e-9,'independently resolved least-squares coefficients')
 stored=np.load(folder/'final-gram.npy',allow_pickle=False);need(stored.dtype==np.float64 and stored.shape==(512,512)and np.max(abs(stored-gram))<1e-13,'full512 saved Gram')
 need(np.max(abs(singular-np.asarray(report['singular_values'])))/singular[0]<1e-11,'full singular values')
 direction=load('candidate-direction.npy');prediction=load('predicted-action.npy');actual=load(calls[258]['result']);need(M.same(direction,load(calls[258]['direction'])),'fresh final physical action direction')
 pe=M.norm(p-direction)/max(1e-12,M.norm(direction));he=M.norm(hp-prediction)/max(1e-12,M.norm(prediction));need(pe<1e-11 and he<1e-11,'independent assembly of full512 correction/action')
 answer=comparison(g,oldload(ocalls['hvp',257]['result']),actual,prediction,orth,True);M.compare(answer,r['comparison']);M.close(answer['baseline_forcing'],prior['comparison']['true_forcing'],'independently qualified256 baseline')
 need(answer['qualified']and answer['material_improvement']and answer['strict_forcing_pass'],'preserved fixed20percent and0.5 gates')
 result=dict(comparison=answer,gram_inf_error=orth,numerical_action_rank=rank,least_squares_coefficient_relative_error=ce,direction_assembly_error=pe,predicted_action_assembly_error=he,recurrence_rows=len(rec))
 save('input-binding.json',dict(producer=str(PARENT),source=execution,pins=PINS,prefix=str(A.PARENT),prefix_pins=A.PINS,independent_prefix=str(PRIOR),independent_prefix_pins=PRIOR_PINS,counts=dict(old_hvp=257,old_gradient=1,new_hvp=258,new_gradient=0),all_at_historical_state018=True));save('arithmetic.json',result)
 return dict(passed=True,native_calls=0,**result,scope='Independent saved binary64 recurrence, full Gram/rank/least squares and fsum physical forcing audit. Shares native autograd outputs and NumPy/BLAS/SVD libraries. Nested larger budget, not equal action count. No step, stationary point, orbit or formal interval theorem.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'300s4GiB budget')
 def stop(*_):raise TimeoutError('owned audit deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<(8 if closing else 6)*1024**2,'8MiB2MiB reserve');q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=execution,python=sys.version,numpy=np.__version__,pid=os.getpid(),files={n:M.entry(ROOT/n)for n in FILES},threads={k:os.environ[k]for k in THREADS}))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT),'literal committed own source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);used+=len(b)
  binding=bind(tick) # inventory/metadata only; scientific npy arrays unopened
  rng=np.random.default_rng(52513);V=np.asfortranarray(rng.standard_normal((98666,512))/np.sqrt(98666));W=np.asfortranarray(V*np.linspace(.5,2,98666)[:,None]);g=rng.standard_normal(98666);samples=[]
  for _ in range(3):
   t=time.monotonic();q,cs=pair_project(V,g);M.norm(q);M.norm(g);samples.append(time.monotonic()-t);tick()
  t=time.monotonic();answer=A.assembly(V,W,np.ones(512)/512);coef=np.linalg.lstsq(W,-g,rcond=1e-12);algebra=time.monotonic()-t;tick();forecast=time.monotonic()-start+2*(511*max(samples)+algebra)+30;passed=forecast<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2
  save('profile.json',dict(passed=passed,rows=98666,columns=512,max_projection_seconds=max(samples),algebra_seconds=algebra,forecast_seconds=forecast,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,target_arrays_unopened=True));need(passed,'full-size measured admission');del V,W,g,q,cs,answer,coef;gc.collect()
  result=audit(binding,tick,save)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):M.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
