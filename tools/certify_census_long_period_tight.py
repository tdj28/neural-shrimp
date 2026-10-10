"""One fixed640bit/radius2^-288 retry from the exact retained refined center."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,resource,signal,subprocess,sys,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from flint import ctx,arb_mat
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import certify_c1 as C
from neural_shrimp import rigorous_orbits as K
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/census-period58/attempt001-target')
PIN='c0f7488ef7836f9c3f631e1636b9f7ca28a85191e7f1834d3060451fb9f45fc8';PRODUCER='3488be385d98b6c38a2280adca7e509ea8ff72da'
BASE=Path('/data2/neural-shrimp-data/explore/certificates/census-period58');SETTINGS=dict(precision=640,radius_exponent=-288,newton_steps=0,max_power=4096)
SOURCES=C.SOURCES+['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py','tools/certify_census_long_period_tight.py','tools/launch_census_long_period_tight.py','tests/test_census_long_period_tight.py','docs/plans/census-long-period-tight.md']
CAP=512*1024**2

def need(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(p.read_bytes())
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def inventory(root,pin):
 need(entry(root/'manifest.json')['sha256']==pin,'exact manifest');rows=read(root/'manifest.json')['files'];need(set(rows)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete inventory')
 for name,item in rows.items():
  p=root/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and entry(p)==item and not p.stat().st_mode&0o222,'immutable source file')

def exact_center(rows):
 need(len(rows)==72,'full72 refined center');out=[]
 for row in rows:
  need(row.get('finite',True)and row['rad'][0]==0 and all(type(x)is int for x in row['mid'])and len(row['mid'])==2,'exact finite dyadic refined center')
  need(abs(row['mid'][1])<=100000,'bounded exponent');out.append(str(F(row['mid'][0])*F(2)**row['mid'][1]))
 return out

def forecast(times):
 need(len(times)==3 and all(np.isfinite(v)and v>0 for v in times),'three finite native costs');return 2*(1+2)*58*max(times)+30

def propose(center,end,M):
 n=len(center);need(len(end)==n and M.nrows()==M.ncols()==n,'full proposal dimensions');residual=arb_mat([[end[i]-center[i]]for i in range(n)]);correction=(M-K.eye(n)).solve(residual);proposal=[(center[i]-correction[i,0]).mid()for i in range(n)];need(all(v.is_finite()for v in proposal),'finite proposed center');return residual,correction,proposal

def proximity(rows,reference):
 need(len(rows)==len(reference)==72,'all72 proximity coordinates');seed=[F(v)for v in reference];scale=max(F(1),*(abs(v)for v in seed));distance=F(0)
 for row,x in zip(rows,seed):
  need(row.get('finite',True),'finite proximity ball');m=F(row['mid'][0])*F(2)**row['mid'][1];r=F(row['rad'][0])*F(2)**row['rad'][1];need(r>=0,'nonnegative radius');distance=max(distance,(abs(m-x)+r)/scale)
 need(distance<=F(1,10**7),'saved representative proximity');return dict(maximum_normalized_distance=str(distance),threshold='1/10000000',scope='Exact whole-box distance only, not branch uniqueness or a statement that floating seed is an exact orbit.')

def prepare():
 inventory(PARENT,PIN);la=read(PARENT/'launcher-result.json');rt=read(PARENT/'run/runtime-start.json');done=read(PARENT/'run/result.json');need(la['exit_code']==2 and la['owned_live_pids']==[]and rt['source_commit']==la['source_commit']==PRODUCER and not done['passed'],'retain exact failed owned parent')
 for name,digest in rt['source_sha256'].items():need(entry(PARENT/'run/source'/name)['sha256']==digest==hashlib.sha256(subprocess.check_output(['git','show',PRODUCER+':'+name],cwd=ROOT)).hexdigest(),'historical source')
 for name in C.SOURCES+['src/neural_shrimp/models.py','src/neural_shrimp/trainability.py','src/neural_shrimp/gdmaps.py']:need(entry(ROOT/name)['sha256']==rt['source_sha256'][name],'unchanged native kernel/generator')
 old=read(PARENT/'run/inputs.json');proof=read(PARENT/'run/certificate/certificate.json')['result'];need(not proof['passed']and proof['settings']==dict(precision=512,radius_exponent=-160,newton_steps=5,max_power=4096),'same failed settings')
 p=old['provenance'];need(p['row_index']==86 and p['map_id']=='S1R-28002-zoom'and p['component']==262,'fixed consumed representative')
 cp=Path(p['certificate_file']);plan=ROOT/'experiments/trainability/EXP-028-plan.json';need(entry(cp)['sha256']==p['certificate_sha256']and entry(plan)['sha256']==p['plan_sha256'],'frozen row/data plan identities')
 fresh=C.prepare(cp,plan,86,p['selection']);need(all(fresh[k]==old[k]for k in ['schema','network','x','y','lr_hidden','lr_output','period','approximation']),'independently regenerated exact physical map and original seed')
 need(old['period']==58 and all(F(old[k])>0 for k in ['lr_hidden','lr_output']),'native58 and positive rates')
 inp=json.loads(json.dumps(old));inp['approximation']=exact_center(proof['center']);inp['provenance']['refined_seed']={'parent_root':str(PARENT),'outer_manifest':PIN,'source':PRODUCER,'certificate':entry(PARENT/'run/certificate/certificate.json'),'field':'result.center','prior_failed':True}
 orbit=np.asarray(read(cp)[86]['certificate']['orbit']);need(orbit.shape==(58,72)and np.isfinite(orbit).all(),'exact full saved phase proposals')
 return inp,orbit,dict(parent_root=str(PARENT),outer_manifest=PIN,source=PRODUCER,original_inputs=entry(PARENT/'run/inputs.json'),refined_center_source=entry(PARENT/'run/certificate/certificate.json'),current_exact_input_sha256=K.sha256(K.canonical_bytes(inp)))

def main(args):
 need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP));out=BASE/f'attempt002-{args.mode}'/'run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limit=60 if args.mode=='profile'else 600;used=0
 def stop(*_):raise TimeoutError('fixed bounded precision-radius unit')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,limit)
 def save(name,v):
  nonlocal used
  b=K.canonical_bytes(v);need(used+len(b)<CAP,'512MiB raw cap');used+=len(b);p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 identity={n:entry(ROOT/n)for n in SOURCES};source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,sources=identity,pid=os.getpid(),mode=args.mode,settings=SETTINGS,explicit_newton_proposals=1,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0),numpy=np.__version__,python=sys.version))
 for n in SOURCES:need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'committed dependency');p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
 result={'passed':False,'mode':args.mode,'completed_native_map_calls':0,'scope':'Single exact-real72D period58 representative; not whole census or geometry.'}
 try:
  inp,orbit,binding=prepare();save('seed-inputs.json',inp);save('input-binding.json',binding)
  if args.mode=='profile':
   times=[]
   with ctx.workprec(640):
    model=K.BallTanhMap(inp)
    for phase in [0,19,38]:
     t=time.monotonic();v,J=model.step([K.exact(K.dyadic(x))for x in orbit[phase]]);times.append(time.monotonic()-t);result['completed_native_map_calls']+=1;need(all(a.is_finite()for a in v)and all(J[i,k].is_finite()for i in range(72)for k in range(72)),'finite full profile');save(f'phase-{phase}.json',dict(value=K.vector_json(v),jacobian=K.matrix_json(J)))
   projection=forecast(times);result.update(passed=projection<600,status='profile_complete',times=times,forecast_seconds=projection,target_proof_started=False)
  else:
   pr=BASE/'attempt002-profile';inventory(pr,args.profile_outer);inventory(pr/'run',args.profile_worker);prt=read(pr/'run/runtime.json');pd=read(pr/'run/outcome.json');pla=read(pr/'launcher-result.json')
   need(pla['exit_code']==0 and pla['owned_live_pids']==[]and prt['mode']=='profile'and prt['sources']==identity and prt['settings']==SETTINGS and prt['numpy']==np.__version__ and prt['python']==sys.version,'same-source owned measured profile')
   need(pd['passed']and pd['completed_native_map_calls']==3 and pd['forecast_seconds']==forecast(pd['times'])<600 and(pr/'run/seed-inputs.json').read_bytes()==K.canonical_bytes(inp),'exact measured input/cost admission');save('profile-binding.json',dict(outer=args.profile_outer,worker=args.profile_worker,source=prt['source_commit'],forecast_seconds=pd['forecast_seconds']))
   with ctx.workprec(640):
    model=K.BallTanhMap(inp);center=[K.exact(v)for v in inp['approximation']];end,M,orbit,Js=model.iterate(center,58);result['completed_native_map_calls']+=58;residual=arb_mat([[end[i]-center[i]]for i in range(72)]);save('newton-before.json',dict(center=K.vector_json(center),residual=K.matrix_json(residual),monodromy=K.matrix_json(M),orbit= [K.vector_json(row)for row in orbit]))
    _,correction,proposal=propose(center,end,M);near=proximity(K.vector_json(proposal),inp['approximation']);save('newton-proposal.json',dict(correction=K.matrix_json(correction),proposal=K.vector_json(proposal),proximity=near))
    actual=json.loads(json.dumps(inp));actual['approximation']=exact_center(K.vector_json(proposal));actual['provenance']['explicit_proposal']={'precision':640,'iterations':1,'record':'newton-proposal.json','record_identity':entry(out/'newton-proposal.json')}
   save('proof-inputs.json',actual);code=C.run(actual,out/'certificate',SETTINGS);result['completed_native_map_calls']+=116;receipt=read(out/'certificate/receipt.json');proof=read(out/'certificate/certificate.json')['result'];near=proximity(proof['box'],read(PARENT/'run/inputs.json')['approximation']);save('representative-proximity.json',near);result.update(passed=code==0,status='certificate_complete',certificate_receipt=receipt,selected_representative_proximity=near)
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),status='failed_or_interrupted')
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result['passed']=result['passed']and result['elapsed_seconds']<limit and result['peak_rss_kib']<2*1024**2;save('outcome.json',result);files={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()};need(sum(v['bytes']for v in files.values())<CAP,'final cumulative raw cap');save('manifest.json',dict(files=files))
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile-outer');p.add_argument('--profile-worker');raise SystemExit(main(p.parse_args()))
