"""Fixed two- and four-period block-PCG comparison with immutable saved baselines."""
import argparse,hashlib,json,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.cifar_restart_diagnostic import Store
from neural_shrimp.response_multiperiod_blocks import probe
RAW=Path('/data2/neural-shrimp-data/explore/clark-response/attempt-002-panel')
PIN='177b909a2274272197dfca42a13ee785924ea31e71cc14ed54f13133081dd364'
EXEC='1b08ac465151a840f748fe2ae2a4936a7ffe0a70'
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/response_spectrum.py','src/neural_shrimp/response_multiperiod_blocks.py','src/neural_shrimp/cifar_restart_diagnostic.py','tools/explore_response_multiperiod_blocks.py','tools/launch_response_multiperiod_blocks.py','tests/test_response_multiperiod_blocks.py','docs/plans/response-multiperiod-blocks.md']
BASE=Path('/data2/neural-shrimp-data/explore/clark-response/time-blocks-001')
BASEPIN='9374e38f7d30879df6a76361a9bf683240106b271e24f1e3427f6aeeb652e0e4'
BASEEXEC='47b6022811bc53b0479dbf8de943515f7de4f380'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_inputs():
 require(sha(RAW/'manifest.json')==PIN,'outer pin');m=json.loads((RAW/'manifest.json').read_text())['files'];require(set(m)=={str(p.relative_to(RAW))for p in RAW.rglob('*')if p.is_file()and p!=RAW/'manifest.json'},'complete inventory')
 for n,d in m.items():
  p=RAW/n;require(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and sha(p)==d['sha256']and p.stat().st_size==d['bytes'],'input bytes')
 start=json.loads((RAW/'run/start.json').read_text());require(start['source_commit']==EXEC,'historical source')
 for n,d in start['sources'].items():require(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+n],cwd=ROOT)).hexdigest()==d['sha256'],'historical executable '+n)
 return m

def verify_baselines():
 require(sha(BASE/'manifest.json')==BASEPIN,'baseline manifest pin');m=json.loads((BASE/'manifest.json').read_text())['files'];require(set(m)=={str(p.relative_to(BASE))for p in BASE.rglob('*')if p.is_file()and p!=BASE/'manifest.json'},'baseline exact inventory')
 for n,d in m.items():
  p=BASE/n;require(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and sha(p)==d['sha256']and p.stat().st_size==d['bytes'],'baseline bytes')
 rt=json.loads((BASE/'run/runtime-start.json').read_text());require(rt['source_commit']==BASEEXEC,'baseline execution')
 for n,d in rt['sources'].items():require(hashlib.sha256(subprocess.check_output(['git','show',BASEEXEC+':'+n],cwd=ROOT)).hexdigest()==d['sha256'],'baseline source')
 result=json.loads((BASE/'run/result.json').read_text());require(result['status']=='completed'and len(result['rows'])==4,'complete baseline unit')
 return result['rows']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10');store=Store(args.out,limit=383*1024**2);start=time.monotonic();stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));seq=[0]
 def tick():
  if stopped[0]or time.monotonic()-start>60:raise TimeoutError('60s/termination')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB cooperative worker')
 def emitter(slot):
  def emit(kind,value):
   seq[0]+=1;tag=f'{seq[0]:05d}-{slot}-{kind}';record=dict(kind=kind,slot=slot,sequence=seq[0],arrays={})
   for name,v in value.items():
    if isinstance(v,np.ndarray):record['arrays'][name]=dict(file=tag+'-'+name+'.npy',**store.array(tag+'-'+name,v))
    else:record[name]=v
   store.json(tag+'.json',record)
  return emit
 result=dict(status='incomplete',rows=[],scope='saved13D midpoint periodic operators, e0column only; no new derivative, cycle, count, training utility or large-model speedup')
 rt=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={n:dict(sha256=sha(ROOT/n),bytes=(ROOT/n).stat().st_size)for n in SOURCES},pid=os.getpid(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k]for k in THREADS},started_unix=time.time());store.json('runtime-start.json',rt)
 try:
  n=13;p=18;m=np.diag(np.exp(np.linspace(-.6,.3,n)))+.01*np.triu(np.ones((n,n)),1)+.003*np.tril(np.ones((n,n)),-1);maps=np.array([m+((t%3)-1)*.0001*np.triu(np.ones((n,n)),1)for t in range(p)]);store.array('synthetic-maps',maps);profile=[]
  for pre in (2,4):
   tick();t=time.monotonic();r=probe(maps,pre,emitter('profile-'+str(pre)),tick);r['readout_column']=list(r['readout_column']);profile.append(dict(wall_seconds=time.monotonic()-t,result=r));store.json('profile-'+str(pre)+'.json',profile[-1])
  projected=4*sum(x['wall_seconds']for x in profile)+5;projected_raw=3*store.used+32*1024**2;result['profile']=dict(projected_seconds=projected,projected_raw_bytes=projected_raw,measured_profile_bytes=store.used,scope='two slots times factor2 on full synthetic two-method cost, gates resources only');store.json('profile-gate.json',result['profile']);require(np.isfinite(projected)and projected<45 and projected_raw<383*1024**2,'resource gate45s/383MiB');verify_inputs();baselines=verify_baselines();store.json('saved-baselines.json',dict(root=str(BASE),manifest=BASEPIN,execution=BASEEXEC,rows=baselines));result['saved_baselines']=baselines;store.json('input-binding.json',dict(root=str(RAW),manifest=PIN,execution=EXEC))
  for label,p in [('p10-row35',10),('p18-row111',18)]:
   tick();identity=json.loads((RAW/'run'/f'{label}-inputs.json').read_text());maps=np.asarray(identity['jacobian_midpoints'],dtype=np.float64);require(maps.shape==(p,13,13)and hashlib.sha256(maps.tobytes()).hexdigest()==identity['midpoints_bytes_sha256'],'exact midpoint maps');direct=json.loads((RAW/'run'/f'{label}-matrixfree-direct.json').read_text());require(direct['solve_qualified']and direct['steps']==p*16 and direct['shift']==-.05 and direct['eta']==.01 and direct['boundary']=='free','matched direct reference');oracle=np.asarray(direct['response_block'])[:,0];store.array(label+'-maps',maps);store.array(label+'-saved-direct-column',oracle)
   old=json.loads((RAW/'run'/f'{label}-matrixfree.json').read_text());store.json(label+'-old-lsmr.json',old)
   for pre in (2,4):
    require(np.array_equal(maps,np.load(BASE/'run'/f'{label}-maps.npy'))and np.array_equal(oracle,np.load(BASE/'run'/f'{label}-saved-direct-column.npy')),'same baseline maps/reference');tag=label+'-'+str(pre);store.json(tag+'-start.json',dict(label=label,block_periods=pre))
    try:
     r=probe(maps,pre,emitter(tag),tick);r['readout_error']=float(np.linalg.norm(np.asarray(r['readout_column'])-oracle));r['endpoint_pass']=bool(r['solve_qualified']and r['readout_error']<=1e-6)
    except (ValueError,np.linalg.LinAlgError)as e:r=dict(failure=type(e).__name__+': '+str(e),endpoint_pass=False)
    r.update(label=label,block_periods=pre);store.json(tag+'-result.json',r);result['rows'].append(r);tick()
  useful=[]
  for label in ('p10-row35','p18-row111'):
   b=next(r for r in baselines if r['label']==label and not r['preconditioned'])
   for c in [r for r in result['rows']if r['label']==label]:
    passed=bool(c['endpoint_pass']and 'counts'in b and c['counts']['JVP']+c['counts']['VJP']<=.5*(b['counts']['JVP']+b['counts']['VJP']));useful.append(dict(label=label,block_periods=c['block_periods'],endpoint_and_half_callbacks=passed))
  result.update(status='completed',utility_screen=useful,fixed_method_screens={str(k):all(x['endpoint_and_half_callbacks']for x in useful if x['block_periods']==k)for k in (2,4)},all_four_pass=all(x['endpoint_and_half_callbacks']for x in useful));tick()
 except BaseException as e:result.update(status='failed_or_interrupted',failure=type(e).__name__+': '+str(e))
 finally:result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,events=seq[0]);store.finish(result)
 print(json.dumps(result));return 0 if result['status']=='completed'else 2
if __name__=='__main__':raise SystemExit(main())
