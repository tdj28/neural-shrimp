"""One fixed operator-only4p flexible-GCR comparison, all phase actions charged."""
import argparse,importlib.util,json,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('prior',ROOT/'tools/explore_response_multiperiod_blocks.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
from neural_shrimp.response_operator_blocks import probe
SOURCES=B.SOURCES+['src/neural_shrimp/response_operator_blocks.py','tools/explore_response_operator_blocks.py','tools/launch_response_operator_blocks.py','tests/test_response_operator_blocks.py','docs/plans/response-operator-blocks.md']
CEILINGS={'p10-row35':96480,'p18-row111':173664}

def profile_projection(seconds,stored_bytes,counts):
 actions=counts['JVP']+counts['VJP']
 B.require(type(actions)is int and 0<actions<=173664 and np.isfinite(seconds)and seconds>0,'positive complete profile cost')
 multiplier=max(3.,1.+sum(CEILINGS.values())/actions)
 return {'projected_seconds':multiplier*seconds+5,'projected_bytes':int(np.ceil(multiplier*stored_bytes))+16*1024**2,'profile_seconds':seconds,'profile_bytes':stored_bytes,'profile_actions':actions,'multiplier':multiplier,'target_action_ceiling_sum':sum(CEILINGS.values()),'scope':'actual profile callback-count normalized full ceilings plus profile; minimum3factor and explicit fixed overhead; resource gate only'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();B.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');B.require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in B.THREADS),'CPU1nice10');store=B.Store(args.out,limit=255*1024**2);started=time.monotonic();stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));seq=[0]
 def tick():
  if stopped[0]or time.monotonic()-started>60:raise TimeoutError('60s/termination')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
 def emitter(slot):return B.durable_emitter(store,seq,slot)
 rt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:{'sha256':B.sha(ROOT/n),'bytes':(ROOT/n).stat().st_size}for n in SOURCES},'pid':os.getpid(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in B.THREADS},'internal_seconds':60,'science_cap':256*1024**2,'action_ceilings':CEILINGS,'started_unix':time.time()};store.json('runtime-start.json',rt)
 result={'status':'incomplete','rows':[],'scope':'two saved13Dmidpoint maps and selected e0response column; all inner/outer callback costs; no model, derivative, geometryutility or measured LLMspeedup'}
 try:
  n=13;p=18;m=np.diag(np.exp(np.linspace(-.6,.3,n)))+.01*np.triu(np.ones((n,n)),1)+.003*np.tril(np.ones((n,n)),-1);maps=np.array([m+((t%3)-1)*.0001*np.triu(np.ones((n,n)),1)for t in range(p)]);store.array('synthetic-maps',maps);start=time.monotonic();profile=probe(maps,173664,emitter('profile'),tick);elapsed=time.monotonic()-start;store.json('profile-result.json',profile)
  gate=profile_projection(elapsed,store.used,profile['counts']);gate['gate']=bool(gate['projected_seconds']<45 and gate['projected_bytes']<255*1024**2);store.json('profile-gate.json',gate);result['profile']=gate;B.require(gate['gate'],'fixed profile resource gate')
  B.verify_inputs();baselines=B.verify_baselines();store.json('input-binding.json',{'root':str(B.RAW),'manifest':B.PIN,'execution':B.EXEC,'baseline_root':str(B.BASE),'baseline_manifest':B.BASEPIN,'baseline_execution':B.BASEEXEC});store.json('retained-baselines.json',baselines)
  for label,p in [('p10-row35',10),('p18-row111',18)]:
   old=next(v for v in baselines if v['label']==label and not v['preconditioned']);B.require(old['counts']['JVP']+old['counts']['VJP']==CEILINGS[label],'exact saved CG300 callback ceiling')
   identity=json.loads((B.RAW/'run'/f'{label}-inputs.json').read_text());maps=np.asarray(identity['jacobian_midpoints'],dtype=np.float64);B.require(maps.shape==(p,13,13) and B.hashlib.sha256(maps.tobytes()).hexdigest()==identity['midpoints_bytes_sha256'],'exact saved midpoint maps')
   direct=json.loads((B.RAW/'run'/f'{label}-matrixfree-direct.json').read_text());B.require(direct['solve_qualified']and direct['steps']==16*p and direct['shift']==-.05 and direct['eta']==.01 and direct['boundary']=='free','same direct reference');oracle=np.asarray(direct['response_block'])[:,0]
   B.require(np.array_equal(maps,np.load(B.BASE/'run'/f'{label}-maps.npy'))and np.array_equal(oracle,np.load(B.BASE/'run'/f'{label}-saved-direct-column.npy')),'same baseline map/reference bytes');store.array(label+'-maps',maps);store.array(label+'-saved-direct-column',oracle);store.json(label+'-start.json',{'label':label,'action_ceiling':CEILINGS[label]})
   try:
    row=probe(maps,CEILINGS[label],emitter(label),tick);row['readout_error']=float(np.linalg.norm(np.asarray(row['readout_column'])-oracle));row['endpoint_pass']=bool(row['solve_qualified']and row['readout_error']<=1e-6)
   except (ValueError,np.linalg.LinAlgError)as e:row={'status':'numerical_failure','failure':type(e).__name__+': '+str(e),'endpoint_pass':False}
   row['label']=label;result['rows'].append(row);store.json(label+'-result.json',row);tick()
  result.update(status='completed',both_endpoints_pass=all(v['endpoint_pass']for v in result['rows']));tick()
 except BaseException as e:result.update(status='failed_or_interrupted',failure=type(e).__name__+': '+str(e))
 finally:result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,events=seq[0]);store.finish(result)
 print(json.dumps(result));return 0 if result['status']=='completed'else 2
if __name__=='__main__':raise SystemExit(main())
