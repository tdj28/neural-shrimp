"""One fixed saved-only physical line/phase geometry diagnosis."""
import hashlib,io,json,os,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from neural_shrimp.cifar_saved_geometry import assess,need
from diagnose_cifar_phase_scaling import entry,read,preflight
OUT=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/geometry001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/corrector001');PIN='966225d5358a9f81479de0519db856c635398d43b8615aee8705a87e87b6a2ea';EXEC='9bc60ebaea7334a72903d71761e085f99f7bdd1a'
FILES=['src/neural_shrimp/cifar_saved_geometry.py','src/neural_shrimp/cifar_phase_scaling.py','tools/diagnose_cifar_phase_scaling.py','tools/diagnose_cifar_phase_corrector.py','tools/launch_cifar_phase_geometry.py','tests/test_cifar_saved_geometry.py','docs/plans/cifar-phase-geometry.md']
def main():
 preflight();need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('30s saved geometry deadline')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 def tick():need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<448*1024,'30s448MiB cap')
 def save(n,v,closing=False):
  nonlocal used
  b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<2*1024**2-(0 if closing else 512*1024),'2MiB/reserve cap')
  p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();sources={n:entry(ROOT/n)for n in FILES};save('runtime.json',dict(source_commit=source,sources=sources,python=sys.version,numpy=np.__version__,pid=os.getpid(),native_calls=0))
  for n,v in sources.items():
   b=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
  # Full-size unrelated source-only linear/secant profile before scientific arrays.
  t=time.monotonic();x=np.linspace(-.2,.2,98666);p=x/10;trial=assess(x,x+.0001,x+.000001,p,-x*.00001,np.r_[p,p],np.r_[x*.0001,-x*.0001],.008,.01);seconds=time.monotonic()-t;save('profile.json',dict(seconds=seconds,forecast_seconds=4*seconds+15,target_arrays_unopened=True));need(4*seconds+15<30,'fixed measured profile');del x,p,trial
  need(entry(PARENT/'manifest.json')['sha256']==PIN,'exact closed correction');inv=read(PARENT/'manifest.json')['files'];need(set(inv)=={str(p.relative_to(PARENT))for p in PARENT.rglob('*')if p.is_file()and p!=PARENT/'manifest.json'},'complete nested inventory')
  for n,v in inv.items():
   tick();p=PARENT/n;need(not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==v,'immutable correction file')
  row=read(PARENT/'run/result.json');outer=read(PARENT/'launcher-result.json');need(row['source_commit']==outer['source_commit']==EXEC and outer['scientific_complete']and outer['exit_code']==0 and not outer['owned_live_pids']and row['status']=='phase_scaled_pilot_complete'and row['qualification']['passed']and row['accepted_index']is None and not row['trials']and not row['linear']['forcing_pass']and row['linear']['iterations']==400 and row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=818,direct_gradient=14),'literal completed negative correction')
  for pid in [outer['launcher_pid'],outer['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'owned termination')
  with tarfile.open(PARENT/'run/source.tar.gz','r:gz')as tar:
   members=tar.getmembers();need({m.name for m in members}==set(row['source_sha256'])and all(m.isfile()and m.size<2*1024**2 for m in members),'full exact source archive')
   for m in members:
    b=tar.extractfile(m).read();h=row['source_sha256'][m.name];need(hashlib.sha256(b).hexdigest()==h==hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+m.name],cwd=ROOT)).hexdigest(),'actual historical native source')
  names=['initial-theta','initial-map1','initial-map2','direction','actual-action','shooting-direction','shooting-action'];arrays=[np.load(PARENT/'run'/f'{n}.npy',allow_pickle=False)for n in names];need(all(v.dtype==np.float64 for v in arrays),'actual binary64 source arrays');need(np.array_equal(np.load(PARENT/'run/initial-residual.npy',allow_pickle=False),arrays[2]-arrays[0]),'actual same-state residual');ident=row['input_identity'];need(ident['eta']==.00827753 and ident['lam']==.01,'actual rate/ridge')
  answer=assess(*arrays,ident['eta'],ident['lam']);need(abs(answer['physical_forcing']-row['linear']['true_forcing'])<1e-12,'physical forcing replay');result.update(passed=True,analysis=answer,parent=dict(root=str(PARENT),outer=PIN,source=EXEC,arrays={n:entry(PARENT/'run'/f'{n}.npy')for n in names}));save('geometry.json',answer)
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
