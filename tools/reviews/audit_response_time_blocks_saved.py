"""Post-outcome independent saved arithmetic, no factor/solve/model calls."""
import hashlib,json,math,os,signal,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RAW=Path('/data2/neural-shrimp-data/explore/clark-response/time-blocks-001');PIN='9374e38f7d30879df6a76361a9bf683240106b271e24f1e3427f6aeeb652e0e4'
def require(ok,m):
 if not ok:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):return math.sqrt(math.fsum(float(y)*float(y)for y in np.asarray(x).ravel()))
def audit():
 require(sha(RAW/'manifest.json')==PIN,'outer pin');inv=json.loads((RAW/'manifest.json').read_text())['files'];require(set(inv)=={str(p.relative_to(RAW))for p in RAW.rglob('*')if p.is_file()and p!=RAW/'manifest.json'},'full inventory')
 for n,d in inv.items():
  f=RAW/n;require(not f.is_symlink()and sha(f)==d['sha256']and f.stat().st_size==d['bytes']and not(f.stat().st_mode&0o222),'member')
 run=RAW/'run';runtime=json.loads((run/'runtime-start.json').read_text());launch=json.loads((RAW/'launch-end.json').read_text());require(runtime['source_commit']==launch['source_commit']=='47b6022811bc53b0479dbf8de943515f7de4f380','source');require(all(not Path('/proc/'+str(launch[k])).exists()for k in ['child_pid','launcher_pid']),'owned exits')
 for n,d in runtime['sources'].items():require(hashlib.sha256(subprocess.check_output(['git','show',runtime['source_commit']+':'+n],cwd=ROOT)).hexdigest()==d['sha256'],'historical source')
 records=[json.loads(f.read_text())for f in sorted(run.glob('[0-9]*.json'))];require([r['sequence']for r in records]==list(range(1,len(records)+1)),'sequence');result=json.loads((run/'result.json').read_text());rows=[];factor_error=0.
 def array(r,n):
  f=run/r['arrays'][n]['file'];require(sha(f)==r['arrays'][n]['sha256'],'array join');a=np.load(f,allow_pickle=False);require(np.isfinite(a).all(),'finite array');return a
 for r in records:
  if r['kind']=='factor':
   l,h=array(r,'factor'),array(r,'matrix');e=norm(l@l.T-h)/norm(h);require(e<=1e-10,'saved Cholesky residual');factor_error=max(factor_error,e)
 for row in result['rows']:
  label=row['label'];slot=label+'-'+str(row['preconditioned']);rr=[r for r in records if r['slot']==slot];solutions=[r for r in rr if r['kind']=='solution'];require(len(solutions)==1,'one returned solution');v=array(solutions[0],'state');a=np.load(run/(label+'-maps.npy'),allow_pickle=False)*np.exp(.05);p,n,_=a.shape;t=16*p;vv=v.reshape(t+1,n);res=np.empty((t,n));normal=np.zeros_like(vv);impulse=8*p
  for i in range(t):
   res[i]=vv[i+1]-a[i%p]@vv[i]
   if i==impulse:res[i,0]-=1
   normal[i+1]+=res[i];normal[i]-=a[i%p].T@res[i]
  normal+=.01**2*vv;actual=[r for r in rr if r['kind']=='actual_residual'];require(len(actual)==1,'actual residual');saved=array(actual[0],'normal_residual');require(norm(normal.ravel()-saved)<1e-12,'independent native K/KT arithmetic');rhs=[r for r in rr if r['kind']=='rhs'][0];relative=norm(normal)/norm(array(rhs,'rhs'));read=vv[impulse+1];direct=np.load(run/(label+'-saved-direct-column.npy'));error=norm(read-direct)
  require(math.isclose(relative,row['true_normal_relative_residual'],rel_tol=1e-7,abs_tol=1e-13)and math.isclose(error,row['readout_error'],rel_tol=1e-10,abs_tol=1e-13),'reductions');iters=[r for r in rr if r['kind']=='iterate'];require(len(iters)==row['iterations']and [r['iteration']for r in iters]==list(range(1,len(iters)+1)),'all callbacks');require(row['counts']['JVP']==t*(row['iterations']+1)and row['counts']['VJP']==t*(row['iterations']+2),'callback cost');require(row['endpoint_pass']==(row['info']==0 and relative<=1e-8 and error<=1e-6),'endpoint gate');rows.append(dict(label=label,preconditioned=row['preconditioned'],relative=relative,readout_error=error,iterations=row['iterations'],endpoint_pass=row['endpoint_pass']))
 require(not result['both_slots_pass']and len(rows)==4,'fixed failed screen');return dict(passed=True,raw_manifest=PIN,rows=rows,max_saved_factor_error=factor_error,files=len(inv)+1,scope='post-outcome producer-agent independent saved K/KT, factor residual and scalar reductions; no factorization/solve/model replay; independent personnel review pending')
def main():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');out=Path('/data2/neural-shrimp-data/audits/response-time-blocks/attempt-001');out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 def stop(*_):raise TimeoutError('10s cap')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,10)
 try:r=audit()
 except BaseException as e:r=dict(passed=False,error=type(e).__name__+': '+str(e))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 r.update(seconds=time.monotonic()-start,pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checker_sha256=sha(Path(__file__)));(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':dict(bytes=(out/'result.json').stat().st_size,sha256=sha(out/'result.json'))}},indent=2)+'\n')
 for f in out.iterdir():f.chmod(0o444)
 out.chmod(0o555);print(json.dumps(r));return 0 if r['passed']else 2
if __name__=='__main__':raise SystemExit(main())
