"""Saved-only independent quadratic/weighted primitive-branch proof review."""
import gc,gzip,hashlib,json,math,os,platform,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from neural_shrimp.rigorous_orbits import canonical_bytes
import review_joined_period18_math as M
import review_quadratic_connection_math as Q
BASE=Path('/data2/neural-shrimp-data/explore/certificates')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
UNITS=[('weighted','d183f0cca438e24a61256cb5beeff3290f2d706b','f945b71d4141434ff48d9645ef3fe6e3b2fa9ee33636d272bd7be9570e8478f2','0d7bb58ead76db13278d1f338b8d0b5896bb3c8dcc48281d9b8e71ef317efd9c',-6),('quadratic','02ab12123a3a3f5bedba6284f3c984f3e3f85c90','e3987a5236d1652abcd49497d5a5d9297a49cf6538cf31fefcfde55ba7d26066','60bddca8270784a7264bd0d9d6650e5efd46ce5eca18b9ff1f30f515c863fbc3',-8)]
FILES=['tools/reviews/review_weighted_connection.py','tools/reviews/review_quadratic_connection_math.py','tools/reviews/review_joined_period18_math.py','tools/reviews/launch_weighted_connection_review.py','tests/reviews/test_quadratic_connection_saved_review.py','docs/plans/weighted-connection-independent-review.md','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','certificates/s2-period18-receipt.json','docs/reviews/s2-period18-weighted-profile.json','docs/reviews/s2-period18-quadratic-profile.json']
ANCHOR_SHA='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277'
WEIGHT_PIN='cbc4c267601135a732268bccf99c6ce91163f7a01171cf0f8c3b5fc1bdd05afa'
WEIGHT_FILE='d2ea032791154ca47d6c57a58af8c6616ef224a10ca49bf3904adb4d47210e5a'

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
  M.need(len(b)<=256*1024**2,'decompression cap');meta=json.loads(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json').read_bytes());M.need(meta=={'uncompressed_bytes':len(b),'uncompressed_sha256':hashlib.sha256(b).hexdigest()},'decompressed exact identity');return json.loads(b)
 return json.loads(p.read_bytes())
def inventory(root,members):
 files={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 M.need(set(files)==set(members),'complete inventory')
 for n,p in files.items():M.need(not p.is_symlink()and not(p.stat().st_mode&0o222)and entry(p)==members[n],'immutable bytes '+n)
 return {'files':len(files),'bytes':sum(v['bytes']for v in members.values())}
def verify(root,pin):
 M.need(entry(root/'manifest.json')['sha256']==pin,'manifest pin '+str(root));return dict(inventory(root,read(root/'manifest.json')['files']),manifest=pin)
def payloads(root,tick):
 for path in root.glob('*.json.gz'):
  h=hashlib.sha256();size=0
  with gzip.open(path,'rb')as f:
   while b:=f.read(1024**2):size+=len(b);M.need(size<=256*1024**2,'payload cap');h.update(b)
  M.need(read(path.with_name(path.name.removesuffix('.json.gz')+'-payload.json'))=={'uncompressed_bytes':size,'uncompressed_sha256':h.hexdigest()},'every compressed payload');tick()

def original_anchor():
 receipt_path=ROOT/'certificates/s2-period18-receipt.json';M.need(entry(receipt_path)['sha256']==ANCHOR_SHA,'exact unchanged canonical anchor')
 receipt=read(receipt_path);root=Path(receipt['external_root']);inventory(root,{**receipt['files'],'receipt.json':entry(receipt_path)})
 original=read(root/'certificate.json')['result'];inputs=read(root/'inputs.json');M.need(original['passed']and receipt['passed']and original['exact_statement']==receipt['statement'],'original proof already established')
 M.need(entry(root/'inputs.json')['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','original consumed exact model/input')
 M.need(inputs['period']==18 and inputs['lr_hidden']=='1254985102288567/17592186044416'and inputs['lr_output']=='1848862009810323/2251799813685248','same exact rates and period')
 phases=original['orbit_enclosures'][:18];return inputs,{'rate':inputs['lr_hidden'],'phase_boxes':phases},[[str(M.pair(v['mid']))for v in row]for row in phases]

def bind_runtime(root,commit):
 run=root/'run';rt=read(run/'runtime-start.json');done=read(run/'outcome.json');launch=read(root/'launcher-result.json')
 M.need(rt['source_commit']==launch['source_commit']==commit,'actual execution identity');M.need(rt['source_sha256']==launch['source_sha256'],'launcher source identity')
 for n,h in rt['source_sha256'].items():M.need(entry(run/'source'/n)['sha256']==h and hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==h,'retained committed source '+n)
 M.need(rt['python_flint']==flint.__version__ and rt['flint']==flint.__FLINT_VERSION__ and rt['python']==platform.python_version(),'same arithmetic runtime')
 M.need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10 and rt['cpu_only']and not rt['paid'],'runtime scope')
 M.need(done['exit_code']==launch['exit_code']==0 and not launch['owned_live_pids'],'successful exited producer')
 for pid in (rt['pid'],launch['launcher_pid'],launch['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
 for value,limit in [(done['elapsed_seconds'],rt['internal_seconds']),(launch['elapsed_seconds'],launch['external_seconds']+7),(done['max_rss_kib'],2*1024**2)]:M.need(type(value)in(int,float)and math.isfinite(value)and 0<=value<limit,'finite bounded resources')
 M.need(done['adapter_counts']=={'map_attempted':144,'map_returned':144,'unit_attempted':288,'unit_returned':288,'map_cap':144,'unit_cap':288}and done['jet_counts']=={'attempted':108,'returned':108,'cap':108},'complete fixed derivative counts')
 return rt,done

def main(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean review source');M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=False;result={'passed':False,'units':[]}
 def save(name,value):
  b=canonical_bytes(value);M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<8*1024**2,'8MiB worker cap')
  with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped and time.monotonic()-start<90,'90s internal');M.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'1.5GiB monitored RSS')
 def stop(*_):
  nonlocal stopped;stopped=True
 save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in FILES},'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k]for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'started_unix':time.time()})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 try:
  model,anchor,initial=original_anchor();tick()
  weightroot=BASE/'s2-period18-quadratic-weights-2026-10-10/attempt-001/run';verify(weightroot,WEIGHT_PIN);M.need(entry(weightroot/'chart-0.json')['sha256']==WEIGHT_FILE,'fixed weight vector identity');weights=read(weightroot/'chart-0.json')['exact_weights']
  for name,commit,outerpin,runpin,half in UNITS:
   root=BASE/('s2-period18-'+name+'-connection-2026-10-10')/'attempt-001-target';run=root/'run';inv=verify(root,outerpin);verify(run,runpin);payloads(run,tick);rt,done=bind_runtime(root,commit)
   profile=read(ROOT/('docs/reviews/s2-period18-'+name+'-profile.json'));proot=Path(profile['external_root']);verify(proot,profile['outer_manifest']['sha256']);verify(proot/'run',profile['run_manifest']['sha256']);pr,pd=bind_runtime(proot,profile['source_commit']);M.need(pr['source_sha256']==rt['source_sha256'],'profile source equality')
   for k in('python','python_flint','flint','numpy','scipy'):M.need(pr[k]==rt[k],'profile library equality')
   M.need(read(run/'cost-gate.json')['manifest_sha256']==profile['run_manifest']['sha256'],'profile dependency')
   inp=read(run/'inputs.json');M.need(inp['model_inputs']==model and inp['anchor']==anchor and inp['initial']==initial and inp['rate']==model['lr_hidden'],'exact original model/native anchor reconstruction')
   proof=read(run/'certificate.json.gz');M.need(proof['inputs_sha256']==entry(run/'inputs.json')['sha256']and proof['runtime']==rt,'proof envelope bindings');r=proof['result']
   M.need(r['passed']and done['passed']and len(r['charts'])==2 and not done['gates']['attraction_evaluated'],'two passed root-only charts')
   for c in r['charts']:
    M.need(c['settings']=={'halfwidth_exponent':half,'radius_exponent':-24,'precision':256,'newton_steps':2},'fixed settings')
    M.need(c['statement']['period']==18 and c['statement']['phase_dimension']==13,'full primitive18/13D scope')
    if name=='weighted':M.need(c['exact_weights']==weights,'same pinned weight vector in both charts')
   if name=='weighted':M.need(read(run/'weights.json')=={'exact_weights':weights,'manifest_sha256':WEIGHT_PIN,'file_sha256':WEIGHT_FILE},'retained weight binding')
   reduced=Q.check_join(r,inp,name=='weighted',tick);save(name+'-arithmetic.json',reduced)
   result['units'].append({'name':name,'passed':reduced['passed'],'execution':commit,'inventory':inv,'exact_parameter_endpoints':reduced['exact_parameter_endpoints'],'precision_bits':[256,384]});del proof,r,reduced,c;gc.collect();tick()
  result.update(passed=True,scope='independent saved Taylor/Krawczyk arithmetic, primitive exclusions and original-anchor/endpoint joins; no attraction or new derivative acquisition')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);inv={str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps(result),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
