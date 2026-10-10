#!/usr/bin/env python3
"""Bounded two-rate fold curve pilot; numerical evidence only."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gzip,hashlib,json,platform,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import flint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.numerical_boundary import TwoRateBoundary,correct_section,null_tangent,qualify_boundary
SOURCES=['src/neural_shrimp/numerical_boundary.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tools/explore_s2_boundary_curves.py','tests/test_numerical_boundary.py','docs/plans/s2-period18-boundary-curves.md']
FOLD=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target/run')
PINS={'inputs.json':'4a5a8c53b51cd9d79c2f5ed760a53b3092c0003978033ab290486d75fcaed470','certificate.json':'e894dedb29bafa2c054ff6eb5dfd6040cdc10711960c5321c341a32af92d8067'}
CAP=128*1024**2

def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 return x

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 data=(json.dumps(clean(x),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def inventory(p):return {str(f.relative_to(p)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(p.rglob('*')) if f.is_file()}
def source():
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source required')
 return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{p:sha(ROOT/p) for p in SOURCES}}

def execute(args):
 identity=source();out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;stopped=False;last=None;trace=[]
 if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10 required')
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGTERM,stop)
 def check():
  if stopped or time.monotonic()-start>(15 if args.mode=='profile' else 120):raise TimeoutError('cooperative time cap')
  if calls>=800:raise TimeoutError('evaluation cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise MemoryError('RSS cap')
  if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>CAP-8*1024**2:raise OSError('output cap')
 save(out/'runtime-start.json',{'source':identity,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'platform':platform.platform(),'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'mode':args.mode,'started_unix':time.time(),'meminfo':Path('/proc/meminfo').read_text()})
 for p in SOURCES:
  dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
 log=gzip.open(out/'evaluations.jsonl.gz','xt')
 def emit(kind,row):
  line=json.dumps(clean({'kind':kind,**row}),allow_nan=False)+'\n';log.write(line);log.flush();os.fsync(log.buffer.fileobj.fileno())
 result={'status':'started','evidence':'floating_two_parameter_boundary_only','directions':[]}
 try:
  if args.mode=='profile':
   inputs=dict(schema='neural-shrimp-c1-input-v1',network=dict(d=2,width=3,activation='tanh',bias=True,in_scale='3/2',out_scale='2/3',loss_scale='1/2',ridge='1/1000'),x=[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],y=[str(Fraction((i*3)%17-8,64)) for i in range(24)],lr_hidden='1/100',lr_output='1/100')
   seed=np.r_[np.arange(13)/128.,.01,.01,np.ones(13)/np.sqrt(13)]
  else:
   if not args.profile or sha(args.profile/'manifest.json')!=args.profile_manifest:raise ValueError('profile identity')
   manifest=json.loads((args.profile/'manifest.json').read_text())['files']
   if set(manifest)!={str(p.relative_to(args.profile)) for p in args.profile.rglob('*') if p.is_file() and p.name!='manifest.json'}:raise ValueError('profile complete inventory')
   for p,h in manifest.items():
    if sha(args.profile/p)!=h['sha256']:raise ValueError('profile bytes')
   prof=json.loads((args.profile/'outcome.json').read_text());runtime=json.loads((args.profile/'runtime-start.json').read_text())
   if prof['status']!='profile_complete' or runtime['source']['files']!=identity['files'] or runtime['python']!=sys.version or runtime['numpy']!=np.__version__ or runtime['flint']!=flint.__version__ or runtime['threads']!={k:'1' for k in THREADS} or runtime['nice']<10 or runtime['mode']!='profile':raise ValueError('profile source/runtime/status')
   times=prof['evaluation_seconds']
   if not all(np.isfinite(t) and t>0 for t in times) or 800*max(times)+5>=110 or prof['peak_rss_kib']>=200*1024:raise ValueError('measured resource gate')
   for p,h in PINS.items():
    if sha(FOLD/p)!=h:raise ValueError('fold input pin')
   inp=json.loads((FOLD/'inputs.json').read_text());cert=json.loads((FOLD/'certificate.json').read_text())['result']
   if not cert['passed']:raise ValueError('fold certificate passed gate')
   inputs=inp['map_inputs']
   def midpoint(b):return float(Fraction(b['mid'][0])*Fraction(2)**b['mid'][1])
   c=np.array([midpoint(v) for v in cert['center']]);seed=np.r_[c[:14],float(Fraction(inputs['lr_output'])),c[14:]]
   save(out/'input-provenance.json',{'parent':str(FOLD),'pins':PINS,'profile':str(args.profile),'profile_manifest':args.profile_manifest})
  save(out/'inputs.json',{'map_inputs':inputs,'seed':seed,'period':18,'sigma':1});engine=TwoRateBoundary(inputs,18,1)
  def evaluate(z):
   nonlocal calls,last
   check();ident=calls;calls+=1;emit('evaluation_start',{'id':ident,'z':z});now=time.monotonic()
   try:data=engine(z)
   except BaseException as error:emit('evaluation_failure',{'id':ident,'error':type(error).__name__+': '+str(error)});raise
   emit('evaluation_return',{'id':ident,'z':z,'data':data,'seconds':time.monotonic()-now});last=data;return data
  if args.mode=='profile':
   times=[]
   for i in range(5):
    now=time.monotonic();evaluate(seed);times.append(time.monotonic()-now)
   result.update(status='profile_complete',evaluation_seconds=times,projected_seconds=800*max(times)+5)
  else:
   scales=np.ones(28);scales[13]=10;scales[14]=.1;normal=np.eye(28)[14]
   point=correct_section(evaluate,seed,normal,scales,emit=emit);save(out/'seed-corrector.json',point)
   if not point['qualified']:raise ValueError('seed correction failed')
   quality=qualify_boundary(point['data'],1,18)
   if not quality['qualified']:raise ValueError('seed boundary failed')
   tangent,sing=null_tangent(point['data']['jacobian']*scales,normal);base=point['state'];save(out/'seed-qualified.json',{'quality':quality,'tangent':tangent,'singular_values':sing})
   for sign in (-1,1):
    z=base.copy();t=sign*tangent;rows=[];reason='fixed_panel_complete'
    for i in range(32):
     found=False
     for half in range(5):
      step=.025*2.**-half;prediction=z+scales*t*step
      node=correct_section(evaluate,prediction,t,scales,emit=emit)
      if node['qualified']:
       qual=qualify_boundary(node['data'],1,18);move=float(np.linalg.norm((node['state']-z)/scales))
       if qual['qualified'] and move<=.06:
        newt,sing=null_tangent(node['data']['jacobian']*scales,t)
        row={'sign':sign,'index':i,'step':step,'half':half,'point':node['state'],'qualification':qual,'scaled_displacement':move,'tangent':newt,'singular_values':sing,'orbit':node['data']['orbit'],'monodromy':node['data']['monodromy']}
        save(out/f'node-{sign:+d}-{i:03d}.json',row);rows.append(row);z=node['state'];t=newt;found=True;break
      emit('rejected_node',{'sign':sign,'index':i,'half':half,'result':node})
     if not found:reason='unresolved_corrector_or_qualification';break
    result['directions'].append({'sign':sign,'accepted':len(rows),'status':reason,'last_point':z});save(out/f'direction-{sign:+d}.json',result['directions'][-1])
   result['status']='numerical_boundary_panel_recorded'
 except BaseException as error:result.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error))
 finally:
  log.close();result.update(augmented_evaluations=calls,elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save(out/'outcome.json',result);save(out/'manifest.json',{'files':inventory(out)})
 print(json.dumps(clean(result)),flush=True);return 0 if result['status'] in ('profile_complete','numerical_boundary_panel_recorded') else 2

def launch(args):
 identity=source();args.out.mkdir(parents=True,exist_ok=False);external=20 if args.mode=='profile' else 140;start=time.monotonic()
 cmd=['nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--mode',args.mode,'--out',str(args.out/'run')]
 if args.profile:cmd+=['--profile',str(args.profile),'--profile-manifest',args.profile_manifest]
 record={'source':identity,'argv':cmd,'launcher_pid':os.getpid(),'external_seconds':external,'kill_grace_seconds':5,'started_unix':time.time()}
 with (args.out/'process.log').open('xb') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['owned_group_pid']=child.pid;save(args.out/'launcher-start.json',record)
  try:code=child.wait(timeout=external)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
 remaining=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   stat=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(stat[2])==child.pid and stat[0]!='Z':remaining.append(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 record.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_live_pids=remaining);save(args.out/'launcher-result.json',record);save(args.out/'manifest.json',{'files':inventory(args.out)})
 for f in args.out.rglob('*'):
  if f.is_file():f.chmod(0o444)
 for p in sorted((p for p in args.out.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 args.out.chmod(0o555);print(json.dumps(record));return code if not remaining else 2

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((execute if a.execute else launch)(a))
