#!/usr/bin/env python3
"""Bounded two-rate fold/flip curve pilots; numerical evidence only."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gzip,hashlib,json,platform,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import flint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.numerical_boundary import TwoRateBoundary,correct_section,null_tangent,qualify_boundary,correct_periodic_state
SOURCES=['src/neural_shrimp/numerical_boundary.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tools/explore_s2_boundary_curves.py','tests/test_numerical_boundary.py','docs/plans/s2-period18-boundary-curves.md']
FOLD=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target/run')
PINS={'inputs.json':'4a5a8c53b51cd9d79c2f5ed760a53b3092c0003978033ab290486d75fcaed470','certificate.json':'e894dedb29bafa2c054ff6eb5dfd6040cdc10711960c5321c341a32af92d8067'}
RESUME=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves/attempt-003-target/run')
RESUME_PINS={'inputs.json':'e700d422430a9b49901558a2decef978997e6700767295e993c43ea1d7357dac','node-+1-031.json':'90a908fe4af6e0d7baf707c18f0b184bb8a79702a94753650165b00b5761c8c7'}
FLIP=Path('/data2/neural-shrimp-data/explore/certificates/s2-ray51-2026-10-09/flip-attempt-001-target/run')
FLIP_PINS={'inputs.json':'39714ef6da8689a2f96337ab753b4d75ee129b7fae888ac6e3977cb00c76fa40','certificate.json':'3328669c3e1d3da79c4136e22cf18a96eaab284a7c8b51559e1e8b05353dcd08'}


ENDPOINT_PANELS={
 'ray32-plus':('attempt-006-ray32-fold','cff307557ed5533fe9801783a658e989dd4270ad6dff0fa62aaad30a479efb0a','1302764a424bdc2411b070c5a2b9d262986b769fdbf29ddeaf8456406898ebf5',1,'node-+1-031.json'),
 'ray13-plus':('attempt-007-ray13-flip','2326c70fcb301ffb4b64132275efd8df4283d9556c2cd295ea82b529b87cd3e5','676091e29cb3d230f549038aad6b6b1fc3392714ab3866535c862898489bd157',-1,'node-+1-031.json'),
 'ray51-upper':('attempt-005-ray51-flip','400ec3746753401371da16e3437f84b19ade3c0560ea8b149e96cf193b2f29dc','f4ea28845e06a29c2e5197eb25af9eb6ec79e12b09ae6461ff5fcb20ac89d4bb',-1,'node-+1-031.json'),
 'ray13-upper':('attempt-007-ray13-flip','2326c70fcb301ffb4b64132275efd8df4283d9556c2cd295ea82b529b87cd3e5','a5878186b42926585c1e32a7092e4d77acc425096ea5783f003b0dd30a1a4a2c',-1,'node--1-031.json')}


def exact_coordinate(q):
    value=Fraction(q)
    if value.denominator & (value.denominator-1):raise ValueError('dyadic coordinate required')
    result=np.ldexp(np.longdouble(str(value.numerator)),-(value.denominator.bit_length()-1))
    if Fraction(*result.as_integer_ratio())!=value:raise ValueError('coordinate not exactly representable')
    return result


def midpoint(b):return np.ldexp(np.longdouble(str(b['mid'][0])),b['mid'][1])


def seeded_panel(kind):
    if kind in ENDPOINT_PANELS:
        name,inputs_pin,node_pin,sigma,node_filename=ENDPOINT_PANELS[kind]
        parent=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves')/name/'run'
        pins={'inputs.json':inputs_pin,node_filename:node_pin}
        for filename,pin in pins.items():
            if sha(parent/filename)!=pin:raise ValueError('endpoint input identity')
        row=json.loads((parent/node_filename).read_text());inp=json.loads((parent/'inputs.json').read_text())
        if not row['qualification']['qualified'] or inp['sigma']!=sigma:raise ValueError('qualified correct-sign parent node required')
        seed=np.array([exact_coordinate(v) for v in row['point_exact_dyadics']])
        return inp['map_inputs'],seed,sigma,np.asarray(row['tangent']),[(1,64)],{'parent':str(parent),'pins':pins,'selection':'Continue exact saved endpoint and its outward carried tangent without repeating prefix.'}
    if kind=='ray32-fold':
        for name,pin in CENTER_PINS.items():
            if sha(CENTER/name)!=pin:raise ValueError('center input identity')
        inp=json.loads((CENTER/'inputs.json').read_text());cert=json.loads((CENTER/'certificate.json').read_text())['result']
        if not cert['passed']:raise ValueError('certified center required')
        seed=np.r_[[midpoint(v) for v in cert['center']],exact_coordinate(inp['lr_hidden']),exact_coordinate(inp['lr_output']),np.ones(13)/np.sqrt(13)]
        return inp,seed,1,np.eye(28)[14],[(-1,32),(1,32)],{'parent':str(CENTER),'pins':CENTER_PINS,'selection':'Historical opposite ray32 fold near hidden68.096714; eight fixed-rate orbit corrections to68.2 then augmented fold correction. Numerical branch association only.'}
    if kind=='fold-plus':
        for name,pin in RESUME_PINS.items():
            if sha(RESUME/name)!=pin:raise ValueError('resume input identity')
        row=json.loads((RESUME/'node-+1-031.json').read_text())
        inputs=json.loads((RESUME/'inputs.json').read_text())['map_inputs']
        seed=np.array([exact_coordinate(v) for v in row['point_exact_dyadics']])
        if not row['qualification']['qualified']:raise ValueError('qualified parent node required')
        return inputs,seed,1,np.asarray(row['tangent']),[(1,64)],{'parent':str(RESUME),'pins':RESUME_PINS}
    parent,pins=(FLIP,FLIP_PINS) if kind=='ray51-flip' else (FLIP13,FLIP13_PINS) if kind=='ray13-flip' else (FOLD,PINS)
    for name,pin in pins.items():
        if sha(parent/name)!=pin:raise ValueError('boundary input identity')
    inp=json.loads((parent/'inputs.json').read_text());cert=json.loads((parent/'certificate.json').read_text())['result']
    if not cert['passed']:raise ValueError('boundary certificate passed gate')
    inputs=inp['map_inputs'];c=np.array([midpoint(v) for v in cert['center']])
    if kind in ('ray51-flip','ray13-flip'):
        from neural_shrimp.rigorous_orbits import exact
        with flint.ctx.workprec(128):
            s=exact(str(Fraction(cert['center'][13]['mid'][0])*Fraction(2)**cert['center'][13]['mid'][1]))
            rates=[((exact(e)+exact(d)*s)*flint.arb(10).log()).exp() for e,d in zip(inp['rate_family']['ell'],inp['rate_family']['direction'])]
            native=[np.longdouble(v.mid().str(35,radius=False)) for v in rates]
        seed=np.r_[c[:13],native,c[14:]];sigma=-1
    else:
        out=Fraction(inputs['lr_output']);native_out=np.longdouble(str(out.numerator))/np.longdouble(str(out.denominator))
        seed=np.r_[c[:14],native_out,c[14:]];sigma=1
    return inputs,seed,sigma,np.eye(28)[14],[(-1,32),(1,32)],{'parent':str(parent),'pins':pins}

FLIP13=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-flip-2026-10-09/attempt-001-target/run')
FLIP13_PINS={'inputs.json':'2b6ca1902882b9aa4270be4f40b53eb497a2bab711c9af0fa993039d4d4d5db5','certificate.json':'63400407ffc4a4b91924db2e97adf3b89747e17d9336b6e0f7f7b208462bd56d'}
CENTER=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-2026-10-09/attempt-001/proof/certificate')
CENTER_PINS={'inputs.json':'9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','certificate.json':'ab1f20f9d87e1adf108bb5e5514901725a684ac59586922d81b3a5d2bb2a11ff'}

CAP=128*1024**2

def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.longdouble):return np.format_float_scientific(x,unique=True,trim='k')
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 return x

def coordinates(z):return [f'{n}/{d}' for n,d in (v.as_integer_ratio() for v in np.asarray(z).ravel())]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 data=(json.dumps(clean(x),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def inventory(p):return {str(f.relative_to(p)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(p.rglob('*')) if f.is_file()}
def source():
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source required')
 return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{p:sha(ROOT/p) for p in SOURCES}}

def execute(args):
 identity=source();
 if np.finfo(np.longdouble).nmant<63:raise ValueError('native extended coordinates required')
 out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;stopped=False;last=None;trace=[]
 if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('nice10 required')
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
 def stop(*_):
  nonlocal stopped;stopped=True
 signal.signal(signal.SIGTERM,stop)
 def check():
  if stopped or time.monotonic()-start>(15 if args.mode=='profile' else 180):raise TimeoutError('cooperative time cap')
  if calls>=800:raise TimeoutError('evaluation cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>512*1024:raise MemoryError('RSS cap')
  if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>CAP-8*1024**2:raise OSError('output cap')
 save(out/'runtime-start.json',{'source':identity,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'coordinate_mantissa_bits':int(np.finfo(np.longdouble).nmant),'platform':platform.platform(),'pid':os.getpid(),'ppid':os.getppid(),'threads':{k:os.environ[k] for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'mode':args.mode,'panel':args.panel,'started_unix':time.time(),'meminfo':Path('/proc/meminfo').read_text()})
 for p in SOURCES:
  dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
 log=gzip.open(out/'evaluations.jsonl.gz','xt')
 def emit(kind,row):
  line=json.dumps(clean({'kind':kind,**row}),allow_nan=False)+'\n';log.write(line);log.flush();os.fsync(log.buffer.fileobj.fileno())
 result={'status':'started','evidence':'floating_two_parameter_boundary_only','directions':[]}
 try:
  if args.mode=='profile':
   inputs=dict(schema='neural-shrimp-c1-input-v1',network=dict(d=2,width=3,activation='tanh',bias=True,in_scale='3/2',out_scale='2/3',loss_scale='1/2',ridge='1/1000'),x=[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))] for i in range(24)],y=[str(Fraction((i*3)%17-8,64)) for i in range(24)],lr_hidden='1/100',lr_output='1/100')
   seed=np.r_[np.arange(13)/128.,.01,.01,np.ones(13)/np.sqrt(13)];sigma=1;normal=np.eye(28)[14];directions=[]
  else:
   if not args.profile or sha(args.profile/'manifest.json')!=args.profile_manifest:raise ValueError('profile identity')
   manifest=json.loads((args.profile/'manifest.json').read_text())['files']
   if set(manifest)!={str(p.relative_to(args.profile)) for p in args.profile.rglob('*') if p.is_file() and p.name!='manifest.json'}:raise ValueError('profile complete inventory')
   for p,h in manifest.items():
    if sha(args.profile/p)!=h['sha256']:raise ValueError('profile bytes')
   prof=json.loads((args.profile/'outcome.json').read_text());runtime=json.loads((args.profile/'runtime-start.json').read_text())
   if prof['status']!='profile_complete' or runtime['source']['files']!=identity['files'] or runtime['python']!=sys.version or runtime['numpy']!=np.__version__ or runtime['flint']!=flint.__version__ or runtime['threads']!={k:'1' for k in THREADS} or runtime['nice']<10 or runtime['mode']!='profile':raise ValueError('profile source/runtime/status')
   times=prof['evaluation_seconds']
   if not all(np.isfinite(t) and t>0 for t in times) or 800*max(times)+5>=170 or prof['peak_rss_kib']>=200*1024:raise ValueError('measured resource gate')
   inputs,seed,sigma,normal,directions,provenance=seeded_panel(args.panel)
   save(out/'input-provenance.json',{**provenance,'panel':args.panel,'profile':str(args.profile),'profile_manifest':args.profile_manifest})
  save(out/'inputs.json',{'map_inputs':inputs,'seed':seed,'seed_exact_dyadics':coordinates(seed),'period':18,'sigma':sigma,'directions':directions});engine=TwoRateBoundary(inputs,18,sigma)
  def evaluate(z):
   nonlocal calls,last
   check();ident=calls;calls+=1;emit('evaluation_start',{'id':ident,'z':z,'z_exact_dyadics':coordinates(z)});now=time.monotonic()
   try:data=engine(z)
   except BaseException as error:emit('evaluation_failure',{'id':ident,'error':type(error).__name__+': '+str(error)});raise
   emit('evaluation_return',{'id':ident,'z':z,'data':data,'seconds':time.monotonic()-now});last=data;return data
  if args.mode=='profile':
   times=[]
   for i in range(5):
    now=time.monotonic();evaluate(seed);times.append(time.monotonic()-now)
   result.update(status='profile_complete',evaluation_seconds=times,projected_seconds=800*max(times)+5)
  else:
   scales=np.ones(28);scales[13]=10;scales[14]=.1
   if args.panel=='ray32-fold':
    for index,rate in enumerate(np.linspace(seed[13],np.longdouble('68.2'),9)[1:]):
     proposal=seed.copy();proposal[13]=rate
     corrected=correct_periodic_state(evaluate,proposal,13,emit=emit)
     save(out/f'orbit-seed-{index:02d}.json',corrected)
     if not corrected['qualified']:raise ValueError('opposite-fold seed path failed')
     orbit=corrected['data']['orbit'];m=corrected['data']['monodromy'];scale=max(1,float(np.max(abs(orbit))))
     if any(np.max(abs(orbit[d]-orbit[0]))/scale<=1e-6 for d in [1,2,3,6,9]):raise ValueError('seed primitive18 unresolved')
     seed=corrected['state']
    eigen,vec=np.linalg.eig(m);critical=int(np.argmin(abs(eigen-1)));q=vec[:,critical]
    if abs(eigen[critical].imag)>1e-8 or np.max(abs(q.imag))>1e-8 or min(abs(np.delete(eigen,critical)-eigen[critical]))<1e-5:raise ValueError('seed simple real mode unresolved')
    if np.linalg.norm(m@q-eigen[critical]*q)/max(1,np.linalg.norm(m)*np.linalg.norm(q))>1e-10:raise ValueError('seed eigensystem residual')
    seed[15:]=q.real/np.linalg.norm(q.real)
    save(out/'fold-eigenvector-seed.json',{'point':seed,'point_exact_dyadics':coordinates(seed),'multipliers_real':eigen.real,'multipliers_imag':eigen.imag,'critical_index':critical})
   point=correct_section(evaluate,seed,normal,scales,emit=emit);save(out/'seed-corrector.json',point)
   if not point['qualified']:raise ValueError('seed correction failed')
   quality=qualify_boundary(point['data'],sigma,18)
   if not quality['qualified']:raise ValueError('seed boundary failed')
   tangent,sing=null_tangent(point['data']['jacobian']*scales,normal);base=point['state'];save(out/'seed-qualified.json',{'quality':quality,'tangent':tangent,'singular_values':sing})
   for sign,node_count in directions:
    z=base.copy();t=sign*tangent;rows=[];reason='fixed_panel_complete'
    for i in range(node_count):
     found=False
     for half in range(5):
      step=.025*2.**-half;prediction=z+scales*t*step
      node=correct_section(evaluate,prediction,t,scales,emit=emit)
      if node['qualified']:
       qual=qualify_boundary(node['data'],sigma,18);move=float(np.linalg.norm((node['state']-z)/scales))
       if qual['qualified'] and move<=.06:
        newt,sing=null_tangent(node['data']['jacobian']*scales,t)
        row={'sign':sign,'index':i,'step':step,'half':half,'point':node['state'],'point_exact_dyadics':coordinates(node['state']),'qualification':qual,'scaled_displacement':move,'tangent':newt,'singular_values':sing,'orbit':node['data']['orbit'],'monodromy':node['data']['monodromy']}
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
 identity=source();args.out.mkdir(parents=True,exist_ok=False);external=20 if args.mode=='profile' else 200;start=time.monotonic()
 cmd=['nice','-n','10',sys.executable,str(Path(__file__).resolve()),'--execute','--mode',args.mode,'--panel',args.panel,'--out',str(args.out/'run')]
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
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--panel',choices=['fold-origin','fold-plus','ray51-flip','ray13-flip','ray32-fold','ray32-plus','ray13-plus','ray51-upper','ray13-upper'],default='fold-origin');p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit((execute if a.execute else launch)(a))
