#!/usr/bin/env python3
"""Fixed saved last-chart simple+1/fold qualification. No model acquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
from neural_shrimp.rigorous_shooting_adjoint import qualify,need
from review_clark_rnn_c1_math import restore
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype')
PARENT=BASE/'attempt-008-wider-march'
PARENT_EXEC='d42d15298ec6fa70259a3e079f74e3108c941040'
PARENT_OUTER='8914b8547ac17ccc2c2ec28589a71c17884420d48b3e0c89827fc7530e3c8e06'
PARENT_RUN='c700c7b393e1565e2a8577a19336dd14b819eb649a915e1f4ddd76433fcbb795'
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_shooting_adjoint.py',
 'tools/reviews/review_clark_rnn_c1_math.py','tools/qualify_eigenpair_shooting_fold.py',
 'tools/launch_eigenpair_shooting_fold.py','tests/test_rigorous_shooting_adjoint.py',
 'tests/test_eigenpair_shooting_fold_driver.py','docs/plans/eigenpair-shooting-fold.md',
 'docs/reviews/eigenpair-wider-march-unit008.json']


def entry(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def read(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def path(root,name):
 p=Path(name);need(not p.is_absolute() and '..' not in p.parts,'relative artifact');q=root/p
 need(not q.is_symlink() and q.resolve().is_relative_to(root.resolve()),'local regular artifact');return q

def inventory(root,pin):
 need(entry(root/'manifest.json')['sha256']==pin,'manifest identity')
 inv=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
 need(set(inv)==actual,'complete inventory')
 for name,item in inv.items():
  p=path(root,name);need(not p.stat().st_mode&0o222 and entry(p)==item,'immutable bytes '+name)
 return {'root':str(root),'manifest':entry(root/'manifest.json'),'files':len(inv)}

def finite_bound(x,limit):return type(x) in (float,int) and math.isfinite(x) and 0<=x<limit

def bind_parent():
 bindings={'outer':inventory(PARENT,PARENT_OUTER),'run':inventory(PARENT/'run',PARENT_RUN)}
 rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json')
 need(rt['commit']==la['commit']==PARENT_EXEC and rt['mode']==la['mode']=='target' and rt['unit']==8,'exact execution')
 need(la['exit_code']==0 and la['stop_reason'] is None and not la['owned_live_pids'] and out['passed'] is True and out['status']=='four_joined_charts','completed parent')
 need(finite_bound(out['elapsed_seconds'],400) and finite_bound(out['peak_rss_kib'],2*1024**2) and finite_bound(la['elapsed_seconds'],427),'parent resource success')
 for name,item in rt['sources'].items():
  need(entry(path(PARENT/'run/source',name))==item,'parent archive')
  need(hashlib.sha256(subprocess.check_output(['git','show',PARENT_EXEC+':'+name],cwd=ROOT)).hexdigest()==item['sha256'],'committed parent source')
 need(len(out['progress']['accepted'])==4 and out['progress']['active_call'] is None,'complete prefix')
 last=out['progress']['last_accepted'];need(last==out['progress']['accepted'][-1] and last['right']=='151/128','fixed last chart')
 p=path(PARENT/'run',last['full_chart_file']);chart=read(p)
 need(chart['passed'] is True and all(chart['gates'].values()) and len(chart['coordinate_inclusions'])==470 and all(chart['coordinate_inclusions']),'parent full proof gates')
 need((chart['period'],chart['dimension'],chart['sigma'])==(18,13,1) and chart['parameter_endpoints']==['147/128','151/128'],'fixed native last segment')
 need([v['divisor'] for v in chart['divisors']]==[1,2,3,6,9] and all(v['excluded'] for v in chart['divisors']) and chart['anchor_check']['passed'],'primitive/anchor premises')
 bindings.update(execution=PARENT_EXEC,full_chart=str(p),full_identity=entry(p),parameter_endpoints=chart['parameter_endpoints'],source_archive_git_verified=True,
   assumption='Producer interval root/derivative certificate authenticated, not independently recomputed here. Wider-chain independent outcome review remains pending.')
 return chart,bindings


def synthetic():
 n=13;p=18;m=2*n*p+2;j=arb_mat(m,m);x=[arb(0)]*m
 for k in range(p):
  z=2*n*k;after=2*n*((k+1)%p);x[z+n]=arb(1)
  for i in range(n):
   val=arb(1) if i==0 else arb(1)/2
   j[z+i,z+i]+=val;j[z+i,after+i]-=1
   j[z+n+i,z+n+i]+=val;j[z+n+i,after+n+i]-=1
  j[z+n,z]=1;j[z,m-2]=1
 return j,x


def profile_gate(root,pin,sources):
 inventory(root,pin);rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 need(rt['mode']==la['mode']=='profile' and rt['commit']==la['commit'] and rt['nice']>=10,'profile mode/source')
 need(rt['sources']==sources and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and out['passed'] is True,'profile byte/source success')
 need(finite_bound(out['elapsed_seconds'],15) and finite_bound(out['peak_rss_kib'],1536*1024) and finite_bound(la['elapsed_seconds'],27),'profile resource bounds')
 need(rt['precision']==256 and rt['python']==sys.version and rt['numpy']==np.__version__ and rt['flint']==flint.__version__ and rt['flint_library']==flint.__FLINT_VERSION__ and rt['threads']==dict.fromkeys(THREADS,'1'),'profile environment')
 projection=3*out['elapsed_seconds']+10
 need(projection<60,'3x profile plus parsing/inventory allowance')
 return {'root':str(root),'outer':pin,'projection_seconds':projection,'scope':'cost screen only; target coefficient gates unchanged'}


def main(args):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');profile=args.mode=='profile'
 need(profile or (args.profile is not None and args.profile_manifest is not None),'pinned profile required')
 out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];serial=[0];total=[0]
 cap=32*1024**2;reserve=4*1024**2;limit=15 if profile else 60;rss=1536*1024
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
 def save(name,obj,final=False):
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True);before=total[0]
  try:
   with p.open('xb') as raw:
    f=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
    try:
     chunks=[];size=0
     for s in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(obj):
      chunks.append(s);size+=len(s)
      if size>65536:
       f.write(''.join(chunks).encode());chunks=[];size=0
       need(before+raw.tell()<(cap if final else cap-reserve),'aggregate byte cap')
     if chunks:f.write(''.join(chunks).encode())
    finally:
     if f is not raw:f.close()
    raw.flush();os.fsync(raw.fileno())
   need(before+p.stat().st_size<(cap if final else cap-reserve),'aggregate byte cap')
  finally:
   if p.exists():total[0]=before+p.stat().st_size
 def check():
  if stop[0] or time.monotonic()-start>=limit:raise TimeoutError('fixed saved-only deadline')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=rss:raise MemoryError('sampled worker RSS')
 def observe(kind,value):
  save(f'events/{serial[0]:03d}-{kind}.json.gz',value);serial[0]+=1;check()
 sources={n:entry(ROOT/n) for n in FILES};commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':sources,'precision':256,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'seconds_cap':limit,'rss_cap_kib':rss,'raw_cap_bytes':cap,'reserve_bytes':reserve,'native_calls':0})
 for n in FILES:
  p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
 outcome={'passed':False,'status':'incomplete','native_calls':0}
 try:
  with ctx.workprec(256):
   if profile:j,x=synthetic();bindings={'synthetic_only':True}
   else:
    gate=profile_gate(args.profile,args.profile_manifest,sources);save('profile-binding.json',gate)
    chart,bindings=bind_parent();save('input-binding.json',bindings);check()
    x=[restore(v) for v in chart['whole_box']];j=arb_mat([[restore(v) for v in row] for row in chart['whole_jacobian']]);del chart
   check();result=qualify(j,x,13,18,observe);save('result.json',result);check()
   outcome.update(passed=result['passed'],status='conditional_fold_qualification_passed' if result['passed'] else 'conditional_fold_gate_unresolved',gates=result['gates'],scope=result['evidence_scope'])
 except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
 outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,events=serial[0],stored_bytes_before_final=total[0])
 if not finite_bound(outcome['elapsed_seconds'],limit) or not finite_bound(outcome['peak_rss_kib'],rss):outcome['passed']=False
 save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
 out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');raise SystemExit(main(p.parse_args()))
