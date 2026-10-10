#!/usr/bin/env python3
"""Fixed saved native +1 quotient stability; no model or derivative acquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import qualify_eigenpair_shooting_fold as P
from neural_shrimp.rigorous_eigenpair_quotient import qualify
from neural_shrimp.rigorous_shooting_adjoint import need
entry,read,inventory,restore,bind_parent,finite_bound,profile_gate=P.entry,P.read,P.inventory,P.restore,P.bind_parent,P.finite_bound,P.profile_gate
FILES=list(dict.fromkeys(P.FILES+['src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_periodic_lyapunov.py','src/neural_shrimp/rigorous_eigenpair_quotient.py','tools/qualify_eigenpair_quotient.py','tools/launch_eigenpair_quotient.py','tests/test_rigorous_eigenpair_quotient.py','docs/plans/eigenpair-neutral-quotient.md']))


def synthetic():
    n=13;p=18
    return [arb_mat([[arb(1) if i==j==0 else (arb('1/2') if i==j else arb(0))for j in range(n)]for i in range(n)])for _ in range(p)],[[arb(int(i==0))for i in range(n)]for _ in range(p)]


def extract_native(chart):
    need((chart['period'],chart['dimension'],chart['sigma'],chart['shooting_dimension'])==(18,13,1,470),'native18 plus mode layout')
    need(chart['passed'] and all(chart['gates'].values()) and len(chart['whole_box'])==470 and len(chart['whole_jacobian'])==470 and all(len(row)==470 for row in chart['whole_jacobian']),'full authenticated chart')
    js=[];qs=[]
    for phase in range(18):
        offset=26*phase
        js.append(arb_mat([[restore(chart['whole_jacobian'][offset+i][offset+j])for j in range(13)]for i in range(13)]))
        qs.append([restore(chart['whole_box'][offset+13+i])for i in range(13)])
    return js,qs


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
    j,x=extract_native(chart);del chart
   check();result=qualify(j,x,precision=256,tick=check,observe=observe);save('result.json',result);check()
   outcome.update(passed=result['passed'],status='conditional_attracting_complement_passed' if result['passed'] else 'complement_bound_unresolved',scope=result['scope'])
 except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
 outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,events=serial[0],stored_bytes_before_final=total[0])
 if not finite_bound(outcome['elapsed_seconds'],limit) or not finite_bound(outcome['peak_rss_kib'],rss):outcome['passed']=False
 save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
 out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');raise SystemExit(main(p.parse_args()))
