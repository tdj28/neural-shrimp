#!/usr/bin/env python3
"""One independent full705 curve proposed in a bounded rotated eight-chart batch."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import profile_native_unit_pair_pal as P
import probe_native_unit_pair as OLD
import probe_native_unit_pair_pal as C
import diagnose_native_unit_pair_pal as D
import probe_native_unit_pair_pal_repair as R
import localize_native_unit_pair_endpoint as E
import probe_native_unit_pair_pal_batch as B
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import exact,dyadic,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE/'rotated001';CAP=768*1024**2
SOURCES=list(dict.fromkeys(B.SOURCES+R.SOURCES+E.SOURCES+['src/neural_shrimp/rigorous_pal_rechart.py','tests/test_rigorous_pal_rechart.py','tools/probe_native705_rotated.py','tools/launch_native705_rotated.py','tests/test_native705_rotated.py','docs/plans/native705-rotated-march.md','docs/plans/native705-rotated-inputs.json']))
read=C.read;entry=C.entry;inventory=C.inventory

def settings(unit):
 require(type(unit)is int and 1<=unit<=8,'fixed eight proposed charts')
 return dict(period=18,precision=256,parameter=str(Fraction(32*unit-17,2048)),halfwidth_exponent=-7,radius_exponent=-10,newton_steps=2)

def paths(batch,unit):
 require(type(batch)is int and 1<=batch<=10,'at most ten batches');settings(unit)
 root=BASE/f'batch{batch:03d}';return root,root/f'chart{unit:03d}/run'

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def validate_card(card,batch,source):
 require(card['source']==source and card['batch']==batch and card['kernel']=='unchanged-full705-quadratic-v1'and card['worker_count']==8,'literal current source/card/batch')
 require(card['settings']==[settings(i)for i in range(1,9)],'all fixed eight exact domains')
 require(len(card['normal'])==len(card['origin'])==len(card['weights'])==len(card['anchor'])==705 and len(card['initials'])==8 and all(len(v)==705 for v in card['initials']),'full705 finite proposals/known anchor')
 require(all(0<Fraction(v)<=1 for v in card['weights']),'positive exact fixed native weights')
 return card

def bind_card(args):
 root,_=paths(args.batch,args.unit);p=root/'input-card.json'
 require(entry(p)['sha256']==args.card_sha,'exact completed parent proposal card')
 card=read(p);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 return validate_card(card,args.batch,head)

def uniform_admission(proof,positive):
 require(proof['shooting_dimension']==705 and proof['dimension']==13 and proof['period']==18,'complete native705 proof')
 expected={'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'anchor_identified':False}
 return bool(proof['gates']==expected and len(proof['coordinate_inclusions'])==705 and all(proof['coordinate_inclusions'])and positive)

def main(args):
 cfg=settings(args.unit);batchroot,target=paths(args.batch,args.unit);identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 require(args.out==target,'exclusive target path');out=args.out;out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();stop=[False];ticks=[0];serial=[0];used=[0];counts={}
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
 def check():
  if stop[0]or time.monotonic()-start>=240:raise TimeoutError('240s target cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*1024**2:raise MemoryError('4GiB sampled target RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n'
  require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'aggregate target raw cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=253,'252 native passes and1preconditioner maximum');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':'target','settings':cfg,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'worker_seconds':240,'sampled_rss_kib':4*1024**2,'hard_as_bytes':6*1024**3,'tick_cap':253})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'batch':args.batch,'unit':args.unit,'status':'unresolved','connected':False,'scope':'One independent uniform705 curve proof only; ordered strict native endpoint joins are a separate completed-batch operation.'}
 try:
  profile=C.bind_profile(args,identity);save('profile-binding.json',profile);check()
  with ctx.workprec(256):
   card=bind_card(args);initial=card['initials'][args.unit-1];normal=card['normal'];origin=card['origin'];weights=card['weights'];inp=card['map_inputs']
   save('inputs.json',{'map_inputs':inp,'card':entry(batchroot/'input-card.json'),'batch':args.batch,'settings':cfg});save('chart-inputs.json',{'initial':initial,'normal':normal,'origin':origin,'weights':weights,'anchor':None,'settings':cfg});check()
   proof,_=quadratic_tube(TwoRateCuspMap(inp),initial,**cfg,normal=normal,origin=origin,weights=weights,anchor=None,tick=tick,observe=observe);save('certificate.json.gz',proof);check()
   require(ticks[0]==253 and counts['direction_return']==144 and counts['pal_path_phase_return']==108,'complete fixed full705 acquisition')
   positive=bool(decode(proof['whole_box'][-1])>0)
   result.update(passed=uniform_admission(proof,positive),status='uniform705_unjoined'if uniform_admission(proof,positive)else'completed_bounds_failed',gates=dict(proof['gates'],whole_angle_positive=positive),contraction=proof['contraction'],strict_inclusions=sum(proof['coordinate_inclusions']),parameter_endpoints=proof['parameter_endpoints'],endpoint_physical=[v[-3:]for v in proof['endpoint_root_boxes']])
 except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--unit',type=int,required=True);p.add_argument('--card-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile-manifest',required=True);p.add_argument('--profile-outer-manifest',required=True);return p

if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
