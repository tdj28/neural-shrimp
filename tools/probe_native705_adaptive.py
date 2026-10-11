#!/usr/bin/env python3
"""One source/card-bound unjoined cubic705 tube in a bounded adaptive packet."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint,numpy as np
from flint import arb,ctx
import probe_native705_cubic_batch as W
from neural_shrimp.rigorous_cubic_proposal import refresh_cubic_proposal
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_correlated_tube,cubic_anchor_test
from neural_shrimp.rigorous_orbits import exact,vector_json
from neural_shrimp.rigorous_eigenpair_locus import exact_mid
P=W.P;C=W.C;BASE=P.BASE/'adaptive001';MAX_ATTEMPTS=32;CAP=1024**3
read=W.read;entry=W.entry;inventory=W.inventory;require=W.require;decode=W.decode;EXPECTED=W.EXPECTED
SOURCES=list(dict.fromkeys(W.SOURCES+['src/neural_shrimp/cubic_rechart.py','src/neural_shrimp/rigorous_pal_rechart.py','tools/probe_native705_adaptive.py','tools/launch_native705_adaptive.py','tests/test_cubic_rechart.py','tests/test_native705_adaptive.py','docs/plans/native705-adaptive-continuation.md','docs/plans/native705-adaptive-policy.json','docs/reviews/native705-cubic-narrow-prefix.json']))

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed adaptive source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def settings(exponent):
 require(type(exponent)is int and -9<=exponent<=-5,'fixed dyadic adaptive width range')
 return dict(period=18,precision=256,parameter=str(Fraction(2)**exponent-Fraction(1,2048)),halfwidth_exponent=exponent,radius_exponent=-10,newton_steps=2,predictor_degree=3,residual_degree=4)

def validate_card(card,source,attempt):
 require(card['source']==source and card['attempt']==attempt and type(attempt)is int and 1<=attempt<=MAX_ATTEMPTS,'literal source and bounded indexed attempt')
 require(card['kernel']=='fresh-cubic705-adaptive-v1'and card['settings']==settings(card['settings']['halfwidth_exponent']),'unchanged native settings with frozen adaptive width')
 require(all(len(card[k])==705 for k in ['normal','origin','weights','initial'])and len(card['anchor']['box'])==705,'complete actual state/modes/rates/angle')
 require('exact_parameter'not in card['anchor'],'rotated gauge cannot inherit exact scalar')
 require(all(0<Fraction(v)<=1 for v in card['weights']),'fixed positive native weights')
 require(all(exact(v).is_exact()and exact(v).is_finite()for k in ['normal','origin','initial']for v in card[k]),'finite exact arbitrary proposals')
 require(card['policy']['sha256']==entry(BASE/'policy.json')['sha256'],'same sealed bounded policy')
 return card

def profile(identity,pins):return W.profile(identity,pins)
def uniform(proof):return W.uniform(proof)

def projected_join(anchor,chart):
 require(len(anchor['box'])==705 and len(chart['weights'])==705,'complete changed-gauge anchor')
 require('exact_parameter'not in anchor,'whole projected parameter only')
 return cubic_anchor_test({'box':[decode(v)for v in anchor['box']],'dimension':13,'period':18},*[[decode(v)for v in chart[k]]for k in ['center','slope','curvature','cubic']],*[[exact(v)for v in chart[k]]for k in ['normal','origin']],exact(chart['parameter_center']),arb((1,chart['settings']['halfwidth_exponent'])),[arb((1,chart['settings']['radius_exponent']))*exact(v)for v in chart['weights']],13,18)

def main(args):
 identity=sources();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();require(type(args.attempt)is int and 1<=args.attempt<=MAX_ATTEMPTS,'bounded adaptive attempt')
 root=BASE/f'attempt{args.attempt:03d}';out=root/'run';require(args.out==out and entry(root/'input-card.json')['sha256']==args.card_sha,'exact exclusive card/path');card=validate_card(read(root/'input-card.json'),source,args.attempt);cfg=card['settings']
 require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=[0];serial=[0];used=[0];counts={};resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
 def halt(*_):raise TimeoutError('240s full cubic worker')
 signal.signal(signal.SIGALRM,halt);signal.signal(signal.SIGTERM,halt);signal.setitimer(signal.ITIMER_REAL,240)
 def check():
  if time.monotonic()-start>=240:raise TimeoutError('240s worker cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=6*1024**2:raise MemoryError('6GiB worker RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n';require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'1GiB raw plus closing reserve');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=331,'324 native and7 algebra ticks');check()
 def observe(kind,value):
  counts[kind]=counts.get(kind,0)+1
  if kind=='cubic_correlated_chart_return':value={'canonical_certificate':'certificate.json.gz','gates':value['gates'],'shooting_dimension':705}
  save(f'events/{serial[0]:04d}-{kind}.json.gz',value);serial[0]+=1;check()
 save('runtime.json',{'commit':source,'sources':identity,'settings':cfg,'attempt':args.attempt,'pid':os.getpid(),'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'worker_seconds':240,'rss_kib':6*1024**2,'hard_as_bytes':12*1024**3,'tick_cap':331})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'connected':False,'attempt':args.attempt,'status':'unresolved'}
 try:
  save('profile-binding.json',profile(identity,card['profile']));save('inputs.json',{'map_inputs':card['map_inputs'],'card':entry(root/'input-card.json'),'parent':card['parent'],'settings':cfg,'policy':card['policy'],'anchor':card['anchor']});check()
  with ctx.workprec(256):
   model=C.TwoRateCuspMap(card['map_inputs']);normal=[exact(v)for v in card['normal']];origin=[exact(v)for v in card['origin']];weights=[exact(v)for v in card['weights']]
   data=refresh_cubic_proposal(model,[exact(v)for v in card['initial']],period=18,normal=normal,origin=origin,parameter=exact(cfg['parameter']),newton_steps=2,tick=tick,observe=observe)
   proof=cubic_correlated_tube(model,data['center'],data['slope'],data['curvature'],data['cubic'],data['C0'],period=18,normal=normal,origin=origin,parameter=cfg['parameter'],halfwidth_exponent=cfg['halfwidth_exponent'],radius_exponent=cfg['radius_exponent'],weights=weights,tick=tick,observe=observe,compact=True);save('certificate.json.gz',proof)
   require(ticks[0]==331 and counts==EXPECTED,'complete fresh native331 acquisition and no hidden return')
   result.update(passed=uniform(proof),status='uniform705_unjoined'if uniform(proof)else'completed_bounds_failed',contraction=proof['bound']['contraction'],strict_inclusions=sum(proof['bound']['coordinate_inclusions']),gates=proof['gates'],parameter_endpoints=proof['parameter_endpoints'],physical_endpoint=proof['endpoint_root_boxes'][1][-3:])
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--card-sha',required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
