#!/usr/bin/env python3
"""One unjoined, freshly corrected full705 cubic chart of a fixed four-chart card."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint,numpy as np
from flint import arb,ctx
import profile_native705_cubic_continuation as P
import probe_native705_correlated_inverse as C
from neural_shrimp.rigorous_cubic_proposal import refresh_cubic_proposal
from neural_shrimp.rigorous_cubic_correlated_pal import cubic_correlated_tube,cubic_anchor_test
from neural_shrimp.rigorous_orbits import exact,vector_json
from neural_shrimp.rigorous_eigenpair_locus import exact_mid
BASE=P.BASE;CAP=1024**3;read=C.read;entry=C.entry;inventory=C.inventory;require=C.require;decode=C.decode
SOURCES=list(dict.fromkeys(C.SOURCES+P.SOURCES+['tools/probe_native705_cubic_batch.py','tools/launch_native705_cubic_batch.py','tests/test_native705_cubic_batch.py','docs/plans/native705-cubic-batch.md']))
APINS={'root':'/data2/neural-shrimp-data/audits/native705-correlated-independent/attempt002','source':'d70eaaf9abd34ea65f1946c7727912b89a57ce55','outer':'db0c695ee52bd019239f683a7158701ad82be9728fcbd0af99359af8c9b66b79','worker':'12157fcd7d23f7aff29c2a8a317ec2a7999348f6162b3347f5fcf8af1537332c','endpoint':'41a904368aeb9458297bf3a436474698d0d95beced993a01102961be13fda078'}
FPINS={'root':str(BASE.parent/'cubic001/attempt001-target'),'source':'c230b809ef43385d624350930c31120f726bd225','outer':'fde50164e0f93636de70a110ed4585d175a2ca0e44bc497eafb2c92075a81579','worker':'dbda7a520bfe1e8f2910aca550ee5d20c9ac8375c5a94f212c4eafd5b2570f7a','certificate':'38a06563fa2f10cea3b0495b7e5133d146cb2b6053062dd42ece0515385363c8'}
EXPECTED={'direction_return':108,'cubic_proposal_input':3,'cubic_proposal_return':3,'cubic_affine_proposal_input':1,'cubic_quadratic_proposal_input':1,'pal_path_phase_return':72,'pal_path_assembly_return':2,'cubic_predictor_return':1,'cubic_correlated_domain':1,'gradient_path_input':72,'gradient_path_return':72,'gradient_path_assembly_return':2,'cubic_path_input':72,'cubic_path_return':72,'cubic_path_assembly_return':2,'cubic_correlated_chart_return':1}

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 return identity

def settings(right,unit):
 require(type(unit)is int and 1<=unit<=4,'fixed four charts')
 center=Fraction(right)+Fraction(1,16)-Fraction(1,2048)+(unit-1)*(Fraction(1,8)-Fraction(1,2048))
 return {'period':18,'precision':256,'parameter':str(center),'halfwidth_exponent':-4,'radius_exponent':-10,'newton_steps':2,'predictor_degree':3,'residual_degree':4}

def profile(identity,pins):
 root=P.OUT.parent;inventory(root,pins['outer']);inventory(root/'run',pins['worker']);rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');out=read(root/'run/outcome.json')
 require(rt['commit']==la['commit']==pins['source']and la['exit_code']==0 and not la['owned_live_pids']and rt['mode']==la['mode']=='synthetic_only','complete sole measured synthetic profile')
 require(set(rt['sources'])==set(P.SOURCES),'complete profiled source closure')
 for n in P.SOURCES:
  require(identity[n]==rt['sources'][n]==entry(root/'run/source'/n),'same measured source '+n)
  require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==identity[n]['sha256'],'historic measured source '+n)
 for k,v in {'settings':P.SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same profile '+k)
 require(out['passed']is True and out['target_acquired']is False and out['ticks']==746,'complete synthetic746 profile')
 expected={k:2*v for k,v in EXPECTED.items()};expected['direction_return']+=36;expected.update(profile_gradient_input=24,profile_gradient_return=24,profile_quartic_input=24,profile_quartic_return=24)
 require(out['events']==expected,'all actual complete profile acquisitions')
 forecast=P.projection(out['chart_times'],out['gradient_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final'])
 require(forecast==out['projection']and forecast['resource_admitted']and out['elapsed_seconds']<240 and out['peak_rss_kib']<6*1024**2,'measured full-refresh caps')
 # Retain measured actual native target costs as a floor; the synthetic
 # matrix fixture alone underestimates the more expensive tanh domains.
 oldroot=Path(FPINS['root']);inventory(oldroot,FPINS['outer']);inventory(oldroot/'run',FPINS['worker']);old=read(oldroot/'run/outcome.json');oldrt=read(oldroot/'run/runtime.json')
 require(oldrt['commit']==FPINS['source']and old['ticks']==149 and old['status']=='completed_bounds_failed'and not old['passed'],'complete retained actual-cost source')
 forecast=actual_floor(forecast,old)
 require(forecast['resource_admitted'],'actual measured native floor fits caps')
 return dict(root=str(root),**pins,forecast=forecast,actual_cost_source=FPINS)

def actual_floor(forecast,old):
 require(all(type(old[k])in(int,float)and old[k]>0 for k in ['seconds','peak_rss_kib','stored_bytes_before_final']),'positive retained actual costs')
 out=dict(forecast);out.update(one_fresh_chart_seconds=max(forecast['one_fresh_chart_seconds'],1.5*old['seconds']+10),rss_kib=max(forecast['rss_kib'],2*old['peak_rss_kib']+128*1024),compressed_storage_bytes=max(forecast['compressed_storage_bytes'],3*old['stored_bytes_before_final']+8*1024**2))
 out['resource_admitted']=out['one_fresh_chart_seconds']<220 and out['rss_kib']<6*1024**2 and out['compressed_storage_bytes']<1024**3
 return out

def uniform(proof):
 g={'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'whole_positive_angle':True,'anchor_identified':False}
 return bool(proof['shooting_dimension']==705 and proof['dimension']==13 and proof['period']==18 and proof['gates']==g and proof['bound']['passed']and len(proof['bound']['coordinate_inclusions'])==705 and all(proof['bound']['coordinate_inclusions']))

def join(anchor,chart):
 require(anchor['normal']==chart['normal']and anchor['origin']==chart['origin'],'identical exact gauge for scalar endpoint use')
 require(len(anchor['box'])==705 and len(chart['bound']['radii'])==705,'all705 joined coordinates')
 return cubic_anchor_test({'box':[decode(v)for v in anchor['box']],'dimension':13,'period':18,'exact_parameter':anchor['parameter']},*[ [decode(v)for v in chart[k]]for k in ['center','slope','curvature','cubic']],*[ [exact(v)for v in chart[k]]for k in ['normal','origin']],exact(chart['parameter_center']),arb((1,chart['settings']['halfwidth_exponent'])),[arb((1,chart['settings']['radius_exponent']))*exact(v)for v in chart['weights']],13,18)

def validate_card(card,source,batch):
 require(card['source']==source and card['batch']==batch and type(batch)is int and 1<=batch<=20,'literal reviewed source/bounded batch')
 require(card['worker_count']==4 and card['kernel']=='fresh-cubic705-v1','fixed full-native packet')
 require(card['settings']==[settings(card['anchor']['parameter'],i)for i in range(1,5)],'fixed contiguous overlapping full705 domains')
 require(all(len(card[k])==705 for k in ['normal','origin','weights'])and len(card['initials'])==4 and all(len(v)==705 for v in card['initials'])and len(card['anchor']['box'])==705,'complete card native state and modes')
 require(card['normal']==card['anchor']['normal']and card['origin']==card['anchor']['origin'],'same actual known-root gauge')
 require(all(0<Fraction(v)<=1 for v in card['weights']),'fixed positive bounded weights')
 return card

def main(args):
 identity=sources();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();require(type(args.batch)is int and 1<=args.batch<=20 and type(args.unit)is int and 1<=args.unit<=4,'bounded batch/unit')
 root=BASE/f'batch{args.batch:03d}';out=root/f'chart{args.unit:03d}/run';require(args.out==out and entry(root/'input-card.json')['sha256']==args.card_sha,'exact exclusive card/path');card=validate_card(read(root/'input-card.json'),source,args.batch);cfg=card['settings'][args.unit-1]
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
 save('runtime.json',{'commit':source,'sources':identity,'settings':cfg,'batch':args.batch,'unit':args.unit,'pid':os.getpid(),'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'worker_seconds':240,'rss_kib':6*1024**2,'hard_as_bytes':12*1024**3,'tick_cap':331})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'connected':False,'batch':args.batch,'unit':args.unit,'status':'unresolved'}
 try:
  save('profile-binding.json',profile(identity,card['profile']));save('inputs.json',{'map_inputs':card['map_inputs'],'card':entry(root/'input-card.json'),'parent':card['parent'],'settings':cfg});check()
  with ctx.workprec(256):
   model=C.TwoRateCuspMap(card['map_inputs']);normal=[exact(v)for v in card['normal']];origin=[exact(v)for v in card['origin']];weights=[exact(v)for v in card['weights']]
   data=refresh_cubic_proposal(model,[exact(v)for v in card['initials'][args.unit-1]],period=18,normal=normal,origin=origin,parameter=exact(cfg['parameter']),newton_steps=2,tick=tick,observe=observe)
   proof=cubic_correlated_tube(model,data['center'],data['slope'],data['curvature'],data['cubic'],data['C0'],period=18,normal=normal,origin=origin,parameter=cfg['parameter'],halfwidth_exponent=-4,radius_exponent=-10,weights=weights,tick=tick,observe=observe,compact=True);save('certificate.json.gz',proof)
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
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--unit',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--card-sha',required=True);return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
