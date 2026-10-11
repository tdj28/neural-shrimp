#!/usr/bin/env python3
"""Synthetic-only full705 correlated-inverse and actual13D tanh cost screen."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint,numpy as np
from flint import arb,arb_mat,ctx
import profile_native_unit_pair_pal as P
from neural_shrimp.rigorous_correlated_pal import correlated_tube
from neural_shrimp.rigorous_gradient_path import GradientPathJet
from neural_shrimp.rigorous_eigenpair_path import PathJet
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_orbits import exact,vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=P.BASE/'correlated001';OUT=BASE/'attempt001-profile/run';CAP=768*1024**2
SETTINGS={'period':18,'precision':256,'parameter':'1/16384','halfwidth_exponent':-15,'radius_exponent':-16,'newton_steps':0,'full_chart_repeats':2,'tanh_pairs_each':12}
SOURCES=list(dict.fromkeys(P.SOURCES+['src/neural_shrimp/rigorous_eigenpair_march.py','src/neural_shrimp/rigorous_gradient_path.py','src/neural_shrimp/rigorous_polynomial_inverse.py','src/neural_shrimp/rigorous_correlated_pal.py','tests/test_rigorous_gradient_path.py','tests/test_rigorous_polynomial_inverse.py','tests/test_rigorous_correlated_pal.py','tools/profile_native705_correlated_inverse.py','tools/launch_native705_correlated_inverse_profile.py','tests/test_native705_correlated_inverse_profile.py','docs/plans/native705-correlated-inverse.md','docs/plans/native705-correlated-inverse-profile.md']))
entry=P.entry

def gradient_pair(model,x,q,r,rates,tick=lambda:None,observe=lambda k,v:None,index=None):
 for name,direction in [('q',q),('r',r)]:
  state=[GradientPathJet(PathJet([v,exact(str(i+1)+'/128'),exact('1/256'),0],[direction[i],exact('1/256'),exact('-1/512'),0]),[int(i==j)for j in range(15)])for i,v in enumerate(x)]
  rr=[GradientPathJet(PathJet([v,exact('1/1024'),exact('-1/2048'),0]),[int(j==13+i)for j in range(15)])for i,v in enumerate(rates)]
  observe('profile_gradient_input',{'sample':index,'direction':name,'inputs':[v.serialized()for v in state+rr]});tick();ret=model.cusp_step(state,*rr)
  observe('profile_gradient_return',{'sample':index,'direction':name,'returns':[v.serialized()for v in ret]})
  require(len(ret)==13 and all(len(v.g)==15 and a.is_finite()for v in ret for p in [v.v,*v.g]for a in p.a+p.b),'full unrelated13D gradient-path cost sample')

def projection(chart_times,gradient_times,path_times,peak,raw):
 require(len(chart_times)==2 and len(gradient_times)==len(path_times)==12,'two full705 and12 cost samples each')
 require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in chart_times+gradient_times+path_times),'positive finite timings')
 require(type(peak)is int and peak>0 and type(raw)is int and raw>0,'positive measured resource values')
 seconds=2*(max(chart_times)+36*max(gradient_times)+36*max(path_times))+20
 rss=2*peak+128*1024;storage=3*raw+8*1024**2
 return {'one_fresh_chart_seconds':seconds,'rss_kib':rss,'compressed_storage_bytes':storage,'resource_admitted':seconds<220 and rss<6*1024**2 and storage<1024**3,'target_caps':{'seconds':240,'rss_kib':6*1024**2,'hard_as_bytes':12*1024**3,'raw_bytes':1024**3},'scope':'Conservative measured proposal only; no target authorization or theorem.'}

def main():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=[0];serial=[0];used=[0];counts={};resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
 def alarm(*_):raise TimeoutError('240s synthetic profile cap')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,240)
 def check():
  if time.monotonic()-start>=240:raise TimeoutError('240s profile cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=6*1024**2:raise MemoryError('6GiB profile RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n'
  require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'768MiB profile raw cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=629,'624 native plus5 bounded algebra ticks');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
 for n,v in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed source '+n)
 save('runtime.json',{'commit':commit,'pid':os.getpid(),'mode':'synthetic_only','sources':identity,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':12*1024**3,'tick_cap':629,'target_acquired':False})
 for n in SOURCES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'target_acquired':False,'chart_times':[],'gradient_times':[],'path_times':[]}
 try:
  with ctx.workprec(256):
   initial,anchor,lift=P.fixture(tick,observe);save('synthetic-lift.json',lift)
   old,data=P.quadratic_tube(P.Synthetic13(),initial,**P.SETTINGS,normal=['0']*704+['1'],origin=['0']*705,anchor=anchor,tick=tick,observe=observe)
   require(old['gates']['uniform_full_native_unit_pair_locus']and not old['gates']['primitive_period'],'full705 repeated2 proposal fixture')
   C0=arb_mat([[decode(v)for v in row]for row in old['preconditioner']]);del old;gc.collect()
   for slot in range(2):
    t=time.monotonic();proof=correlated_tube(P.Synthetic13(),data['center'],data['slope'],data['curvature'],C0,period=18,normal=[arb(0)]*704+[arb(1)],origin=[arb(0)]*705,parameter=SETTINGS['parameter'],halfwidth_exponent=-15,radius_exponent=-16,weights=[arb(1)]*705,tick=tick,observe=observe)
    result['chart_times'].append(time.monotonic()-t)
    require(proof['bound']['passed']and proof['gates']['realified_frame_independent']and proof['gates']['whole_positive_angle']and not proof['gates']['primitive_period']and not proof['gates']['anchor_identified'],'full705 correlated conditional control without false primitive/anchor claim')
    del proof;gc.collect()
   inp,x,q,r,rates=P.gd_fixture();save('synthetic-gd-inputs.json',{'map_inputs':inp,'x':vector_json(x),'q':vector_json(q),'r':vector_json(r),'rates':vector_json(rates)});model=TwoRateCuspMap(inp)
   for slot in range(12):
    t=time.monotonic();gradient_pair(model,x,q,r,rates,tick,observe,slot);result['gradient_times'].append(time.monotonic()-t)
   for slot in range(12):
    t=time.monotonic();P.path_pair(model,x,q,r,rates,tick,observe,slot);result['path_times'].append(time.monotonic()-t)
   require(ticks[0]==629,'complete synthetic profile operation count')
   result['projection']=projection(result['chart_times'],result['gradient_times'],result['path_times'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,used[0]);result['passed']=result['projection']['resource_admitted']
 except BaseException as exc:result['failure']={'type':type(exc).__name__,'message':str(exc)}
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=6*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},True)
 for p in sorted(OUT.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 OUT.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
