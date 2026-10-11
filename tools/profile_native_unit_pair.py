#!/usr/bin/env python3
"""Synthetic-only full704 matrix and native tanh two-direction cost screen."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint,numpy as np
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from neural_shrimp.rigorous_unit_pair import phase,native_lift,unit_pair_tube
from neural_shrimp.rigorous_eigenpair_locus import require,exact_mid
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair')
SETTINGS={'period':18,'precision':256,'parameter':'1/16384','halfwidth_exponent':-14,'radius_exponent':-16,'newton_steps':2}
SOURCES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_cusps.py',
 'src/neural_shrimp/rigorous_eigenpair_locus.py','src/neural_shrimp/rigorous_unit_pair.py',
 'tests/test_rigorous_unit_pair.py','tools/profile_native_unit_pair.py','tools/launch_native_unit_pair_profile.py',
 'tests/test_native_unit_pair_profile.py','docs/plans/native-unit-pair-continuation.md','docs/plans/native-unit-pair-profile.md']
CAP=512*1024**2

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

class Synthetic13:
 n=13
 def cusp_step(self,z,h,o):
  x,y,w=z[:3];e=exact('1/18')
  a=x+y*e;b=y+((h-1)+x*x*exact('1/2')+(o-1)*y+2*x*y)*e
  return [a,b,w*exact('-3/2')+w*w*w*exact('1/2')]+[v*exact('1/2')for v in z[3:]]

def fixture(tick=lambda:None,observe=lambda k,v:None):
 x=[arb(0)]*13;x[2]=arb(1);q=[arb(1)]+[arb(0)]*12;r=[arb(0),arb(1)]+[arb(0)]*11
 z=x+[arb(1),arb(1)]+q+r
 flat,lift=native_lift(Synthetic13(),z,18,tick,observe)
 require(lift['closure_contains_zero'],'known repeated2 synthetic Jordan chain')
 return [exact_mid(v)for v in flat],{'box':flat,'dimension':13,'period':18,'parameter':'0'},lift

def gd_fixture():
 inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},
      'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))]for i in range(24)],
      'y':[str(Fraction((i*3)%17-8,64))for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
 x=[arb(exact(str(Fraction(i-6,64))),arb((1,-20)))for i in range(13)]
 q=[arb(exact(str(Fraction(i+1,32))),arb((1,-20)))for i in range(13)]
 r=[arb(exact(str(Fraction(i-3,64))),arb((1,-20)))for i in range(13)]
 rates=[arb(exact('1/100'),arb((1,-20)))]*2
 return inp,x,q,r,rates

def projection(chart_times,phase_times,peak,raw):
 require(len(chart_times)==2 and len(phase_times)==12,'full measured profile count')
 require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in chart_times+phase_times),'positive finite measured times')
 require(type(peak)is int and peak>0 and type(raw)is int and raw>0,'positive actual resource measurements')
 # A target card may request at most one five-evaluation chart and one lift:
 # 5*18 + 18 =108 paired native phases. Synthetic chart cost is added intact.
 seconds=2*(max(chart_times)+108*max(phase_times))+20
 rss=2*peak+128*1024;storage=3*raw+8*1024**2
 return {'one_chart_plus_lift_seconds':seconds,'rss_kib':rss,'compressed_storage_bytes':storage,
         'resource_admitted':seconds<200 and rss<6*1024**2 and storage<768*1024**2,
         'target_ceiling_proposal':{'worker_seconds':240,'sampled_rss_kib':6*1024**2,'hard_as_bytes':10*1024**3,'raw_bytes':768*1024**2},
         'scope':'Conservative measured heuristic, not target authorization. Conditioning and compression can differ; actual caps remain binding.'}

def main():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');out=BASE/'attempt001-profile/run';out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();stop=[False];ticks=[0];serial=[0];used=[0];counts={}
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2)
 def check():
  if stop[0]or time.monotonic()-start>=240:raise TimeoutError('240s profile cap')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*1024**2:raise MemoryError('4GiB sampled profile RSS')
 def save(name,value,final=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();data=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n'
  require(used[0]+len(data)<CAP-(0 if final else 4*1024**2),'profile raw cap');p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
  used[0]+=len(data)
 def tick():ticks[0]+=1;require(ticks[0]<=422,'420 native passes plus2 preconditioner ticks');check()
 def observe(kind,row):
  counts[kind]=counts.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',row);serial[0]+=1;check()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES}
 for n,row in identity.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
 save('runtime.json',{'commit':commit,'pid':os.getpid(),'mode':'synthetic_only','sources':identity,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'worker_seconds':240,'sampled_rss_kib':4*1024**2,'hard_as_bytes':8*1024**3,'tick_cap':422,'target_acquired':False})
 for n in SOURCES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False,'target_acquired':False,'chart_times':[],'phase_times':[]}
 try:
  with ctx.workprec(256):
   initial,anchor,lift=fixture(tick,observe);save('synthetic-lift.json',lift);save('synthetic-inputs.json',{'initial':initial,'settings':SETTINGS,'known_root':'18 phases repeating an exact primitive2 with normalized +1 Jordan chain'})
   for slot in range(2):
    t=time.monotonic();proof,_=unit_pair_tube(Synthetic13(),initial,**SETTINGS,anchor=anchor,tick=tick,observe=observe)
    result['chart_times'].append(time.monotonic()-t);save(f'chart-{slot}.json.gz',proof)
    require(proof['gates']['uniform_normalized_unit_pair']and proof['gates']['anchor_identified']and proof['gates']['realified_frame_independent']and not proof['gates']['primitive_period'],'full704 positive tube with repeated2 control')
   inp,x,q,r,rates=gd_fixture();save('synthetic-gd-inputs.json',{'map_inputs':inp,'x':vector_json(x),'q':vector_json(q),'r':vector_json(r),'rates':vector_json(rates)});model=TwoRateCuspMap(inp)
   for slot in range(12):
    t=time.monotonic();v,b,e=phase(model,x,q,r,rates,tick,observe,slot)
    result['phase_times'].append(time.monotonic()-t);save(f'gd-pair-{slot}.json.gz',{'value':matrix_json(v),'block':matrix_json(b),'rates':matrix_json(e)})
   require(ticks[0]==422,'complete profile count')
   result['projection']=projection(result['chart_times'],result['phase_times'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,used[0]);result['passed']=result['projection']['resource_admitted']
 except BaseException as exc:result['failure']={'type':type(exc).__name__,'message':str(exc)}
 result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],events=counts,stored_bytes_before_final=used[0])
 if result['elapsed_seconds']>=240 or result['peak_rss_kib']>=4*1024**2:result['passed']=False
 save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
