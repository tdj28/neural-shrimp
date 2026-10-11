"""Bounded floating negative-one branch scout from a proved native470 tube."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,json,hashlib,math,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native470_first_flip as C
from neural_shrimp.unit_pair_scout import TanhMap
from neural_shrimp.signed_flip_scout import evaluate,survey
need,read,entry=C.need,C.read,C.entry
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/signed-flip-scout001/attempt001/run')
CARD='docs/plans/native470-flip-scout-inputs.json';SETTINGS=dict(period=18,step=-.25,max_points=48,max_evaluations=320,profile_evaluations=12,native_calls=5976,seconds=240,rss_bytes=3*1024**3,as_bytes=8*1024**3,raw_bytes=256*1024**2)
FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-flip-scout.md','docs/reviews/native470-weighted-flip002.json','docs/reviews/native470-first-diagnosis001.json','docs/reviews/native470-independent001.json','src/neural_shrimp/unit_pair_scout.py','src/neural_shrimp/signed_flip_scout.py','tools/scout_native470_flip.py','tools/launch_native470_flip_scout.py','tests/test_native470_flip_scout.py']))
def safe(v):
 if isinstance(v,np.ndarray):return safe(v.tolist())
 if isinstance(v,(np.floating,float)):return float(v)if math.isfinite(v)else dict(nonfinite=str(v))
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [safe(x)for x in v]
 return v
class Work:
 def __init__(self):self.start=time.monotonic();self.calls=0;self.serial=0;self.used=0;self.stage='profile';self.chart=None;self.first=None
 def check(self):
  if not(time.monotonic()-self.start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2):raise RuntimeError('240s3GiB resource cap')
 def tick(self):
  if self.calls>=SETTINGS['native_calls']:raise RuntimeError('inclusive native phase cap')
  self.calls+=1;self.check()
 def save(self,name,row,closing=False):
  raw=(json.dumps(safe(row),sort_keys=True,allow_nan=False)+'\n').encode();b=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw;
  if self.used+len(b)>=(256 if closing else 248)*1024**2:raise RuntimeError('256MiB8MiBreserve')
  q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  self.used+=len(b)
  if not closing:self.check()
 def observe(self,kind,row):
  self.save(f'events/{self.serial:05d}-{self.stage}-{kind}.json.gz',row);self.serial+=1
  if kind=='accepted_point'and self.first is None:
   a=association(row['native_flat'],self.chart);self.save('initial-tube-association.json',a);need(a['passed'],'all470 numerical seed lies strictly in known curved tube');self.first=a

def rational(v):
 need(v['rad'][0]==0,'exact saved proposal coefficient');return F(v['mid'][0])*F(2)**v['mid'][1]
def association(flat,chart):
 need(len(flat)==470 and np.isfinite(flat).all(),'entire finite470 proposal');x=list(map(F.from_float,map(float,flat)));c,s,a=[[rational(v)for v in chart[k]]for k in ('center','slope','curvature')];normal=list(map(F,chart['normal']));origin=list(map(F,chart['origin']));d=sum(n*(z-o)for n,z,o in zip(normal,x,origin))-F(chart['parameter_center']);h=F(2)**chart['settings']['halfwidth_exponent'];rad=[rational(v)for v in chart['coordinate_radii']];checks=[abs(x[i]-c[i]-s[i]*d-a[i]*d*d/2)<rad[i]for i in range(470)];return dict(passed=all(checks)and abs(d)<h,coordinates=sum(checks),coordinate_inclusions=checks,projected_parameter=str(d),parameter_inside=abs(d)<h,scope='Exact rational comparison of a floating numerical candidate with a proved full470 tube. A finite-residual numerical point is not itself an exact periodic root or a rigorous continued branch.')

def profile(w):
 inp,x=C.L.K.fixture();n=13;z=np.r_[[float(v.mid())for v in x[:n]],float(x[n+1].mid()),[float(v.mid())for v in x[n+2:2*n+2]]];hidden=float(x[n].mid());model=TanhMap(inp);times=[];sizes=[];w.save('profile-inputs.json',dict(map_inputs=inp,z=z,hidden=hidden,period=18))
 for i in range(12):
  used=w.used;t=time.monotonic();w.observe('evaluation_input',dict(index=i,z=z,hidden=hidden));v,J,dh,d=evaluate(model,z,hidden,18,w.tick,w.observe);w.observe('evaluation_return',dict(index=i,value=v,jacobian=J,hidden_derivative=dh,native=d));times.append(time.monotonic()-t);sizes.append(w.used-used)
 old=read(ROOT/'docs/reviews/native470-first-diagnosis001.json')['outcome'];seconds=time.monotonic()-w.start+2*SETTINGS['max_evaluations']*max(times)+2*old['seconds']+20;raw=2*SETTINGS['max_evaluations']*max(sizes)+32*1024**2;rss=max(2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024,2*old['peak_rss_kib']+256*1024);ans=dict(passed=seconds<240 and raw<256*1024**2 and rss<3*1024**2,seconds=times,bytes=sizes,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True,native_calls=w.calls);w.save('profile.json',ans);need(ans['passed']and w.calls==216,'full native18 timing/storage admission')

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS,'fixed reviewed scout');r=card['proof'];p=Path(r['root'])
 for b,k in((p,'outer'),(p/'run','worker')):C.L.K.inventory(b,r[k]['sha256'],w)
 rt=read(p/'run/runtime.json');la=read(p/'launcher-result.json');o=read(p/'run/outcome.json');need(rt['source_commit']==la['source_commit']==r['source']and o==r['outcome']and o['passed']and o['strict_inclusions']==o['anchor_inclusions']==470 and all(o['gates'].values()),'complete connected primitive18 signed470 authority')
 need(la['exit_code']==0 and la['failure']is None and not la['owned_live_pids']and all(not Path('/proc',str(la[k])).exists()for k in ('launcher_pid','owned_group_pid')),'owned proof group gone')
 for n,v in rt['sources'].items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',r['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'literal proof source');w.check()
 a=card['independent'];ap=Path(a['root'])
 for b,k in ((ap,'outer'),(ap/'run','worker')):C.L.K.inventory(b,a[k]['sha256'],w)
 ar=read(ap/'run/runtime.json');al=read(ap/'launcher-result.json');ao=read(ap/'run/outcome.json');ib=read(ap/'run/input-binding.json')
 need(ar['source_commit']==al['source_commit']==a['source']and ao==a['outcome']and ao['passed']and ao['root_coordinates']==ao['anchor_coordinates']==470 and ao['native_calls']==0,'independent all470 root/anchor authority')
 need(ib==a['binding']and ib['card']['producer']==r,'independent exact same producer/domain binding')
 need(al['exit_code']==0 and al['failure']is None and not al['owned_live_pids']and all(not Path('/proc',str(al[k])).exists()for k in ('launcher_pid','owned_group_pid')),'independent owned group gone')
 for n,v in ar['sources'].items():need(entry(ap/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',a['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'independent exact source');w.check()
 need(entry(p/'run/target-certificate.json.gz')==r['certificate']and entry(p/'run/target-inputs.json')==r['inputs'],'exact proof/domain');
 if metadata_only:return p
 inp=read(p/'run/target-inputs.json');w.save('input-binding.json',dict(proof=r,map_inputs=inp['map_inputs'],scope='Floating version of exactly the same analytic model; all conclusions remain numerical proposals.'))
 with gzip.open(p/'run/target-certificate.json.gz','rb')as f:b=f.read(256*1024**2+1)
 need(len(b)<=256*1024**2,'bounded original certificate read');c=json.loads(b);need(c['passed']and c['period']==18 and c['sigma']==-1 and c['shooting_dimension']==470,'same proved signed tube');h=F(2)**c['settings']['halfwidth_exponent'];d=-h/2;full=[float(rational(v)+rational(s)*d+rational(a)*d*d/2)for v,s,a in zip(c['center'],c['slope'],c['curvature'])];z=np.r_[full[:13],full[-1],full[13:26]];hidden=full[-2];w.chart=c;w.save('seed-proposal.json',dict(z=z,hidden=hidden,full470=full,old_parameter=str(d),scope='Interior exact-predictor finite proposal; corrected actual470 rational tube association required before any advance.'));return TanhMap(inp['map_inputs']),z,hidden

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False,scope='Numerical negative-one branch anatomy proposals only; no additional rigorous extent or stability.')
 def stop(*_):raise TimeoutError('240s owned')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,sources={n:entry(ROOT/n)for n in FILES},numpy=np.__version__,python=platform.python_version(),threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);w.used+=len(b)
  profile(w);card=read(ROOT/CARD);model,z,h=binding(w,card);w.stage='target';ans['target_acquired']=True;result=survey(model,z,h,step=SETTINGS['step'],max_points=SETTINGS['max_points'],max_evaluations=SETTINGS['max_evaluations'],tick=w.tick,observe=w.observe);w.save('survey.json',result);ans.update(passed=True,status=result['status'],points=len(result['points']),evaluations=result['evaluations'],last=result['points'][-1]['diagnostics'],last_hidden=result['points'][-1]['hidden'])
 except BaseException as e:ans.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);ans.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.calls,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
