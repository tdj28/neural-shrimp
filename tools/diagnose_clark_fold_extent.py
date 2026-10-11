"""Saved coordinate geometry for a finite native6 fold continuation proposal."""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2)
import hashlib,json,math,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import review_clark_rnn_n12_joint_curves as J
import review_clark_rnn_n12_organizer as O
OUT=Path('/data2/neural-shrimp-data/audits/clark-fold-extent-geometry/attempt001')
CARD='docs/plans/clark-fold-extent-geometry-inputs.json'
SPECS=[dict(root=str(a.RAW),source=a.EXECUTION,outer=a.OUTER_PIN,worker=a.WORKER_PIN,count=n,receipt=r)for a,n,r in[(J,49,'docs/reviews/clark-rnn-n12-fold-continuation-002.json'),(O,17,'docs/reviews/clark-rnn-n12-organizer-connection-001.json')]]
MATRIX='1be92c769bfd21bf23a2a7ff79aca118cd58454b311e4bb095db117c865f265f'
CAPS=dict(seconds=30,rss_bytes=512*1024**2,as_bytes=2*1024**3,raw_bytes=8*1024**2,native_calls=0)
FILES=[CARD,'docs/plans/clark-fold-extent-geometry.md','tools/diagnose_clark_fold_extent.py','tools/launch_clark_fold_extent_geometry.py','tests/test_clark_fold_extent_geometry.py','tools/review_clark_rnn_n12_joint_curves.py','tools/review_clark_rnn_n12_organizer.py']+[s['receipt']for s in SPECS]
def need(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(p.read_text())
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def norm(x):return math.sqrt(math.fsum(float(z)*float(z)for z in np.ravel(x)))
def coordinates(pair,cycle_x,cycle_y,projection):
 need(pair.shape==(25,)and cycle_x.shape==cycle_y.shape==(6,12)and projection.shape==(12,)and all(np.isfinite(v).all()for v in[pair,cycle_x,cycle_y,projection]),'literal finite full native pair')
 x,y=pair[:12],pair[12:24];need(np.array_equal(x,cycle_x[0])and np.array_equal(y,cycle_y[0]),'actual stored native phase0')
 a=norm(x-y)/2;need(a>0,'positive numerical pair amplitude');b=a*a;m=(x+y)/2;q=(x-y)/(2*a);scale=max(1.,float(np.max(np.abs([cycle_x,cycle_y]))));partner=float(np.max(np.abs(cycle_x+np.roll(cycle_y,3,axis=0))))/scale
 need(partner<1e-8 and abs(norm(q)-1)<1e-12 and np.max(np.abs(m+a*q-x))<1e-12 and np.max(np.abs(m-a*q-y))<1e-12,'native signed partner and reversible coordinates')
 projected=math.fsum(float(u)*float(v)for u,v in zip(projection,x-y))/2
 return dict(b=b,amplitude=a,m=m.tolist(),q=q.tolist(),leak=float(pair[-1]),projected_amplitude=projected,projected_squared=projected*projected,projected_squared_over_b=projected*projected/b,partner_identity_error=partner,normalization_error=abs(norm(q)-1),original_native_cycles=[cycle_x.tolist(),cycle_y.tolist()])
class Work:
 def __init__(self):self.root=OUT/'run';self.root.mkdir(parents=True,exist_ok=False);self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'30s512MiB')
 def save(self,n,v,closing=False):
  raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();need(self.used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB reserve');p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
def bind(w,s):
 p=Path(s['root']);need(entry(p/'manifest.json')['sha256']==s['outer']and entry(p/'run/manifest.json')['sha256']==s['worker'],'literal original manifests');J.inventory(p);J.inventory(p/'run');w.tick();rt=read(p/'run/runtime-start.json');owner=read(p/'launch-end.json')
 need(rt['source']['commit']==owner['source_commit']==s['source']and owner['exit_code']==0 and owner['owned_group_empty']and owner['child_pid_absent'],'original complete numerical packet')
 for pid in[rt['pid'],owner['child_pid'],owner['launcher_pid']]:need(not Path('/proc',str(pid)).exists(),'original owners absent')
 for n,pin in rt['source']['sha256'].items():w.tick();need(hashlib.sha256(subprocess.check_output(['git','show',s['source']+':'+n],cwd=ROOT)).hexdigest()==pin,'historical exact source')
 receipt=read(ROOT/s['receipt']);audit=receipt['independent_saved_audit'];need(audit['passed']and audit['execution_commit']==s['source']and audit['points']==s['count']and audit['joins']==s['count']-1 and audit['outer']['manifest_sha256']==s['outer']and audit['worker']['manifest_sha256']==s['worker'],'existing independent full-native numeric authority')
 binding=read(p/'run/input-binding.json')
 def matrix_pins(value):
  if not isinstance(value,dict):return []
  return ([value['matrix_input_sha256']]if 'matrix_input_sha256'in value else[])+[pin for x in value.values()for pin in matrix_pins(x)]
 pins=matrix_pins(binding);need(pins and set(pins)=={MATRIX},'literal same fixed matrix ancestry')
 if s['count']==17:need(binding['parents']['joint']==[Path(SPECS[0]['root']).name,SPECS[0]['outer'],SPECS[0]['worker'],SPECS[0]['source']],'literal overlapping numerical parent')
 result=read(p/'run/result.json');need(len(result['points'])==s['count']and len(result['associations'])==s['count']-1,'complete point/association inventory');return p/'run',result

def diagnose(w):
 rows=[]
 for si,s in enumerate(SPECS):
  run,result=bind(w,s);events={}
  for p in sorted(run.glob('[0-9]*.json')):
   w.tick();d=read(p)
   if d['kind']=='fold_curve_point':need(d['tag']not in events,'unique full fold row');events[d['tag']]=d
  for point in result['points']:
   w.tick();tag=point['fold']['tag'];d=events[tag];need(all(d[k]==v for k,v in point['fold'].items())and d['passed']and d['gain']==point['gain'],'literal final qualified point')
   f=run/d['array_file'];need(entry(f)['sha256']==d['array_sha256'],'exact consumed full arrays')
   with np.load(f,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
   value=coordinates(a['paired_physical'],a['cycle_x'],a['cycle_y'],a['projection']);need(abs(value['projected_amplitude']-d['amplitude'])<1e-12,'original projection amplitude association');rows.append(dict(parent=si,tag=tag,gain=d['gain'],array=entry(f),**value))
 need(len(rows)==66,'all fixed numerical points retained');old=rows[48];new=rows[49];need(old['gain']==new['gain']==4.08 and np.max(np.abs(np.asarray(old['original_native_cycles'])-np.asarray(new['original_native_cycles'])))<1e-7 and abs(old['b']-new['b'])<1e-10,'actual overlap with no phase relabel')
 increasing=list(reversed(rows[49:]))+list(reversed(rows[:48]));steps=[]
 for a,b in zip(increasing,increasing[1:]):
  steps.append(dict(gain0=a['gain'],gain1=b['gain'],b0=a['b'],b1=b['b'],db=b['b']-a['b'],dq=norm(np.asarray(b['q'])-a['q']),q_dot=math.fsum(x*y for x,y in zip(a['q'],b['q']))))
 w.save('points.json',rows);w.save('increasing-gain-steps.json',steps)
 return dict(passed=True,native_calls=0,matrix_input_sha256=MATRIX,points=66,unique_points=65,raw_overlap_gain=4.08,first_gain=increasing[0]['gain'],last_gain=increasing[-1]['gain'],b_range=[increasing[0]['b'],increasing[-1]['b']],b_strictly_increases_with_gain=all(x['db']>0 for x in steps),minimum_forward_q_dot=min(x['q_dot']for x in steps),projected_squared_over_b_range=[min(x['projected_squared_over_b']for x in rows),max(x['projected_squared_over_b']for x in rows)],scope='Saved numerical coordinates/path planning only. Existing independent native6 numerical ancestry is bound; no new rigorous tube, phase-root identity, 12/24 connection or window proof.')
def synthetic(w):
 cx=np.arange(72,dtype=np.float64).reshape(6,12)/100;cy=-np.roll(cx,-3,axis=0);projection=np.ones(12)/math.sqrt(12);pair=np.r_[cx[0],cy[0],.8];start=time.monotonic()
 for _ in range(66):coordinates(pair,cx,cy,projection);w.tick()
 seconds=time.monotonic()-start;forecast=8*seconds+8;report=dict(passed=forecast<30,seconds=seconds,forecast_seconds=forecast,points=66,native_calls=0);w.save('profile.json',report);need(report['passed'],'fixed66 geometry arithmetic admission before arrays')

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10');card=read(ROOT/CARD);need(card==dict(status='reviewed',inputs=SPECS,caps=CAPS),'fixed reviewed metadata');w=Work();out=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned30s saved diagnosis')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'own committed source');p=w.root/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);need(w.used+len(raw)<6*1024**2,'source reserve');p.write_bytes(raw);w.used+=len(raw)
  synthetic(w);out=diagnose(w);w.tick()
 except BaseException as e:out.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out['passed']=bool(out['passed']and out['seconds']<30 and out['peak_rss_kib']<512*1024);w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(w.root)):entry(p)for p in w.root.rglob('*')if p.is_file()}),True)
 print(json.dumps(out));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
