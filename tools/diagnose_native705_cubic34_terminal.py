"""Fixed saved-only diagnosis of terminal failed narrow chart34/domain63 cell."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
import json,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic34_narrow as N
P=N.P
import native705_metric_diagnosis as H
from flint import arb,ctx
O,D,U,need,read=P.O,P.D,P.C.U,P.need,P.read
OUT=N.OUT.parent.with_name('cubic34-terminal-diagnosis001')/'run';CARD='docs/plans/native705-cubic34-terminal-diagnosis-inputs.json'
CAPS=dict(seconds=90,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=64*1024**2,native_calls=0)
FILES=sorted(set(N.FILES+['tools/native705_metric_diagnosis.py','tools/diagnose_native705_cubic34_terminal.py','tools/launch_native705_cubic34_terminal_diagnosis.py','tests/test_native705_cubic34_terminal_diagnosis.py','docs/plans/native705-cubic34-terminal-diagnosis.md',CARD,'docs/reviews/native705-cubic34-narrow001.json']))

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.prefix='profile'
 def tick(self):need(time.monotonic()-self.start<90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'90s1GiB saved diagnosis')
 def save(self,name,value,closing=False):
  raw=D.canonical_bytes(value);need(self.used+len(raw)<(64 if closing else 62)*1024**2,'64MiB2MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def observe(self,k,v):self.save(self.prefix+'/'+k+'.json',v);self.tick()


def card_check(c):
 need(c['status']=='reviewed'and c['caps']==CAPS and c['ordinal']==76 and c['chart']==34 and c['cell']==dict(domain=63,depth=1,index=0,endpoints=['57/256','457/2048'],center='913/4096',halfwidth='1/4096'),'fixed reviewed smallest failed cell')
 need(c['producer']==dict(root=str(N.OUT.parent),source='191b325793e7e4c689d392e940df6657aef424d4',outer='286f719119f39ecdf52171b4e146a2e364444d1a98d682b956cc9f7fb26b6ff8',worker='2db91b947eb2ac17ff3e3bcd169e5600fee99300a445bea2d3a776b69569d7f0'),'literal sealed full panel')


def bind(card,w):
 card_check(card);spec=card['producer'];root=Path(spec['root']);run=root/'run';O.verify(root,spec['outer']);O.verify(run,spec['worker']);rt=D.B.archived(run,spec['source'],'runtime.json','source');owner=read(root/'launcher-result.json');out=read(run/'outcome.json')
 need(not out['passed']and out['domains_done']==63 and len(out['completed_attempts'])==76 and len(out['accepted_cells'])==68 and out['native_calls']==15390 and out['error']=='ValueError: fixed maximum cell depth1','complete retained mathematical bound failure')
 need(owner['exit_code']==2 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'actual owned negative execution')
 for pid in(rt['pid'],owner['launcher_pid'],owner['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'actual owners absent')
 for n,pin in rt['sources'].items():need(O.entry(ROOT/n)==pin,'unchanged actual producer source '+n)
 need(sum(v['bytes']for v in read(run/'manifest.json')['files'].values())==8124766085,'literal complete rawbyte cold-cost size')
 cell=run/'cells/076';row=out['completed_attempts'][-1];need(row['cell']==card['cell']and not row['passed']and row['completed']and row['native_calls']==126,'literal terminal cell')
 need(read(cell/'cell-input.json')==card['cell']and read(cell/'cell-outcome.json')==row,'actual terminal cell records')
 qd=read(cell/'quotient-domain-inputs.json');dd=read(run/'domains/063/quotient-domain-inputs.json');ret=read(cell/'retained-derivatives.json')
 need(qd['cell']==card['cell']and qd['parameter_endpoints']==card['cell']['endpoints']and qd['halfwidth']=='1/4096'and dd['domain']==dict(index=63,endpoints=['57/256','229/1024']),'fixed true derivative domain and center subinterval')
 for k in('whole','tangent'):need(qd['full705'][k]==dd['full705'][k],'unchanged actual full705 '+k)
 need(qd['complements']==dd['complements']and qd['metrics']==dd['metrics']and ret['domain']==dd['domain'],'same actual cyclic frame and moving-plane domain')
 need([i+1 for i,x in enumerate(out['completed_attempts'])if not x['passed']]==[1,4,7,66,69,72,75,76]and out['pending_cells']==[N.C.H.cell(N.H.partition(),63,1,k)for k in(0,1)],'all negatives and exact terminal queue retained')
 paths=sorted((cell/'events').glob('*-phase.json'));need(len(paths)==18,'all18 complete actual quotient phases');proof=read(cell/'complement.json');need(not proof['passed'],'literal failed original strict metric')
 need({p.name for p in(cell/'native').iterdir()}=={f'{j:04d}-{k}.json'for j in range(row['first_call'],row['last_call']+1)for k in('input','return')},'all126 physical acquisitions retained')
 w.save('input-binding.json',dict(producer=spec,cell=row,quotient=O.entry(cell/'quotient-domain-inputs.json'),domain=O.entry(run/'domains/063/quotient-domain-inputs.json'),retained_derivatives=ret,proof=O.entry(cell/'complement.json'),phases={str(p.relative_to(run)):O.entry(p)for p in paths},scope='Only terminal cell76 is diagnosed; complete63-domain/76-attempt/68-positive/eight-negative producer remains immutable, with no whole34 claim. No new derivative or matrix domain is created.'));return paths


def main():
 card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned90s saved diagnosis')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:O.entry(ROOT/n)for n in FILES},caps=CAPS,precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();test=[(H.eye(11)/2,H.eye(11)*arb(0,arb(1)/64))for _ in range(18)];trial=H.analyze(test,arb(1)/4096,tick=w.tick,observe=w.observe);elapsed=time.monotonic()-begin;ratio=8124766085/4285184214;wall=time.monotonic()-w.start+3*elapsed+2*ratio*13.163213+5;raw=3*w.used+4*1024**2;ok=all(v['metric_passed']for v in trial['scenarios'])and wall<90 and raw<64*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024<1024**2;w.save('profile.json',dict(passed=ok,seconds=elapsed,forecast_seconds=wall,forecast_bytes=raw,actual_full_inventory_metadata_floor=ratio*13.163213,reference_inventory_bytes=4285184214,target_inventory_bytes=8124766085,target_arrays_unopened=True));need(ok,'full18phase11mode/source-bound measured admission')
   paths=bind(card,w);phases=[]
   for j,p in enumerate(paths):
    row=read(p);need(row['phase']==j and row['operator_bound']['passed'],'literal actual phase/qualified norm');phases.append((U.R.matrix(row['center_quotient']),U.R.matrix(row['whole_quotient_derivative'])))
   w.prefix='target';result=H.analyze(phases,U.R.exact(F(card['cell']['halfwidth'])),tick=w.tick,observe=w.observe);need(not result['scenarios'][0]['metric_passed'],'original unresolved bound retained');w.tick();answer.update(passed=True,**result)
 except BaseException as exc:answer.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;answer['passed']=bool(answer['passed']and elapsed<90 and peak<1024**2);answer.update(seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
