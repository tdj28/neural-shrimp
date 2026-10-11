"""Fixed saved-only diagnosis of smallest failed chart34/domain1 cell."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
import json,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_panel as P
import native705_metric_diagnosis as H
from flint import arb,ctx
O,D,U,need,read=P.O,P.D,P.C.U,P.need,P.read
OUT=P.OUT.parent.with_name('cubic34-diagnosis001')/'run';CARD='docs/plans/native705-cubic34-diagnosis-inputs.json'
CAPS=dict(seconds=60,rss_bytes=1024**3,as_bytes=3*1024**3,raw_bytes=64*1024**2,native_calls=0)
FILES=sorted(set(P.FILES+['tools/native705_metric_diagnosis.py','tools/diagnose_native705_cubic34.py','tools/launch_native705_cubic34_diagnosis.py','tests/test_native705_cubic34_diagnosis.py','docs/plans/native705-cubic34-diagnosis.md',CARD,'docs/reviews/native705-cubic-panel001.json']))

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.prefix='profile'
 def tick(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB saved diagnosis')
 def save(self,name,value,closing=False):
  raw=D.canonical_bytes(value);need(self.used+len(raw)<(64 if closing else 62)*1024**2,'64MiB2MiB reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def observe(self,k,v):self.save(self.prefix+'/'+k+'.json',v);self.tick()


def card_check(c):
 need(c['status']=='reviewed'and c['caps']==CAPS and c['ordinal']==74 and c['chart']==34 and c['cell']==dict(domain=1,depth=3,index=0,endpoints=['83/512','167/1024'],center='333/2048',halfwidth='1/2048'),'fixed reviewed smallest failed cell')
 need(c['producer']==dict(root=str(P.OUT.parent),source='8cbb5f429375cb92d0619f093dc50f857977057c',outer='179857405bbbaa7d1bd32f0decd0f788ca28af27c9585a690e3673e945a762ae',worker='81232683b6a1cac7a4ad1fed4da890c11f302a91f3f97a32e5951155f7620a22'),'literal sealed full panel')


def bind(card,w):
 card_check(card);spec=card['producer'];root=Path(spec['root']);run=root/'run';O.verify(root,spec['outer']);O.verify(run,spec['worker']);rt=D.B.archived(run,spec['source'],'runtime.json','source');owner=read(root/'launcher-result.json');out=read(run/'outcome.json')
 need(not out['passed']and out['domains_done']==9 and len(out['completed_attempts'])==74 and len(out['accepted_cells'])==39 and out['native_calls']==10278 and out['error']=='ValueError: fixed maximum refinement depth','complete retained mathematical bound failure')
 need(owner['exit_code']==2 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'actual owned negative execution')
 for pid in(rt['pid'],owner['launcher_pid'],owner['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'actual owners absent')
 for n,pin in rt['sources'].items():need(O.entry(ROOT/n)==pin,'unchanged actual producer source '+n)
 cell=run/'charts/034/cells/074';row=out['completed_attempts'][-1];need(row['cell']==card['cell']and row['chart']==34 and row['ordinal']==74 and not row['passed']and row['completed']and row['native_calls']==126,'literal terminal cell')
 need(read(cell/'cell-input.json')==card['cell']and read(cell/'cell-outcome.json')=={k:v for k,v in row.items()if k not in('chart','ordinal')},'actual terminal cell records')
 qd=read(cell/'quotient-domain-inputs.json');dd=read(run/'charts/034/domains/001/quotient-domain-inputs.json');ret=read(cell/'retained-derivatives.json')
 need(qd['cell']==card['cell']and qd['parameter_endpoints']==card['cell']['endpoints']and qd['halfwidth']=='1/2048'and dd['domain']==dict(index=1,endpoints=['83/512','87/512']),'fixed true derivative domain and center subinterval')
 for k in('whole','tangent'):need(qd['full705'][k]==dd['full705'][k],'unchanged actual full705 '+k)
 need(qd['complements']==dd['complements']and qd['metrics']==dd['metrics']and ret['domain']==dd['domain'],'same actual cyclic frame and moving-plane domain')
 paths=sorted((cell/'events').glob('*-phase.json'));need(len(paths)==18,'all18 complete actual quotient phases');proof=read(cell/'complement.json');need(not proof['passed'],'literal failed original strict metric')
 need({p.name for p in(cell/'native').iterdir()}=={f'{j:04d}-{k}.json'for j in range(row['first_call'],row['last_call']+1)for k in('input','return')},'all126 physical acquisitions retained')
 w.save('input-binding.json',dict(producer=spec,cell=row,quotient=O.entry(cell/'quotient-domain-inputs.json'),domain=O.entry(run/'charts/034/domains/001/quotient-domain-inputs.json'),retained_derivatives=ret,proof=O.entry(cell/'complement.json'),phases={str(p.relative_to(run)):O.entry(p)for p in paths},scope='Only the smallest completed failed cell is diagnosed; full producer including whole33 and all four34 failures remains immutable. No new derivative or matrix domain is created.'));return paths


def main():
 card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned60s saved diagnosis')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:O.entry(ROOT/n)for n in FILES},caps=CAPS,precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();test=[(H.eye(11)/2,H.eye(11)*arb(0,arb(1)/64))for _ in range(18)];trial=H.analyze(test,arb(1)/2048,tick=w.tick,observe=w.observe);elapsed=time.monotonic()-begin;wall=time.monotonic()-w.start+3*elapsed+2*13.163213+5;raw=3*w.used+4*1024**2;ok=all(v['metric_passed']for v in trial['scenarios'])and wall<60 and raw<64*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024<1024**2;w.save('profile.json',dict(passed=ok,seconds=elapsed,forecast_seconds=wall,forecast_bytes=raw,actual_full_inventory_metadata_floor=13.163213,target_arrays_unopened=True));need(ok,'full18phase11mode/source-bound measured admission')
   paths=bind(card,w);phases=[]
   for j,p in enumerate(paths):
    row=read(p);need(row['phase']==j and row['operator_bound']['passed'],'literal actual phase/qualified norm');phases.append((U.R.matrix(row['center_quotient']),U.R.matrix(row['whole_quotient_derivative'])))
   w.prefix='target';result=H.analyze(phases,U.R.exact(F(card['cell']['halfwidth'])),tick=w.tick,observe=w.observe);need(not result['scenarios'][0]['metric_passed'],'original unresolved bound retained');w.tick();answer.update(passed=True,**result)
 except BaseException as exc:answer.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;answer['passed']=bool(answer['passed']and elapsed<60 and peak<1024**2);answer.update(seconds=elapsed,peak_rss_kib=peak);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
