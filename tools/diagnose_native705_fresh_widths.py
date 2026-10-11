"""Fixed saved five-width cocycle replay; same full-quarter derivative domain."""
import json, os, resource, signal, subprocess, sys, time, traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_fresh_center_quarter as Q
from neural_shrimp.rigorous_correlated_cocycle import phase_model
from neural_shrimp.rigorous_norm_cocycle import operator_upper,frobenius_upper,advance,scalar_remainder_metric
from neural_shrimp.rigorous_orbits import eye,ball_json,matrix_json
from flint import arb,arb_mat,ctx
P,O,D,U,need,read=Q.P,Q.O,Q.D,Q.U,Q.need,Q.read
WIDTHS=[F(1,2**k)for k in(11,10,9,8,7)]
OUT=Q.PILOT.with_name('fresh-widths002')/'run'
FILES=sorted(set(Q.FILES+['tools/diagnose_native705_fresh_widths.py','tools/launch_native705_fresh_widths.py','tests/test_native705_fresh_widths.py','docs/plans/native705-fresh-widths.md','docs/reviews/native705-fresh-widths001-profile-failure.json']))


def geometry(center,widths,domain):
 need(center==F(-33,2048)and widths==WIDTHS and domain==[F(-49,2048),F(-17,2048)],'fixed literal original center/five widths/quarter')
 need(all(domain[0]<=center-h<center+h<=domain[1]for h in widths),'complete closed interval lies in original derivative domain')


def replay(phases,h,tick=lambda:None):
 need(len(phases)==18 and h>0,'all18 phases and positive halfwidth')
 acc=([eye(11),arb_mat(11,11),arb_mat(11,11)],arb(0),arb(1));rows=[]
 for j,(center,derivative)in enumerate(phases):
  need(center.nrows()==center.ncols()==derivative.nrows()==derivative.ncols()==11,'full11 quotient')
  coefficients,error=phase_model(center,derivative,U.R.exact(h));op,gamma=operator_upper(coefficients[0],tick);need(op['passed'],'fresh actual center-operator bound')
  epsilon=frobenius_upper(error);acc,detail=advance(acc,(coefficients,epsilon,gamma));rows.append(dict(phase=j,operator=op,error_norm=ball_json(epsilon),product=detail));tick()
 metric=scalar_remainder_metric(*acc,tick=tick)
 return dict(halfwidth=str(h),passed=metric['passed'],remainder=ball_json(acc[1]),metric=metric,phases=rows)


class Work:
 def __init__(self):self.start=time.monotonic();self.used=0
 def tick(self):need(time.monotonic()-self.start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'60s1GiB saved width cap')
 def save(self,name,row,closing=False):
  raw=D.canonical_bytes(row);need(self.used+len(raw)<(64 if closing else 62)*1024**2,'64MiB2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)


def preserved_profile(w):
 r=read(ROOT/'docs/reviews/native705-fresh-widths001-profile-failure.json');root=Path(r['root'])
 need(root==Q.PILOT.with_name('fresh-widths001')and r['execution']=='5dae38f39cddf0fd3f37e8a774665c87ace50ef7'and r['outer']['sha256']=='6fc1d58bbfe58ccc095280db2b1cac469f3be63adb94574fddc2d44728f0a248'and r['worker']['sha256']=='ded4ae76d1e10a342ab3f41528e2695bb527250946702e04808fb32a8129465d','literal old profile-only failure')
 O.verify(root,r['outer']['sha256']);O.verify(root/'run',r['worker']['sha256']);rt=D.B.archived(root/'run',r['execution'],'runtime.json','source');own=read(root/'launcher-result.json');p=read(root/'run/profile.json');out=read(root/'run/outcome.json')
 need(not out['passed']and out['native_calls']==0 and not p['passed']and p['target_arrays_unopened']and p['forecast_bytes']==63409480 and p['forecast_seconds']<60 and own['exit_code']==2 and not own['owned_live_pids']and not(root/'run/input-binding.json').exists(),'unchanged complete pretarget storage rejection')
 for pid in[rt['pid'],own['launcher_pid'],own['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'old failure owners gone')
 w.save('preserved-profile-failure.json',r)


def bind(w):
 preserved_profile(w)
 card=read(ROOT/Q.CARD);Q.card_check(card);Q.independent(w,card)
 O.verify(Q.PILOT,Q.PILOT_PINS[0]);O.verify(Q.PILOT/'run',Q.PILOT_PINS[1]);rt=D.B.archived(Q.PILOT/'run',Q.PILOT_EXEC,'runtime.json','source');own=read(Q.PILOT/'launcher-result.json');out=read(Q.PILOT/'run/outcome.json')
 need(out['passed']and out['native_calls']==270 and out['parameter_endpoints']==['-17/1024','-1/64']and own['exit_code']==0 and not own['owned_live_pids']and own['worker_pid_absent'],'literal independently qualified pilot')
 for pid in[rt['pid'],own['launcher_pid'],own['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual old owner gone')
 O.verify(Q.OLD.PARENT,Q.OLD.PARENT_PIN);parent=D.B.archived(Q.OLD.PARENT/'run',Q.OLD.PARENT_EXEC,'runtime.json','source');full=read(Q.OLD.PARENT/'run/quotient-domain-inputs.json');point=read(Q.PILOT/'run/quotient-domain-inputs.json')
 need(full['parameter_endpoints']==['-49/2048','-17/2048']and point['full705']['whole']==full['full705']['whole']and point['full705']['tangent']==full['full705']['tangent']and point['complements']==full['complements']and point['metrics']==full['metrics'],'same entire quarter domains/tangent/frames')
 for n,v in rt['sources'].items():need(O.entry(ROOT/n)==v,'unchanged actual pilot source '+n)
 paths=sorted((Q.PILOT/'run/events').glob('*-phase.json'));need(len(paths)==18,'all18 actual phase reductions')
 geometry(F(-33,2048),WIDTHS,list(map(F,full['parameter_endpoints'])))
 w.save('input-binding.json',dict(pilot=card['pilot'],independent_pilot=card['independent_pilot'],whole_parent=dict(root=str(Q.OLD.PARENT),outer=Q.OLD.PARENT_PIN,source=Q.OLD.PARENT_EXEC),phases={str(p.relative_to(Q.PILOT)):O.entry(p)for p in paths},center='-33/2048',widths=list(map(str,WIDTHS)),quarter=full['parameter_endpoints'],scope='The actual known center, full705 whole-quarter root/tangent, native DF/DFprime and exact frames stay fixed; only the centered parameter halfwidth changes. No uncertainty is removed.'))
 return paths


def main():
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();out=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('owned60s saved width replay')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:O.entry(ROOT/n)for n in FILES},precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,native_calls=0))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed archive');need(w.used+len(raw)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   before=time.monotonic();synthetic=[(eye(11)/2,eye(11)*arb(0,arb(1)/64))for _ in range(18)]
   profiles=[replay(synthetic,h,w.tick)for h in WIDTHS];need(all(r['passed']for r in profiles),'all full18phase11mode synthetic controls');elapsed=time.monotonic()-before;forecast=time.monotonic()-w.start+3*elapsed+20;raw=4*(w.used+len(D.canonical_bytes(profiles)))+2*1024**2;passed=forecast<60 and raw<64*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+128*1024<1024**2;w.save('profile.json',dict(passed=passed,seconds=elapsed,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True));need(passed,'full source-bound measured replay admission')
   paths=bind(w);phases=[]
   for j,p in enumerate(paths):
    row=read(p);need(row['phase']==j and row['operator_bound']['passed'],'literal phase order/old qualified reduction');phases.append((U.R.matrix(row['center_quotient']),U.R.matrix(row['whole_quotient_derivative'])))
   results=[]
   for i,h in enumerate(WIDTHS):
    r=replay(phases,h,w.tick);w.save(f'width-{i:02d}.json',r);results.append(dict(halfwidth=str(h),endpoints=[str(F(-33,2048)-h),str(F(-33,2048)+h)],passed=r['passed'],remainder=r['remainder'],penalty=r['metric']['remainder_penalty']))
   need(results[0]['passed'],'original independent narrow width reproduced');out.update(passed=True,results=results,scope='Fixed saved width diagnosis under actual full-quarter domains. Larger passing intervals are source-conditional producer arithmetic; no newly independently promoted coverage and no evidence for other chart centers/quarters.')
 except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(out));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
