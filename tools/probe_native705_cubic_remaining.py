"""One known cubic32 root and all remaining exact-domain complementary cells."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_complement as C
S,G,H,OLD,P,O,D,U,B,need,read=C.S,C.G,C.H,C.OLD,C.P,C.O,C.D,C.U,C.B,C.need,C.read
from flint import ctx
OUT=C.OUT.parent.with_name('cubic32-remaining001')/'run';CARD='docs/plans/native705-cubic32-remaining-inputs.json'
FIRST=dict(root=str(C.OUT.parent),source='e63d1c4a35d539b520f7c87ed2ab67ba5a1b58b0',outer='7d44b53806dd8b930e14cb12a8d2c9efaa84b82da5ce56ccfa2ff916ee2fec88',worker='7c1ca53feec8d2f03bdfc3bf1b215f898cfae6daaedc2df7cc0f9322c0a4b4e4')
CAPS=dict(seconds=9000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=18*1024**3,profile_calls=144,maximum_target_calls=11880,maximum_total_calls=12024)
FILES=sorted(set(C.FILES+['tools/probe_native705_cubic_remaining.py','tools/launch_native705_cubic_remaining.py','tests/test_native705_cubic_remaining.py','docs/plans/native705-cubic32-remaining.md',CARD,'docs/reviews/native705-cubic32-first-quarter001.json']))


def card_check(card):
 need(card['status']=='reviewed'and card['ordinal']==32 and card['parameter_endpoints']==['55/1024','103/1024']and card['whole_chart_endpoints']==['39/1024','103/1024'],'fixed reviewed remaining cubic32 interval')
 need(card['domains']==H.domains(*card['parameter_endpoints'],6)and card['maximum_depth']==3 and card['maximum_centers']==90 and card['caps']==CAPS and card['first_producer']==FIRST,'exact six domains/depth3/inclusive caps')
 a=card['first_independent'];need(type(a)is dict and set(a)=={'root','source','outer','worker'}and all(type(v)is str and v for v in a.values()),'completed independent first portion required before any acquisition')


def forecast(profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib,first_seconds,first_bytes):
 base=S.H.forecast(profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib)
 need(first_seconds>0 and first_bytes>0,'actual complete first cubic cost floor');ratio=F(12024,1080)
 seconds=max(3*base['forecast_seconds'],elapsed+2*float(ratio)*first_seconds+2*61.466+60)
 raw=max(3*base['forecast_bytes'],profile_bytes+2*float(ratio)*first_bytes+64*1024**2)
 return dict(passed=seconds<9000 and raw<18*1024**3 and peak_kib+512*1024<4*1024**2,forecast_seconds=seconds,forecast_bytes=raw,maximum_centers=90,maximum_derivative_domains=6,maximum_total_calls=12024,profile_calls=144,complete_first_cost=dict(seconds=first_seconds,bytes=first_bytes,native_calls=1080))


def metadata(w,card):
 r=Path(FIRST['root']);O.verify(r,FIRST['outer']);O.verify(r/'run',FIRST['worker']);rt=D.B.archived(r/'run',FIRST['source'],'runtime.json','source');owner=read(r/'launcher-result.json');out=read(r/'run/outcome.json');original_card=read(r/'run/source'/C.CARD)
 need(out['passed']and out['status']=='cubic32_first_portion_complement_covered'and out['native_calls']==1080 and out['union']['endpoints']==['39/1024','55/1024']and out['union']['accepted_cells']==4,'complete original cubic first portion including failed coarse cells')
 need(owner['source_commit']==FIRST['source']and owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'complete first producer ownership')
 for pid in [rt['pid'],owner['launcher_pid'],owner['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual prior owners absent')
 C.card_check(original_card);need(original_card==read(ROOT/C.CARD),'literal original same-root native card');data=G.metadata(original_card,w.tick)
 independent=S.authority(card['first_independent']);ir=Path(card['first_independent']['root']);ib=read(ir/'run/authority-binding.json')
 need(independent['parameter_endpoints']==['39/1024','55/1024']and independent['total_cells']==4 and independent['complement_dimension']==11 and ib['producer']==FIRST,'exact separately replayed first portion')
 need(ib['original_root']==data['index']and ib['prior_stable']==data['prior']and ib['projection']==data['anchor_projection']and independent['prior_stable_projection']==data['anchor_projection'],'independent complete same root/gauge/previous stable genealogy')
 need(card['parameter_endpoints'][0]==independent['parameter_endpoints'][1]and card['parameter_endpoints'][1]==data['index']['parameter_endpoints'][1]and card['whole_chart_endpoints']==data['index']['parameter_endpoints'],'exact fullchart union with independent first portion')
 for n in G.B.NATIVE+G.M.T.PATH_FILES:need(rt['sources'][n]==O.entry(ROOT/n),'unchanged actual native/path '+n)
 w.save('first-portion-binding.json',dict(producer=FIRST,producer_outcome=out,independent=card['first_independent'],independent_outcome=independent,independent_binding=ib,original_card=original_card));return original_card,independent


class Work(S.Work):
 output_dir=OUT;maximum_calls=12024;raw_cap=18*1024**3
 def tick(self):need(not self.stopped and time.monotonic()-self.start<9000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'9000s4GiB shared cubic context')


def target(w,card):
 original_card,prior=metadata(w,card);w.prefix='context';context=G.bind(w,original_card);w.prefix='';domains=card['domains'];w.save('scope-binding.json',dict(parameter_endpoints=card['parameter_endpoints'],domains=domains,prior_endpoints=prior['parameter_endpoints'],historical_context='Original first-portion context has the same full cubic root and affine inverse; new acquisition scope is this exact remaining interval.'))
 for drow in domains:
  w.prefix=f'domains/{drow["index"]:03d}';qi,tan,derivatives=C.derivative_domain(w,drow,context,domains);queue=[H.cell(domains,drow['index'])]
  while queue:
   need(len(w.attempts)<90,'fixed90 total center ceiling');row=queue.pop(0);ordinal=len(w.attempts)+1;w.prefix=f'cells/{ordinal:03d}';w.save('cell-input.json',row);out=C.fresh_cell(w,row,context,qi,tan,derivatives,domains);w.save('cell-outcome.json',out);w.attempts.append(out);w.prefix='';w.save(f'completed-attempt-{ordinal:03d}.json',out)
   if out['passed']:w.accepted.append(out)
   else:queue=H.children(domains,row,out)+queue
   gc.collect()
 union=H.complete(domains,w.accepted);need(union['endpoints']==card['parameter_endpoints']and prior['parameter_endpoints'][1]==union['endpoints'][0],'literal adjacent proved union');need(w.calls==144+540+126*len(w.attempts)<=12024,'complete inclusive native accounting');union.update(prior_endpoints=prior['parameter_endpoints'],whole_chart_endpoints=card['whole_chart_endpoints']);w.save('union.json',union)
 return dict(passed=True,status='cubic32_remaining_complement_covered',union=union,scope='Producer all11 complementary contraction on remaining cubic32, joined to independently qualified first portion. New cells require independent replay; neutral critical pair retained. No complete window or genericNS assertion.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,9000)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<18*1024**3-8*1024**2,'complete bounded source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   start=time.monotonic();C.synthetic(w);seconds=time.monotonic()-start;costs=[]
   for spec in [dict(root=str(S.Q.PILOT),outer=S.Q.PILOT_PINS[0],worker=S.Q.PILOT_PINS[1]),FIRST]:
    r=Path(spec['root']);need(O.entry(r/'manifest.json')['sha256']==spec['outer']and O.entry(r/'run/manifest.json')['sha256']==spec['worker'],'literal completed cost manifests');m=read(r/'run/manifest.json');need(O.entry(r/'run/outcome.json')==m['files']['outcome.json'],'literal completed runtime floor');o=read(r/'run/outcome.json');need(o['passed'],'completed positive actual cost');costs.append((o['elapsed_seconds'],sum(v['bytes']for v in m['files'].values())))
   report=forecast(seconds,w.used,*costs[0],time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,*costs[1]);w.save('batch-profile.json',report);need(report['passed'],'complete90centers6domains measured admission before arrays');result=target(w,card);w.tick()
 except BaseException as e:result.update(passed=False,status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),completed_attempts=w.attempts,accepted_cells=w.accepted,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
