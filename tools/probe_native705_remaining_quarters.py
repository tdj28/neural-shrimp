"""One context and fixed remaining chart31 quarters; original proof kernels."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_second_quarter as S
Q,OLD,P,O,D,U,B,H,need,read=S.Q,S.OLD,S.P,S.O,S.D,S.U,S.B,S.H,S.need,S.read
from flint import ctx
OUT=S.OUT.parent.with_name('remaining-quarters001')/'run';CARD='docs/plans/native705-remaining-quarters-inputs.json'
PRODUCER=dict(root=str(S.OUT.parent),source='dda943e5bd234d2f2875a0b9dff2deea56467c77',outer='070b819ada019aaeee9f58fca3e3f6145740ffb6e3b61e9f1f8dfeeb9dc5bc1f',worker='637d9cf3603853cd613f6135fed0914e3525b848f2daa99ffdc601ee6261eb19')
CAPS=dict(seconds=6000,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=12*1024**3,profile_calls=144,maximum_target_calls=7920,maximum_total_calls=8064)
FILES=sorted(set(S.FILES+['tools/probe_native705_remaining_quarters.py','tools/launch_native705_remaining_quarters.py','tests/test_native705_remaining_quarters.py','docs/plans/native705-remaining-quarters.md',CARD,'docs/reviews/native705-second-quarter001.json']))


def card_check(card):
 need(card['status']=='reviewed'and card['quarters']==[3,4]and card['domains']=={str(k):H.domains(k)for k in(3,4)}and card['caps']==CAPS and card['second_quarter_producer']==PRODUCER,'literal fixed two remaining quarters')
 a=card['second_quarter_independent'];need(isinstance(a,dict)and set(a)=={'root','source','outer','worker'}and all(isinstance(x,str)and x for x in a.values()),'completed independently replayed secondquarter required')


def forecast(profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib,second_seconds,second_bytes):
 first=H.forecast(profile_seconds,profile_bytes,pilot_seconds,pilot_bytes,elapsed,peak_kib)
 need(second_seconds>0 and second_bytes>0,'actual secondquarter complete cost floor');ratio=8064/1080
 seconds=max(2*first['forecast_seconds'],elapsed+2*ratio*second_seconds+2*61.466+60)
 raw=max(2*first['forecast_bytes'],profile_bytes+2*ratio*second_bytes+64*1024**2)
 return dict(passed=seconds<6000 and raw<12*1024**3 and peak_kib+512*1024<4*1024**2,forecast_seconds=seconds,forecast_bytes=raw,profile_calls=144,maximum_total_calls=8064,maximum_centers=60,maximum_derivative_domains=4,complete_second_quarter_cost=dict(seconds=second_seconds,bytes=second_bytes,native_calls=1080))


def metadata(w,card):
 old=S.read(ROOT/S.CARD);S.bind_meta(w,old);r=Path(PRODUCER['root']);O.verify(r,PRODUCER['outer']);O.verify(r/'run',PRODUCER['worker']);rt=D.B.archived(r/'run',PRODUCER['source'],'runtime.json','source');owner=read(r/'launcher-result.json');out=read(r/'run/outcome.json')
 need(out['passed']and out['native_calls']==1080 and out['union']['endpoints']==['-17/2048','15/2048']and out['union']['accepted_cells']==4 and owner['exit_code']==0 and not owner['owned_live_pids']and owner['worker_pid_absent'],'complete sealed secondquarter producer')
 for pid in[rt['pid'],owner['launcher_pid'],owner['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual prior owners absent')
 independent=S.authority(card['second_quarter_independent']);ir=Path(card['second_quarter_independent']['root']);ib=read(ir/'run/authority-binding.json')
 need(independent['passed']and independent['native_calls']==0 and independent['parameter_endpoints']==['-17/2048','15/2048']and independent['complement_dimension']==11 and ib['producer']==PRODUCER and ib['first_quarter']==S.FIXED_FIRST,'literal independently qualified joined secondquarter')
 for name in P.B.NATIVE+P.T.PATH_FILES:need(rt['sources'][name]==O.entry(ROOT/name),'same actual secondquarter native/path '+name)
 w.save('second-quarter-binding.json',dict(producer=PRODUCER,producer_outcome=out,independent=card['second_quarter_independent'],independent_outcome=independent,independent_binding=ib));return independent


class Work(S.Work):
 output_dir=OUT;maximum_calls=8064;raw_cap=12*1024**3
 def __init__(self):super().__init__();self.unions=[]
 def tick(self):need(not self.stopped and time.monotonic()-self.start<6000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'6000s4GiB shared-context cap')


def target(w,card):
 prior=metadata(w,card);w.prefix='context';context=P.bind(w);w.prefix='';w.save('scope-binding.json',dict(quarters=card['quarters'],domains=card['domains'],parameter_endpoints=['15/2048','79/2048'],historical_input_binding='Original source-bound fullchart authority retained unchanged; historical firstquarter text is not this packet scope.'))
 for quarter in card['quarters']:
  qa=[];attempts=0
  for row in H.domains(quarter):
   w.prefix=f'quarters/{quarter}/domains/{row["index"]:03d}';qi,tan,derivatives=S.derivative_domain(w,row,context,quarter);queue=[H.cell(row['index'],quarter=quarter)]
   while queue:
    need(attempts<30 and len(w.attempts)<60,'fixed perquarter and complete center caps');spec=queue.pop(0);ordinal=len(w.attempts)+1;w.prefix=f'cells/{ordinal:03d}';w.save('cell-input.json',dict(quarter=quarter,cell=spec));out=S.fresh_cell(w,spec,context,qi,tan,derivatives,quarter);out['quarter']=quarter;w.save('cell-outcome.json',out);w.attempts.append(out);attempts+=1;w.prefix='';w.save(f'completed-attempt-{ordinal:03d}.json',out)
    if out['passed']:qa.append(out);w.accepted.append(out)
    else:queue=H.children(spec,out,quarter)+queue
    gc.collect()
  union=H.complete(qa,quarter);w.unions.append(dict(quarter=quarter,**union));w.save(f'quarter-{quarter}-union.json',w.unions[-1])
 need(w.unions[0]['endpoints'][0]==prior['parameter_endpoints'][1]and w.unions[0]['endpoints'][1]==w.unions[1]['endpoints'][0]and w.unions[1]['endpoints'][1]=='79/2048','exact fullchart adjacency to independently qualified prefix');need(w.calls==144+360+126*len(w.attempts)<=8064,'complete declared inclusive native count');union=dict(passed=True,new_quarters=w.unions,whole_chart_endpoints=['-49/2048','79/2048'],prior_quarter_endpoints=prior['parameter_endpoints']);w.save('union.json',union)
 return dict(passed=True,status='remaining_chart31_quarters_complement_covered',union=union,scope='Producer full11 complementary contraction on chart31 remaining two quarters. Prior two quarters independently qualified; new cells require independent replay before wholechart stability promotion. Critical unit pair remains neutral; no fullwindow/genericNS assertion.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,6000)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:O.entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<12*1024**3-8*1024**2,'complete bounded source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();OLD.synthetic(w);seconds=time.monotonic()-begin;costs=[]
   for spec in[dict(root=str(Q.PILOT),outer=Q.PILOT_PINS[0],worker=Q.PILOT_PINS[1]),PRODUCER]:
    r=Path(spec['root']);need(O.entry(r/'manifest.json')['sha256']==spec['outer']and O.entry(r/'run/manifest.json')['sha256']==spec['worker'],'literal completed cost manifests');m=read(r/'run/manifest.json');need(O.entry(r/'run/outcome.json')==m['files']['outcome.json'],'literal completed runtime floor');o=read(r/'run/outcome.json');need(o['passed'],'completed positive actual cost');costs.append((o['elapsed_seconds'],sum(v['bytes']for v in m['files'].values())))
   report=forecast(seconds,w.used,*costs[0],time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,*costs[1]);w.save('batch-profile.json',report);need(report['passed'],'all60centers4domains measured admission before arrays');result=target(w,card)
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  w.prefix='';signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),completed_attempts=w.attempts,accepted_cells=w.accepted,completed_quarters=w.unions,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
