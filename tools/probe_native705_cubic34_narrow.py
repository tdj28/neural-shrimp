"""Whole34-only repair with64 narrower derivative domains and complete192-cell budget."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import diagnose_native705_cubic34 as V
from neural_shrimp import cubic_complement_narrow as H
P,C,G,S,O,D,need,read=V.P,V.P.C,V.P.G,V.P.S,V.O,V.D,V.need,V.read
from flint import ctx
OUT=P.OUT.parent.with_name('cubic34-narrow001')/'run';CARD='docs/plans/native705-cubic34-narrow-inputs.json'
CAPS=dict(seconds=12600,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=32*1024**3,profile_calls=144,maximum_target_calls=29952,maximum_total_calls=30096)
INDEPENDENT=dict(root='/data2/neural-shrimp-data/audits/native705-cubic-panel-independent/attempt001',source='5f6e80313d7dc7f02eef5a0e62b3243fe1396708',outer='a0a9789cce6df7cef235401a9341257bc57e34d9546421811b96196e3564263b',worker='0acda2263797d2b8b8901ccebdc85ac7e4c70f6f68516fe978da6413795c8801')
DIAG=dict(root=str(V.OUT.parent),source='f42b3f5af085d376b72498996ee6f13b08f12706',outer='c6ee28574fb2c28a033d4ae457e9c7ecff4e77680c30c53dca5bd8cbc0deb4c9',worker='e04355589b063a7bf6b42277708c7ed3157b7951d10a284cb11c0a81b072b771')
FILES=sorted(set(V.FILES+['src/neural_shrimp/cubic_complement_narrow.py','tools/probe_native705_cubic34_narrow.py','tools/launch_native705_cubic34_narrow.py','tests/test_native705_cubic34_narrow.py','docs/plans/native705-cubic34-narrow.md',CARD,'docs/reviews/native705-cubic34-diagnosis001.json']))

def card_check(c):
 need(c['status']=='reviewed'and c['chart']==read(ROOT/P.CARD)['charts'][1]and c['chart']['ordinal']==34,'literal reviewed existing whole34')
 need(c['caps']==CAPS and c['domains']==H.partition()and c['maximum_centers']==192 and c['maximum_depth']==1,'full64/192 finite cover')
 need(c['independent_previous']==INDEPENDENT and c['diagnosis']==DIAG and c['wide_failure']==read(ROOT/V.CARD)['producer'],'literal completed independent33 and all wide failures')

def bind(w,c):
 card_check(c);w.prefix='history/wide-failure';V.bind(read(ROOT/V.CARD),w)
 w.prefix='history';diag=S.authority(DIAG);need(diag['passed']and len(diag['scenarios'])==7 and not diag['scenarios'][0]['metric_passed'],'complete original negative metric diagnosis')
 out=S.authority(INDEPENDENT);ir=Path(INDEPENDENT['root'])/'run';authority=read(ir/'authority-binding.json');initial=read(ir/'initial-authority.json');union=read(ir/'charts/033/union.json');idx=G.M.select(read(ROOT/G.M.CARD),33)
 need(len(out['completed_charts'])==1 and out['completed_charts'][0]['chart']==33 and out['completed_charts'][0]['original_root']==idx and out['completed_charts'][0]['union']==union and union==dict(passed=True,endpoints=idx['parameter_endpoints'],total_cells=39,complement_dimension=11),'entire independently stable whole33')
 need(len(out['completed_cells'])==39 and all(v['passed']and v['complement_dimension']==11 for v in out['completed_cells'])and authority['producer']==c['wide_failure']and authority['retained_failed_attempts']==[71,72,73,74]and authority['retained_unqualified_chart']==c['chart'],'exact39 accepted leaves and four retained34 failures')
 need(authority['initial_independent']==read(ROOT/P.CARD)['initial_independent']and initial['previous']['complete_complement_endpoints']==['39/1024','103/1024'],'complete historical known-root genealogy')
 previous=dict(index=idx,complete_complement_endpoints=idx['parameter_endpoints'],independent=INDEPENDENT)
 data=G.metadata(c['chart'],previous,w.tick);w.save('authority-binding.json',dict(independent_previous=INDEPENDENT,previous=previous,wide_failure=c['wide_failure'],diagnosis=DIAG,domain_strategy='Fresh64 actual derivative domains, depth1 only after complete negative full11 metric; no transferred chart33 acquisitions.',original_root=data['index']))
 return previous

def measured(c):
 root=Path(c['wide_failure']['root']);O.verify(root,c['wide_failure']['outer']);O.verify(root/'run',c['wide_failure']['worker']);D.B.archived(root/'run',c['wide_failure']['source'],'runtime.json','source');m=read(root/'run/manifest.json')['files'];out=read(root/'run/outcome.json');owner=read(root/'launcher-result.json')
 need(owner['exit_code']==2 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent']and out['native_calls']==10278 and out['domains_done']==9 and len(out['completed_attempts'])==74,'entire actual completed cold-work reference including all failures')
 groups={}
 for name,pin in m.items():
  parts=name.split('/')
  if len(parts)>=5 and parts[0]=='charts'and parts[2]in('domains','cells'):
   key='/'.join(parts[:4]);groups[key]=groups.get(key,0)+pin['bytes']
 domain=[v for k,v in groups.items()if '/domains/'in k];cell=[v for k,v in groups.items()if '/cells/'in k]
 need(len(domain)==9 and len(cell)==74,'all actual domain/cell byte histories')
 return dict(seconds=out['elapsed_seconds'],domains=9,centers=74,calls=10278,max_domain_bytes=max(domain),max_cell_bytes=max(cell),raw_bytes=sum(v['bytes']for v in m.values()))

def forecast(seconds,size,elapsed,peak,cost):
 # Separate category byte costs; conservatively charge the entire cold runtime
 # against the larger category expansion, including unused historical work.
 ratio=max(64/cost['domains'],192/cost['centers']);wall=elapsed+max(2*ratio*cost['seconds'],2*192*seconds)+2*61.466+60
 raw=size+2*(64*cost['max_domain_bytes']+192*cost['max_cell_bytes'])+256*1024**2
 return dict(passed=wall<CAPS['seconds']and raw<CAPS['raw_bytes']and peak+512*1024<4*1024**2,forecast_seconds=wall,forecast_bytes=raw,maximum_domains=64,maximum_centers=192,maximum_total_calls=30096,cold_history=cost,profile_seconds=seconds,target_arrays_unopened=True)

class Work(S.Work):
 output_dir=OUT;maximum_calls=30096;raw_cap=32*1024**3
 def __init__(self):super().__init__();self.domains_done=0;self.pending=[]
 def tick(self):need(not self.stopped and time.monotonic()-self.start<12600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'12600s4GiB whole34 narrow-domain worker')

def target(w,c):
 previous=bind(w,c);w.prefix='context';context=G.bind(w,c['chart'],previous);domains=c['domains']
 for dom in domains:
  H.reserve(w.calls,w.domains_done,len(w.attempts),'domain');w.prefix=f"domains/{dom['index']:03d}";qi,tan,derivatives=C.derivative_domain(w,dom,context,domains);w.domains_done+=1;w.pending=[C.H.cell(domains,dom['index'])]
  while w.pending:
   H.reserve(w.calls,w.domains_done,len(w.attempts),'cell');cell=w.pending[0];number=len(w.attempts)+1;w.prefix=f'cells/{number:03d}';w.save('cell-input.json',cell);out=C.fresh_cell(w,cell,context,qi,tan,derivatives,domains);w.save('cell-outcome.json',out);w.attempts.append(out);w.prefix='';w.save(f'completed-attempt-{number:03d}.json',out)
   if out['passed']:w.accepted.append(out);w.pending=w.pending[1:]
   else:w.pending=H.children(domains,cell,out)+w.pending[1:]
   gc.collect()
 union=H.complete(domains,w.accepted);need(w.domains_done==64 and w.calls==144+90*64+126*len(w.attempts)<=30096 and not w.pending,'complete declared full34 phase/union accounting');w.prefix='';w.save('union.json',union)
 return dict(passed=True,status='whole34_narrow_domains_complement_covered',union=union,scope='Producer full11 complementary stability on entire original cubic34, joined to independently qualified whole33. New outcomes require independent replay. All original wide-domain failures retained; neutral critical pair remains, no full-window claim.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);c=read(ROOT/CARD);card_check(c);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,12600)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:O.entry(ROOT/n)for n in FILES},precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,caps=CAPS,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact source');need(w.used+len(raw)<w.raw_cap-8*1024**2,'bounded archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   begin=time.monotonic();C.synthetic(w);seconds=time.monotonic()-begin;cost=measured(c);report=forecast(seconds,w.used,time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cost);w.save('batch-profile.json',report);need(report['passed'],'entire64domain192center measured cold-cost admission before arrays');answer=target(w,c);w.tick()
 except BaseException as exc:answer.update(passed=False,status='failed_or_incomplete',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);w.prefix='';answer.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),domains_done=w.domains_done,completed_attempts=w.attempts,accepted_cells=w.accepted,pending_cells=w.pending,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
