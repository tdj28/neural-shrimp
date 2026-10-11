"""One bounded sequential complementary cover of two existing cubic root charts."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import probe_native705_cubic_remaining as R
import bind_native705_cubic_panel as G
from neural_shrimp import cubic_complement_panel as H
from flint import ctx
C,S,O,D,need,read=R.C,R.S,R.O,R.D,R.need,R.read
OUT=R.OUT.parent.with_name('cubic-panel001')/'run';CARD='docs/plans/native705-cubic-panel-inputs.json'
CAPS=dict(seconds=10800,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=24*1024**3,profile_calls=144,maximum_target_calls=13536,maximum_total_calls=13680)
MEASURED=dict(root=str(R.OUT.parent),source='6920efa4ebc6570ea131eb91b6907435d78dcd76',outer='987a2d872dca54262bf49705b6abaaaa6e8ed56d465092e46780a8944cea41e0',worker='43f215c51e2ae8ef07458afdb95063835c16ffbab1fd7aac9a83c67913b06d00')
FILES=sorted(set(R.FILES+['tools/bind_native705_cubic_panel.py','tools/probe_native705_cubic_panel.py','tools/launch_native705_cubic_panel.py','src/neural_shrimp/cubic_complement_panel.py','tests/test_cubic_complement_panel.py','tests/test_native705_cubic_panel.py','docs/plans/native705-cubic-panel.md',CARD,'docs/reviews/native705-cubic32-remaining001.json']))


def card_check(card):
 need(card['status']=='reviewed'and [v['ordinal']for v in card['charts']]==[33,34]and card['caps']==CAPS,'literal reviewed two-chart packet')
 need(card['maximum_centers']==96 and card['maximum_domains']==16 and card['maximum_depth']==3,'bounded global adaptation')
 spec=card['initial_independent'];need(set(spec)=={'root','source','outer','worker'}and all(type(v)is str and v for v in spec.values()),'complete independent entire chart32 prerequisite')
 index=read(ROOT/G.M.CARD)
 for row in card['charts']:
  need(row['root_index']==G.M.select(index,row['ordinal'])and row['parameter_endpoints']==row['root_index']['parameter_endpoints'],'literal independently proved root index')
 need(sum(len(H.partition(v['parameter_endpoints']))for v in card['charts'])==16 and H.call_cap(16,96)==CAPS['maximum_total_calls'],'exact full domains/inclusive cap')


def initial(w,card):
 spec=card['initial_independent'];out=S.authority(spec);ir=Path(spec['root'])/'run';binding=read(ir/'authority-binding.json');union=read(ir/'union.json');index=G.M.select(read(ROOT/G.M.CARD),32)
 need(out['passed']and out['native_calls']==0 and out['complement_dimension']==11 and out['whole_chart_endpoints']==index['parameter_endpoints']and union['whole_chart_endpoints']==index['parameter_endpoints'],'complete independently attracting chart32 quotient')
 need(binding['original_root']==index and binding['producer']==MEASURED and binding['first_independent']==R.read(ROOT/R.CARD)['first_independent'],'literal entire known-root and first/remaining independent genealogy')
 need(out['parameter_endpoints']==['55/1024','103/1024']and out['total_cells']==18 and len(out['completed_cells'])==18 and all(v['passed']and v['complement_dimension']==11 for v in out['completed_cells']),'all18 independent remaining leaves')
 first=S.authority(binding['first_independent']);need(first['passed']and first['parameter_endpoints']==['39/1024','55/1024']and first['total_cells']==4,'independent full first portion')
 previous=dict(index=index,complete_complement_endpoints=index['parameter_endpoints'],independent=spec)
 w.save('initial-authority.json',dict(previous=previous,independent_outcome=out,first_independent=binding['first_independent'],first_outcome=first));return previous


def measured():
 p=Path(MEASURED['root']);O.verify(p,MEASURED['outer']);O.verify(p/'run',MEASURED['worker']);D.B.archived(p/'run',MEASURED['source'],'runtime.json','source');owner=read(p/'launcher-result.json');out=read(p/'run/outcome.json')
 need(owner['exit_code']==0 and owner['failure']is None and out['passed']and out['native_calls']==4464,'entire measured six-domain30-center cubic pipeline')
 old=read(p/'run/batch-profile.json');need(old['passed']and old['maximum_total_calls']==12024,'retained conservative wholepacket admission');return out,old,sum(v['bytes']for v in read(p/'run/manifest.json')['files'].values())


def forecast(seconds,size,elapsed,peak,out,old,raw):
 ratio=13680/4464;upperratio=13680/12024
 wall=max(elapsed+2*ratio*out['elapsed_seconds']+3*61.466+60,upperratio*old['forecast_seconds']+2*seconds+60)
 storage=max(size+2*ratio*raw+128*1024**2,upperratio*old['forecast_bytes']+size+64*1024**2)
 return dict(passed=wall<10800 and storage<24*1024**3 and peak+512*1024<4*1024**2,forecast_seconds=wall,forecast_bytes=storage,maximum_total_calls=13680,maximum_domains=16,maximum_centers=96,actual_complete_cubic_cost=dict(seconds=out['elapsed_seconds'],native_calls=4464,bytes=raw),target_arrays_unopened=True)


class Work(S.Work):
 output_dir=OUT;maximum_calls=13680;raw_cap=24*1024**3
 def __init__(self):super().__init__();self.domains_done=0;self.completed_charts=[];self.pending=[]
 def tick(self):need(not self.stopped and time.monotonic()-self.start<10800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'10800s4GiB bounded cubic panel')


def target(w,card):
 previous=initial(w,card)
 for spec in card['charts']:
  ordinal=spec['ordinal'];base=f'charts/{ordinal:03d}';w.prefix=base+'/context';context=G.bind(w,spec,previous);domains=H.partition(spec['parameter_endpoints']);local=[]
  for domain in domains:
   H.reserve(w.calls,w.domains_done,len(w.attempts),'domain',16,96);w.prefix=base+f'/domains/{domain["index"]:03d}';qi,tan,derivatives=C.derivative_domain(w,domain,context,domains);w.domains_done+=1;w.pending=[C.H.cell(domains,domain['index'])]
   while w.pending:
    H.reserve(w.calls,w.domains_done,len(w.attempts),'cell',16,96);cell=w.pending[0];number=len(w.attempts)+1;w.prefix=base+f'/cells/{number:03d}';w.save('cell-input.json',cell);out=C.fresh_cell(w,cell,context,qi,tan,derivatives,domains);w.save('cell-outcome.json',out);record=dict(chart=ordinal,ordinal=number,**out);w.attempts.append(record);w.prefix='';w.save(f'completed-attempt-{number:03d}.json',record)
    w.pending=w.pending[1:]
    if out['passed']:local.append(out);w.accepted.append(record)
    else:w.pending=C.H.children(domains,cell,out)+w.pending
    gc.collect()
  covered=H.chart_complete(domains,local);need(covered['endpoints']==spec['parameter_endpoints'],'complete original chart before advancing');record=dict(chart=ordinal,root_index=spec['root_index'],union=covered,predecessor=previous);w.prefix='';w.save(f'complete-chart-{ordinal:03d}.json',record);w.completed_charts.append(record);previous=dict(index=spec['root_index'],complete_complement_endpoints=covered['endpoints'],producer_prefix_ordinal=ordinal);del context,qi,tan,derivatives;gc.collect()
 need(len(w.completed_charts)==2 and not w.pending and w.calls==144+90*w.domains_done+126*len(w.attempts)<=13680,'complete bounded two-chart panel')
 return dict(passed=True,status='two_cubic_charts_complement_covered',scope='Producer full11 complementary stability on whole independently connected cubic33 and34, joined to independently qualified whole32. Independent replay pending; neutral critical pair retained. No final-junction or complete-window claim.')


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);card=read(ROOT/CARD);card_check(card);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,status='incomplete');signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,10800)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:O.entry(ROOT/n)for n in FILES},precision=480,python=D.platform.python_version(),numpy=D.np.__version__,python_flint=D.flint.__version__,flint=D.flint.__FLINT_VERSION__,caps=CAPS,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(O.THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<w.raw_cap-8*1024**2,'bounded source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   started=time.monotonic();C.synthetic(w);seconds=time.monotonic()-started;old,profile,raw=measured();report=forecast(seconds,w.used,time.monotonic()-w.start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,old,profile,raw);w.save('batch-profile.json',report);need(report['passed'],'complete96centers16domains measured admission before arrays');answer=target(w,card);w.tick()
 except BaseException as e:answer.update(passed=False,status='failed_or_incomplete',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);w.prefix='';answer.update(native_calls=w.calls,profile_native_calls=min(w.calls,144),target_native_calls=max(0,w.calls-144),domains_done=w.domains_done,completed_attempts=w.attempts,accepted_cells=w.accepted,completed_charts=w.completed_charts,pending_cells=w.pending,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):O.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
