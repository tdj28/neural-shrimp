"""Independent continuation of the reviewed470D saved-evidence audit."""
import gc,gzip,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_eigenpair_path as B
E=B.E;O=B.O;need=E.need;read=B.read
from neural_shrimp.rigorous_orbits import canonical_bytes
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype')
PRIOR=Path('/data2/neural-shrimp-data/audits/eigenpair-quadratic-path/attempt-003');PRIOR_PIN='ff9f38c0e3e769ef9a45e96d761cbc1313a0e3b1bb15711fbd57e679f7efbbae'
UNITS=[('b50d587bc6f4bb12f0832b7ce33509ff90348f8d','422a2948f044bb0beffe6697a813e26c684d401f1303502ab1add24528cc6c14','cdaa9e927fa709ec022dc74dda64f23995a72db2bade379383993192d6491fe7'),('692943933a061c718bfb8ccdb8d17c8a5c99feb8','6984dff87e855a4fa43d036358688ebf536d9feac88bbe1a0fcdc2d0fc9a2715','f1bb8a6acc3b6b50e53d71ef8fa57be1ea1c9035f3dfbec8d5d6cacc1c8a2cad')]
FILES=B.FILES+['tools/reviews/review_eigenpair_march.py','docs/plans/eigenpair-march-independent-audit.md']
def compact(c):return {k:c[k] for k in ('center','curvature','dimension','endpoint_root_boxes','gates','normal','origin','parameter_center','parameter_endpoints','passed','period','schema','settings','shooting_dimension','sigma','slope','weights')}
def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed auditor');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10')
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);need(len(b)+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<24*1024**2,'24MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'180s/3GiBcap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,180)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1')})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  O.verify(PRIOR,PRIOR_PIN);need(read(PRIOR/'run/outcome.json')['passed'],'prior audited native pair')
  O.verify(B.RAW,B.OUTER);initial=read(B.RAW/'run/result.json.gz')['charts'][-1];previous=compact(initial);fresh=read(PRIOR/'run/endpoint.json');del initial;gc.collect();tick()
  for unit,(execution,outer,workerpin) in enumerate(UNITS,1):
   root=BASE/f'attempt-{unit:03d}-march-target';run=root/'run';O.verify(root,outer);O.verify(run,workerpin);rt=read(run/'runtime.json');la=read(root/'launcher-result.json');done=read(run/'outcome.json');inp=read(run/'inputs.json');agg=read(run/'march.json')
   need(rt['commit']==la['commit']==execution and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None,'exact successful producer')
   need(done['passed'] and done['calls']==4 and not done['rebox_used'] and agg['completed'],'four complete unreboxed charts only')
   need(inp['parent']==previous and inp['map_inputs']==read(B.RAW/'run/exact-map-inputs.json'),'full immediate native parent and map')
   for name,v in rt['sources'].items():need(O.entry(run/'source'/name)==v and __import__('hashlib').sha256(subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'committed source '+name)
   need(not Path('/proc',str(rt['pid'])).exists(),'owned worker terminated');tick()
   events=[];index=0
   for path in sorted((run/'events').glob('*.json.gz')):
    kind=path.name.split('-',1)[1].removesuffix('.json.gz')
    if kind in ('phase_return','path_phase_return','path_assembly_return'):events.append((kind,read(path)))
    elif kind=='chart_return':
     c=read(path);record=done['progress']['accepted'][index];accepted=read(run/record['accepted_file'])
     need(record['full_chart_file']==str(path.relative_to(run)) and record['index']==index and record['call']==index and accepted['chart']==compact(c),'durable accepted/full witness')
     need(c['normal']==previous['normal'] and c['origin']==previous['origin'] and c['parameter_endpoints'][0]==previous['parameter_endpoints'][1],'identical native gauge/common endpoint')
     known={'box':previous['endpoint_root_boxes'][1],'exact_parameter':previous['parameter_endpoints'][1]}
     r256=E.check_chart(c,known,events,256);save(f'unit{unit}-chart{index}-256.json',r256);tick();r384=E.check_chart(c,fresh,events,384);save(f'unit{unit}-chart{index}-384.json',r384);tick()
     need(r256['passed'] and r384['passed'],'fresh strict470D proof and full endpoint join')
     result['charts'].append({'unit':unit,'index':index,'right':c['parameter_endpoints'][1],'contraction384':r384['contraction'],'inclusions':r384['inclusions']})
     previous=compact(c);fresh={'box':r384['recomputed_endpoints'][1],'exact_parameter':c['parameter_endpoints'][1]};index+=1;events=[];del c,r256,r384;gc.collect()
   need(index==4 and not events and agg['last_chart']==previous,'complete unit prefix');save(f'unit{unit}-endpoint.json',{'original':previous,'independent384':fresh});tick()
  result.update(passed=True,total_connected_charts=10,exact_gauge=['-1/32768','19/32768'],scope='Same independently audited normalized primitive18 +1 native eigenpair locus. Independent saved assembly and256/384 root/full join reduction; shared analytic derivatives and Arb. No generic fold, cusp, complementary stability or complete shrimp claim.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
