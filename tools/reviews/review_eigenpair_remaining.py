"""One pinned saved-evidence unit beyond the independently joined first20 charts.

Independent native assembly and256/384 reductions; shared analytic phase/path
acquisitions and Arb. No producer proof routine or new model acquisition.
"""
import argparse,gc,gzip,hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from flint import ctx
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_eigenpair_path_math as E
from neural_shrimp.rigorous_orbits import canonical_bytes
need=E.need
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
BASE=Path('/data2/neural-shrimp-data/audits/eigenpair-remaining-locus')
MAP_INPUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-003-quadratic-target/run/exact-map-inputs.json')
MAP_SHA='16fc4b1c5ea4e7cb30fe321ca48147c209ecba91fca60872c1583de1a3a81f0a'
KEYS=('center','curvature','dimension','endpoint_root_boxes','gates','normal','origin','parameter_center','parameter_endpoints','passed','period','schema','settings','shooting_dimension','sigma','slope','weights')
def compact(c):return {k:c[k]for k in KEYS}
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
  need(len(b)<=256*1024**2,'bounded decompression');return json.loads(b)
 return json.loads(p.read_bytes())
def verify(root,pin):
 need(entry(root/'manifest.json')['sha256']==pin,'fixed complete manifest')
 expected=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 need(set(actual)==set(expected),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink() and not p.stat().st_mode&0o222 and entry(p)==expected[n],'immutable inventory member '+n)
def source_bind(run,execution):
 rt=read(run/'runtime.json')
 need(rt['commit']==execution and rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'exact native producer settings')
 for n,row in rt['sources'].items():
  need(entry(run/'source'/n)==row and hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical archived source '+n)
 return rt
def event_group(run,selected):
 """Only the full attempted chart immediately preceding this exact return."""
 stop=run/selected;need(stop.parent==run/'events' and stop.name.endswith('-chart_return.json.gz'),'native full chart return')
 events=[];returns=[]
 for p in sorted((run/'events').glob('*.json.gz')):
  if p.name>stop.name:break
  kind=p.name.split('-',1)[1].removesuffix('.json.gz')
  if kind=='chart_return':
   returns.append(str(p.relative_to(run)))
   if p==stop:return events,returns
   events=[]
  elif kind in ('phase_return','path_phase_return','path_assembly_return'):events.append((kind,read(p)))
 raise ValueError('missing exact full chart return')
def same_map(actual,known):
 need(set(known)=={'schema','network','x','y','lr_hidden','lr_output'}and actual==known,'exact independently audited two-rate map input')
def joined(previous,chart):
 need(chart['normal']==previous['normal'] and chart['origin']==previous['origin'],'identical native gauge')
 need(chart['parameter_endpoints'][0]==previous['parameter_endpoints'][1],'exact shared endpoint')
 need(all(chart[k]==previous[k]for k in ('period','dimension','sigma','shooting_dimension')),'same full native convention')
 return {'box':previous['endpoint_root_boxes'][1],'exact_parameter':previous['parameter_endpoints'][1]}
def worker(out,specfile):
 need(specfile.parent==ROOT/'docs/plans'and specfile.name=='eigenpair-remaining-unit000.json','first fixed saved review unit only')
 spec=read(specfile);need(spec['index']==0 and len(spec['selected'])==1,'one previously unaudited repaired chart')
 need(out==BASE/'unit000-attempt002/run','exclusive first audit output')
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
 files=sorted(set([str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]+[str(specfile.relative_to(ROOT)),'tools/reviews/launch_eigenpair_remaining.py','tests/reviews/test_eigenpair_remaining.py','tests/reviews/test_eigenpair_path_review.py','docs/plans/eigenpair-remaining-independent.md','docs/reviews/eigenpair-remaining-helper-provenance.json']))
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'charts':[]};resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
 def save(n,v):
  b=canonical_bytes(v);need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<24*1024**2,'24MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0]and time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'180s/3GiBcap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,180)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in files},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1'),'precision':[256,384],'model_calls':0})
 for n in files:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  prior=Path(spec['prior_root']);verify(prior,spec['prior_manifest']);need(read(prior/'run/outcome.json')['passed'],'complete independent predecessor')
  endpoint=read(prior/spec['prior_endpoint']);previous=endpoint['original'];fresh=endpoint['independent384'];need(fresh['exact_parameter']==previous['parameter_endpoints'][1]=='19/128','exact prior independent endpoint')
  root=Path(spec['root']);run=root/'run';verify(root,spec['outer']);verify(run,spec['worker']);rt=source_bind(run,spec['execution']);la=read(root/'launcher-result.json');done=read(run/'outcome.json');inp=read(run/'inputs.json')
  need(la['commit']==spec['execution']and la['exit_code']==0 and la['owned_live_pids']==[]and la['stop_reason']is None and done['passed']is spec['outcome_passed'],'exact complete producer unit')
  need(inp['parent']==previous,'complete original native predecessor')
  need(entry(MAP_INPUT)['sha256']==MAP_SHA,'exact independently audited eigenpair map file');same_map(inp['map_inputs'],read(MAP_INPUT))
  need(not Path('/proc',str(rt['pid'])).exists(),'producer absent');gc.collect();tick()
  for index,row in enumerate(spec['selected']):
   p=run/row['path'];need(entry(p)==row['identity'],'exact saved full witness');c=read(p);need(c['parameter_endpoints']==row['expected_endpoints']and c['settings']==row['settings'],'fixed chart and settings')
   known=joined(previous,c);events,attempts=event_group(run,row['path']);r256=E.check_chart(c,known,events,256,expected_settings=row['settings']);save(f'chart{index}-256.json',r256);tick()
   r384=E.check_chart(c,fresh,events,384,expected_settings=row['settings']);save(f'chart{index}-384.json',r384);tick();need(r256['passed']and r384['passed'],'independent native root primitive and full join')
   result['charts'].append({'global_chart':spec['first_global_chart']+index,'exact_endpoints':c['parameter_endpoints'],'passed':True,'contraction384':r384['contraction'],'coordinate_inclusions':r384['inclusions'],'earlier_attempt_returns':attempts[:-1]})
   previous=compact(c);fresh={'box':r384['recomputed_endpoints'][1],'exact_parameter':c['parameter_endpoints'][1]};del c,events,r256,r384;gc.collect();tick()
  save('endpoint.json',{'original':previous,'independent384':fresh});result.update(passed=True,total_connected_charts=21,scope='Independent saved native assembly,256 replay and384 full470 root/primitive/endpoint join. Shared analytic phase/path acquisitions and Arb. No generic unfolding or attraction claim.')
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out,a.spec))
