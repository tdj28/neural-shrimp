"""Bounded saved-only initial arclength pair and fixed-gauge batch review."""
import gc,gzip,hashlib,json,math,os,platform,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from flint import ctx,arb
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json
import review_joined_period18_math as M
import review_arclength_connection_math as A
from review_quadratic_connection_math import matrix,exact,original_ball
BASE=Path('/data2/neural-shrimp-data/explore/certificates')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
PAIR=('s2-period18-arclength-connection-2026-10-10','11eb9a2dd76a00f592e1d1d28a52fc54cdcabc21','bdf1b2e61b6577d86bb16c4c5c9e11f2f92ce9e72119e4aa83e3a2e14fc39301','0460fd73c44e52b479538908e55df12c61a082e2f3541c3c54186bfd7750f71c')
BATCH=('s2-period18-arclength-batch-2026-10-10','ed708aa36f7ee93bb3b6b730c134dcfbb5bc736e','4af41a333670373f982f9a1d021ff30e9ee80a5e8fb75f13c7cb3ed3e06958e1','7baaf7af18611786403ef8e019f8f0c781ed47867183b6b59bdc55f1cf037a35')
FAILED_BATCH=('c802f6902529ae4ae0b3900594d2dd5b77f2304e620e2eb2632c17ca411ded81','8b65c16802b8560d241bf97bcea4eaab8308aa39843d71c7af233bf4e4cc9b2b')
PRIOR=Path('/data2/neural-shrimp-data/audits/s2-period18-weighted-connection/attempt-002');PRIOR_SHA='c62a15eeda71f73356b5e082e92215a386240ad84ff28f459d06c096ef3b9ad5'
SETTINGS={'precision':256,'halfwidth_exponent':-6,'radius_exponent':-22,'newton_steps':2}
FILES=['tools/reviews/review_arclength_connection.py','tools/reviews/review_arclength_connection_math.py','tools/reviews/review_quadratic_connection_math.py','tools/reviews/review_joined_period18_math.py','tools/reviews/launch_arclength_connection_review.py','tests/reviews/test_arclength_saved_independent.py','docs/plans/arclength-independent-review.md','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','certificates/s2-period18-receipt.json']
def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
  M.need(len(b)<=256*1024**2,'decompression cap');meta=read(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json'));M.need(meta=={'uncompressed_bytes':len(b),'uncompressed_sha256':hashlib.sha256(b).hexdigest()},'decompressed hash');return json.loads(b)
 return json.loads(p.read_bytes())
def verify(root,pin):
 M.need(entry(root/'manifest.json')['sha256']==pin,'manifest pin');m=read(root/'manifest.json')['files'];paths={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};M.need(set(paths)==set(m),'complete inventory')
 for n,p in paths.items():M.need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==m[n],'immutable bytes '+n)
 return {'manifest_sha256':pin,'files':len(paths),'bytes':sum(v['bytes']for v in m.values())}
def source_bind(run,commit):
 rt=read(run/'runtime-start.json');M.need(rt['source_commit']==commit,'execution identity')
 for n,h in rt['source_sha256'].items():M.need(entry(run/'source'/n)['sha256']==h and hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==h,'committed source '+n)
 M.need(rt['python_flint']==flint.__version__ and rt['flint']==flint.__FLINT_VERSION__ and rt['python']==platform.python_version(),'same arithmetic libraries');M.need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10 and rt['cpu_only']and not rt['paid'],'runtime scope');return rt

def bind_unit(unit,count,tick):
 name,commit,outer,worker=unit;root=BASE/name/('attempt-002-target'if unit==BATCH else'attempt-001-target');run=root/'run';inv=verify(root,outer);verify(run,worker);rt=source_bind(run,commit);done=read(run/'outcome.json');la=read(root/'launcher-result.json')
 M.need(la['source_commit']==commit and la['source_sha256']==rt['source_sha256']and la['exit_code']==done['exit_code']==0 and not la['owned_live_pids'],'completed producer')
 for pid in (rt['pid'],la['launcher_pid'],la['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'producer PID absent')
 for v,b in ((done['elapsed_seconds'],rt['internal_seconds']),(done['max_rss_kib'],2*1024**2),(la['elapsed_seconds'],la['external_seconds']+7)):M.need(type(v)in(int,float)and math.isfinite(v)and 0<=v<b,'bounded resources')
 M.need(done['adapter_counts']=={'map_attempted':72*count,'map_returned':72*count,'unit_attempted':144*count,'unit_returned':144*count,'map_cap':72*count,'unit_cap':144*count}and done['jet_counts']=={'attempted':54*count,'returned':54*count,'cap':54*count},'complete fixed acquisition counts')
 cost=read(run/'cost-gate.json');profile=root.parent/('attempt-002-profile/run'if unit==BATCH else'attempt-001-profile/run');verify(profile,cost['manifest_sha256']);pr=read(profile/'runtime-start.json');pd=read(profile/'outcome.json')
 M.need(pr['source_sha256']==rt['source_sha256']and pr['mode']=='profile'and pd['exit_code']==0,'profile source/completion');M.need(all(pr[k]==rt[k]for k in('python','python_flint','flint','numpy','scipy','threads'))and pr['nice']>=10,'profile library equality')
 for v,b in ((pd['elapsed_seconds'],20),(pd['max_rss_kib'],2*1024**2)):M.need(type(v)in(int,float)and math.isfinite(v)and 0<=v<b,'profile resources')
 for path in run.glob('*.json.gz'):
  h=hashlib.sha256();size=0
  with gzip.open(path,'rb')as f:
   while b:=f.read(1024**2):h.update(b);size+=len(b);M.need(size<=256*1024**2,'payload cap')
  M.need(read(path.with_name(path.name.removesuffix('.json.gz')+'-payload.json'))=={'uncompressed_bytes':size,'uncompressed_sha256':h.hexdigest()},'all payload identities');tick()
 return run,rt,done,inv

def initial_binding(inp):
 verify(PRIOR,PRIOR_SHA);oldreview=read(PRIOR/'run/outcome.json');M.need(oldreview['passed'],'independently reviewed weighted parent')
 for name,e in read(PRIOR/'run/runtime-start.json')['sources'].items():M.need(entry(PRIOR/'run/source'/name)==e,'retained prior-review source')
 parent=BASE/'s2-period18-weighted-connection-2026-10-10/attempt-001-target/run';pin='0d7bb58ead76db13278d1f338b8d0b5896bb3c8dcc48281d9b8e71ef317efd9c';verify(parent,pin)
 old=read(parent/'certificate.json.gz')['result'];M.need(old['passed'],'proved weighted root');last=old['charts'][-1];endpoint=last['exact_parameter_endpoints'][1]
 original=read(ROOT/'certificates/s2-period18-receipt.json');M.need(entry(ROOT/'certificates/s2-period18-receipt.json')['sha256']=='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277','original receipt');model_path=Path(original['external_root'])/'inputs.json';M.need(entry(model_path)['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f','original model');M.need(inp['model_inputs']==read(model_path),'exact original model')
 with ctx.workprec(256):
  box=last['endpoint_root_enclosures'][1]+[ball_json(exact(endpoint))];initial=[str(M.pair(v['mid']))for v in box];s=matrix(last['predictor_slope']);t=matrix(last['predictor_curvature']);delta=exact('1/64');v=[(s[i,0]+t[i,0]*delta).mid()for i in range(234)]+[arb(1)];sq=sum((x*x for x in v),arb(0));normal=[str(M.pair(ball_json((x/sq).mid())['mid']))for x in v]
 M.need(inp['period']==18 and inp['initial']==inp['origin']==initial and inp['normal']==normal and inp['weights']==last['exact_weights']+['1']and inp['anchor']=={'state_box':box},'exact weighted endpoint/gauge reconstruction')
 prior=read(PRIOR/'run/weighted-arithmetic.json');M.need(prior['passed']and prior['exact_parameter_endpoints'][-1]==endpoint,'bound reviewed endpoint');fresh=prior['charts'][-1]['independent_higher_precision']['recomputed_endpoints'][1]+[box[-1]]
 del old,last;gc.collect();return {'state_box':fresh}

def main(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];result={'passed':False,'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<16*1024**2,'16MiB cap');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped[0]and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'time/RSS cap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in FILES},'pid':os.getpid(),'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  failed=BASE/BATCH[0]/'attempt-001-target';verify(failed,FAILED_BATCH[0]);verify(failed/'run',FAILED_BATCH[1]);failure=read(failed/'run/outcome.json');M.need(failure['exit_code']==2 and failure['ticks']==failure['stages']==0 and failure['error']=={'type':'ValueError','message':'bounded map calls'},'preserved pre-acquisition wrapper failure');result['preserved_producer_failure']=failure
  run,rt,done,inv=bind_unit(PAIR,2,tick);inp=read(run/'inputs.json');fresh_anchor=initial_binding(inp);pair=read(run/'certificate.json.gz');M.need(pair['inputs_sha256']==entry(run/'inputs.json')['sha256']and pair['runtime']==rt and pair['result']['passed'],'pair envelope');charts=pair['result']['charts'];M.need(len(charts)==2,'initial pair');normal,origin,weights=inp['normal'],inp['origin'],inp['weights'];previous=None;fresh_previous=None;index=0
  def consume(c,index,original_anchor,fresh_anchor):
   M.need(c['settings']==SETTINGS and c['statement']['period']==18 and c['statement']['phase_dimension']==13 and c['parameter_center']==str(Fraction(index,32)),'fixed chart schedule/settings')
   M.need(c['anchor']==original_anchor,'complete saved prior anchor identity')
   original=A.chart_check(c,original_anchor,normal,origin,weights,256);tick();fresh=A.chart_check(c,fresh_anchor,normal,origin,weights,384);tick();save(f'chart-{index:03d}.json',{'original':original,'independent384':fresh})
   M.need(original['passed']and fresh['passed'],'qualified native chart and join');result['charts'].append({'index':index,'exact_gauge_endpoints':c['exact_parameter_endpoints'],'contraction':fresh['contraction'],'max_image_ratio':fresh['max_image_ratio'],'hidden_rate_endpoint_enclosures':[e[-1]for e in fresh['fresh_endpoints']],'passed':True});return fresh
  for c in charts:
   original_anchor=inp['anchor']if index==0 else {'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]}
   if index:fresh_anchor={'state_box':fresh_previous['fresh_endpoints'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]}
   fresh_previous=consume(c,index,original_anchor,fresh_anchor);previous=c;index+=1
  # Keep only native endpoint metadata before releasing the two large graphs.
  previous={k:previous[k]for k in('endpoint_root_enclosures','exact_parameter_endpoints','normal','origin','settings','center','parameter_center','predictor_slope','predictor_curvature','exact_weights','statement','passed','gates')};del pair,charts,c;gc.collect()
  br,brt,bd,bi=bind_unit(BATCH,16,tick);binp=read(br/'inputs.json');summary=read(br/'batch.json');M.need(binp['model_inputs']==inp['model_inputs']and binp['previous']==previous,'same previous native chart/model');M.need(summary['passed']and summary['completed_charts']==summary['qualified_charts']==summary['requested_charts']==16,'complete fixed batch')
  for row in summary['charts']:
   artifact=row['artifact'];path=br/artifact['path'];M.need(entry(path)=={k:artifact[k]for k in('bytes','sha256')}and read(path.with_name(path.name.removesuffix('.json.gz')+'-payload.json'))==artifact['payload'],'summary proof identity');c=read(path)
   original_anchor={'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]};fresh_anchor={'state_box':fresh_previous['fresh_endpoints'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]}
   fresh_previous=consume(c,index,original_anchor,fresh_anchor);M.need(row['index']==index-2 and row['passed']and row['gates']==c['gates']and row['exact_parameter_endpoints']==c['exact_parameter_endpoints'],'summary scientific identity');previous={k:c[k]for k in('endpoint_root_enclosures','exact_parameter_endpoints')};del c;index+=1;gc.collect();tick()
  M.need(summary['new_exact_gauge_endpoints']==['3/64',previous['exact_parameter_endpoints'][1]],'batch coverage');result.update(passed=True,chart_count=index,exact_gauge_endpoints=['-1/64',previous['exact_parameter_endpoints'][1]],pair_inventory=inv,batch_inventory=bi,scope='same anchored primitive18 branch in one exact affine gauge; no attraction, monotonic hidden-rate, fold identity or full shrimp claim')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps({k:v for k,v in result.items()if k!='charts'}),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
