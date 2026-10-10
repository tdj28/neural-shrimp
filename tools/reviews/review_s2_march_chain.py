"""Bounded independent saved-arithmetic audit, one exact producer unit at a time.

No new model, jet, Newton or decomposition acquisition. Original analytic
Jacobians/path jets remain trusted inputs; the independent reducer is unchanged.
"""
import argparse,gc,json,math,os,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_arclength_connection_math as A
import review_joined_period18_math as M
from neural_shrimp.rigorous_orbits import canonical_bytes
from neural_shrimp.rigorous_arclength_batch import compact_chart
CONFIG='docs/plans/s2-march-chain-review-inputs.json'
CARD='docs/plans/s2-march-chain-review.md'
BASE=Path('/data2/neural-shrimp-data/audits/s2-period18-march-chain')
INITIAL=Path('/data2/neural-shrimp-data/audits/s2-period18-arclength-connection/attempt-001')
INITIAL_SHA='5afa111cec0d00ac8ad34d6c51d99d740798ac80d2274065aac58fa6f7092101'
OLD_PRODUCER=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-arclength-batch-2026-10-10/attempt-002-target/run')
OLD_SHA='7baaf7af18611786403ef8e019f8f0c781ed47867183b6b59bdc55f1cf037a35'
FILES=list(dict.fromkeys(O.FILES+[CONFIG,CARD,'tools/reviews/review_s2_march_chain.py','tests/reviews/test_s2_march_chain.py','src/neural_shrimp/rigorous_arclength_batch.py']))

def validate_row(c,row,previous,index):
 M.need(row['index']==index and c['passed']==row['passed'] and c['gates']==row['gates'],'ordered mirrored chart verdict')
 M.need(c['statement']['period']==18 and c['statement']['phase_dimension']==13,'native primitive18 dimensions')
 st=c['settings'];M.need(st['precision']==256 and st['halfwidth_exponent']==-6 and st['newton_steps']==2 and -100<=st['radius_exponent']<=-10,'bounded original chart settings')
 M.need((c['normal'],c['origin'])==(previous['normal'],previous['origin']),'unchanged exact gauge')
 M.need(c['anchor']=={'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]},'whole native predecessor identity')
 M.need(c['exact_parameter_endpoints'][0]==previous['exact_parameter_endpoints'][1],'no skipped gauge interval')
 M.need(Fraction(c['parameter_center'])==Fraction(previous['parameter_center'])+Fraction(1,32),'fixed next center')
 M.need('reboxing' not in c,'separate decoder audit required for reboxing')

def pinned_chart(run,row):
 a=row['artifact'];name=f"chart-{row['index']:03d}.json.gz";M.need(a['path']==name,'fixed artifact name')
 p=run/name;M.need(O.entry(p)=={k:a[k]for k in('bytes','sha256')},'chart stored identity')
 M.need(O.read(p.with_name(p.name.removesuffix('.json.gz')+'-payload.json'))==a['payload'],'payload identity')
 return O.read(p)

def bind(unit,tick):
 root=Path(unit['root']);run=root/'run';O.verify(root,unit['outer_sha256']);O.verify(run,unit['worker_sha256']);tick()
 rt=O.source_bind(run,unit['execution']);end=O.read(run/'outcome.json');la=O.read(root/'launcher-result.json');batch=O.read(run/'batch.json')
 M.need(rt['mode']=='target' and la['source_commit']==unit['execution'] and la['source_sha256']==rt['source_sha256'] and not la['owned_live_pids'],'producer execution identity')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned producer PID absent')
 good=unit['qualified'];done=unit['completed'];M.need(1<=good<=done<=16 and done-good<=1,'exact qualified prefix length')
 M.need(batch['qualified_charts']==good and batch['completed_charts']==done and len(batch['charts'])==done and batch['new_exact_gauge_endpoints']==unit['new_gauge_endpoints'],'fixed prefix summary')
 M.need(end['exit_code']==la['exit_code']==(0 if done==good else 2) and batch['passed']==(done==good),'retained producer completion/failure')
 M.need(end['adapter_counts']==dict(map_attempted=72*done,map_returned=72*done,unit_attempted=144*done,unit_returned=144*done,map_cap=1152,unit_cap=2304),'complete derivative counts')
 M.need(end['jet_counts']==dict(attempted=54*done,returned=54*done,cap=864),'complete jet counts')
 for v,cap in((end['elapsed_seconds'],rt['internal_seconds']),(end['max_rss_kib'],2*1024**2),(la['elapsed_seconds'],la['external_seconds']+7)):
  M.need(type(v)in(int,float) and math.isfinite(v) and 0<=v<cap,'producer resource bound')
 argv=rt['argv'];profile=Path(argv[argv.index('--profile')+1]);pin=argv[argv.index('--profile-manifest')+1]
 M.need(profile.is_relative_to(Path('/data2/neural-shrimp-data/explore/certificates')),'local profile scope')
 O.verify(profile,pin);pr=O.read(profile/'runtime-start.json');pe=O.read(profile/'outcome.json');cost=O.read(run/'cost-gate.json')
 M.need(pr['source_sha256']==rt['source_sha256'] and pr['mode']=='profile' and pe['exit_code']==0 and cost['manifest_sha256']==pin,'profile source/completion binding')
 M.need(all(pr[k]==rt[k]for k in('python','python_flint','flint','numpy','scipy','threads')),'profile library identity')
 return run,rt,batch

def initial_anchor(index,prior_pin,units,tick):
 if index==0:
  M.need(prior_pin==INITIAL_SHA,'initial audit pin');O.verify(INITIAL,prior_pin);O.verify(OLD_PRODUCER,OLD_SHA)
  result=O.read(INITIAL/'run/outcome.json');M.need(result['passed'] and result['chart_count']==18,'reviewed initial18')
  old=O.read(OLD_PRODUCER/'batch.json');previous=pinned_chart(OLD_PRODUCER,old['charts'][-1]);previous=compact_chart(previous)
  fresh=O.read(INITIAL/'run/chart-017.json')['independent384'];count=18
 else:
  prior=BASE/f'unit-{index:03d}';O.verify(prior,prior_pin);result=O.read(prior/'run/outcome.json')
  M.need(result['passed'] and result['unit_index']==index-1 and result['producer']==units[index-1],'immediate successful preceding audit')
  state=O.read(prior/'run/endpoint.json');previous=state['original'];fresh={'fresh_endpoints':[None,state['independent384_endpoint']]};count=result['total_qualified_charts']
 tick();return previous,fresh,count

def run_worker(index,prior_pin,out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
 M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1 nice10')
 units=O.read(ROOT/CONFIG)['units'];M.need(type(index)is int and 0<=index<len(units),'fixed unit index')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'unit_index':index,'producer':units[index],'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<24*1024**2,'worker24MiB cap')
  p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stop[0] and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'time/RSS budget')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n)for n in FILES},'prior_audit_sha256':prior_pin,'pid':os.getpid(),'threads':dict.fromkeys(O.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  previous,fresh,total=initial_anchor(index,prior_pin,units,tick);run,rt,batch=bind(units[index],tick);inp=O.read(run/'inputs.json')
  M.need(inp['previous']==previous,'exact already-reviewed previous chart')
  model=O.read(OLD_PRODUCER/'inputs.json')['model_inputs'];M.need(inp['model_inputs']==model,'exact original consumed model/data')
  for i,row in enumerate(batch['charts']):
   c=pinned_chart(run,row);validate_row(c,row,previous,i);tick()
   original=A.chart_check(c,c['anchor'],c['normal'],c['origin'],c['exact_weights'],256);tick()
   anchor={'state_box':fresh['fresh_endpoints'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]}
   stronger=A.chart_check(c,anchor,c['normal'],c['origin'],c['exact_weights'],384);tick()
   save(f'chart-{i:03d}.json',{'original':original,'independent384':stronger,'producer_artifact':row['artifact']})
   if i<units[index]['qualified']:
    M.need(original['passed'] and stronger['passed'],'qualified primitive chart and full join')
    previous=compact_chart(c);fresh=stronger;total+=1
   else:
    M.need(i==len(batch['charts'])-1 and not original['passed'] and original['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'retained failed tail cannot anchor')
   result['charts'].append({'local_index':i,'producer_passed':row['passed'],'original_passed':original['passed'],'independent384_passed':stronger['passed'],'contraction':stronger['contraction'],'max_image_ratio':stronger['max_image_ratio']})
   del c;gc.collect();tick()
  save('endpoint.json',{'original':previous,'independent384_endpoint':fresh['fresh_endpoints'][1]})
  result.update(passed=True,total_qualified_charts=total,exact_gauge_endpoints=['-1/64',previous['exact_parameter_endpoints'][1]],hidden_rate_endpoint=fresh['fresh_endpoints'][1][-1],scope='Connected native primitive18 branch; no whole-branch attraction, monotone hidden rate, fold identification or full shrimp. Independent saved arithmetic, shared analytic acquisition/Arb; root previously reviewed producer source.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
 print(json.dumps({k:v for k,v in result.items()if k!='charts'}),flush=True);return 0 if result['passed']else 2

def launch(index,prior_pin):
 out=BASE/f'unit-{index+1:03d}';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None;code=2
 M.need(int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))>3*1024**2,'host headroom')
 def save(n,v):
  with(out/n).open('xb')as f:f.write(canonical_bytes(v));f.flush();os.fsync(f.fileno())
 command=['timeout','--signal=TERM','--kill-after=5s','135s',sys.executable,str(Path(__file__).resolve()),'--worker','--index',str(index),'--prior-manifest',prior_pin,'--out',str(out/'run')]
 meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'launcher_pid':os.getpid(),'argv':command,'started_unix':time.time(),'load':os.getloadavg(),'internal_seconds':120,'external_seconds':135,'kill_grace_seconds':5}
 save('launcher-start.json',meta)
 with(out/'process.log').open('xb')as log:
  try:proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('owned-process.json',{'group_pid':proc.pid});code=proc.wait(timeout=142)
  except BaseException as e:meta['failure']=type(e).__name__+': '+str(e)
  finally:
   if proc is not None:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()
 live=[]
 if proc is not None:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    f=(p/'stat').read_text().rsplit(')',1)[1].split()
    if int(f[2])==proc.pid and f[0]!='Z':live.append(int(p.name))
   except(OSError,ValueError):pass
 if live or sum(p.stat().st_size for p in out.rglob('*')if p.is_file())>25*1024**2:code=2
 meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,owned_group_pid=None if proc is None else proc.pid,owned_live_pids=live);save('launcher-result.json',meta)
 inv={str(p.relative_to(out)):O.entry(p)for p in sorted(out.rglob('*'))if p.is_file()};save('manifest.json',{'files':inv})
 for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
 out.chmod(0o555);print(json.dumps({'root':str(out),'exit_code':code,'manifest':O.entry(out/'manifest.json')}),flush=True);return code

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--prior-manifest',required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args()
 raise SystemExit(run_worker(a.index,a.prior_manifest,a.out)if a.worker else launch(a.index,a.prior_manifest))
