"""Independent saved-arithmetic review of authenticated new-gauge chart prefixes."""
import argparse,gc,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_arclength_connection as O
import review_arclength_connection_math as A
import review_reboxed_arclength_math as RB
import review_joined_period18_math as M
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_orbits import canonical_bytes
CONFIG='docs/plans/s2-newgauge-audit-inputs.json';CARD='docs/plans/s2-newgauge-audit.md'
BASE=Path('/data2/neural-shrimp-data/audits/s2-period18-newgauge-chain')
INITIAL=Path('/data2/neural-shrimp-data/audits/s2-period18-endpoint-localization/attempt-001')
INITIAL_PIN='be793088b7a6e2fed389e54da2c107be70e203cc13dd55c5a8a9e1a63516b408'
FILES=list(dict.fromkeys(O.FILES+[CONFIG,CARD,'tools/reviews/review_s2_newgauge_chain.py','tools/reviews/review_reboxed_arclength_math.py','src/neural_shrimp/rigorous_arclength_batch.py','tests/reviews/test_s2_newgauge_chain.py']))

def validate_transition(previous,c):
 M.need(c['normal']==previous['normal'] and c['origin']==previous['origin'],'same exact gauge')
 M.need(c['anchor']=={'state_box':previous['endpoint_root_enclosures'][1],'exact_parameter':previous['exact_parameter_endpoints'][1]},'complete native predecessor')
 M.need(c['exact_parameter_endpoints'][0]==previous['exact_parameter_endpoints'][1],'no parameter gap')
 M.need(Fraction(c['parameter_center'])-Fraction(previous['parameter_center'])==Fraction(2)**c['settings']['halfwidth_exponent']+Fraction(2)**previous['settings']['halfwidth_exponent'],'successive chart centers')

def bind(unit,tick):
 root=Path(unit['root']);run=root/'run';O.verify(root,unit['outer']);O.verify(run,unit['worker']);tick()
 rt=O.read(run/'runtime-start.json');M.need(rt['source_commit']==unit['execution'] and rt['mode']=='target','producer execution')
 for n,h in rt['source_sha256'].items():M.need(O.entry(run/'source'/n)['sha256']==h and O.entry(run/'source'/n)['sha256']==__import__('hashlib').sha256(subprocess.check_output(['git','show',unit['execution']+':'+n],cwd=ROOT)).hexdigest(),'retained committed producer source '+n)
 M.need(rt['threads']==dict.fromkeys(O.THREADS,'1') and rt['nice']>=10 and rt['cpu_only'] and not rt.get('paid',False),'local runtime scope')
 done=O.read(run/'outcome.json');la=O.read(root/'launcher-result.json')
 M.need(done['exit_code']==la['exit_code']==unit['exit'] and not la['owned_live_pids'],'retained completion/failure')
 M.need(la['source_commit']==unit['execution'] and la['source_sha256']==rt['source_sha256'],'launcher execution binding')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
 c=unit['completed'];a=unit['reboxed'];counts=done['adapter_counts'];jets=done['jet_counts']
 M.need(counts['map_attempted']==counts['map_returned']==72*c+18*a and counts['unit_attempted']==counts['unit_returned']==144*c+36*a and jets['attempted']==jets['returned']==54*c,'complete native derivatives and jets')
 for x,cap in((done['elapsed_seconds'],rt['internal_seconds']),(done['max_rss_kib'],2*1024**2)):
  M.need(type(x)in(int,float) and math.isfinite(x) and 0<=x<cap,'bounded producer resources')
 return run,O.read(run/'inputs.json'),done

def chart_source(run,index,unit):
 if unit['kind']=='reanchor':
  result=O.read(run/'certificate.json.gz');M.need(result['passed'] and len(result['charts'])==2,'qualified new gauge pair');return result['charts'][index]
 return O.read(run/f'chart-{index:03d}.json.gz')

def run_worker(index,prior_pin,out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1 nice10')
 units=O.read(ROOT/CONFIG)['units'];M.need(0<=index<len(units),'frozen unit index');unit=units[index]
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];result={'passed':False,'unit_index':index,'producer':unit,'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);M.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<24*1024**2,'24MiB worker cap');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped[0] and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'time/RSS cap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'prior_audit_sha256':prior_pin,'pid':os.getpid(),'threads':dict.fromkeys(O.THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0)})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  run,inp,done=bind(unit,tick)
  original=O.read(ROOT/'certificates/s2-period18-receipt.json');M.need(O.entry(ROOT/'certificates/s2-period18-receipt.json')['sha256']=='f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277','original model receipt')
  mp=Path(original['external_root'])/'inputs.json';M.need(O.entry(mp)['sha256']=='9c01121b61a195baf20050eaf3bf567ebba6b7a0310d8fbced0aabc67f98248f' and inp['model_inputs']==O.read(mp),'same exact consumed native model')
  previous=None;fresh=None;count=127
  if index==0:
   M.need(prior_pin==INITIAL_PIN,'localized audit identity');O.verify(INITIAL,prior_pin);known=O.read(INITIAL/'run/reduction-256.json')
   M.need(known['passed'] and known['precision']==256 and inp['anchor']=={'state_box':known['final_state_box']},'independently replayed256 known root; supplementary384 localization is not claimed')
   anchor=inp['anchor'];fresh_anchor=anchor
  else:
   prior=BASE/f'unit-{index:03d}';O.verify(prior,prior_pin);pd=O.read(prior/'run/outcome.json');state=O.read(prior/'run/endpoint.json')
   M.need(pd['passed'] and pd['producer']==units[index-1] and pd['unit_index']==index-1,'successful immediate audit predecessor')
   previous=state['original'];fresh=state['independent384_endpoint'];count=pd['total_qualified_charts'];M.need(inp['previous']==previous,'whole producer endpoint equals prior audited endpoint')
  adapted=0
  for i in range(unit['qualified']):
   c=chart_source(run,i,unit);M.need(c['passed'] and all(c['gates'].values()) and c['statement']['period']==18 and c['statement']['phase_dimension']==13,'qualified native18 chart')
   M.need(c['settings']['precision']==256 and c['settings']['halfwidth_exponent']==-7 and c['settings']['newton_steps']==2,'fixed scientific settings')
   if previous is not None:
    validate_transition(previous,c);anchor=c['anchor'];fresh_anchor={'state_box':fresh,'exact_parameter':previous['exact_parameter_endpoints'][1]}
   else:M.need(c['anchor']==anchor and c['normal']==inp['normal'] and c['origin']==inp['origin'],'full correlated first anchor')
   initial=None
   if 'reboxing' in c:
    adapted+=1;ident=c['reboxing']['initial_artifact'];p=run/f'initial-{i:03d}.json.gz';M.need(ident['path']==p.name and O.entry(p)=={k:ident[k] for k in ('bytes','sha256')},'retained failed initial')
    initial=O.read(p)
    for key in('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):M.need(initial[key]==c[key],'same reboxed predictor/anchor '+key)
    ri=A.chart_check(initial,initial['anchor'],initial['normal'],initial['origin'],initial['exact_weights'],256)
    M.need(not ri['passed'] and ri['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'independently reproduced initial root failure');save(f'initial-{i:03d}.json',ri);del initial;gc.collect();tick()
   f=RB.chart_check if 'reboxing' in c else A.chart_check
   r256=f(c,anchor,c['normal'],c['origin'],c['exact_weights'],256);tick();r384=f(c,fresh_anchor,c['normal'],c['origin'],c['exact_weights'],384);tick()
   save(f'chart-{i:03d}.json',{'original256':r256,'independent384':r384})
   M.need(r256['passed'] and r384['passed'],'independent root/primitive/full join')
   if 'known_fold_membership' in c:M.need(not any(v['contained_in_qualified_chart'] for v in c['known_fold_membership']['all_cyclic_alignments']),'positive fold membership requires separate independent audit')
   previous=compact_chart(c);fresh=r384['fresh_endpoints'][1];count+=1
   result['charts'].append({'index':i,'passed':True,'contraction':r384['contraction'],'max_image_ratio':r384['max_image_ratio']});del c;gc.collect()
  M.need(adapted==unit['reboxed'],'complete successful rebox count')
  if unit['kind']=='partial':
   M.need(not (run/'batch.json').exists() and done['error']=={'type':'ValueError','message':'nonnegative finite radii proposal'},'disclosed interrupted batch/proposal')
   failed=O.read(run/f"initial-{unit['qualified']:03d}.json.gz");validate_transition(previous,failed)
   bad=A.chart_check(failed,failed['anchor'],failed['normal'],failed['origin'],failed['exact_weights'],256);save('failed-tail-256.json',bad)
   M.need(not bad['passed'] and bad['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'rejected tail cannot anchor');tick()
  save('endpoint.json',{'original':previous,'independent384_endpoint':fresh})
  result.update(passed=True,total_qualified_charts=count,exact_new_gauge=['-1/128',previous['exact_parameter_endpoints'][1]],hidden_rate_endpoint=fresh[-1],scope='Independent256 replay and384 saved-arithmetic contraction/primitive/native joins, including correlated reanchoring. Prior localization only256 replayed. Shared analytic derivative/path-jet acquisition and Arb; no new model calls, whole-branch attraction, fold identity or entire shrimp.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps({k:v for k,v in result.items() if k!='charts'}));return 0 if result['passed'] else 2

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
