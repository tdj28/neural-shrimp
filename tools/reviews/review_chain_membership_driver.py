"""Fixed saved-only native chain lift and final independent-tube membership audit."""
import hashlib,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import review_chain_membership_math as M
need=M.need
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
BASE=Path('/data2/neural-shrimp-data/audits/eigenpair-chain-membership-independent')
CARD=ROOT/'docs/plans/eigenpair-chain-membership-independent.json'
CHAIN=Path('/data2/neural-shrimp-data/audits/generalized-chain-independent/attempt002')
CHAIN_OUTER='ccc9147597490b78bcfc92f87a13093097694fb402c0ef1303ef7c4a52b64d91'
CHAIN_WORKER='f8eb89d3e205f9d1af994b93e27cae0a9657569aa25d35346df03dc45ba0d947'
CHAIN_SOURCE='4e8e0ec0844ed806d7a9c6452da9e7c603e63f02'
LIFT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/native-lift/attempt001-target')
LIFT_OUTER='96ab6b8c668f7631805a7a6eb78bf90fa924cce7ca280891c51ca02dd1b4c84f'
LIFT_WORKER='4dd882c89745b946e42540dc46a62b704c821057c83d46c4ba78acc913fd9ffb'
LIFT_SOURCE='bc2f0ad7a435d0d04f859d683223c688f178db9e'
FINAL=Path('/data2/neural-shrimp-data/audits/eigenpair-remaining-locus/unit024-attempt001')
MAP=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-003-quadratic-target/run/exact-map-inputs.json')
MAP_SHA='16fc4b1c5ea4e7cb30fe321ca48147c209ecba91fca60872c1583de1a3a81f0a'
def read(p):return json.loads(p.read_bytes())
def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def inventory(root,pin):
 need(entry(root/'manifest.json')['sha256']==pin,'fixed manifest pin')
 files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 need(set(files)==set(actual),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable source/data '+n)
def source(run,commit,subdir='source',rtkey='sources'):
 rt=read(run/'runtime.json');need(rt.get('commit',rt.get('source_commit'))==commit,'exact execution commit')
 need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'native CPU1nice10')
 archived=run/subdir
 for p in sorted(archived.rglob('*')):
  if not p.is_file():continue
  n=str(p.relative_to(archived));expected=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()
  need(entry(p)['sha256']==expected,'historical archived source '+n)
  if rtkey is not None:need(entry(p)==rt[rtkey][n],'runtime source identity '+n)
 need(not Path('/proc',str(rt['pid'])).exists(),'prior worker exited');return rt

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
 card=read(CARD);need(card['final_root']==str(FINAL)and card['total_connected_charts']==107,'exact complete107 prerequisite')
 need(len(card['final_outer'])==64 and len(card['final_worker'])==64,'literal final immutable pins')
 out=BASE/'attempt001/run';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];used=[0]
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
 signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,60)
 def tick():need(not stop[0]and time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'60s/512MiB cap')
 def save(name,data):
  b=(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();need(used[0]+len(b)<16*1024**2,'16MiB raw cap')
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used[0]+=len(b)
 files=['tools/reviews/review_chain_membership_math.py','tools/reviews/review_chain_membership_driver.py','tools/reviews/launch_chain_membership_review.py','tests/reviews/test_chain_membership_review.py','docs/plans/eigenpair-chain-membership-independent.md',str(CARD.relative_to(ROOT))]
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:entry(ROOT/n)for n in files},'pid':os.getpid(),'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'model_calls':0,'arithmetic':'exact fractions'})
 for n in files:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
 result={'passed':False}
 try:
  inventory(CHAIN,CHAIN_OUTER);inventory(CHAIN/'run',CHAIN_WORKER);rt=source(CHAIN/'run',CHAIN_SOURCE,'audit-source',None)
  chainout=read(CHAIN/'run/outcome.json');launch=read(CHAIN/'launch-end.json');bind=read(CHAIN/'run/input-binding.json')
  need(launch['source_commit']==CHAIN_SOURCE and launch['exit_code']==0 and launch['owned_group_empty']and launch['child_pid_absent'],'completed independent chain launch')
  need(chainout['passed']and chainout['root_passed']and chainout['positive_actual_rates']and chainout['primitive_period']==18 and chainout['strict_inclusions']==41 and chainout['exact_algebraic_multiplicity_two']and chainout['complement_attracting'],'independent normalized primitive18 Jordan root')
  rootbox=read(CHAIN/'run/fresh-box.json')['box'];tick()
  inventory(LIFT,LIFT_OUTER);inventory(LIFT/'run',LIFT_WORKER);source(LIFT/'run',LIFT_SOURCE)
  inp=read(LIFT/'run/inputs.json');lift=read(LIFT/'run/lift.json');lout=read(LIFT/'run/outcome.json');lla=read(LIFT/'launcher-result.json')
  need(lla['commit']==LIFT_SOURCE and lla['exit_code']==0 and lla['owned_live_pids']==[]and lla['stop_reason']is None,'completed native lift launch')
  need(lout['passed']and lout['status']=='authenticated_native_lift_completed'and lout['observations']==lout['ticks']==18,'complete18 phase native lift')
  need(inp['bindings']['proof']['sha256']==bind['proof_sha256']and inp['bindings']['inputs']['sha256']==bind['inputs_sha256'],'identical independently proved generalized root')
  need(entry(MAP)['sha256']==MAP_SHA and inp['map_inputs']==bind['map_inputs']==read(MAP),'identical full native physical map')
  need(all(M.contains(M.ball(a),M.ball(b))for a,b in zip(inp['box'],rootbox))and len(inp['box'])==len(rootbox)==41,'lift input covers independent enlarged41D box')
  phasefiles=sorted((LIFT/'run').glob('event-*-phase.json'));need(len(phasefiles)==18,'all native lift phase events')
  phases=[read(p)for p in phasefiles];domain=M.check_phase_domains(rootbox,lift,phases);save('native-domain-check.json',domain);tick()
  inventory(FINAL,card['final_outer']);inventory(FINAL/'run',card['final_worker']);source(FINAL/'run',card['final_source'])
  fout=read(FINAL/'run/outcome.json');fla=read(FINAL/'launcher-result.json');endpoint=read(FINAL/'run/endpoint.json');chart=endpoint['original']
  need(fla['source_commit']==card['final_source']and fla['exit_code']==0 and fla['owned_live_pids']==[],'completed final independent launch')
  need(fout['passed']and fout['total_connected_charts']==107 and [r['global_chart']for r in fout['charts']]==[106,107]and all(r['passed']for r in fout['charts']),'complete independent boundary107')
  need(chart['parameter_endpoints']==['293/128','295/128']and chart['passed']is True,'exact final independently proved chart')
  need(chart['normal']==inp['native_event']['normal']and chart['origin']==inp['native_event']['origin'],'unchanged native chart gauge since root source')
  member=M.check_membership(lift,chart);save('membership.json',member);tick();need(member['passed'],'exact-rational all470 strict membership')
  save('input-bindings.json',{'chain':[str(CHAIN),CHAIN_OUTER,CHAIN_WORKER,CHAIN_SOURCE],'lift':[str(LIFT),LIFT_OUTER,LIFT_WORKER,LIFT_SOURCE],'final':card,'proof':bind['proof_sha256'],'phase_domains':{p.name:entry(p)for p in phasefiles}})
  result.update(passed=True,total_connected_charts=107,primitive_period=18,native_coordinate_inclusions=len(member['coordinate_inclusions']),parameter=member['parameter'],scope='Independent source/domain binding and exact-rational full470 membership in independently replayed107-chart native locus. Shared analytic phase acquisitions; independently audited41D normalized double+1 root and stable11D complement are prior prerequisites. No generic unfolding or full shrimp claim.')
 except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
