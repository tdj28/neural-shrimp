"""Independent235D native continuation through repaired small-chart prefix."""
import gc,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_s2_newgauge_chain as B
O,A,RB,M=B.O,B.A,B.RB,B.M
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_orbits import canonical_bytes
BASE=Path('/data2/neural-shrimp-data/explore/certificates')
SPECS=[('s2-period18-fold-rechart-2026-10-10/attempt-001-target','f5a1490549cfc3e44ba045cfcb45d564f862401b','8a8f85aa6a57538cba046f62d314d15f93ebce1c8d998c3e13c12bda1e30dc86','2b900f6b1928ed512d0395200e276849a27c777e4284ed991950077fc0c7e5f1',2),('s2-period18-fold-saved-rebox-2026-10-10/attempt-001-target','bf688606f5afb5daf6bdfb20a02fd458c1ee7d94','2bd9aa9a8d797f7baa21225871e075e9b1af5f870324095d997320e0b2d7231a','68abff9e1ccdb5c4b8170acd174c4e5e8f27e585698ea1c5436281d89bc56d2d',0),('s2-period18-fold-small-march-2026-10-10/unit-001','e987c8669efb5fb9275046f401cd4cfa172cfa9b','6e46898411bdcbb6af977fafe7b6048a825f71b50c7565d624167d38e69c0b85','a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2',2)]
INITIAL=Path('/data2/neural-shrimp-data/audits/s2-period18-fold-endpoint-localization/attempt-001');PIN='5e1b1959fdeb2366fdf4ee38cd335f233757b5ee5b2e34157308fe3116b61384'
FILES=B.FILES+['tools/reviews/review_s2_small_march.py','docs/plans/s2-small-march-independent-audit.md']
def bind(spec):
 rel,execution,outer,worker,code=spec;root=BASE/rel;run=root/'run';O.verify(root,outer);O.verify(run,worker);rt=O.read(run/'runtime-start.json');done=O.read(run/'outcome.json');la=O.read(root/'launcher-result.json')
 M.need(rt['source_commit']==la['source_commit']==execution and done['exit_code']==la['exit_code']==code and not la['owned_live_pids'],'retained exact execution/outcome')
 for n,h in rt['source_sha256'].items():M.need(O.entry(run/'source'/n)['sha256']==hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==h,'archived committed source '+n)
 M.need(rt['mode']=='target' and rt['nice']>=10 and rt['threads']==dict.fromkeys(O.THREADS,'1') and not Path('/proc',str(rt['pid'])).exists(),'local owned execution terminated')
 return run,done

def worker(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];result={'passed':False,'charts':[]};previous=None;fresh=None
 def save(n,v):
  b=canonical_bytes(v);M.need(len(b)+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<24*1024**2,'24MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped[0] and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'120s/1.5GiBcap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:O.entry(ROOT/n) for n in FILES},'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1')})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 def check(c,label,anchor=None,qualified=True):
  nonlocal previous,fresh
  M.need(c['settings']['halfwidth_exponent']==-8 and c['settings']['precision']==256 and c['statement']['period']==18 and c['statement']['phase_dimension']==13,'native settings')
  if previous is not None:B.validate_transition(previous,c);anchor=c['anchor'];aa={'state_box':fresh,'exact_parameter':previous['exact_parameter_endpoints'][1]}
  else:aa=anchor;M.need(c['anchor']==anchor,'complete initial known root')
  method=RB.chart_check if 'reboxing' in c else A.chart_check
  q=method(c,anchor,c['normal'],c['origin'],c['exact_weights'],256);save(label+'-256.json',q);tick()
  if qualified:
   r=method(c,aa,c['normal'],c['origin'],c['exact_weights'],384);save(label+'-384.json',r);tick();M.need(q['passed'] and r['passed'],'independent full native proof')
   M.need(not any(x['contained_in_qualified_chart'] for x in c['known_fold_membership']['all_cyclic_alignments']),'fold identity not implied')
   previous=compact_chart(c);fresh=r['fresh_endpoints'][1];result['charts'].append({'label':label,'right':c['exact_parameter_endpoints'][1],'contraction384':r['contraction']})
  else:M.need(not q['passed'] and q['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'retained root-only failure')
 try:
  O.verify(INITIAL,PIN);known=O.read(INITIAL/'run/reduction-256.json');M.need(known['passed'] and known['exact_gauge']=='33/128','independent144endpoint');anchor={'state_box':known['final_state_box'],'exact_parameter':'33/128'}
  run,_=bind(SPECS[0]);inp=O.read(run/'inputs.json');model=inp['model_inputs'];pair=O.read(run/'certificate.json.gz');M.need(len(pair['charts'])==2 and not pair['passed'],'exact partial pair');check(pair['charts'][0],'rechart-first',anchor);failed=pair['charts'][1];check(failed,'rechart-second-failed',qualified=False)
  run,done=bind(SPECS[1]);rinp=O.read(run/'inputs.json.gz');c=O.read(run/'certificate.json.gz');M.need(rinp['model_inputs']==model and rinp['chart']==failed and done['passed'],'same exact repaired initial')
  for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):M.need(c[key]==failed[key],'unchanged rebox '+key)
  check(c,'saved-repair');del pair,failed,c;gc.collect();run,done=bind(SPECS[2]);mi=O.read(run/'inputs.json');M.need(mi['model_inputs']==model and mi['previous']==previous and not (run/'batch.json').exists(),'bound partial prefix')
  M.need(done['last_durable_chart']['index']==9 and done['last_durable_initial']['index']==10 and done['error']=={'type':'ValueError','message':'nonnegative finite radii proposal'},'exact retained partial cardinality/error')
  repaired=0
  for i in range(10):
   c=O.read(run/f'chart-{i:03d}.json.gz')
   if 'reboxing' in c:
    repaired+=1;initial=O.read(run/f'initial-{i:03d}.json.gz');ident=c['reboxing']['initial_artifact'];M.need(ident['path']==f'initial-{i:03d}.json.gz' and O.entry(run/ident['path'])=={k:ident[k] for k in ('bytes','sha256')},'full failed rebox input identity')
    for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','predictor_residual','center_path_derivatives','whole_path_third_derivative','anchor'):M.need(c[key]==initial[key],'same fresh-domain predictor '+key)
    check(initial,f'initial-{i}',qualified=False);del initial
   check(c,f'march-{i}');del c;gc.collect()
  M.need(repaired==4,'all four successful fresh-domain repairs');failed=O.read(run/'initial-010.json.gz');check(failed,'failed-tail',qualified=False);save('endpoint.json',{'original':previous,'independent384_endpoint':fresh})
  result.update(passed=True,total_connected_charts=156,exact_new_gauge=['-1/128','45/128'],scope='Independent256 exact replay and384 fresh saved reduction, primitive and full235D native joins through qualified prefix; all failed roots retained. Shared analytic derivatives/pathjets and Arb, no fold identity or whole-path attraction.')
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
