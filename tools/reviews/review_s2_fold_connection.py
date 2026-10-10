"""Independent saved235D fold-to-center branch connection reduction."""
import gc,hashlib,json,os,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_s2_small_march as B
O,A,RB,M=B.O,B.A,B.RB,B.M
from review_quadratic_connection_math import original_ball,exact
from neural_shrimp.rigorous_arclength_batch import compact_chart
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json
BASE=Path('/data2/neural-shrimp-data/explore/certificates')
RESTRICT=('s2-period18-fold-restricted-chart-2026-10-10/attempt-001-target','c5f9c0690e4364199e5a70b39c577854ef305c4f','0e2a811737574802c0ed27af57bf45941c4f65932698092b720fff53f9cd4f95','a1f641992973e0a3b1eee34cca5c5ca0a97a7efadcd1ae75e1322e3c48e332c6',0)
FAILED=('s2-period18-fold-reverse-2026-10-10/attempt-001-target','136b77b4ef7102a49951d44007610d67fcaf596e','f872e5f6640018428004d8945e2fec1b982aa2e1ee694da08bc04a070f1a7e01','9aee1856145ff51a5f76ab809118a406ef40d478344217dddef0aa5472432f71',2)
UNITS=[('s2-period18-fold-reverse-march-2026-10-10/unit-001','366bb15f5bf27218f569fffa1d0232c44a4670fe','22c396c0eafbdd4a93d9cc8749dbe29f529b98ee80850cf228e3172e25139a47','e1a4d1514a068c2791e24fb834a07c09f4729e120f4b3287dad8395898e574d1',0),('s2-period18-fold-reverse-march-2026-10-10/unit-002','7f9346e9f1c92bdecd0c18f1e6ec71ce3e0af931','e9774d0f9a46b771533f12ae43d65d9b00fa98bc30cb47e5ade23c67a15dbc9e','d66af8427a2eeb8da0f5eb8b1cf7065879d18a707e1956a26e8316df4d5bc11f',0)]
ENDPOINT=Path('/data2/neural-shrimp-data/audits/s2-period18-near-fold-endpoint-localization/attempt-001');ENDPIN='220b77b44a053a598bdbdbc264458f33a77887d27517256e60ab3c65f6789cb9'
FILES=B.FILES+['tools/reviews/review_s2_fold_connection.py','tests/reviews/test_s2_fold_connection_review.py','docs/plans/s2-fold-connection-independent.md']
def contains(a,b):
 lo,hi=M.interval(a);u,v=M.interval(b);return lo<=u<=v<=hi

def restriction(c,old):
 for key in ('center','normal','origin','parameter_center','predictor_slope','predictor_curvature','preconditioner','center_path_derivatives','whole_path_third_derivative','anchor'):
  M.need(c[key]==old[key],'same restricted exact predictor/Taylor source '+key)
 M.need(c['settings']['halfwidth_exponent']==-10<old['settings']['halfwidth_exponent']==-7,'strict H3 parameter restriction')
 M.need(c['settings']['radius_exponent']==-18 and c['settings']['precision']==256,'fixed new whole error domain')
 # Exact predictor identity and interval inclusion imply all true value,
 # velocity and unchanged acceleration inputs are a subset of the old H3 path.
 with ctx.workprec(384):
  oldd=arb(0,exact(Fraction(2)**-7));newd=arb(0,exact(Fraction(2)**-10));rows=[]
  for i in range(235):
   z=exact(M.point(c['center'][i]));s=exact(M.point(c['predictor_slope'][i][0]));t=exact(M.point(c['predictor_curvature'][i][0]))
   op=z+s*oldd+t*oldd*oldd/2;np=z+s*newd+t*newd*newd/2;ov=s+t*oldd;nv=s+t*newd
   rows.append(bool(op.contains(np) and ov.contains(nv)))
 M.need(all(rows),'independent restricted value/velocity path containment')
 return {'passed':True,'coordinates':235,'old_halfwidth':'1/128','new_halfwidth':'1/1024','retained_all_Taylor_terms':True,'new_C1_error_domain_checked_by_chart_reduction':True}

def membership(c,box,precision):
 p,n=18,13;N=235;rows=[]
 M.need(len(box)==N,'full235 native root')
 with ctx.workprec(precision):
  decoder=M.stored if precision==256 else original_ball
  nn=list(map(exact,c['normal']));oo=list(map(exact,c['origin']));sc=exact(c['parameter_center']);delta=exact(Fraction(2)**c['settings']['halfwidth_exponent'])
  center=[decoder(v) for v in c['center']];S=[decoder(v[0]) for v in c['predictor_slope']];T=[decoder(v[0]) for v in c['predictor_curvature']];radii=[decoder(v) for v in c['coordinate_radii']]
  for phase in range(p):
   bb=[decoder(box[((j+phase)%p)*n+i]) for j in range(p) for i in range(n)]+[decoder(box[-1])]
   projection=sum((nn[i]*(bb[i]-oo[i]) for i in range(N)),arb(0));d=projection-sc
   errors=[abs(bb[i]-(center[i]+S[i]*d+T[i]*d*d/2)).upper() for i in range(N)];checks=[bool(errors[i]<radii[i]) for i in range(N)];inside=bool(abs(d).upper()<delta)
   rows.append({'phase':phase,'parameter_inside':inside,'coordinate_inclusions':checks,'max_ratio':ball_json(max((errors[i]/radii[i]).upper() for i in range(N))),'passed':inside and all(checks)})
 return rows

def worker(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=[False];result={'passed':False,'charts':[]};previous=None;fresh=None
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2)
 def save(n,v):
  b=canonical_bytes(v);M.need(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(b)<64*1024**2,'64MiB');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped[0] and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1536*1024,'240s/1.5GiB')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'pid':os.getpid(),'sources':{n:O.entry(ROOT/n) for n in FILES},'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':[256,384],'model_calls':0})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 def check(c,label,anchor=None):
  nonlocal previous,fresh
  M.need(c['settings']['halfwidth_exponent']==-10 and c['settings']['precision']==256 and c['statement']['period']==18 and c['statement']['phase_dimension']==13,'fixed native settings')
  if previous is not None:B.B.validate_transition(previous,c);anchor=c['anchor'];aa={'state_box':fresh,'exact_parameter':previous['exact_parameter_endpoints'][1]}
  else:aa=anchor;M.need(c['anchor']==anchor,'correlated whole fold anchor')
  method=RB.chart_check if 'reboxing' in c else A.chart_check
  q=method(c,anchor,c['normal'],c['origin'],c['exact_weights'],256);save(label+'-256.json',q);tick()
  r=method(c,aa,c['normal'],c['origin'],c['exact_weights'],384);save(label+'-384.json',r);tick();M.need(q['passed'] and r['passed'],'full native root/primitive/join passes independently')
  previous=compact_chart(c);fresh=r['fresh_endpoints'][1];result['charts'].append({'label':label,'right':c['exact_parameter_endpoints'][1],'contraction384':r['contraction']})
 try:
  O.verify(ENDPOINT,ENDPIN);known=O.read(ENDPOINT/'run/reduction-256.json');M.need(known['passed'] and known['exact_gauge']=='45/128','independently localized forward endpoint')
  fd=O.read(ROOT/'docs/reviews/s2-period18-fold-certificate-independent-review.json');M.need(fd['status']=='passed','independent fold review');O.verify(Path(fd['raw_root']),fd['raw_manifest_sha256'])
  receipt=O.read(ROOT/'certificates/s2-period18-fold-receipt.json');foldroot=Path(receipt['external_root'])
  for name,identity in receipt['files'].items():M.need(O.entry(foldroot/name)==identity,'canonical fold bytes '+name)
  fold=O.read(foldroot/'run/certificate.json')['result'];fi=O.read(foldroot/'run/inputs.json');M.need(fold['passed'] and all(fold['gates'].values()) and fi['period']==18,'certified simple fold')
  fr,_=B.bind(FAILED);old=O.read(fr/'certificate.json.gz')['charts'][0];oi=O.read(fr/'inputs.json');M.need(not old['passed'] and old['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'retained original root-only failure')
  foldbox=[v for row in fold['orbit_enclosures'][:18] for v in row]+[fold['box'][13]]
  M.need(len(old['anchor']['state_box'])==235 and all(contains(a,b) for a,b in zip(old['anchor']['state_box'],foldbox,strict=True)) and 'exact_parameter' not in old['anchor'],'whole actual fold root without false zero gauge')
  rr,_=B.bind(RESTRICT);inp=O.read(rr/'inputs.json.gz');c=O.read(rr/'certificate.json.gz');M.need(inp['chart']==old and inp['model_inputs']==oi['model_inputs'],'exact failed source/model binding')
  original=O.read(ROOT/'certificates/s2-period18-receipt.json');mp=Path(original['external_root'])/'inputs.json';M.need(inp['model_inputs']==O.read(mp),'same exact certified central training map')
  for key in fi['map_inputs']:
   if key!='lr_hidden':M.need(fi['map_inputs'][key]==inp['model_inputs'][key],'same fold model '+key)
  M.need(inp['target_box']==known['final_state_box'],'same independently localized full target')
  save('restriction.json',restriction(c,old));check(c,'restricted-fold',c['anchor']);target=inp['target_box'];model=inp['model_inputs'];del old,oi,c,inp;gc.collect();index=1
  for unit,spec in enumerate(UNITS):
   run,done=B.bind(spec);inp=O.read(run/'inputs.json');batch=O.read(run/'batch.json');count=16 if unit==0 else 14
   M.need(inp['model_inputs']==model and inp['target_box']==target and inp['previous']==previous,'complete exact unit predecessor/model/target')
   M.need(batch['passed'] and batch['completed_charts']==batch['qualified_charts']==count and batch['requested_charts']==(16 if unit==0 else 15) and len(batch['charts'])==count,'all bounded charts retained')
   M.need(done['adapter_counts']['map_attempted']==done['adapter_counts']['map_returned']==72*count and done['jet_counts']['attempted']==done['jet_counts']['returned']==54*count,'complete unchanged native acquisition')
   for j,row in enumerate(batch['charts']):
    path=run/f'chart-{j:03d}.json.gz';M.need(row['index']==j and row['artifact']['path']==path.name and O.entry(path)=={k:row['artifact'][k] for k in ('bytes','sha256')},'all proof identities')
    c=O.read(path);M.need('reboxing' not in c,'no repairs in these exact units');check(c,f'march-{index:03d}')
    if unit==1 and j==13:
     allrows={str(prec):membership(c,target,prec) for prec in (256,384)};save('target-membership.json',allrows)
     saved=c['known_fold_membership']['all_cyclic_alignments'];M.need(len(saved)==18,'all saved phase alignments')
     for q,s in zip(allrows['256'],saved,strict=True):M.need(q['phase']==s['cyclic_shift'] and q['passed']==s['contained_in_qualified_chart'] and q['parameter_inside']==s['strict_parameter_inside'] and q['coordinate_inclusions']==s['strict_coordinate_inclusions'],'independent strict membership decisions')
     M.need([q['phase'] for q in allrows['384'] if q['passed']]==[0],'same actual full primitive orbit at phase0')
    index+=1;del c;gc.collect();tick()
  M.need(index==31,'all31 reverse charts');result.update(passed=True,reverse_charts=31,forward_charts=156,fold_to_center_branch_identity=True,matched_phase=0,scope='Independent exact256 and fresh384 saved reductions connect the canonical primitive18 fold to the independently localized central-branch endpoint, retaining full235D states/rate and all proper divisors. This proves branch identity; whole-branch attraction and global shrimp boundary organization are separate. Shared source-bound analytic derivatives/pathjets and Arb remain disclosed.')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps({k:v for k,v in result.items() if k!='charts'}));return 0 if result['passed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
