"""One independent saved finite704 complement audit; no model/derivative calls."""
import gc,gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_native704_complement_math as H
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json
need=H.need;R=H.R;M=H.M
OUT=Path('/data2/neural-shrimp-data/audits/native704-complement-independent/attempt003/run')
CARD='docs/plans/native704-complement-independent-inputs.json'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
NATIVE=['src/neural_shrimp/'+n+'.py'for n in('rigorous_cusps','rigorous_folds','rigorous_orbits','controls','models')]
FILES=[CARD,'docs/plans/native704-complement-independent.md','tools/reviews/review_native704_complement.py','tools/reviews/launch_native704_complement_review.py','tests/test_native704_complement_driver.py','tests/test_native704_complement_review.py','docs/reviews/native704-complement-independent-attempt002-failure.json']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
  need(len(b)<=256*1024**2,'bounded256MiB decompression')
  # Original704 source wrote gzip only. Complete immutable compressed bytes
  # are authenticated by inventory() before any bounded decompression.
  return json.loads(b)
 return json.loads(p.read_bytes())

def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal manifest pin');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 need(set(actual)==set(files),'entire immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable exact file '+n);tick()
 return {'root':str(root),'manifest':pin,'files':len(files),'bytes':sum(v['bytes']for v in files.values())}

def historical(run,source):
 rt=read(run/'runtime.json');need(rt.get('commit',rt.get('source_commit'))==source,'literal archived execution')
 paths={str(p.relative_to(run/'source')):p for p in(run/'source').rglob('*')if p.is_file()};need(set(paths)==set(rt['sources']),'entire historical source archive')
 for n,p in paths.items():
  v=entry(p);need(v==rt['sources'][n]and v['sha256']==hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest(),'historical exact source '+n)
 need(not Path('/proc',str(rt['pid'])).exists(),'prior worker absent');return rt

def bundle(spec,tick,exit_code=0):
 root=Path(spec['root']);rows=[inventory(root,spec['outer'],tick),inventory(root/'run',spec['worker'],tick)];rt=historical(root/'run',spec['source']);la=read(root/'launcher-result.json')
 need(la.get('commit',la.get('source_commit'))==spec['source']and la['exit_code']==exit_code and not la['owned_live_pids'],'completed exact owned execution')
 for key in('launcher_pid','owned_group_pid'):
  if key in la:need(not Path('/proc',str(la[key])).exists(),'prior owner absent')
 return root,rt,read(root/'run/outcome.json'),rows

def exact_cover(outer,inner,label):
 need(len(outer)==len(inner),'full domain length '+label)
 for a,b in zip(outer,inner,strict=True):H.U.contains(a,ball_json(b),label)

def equal_balls(a,b,label):
 need(len(a)==len(b),'complete equal domain '+label)
 for x,y in zip(a,b,strict=True):need(M.pair(x['mid'])==M.pair(y['mid'])and M.pair(x['rad'])==M.pair(y['rad']),label)

def acquired_input(row,domains,kind,j):
 need(row['kind']==kind and row['phase']==j and 0<=j<18,'literal native phase and kind');off=39*j;v=domains['center'if kind=='center'else'whole'];t=domains['tangent']
 equal_balls(row['state'],v[off:off+13],'same actual native state');equal_balls(row['rates'],v[-2:],'both actual native rates')
 if kind=='center':need(len(row['state_tangent'])==13 and len(row['rate_tangent'])==2 and all(R.original_ball(x).is_zero()for x in row['state_tangent']+row['rate_tangent']),'exact zero center tangents')
 else:equal_balls(row['state_tangent'],t[off:off+13],'true native state tangent');equal_balls(row['rate_tangent'],t[-2:],'both true physical rate tangents')

def save_bound(root,name,value,used,closing=False):
 data=canonical_bytes(value);need(used[0]+len(data)<64*1024**2-(0 if closing else 2*1024**2),'worker64MiB with2MiB closing reserve');p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 used[0]+=len(data)

def synthetic(n,tick):
 need(type(n)is int and 1<=n<=704,'fixed bounded synthetic size');small=arb((1,-16));C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)]);D=M.ident(n)-C*J;tick()
 w=[arb(1)]*n;k=H.infnorm(D);v=arb_mat([[arb((1,-80))]for _ in range(n)]);initial=arb_mat([[arb(0,arb((1,-11)))]for _ in range(n)])
 for sign in(1,-1,2):
  box=H.enclosure(D,-C*(v*sign),w,k,initial,iterations=6 if sign==2 else 3);need(H.finite(box),'full synthetic retained enclosure');tick()
 return {'passed':True,'dimension':n,'neumann':ball_json(k),'intersections':12,'dense_matrix_product':True}

def bind(card,tick):
 rows=[];failed,_,failure,rr=bundle(card['previous_failure'],tick,2);rows+=rr
 need(failure['passed']is False and failure['native_calls']==0 and failure['error']==card['previous_failure']['error'],'retained original packaging failure')
 _,_,failure,rr=bundle(card['previous_domain_failure'],tick,2);rows+=rr
 need(failure['passed']is False and failure['native_calls']==0 and failure['error']==card['previous_domain_failure']['error'],'retained original strict tangent-domain failure')
 pr,rt,done,rr=bundle(card['producer'],tick);rows+=rr
 need(rt['mode']=='target'and rt['settings']==card['settings']and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'original native settings')
 need(done['passed']and done['status']=='qualified_unit_pair_tube'and done['strict_inclusions']==704 and done['parameter_endpoints']==['0','1/32768']and done['events']=={'chart_return':1,'direction_return':216,'newton_proposal':2,'preconditioner_return':1}and done['ticks']==217,'complete original native704 theorem')
 cert=pr/'run/certificate.json.gz';need(entry(cert)['sha256']==card['producer']['certificate']==entry(pr/'run/events/0219-chart_return.json.gz')['sha256'],'literal complete original chart')
 files=sorted((pr/'run/events').glob('*-direction_return.json.gz'));need(len(files)==216,'complete216 original acquisitions');fp={str(p.relative_to(pr/'run')):entry(p)for p in files}
 ir,irt,io,rr=bundle(card['independent'],tick);rows+=rr;ib=read(ir/'run/input-bindings.json')
 need(io['passed']and io['strict_inclusions']==io['anchor_coordinates']==704 and io['primitive_period']==18 and io['same_independently_connected_root']and io['parameter_endpoints']==['0','1/32768'],'prior independently proved same primitive18 curve')
 need(ib['card']['producer']==card['producer']and ib['card']['chain']==card['chain']and ib['card']['connection']==card['connection']and ib['certificate']==entry(cert)and ib['native_direction_files']==fp,'literal prior independent proof and all native acquisition identities')
 oldproof=read(ir/'run/native704-384.json');need(oldproof['passed']and oldproof['inclusions']==oldproof['anchor_coordinates']==704,'completed independent full arithmetic');del oldproof
 lr,lrt,lo,rr=bundle(card['localization'],tick);rows+=rr;lb=read(lr/'run/bindings.json');sel=read(lr/'run/selected-source-inputs.json')
 need(lo['passed']and lo['status']=='conditional_full704_localization_completed'and lo['native_calls']==0 and lo['shared_native_acquisitions']==108 and not lo['complementary_stability_evaluated'],'complete saved localization context')
 need(lb['card']['producer']==card['producer']and lb['card']['independent']==card['independent']and sel['card']==lb['card']and lb['certificate']==sel['certificate']==entry(cert),'same literal localization parent')
 selected=files[108:];sf={str(p.relative_to(pr/'run')):entry(p)for p in selected}
 need(sel['direction_files']==sf and sel['groups']=='center,predictor,whole: original groups4,5,6'and sel['new_native_calls']==0 and sel['shared_native_acquisitions']==108,'exact final localization acquisition groups')
 need(entry(lr/'run/localization.json')['sha256']==card['localization']['proof'],'literal saved localization')
 fr,frt,fo,rr=bundle(card['failed_orthogonal'],tick,2);rows+=rr;fb=read(fr/'run/bindings.json');proposal=read(fr/'run/events/001-proposal.json')
 need(not fo['passed']and fo['status']=='completed_sufficient_bound_unresolved'and fo['native_calls']==0 and fb['card']['producer']==card['producer']and fb['card']['independent']==card['independent']and proposal['strategy']=='periodic_lyapunov_balanced_orthogonal','retained failure and exact fixed proposals')
 cr,crt,co,rr=bundle(card['complement'],tick);rows+=rr;cb=read(cr/'run/bindings.json');domains=read(cr/'run/actual-domain-inputs.json');oldloc=read(lr/'run/localization.json')
 need(co['passed']and co['status']=='finite_complement_attracting'and co['native_calls']==crt['native_call_cap']==72 and crt['precision']==384 and co['parameter_endpoints']==['0','1/32768'],'complete36+36 acquired complement run')
 need(cb['parent']['card']['producer']==card['producer']and cb['parent']['card']['independent']==card['independent']and cb['parent']['certificate']==entry(cert)and cb['localization']==card['localization']and cb['fixed_proposals']==entry(fr/'run/events/001-proposal.json'),'exact complement parents')
 inp=read(pr/'run/inputs.json')['map_inputs'];need(cb['map_inputs']==inp and cb['native_source_matches']==NATIVE,'identical native map and data')
 for n in NATIVE:need(rt['sources'][n]==crt['sources'][n]==entry(pr/'run/source'/n)==entry(cr/'run/source'/n),'shared exact analytic derivative source '+n)
 need(domains['exact_endpoints']==['0','1/32768']and domains['halfwidth']=='1/65536','entire original finite parameter segment')
 for key,old in [('center',oldloc['center_box']),('whole',oldloc['whole_box']),('tangent',[v[0]for v in oldloc['tangent_box']])]:equal_balls(domains[key],[ball_json(R.original_ball(v))for v in old],'literal actual domains restored from localization')
 need(len(domains['complements'])==len(domains['metrics'])==len(proposal['phase_complements'])==len(proposal['upper_cholesky_factors'])==18,'all fixed18 frame/metric proposals')
 E=[R.matrix(v)for v in domains['complements']];B=[R.matrix(v)for v in domains['metrics']]
 for j in range(18):
  old=R.matrix(proposal['phase_complements'][j]);b=arb_mat([[R.exact(Fraction.from_float(x))for x in row]for row in proposal['upper_cholesky_factors'][j]])
  need(E[j]==old and B[j]==b,'same exact retained frame/metric proposals')
 nativefiles={p.name for p in(cr/'run/native').iterdir()if p.is_file()};need(nativefiles=={f'{i:03d}-{kind}.json'for i in range(1,73)for kind in('input','return')},'all72 input/return native acquisition pairs')
 native=[]
 for i in range(1,73):
  a=read(cr/f'run/native/{i:03d}-input.json');v=read(cr/f'run/native/{i:03d}-return.json')
  kind='synthetic'if i<=36 else('center'if i%2 else'whole_tangent');phase=i-1 if i<=36 else(i-37)//2
  need(a['kind']==v['kind']==kind and a['phase']==v['phase']==phase,'full acquired native inventory/order')
  if i>36:
   acquired_input(a,domains,kind,phase);A=R.matrix(v['A']);Ap=R.matrix(v['Aprime']);need(A.nrows()==A.ncols()==Ap.nrows()==Ap.ncols()==13 and H.finite(A)and H.finite(Ap),'full actual13x13 derivative returns')
   if kind=='center':need(all(x.is_zero()for row in Ap.tolist()for x in row),'center zero derivative consistency');A0=A
   else:native.append((A0,A,Ap))
 need(entry(cr/'run/complement.json')['sha256']==card['complement']['proof'],'literal complement proof')
 records=[read(cr/f'run/events/{j+2:03d}-phase.json')for j in range(18)];metric=read(cr/'run/events/020-metric.json')
 groups=[[read(p)for p in selected[k:k+36]]for k in range(0,108,36)];chart=read(cert)
 keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','weights','coordinate_radii','preconditioner','whole_box'}
 for k in list(chart):
  if k not in keep:del chart[k]
 gc.collect();tick()
 return chart,groups,domains,E,B,native,records,metric,{'card':card,'inventories':rows,'original_certificate':entry(cert),'direction_files':sf,'actual_native_files':{str(p.relative_to(cr/'run')):entry(p)for p in sorted((cr/'run/native').iterdir())if p.is_file()},'shared_native_source':{n:rt['sources'][n]for n in NATIVE}}

def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stopped=[False];result={'passed':False,'native_calls':0,'status':'incomplete'}
 def save(n,v,closing=False):save_bound(OUT,n,v,used,closing)
 def tick():need(not stopped[0]and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'120s/4GiB cap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':commit,'pid':os.getpid(),'sources':{n:entry(ROOT/n)for n in FILES},'precision':480,'native_calls':0,'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1')})
  for n in FILES:
   b=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(b).hexdigest(),'committed archive source');need(used[0]+len(b)<62*1024**2,'source bytes cap');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();profile=synthetic(704,tick);seconds=time.monotonic()-before;forecast=4*seconds+60;save('profile.json',dict(profile,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<120,'measured full704 arithmetic admission');gc.collect();tick()
   card=read(ROOT/CARD);need(card['audit']=={'output':str(OUT),'precision':480,'seconds':120,'rss_bytes':4*1024**3,'as_bytes':6*1024**3,'output_bytes':64*1024**2,'closing_reserve_bytes':2*1024**2,'synthetic_dimension':704,'forecast_multiplier':4,'forecast_allowance_seconds':60,'tangent_intersections':6},'literal fixed audit settings')
   chart,groups,domains,E,B,native,records,metric,binding=bind(card,tick);save('input-bindings.json',binding);center,whole,tangent,localized=H.localization(chart,groups,tick);save('localization-480.json',localized);del chart,groups;gc.collect();tick()
   for key,values in [('center',center),('whole',whole),('tangent',tangent)]:exact_cover(domains[key],values,'fresh480 enclosure inside actual '+key+' domain')
   save('domain-coverage.json',{'passed':True,'coordinates':704,'domains':['center','whole','tangent'],'exact_rational_containment':True});proof=H.reduction(center,whole,tangent,E,B,native,records,metric,tick);save('complement-480.json',proof);tick()
   result.update(passed=True,status='independently_qualified_finite_complement',parameter_endpoints=['0','1/32768'],primitive_period=18,complement_dimension=11,localization_passed=True,domain_coverage=True,complementary_modulus_upper=proof['complementary_modulus_upper'],scope=proof['scope'])
 except BaseException as exc:result.update(status='failed_or_interrupted',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
