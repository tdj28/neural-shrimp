"""Saved full51 subset-radius screen of all ten retained wider failures.

This is a conditional inequality feasibility screen, not a native qualification
or an independent proof of the original22-tube ancestry. No model is called.
"""
import os,resource
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
import gzip,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews')]
import flint,numpy as np,scipy
from flint import arb,arb_mat,ctx
import diagnose_clark_correlated_extent as B
import review_clark_correlated as D
import clark_subset_radius_math as S
A,H,M,need=D.A,D.H,D.M,D.need
PARENT,SOURCE,PINS=B.PARENT,B.SOURCE,B.PINS
OUT=Path('/data2/neural-shrimp-data/audits/clark-subset-radius-screen/attempt002/run')
CARD='docs/plans/clark-subset-radius-screen-inputs.json'
CAPS=dict(seconds=180,rss_bytes=2*1024**3,as_bytes=4*1024**3,raw_bytes=64*1024**2,native_calls=0,precision=512,lp_seconds_each=1)
FAILURES=list(range(3,31,3))
FAILED=dict(root='/data2/neural-shrimp-data/audits/clark-subset-radius-screen/attempt001',source='a4883183b44980491509f485633811ed26672f70',outer='e0f18f29ac49a07dff10c05284cb28fee2a0ef17172de7c84b8037c39618db97',worker='c79a09912d3197f1584efacbf48d1db3e0610f6113cdb773091e7b1da615b7fe')
EXPECTED={'clark_gradient_path_input':480,'path_attempt':160,'clark_correlated_path_return':160,'native_even_odd_phase':294,'explicit_amplitude_fold_return':97,'clark_gradient_path_return':480,'explicit_family_attempt':97,'paired_native_result':22,'validated_inverse':88,'cached_residual_reuse':64,'correlated_center_newton_proposal':64,'context_attempt':22,'paired_native_phase':132,'correlated_radius_proposal':32,'quadratic_predictor_proposal':32,'correlated_explicit_tube_return':32,'explicit_organizer_anchor':1}
FILES=sorted(set(D.FILES+B.FILES+[CARD,'tools/screen_clark_subset_radii.py','tools/clark_subset_radius_math.py','tools/launch_clark_subset_radius_screen.py','tests/test_clark_subset_radius_math.py','tests/test_clark_subset_radius_driver.py','docs/plans/clark-subset-radius-screen.md','docs/reviews/clark-correlated-extent-diagnosis.json','docs/reviews/clark-correlated-extension-authority.json','docs/reviews/clark-subset-radius-screen-attempt001.json']))
read,entry=D.read,D.entry

def serial(v):
 if isinstance(v,F):return str(v)
 if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [serial(x)for x in v]
 return D.serial(v)

class Work(D.Work):
 def save(self,n,v,closing=False):
  raw=(json.dumps(serial(v),indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
  if n.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(64 if closing else 60)*1024**2,'64MiB4MiB reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.tick()

def history(spec,w,exitcode=0):
 p=Path(spec['root']);D.inventory(p,spec['outer'],w);D.inventory(p/'run',spec['worker'],w);D.archive(p,spec['source'],exitcode,w);return p

def metadata(w):
 card=read(ROOT/CARD);need(card['status']=='reviewed'and card['caps']==CAPS and card['failures']==FAILURES and card['parent']==dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1]),'literal reviewed fixed10 screen')
 need(card['failed_profile']==FAILED,'literal retained failed profile');fp=history(FAILED,w,2);fo=read(fp/'run/outcome.json');need(not fo['passed']and fo['native_calls']==0 and fo['error']=='ValueError: full51 synthetic reduction/LP/exact verification'and not list((fp/'run').glob('attempt-*')),'retained zero-target synthetic admission failure')
 cont=B.bind(w);need([r['attempt']for r in cont['attempts']if not r['passed']]==FAILURES,'all ten retained failures')
 run=PARENT/'run';original=read(run/'source/docs/plans/clark-correlated-extent-inputs.json');spec=original['independent_extension'];need(spec==card['independent_extension'],'same exact prior root authority');p=history(spec,w);old=read(p/'run/outcome.json');ep=read(p/'run/connected-endpoint.json');ib=read(run/'input-binding.json');need(old['passed']and old['native_calls']==0 and old['new_independent_tubes']==2 and old['range']==['0','1/2048']and entry(p/'run/connected-endpoint.json')==spec['endpoint']and len(ep['box'])==51 and A.point(ep['parameter'])==F(1,2048),'full independent original51 authority')
 need(read(run/'known-endpoint-binding.json')['endpoint']==ep and read(p/'run/input-binding.json')['map']==ib['map']and entry(Path(ib['card']['matrix']['path']))==ib['card']['matrix']['entry'],'same original phase/map/whole starting box')
 diag=history(card['diagnosis'],w);do=read(diag/'run/outcome.json');need(do['passed']and do['native_calls']==0 and do['attempts']==32 and do['whole_bound_failures']==10 and read(diag/'run/input-binding.json')==dict(root=str(PARENT),source=SOURCE,pins=PINS,events=2257,native_phases=906,attempts=32,accepted=22),'literal complete saved diagnosis')
 paths=sorted((run/'events').iterdir());kinds=[p.name.split('-',1)[1].split('.json')[0]for p in paths];need(len(paths)==2257 and [int(p.name[:4])for p in paths]==list(range(1,2258))and Counter(kinds)==EXPECTED,'all2257 immutable ordered events')
 w.save('screen-binding.json',dict(card=card,events=EXPECTED,map=ib['map'],initial_endpoint=ep,scope='Original accepted22 ancestry and full predecessor51 boxes remain source-authenticated producer proof authority; this screen independently reconstructs only failed-tube raw path coefficients and subset inequalities. No new native or complement claim.'))
 return paths,kinds,ib,old

def exact_projection(known,parameter,t):
 c,s,a,r=[[A.point(x)for x in t[k]]for k in('center','slope','curvature','radii')];b,h=A.point(t['parameter']),A.point(t['halfwidth']);n=len(c)
 need(n==51 and all(len(x)==n for x in(s,a,r,known))and b-h<=parameter<=b+h and all(x>0 for x in r),'full known51 box and exact scalar inside support')
 delta=parameter-b;prediction=[c[i]+s[i]*delta+a[i]*delta*delta/2 for i in range(n)];bounds=[max(abs(lo-prediction[i]),abs(hi-prediction[i]))for i,x in enumerate(known)for lo,hi in[A.interval(x)]]
 return [bounds[i]/r[i]for i in range(n)],bounds

def predecessor(attempt,t):
 root=PARENT/'run/charts'/f'{attempt-1:02d}';ep=read(root/'connected-endpoint.json');proof=read(root/'tube.json.gz');decision=read(root/'decision.json');join=read(root/'join.json');summary=B.diagnose(proof)
 need(decision['passed']and summary['passed_saved']and summary['inclusions']==51 and summary['saved_contraction']<1 and join['passed']and join['coordinates']==51,'entire source-authenticated predecessor proof and join')
 need(len(ep['box'])==51 and ep['box']==proof['endpoints'][1]and A.point(ep['parameter'])==A.point(proof['parameter'])+A.point(proof['halfwidth'])==A.point(t['parameter'])-A.point(t['halfwidth']),'literal full original endpoint, not center shortcut')
 anchor,bounds=exact_projection(ep['box'],A.point(ep['parameter']),t)
 return anchor,dict(attempt=attempt-1,endpoint_file=entry(root/'connected-endpoint.json'),proof_file=entry(root/'tube.json.gz'),full_original_endpoint=ep,projection_absolute_bounds=bounds,original_root_summary={k:summary[k]for k in('passed_saved','inclusions','saved_contraction')})

def full_error_reduction(paths,recent,t,matrix,w):
 found={};r=[M.decode(x)for x in t['radii']]
 need(len(r)==51 and all(x.is_exact()and x>0 for x in r),'all51 exact original radii')
 for start in recent[-5:]:
  req=read(paths[start])
  if any(req[k]!=t[k]for k in('center','slope','curvature','parameter')):continue
  zero=all(A.interval(v)==(F(0),F(0))for v in req['error']);center=A.interval(req['d'])==(F(0),F(0));key=(zero,center);need(key not in found,'unique actual residual/fullE groups')
  need(len(req['error'])==51,'all51 actual error coordinates')
  if not zero:need(all(A.covers(v,(-A.point(z),A.point(z)))for v,z in zip(req['error'],t['radii'],strict=True)),'same full original acquired E')
  if not center:need(A.covers(req['d'],(-A.point(t['halfwidth']),A.point(t['halfwidth']))),'whole actual parameter support')
  found[key]=D.path_group(paths,start,matrix);w.tick()
 need(set(found)=={(True,True),(True,False),(False,True),(False,False)}and found[False,True][0]['error']==found[False,False][0]['error']==t['actual_error'],'four complete same-fullE groups')
 fc,fw=found[True,True][1],found[True,False][1];J=[found[False,True][2][0],found[False,True][2][1],found[False,False][2][2]];R=[M.col([fc[i,k]if k<3 else fw[i,3]for i in range(51)])for k in range(4)];C=[M.matrix(x)for x in t['inverse_coefficients']]
 need(len(C)==2 and all(x.is_exact()for m in C for row in m.tolist()for x in row),'same fixed finite C0/C1')
 # Preserve original enclosure as well as own 512-bit raw-return reduction.
 data=H.reduction(C,J,R,r,M.decode(t['halfwidth']))
 Aout=[[max(B.bound(D.serial(data['defect'][i,j])),B.bound(t['whole_defect'][i][j]))*A.point(t['radii'][j])/A.point(t['radii'][i])for j in range(51)]for i in range(51)]
 beta=[max(B.bound(D.serial(data['forcing'][i,0])),B.bound(t['preconditioned_residual'][i][0]))/A.point(t['radii'][i])for i in range(51)]
 return Aout,beta,dict(precision=512,inverse_coefficients=t['inverse_coefficients'],actual_error=t['actual_error'],halfwidth=t['halfwidth'],defect=data['defect'],forcing=data['forcing'],own_original_radius_contraction=data['contraction'],scope='Independent raw phase assembly, fixed original C polynomial, noncommuting convolution/parity evaluation on actual unchanged fullE. Scalar LP uses max of original and independent absolute envelopes, never clips either.')

def profile(w,old):
 start=time.monotonic();n=51;I=M.eye(n);Z=arb_mat(n,n);J0=arb_mat([[arb(int(i==j))*arb(7)/8+arb(1,1)/2**25 for j in range(n)]for i in range(n)]);data=H.reduction([I,Z],[J0,Z,Z],[M.col([arb(1)/128]*n)]+[arb_mat(n,1)]*3,[arb(1)]*n,arb(1)/16)
 aa=[[B.bound(D.serial(data['defect'][i,j]))for j in range(n)]for i in range(n)];beta=[B.bound(D.serial(data['forcing'][i,0]))for i in range(n)];anchor=[F(1,4)]*n;proposal=S.propose(aa,beta,anchor)
 before=w.used;w.save('profile-full51.json.gz',dict(A=aa,beta=beta,anchor=anchor,result=proposal));need(proposal['verification'] and proposal['verification']['passed'],'full51 synthetic reduction/LP/exact verification');unit=time.monotonic()-start;cost=max(2*10*(unit+1)+30,2*10*old['elapsed_seconds']+30);raw=max(2*10*(w.used-before)+8*1024**2,2*10*1024**2)
 w.save('profile.json',dict(unit_seconds=unit,forecast_seconds=cost,forecast_bytes=raw,actual_old_extension_seconds=old['elapsed_seconds'],passed=cost<180 and raw<60*1024**2,target_arrays_unopened=True,native_calls=0));need(cost<180 and raw<60*1024**2,'full51 measured admission with old actual raw-assembly floor')

def target(w,paths,kinds,ib):
 matrix=read(paths[kinds.index('path_attempt')])['matrix'];need([[A.point(x)for x in row]for row in matrix]==[[F(x)for x in row]for row in ib['map']['j0']],'same original literal matrix');recent=[];attempt=0;rows=[]
 for i,kind in enumerate(kinds):
  if kind=='path_attempt':recent.append(i)
  if kind!='correlated_explicit_tube_return':continue
  attempt+=1
  if attempt not in FAILURES:continue
  t=read(paths[i]);root=PARENT/'run/charts'/f'{attempt:02d}';need(t==read(root/'tube.json.gz')and not t['passed']and not read(root/'decision.json')['passed']and A.point(t['halfwidth'])==F(1,2**14),'literal complete original wider failure');original=B.diagnose(t);anchor,known=predecessor(attempt,t);aa,beta,reduction=full_error_reduction(paths,recent,t,matrix,w);tag=f'attempt-{attempt:02d}'
  # Inputs/old failure and full known box are durable before proposing/deciding.
  w.save(tag+'/original.json',dict(original=original,known=known,proof_file=entry(root/'tube.json.gz'),full_error=t['actual_error'],old_radii=t['radii']))
  w.save(tag+'/reduction.json.gz',reduction);w.save(tag+'/bounds.json.gz',dict(A=aa,beta=beta,anchor=anchor,original_radii=[A.point(x)for x in t['radii']]))
  proposal=S.propose(aa,beta,anchor,observe=lambda record:w.save(tag+'/proposal.json',record));verification=proposal['verification'];w.save(tag+'/verification.json',dict(verification=verification,scope=proposal.get('scope','Floating LP is a proposal; exact rational verification alone decides.')))
  passed=bool(verification and verification['passed']);rows.append(dict(attempt=attempt,subset_inequalities_passed=passed,original_contraction=original['saved_contraction'],floating_success=bool(proposal['proposal'].get('success')),status='exact_subset_root_and_full_known_box_inequalities'if passed else'no_verified_subset_proposal',contraction=None if verification is None else verification.get('contraction'),minimum_radius_ratio=None if verification is None else min(verification['proposal']),maximum_radius_ratio=None if verification is None else max(verification['proposal'])));w.tick()
 need(attempt==32 and [r['attempt']for r in rows]==FAILURES,'all ten wider failures processed, no selection')
 return dict(passed=True,status='completed_fixed_saved_subset_screen',native_calls=0,attempts=rows,verified=sum(r['subset_inequalities_passed']for r in rows),scope='Exact rational subset root/known-box inequalities on independently assembled unchanged acquired domains. Original known-root ancestry is source-authenticated producer proof authority; independent22-prefix audit remains separate. No new native primitive/stability/genericity/full-window claim; failed LP is not infeasibility.')

def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();out=dict(passed=False,native_calls=0)
 def stop(*_):raise TimeoutError('fixed180s saved screen')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,180)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),caps=CAPS))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed closure');need(w.used+len(raw)<60*1024**2,'source within raw reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(512):paths,kinds,ib,old=metadata(w);profile(w,old);out=target(w,paths,kinds,ib)
 except BaseException as e:out.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);out['passed']=bool(out['passed']and out['seconds']<180 and out['peak_rss_kib']<2*1024**2);w.stopped=False;w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:serial(out.get(k))for k in('passed','seconds','verified','error')}));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
