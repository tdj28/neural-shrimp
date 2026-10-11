"""Fixed zero-native independent replay of the six earned Clark51 tubes."""
import gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
import review_clark_correlated_acquisitions as A
H,M,need=A.H,A.M,A.need
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/data2/neural-shrimp-data/explore/clark-rnn/correlated-fold-bridge/attempt002')
SOURCE='d5dcb8537c95727ac75a5c1e3617f5b868c60c96';PINS=['1969c37c7b7864a55ca8c714e523e9fe64e82b8301062dcfe5ff5934a514e15e','b32b5fa607afa0df8127e41c53875162a603160491ea94f5010851c1424c601d']
OUT=Path('/data2/neural-shrimp-data/audits/clark-correlated-fold-independent/attempt001/run')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
EXPECTED={'native_even_odd_phase':78,'explicit_organizer_anchor':1,'explicit_family_attempt':25,'explicit_amplitude_fold_return':25,'correlated_center_newton_proposal':16,'path_attempt':40,'clark_gradient_path_input':120,'clark_gradient_path_return':120,'clark_correlated_path_return':40,'quadratic_predictor_proposal':8,'correlated_radius_proposal':8,'cached_residual_reuse':16,'correlated_explicit_tube_return':8,'context_attempt':6,'paired_native_phase':36,'validated_inverse':24,'paired_native_result':6}
FILES=['tools/reviews/review_clark_correlated.py','tools/reviews/review_clark_correlated_math.py','tools/reviews/review_clark_correlated_acquisitions.py','tools/reviews/review_clark_organizer_math.py','tools/launch_clark_correlated_review.py','tests/reviews/test_clark_correlated_review.py','tests/reviews/test_clark_correlated_acquisitions.py','tests/reviews/test_clark_correlated_driver.py','docs/plans/clark-correlated-independent.md','docs/reviews/clark-correlated-fold-attempt002.json']

def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:raw=f.read(64*1024**2+1)
  need(len(raw)<=64*1024**2,'bounded saved decompression')
 else:raw=p.read_bytes();need(len(raw)<=64*1024**2,'bounded saved metadata')
 return json.loads(raw)

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while chunk:=f.read(1024**2):h.update(chunk)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())

def serial(v):
 if isinstance(v,arb):
  need(v.is_finite(),'finite saved independent ball');return dict(mid=[int(x)for x in v.mid().man_exp()],rad=[int(x)for x in v.rad().man_exp()],display=v.str(22))
 if isinstance(v,arb_mat):return [[serial(x)for x in row]for row in v.tolist()]
 if isinstance(v,dict):return {k:serial(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [serial(x)for x in v]
 return v

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False
 def tick(self):need(not self.stopped and time.monotonic()-self.start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'180s/2GiB independent cap')
 def save(self,n,v,closing=False):
  raw=(json.dumps(serial(v),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
  if n.endswith('.gz'):raw=gzip.compress(raw,compresslevel=1,mtime=0)
  need(self.used+len(raw)<(128 if closing else 124)*1024**2,'128MiB4MiB closing reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.tick()

def inventory(root,pin,w):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable inventory');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(actual),'complete immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'actual immutable bytes '+n);w.tick()

def archive(root,source,expected_exit,w):
 rt=read(root/'run/runtime.json');la=read(root/'launcher-result.json');need(rt['source_commit']==la['source_commit']==source and la['exit_code']==expected_exit and not la['owned_live_pids'],'complete literal historical source/exit')
 for pid in(rt['pid'],la['launcher_pid'],la['owned_group_pid']):need(not Path('/proc',str(pid)).exists(),'historical owned PID absent')
 actual={str(p.relative_to(root/'run/source')):p for p in(root/'run/source').rglob('*')if p.is_file()};need(set(actual)==set(rt['sources']),'complete historical source archive')
 for n,pin in rt['sources'].items():need(entry(actual[n])==pin and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'historical committed source '+n);w.tick()
 return rt

def rational(x):return arb(str(x))
def covered(raw,x):
 # Arb endpoints are dyadic at working precision; compare exact rationals.
 def q(v):return A.dyadic([int(k)for k in v.man_exp()])
 return A.covers(raw,(q(x.lower()),q(x.upper())))

def anchor_binding(record,event,binding):
 need(event==record['proof'] and record['known']==event['known'] and event['known'][:26]==binding['organizer_box'],'literal actual anchor input equals bound organizer')

def metadata(w):
 for p,pin in zip((PARENT,PARENT/'run'),PINS,strict=True):inventory(p,pin,w)
 rt=archive(PARENT,SOURCE,2,w);run=PARENT/'run';out=read(run/'outcome.json');cont=read(run/'continuation.json')
 need(not out['passed']and out['error']=='ValueError: bounded correlated critical range incomplete'and out['calls']==dict(algebra=24,anchor=1,cache_reuses=16,context=6,events=577,explicit=25,implicit=0,native_phase_returns=234,path=40),'literal bounded source outcome')
 need(not cont['passed']and len(cont['attempts'])==8 and cont['accepted']==6 and A.point(cont['last_parameter'])==F(3,8192)and [x['attempt']for x in cont['attempts']if x['passed']]==list(range(3,9)),'fixed six accepted prefix and two failed attempts')
 paths=sorted((run/'events').iterdir());kinds=[p.name.split('-',1)[1].split('.json')[0]for p in paths];need(len(paths)==577 and [int(p.name[:4])for p in paths]==list(range(1,578))and Counter(kinds)==EXPECTED,'all577 exact acquired events')
 card=read(run/'source/docs/plans/clark-correlated-fold-inputs.json')
 for key in('profile','failed_correlated','failed_explicit','diagnosis'):
  s=card[key];p=Path(s['root']);inventory(p,s['outer'],w);inventory(p/'run',s['worker'],w);archive(p,s['source'],0 if key in('profile','diagnosis')else 2,w)
 binding=read(run/'input-binding.json');anchor_binding(read(run/'explicit-anchor.json'),read(paths[3]),binding);organizer=binding['card']['independent'];p=Path(organizer['root']);inventory(p,organizer['outer'],w);inventory(p/'run',organizer['worker'],w);org_rt=archive(p,organizer['source'],0,w);oo=read(p/'run/outcome.json');fresh=read(p/'run/fresh-root.json');same=read(p/'run/same-root.json');unfold=read(p/'run/unfolding.json')
 need(oo['passed']and oo['primitive_period']==6 and oo['complement_dimension']==11 and fresh['passed']and len(fresh['coordinate_inclusions'])==26 and all(fresh['coordinate_inclusions'])and same['all26_original_boxes_strictly_inside'],'exact independent same normalized c3zero organizer')
 need(read(p/'run/input-binding.json')['map']==binding['map'],'same exact native matrix/physical map');known=[M.decode(x)for x in fresh['box']];need(len(binding['organizer_box'])==26 and all(covered(x,y)for x,y in zip(binding['organizer_box'],known,strict=True)),'whole known organizer covered')
 matrix=binding['card']['matrix'];need(entry(Path(matrix['path']))==matrix['entry'],'same fixed original matrix bytes')
 bridge=read(run/'fold-input-binding.json')['card']['bridge'];bp=Path(bridge['root']);inventory(bp,bridge['outer'],w);inventory(bp/'run',bridge['worker'],w);archive(bp,bridge['source'],0,w);need(read(bp/'run/outcome.json')['passed'],'original exact local critical bridge authority')
 w.save('input-binding.json',dict(root=str(PARENT),source=SOURCE,outer=PINS[0],worker=PINS[1],runtime=rt,organizer=organizer,bridge=bridge,matrix=matrix,map=binding['map'],events=EXPECTED,scope='Native analytic acquisitions and Arb shared; independent coefficient/root/endpoint/native reductions, zero new model calls.'))
 return paths,kinds,known,M.decode(unfold['multiplier_leak_derivative']),binding

def anchor(paths,known,crossing,w):
 rows=[read(paths[k])for k in range(3)];need([x['phase']for x in rows]==[0,1,2],'all anchor native phases');last=rows[-1];n=12;K=25;N=26;A0=arb_mat(K,K);rhs=arb_mat(K,1)
 for i,x in enumerate(last['even']+last['odd']):
  need(len(x['g'])==len(x['m'])==27,'full anchor derivative columns');offset=i if i<n else i+1
  for k in range(K):A0[i,k]=-M.decode(x['g'][k])+int(k==offset)*(-1 if i<n else 1)
  rhs[i,0]=M.decode(x['g'][N])
 for i in range(n):A0[24,13+i]=known[13+i]
 z=H.anchor(A0,rhs,known,crossing);w.save('independent-anchor.json.gz',dict(known=z,paired_jacobian=A0,rhs=rhs,lambda_leak=crossing,scope='Exact normalized c3zero known organizer implies vleak=0; own validated25 inverse encloses all remainingv.'));return z

def path_group(paths,start,matrix):
 request=read(paths[start]);need(request['matrix']==matrix,'literal original matrix in every acquisition');inputs=[];returns=[]
 for k in range(3):
  need('clark_gradient_path_input'in paths[start+1+2*k].name and 'clark_gradient_path_return'in paths[start+2+2*k].name,'complete raw path order');inputs.append(read(paths[start+1+2*k]));returns.append(read(paths[start+2+2*k]))
 last=read(paths[start+7]);need('clark_correlated_path_return'in paths[start+7].name and all(last[k]==request[k]for k in('center','slope','curvature','error','parameter','d')),'same actual complete path source return');F0,J=A.assemble_path(request,inputs,returns);need(all(x.contains(0)for row in(F0-M.matrix(last['value_coefficients'])).tolist()for x in row)and all(x.contains(0)for a,b in zip(J,last['jacobian_coefficients'],strict=True)for row in(a-M.matrix(b)).tolist()for x in row),'independent full51 assembly consistency');return request,F0,J

def qualify_native(paths,start,t,data,w,tag):
 req=read(paths[start]);n=12;c,s,a=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature')];b=M.decode(t['parameter']);h=M.decode(t['halfwidth']);d=arb(0,h);whole=[c[i]+s[i]*d+a[i]*d*d/2+data['image'][i,0]for i in range(51)]
 need(len(req['input'])==26 and len(req['paired_tangent'])==25 and all(covered(x,y)for x,y in zip(req['input']+[v[0]for v in req['paired_tangent']],whole,strict=True)),'actual native whole51 domain covers independent root image');lo=b-h;hi=b+h;need(req['support']is not None and len(req['support'])==2 and A.point(req['support'][0])==A.point(serial(lo))and A.point(req['support'][1])==A.point(serial(hi))and covered(req['squared_amplitude'],lo)and covered(req['squared_amplitude'],hi),'exact supported squared amplitude interval')
 amplitude=lo.sqrt().union(hi.sqrt());bb=b+d;z=whole[:26];v=whole[26:];q=z[13:25];need(lo>=0 and z[25]>0 and 0<z[12]<1 and sum((x*x for x in q),arb(0))>0,'actual physical gain/leak and nonzero paired mode');starts=[[z[i]+sign*amplitude*q[i]for i in range(n)]for sign in(1,-1)];modes=[[sign*(q[i]+2*bb*v[13+i])+2*amplitude*v[i]for i in range(n)]for sign in(1,-1)];rows=[read(paths[start+k])for k in range(1,7)];orbits=[];blocks=[]
 for partner in(0,1):
  raw=rows[3*partner:3*partner+3];need([(x['partner'],x['phase'])for x in raw]==[(partner,k)for k in range(3)],'literal actual native partner/phase order');need(all(covered(x,y)for x,y in zip(raw[0]['input'],starts[partner],strict=True)),'actual phase0 acquisition covers same root partner')
  for k in(1,2):need(raw[k]['input']==raw[k-1]['state'],'literal native state chain')
  orbits.append([[M.decode(x)for x in raw[0]['input']]]+[[M.decode(x)for x in row['state']]for row in raw]);blocks.append([M.matrix(row['jacobian'])for row in raw])
 need(all((orbits[p][3][i]+starts[1-p][i]).contains(0)for p in(0,1)for i in range(n)),'literal native F3 partner state consistency')
 contexts=[]
 for partner in(0,1):
  need(all(x.contains(0)for x in M.vec(blocks[partner][2]*blocks[partner][1]*blocks[partner][0]*M.col(modes[partner])+M.col(modes[1-partner]))),'same exact paired mode transport consistency')
  need(sum((modes[partner][i]*(1 if partner==0 else-1)*q[i]for i in range(n)),arb(0))>0,'same original normalized orientation')
  cycle=orbits[partner][:3]+[[-x for x in row]for row in orbits[1-partner][:3]];js=blocks[partner]+blocks[1-partner]
  for shift in(0,1):
   mode=modes[partner]if shift==0 else M.vec(js[0]*M.col(modes[partner]));out=H.native_context(cycle[shift:]+cycle[:shift],js[shift:]+js[:shift],mode);need(out['passed'],'all native primitive6 and11stable quotient');contexts.append(dict(partner=partner,shift=shift,**out))
 w.save(tag+'/native.json.gz',dict(passed=True,contexts=contexts,parameter=[lo,hi],positive_amplitude=bool(lo>0),scope='Exact paired identities inherited from authenticated full51 root; own native products, normalized line, primitive divisors and quotient power reductions. Shared actual native derivative returns.'));return contexts

def target(w):
 paths,kinds,known,crossing,binding=metadata(w);known=anchor(paths,known,crossing,w);matrix=read(paths[kinds.index('path_attempt')])['matrix'];need([[A.point(x)for x in row]for row in matrix]==[[F(x)for x in row]for row in binding['map']['j0']],'all exact original matrix entries')
 recent=[];left=arb(0);accepted=[];attempts=[]
 for i,kind in enumerate(kinds):
  if kind=='path_attempt':recent.append(i)
  if kind!='correlated_explicit_tube_return':continue
  attempt=len(attempts)+1;tag=f'charts/{attempt:02d}';t=read(paths[i]);need(t==read(PARENT/f'run/charts/{attempt:02d}/tube.json.gz')and len(recent)>=5,'literal complete retained tube');c,s,a,r=[[M.decode(v)for v in t[k]]for k in('center','slope','curvature','radii')];b=M.decode(t['parameter']);h=M.decode(t['halfwidth']);need(all(x.is_exact()for x in c+s+a+r+[b,h])and len(c)==len(s)==len(a)==len(r)==51,'complete exact curved geometry');found={}
  for k in recent[-5:]:
   req=read(paths[k])
   if any(req[x]!=t[y]for x,y in [('center','center'),('slope','slope'),('curvature','curvature'),('parameter','parameter')]):continue
   zero=all(A.point(v)==0 for v in req['error'])if all(A.interval(v)[0]==A.interval(v)[1]for v in req['error'])else False;center=A.interval(req['d'])==(F(0),F(0));key=(zero,center);need(key not in found,'unique actual proof acquisition')
   if not zero:need(all(A.covers(v,(-A.point(z),A.point(z)))for v,z in zip(req['error'],t['radii'],strict=True)),'same full actual error domain')
   if not center:need(A.covers(req['d'],(-A.point(t['halfwidth']),A.point(t['halfwidth']))),'whole acquired parameter covers original interval')
   found[key]=path_group(paths,k,matrix);w.tick()
  need(set(found)=={(True,True),(True,False),(False,True),(False,False)},'four complete residual/fullE acquisitions');need(found[False,True][0]['error']==found[False,False][0]['error']==t['actual_error'],'same literal fullE in center/whole Jacobian acquisitions');fc=found[True,True][1];fw=found[True,False][1];J=[found[False,True][2][0],found[False,True][2][1],found[False,False][2][2]];R=[M.col([fc[k,j]if j<3 else fw[k,3]for k in range(51)])for j in range(4)];C=[M.matrix(v)for v in t['inverse_coefficients']];data=H.reduction(C,J,R,r,h);w.save(tag+'/root.json.gz',data);decision=read(PARENT/f'run/charts/{attempt:02d}/decision.json');record=dict(attempt=attempt,root_passed=data['passed'],contraction=data['contraction'],original_accepted=decision['passed'])
  if decision['passed']:
   need(attempt in range(3,9)and data['passed']and left==b-h,'ordered original accepted closed support');join=H.join(known,left,c,s,a,b,h,r);w.save(tag+'/join.json',join);need(join['passed']and join['coordinates']==51,'all51 exact known phase/root join');need(kinds[i+1]=='context_attempt','native context immediately after accepted root');native=qualify_native(paths,i+1,t,data,w,tag);known=H.endpoint(data,c,s,a,1);left=b+h;w.save(tag+'/endpoint.json',dict(parameter=left,box=known));accepted.append(dict(attempt=attempt,parameter=left,contraction=data['contraction'],native_contexts=len(native)))
  attempts.append(record)
 need(len(attempts)==8 and len(accepted)==6 and left==arb(3)/8192,'complete exact six-prefix audit');w.save('connected-endpoint.json',dict(parameter=left,box=known,dimension=51,actual_native_period=6,organizer='independently exact same c3zero normalized phase'));return dict(passed=True,status='independent_six_correlated_native6_critical_tubes',accepted=accepted,attempts=attempts,range=['0','3/8192'],full_requested_range_completed=False,endpoint_generic_fold_proved=False,scope='Separate saved coefficient/root/native reductions. Native analytic path/DF acquisitions and Arb shared. All51 organizer/ordered endpoint joins and24 actual primitive6/+1/11stable contexts; b0 remains degenerate. No uniform genericity or12/24/global-window claim.')

def synthetic(w):
 begin=time.monotonic();n=51;C=[M.eye(n),arb_mat(n,n)];J=[M.eye(n),arb_mat(n,n),arb_mat(n,n)];R=[arb_mat(n,1)for _ in range(4)];data=H.reduction(C,J,R,[arb(1)/128]*n,arb(1)/64);need(data['passed'],'full51 synthetic strict reduction');w.save('profile-root.json.gz',data);root_time=time.monotonic()-begin;begin=time.monotonic();n=12;cycle=[[arb(i+k)for i in range(n)]for k in range(6)];block=arb_mat([[arb(int(i==k))*(arb(1)if i==0 else arb(1)/2)for k in range(n)]for i in range(n)]);out=H.native_context(cycle,[block]*6,[arb(int(i==0))for i in range(n)]);need(out['passed'],'full12 synthetic native algebra');w.save('profile-native.json.gz',out);native_time=time.monotonic()-begin;forecast=2*(8*root_time+24*native_time)+80;need(forecast<180 and 32*w.used+4*1024**2<124*1024**2,'full-size measured saved audit admission');w.save('profile.json',dict(passed=True,root_seconds=root_time,native_algebra_seconds=native_time,forecast_seconds=forecast,forecast_bytes=32*w.used+4*1024**2,target_arrays_unopened=True,native_calls=0))

def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'clean CPU1nice10 source');OUT.mkdir(parents=True,exist_ok=False);w=Work();out=dict(passed=False,status='incomplete');signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,180)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,sources={n:entry(ROOT/n)for n in FILES},pid=os.getpid(),precision=512,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,native_call_cap=0,seconds=180,rss_bytes=2*1024**3,raw_bytes=128*1024**2))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed own source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
  with ctx.workprec(512):synthetic(w);out=target(w)
 except BaseException as e:out.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);out.update(native_calls=0,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.stopped=False;w.save('outcome.json',out,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(serial(out)));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
