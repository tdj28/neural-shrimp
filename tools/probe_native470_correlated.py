"""One measured full-error correlated signed470 chart from an independent root."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gzip,hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
import flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_native470_first_flip as C
from neural_shrimp.rigorous_signed_correlated import signed_correlated_tube
from neural_shrimp.rigorous_signed_proposal import refresh_signed_proposal
from neural_shrimp.rigorous_signed_gradient_path import signed_jacobian_path,GradientPathJet
from neural_shrimp.native470_first_components import encode,SyntheticFlip,synthetic_fixture
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json,canonical_bytes
from neural_shrimp.rigorous_eigenpair_locus import phase,exact_mid
from neural_shrimp.rigorous_eigenpair_path import path_coefficients
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball
need,entry=C.need,C.entry
def read(p):
 with (gzip.open(p,'rb')if p.suffix=='.gz'else p.open('rb'))as f:raw=f.read(256*1024**2+1)
 need(len(raw)<=256*1024**2,'bounded256MiB JSON decode');return json.loads(raw)
OUT=C.OUT.parents[2]/'signed-flip-correlated001/attempt003/run';CARD='docs/plans/native470-correlated-inputs.json'
SETTINGS=dict(period=18,sigma=-1,halfwidth_exponent=-3,radius_exponent=-12,newton_steps=2,precision=256,parameter='-49/256',known_parameter='-17/256')
LIMITS=dict(seconds=240,rss_bytes=4*1024**3,as_bytes=8*1024**3,raw_bytes=768*1024**2,profile_ticks=186,target_ticks=150,native_calls=324)
FILES=sorted(set(C.FILES+[CARD,'docs/plans/native470-correlated.md','tools/probe_native470_correlated.py','tools/launch_native470_correlated.py','tests/test_native470_correlated_driver.py','tests/test_signed_gradient_path.py','tests/test_signed_correlated.py','tests/test_signed_proposal.py','docs/reviews/native470-correlated-profile001.json','docs/reviews/native470-correlated-profile002.json']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def jet(v):return dict(kind='gradient_path',**v.serialized())if isinstance(v,GradientPathJet)else encode(v)
class Retained:
 def __init__(self,model,observe):self.model=model;self.n=model.n;self.observe=observe;self.calls=0
 def cusp_step(self,x,h,o):
  i=self.calls;self.calls+=1;self.observe('native_input',dict(index=i,inputs=[jet(v)for v in list(x)+[h,o]]));ret=self.model.cusp_step(x,h,o);self.observe('native_return',dict(index=i,returns=[jet(v)for v in ret]));return ret
class Work(C.Work):
 def tick(self):self.ticks+=1;need(self.ticks<=336,'186profile+150target inclusive ticks');self.check()
 def save(self,name,row,closing=False):
  old=C.OUT;C.OUT=OUT
  try:return super().save(name,row,closing)
  finally:C.OUT=old
 def observe(self,kind,row):
  if kind=='native_input':self.native+=1;need(self.native<=324,'all324 inclusive native acquisitions')
  if kind=='signed_correlated_chart_return':self.save(self.stage+'-certificate.json.gz',row)
  else:self.save(f'events/{self.serial:04d}-{self.stage}-{kind}.json.gz',row);self.serial+=1

def chart(model,initial,normal,origin,parameter,weights,anchor,settings,w):
 z=[exact(v)for v in initial];n=[exact(v)for v in normal];o=[exact(v)for v in origin];p=exact(parameter)
 proposal=refresh_signed_proposal(model,z,period=18,sigma=-1,normal=n,origin=o,parameter=p,newton_steps=2,tick=w.tick,observe=w.observe)
 return signed_correlated_tube(model,proposal['center'],proposal['slope'],proposal['curvature'],proposal['C0'],period=18,sigma=-1,normal=n,origin=o,parameter=parameter,halfwidth_exponent=settings['halfwidth_exponent'],radius_exponent=settings['radius_exponent'],weights=[exact(v)for v in weights],anchor=anchor,tick=w.tick,observe=w.observe)

def profile(w):
 retained_failure(w);retained_second_failure(w)
 ini,kw=synthetic_fixture();w.save('profile-fixture.json',dict(initial=ini,settings={k:v for k,v in kw.items()if k!='anchor'},scope='Unrelated full470 repeated primitive2, false primitive18'))
 used=w.used;t=time.monotonic();r=chart(Retained(SyntheticFlip(),w.observe),ini,kw['normal'],ini,'0',['1']*470,kw['anchor'],kw,w);charttime=time.monotonic()-t
 need(r['gates']['uniform_normalized_eigenpair_locus']and r['gates']['anchor_identified']and not r['gates']['primitive_period'],'full470 synthetic proof/anchor and false primitive18');del r
 inp,z=C.L.K.fixture();z=[v.mid()for v in z];n=13;x=z[:n];q=z[n+2:2*n+2];rates=z[n:n+2];w.save('profile-native-fixture.json',dict(map_inputs=inp,state=vector_json(x),mode=vector_json(q),rates=vector_json(rates)))
 model=Retained(C.TwoRateCuspMap(inp),w.observe);initial=x+q+rates;slope=[exact(str(F(i-14,128)))for i in range(28)];curve=[exact(str(F(7-i,256)))for i in range(28)];normal=[arb(0)]*26+[arb(1),arb(0)];delta=arb(0,arb((1,-14)));err=[arb(0,arb((1,-20)))]*28;times={k:[]for k in ('mixed','gradient','path')}
 for k in range(12):
  w.tick();t=time.monotonic();v,j,e=phase(model,x,q,rates);w.save(f'profile-mixed-{k}.json.gz',dict(value=matrix_json(v),block=matrix_json(j),rates=matrix_json(e)));times['mixed'].append(time.monotonic()-t)
  t=time.monotonic();jj=signed_jacobian_path(model,initial,slope,curve,err,delta,1,-1,normal,tick=w.tick,observe=w.observe,retain_assembly=False);del jj;times['gradient'].append(time.monotonic()-t)
  t=time.monotonic();pp=path_coefficients(model,initial,slope,curve,delta,1,-1,normal,initial,arb(0),w.tick,w.observe);w.save(f'profile-path-{k}.json.gz',dict(coefficients=matrix_json(pp)));times['path'].append(time.monotonic()-t)
 seconds=time.monotonic()-w.start+2*(charttime+54*max(times['mixed'])+36*max(times['gradient'])+54*max(times['path']))+15;raw=4*(w.used-used)+32*1024**2;rss=2*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+256*1024
 ok=seconds<240 and raw<768*1024**2 and rss<4*1024**2
 w.save('profile.json',dict(passed=ok,chart_seconds=charttime,times=times,forecast_seconds=seconds,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True,ticks=w.ticks,native_calls=w.native));need(ok and w.ticks==186 and w.native==180,'complete full470 same-source measured profile before target')

def history(w,spec,expected_exit=0):
 p=Path(spec['root'])
 for b,k in ((p,'outer'),(p/'run','worker')):C.L.K.inventory(b,spec[k]['sha256'],w)
 rt=read(p/'run/runtime.json');la=read(p/'launcher-result.json');owner=read(p/'owned-process.json');group=owner.get('group_pid',owner.get('pid'))
 need(type(group)is int and la['source_commit']==rt['source_commit']==spec['source']and la['exit_code']==expected_exit and la['failure']is None and not la['owned_live_pids']and la['worker_pid_absent'],'complete successful source and ownership')
 need(la.get('owned_group_pid',group)==group and all(not Path('/proc',str(pid)).exists()for pid in [group,la.get('launcher_pid',-1)]),'historical owned processes gone')
 for n,v in rt['sources'].items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'historical exact source '+n);w.check()
 return p,rt

def retained_failure(w):
 spec=read(ROOT/'docs/reviews/native470-correlated-profile001.json');p,rt=history(w,spec,2);o=read(p/'run/outcome.json')
 need(spec['root']==str(OUT.parents[1]/'attempt001')and spec['source']=='2fb0cd3ebd06331f85cb697a49b70a5440aaa896','literal first profile source/path')
 need(o==spec['outcome']and not o['passed']and not o['target_acquired']and o['native_calls']==145 and o['ticks']==151 and o['error']=='ValueError: exact finite predictor and gauge'and not list((p/'run/events').glob('*-target-*')),'immutable failed exact-fixture profile before target')
 w.save('retained-profile-failure.json',dict(source=spec['source'],outer=spec['outer'],worker=spec['worker'],outcome=o))

def retained_second_failure(w):
 spec=read(ROOT/'docs/reviews/native470-correlated-profile002.json');p,rt=history(w,spec,2);o=read(p/'run/outcome.json');pr=read(p/'run/profile.json')
 need(spec['root']==str(OUT.parents[1]/'attempt002')and spec['source']=='a3f1d2484f34dcc22d7010159429ec0623c3fccf','literal second failed binding source/path')
 need(o==spec['outcome']and pr==spec['profile']and pr['passed']and not o['passed']and not o['target_acquired']and o['native_calls']==180 and o['ticks']==186 and o['error'].startswith('UnicodeDecodeError:')and not list((p/'run/events').glob('*-target-*')),'completed profile and zero-target decoder failure retained')
 w.save('retained-profile002-failure.json',dict(source=spec['source'],outer=spec['outer'],worker=spec['worker'],outcome=o))

def binding(w,card,metadata_only=False):
 need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'fixed reviewed wider correlated design')
 ap,ar=history(w,card['independent']);p,pr=history(w,card['producer']);ao=read(ap/'run/outcome.json');po=read(p/'run/outcome.json');ib=read(ap/'run/input-binding.json')
 need(ao==card['independent']['outcome']and ao['passed']and ao['root_coordinates']==ao['anchor_coordinates']==470 and ao['native_calls']==0,'complete independent advancing endpoint authority')
 need(po==card['producer']['outcome']and po['passed']and po['strict_inclusions']==po['anchor_inclusions']==470 and po['native_calls']==276,'complete original advancing acquisition')
 need(ib['card']['producer']==card['producer']and ib['card']['known']==card['known_parent'],'literal entire independent predecessor genealogy')
 for path,k in [('connected-left-endpoint.json','endpoint'),('independent512.json','proof')]:need(entry(ap/'run'/path)==card['independent'][k],'literal independent file '+path)
 for path,k in [('target-inputs.json','inputs'),('target-certificate.json.gz','certificate')]:need(entry(p/'run'/path)==card['producer'][k],'literal acquired file '+path)
 for n,v in ib['native_source'].items():need(pr['sources'][n]==v==entry(ROOT/n),'identical actual native equations '+n)
 total=sum(v['bytes']for v in read(p/'run/manifest.json')['files'].values());floor=dict(seconds=time.monotonic()-w.start+2*po['seconds']+15,rss_kib=2*po['peak_rss_kib']+256*1024,bytes=2*total+32*1024**2)
 need(floor['seconds']<240 and floor['rss_kib']<4*1024**2 and floor['bytes']<768*1024**2,'actual complete native470 resource floor');w.save('actual-cost-floor.json',floor)
 if metadata_only:return p,ap
 inp=read(p/'run/target-inputs.json');known=read(ap/'run/connected-left-endpoint.json');audit=read(ap/'run/independent512.json');old=read(p/'run/target-certificate.json.gz')
 need(known['box']==audit['recomputed_endpoints'][0]and len(known['box'])==470 and audit['passed']and audit['inclusions']==audit['anchor_coordinates']==470,'literal full470 independent known root')
 need(known['exact_parameter']==SETTINGS['known_parameter']and known['normal']==inp['normal']==old['normal']==['0']*468+['1','0']and known['origin']==inp['origin']==old['origin']and known['period']==18 and known['sigma']==-1 and known['dimension']==13,'same exact phase/sign/native hidden gauge')
 need(inp['map_inputs']==ib['model']and old['weights']==inp['weights']==card['weights']and old['parameter_center']==inp['parameter']=='-9/256'and old['parameter_endpoints'][0]==SETTINGS['known_parameter'],'same actual map/weights/old chart')
 need(F(SETTINGS['parameter'])+F(2)**SETTINGS['halfwidth_exponent']==F(known['exact_parameter']),'new right endpoint is literal old left endpoint')
 with ctx.workprec(512):box=[original_ball(v)for v in known['box']]
 with ctx.workprec(256):
  d=exact(str(F(SETTINGS['parameter'])-F(old['parameter_center'])));initial=[exact_mid(original_ball(c)+original_ball(s)*d+original_ball(a)*d*d/2)for c,s,a in zip(old['center'],old['slope'],old['curvature'],strict=True)]
 anchor=dict(box=box,exact_parameter=known['exact_parameter'],period=18,sigma=-1,dimension=13)
 w.save('input-binding.json',dict(independent=card['independent'],producer=card['producer'],native_source=ib['native_source'],map_inputs=inp['map_inputs'],known=known,parameter=SETTINGS['parameter'],normal=known['normal'],origin=known['origin'],settings=SETTINGS,weights=card['weights'],scope='Old extrapolation is finite only; all new derivatives and strict whole470 root/join required.'))
 w.save('target-inputs.json',dict(initial=initial,anchor=known,map_inputs=inp['map_inputs'],normal=known['normal'],origin=known['origin'],parameter=SETTINGS['parameter'],weights=card['weights'],settings=SETTINGS));return inp['map_inputs'],initial,known,anchor

def main():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();ans=dict(passed=False,target_acquired=False)
 def stop(*_):raise TimeoutError('240s owned correlated470')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),settings=SETTINGS,limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'committed source '+n);pp=OUT/'source'/n;pp.parent.mkdir(parents=True,exist_ok=True);pp.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(256):profile(w);card=read(ROOT/CARD);inp,initial,known,anchor=binding(w,card);w.stage='target';ans['target_acquired']=True;r=chart(Retained(C.TwoRateCuspMap(inp),w.observe),initial,known['normal'],known['origin'],SETTINGS['parameter'],card['weights'],anchor,SETTINGS,w)
  need(w.ticks==336 and w.native==324,'entire fixed profile and target acquisition');ans.update(passed=r['passed'],status='connected_correlated_signed470_passed'if r['passed']else'complete_bound_failure',gates=r['gates'],contraction=r['bound']['contraction'],strict_inclusions=sum(r['bound']['coordinate_inclusions']),anchor_inclusions=sum(r['anchor_check']['coordinate_inclusions']),anchor_parameter_inside=r['anchor_check']['parameter_inside'],parameter_endpoints=r['parameter_endpoints'],physical_endpoints=[v[-2:]for v in r['endpoint_root_boxes']],scope='Primitive18 negative-one locus with full known-root join. No complementary stability, generic flip or complete window claim.')
 except BaseException as e:ans.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if elapsed>=240 or rss>=4*1024**2:ans['passed']=False
  ans.update(seconds=elapsed,peak_rss_kib=rss,ticks=w.ticks,native_calls=w.native,events=w.serial);w.save('outcome.json',ans,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(ans));return 0 if ans['passed']else 2
if __name__=='__main__':raise SystemExit(main())
