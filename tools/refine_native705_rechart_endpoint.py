#!/usr/bin/env python3
"""Reusable full-source known-endpoint refinement; fixed retained root5 pilot."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import flint
from flint import arb,arb_mat,ctx
import launch_native705_maintained as L
import localize_cubic705_endpoint_point as P
from neural_shrimp import rigorous_saved_cubic_endpoint as R
from neural_shrimp.rigorous_cubic_endpoint import localize_fresh_endpoint
from neural_shrimp.rigorous_cubic_gradient_path import cubic_jacobian_coefficients
from neural_shrimp.rigorous_unit_pair_pal import assemble
from neural_shrimp.rigorous_unit_pair import phase
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,eye,ball_json,vector_json,matrix_json
W=L.W;need=W.need;read=W.read;entry=W.entry
BASE=W.BASE.parent/'maintained-fallback001';OUT=BASE/'attempt002/run';CAP=512*1024**2
CARD='docs/plans/native705-rechart-refinement-inputs.json'
SOURCES=list(dict.fromkeys(W.SOURCES+['src/neural_shrimp/rigorous_saved_cubic_endpoint.py','tools/refine_native705_rechart_endpoint.py','tools/launch_native705_rechart_refinement.py','tests/test_saved_cubic_endpoint.py','tests/test_native705_rechart_refinement.py','docs/plans/native705-reusable-endpoint.md','docs/reviews/native705-maintained001.json','docs/reviews/native705-rechart-refinement-profile001.json',CARD]))

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.native=0;self.saved=0;self.events=0;self.stopped=False
 def check(self):need(not self.stopped and time.monotonic()-self.start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'300s/6GiB endpoint refinement cap')
 def save(self,name,value,closing=False):
  raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode();raw=gzip.compress(raw,mtime=0)if name.endswith('.gz')else raw+b'\n';need(self.used+len(raw)<CAP-(0 if closing else 4*1024**2),'512MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def tick(self):self.native+=1;need(self.native<=72,'36profile36target phases');self.check()
 def replay_tick(self):self.saved+=1;need(self.saved<=144,'72synthetic72saved replay operations');self.check()
 def observe(self,kind,value):self.save(f'events/{self.events:03d}-{kind}.json.gz',value);self.events+=1;self.check()


def exact_inverse(rows):
 need(len(rows)==705 and all(len(r)==705 for r in rows),'entire saved inverse polynomial')
 for row in rows:
  for v in row:need(type(v)is list and len(v)==2 and all(type(x)is int for x in v),'literal dyadic inverse coefficient')
 out=arb_mat([[arb(tuple(v))for v in row]for row in rows]);need(all(x.is_exact()and x.is_finite()for row in out.tolist()for x in row),'exact finite saved inverse');return out


def acquisitions(run,group):
 need(group in(0,1),'two retained fullE domains');ins=sorted((run/'events').glob('*-gradient_path_input.json.gz'));outs=sorted((run/'events').glob('*-gradient_path_return.json.gz'));need(len(ins)==len(outs)==72,'72 retained native gradient pairs');rows=[]
 for k in range(group*36,(group+1)*36):
  need(int(outs[k].name.split('-')[0])==1+int(ins[k].name.split('-')[0]),'literal adjacent input and return');rows.append({'input':read(ins[k]),'return':read(outs[k])})
 return rows


def geometry(chart,domain):
 need(chart['shooting_dimension']==705 and chart['period']==18 and chart['dimension']==13,'native705 known root');vectors=[[R.original_ball(v)for v in chart[k]]for k in('center','slope','curvature','cubic')];c,s,a,b=vectors
 need(all(v.is_exact()for row in vectors for v in row),'literal finite cubic predictor');need(all(domain[k]==chart[k]for k in('center','slope','curvature','cubic'))and domain['parameter']==chart['parameter_center']and domain['halfwidth_exponent']==chart['settings']['halfwidth_exponent']and domain['radius_exponent']==chart['settings']['radius_exponent']==-10,'same acquired cubic parameter/correction domain')
 h=arb((1,chart['settings']['halfwidth_exponent']));need(0<h<=arb(1)/8 and len(domain['error'])==705,'bounded complete correction domain');error=[R.original_ball(v)for v in domain['error']];weights=[exact(v)for v in chart['weights']];normal=[exact(v)for v in chart['normal']];origin=[exact(v)for v in chart['origin']];need(all(0<w<=1 and w.is_exact()for w in weights),'exact positive weights')
 for e,w in zip(error,weights,strict=True):need(e.contains(0)and e.lower()<=-arb((1,-10))*w and e.upper()>=arb((1,-10))*w,'actual original full error covers requested correction')
 return c,s,a,b,error,weights,normal,origin,h


def replay_operator(run,chart,domain,w):
 c,s,a,b,error,weights,normal,origin,h=geometry(chart,domain);zero=acquisitions(run,0);whole=acquisitions(run,1);d=R.original_ball(whole[0]['input']['d']);need(d.contains(-h)and d.contains(h)and R.original_ball(zero[0]['input']['d']).is_zero(),'same original whole and zero parameter acquisitions')
 J0=R.replay_jacobian(c,s,a,b,error,arb(0),18,normal,zero,tick=w.replay_tick);Jw=R.replay_jacobian(c,s,a,b,error,d,18,normal,whole,tick=w.replay_tick);J=[J0[0],J0[1],Jw[2]];del J0,Jw,zero,whole;gc.collect()
 need(chart['inverse_polynomial_encoding']=='exact-dyadic-pairs'and len(chart['inverse_polynomial_exact'])==2,'entire literal inverse polynomial');C0,C1=[exact_inverse(v)for v in chart['inverse_polynomial_exact']];C,D,q=R.endpoint_operator(J,C0,C1,h,weights,tick=w.check);del J,C0,C1;gc.collect();return C,D,q,weights,(c,s,a,b,error,normal,origin,h)


def refine_pair(model_inputs,oldrun,oldchart,newchart,known,domain,w):
 """Reusable caller-bound fallback; caller supplies the authentic old root."""
 C,D,q,weights,data=replay_operator(oldrun,oldchart,domain,w);c,s,a,b,error,normal,origin,h=data;predictor=[c[i]+s[i]*h+a[i]*h*h/2+b[i]*h*h*h for i in range(705)];box=[R.original_ball(v)for v in known];point=[x.mid()for x in box]
 P.old_domain_contains(domain['error'],predictor,box);P.old_domain_contains(domain['error'],predictor,point);parameter=exact(oldchart['parameter_endpoints'][1]);need(parameter==exact(oldchart['parameter_center'])+h,'literal exact proved right endpoint')
 w.save('endpoint-operator.json.gz',dict(C=matrix_json(C),D=matrix_json(D),weighted_bound=ball_json(q)));w.save('point-input.json',dict(model=model_inputs,point=vector_json(point),known=known,predictor=vector_json(predictor),original_error=domain['error'],normal=oldchart['normal'],origin=oldchart['origin'],parameter=oldchart['parameter_endpoints'][1]))
 residual,J,_=assemble(TwoRateCuspMap(model_inputs),point,18,normal,origin,parameter,w.tick,w.observe);del J;w.save('fresh-point-residual.json',matrix_json(residual));zero=arb_mat(705,705)
 localized,proof=localize_fresh_endpoint([D,zero,zero,zero],C,zero,residual,arb(0),h,point,box,weights,tick=w.check);w.save('localized-known-endpoint.json',proof);join=W.join({'box':vector_json(localized)},newchart);w.save('projected-join.json',join)
 return localized,join,q


def synthetic(w):
 failure=read(ROOT/'docs/reviews/native705-rechart-refinement-profile001.json');failed=BASE/'attempt001';need(failure['root']==str(failed)and failure['source']=='9248ae90876b5dc53c1b1d2b527d5ad634bfaae3'and failure['outer']['sha256']=='fc65a69bd6ec11644c02318a0fd0781f0560607740dff2fffed7d65332345fbc','literal retained time-only profile failure');W.inventory(failed,failure['outer']['sha256']);need(read(failed/'run/profile.json')==failure['profile']and read(failed/'run/outcome.json')==failure['outcome']and failure['outcome']['native_calls']==36 and failure['outcome']['saved_replay_calls']==72 and failure['profile']['target_arrays_unopened']and 240<failure['profile']['forecast_seconds']<300 and failure['profile']['forecast_bytes']<CAP and failure['profile']['forecast_peak_bytes']<6*1024**3,'unchanged successful memory/storage gates and rejected time forecast')
 W.inventory(Path(L.PARENT['root']),L.PARENT['outer']);prior=read(Path(L.PARENT['root'])/'run/outcome.json');need(prior['passed']and prior['native_calls']==72 and 0<prior['seconds']<120,'literal completed actual endpoint cost floor');floor=prior['seconds'];before=time.monotonic();used=w.used;n=13;N=705
 class Linear:
  n=13
  def cusp_step(self,x,h,o):return[(1-h/128)*v+o*sum(x)/4096 for v in x]
 c=[arb((i%13+1,-8))for i in range(702)]+[arb(1),arb(1),arb((1,-6))];s=[arb((i%5+1,-14))for i in range(N)];a=[arb((i%3-1,-18))for i in range(N)];b=[arb((i%7-3,-22))for i in range(N)];E=[arb(0,arb((1,-22)))]*N;normal=[arb(0)]*(N-1)+[arb(1)]
 for d in(arb(0),arb(0,arb((1,-6)))):
  rows=[];pending=[]
  def observe(k,v):
   if k=='gradient_path_input':pending.append(v)
   elif k=='gradient_path_return':rows.append({'input':pending[-1],'return':v})
  original=cubic_jacobian_coefficients(Linear(),c,s,a,b,E,d,18,normal,observe=observe,retain_assembly=False);replay=R.replay_jacobian(c,s,a,b,E,d,18,normal,rows,tick=w.replay_tick)
  need(all(all((original[k][i,j]-replay[k][i,j]).contains(0)for i in range(N)for j in range(N))for k in range(3)),'full705 saved-return synthetic reconstruction');del original,replay,rows,pending;gc.collect()
 A=arb_mat([[arb(((i+1)*6364136223846793005+(j+1)*1442695040888963407)**4%2**250)/arb((1,270))for j in range(N)]for i in range(N)]);I=eye(N);C,D,q=R.endpoint_operator([I-A,A/8,A/64],I+A,-A/8,arb(1)/64,[arb(1)]*N,tick=w.check);need(q<1,'unrelated full705 point operator');w.save('profile-operator.json.gz',dict(C=matrix_json(C),D=matrix_json(D),q=ball_json(q)));del A,I,C,D;gc.collect()
 inp,x,q,r,rates=P.P.gd_fixture();w.save('profile-input.json',dict(model=inp,x=vector_json(x),q=vector_json(q),r=vector_json(r),rates=vector_json(rates)))
 for k in range(18):phase(TwoRateCuspMap(inp),x,q,r,rates,w.tick,w.observe,k)
 elapsed=time.monotonic()-before;stored=w.used-used;forecast=8*elapsed+2*floor+20;raw=4*stored+64*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3
 report=dict(passed=bool(w.native==36 and w.saved==72 and forecast<300 and raw<CAP and peak<6*1024**3),seconds=elapsed,stored_bytes=stored,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,actual_prior_seconds=floor,target_arrays_unopened=True);w.save('profile.json',report);need(report['passed'],'complete measured705 replay/residual admission before target')


def bind(identity,w):
 card=read(ROOT/CARD);need(card['pilot']==dict(root=str(L.BASE),source='7edc90109fba137b9cc1ef5439cc615a273524e8',outer='590728cd05104c9c9892dc984ef09264b8124f0b326e13d7fa39d5533dff4933',old_attempt=4,new_attempt=5),'fixed retained five-attempt genealogy');root=L.BASE;W.inventory(root,card['pilot']['outer']);done=read(root/'outcome.json');need(done['source']==card['pilot']['source']and done['status']=='mathematical_stop'and done['connected_count']==3 and done['attempted']==5 and not done['owned_live_pids'],'complete retained prefix and unjoined root')
 # Initial historical root proof is source-bound by its existing complete binder.
 oldbase=L.BASE;oldsave=L.save
 try:
  L.save=lambda folder,name,value:w.save(str(folder.relative_to(OUT)/name),value)
  L.BASE=OUT/'initial-authority';L.BASE.mkdir();model,_,anchor,_=L.initial_parent(identity,w.check)
 finally:L.BASE=oldbase;L.save=oldsave
 chart=None
 for index,row in enumerate(done['attempts'],1):
  need(row['attempt']==index,'literal all-attempt order');folder=root/f'attempt{index:03d}';W.inventory(folder,row['outer_manifest']['sha256']);W.inventory(folder/'run',row['worker']['sha256']);rt=read(folder/'run/runtime.json');ic=read(folder/'input-card.json');o=read(folder/'run/outcome.json');owner=read(folder/'launcher-result.json');owned=read(folder/'owned-process.json')
  need(rt['commit']==ic['source']==card['pilot']['source']and rt['settings']==ic['settings']and o['ticks']==331 and o['events']==W.EXPECTED and not owner['owned_live_pids']and not Path('/proc',str(owned['group'])).exists(),'complete actual retained331 source/native owner')
  for n,v in rt['sources'].items():need(entry(folder/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'exact native original source '+n)
  for n in W.W.C.NATIVE_SOURCES:need(identity[n]==rt['sources'][n],'unchanged actual native map/path '+n)
  need(ic['map_inputs']==model,'same model through maintained prefix');chart=read(folder/'run/certificate.json.gz');other=L.OLD.complete_result(o,chart,index,ic['settings']);need(other and o['passed']==row['root_passed'],'exact original root gates')
  if index in(1,2,4):
   need(ic['anchor']==anchor and ic['mode']=='same_gauge','whole prior accepted anchor and same gauge');join=W.join(anchor,chart);need(join['passed']and all(join['coordinate_inclusions']),'recheck entire literal prefix identity');anchor=L.anchor_from(chart)
  elif index==3:need(not o['passed']and not row['joined'],'retained larger-width failure')
  else:need(o['passed']and not row['joined']and ic['mode']=='rotated_gauge'and ic['anchor']=={'box':anchor['box']},'retained successful root has only unresolved projected join')
  if index==4:oldchart=chart;oldrun=folder/'run'
  w.check()
 paths=list((oldrun/'events').glob('*-cubic_correlated_domain.json.gz'));need(len(paths)==1,'unique original derivative domain');domain=read(paths[0]);w.save('binding.json',dict(card=card,old_worker=done['attempts'][3]['worker'],new_worker=done['attempts'][4]['worker'],old_gauge=oldchart['normal'],old_parameter=oldchart['parameter_endpoints'][1],initial_authority=L.PARENT,scope='Source-bound complete producer existence chain; outcome independence remains separate.'))
 return model,oldrun,oldchart,chart,anchor['box'],domain


def main():
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');OUT.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,300);result=dict(passed=False,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identity={n:entry(ROOT/n)for n in SOURCES};w.save('runtime.json',dict(commit=source,sources=identity,precision=480,pid=os.getpid(),python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0)))
  for n,v in identity.items():need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed exact source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());w.used+=p.stat().st_size
  with ctx.workprec(480):
   card=read(ROOT/CARD);need(card['status']=='measured-reviewed'and card['caps']==dict(seconds=300,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=CAP,native_calls=72,saved_replay_calls=144),'literal reviewed fixed fallback card');synthetic(w);model,run,old,new,known,domain=bind(identity,w);localized,join,q=refine_pair(model,run,old,new,known,domain,w);need(w.native==72 and w.saved==144,'all fixed replay and fresh residual operations');passed=bool(join['passed']and len(join['coordinate_inclusions'])==705 and all(join['coordinate_inclusions']));w.save('known-endpoint.json',dict(chart={k:old[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=vector_json(localized)))
   if passed:w.save('connected-endpoint.json',dict(chart={k:new[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=new['endpoint_root_boxes'][1]))
   result.update(passed=passed,status='retained_root_connected'if passed else'completed_join_unresolved',strict_inclusions=sum(join['coordinate_inclusions']),weighted_bound=ball_json(q),scope='Identical known endpoint refined by source-bound saved fullE derivatives and36 fresh native phases, joining retained full705 root5. No uniform-root reacquisition, complementary or whole-window claim.')
 except BaseException as exc:result.update(status='failed_or_interrupted',failure=type(exc).__name__+': '+str(exc))
 finally:signal.setitimer(signal.ITIMER_REAL,0)
 result.update(seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native,saved_replay_calls=w.saved,events=w.events);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
