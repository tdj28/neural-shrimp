"""One saved independent output-gauge root plus original-root refinement audit."""
import os,resource
if __name__=='__main__':resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,gzip,json,platform,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_signed_correlated as S
import review_signed_march as T
import review_signed_output_math as P
import review_signed_output_curve as Q
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,vector_json,matrix_json
need,read,entry,inventory,archive=S.need,S.read,S.entry,S.inventory,S.archive
OUT=Path('/data2/neural-shrimp-data/audits/signed470-output-independent/attempt002/run');CARD='docs/plans/native470-output-independent-inputs.json'
LIMITS=dict(seconds=300,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=768*1024**2,precision=512,native_calls=0)
FILES=sorted(set(T.FILES+[CARD,'docs/plans/native470-output-independent.md','tools/reviews/review_signed_output.py','tools/reviews/review_signed_output_math.py','tools/reviews/review_signed_output_curve.py','tools/launch_signed_output_review.py','tests/reviews/test_signed_output_math.py','tests/reviews/test_signed_output_driver.py']))
class Work(S.Work):
 def save(self,n,v,closing=False):
  raw=canonical_bytes(v);need(self.used+len(raw)<(768 if closing else 764)*1024**2,'768MiB4MiBreserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
  if not closing:self.tick()
 def matrix(self,n,a):
  old=S.K.OUT;S.K.OUT=OUT
  try:return S.K.Work.matrix(self,n,a)
  finally:S.K.OUT=old

class Prefix:
 def __init__(self,w,p):self.w,self.p=w,p
 def tick(self):self.w.tick()
 def save(self,n,v,closing=False):self.w.save(self.p+n,v,closing)

def sealed(spec,w,expected=0,expected_outcome=None):
 p=Path(spec['root']);inventory(p,spec['outer']['sha256'],w);inventory(p/'run',spec['worker']['sha256'],w);rt=archive(p,spec['source'],w);la=read(p/'launcher-result.json');out=read(p/'run/outcome.json');need(la['exit_code']==expected and la['failure']is None and la['worker_pid_absent']and out==(spec['outcome']if expected_outcome is None else expected_outcome),'literal entire sealed owned result')
 for n,e in spec.get('files',{}).items():need(entry(p/'run'/n)==e,'exact retained source/input evidence '+n)
 return p,rt,out

def preserved(card,w):
 p,rt,o=sealed(card['prior_failure'],w,2)
 need(card['prior_failure']['source']=='124e184d828e1038a610779bc5535e579a2babc6'and not o['passed']and o['native_calls']==0 and o['error']=='ValueError: entire predictor and phase/rate/gauge proposal','literal prior representation-only failure')
 need(read(p/'run/profile.json')['passed']and read(p/'run/complete-admission.json')['passed']and all(read(p/f'run/endpoint-{i}-independent.json')['passed']for i in range(3))and not(p/'run/independent512.json').exists(),'retain completed three localizations and no uniform-root promotion')
 w.save('preserved-failure.json',dict(source=card['prior_failure']['source'],outer=card['prior_failure']['outer'],worker=card['prior_failure']['worker'],outcome=o))
 return o

def bind(card,w,metadata_only=False):
 need(card['status']=='reviewed'and card['limits']==LIMITS,'fixed reviewed output-gauge audit');preserved(card,w);ap,ar,ao=sealed(card['known'],w,0,card['known_outcome']);p,rt,out=sealed(card['producer'],w);fp,fr,fo=sealed(card['failed'],w,2);dp,dr,do=sealed(card['diagnosis'],w)
 ai=read(ap/'run/input-binding.json');need(ao['passed']and ao['native_calls']==0 and ao['accepted_indices']==[1,2,3,4,6,7,9,10,12,13,15,16,18,19,21,22]and ao['exact_parameter']=='-2001/256'and all(r['passed']and r['root_coordinates']==r['anchor_coordinates']==470 for r in ao['rows']),'independent complete16 full470 roots/joins')
 mp,oldap=T.bind(ai['card'],Prefix(w,'original-march-'),True);need(entry(ap/'run/connected-left-endpoint.json')==card['known']['endpoint'],'literal independent endpoint')
 hb=read(p/'run/history-binding.json');need(hb['independent']==card['known']and hb['failed_output']==card['failed']and hb['diagnosis']==card['diagnosis'],'literal all weighted-target ancestors')
 ib=read(fp/'run/input-binding.json');need(ib['known']['exact_parameter']=='-2001/256'and ib['model']==hb['model']==ai['model']and ib['native_source']==hb['native_source']==ai['native_source'],'complete same model and native equations')
 need(out['passed']and out['strict_inclusions']==out['anchor_inclusions']==470 and out['native_calls']==324 and out['ticks']==336 and out['events']==1096,'complete newly scaled full470 producer')
 need(not fo['passed']and fo['endpoint_refined']and fo['native_calls']==432 and fo['ticks']==450 and do['passed']and do['native_calls']==0,'original failed chart/refinement and numerical diagnosis retained')
 for n,e in ai['native_source'].items():need(rt['sources'][n]==fr['sources'][n]==e,'same actual native map '+n)
 for n in ('profile.json','measured-admission.json'):need(read(p/'run'/n)['passed'],'complete measured fresh admission')
 need(entry(mp/'run/attempt022/target-certificate.json.gz')==ib['chart']and entry(Path(ib['domain']['path']))=={k:ib['domain'][k]for k in('bytes','sha256')},'literal original chart22 fixed domain')
 ev=sorted((fp/'run/events').glob('*-endpoint-*.json.gz'));kinds={}
 for f in ev:k=f.name.split('-endpoint-')[1].removesuffix('.json.gz');kinds[k]=kinds.get(k,0)+1
 need(kinds==dict(native_input=54,native_return=54,phase_return=54),'all54 endpoint acquisitions retained')
 need(read(ap/'run/chart022/outcome.json')['passed']and read(ap/'run/chart022/outcome.json')['root_coordinates']==470,'selected old independent chart22')
 w.save('input-binding.json',dict(card=card,model=ai['model'],native_source=ai['native_source'],old_producer=ai['card']['producer'],old_index=22,old_parameter='-2001/256',scope='Independent exact same-root refinement and full470 output-gauge proof from authenticated native returns; no native calls.'))
 Q.geometry_header(p/'run')
 if metadata_only:return p,fp,ap,mp
 known=read(ap/'run/connected-left-endpoint.json');need(known==read(ap/'run/chart022/connected-left-endpoint.json')and known['exact_parameter']=='-2001/256'and known['normal']==['0']*468+['1','0']and len(known['box'])==470,'exact independently reconstructed old known root');return p,fp,ap,mp,known

def load_matrix(p):
 x=read(p);need(x['encoding']=='mid-rad-dyadic-pairs'and len(x['rows'])==470 and all(len(r)==470 for r in x['rows']),'complete independent470 matrix');return arb_mat([[P.R.original_ball(dict(mid=v[0],rad=v[1]))for v in row]for row in x['rows']])

def endpoint_events(fp):
 events=[(f.name.split('-endpoint-')[1].removesuffix('.json.gz'),read(f))for f in sorted((fp/'run/events').glob('*-endpoint-*.json.gz'))];need(len(events)==162,'all54 endpoint native pairs and phase outputs');rows=[]
 for i in range(54):
  trio=events[3*i:3*i+3];need([k for k,_ in trio]==['native_input','native_return','phase_return'],'exact endpoint input/return/phase order');a,b,c=[v for _,v in trio];need(a['index']==b['index']==i and c['phase']==i%18,'literal endpoint acquisition index/phase')
  need(len(a['inputs'])==15 and len(b['returns'])==13,'full native endpoint dimensions');rows.append(dict(phase=c['phase'],inputs=a['inputs'],returns=b['returns']))
 return rows

def sharpen(fp,ap,mp,known,w):
 old=read(mp/'run/attempt022/target-certificate.json.gz');paths=list((mp/'run/attempt022/events').glob('*-signed_correlated_domain.json.gz'));need(len(paths)==1,'one original domain');domain=read(paths[0]);need(old['center']==domain['center']and old['slope']==domain['slope']and old['curvature']==domain['curvature']and old['parameter_center']==domain['parameter']and old['parameter_endpoints'][0]==known['exact_parameter']and old['normal']==known['normal']and old['origin']==known['origin'],'entire same original fixed-parameter predictor/domain')
 d=P.R.exact(F(known['exact_parameter'])-F(old['parameter_center']));raw=old['inverse_polynomial_exact'];need(old['inverse_polynomial_encoding']=='exact-dyadic-pairs'and len(raw)==2 and all(len(a)==470 and all(len(r)==470 for r in a)for a in raw),'literal finite inverse proposal');C=[arb_mat([[arb(tuple(v))for v in row]for row in a])for a in raw];Cpoint=C[0]+C[1]*d;need(all(v.is_exact()for row in Cpoint.tolist()for v in row),'exact same point inverse');del C,raw;gc.collect()
 J=None;power=arb(1)
 for k in range(3):
  a=load_matrix(ap/f'run/chart022/target/acquired_J{k}.json.gz');J=a if k==0 else J+a*power;power*=d;w.tick()
 del a;gc.collect();weights=[P.R.exact(F(v))for v in old['weights']];normal=[P.R.exact(F(v))for v in known['normal']];origin=[P.R.exact(F(v))for v in known['origin']];rows=endpoint_events(fp);box=known['box'];proofs=[]
 for i in range(3):
  actual=read(fp/f'run/endpoint-refinement-{i}-input.json');need(actual['exact_parameter']==known['exact_parameter'],'literal original point parameter');point=[P.R.exact(P.M.point(v))for v in actual['point']];P.fixed_domain(old,domain,box,actual['point'],known['exact_parameter']);g=P.point_residual(point,normal,origin,P.R.exact(F(known['exact_parameter'])),rows[18*i:18*(i+1)],13,18,-1,w.tick);record=P.localize(Cpoint,J,g,point,box,old,domain,known['exact_parameter'],weights);w.save(f'endpoint-{i}-independent.json',dict(point=actual['point'],residual=matrix_json(g),**record));box=record['box'];proofs.append(dict(coordinates=record['dimension'],weighted_bound=record['weighted_bound']));w.tick()
 refined=dict(known,box=box);w.save('independent-refined-known470.json',refined);del Cpoint,J;gc.collect();return refined,proofs

def profile(w,card):
 S.synthetic(w);pr=read(OUT/'profile.json');t=pr['seconds'];old=card['producer']['outcome'];failed=card['failed']['outcome'];forecast=dict(seconds=time.monotonic()-w.start+max(4*t+80,2*old['seconds']+failed['seconds'],2*card['prior_failure']['outcome']['seconds']+60),bytes=w.used+4*pr['bytes']+64*1024**2,rss_kib=max(pr['forecast']['rss_kib'],2*old['peak_rss_kib']+256*1024));ok=forecast['seconds']<300 and forecast['bytes']<636*1024**2 and forecast['rss_kib']<6*1024**2;w.save('complete-admission.json',dict(passed=ok,forecast=forecast,retained_success_seconds=old['seconds'],retained_failure_seconds=failed['seconds'],retained_audit_failure_seconds=card['prior_failure']['outcome']['seconds'],target_arrays_unopened=True));need(ok,'full470 proof/refinement profile and all actual cold costs')

def main():
 need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();answer=dict(passed=False,native_calls=0)
 def stop(*_):w.stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,300)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),limits=LIMITS,sources={n:entry(ROOT/n)for n in FILES},python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'full committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);w.used+=len(data)
  card=read(ROOT/CARD);need(card['status']=='reviewed'and card['limits']==LIMITS,'reviewed card before profile')
  with ctx.workprec(512):
   profile(w,card);p,fp,ap,mp,known=bind(card,w);w.stage='target';refined,rows=sharpen(fp,ap,mp,known,w);endpoint,result=Q.replay_curve(p/'run',refined,w);answer.update(result,refinement_rows=rows,exact_parameter=endpoint['exact_parameter'])
 except BaseException as e:answer.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);elapsed=time.monotonic()-w.start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;answer['passed']=bool(answer['passed']and elapsed<300 and rss<6*1024**2);answer.update(seconds=elapsed,peak_rss_kib=rss);w.save('outcome.json',answer,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(answer));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
