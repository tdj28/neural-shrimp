#!/usr/bin/env python3
"""One fixed saved-only independent wider705 polynomial tube audit."""
import gc,gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import review_correlated705_math as H
from neural_shrimp.rigorous_orbits import ball_json,vector_json,canonical_bytes
need,P,R,M,U=H.need,H.P,H.R,H.M,H.U
OUT=Path('/data2/neural-shrimp-data/audits/native705-correlated-independent/attempt002/run');CARD='docs/plans/native705-correlated-independent-inputs.json'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
NATIVE=['src/neural_shrimp/'+n+'.py'for n in('__init__','models','controls','rigorous_orbits','rigorous_folds','rigorous_cusps','rigorous_eigenpair_locus','rigorous_eigenpair_path','rigorous_unit_pair','rigorous_unit_pair_pal')]
FILES=[CARD,'docs/plans/native705-correlated-independent.md','tools/reviews/review_correlated705.py','tools/launch_correlated705_review.py','tests/test_correlated705_review.py','tests/test_correlated705_driver.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
  need(len(b)<=256*1024**2,'bounded256MiB decoded input; no huge derived producer certificate');return json.loads(b)
 return json.loads(p.read_bytes())
def inventory(root,pin,tick):
 need(entry(root/'manifest.json')['sha256']==pin,'literal immutable manifest');files=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(files)==set(actual),'entire immutable inventory')
 for n,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==files[n],'immutable exact file '+n);tick()
 return dict(root=str(root),manifest=pin,files=len(files),bytes=sum(v['bytes']for v in files.values()))
def historical(run,source):
 rt=read(run/'runtime.json');need(rt.get('commit',rt.get('source_commit'))==source,'exact execution source');actual={str(p.relative_to(run/'source')):p for p in(run/'source').rglob('*')if p.is_file()};need(set(actual)==set(rt['sources']),'full source archive')
 for n,p in actual.items():need(entry(p)==rt['sources'][n]and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==rt['sources'][n]['sha256'],'historical source '+n)
 need(not Path('/proc',str(rt['pid'])).exists(),'prior worker absent');return rt

def bundle(spec,tick):
 root=Path(spec['root']);rows=[inventory(root,spec['outer'],tick),inventory(root/'run',spec['worker'],tick)];rt=historical(root/'run',spec['source']);la=read(root/'launcher-result.json');need(la.get('commit',la.get('source_commit'))==spec['source']and la['exit_code']==0 and not la['owned_live_pids'],'completed immutable execution')
 for k in('launcher_pid','owned_group_pid'):
  if k in la:need(not Path('/proc',str(la[k])).exists(),'prior owned PID absent')
 return root,rt,read(root/'run/outcome.json'),rows

def retained_events(run):
 direction=sorted((run/'events').glob('*-direction_return.json.gz'));paths=sorted((run/'events').glob('*-pal_path_phase_return.json.gz'));need(len(direction)==144 and len(paths)==108,'complete original705 derivative/path acquisitions');return [read(p)for p in direction],[read(p)for p in paths]

def strip_chart(p):
 chart=read(p);keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','curvature','weights','coordinate_radii','preconditioner','whole_box','normal','origin'}
 for k in list(chart):
  if k not in keep:del chart[k]
 gc.collect();return chart

def preserved_failure(card,w):
 spec=card['prior_failure'];root=Path(spec['root']);rows=[inventory(root,spec['outer'],w.tick),inventory(root/'run',spec['worker'],w.tick)];rt=historical(root/'run',spec['source']);la=read(root/'launcher-result.json');profile=read(root/'run/profile.json');out=read(root/'run/outcome.json')
 need(la['source_commit']==spec['source']and la['exit_code']==2 and not la['owned_live_pids']and not out['passed']and out['native_calls']==0,'preserved failed admission')
 need(profile['target_arrays_unopened']and not profile['passed']and profile['forecast_bytes']==560985542 and profile['forecast_seconds']<240 and profile['forecast_peak_bytes']<6*1024**3 and read(root/'run/profile/correlated_root.json')['passed']and not(root/'run/input-binding.json').exists(),'actual prior storage-only failure')
 for key in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(la[key])).exists(),'failed attempt owned PID absent')
 return rows

def bind(card,w):
 prior=preserved_failure(card,w)
 pr,rt,out,rows=bundle(card['producer'],w.tick);rows+=prior;run=pr/'run';need(rt['settings']==card['settings']and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'same actual source/settings')
 need(out['passed']and out['connected']and out['ticks']==148 and out['strict_inclusions']==out['anchor_inclusions']==705 and out['events']=={'correlated_chart_return':1,'correlated_domain':1,'gradient_path_assembly_return':2,'gradient_path_input':72,'gradient_path_return':72,'pal_path_assembly_return':2,'pal_path_phase_return':72},'complete wider producer record')
 binding=read(run/'input-binding.json');need(binding['independent']==card['independent']and binding['packet']==card['packet']and binding['parent']['root']==card['parent']['root']and binding['parent']['worker']['sha256']==card['parent']['worker']and binding['proposal']['root']==card['proposal']['root']and binding['proposal']['worker']['sha256']==card['proposal']['worker']and binding['prior_failure_preserved']and binding['settings']==card['settings'],'all exact original parent bindings')
 packet=Path(card['packet']['root']);rows.append(inventory(packet,card['packet']['outer'],w.tick));oldout=read(packet/'outcome.json');need(oldout['source']==card['packet']['source']and oldout['connected_charts']==24 and len(oldout['batches'])==4 and not oldout['passed']and not oldout['owned_live_pids'],'retained24 connected and8 failed packet')
 ir,irt,io,rr=bundle(card['independent'],w.tick);rows+=rr;ib=read(ir/'run/input-binding.json');ep=read(ir/'run/connected-endpoint.json');need(entry(ir/'run/connected-endpoint.json')['sha256']==card['independent']['endpoint']and len(ep['endpoint'])==705,'literal independent full705 endpoint')
 need(io['passed']and io['connected']and io['connected_count']==8 and io['connected_parameter_endpoints']==['-1/2048','255/2048'],'independently connected final eight')
 parent=Path(card['parent']['root']);proposal=Path(card['proposal']['root']);prs=[]
 for spec in(card['parent'],card['proposal']):
  rr=Path(spec['root'])/'run';rows.append(inventory(rr,spec['worker'],w.tick));prs.append(historical(rr,card['packet']['source']))
 need(ib['card']['root']==str(parent.parent)and ib['card']['source']==card['packet']['source']and ib['card']['qualified_prefix']==8 and ib['card']['charts'][-1]==dict(path='chart008/run',unit=8,worker=card['parent']['worker']),'same independent known predecessor')
 for n in NATIVE:need(rt['sources'][n]==prs[0]['sources'][n]==prs[1]['sources'][n],'identical fixed native model '+n)
 inp=read(run/'inputs.json');need(inp['map_inputs']==read(parent/'run/inputs.json')['map_inputs']==read(proposal/'run/inputs.json')['map_inputs'],'same full map/data/both rates')
 po=read(parent/'run/outcome.json');fo=read(proposal/'run/outcome.json');need(po['passed']and po['ticks']==253 and po['strict_inclusions']==705 and not fo['passed']and fo['status']=='completed_bounds_failed'and fo['ticks']==253 and fo['strict_inclusions']==704,'actual passed root and failed finite proposal')
 ev=sorted((run/'events').glob('*.json.gz'));need(len(ev)==222 and [int(p.name.split('-')[0])for p in ev]==list(range(222)),'all222 actual source events')
 # Do not parse either copy of the huge derived correlated certificate.
 need(entry(run/'certificate.json.gz')==entry(run/'events/0221-correlated_chart_return.json.gz')==dict(bytes=184464912,sha256='c28c7a95867b1aa1df3dcfc76867e2aff0d36e1fec08225d4294f1a9799dd06a'),'complete exact derived certificate retained')
 domain=read(run/'events/0000-correlated_domain.json.gz');chart=strip_chart(parent/'run/certificate.json.gz')
 for k,v in ep['chart'].items():need(chart[k]==v,'literal independent preceding geometry '+k)
 direction,paths=retained_events(parent/'run');w.save('input-binding.json',dict(card=card,inventories=rows,source_native={n:rt['sources'][n]for n in NATIVE},producer_domain=entry(run/'events/0000-correlated_domain.json.gz'),source_events={p.name:entry(p)for p in ev},huge_derived_proof_unparsed=True))
 return run,proposal,domain,chart,direction,paths,ep['endpoint']

def gradients(run,group):
 need(group in(0,1),'two actual gradient groups');base=1+73*group;rows=[]
 for i in range(36):
  a=read(run/f'events/{base+2*i:04d}-gradient_path_input.json.gz');b=read(run/f'events/{base+2*i+1:04d}-gradient_path_return.json.gz');need(a['phase']==b['phase']==i//2 and a['direction']==b['direction']==('q'if i%2==0 else'r'),'actual paired native gradient order');rows.append(dict(**a,returns=b['returns']))
 return rows

def paths(run,group):
 need(group in(0,1),'two actual cubic groups');base=147+37*group;return [read(run/f'events/{base+i:04d}-pal_path_phase_return.json.gz')for i in range(36)]

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.stage='profile'
 def tick(self):need(not self.stopped and time.monotonic()-self.start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'240s/6GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(640 if closing else 636)*1024**2,'640MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def matrix(self,name,a):
  # Stream exact dyadic pairs row-by-row: no full JSON string/matrix copy.
  need(H.finite(a),'finite derived matrix');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);start=self.used
  try:
   with p.open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as f:
     f.write(b'{"encoding":"mid-rad-dyadic-pairs","rows":[')
     for i in range(a.nrows()):
      row=[[[int(t)for t in a[i,j].mid().man_exp()],[int(t)for t in a[i,j].rad().man_exp()]]for j in range(a.ncols())]
      f.write(((','if i else'')+json.dumps(row,separators=(',',':'))).encode());need(start+raw.tell()<636*1024**2,'derived matrix cap preserves final reserve');self.tick()
     f.write(b']}')
    raw.flush();os.fsync(raw.fileno())
  finally:self.used=start+(p.stat().st_size if p.exists()else 0)
  need(self.used<636*1024**2,'full derived matrix byte cap')

 def emit(self,tag,value):
  if isinstance(value,arb_mat):self.matrix(self.stage+'/'+tag+'.json.gz',value)
  else:self.save(self.stage+'/'+tag+'.json',value)
  self.tick()

def synthetic(w):
 n=705;den=arb((1,270));A=arb_mat([[arb(((i+1)*6364136223846793005+(j+1)*1442695040888963407)**4%2**250)/den for j in range(n)]for i in range(n)]);I=M.ident(n);C=I+A;J=[I-A,A/8,A/32];res=[arb_mat([[arb((1,-80))*(i+1)/n]for i in range(n)])]+[arb_mat(n,1)for _ in range(3)];h=arb(1)/32;rho=arb(1)/1024;before=time.monotonic();old=w.used
 proof=H.polynomial_root(C,J,res,[arb(1)]*n,rho,h,[arb(0,rho)]*n,w.tick,w.emit);need(proof['passed'],'full705 synthetic correlated control');elapsed=time.monotonic()-before;stored=w.used-old;forecast=3*elapsed+75;raw=3*stored+32*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3
 result=dict(passed=bool(forecast<240 and raw<640*1024**2 and peak<6*1024**3),dimension=n,seconds=elapsed,stored_bytes=stored,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True);w.save('profile.json',result);need(result['passed'],'measured full705 source/serialization admission');del A,C,J,res;gc.collect()

def actual_geometry(proposal,domain):
 c,s,a=[[R.exact(M.point(v))for v in proposal[k]]for k in('center','slope','curvature')];weights=[R.exact(F(v))for v in proposal['weights']];normal=[R.exact(F(v))for v in proposal['normal']];origin=[R.exact(F(v))for v in proposal['origin']];need(all(len(v)==705 for v in(c,s,a,weights,normal,origin))and all(v>0 for v in weights),'complete exact705 proposals')
 need(proposal['parameter_center']==domain['parameter']=='15/2048'and proposal['settings']['halfwidth_exponent']==-7 and proposal['settings']['radius_exponent']==domain['radius_exponent']==-10 and domain['halfwidth_exponent']==-5,'fixed wider parameter geometry')
 for k in('center','slope','curvature'):need(proposal[k]==domain[k],'same fixed source predictor '+k)
 h=arb(1)/32;rho=arb(1)/1024;d=arb(0,h);radii=[rho*v for v in weights];error=[R.original_ball(v)for v in domain['error']];need(len(error)==705,'entire actual correction box')
 for old,r in zip(domain['error'],radii,strict=True):lo,hi=M.interval(old);rr=M.point(ball_json(r));need(lo<=-rr and rr<=hi and lo==-hi,'actual constructed correction covers exact requested radius')
 whole=[c[i]+s[i]*d+a[i]*d*d/2+error[i]for i in range(705)]
 for old,new in zip(domain['whole_box'],whole,strict=True):U.contains(old,ball_json(new),'exact acquired whole curved domain')
 need(all(v>0 for v in whole[-3:])and abs(whole[-1]).upper()<=arb(1)/4,'whole positive actual rates/angle');return c,s,a,weights,normal,origin,h,rho,d,radii,error,whole

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed audit');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,240);result=dict(passed=False,native_calls=0,status='incomplete')
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'exact source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   synthetic(w);card=read(ROOT/CARD);need(card['audit']==dict(output=str(OUT),precision=480,seconds=240,rss_bytes=6*1024**3,as_bytes=10*1024**3,output_bytes=640*1024**2,closing_reserve_bytes=4*1024**2,synthetic_dimension=705,forecast_multiplier=3,forecast_allowance_seconds=75),'literal audit bounds')
   run,proposal_root,domain,old,dg,pg,known=bind(card,w);localized=H.L.localize_right(old,dg,pg,known,precision=480,tick=w.tick);w.save('known-endpoint-480.json',localized);need(localized['passed']and len(localized['box'])==705,'independent existing root localization');del old,dg,pg,known;gc.collect();w.tick()
   proposal=strip_chart(proposal_root/'run/certificate.json.gz');c,s,a,weights,normal,origin,h,rho,d,radii,error,whole=actual_geometry(proposal,domain);C0=R.matrix(proposal['preconditioner']);del proposal;gc.collect();w.stage='target';sc=R.exact(F(15,2048))
   ac=P.anchor_membership(localized['box'],c,s,a,normal,origin,sc,h,radii);w.save('known-root-membership.json',ac)
   J0=H.jacobian_coefficients(c,s,a,error,arb(0),gradients(run,0),13,18,normal,w.tick);Jw=H.jacobian_coefficients(c,s,a,error,d,gradients(run,1),13,18,normal,w.tick);J=[J0[0],J0[1],Jw[2]];del J0,Jw;gc.collect()
   for k,j in enumerate(J):w.matrix(f'target/jacobian_coefficient_{k}.json.gz',j)
   coeff=P.path_coefficients(c,s,a,arb(0),paths(run,0),13,18,normal,origin,sc);wc=P.path_coefficients(c,s,a,d,paths(run,1),13,18,normal,origin,sc);res=[arb_mat([[coeff[i,k]if k<3 else wc[i,3]]for i in range(705)])for k in range(4)]
   for k,r in enumerate(res):w.matrix(f'target/residual_coefficient_{k}.json.gz',r)
   proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[39*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];independent=[i for i in range(13)if not whole[26+i].contains(0)];gates=dict(uniform_root=proof['passed'],known_root=ac['passed']and ac['coordinates']==705,primitive18=all(v['coordinates']for v in proper),independent_frame=bool(independent),positive_rates_angle=True)
   endpoints=[]
   for sign,correction in zip((-1,1),proof['endpoint_corrections'],strict=True):
    dd=sign*h;endpoints.append(vector_json([c[i]+s[i]*dd+a[i]*dd*dd/2+R.original_ball(correction[i])for i in range(705)]))
   chart=dict(shooting_dimension=705,normal=[str(M.point(ball_json(v)))for v in normal],origin=[str(M.point(ball_json(v)))for v in origin],parameter_center='15/2048',parameter_endpoints=['-49/2048','79/2048'],settings=dict(halfwidth_exponent=-5,radius_exponent=-10))
   w.save('independent-proof.json',dict(passed=all(gates.values()),gates=gates,root=proof,primitive=proper,independent_frame_coordinates=independent,anchor=ac,endpoints=endpoints,chart=chart));w.save('connected-endpoint.json'if all(gates.values())else'unqualified-endpoint-candidate.json',dict(chart=chart,endpoint=endpoints[1]));w.tick();result.update(passed=all(gates.values()),connected=all(gates.values()),status='independent_connected_correlated705'if all(gates.values())else'completed_independent_bound_failure',strict_inclusions=sum(proof['coordinate_inclusions']),anchor_coordinates=ac['coordinates'],contraction=proof['contraction'],primitive_period=18,parameter_endpoints=chart['parameter_endpoints'],scope='Independent480 full705 native-return Jacobian/residual polynomial reductions and full projected known-root join. Shared acquired derivative arrays/Arb; no complementary spectrum, signed criticality, junction or complete window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
