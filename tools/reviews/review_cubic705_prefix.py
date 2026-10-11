#!/usr/bin/env python3
"""Independent480 proof of a fixed three-chart native cubic705 connected prefix."""
import gc,gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import review_cubic705_math as H
import review_correlated705 as OLD
from neural_shrimp.rigorous_orbits import ball_json,vector_json,canonical_bytes
need,P,R,M,U=H.need,H.P,H.R,H.M,H.U
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic-prefix-independent/attempt001/run');CARD='docs/plans/native705-cubic-independent-inputs.json'
THREADS=OLD.THREADS;entry=OLD.entry;read=OLD.read;inventory=OLD.inventory;historical=OLD.historical
FILES=[CARD,'docs/plans/native705-cubic-independent.md','tools/reviews/review_cubic705_prefix.py','tools/launch_cubic705_prefix_review.py','tests/test_cubic705_review.py','tests/test_cubic705_prefix_driver.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
EXPECTED={'direction_return':108,'cubic_proposal_input':3,'cubic_proposal_return':3,'cubic_affine_proposal_input':1,'cubic_quadratic_proposal_input':1,'pal_path_phase_return':72,'pal_path_assembly_return':2,'cubic_predictor_return':1,'cubic_correlated_domain':1,'gradient_path_input':72,'gradient_path_return':72,'gradient_path_assembly_return':2,'cubic_path_input':72,'cubic_path_return':72,'cubic_path_assembly_return':2,'cubic_correlated_chart_return':1}

def exact_inverse(rows):
 need(type(rows)is list and len(rows)==705 and all(type(v)is list and len(v)==705 for v in rows),'complete finite C0 proposal')
 for row in rows:
  for v in row:need(type(v)is list and len(v)==2 and all(type(q)is int for q in v)and -10000<=v[1]<=10000,'finite dyadic inverse encoding')
 out=arb_mat([[arb((v[0],v[1]))for v in row]for row in rows]);need(H.finite(out)and all(v.is_exact()for row in out.tolist()for v in row),'exact finite C0 without producer inverse authority');return out

def acquisitions(run,kind,group):
 need(kind in('gradient_path','cubic_path')and group in(0,1),'fixed actual path groups')
 ip=sorted((run/'events').glob('*-'+kind+'_input.json.gz'));op=sorted((run/'events').glob('*-'+kind+'_return.json.gz'));need(len(ip)==len(op)==72,'complete actual acquired input/return pairs');rows=[]
 for j in range(36*group,36*(group+1)):
  need(int(op[j].name.split('-')[0])==int(ip[j].name.split('-')[0])+1,'adjacent actual native input/output');a=read(ip[j]);b=read(op[j]);need(a['phase']==b['phase']==(j%36)//2 and a['direction']==b['direction']==('q'if j%2==0 else'r'),'fixed phase/direction order');rows.append(dict(**a,returns=b['returns']))
 return rows

def geometry(domain,proposal,cfg,card):
 need(cfg['period']==18 and cfg['precision']==256 and cfg['newton_steps']==2 and cfg['halfwidth_exponent']==-5 and cfg['radius_exponent']==-10 and cfg['predictor_degree']==3 and cfg['residual_degree']==4,'same fixed source settings')
 c,s,a,b=[[R.exact(M.point(v))for v in proposal[k]]for k in('center','slope','curvature','cubic')];normal=[R.exact(F(v))for v in card['normal']];origin=[R.exact(F(v))for v in card['origin']];weights=[R.exact(F(v))for v in card['weights']]
 need(all(len(v)==705 for v in(c,s,a,b,normal,origin,weights))and all(0<v<=1 for v in weights),'entire native predictor/gauge/weights')
 for k in('center','slope','curvature','cubic'):need(domain[k]==proposal[k],'literal finite predictor '+k)
 need(domain['parameter']==cfg['parameter']and domain['halfwidth_exponent']==-5 and domain['radius_exponent']==-10,'actual source domain scales')
 h=arb(1)/32;rho=arb(1)/1024;d=arb(0,h);radii=[rho*v for v in weights];error=[R.original_ball(v)for v in domain['error']];need(len(error)==705,'all actual correction variables')
 for old,r in zip(domain['error'],radii,strict=True):
  lo,hi=M.interval(old);rr=M.point(ball_json(r));need(lo==-hi and lo<=-rr and hi>=rr,'original exact acquired domain covers requested correction')
 whole=[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d+error[i]for i in range(705)]
 need(len(domain['whole_box'])==705,'full original whole box')
 for old,new in zip(domain['whole_box'],whole,strict=True):U.contains(old,ball_json(new),'every actual whole cubic acquisition domain')
 need(all(v>0 for v in whole[-3:])and abs(whole[-1]).upper()<=arb(1)/4,'whole physical rates and positive angle')
 return c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole

def bind(card,w):
 spec=card['producer'];root=Path(spec['root']);rows=[inventory(root,spec['outer'],w.tick)];out=read(root/'outcome.json');need(out['source']==spec['source']and not out['passed']and out['connected_count']==3 and out['connected_right']=='115/512'and not out['owned_live_pids']and len(out['workers'])==4,'exact qualified prefix and failed suffix retained')
 need([r['exit_code']for r in out['workers']]==[0,0,0,2]and all(not r['owned_live_pids']for r in out['workers']),'all four actual workers terminated')
 owned=read(root/'owned-processes.json');need(len(owned['workers'])==4,'complete ownership')
 for v in owned['workers']:need(not Path('/proc',str(v['group'])).exists(),'prior owned worker absent')
 pc=read(root/'input-card.json');need(pc['source']==spec['source']and pc['worker_count']==4 and pc['batch']==1 and pc['kernel']=='fresh-cubic705-v1','literal original card')
 ar,art,ao,rr=OLD.bundle(card['anchor'],w.tick);rows+=rr;endpoint=read(ar/'run/connected-endpoint.json');need(entry(ar/'run/connected-endpoint.json')['sha256']==card['anchor']['endpoint']and ao['passed']and ao['connected']and ao['strict_inclusions']==ao['anchor_coordinates']==705 and ao['primitive_period']==18,'complete independently known endpoint')
 need(endpoint['chart']['parameter_endpoints']==['-49/2048','79/2048']and pc['anchor']==dict(box=endpoint['endpoint'],normal=endpoint['chart']['normal'],origin=endpoint['chart']['origin'],parameter='79/2048'),'entire original exact anchor equality')
 need(pc['normal']==endpoint['chart']['normal']and pc['origin']==endpoint['chart']['origin'],'literal affine gauge identity')
 prior=read(ar/'run/input-binding.json')['card']['producer'];need(prior==card['model_authority'],'same audited native source');pr,prt,po,rr=OLD.bundle(prior,w.tick);rows+=rr
 need(pc['map_inputs']==read(pr/'run/inputs.json')['map_inputs'],'identical original objective/model/data')
 indexed=[]
 for unit,worker in enumerate(spec['workers'],1):
  run=root/f'chart{unit:03d}/run';rows.append(inventory(run,worker,w.tick));rt=historical(run,spec['source']);o=read(run/'outcome.json');need(rt['unit']==o['unit']==unit and rt['settings']==pc['settings'][unit-1]and rt['threads']==dict.fromkeys(THREADS,'1'),'same exact indexed worker/runtime')
  need(o['passed']and o['status']=='uniform705_unjoined'and not o['connected']and o['ticks']==331 and o['strict_inclusions']==705 and o['events']==EXPECTED,'all original full unjoined proof acquisitions')
  for n in OLD.NATIVE:need(rt['sources'][n]==prt['sources'][n],'unchanged source-authenticated model '+n)
  inp=read(run/'inputs.json');need(inp['map_inputs']==pc['map_inputs']and inp['card']==entry(root/'input-card.json')and inp['settings']==rt['settings'],'complete actual worker inputs')
  ev=sorted((run/'events').glob('*.json.gz'));need(len(ev)==sum(EXPECTED.values())and[int(p.name.split('-')[0])for p in ev]==list(range(len(ev))),'entire contiguous event stream')
  for kind,num in EXPECTED.items():need(len(list((run/'events').glob('*-'+kind+'.json.gz')))==num,'all actual event kinds '+kind)
  indexed.append((run,rt['settings']))
 w.save('input-binding.json',dict(card=card,inventories=rows,producer_input_card=entry(root/'input-card.json'),source_native={n:prt['sources'][n]for n in OLD.NATIVE},derived_producer_certificates_unparsed=True));return indexed,pc,endpoint
class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.stage='profile'
 def tick(self):need(not self.stopped and time.monotonic()-self.start<900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'900s/6GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(2048 if closing else 2044)*1024**2,'2GiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
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
      f.write(((','if i else'')+json.dumps(row,separators=(',',':'))).encode());need(start+raw.tell()<2044*1024**2,'derived matrix cap preserves final reserve');self.tick()
     f.write(b']}')
    raw.flush();os.fsync(raw.fileno())
  finally:self.used=start+(p.stat().st_size if p.exists()else 0)
  need(self.used<2044*1024**2,'full derived matrix byte cap')

 def emit(self,tag,value):
  if isinstance(value,arb_mat):self.matrix(self.stage+'/'+tag+'.json.gz',value)
  else:self.save(self.stage+'/'+tag+'.json',value)
  self.tick()

def synthetic(w):
 n=705;den=arb((1,270));A=arb_mat([[arb(((i+1)*6364136223846793005+(j+1)*1442695040888963407)**4%2**250)/den for j in range(n)]for i in range(n)]);I=M.ident(n);C=I+A;J=[I-A,A/8,A/32];res=[arb_mat([[arb((1,-80))*(i+1)/n]for i in range(n)])]+[arb_mat(n,1)for _ in range(3)]+[arb_mat([[arb((1,-60))*(i+1)/n]for i in range(n)])];h=arb(1)/32;rho=arb(1)/1024;before=time.monotonic();used=w.used
 proof=H.polynomial_root(C,J,res,[arb(1)]*n,rho,h,[arb(0,rho)]*n,w.tick,w.emit);need(proof['passed'],'fresh full705 degree5 polynomial control');elapsed=time.monotonic()-before;stored=w.used-used;forecast=elapsed+3*(3*elapsed+75);raw=10*stored+64*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3
 report=dict(passed=bool(forecast<900 and raw<2048*1024**2 and peak<6*1024**3),seconds=elapsed,stored_bytes=stored,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,dimension=705,target_arrays_unopened=True);w.save('profile.json',report);need(report['passed'],'complete measured900s/6GiB/2GiB source/serialization admission');del A,C,J,res;gc.collect()

def check_one(run,cfg,pc,previous,w,unit):
 dom=sorted((run/'events').glob('*-cubic_correlated_domain.json.gz'));pro=sorted((run/'events').glob('*-cubic_predictor_return.json.gz'));need(len(dom)==len(pro)==1,'unique literal domain/finite proposal')
 domain=read(dom[0]);proposal=read(pro[0]);c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=geometry(domain,proposal,cfg,pc);C0=exact_inverse(proposal['preconditioner_exact']);del proposal;gc.collect();w.stage=f'chart{unit:03d}';sc=R.exact(F(cfg['parameter']))
 need(previous['chart']['normal']==pc['normal']and previous['chart']['origin']==pc['origin'],'same independently proved exact affine gauge before scalar join')
 ac=H.cubic_membership(previous['endpoint'],c,s,a,b,normal,origin,sc,h,radii,known_parameter=previous['chart']['parameter_endpoints'][1]);w.save(w.stage+'/known-root-membership.json',ac)
 J0=H.jacobian_coefficients(c,s,a,b,error,arb(0),acquisitions(run,'gradient_path',0),13,18,normal,w.tick);Jw=H.jacobian_coefficients(c,s,a,b,error,d,acquisitions(run,'gradient_path',1),13,18,normal,w.tick);J=[J0[0],J0[1],Jw[2]];del J0,Jw;gc.collect()
 for k,j in enumerate(J):w.matrix(w.stage+f'/jacobian_coefficient_{k}.json.gz',j)
 coeff=H.residual_coefficients(c,s,a,b,arb(0),acquisitions(run,'cubic_path',0),13,18,normal,origin,sc);wc=H.residual_coefficients(c,s,a,b,d,acquisitions(run,'cubic_path',1),13,18,normal,origin,sc);res=[arb_mat([[coeff[i,k]if k<4 else wc[i,4]]for i in range(705)])for k in range(5)]
 for k,v in enumerate(res):w.matrix(w.stage+f'/residual_coefficient_{k}.json.gz',v)
 proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[39*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];frame=[i for i in range(13)if not whole[26+i].contains(0)];gates=dict(uniform_root=proof['passed'],known_root=ac['passed']and ac['coordinates']==705,primitive18=all(v['coordinates']for v in proper),independent_frame=bool(frame),positive_rates_angle=True);endpoints=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  dd=sign*h;endpoints.append(vector_json([c[i]+s[i]*dd+a[i]*dd*dd/2+b[i]*dd*dd*dd+R.original_ball(corr[i])for i in range(705)]))
 chart=dict(shooting_dimension=705,normal=pc['normal'],origin=pc['origin'],parameter_center=cfg['parameter'],parameter_endpoints=[str(F(cfg['parameter'])-F(1,32)),str(F(cfg['parameter'])+F(1,32))],settings=dict(halfwidth_exponent=-5,radius_exponent=-10));passed=all(gates.values());w.save(w.stage+'/independent-proof.json',dict(passed=passed,gates=gates,root=proof,anchor=ac,proper_divisors=proper,frame_nonzero_coordinates=frame,chart=chart,endpoints=endpoints))
 return dict(chart=chart,endpoint=endpoints[1]),dict(unit=unit,passed=passed,strict_inclusions=sum(proof['coordinate_inclusions']),anchor_coordinates=ac['coordinates'],contraction=proof['contraction'],parameter_endpoints=chart['parameter_endpoints'],physical_right=endpoints[1][-3:])

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,900);result=dict(passed=False,connected=False,native_calls=0,status='incomplete',charts=[])
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0)))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'own exact source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   card=read(ROOT/CARD);need(card['audit']==dict(output=str(OUT),precision=480,seconds=900,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=2048*1024**2,charts=3),'literal full audit resource card');synthetic(w);runs,pc,previous=bind(card,w)
   for unit,(run,cfg)in enumerate(runs,1):
    previous,rec=check_one(run,cfg,pc,previous,w,unit);result['charts'].append(rec);need(rec['passed'],'full independent root/known-root/divisor/frame gates')
    w.save(f'connected-endpoint-{unit:03d}.json',previous);w.tick();gc.collect()
   need(len(result['charts'])==3 and previous['chart']['parameter_endpoints'][1]=='115/512','complete ordered three-chart prefix');w.save('connected-endpoint.json',previous);result.update(passed=True,connected=True,status='independently_connected_three_cubic705_charts',connected_count=3,connected_right='115/512',physical_right=previous['endpoint'][-3:],scope='Independent480 full native acquired-domain, Jacobian/quartic residual/degree5forcing reductions and full705 known-root joins. Shared native derivatives/Arb. No complement, distant junction or complete window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
