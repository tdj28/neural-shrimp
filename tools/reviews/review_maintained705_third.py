"""Fixed saved24 maintained003 roots from an independently proved705 endpoint."""
import gc,gzip,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import review_native705_minimum as A
B,H,R,M,U=A.B,A.H,A.R,A.M,A.U
need,read,entry,inventory,historical=A.need,A.read,A.entry,A.inventory,A.historical
from neural_shrimp.rigorous_orbits import ball_json,vector_json,canonical_bytes
OUT=Path('/data2/neural-shrimp-data/audits/native705-maintained003-independent/attempt001/run')
CARD='docs/plans/native705-maintained003-independent-inputs.json'
SELECTED=list(range(1,25))
FILES=[CARD,'docs/plans/native705-maintained003-independent.md','docs/reviews/native705-maintained-tail-independent.json','tools/reviews/review_maintained705_third.py','tools/launch_maintained705_third_review.py','tests/test_maintained705_third_review.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def settings():return dict(precision=480,seconds=4500,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=14*1024**3,closing_reserve=4*1024**2,charts=24,native_calls=0)

def selection(card):
 need(card['status']=='reviewed'and card['schema']=='independent-maintained003-all24-v1'and card['settings']==settings()and card['selected_attempts']==SELECTED,'fixed selected24 bounded design')
 rows=card['producer']['workers'];need([x['attempt']for x in rows]==list(range(1,25))and[x['attempt']for x in rows if x['joined']]==SELECTED,'complete24 all accepted histories')
 need(all(x['root_passed']==x['joined']for x in rows)and[x['attempt']for x in rows if x['mode']=='rotated_gauge']==[24],'one exact rotation and no inferred joins')
 return rows

def same_gauge_parameter(previous,pc):
 need(previous['chart']['normal']==pc['normal']and previous['chart']['origin']==pc['origin'],'literal inherited affine gauge')
 value=F(previous['chart']['parameter_endpoints'][1]);center=F(pc['settings']['parameter']);h=F(2)**pc['settings']['halfwidth_exponent']
 need(pc['anchor']['exact_parameter']==str(value)and center-h<=value<=center+h and value-(center-h)==F(1,2048),'exact known scalar and original overlap')
 return str(value)

def bind(card,w):
 rowspec=selection(card);anchor,rt,out,rows=B.OLD.bundle(card['anchor'],w.tick);ep=read(anchor/'run/connected-endpoint.json');ib=read(anchor/'run/input-binding.json');proof=read(anchor/'run/chart019/independent-proof.json')
 need(out['passed']and out['connected']and out['native_calls']==0 and len(out['charts'])==19 and rt['precision']==480,'complete independent remaining19-root authority')
 need(entry(anchor/'run/connected-endpoint.json')==card['anchor']['endpoint']and entry(anchor/'run/chart019/independent-proof.json')==card['anchor']['proof']and ep['chart']==proof['chart']and ep['endpoint']==proof['endpoints'][1],'exact independently computed successor endpoint')
 need(proof['passed']and all(proof['gates'].values())and len(proof['root']['coordinate_inclusions'])==705 and all(proof['root']['coordinate_inclusions'])and proof['anchor']['coordinates']==705 and ep['chart']['parameter_endpoints'][1]=='2995/2048','all705 prior root and join')
 spec=card['producer'];root=Path(spec['root']);rows.append(inventory(root,spec['outer'],w.tick));po=read(root/'outcome.json');start=read(root/'launcher-start.json')
 need(po['source']==start['source']==spec['source']and po['passed']and po['status']=='attempt_cap_reached'and po['attempted']==po['launched_attempts']==24 and po['connected_count']==24 and po['fallback_count']==0 and not po['owned_live_pids'],'full completed actual24-attempt packet')
 need(not Path('/proc',str(start['pid'])).exists(),'original parent exited');need(len(po['attempts'])==24,'all original history rows')
 indexed=[];prior=rowspec[0]
 for v,original in zip(rowspec,po['attempts'],strict=True):
  i=v['attempt'];base=root/f'attempt{i:03d}';run=base/'run';rows.append(inventory(base,v['outer'],w.tick));rows.append(inventory(run,v['worker'],w.tick));rt=historical(run,spec['source']);pc=read(base/'input-card.json');inp=read(run/'inputs.json');o=read(run/'outcome.json');owner=read(base/'owned-process.json');la=read(base/'launcher-result.json')
  need(original['attempt']==i and original['worker']['sha256']==v['worker']and original['outer_manifest']['sha256']==v['outer']and original['joined']==v['joined']and original['root_passed']==v['root_passed'],'literal complete original outcome order')
  need(not la['owned_live_pids']and la['exit_code']==(0 if v['root_passed']else 2)and not Path('/proc',str(owner['group'])).exists(),'every accepted and failed worker exited')
  need(entry(base/'input-card.json')==v['card']and pc['source']==spec['source']and pc['attempt']==rt['attempt']==o['attempt']==i and pc['mode']==v['mode']and pc['kernel']=='fresh-cubic705-maintained-v1','literal source/card/index/mode')
  need(pc['settings']==rt['settings']==inp['settings']==v['settings']and pc['map_inputs']==inp['map_inputs']==ib['model']and pc['parent']==inp['parent']==v['parent']and pc['anchor']==inp['anchor']and inp['card']==v['card'],'complete original map/geometry/parent binding')
  need(rt['threads']==dict.fromkeys(B.THREADS,'1')and rt['nice']>=10,'actual single-thread native runtime')
  for n in A.NATIVE:need(rt['sources'][n]==ib['native_source'][n],'identical original derivative source '+n)
  need(o['passed']==v['root_passed']and o['ticks']==331 and o['events']==B.EXPECTED and not o['connected'],'every complete331 acquisition history')
  ev=sorted((run/'events').glob('*.json.gz'));need(len(ev)==sum(B.EXPECTED.values())and[int(p.name.split('-')[0])for p in ev]==list(range(len(ev))),'all ordered retained native events')
  for k,num in B.EXPECTED.items():need(len(list((run/'events').glob('*-'+k+'.json.gz')))==num,'actual event-kind counts')
  if i==1:
   inherited=ib['card']['producer'];last=inherited['workers'][-1]
   expected=dict(kind='accepted-maintained-endpoint',root=str(Path(inherited['root'])/'attempt024'),source=inherited['source'],outer=last['outer'],worker=last['worker'],endpoint=last['endpoint']['sha256'])
   need(pc['parent']==expected and ep['chart']['normal']==pc['normal']and ep['chart']['origin']==pc['origin'],'first chart follows exactly the independently audited preceding source')
  else:
   need(pc['parent']==dict(kind='accepted-maintained-endpoint',root=str(root/f"attempt{prior['attempt']:03d}"),source=spec['source'],outer=prior['outer'],worker=prior['worker'],endpoint=prior['endpoint']['sha256']),'original ordered last-accepted genealogy')
  if v['joined']:
   need(o['status']=='uniform705_unjoined'and o['strict_inclusions']==705 and entry(base/'connected-endpoint.json')==v['endpoint'],'complete original root/endpoint history')
   prior=v
   indexed.append((run,pc))
 need([pc['attempt']for _,pc in indexed]==SELECTED,'fixed24 exact queue')
 w.save('input-binding.json',dict(card=card,inventories=rows,native_source=ib['native_source'],model=ib['model'],all_attempts_retained=24,prior_independently_passed_packet='maintained002-all20',derived_producer_proofs_unparsed=True));return indexed,ep

def geometry(run,pc):
 ds=sorted((run/'events').glob('*-cubic_correlated_domain.json.gz'));ps=sorted((run/'events').glob('*-cubic_predictor_return.json.gz'));need(len(ds)==len(ps)==1,'unique actual domain/proposal');dom=read(ds[0]);proposal=read(ps[0]);data=B.geometry(dom,proposal,pc['settings'],pc,allowed_widths=(-7,-6,-5,-4,-3));C0=B.exact_inverse(proposal['preconditioner_exact']);del proposal;gc.collect();return data,C0,dom

def tube(run,pc,previous,w,index,rotated):
 data,C0,dom=geometry(run,pc);c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=data;sc=R.exact(F(pc['settings']['parameter']));w.stage=f'chart{index:03d}'
 if rotated:
  need(previous['chart']['normal']!=pc['normal']or previous['chart']['origin']!=pc['origin'],'actual changed gauge');parameter=None
 else:
  parameter=same_gauge_parameter(previous,pc)
 join=H.cubic_membership(previous['endpoint'],c,s,a,b,normal,origin,sc,h,radii,known_parameter=parameter);w.save(w.stage+'/known-root-membership.json',join);need(join['passed']and join['coordinates']==705,'all705 strict known-root projected/exact join')
 J=A.rebuild_J(run,data,w)
 for k,v in enumerate(J):w.matrix(w.stage+f'/jacobian_coefficient_{k}.json.gz',v)
 cc=H.residual_coefficients(c,s,a,b,arb(0),B.acquisitions(run,'cubic_path',0),13,18,normal,origin,sc);wc=H.residual_coefficients(c,s,a,b,d,B.acquisitions(run,'cubic_path',1),13,18,normal,origin,sc);res=[arb_mat([[cc[i,k]if k<4 else wc[i,4]]for i in range(705)])for k in range(5)]
 for k,v in enumerate(res):w.matrix(w.stage+f'/residual_coefficient_{k}.json.gz',v)
 proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[39*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];frame=[i for i in range(13)if not whole[26+i].contains(0)]
 gates=dict(uniform_root=proof['passed'],known_root=join['passed'],primitive18=all(v['coordinates']for v in proper),independent_frame=bool(frame),positive_rates_angle=all(x>0 for x in whole[-3:]));ends=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  x=sign*h;ends.append(vector_json([c[i]+s[i]*x+a[i]*x*x/2+b[i]*x*x*x+R.original_ball(corr[i])for i in range(705)]))
 scf=F(pc['settings']['parameter']);fh=F(2)**pc['settings']['halfwidth_exponent'];chart=dict(shooting_dimension=705,normal=pc['normal'],origin=pc['origin'],parameter_center=str(scf),parameter_endpoints=list(map(str,[scf-fh,scf+fh])),settings=dict(halfwidth_exponent=pc['settings']['halfwidth_exponent'],radius_exponent=-10));passed=all(gates.values())
 w.save(w.stage+'/independent-proof.json',dict(passed=passed,gates=gates,root=proof,anchor=join,proper_divisors=proper,frame_nonzero_coordinates=frame,chart=chart,endpoints=ends));need(passed,'all strict independent root/primitive/frame/rate gates')
 return dict(chart=chart,endpoint=ends[1]),dict(passed=passed,strict_inclusions=sum(proof['coordinate_inclusions']),anchor_coordinates=join['coordinates'],contraction=proof['contraction'],rotated=rotated,parameter_endpoints=chart['parameter_endpoints'])

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.stage='profile'
 def tick(self):need(not self.stopped and time.monotonic()-self.start<4500 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'4500s/6GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(14336 if closing else 14332)*1024**2,'14GiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
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
      f.write(((','if i else'')+json.dumps(row,separators=(',',':'))).encode());need(start+raw.tell()<14332*1024**2,'derived matrix cap preserves final reserve');self.tick()
     f.write(b']}')
    raw.flush();os.fsync(raw.fileno())
  finally:self.used=start+(p.stat().st_size if p.exists()else 0)
  need(self.used<14332*1024**2,'full derived matrix byte cap')

 def emit(self,tag,value):
  if isinstance(value,arb_mat):self.matrix(self.stage+'/'+tag+'.json.gz',value)
  else:self.save(self.stage+'/'+tag+'.json',value)
  self.tick()

def synthetic(w,card):
 # The completed19 replay supplies a conservative two-times average cost,
 # including full prior profile/binding overhead; fresh full705 floor also applies.
 spec=card['anchor'];r=Path(spec['root']);need(entry(r/'manifest.json')['sha256']==spec['outer']and entry(r/'run/manifest.json')['sha256']==spec['worker'],'literal measured prior manifest')
 manifest=read(r/'run/manifest.json');po=read(r/'run/outcome.json');need(entry(r/'run/outcome.json')==manifest['files']['outcome.json']and po['passed']and po['native_calls']==0,'measured actual prior result')
 C,J,res,h,rho=B.synthetic_fixture();before=time.monotonic();used=w.used;proof=H.polynomial_root(C,J,res,[arb(1)]*705,rho,h,[arb(0,rho)]*705,w.tick,w.emit);need(proof['passed'],'fresh full705 degree5 synthetic');elapsed=time.monotonic()-before;stored=w.used-used
 priorbytes=sum(x['bytes']for x in read(r/'manifest.json')['files'].values());seconds=max(elapsed+24*(3*elapsed+75),2*24*po['elapsed_seconds']/19+90+elapsed);raw=max(73*stored+128*1024**2,2*24*priorbytes/19+128*1024**2);peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3,2*po['peak_rss_kib']*1024+256*1024**2)
 report=dict(passed=bool(seconds<4500 and raw<14*1024**3 and peak<6*1024**3),seconds=elapsed,stored_bytes=stored,forecast_seconds=seconds,forecast_bytes=raw,forecast_peak_bytes=peak,charts=24,dimension=705,target_arrays_unopened=True,prior=spec);w.save('profile.json',report);need(report['passed'],'measured fixed24 time/RSS/storage admission');del C,J,res;gc.collect()

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in B.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,4500);result=dict(passed=False,connected=False,status='incomplete',native_calls=0,charts=[])
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads=dict.fromkeys(B.THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(w.used+len(raw)<14332*1024**2,'source bytes reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   card=read(ROOT/CARD);selection(card);synthetic(w,card);runs,previous=bind(card,w);turn={}
   for unit,(run,pc)in enumerate(runs,1):
    rotated=pc['attempt']==24;need(rotated==(pc['mode']=='rotated_gauge'),'only exact indexed rotation');previous,row=tube(run,pc,previous,w,unit,rotated);row['attempt']=pc['attempt'];result['charts'].append(row);w.save(f'connected-endpoint-{unit:03d}.json',previous)
    if unit in(16,17,18):turn[unit]=R.original_ball(previous['endpoint'][-1])
    w.tick();gc.collect()
   need(len(result['charts'])==24 and previous['chart']['parameter_endpoints'][1]=='511/2048','complete fixed24 successor range');gaps=[turn[17]-turn[16],turn[17]-turn[18]];result['interior_angle_maximum']=dict(passed=all(v>0 for v in gaps),endpoint_indices=[16,17,18],gaps=[ball_json(v)for v in gaps],scope='Continuity and strict middle endpoint inequalities give at least one interior angle maximum; no uniqueness or nondegeneracy.');w.save('angle-maximum.json',result['interior_angle_maximum']);w.save('connected-endpoint.json',previous);result.update(passed=True,connected=True,status='independently_connected_all24_maintained003_charts',connected_count=24,physical_right=previous['endpoint'][-3:],scope='Independent480 acquired-domain full705 cubic root and exact ordered full-state joins from independently proved maintained002 final endpoint. Shared native acquisitions and Arb. All24 histories retained. No complementary spectrum, intermediate physical monotonicity, distant junction or complete-window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
