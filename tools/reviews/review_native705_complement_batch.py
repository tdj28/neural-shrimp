"""Four fixed saved-only independent705 tangent/complement audits, CPU1."""
import gc,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_native705_complement as A
H,B=A.H,A.B
need,R,M=A.need,A.R,A.M
entry,read,inventory,historical,bundle,exact_cover,equal_balls,save_bound=A.entry,A.read,A.inventory,A.historical,A.bundle,A.exact_cover,A.equal_balls,A.save_bound
THREADS,NATIVE,PATH_FILES=A.THREADS,A.NATIVE,A.PATH_FILES
OUT=Path('/data2/neural-shrimp-data/audits/native705-complement-batch-independent/attempt001/run')
CARD='docs/plans/native705-complement-batch-independent-inputs.json'
FILES=sorted(set(A.FILES+[CARD,'docs/plans/native705-complement-batch-independent.md','tools/reviews/review_native705_complement_batch.py','tools/reviews/launch_native705_complement_batch_review.py','tests/test_native705_complement_batch_review.py','docs/reviews/native705-complement-independent.json']))


def fixed_settings():
 return dict(output=str(OUT),units=[1,2,3,4],precision=480,seconds=240,rss_bytes=4*1024**3,as_bytes=6*1024**3,output_bytes=64*1024**2,closing_reserve_bytes=2*1024**2,own_intersections=6,forecast_multiplier=16,forecast_allowance_seconds=170)


def geometry(unit):
 need(type(unit)is int and 1<=unit<=4,'only fixed successor1..4');return H.geometry(unit)


def path_event(inp,out,index):
 need(1<=index<=72 or 109<=index<=180,'only actual four path groups');need(inp['kind']==out['kind']=='path','literal native path kind')
 return A.path_event(inp,out,index if index<=72 else index-36)


def batch_done(row,source,original=False):
 need(row.get('commit',row.get('source_commit'))==source and row['passed']and not row['owned_live_pids']and len(row['workers'])==4,'complete literal owned four-chart batch')
 if original:need(row['connected_count']==4 and row['connected_right']=='319/4096','complete original ordered connection')
 for i,v in enumerate(row['workers'],1):
  need(v['unit']==i and v['exit_code']==0 and not v['owned_live_pids'],'all ordered completed workers')
  pid=v['group']if original else v['group_pid'];need(not Path('/proc',str(pid)).exists(),'prior exact owned group absent')


def shared_bind(card,tick):
 need(card['schema']=='independent-four705-complement-inputs-v1'and card['original']['units']==[1,2,3,4]and card['audit']==fixed_settings(),'fixed entire four-chart audit card')
 old=card['original'];pr=Path(old['producer']['root']);qr=Path(card['complement']['root']);rows=[inventory(pr,old['producer']['outer'],tick),inventory(qr,card['complement']['outer'],tick)]
 batch_done(read(pr/'outcome.json'),old['producer']['source'],True);batch_done(read(qr/'launcher-result.json'),card['complement']['source'])
 ir,irt,io,rr=bundle(old['independent'],tick);rows+=rr;ib=read(ir/'run/input-binding.json')
 need(io['passed']and io['connected']and io['connected_count']==4 and io['native_calls']==0 and len(ib['charts'])==4,'independently connected full four-chart root family')
 need(ib['batch']==str(pr)and ib['batch_manifest']==old['producer']['outer']and ib['producer_source']==old['producer']['source'],'literal independent parent root identity')
 fr,frt,fo,rr=bundle(old['frozen_basis'],tick,2);rows+=rr;fb=read(fr/'run/bindings.json');prop=read(fr/'run/events/001-proposal.json')
 need(not fo['passed']and fo['native_calls']==0 and fo['status']=='completed_sufficient_bound_unresolved'and prop['strategy']=='periodic_lyapunov_balanced_orthogonal'and fb['card']['producer']==old['physical_origin'],'retained same-map basis proposal failure')
 phys,prt,po,rr=bundle(old['physical_origin'],tick);rows+=rr;need(po['passed']and entry(phys/'run/certificate.json.gz')['sha256']==old['physical_origin']['certificate'],'literal original physical root')
 first,_,done,rr=bundle(card['first_audit'],tick);rows+=rr;need(done['passed']and done['native_calls']==0 and done['status']=='independently_qualified_first705_complement'and done['parameter_endpoints']==['-1/4096','63/4096']and done['elapsed_seconds']<19,'passed measured first705 predecessor')
 # Bind the original four joins to exactly the independently audited first tube.
 previous=read(first/'run/input-bindings.json')['card']['inputs']['original']['curve'];need(ib['prior']=={k:previous['producer'][k]for k in ('outer','source','worker')}and ib['prior_join']==previous['connection']['root']and ib['prior_uniform']==previous['independent']['root'],'exact original first705 branch connection')
 return dict(pr=pr,qr=qr,ir=ir,ib=ib,prop=prop,physical=read(phys/'run/inputs.json')['map_inputs'],physical_runtime=prt,rows=rows)


def bind_unit(card,unit,shared,tick):
 sc,ends=geometry(unit);old=card['original'];pr=shared['pr']/f'chart{unit:03d}/run';qr=shared['qr']/f'chart{unit:03d}/run';rows=[inventory(pr,old['producer']['workers'][unit-1],tick),inventory(qr,card['complement']['workers'][unit-1],tick)];rt=historical(pr,old['producer']['source']);qt=historical(qr,card['complement']['source']);po=read(pr/'outcome.json');qo=read(qr/'outcome.json')
 need(po['passed']and po['status']=='uniform705_unjoined'and not po['connected']and po['unit']==unit and po['strict_inclusions']==705 and po['ticks']==253 and po['events']['direction_return']==144 and po['events']['pal_path_phase_return']==108 and po['parameter_endpoints']==ends,'complete original exact unit')
 need(qo['passed']and qo['status']=='whole_chart_complement_attracting'and qo['unit']==unit and qo['native_calls']==216 and qo['complement_dimension']==11 and qo['parameter_endpoints']==ends and entry(qr/'complement.json')==card['complement']['proofs'][unit-1],'literal completed216-call complement unit')
 need(qt['precision']==384 and qt['native_call_cap']==216 and qt['nice']>=10 and qt['threads']==dict.fromkeys(THREADS,'1'),'actual native settings')
 selected=shared['ib']['charts'][unit-1];cert=pr/'certificate.json.gz';need(selected==dict(root=str(pr),manifest=entry(pr/'manifest.json'),certificate=entry(cert)),'exact independently proved certificate')
 ir=shared['ir'];upath=ir/f'run/chart-{unit}/uniform.json';jpath=ir/f'run/chart-{unit}/join.json';uniform=read(upath);join=read(jpath)
 need(uniform['inclusions']==705 and all(v for k,v in uniform['gates'].items()if k!='anchor_identified')and join['passed']and join['coordinates']==705 and join['parameter']==ends[0],'all705 independent uniform and exact current join')
 qb=read(qr/'bindings.json');inp=read(pr/'inputs.json')['map_inputs'];need(qb['card']==old and qb['unit']==unit and qb['parameter_endpoints']==ends and qb['original_run']==str(pr)and qb['original_certificate']==entry(cert)and qb['independent_uniform']==entry(upath)and qb['independent_join']==entry(jpath)and qb['shared_native_acquisitions']==252,'exact complement same-root binding')
 need(inp==qb['map_inputs']==shared['physical'],'exact model/data/both physical rates')
 for name in NATIVE:need(qt['sources'][name]==rt['sources'][name]==shared['physical_runtime']['sources'][name],'unchanged native derivative source '+name)
 for name in PATH_FILES:need(qt['sources'][name]==rt['sources'][name],'unchanged native path source '+name)
 directions=sorted((pr/'events').glob('*-direction_return.json.gz'));paths=sorted((pr/'events').glob('*-pal_path_phase_return.json.gz'));need(len(directions)==144 and len(paths)==108,'all252 original actual acquisitions')
 td=read(qr/'tangent-domain-inputs.json');dom=read(qr/'quotient-domain-inputs.json');loc=read(qr/'localization.json');tan=read(qr/'tangent.json');need(td['parameter_endpoints']==dom['parameter_endpoints']==ends and dom['halfwidth']=='1/128'and td['scale']=='1/256','same complete actual gauge')
 with ctx.workprec(384):full={'center':A.restored(loc['center_box']),'whole':A.restored(loc['whole_box']),'tangent':A.restored([v[0]for v in tan['tangent_box']])}
 for k,v in full.items():need(len(v)==705,'full705 '+k);equal_balls(dom['full705'][k],v,'actual all705 '+k);equal_balls(dom[k],v[:-1],'drop only angle '+k)
 E=[R.matrix(v)for v in dom['complements']];metrics=[R.matrix(v)for v in dom['metrics']];need(len(E)==len(metrics)==18,'all18 unchanged frames')
 for j in range(18):need(E[j]==R.matrix(shared['prop']['phase_complements'][j])and metrics[j]==arb_mat([[R.exact(F.from_float(x))for x in row]for row in shared['prop']['upper_cholesky_factors'][j]]),'exact frozen E/B proposals')
 A.native_inventory(qr,216);events=[];native=[]
 for i in range(1,217):
  a=read(qr/f'native/{i:03d}-input.json');v=read(qr/f'native/{i:03d}-return.json')
  if i<=72 or 109<=i<=180:
   event=path_event(a,v,i)
   if i>=109:events.append(event)
  elif i<=108:need(a['kind']==v['kind']=='synthetic'and a['phase']==v['phase']==i-73,'complete36 unrelated derivative profile')
  else:
   j=(i-181)//2;kind='center'if i%2 else'whole_tangent';matrix,deriv=A.native_return(a,v,dom,kind,j)
   if kind=='center':A0=matrix
   else:native.append((A0,matrix,deriv))
 need(len(events)==72 and len(native)==18,'all target72paths and36 derivative acquisitions')
 records=[read(qr/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(qr/'events/020-metric.json');profile=read(qr/'profile.json');need(profile['passed']and profile['native_calls']==108 and profile['target_arrays_unopened']and profile['forecast_seconds']<400,'original measured profile admission')
 chart=read(cert);ci=read(pr/'chart-inputs.json');need(all(chart[k]==ci[k]for k in ('normal','origin','weights')),'same exact gauge and weights')
 keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','curvature','weights','coordinate_radii','preconditioner','whole_box','normal','origin'}
 for k in list(chart):
  if k not in keep:del chart[k]
 dg=[read(p)for p in directions];pg=[read(p)for p in paths];gc.collect();tick()
 return chart,dg,pg,td,dom,E,metrics,native,records,metric,events,dict(unit=unit,parameter_endpoints=ends,inventories=rows,certificate=entry(cert),original_native_files={str(p.relative_to(pr)):entry(p)for p in directions+paths},independent_uniform=entry(upath),independent_join=entry(jpath),native_sources={n:rt['sources'][n]for n in NATIVE+PATH_FILES})


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];result=dict(passed=False,native_calls=0,status='incomplete',rows=[])
 def save(n,v,closing=False):save_bound(OUT,n,v,used,closing)
 def tick():need(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=os.getpriority(os.PRIO_PROCESS,0),threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(b).hexdigest(),'committed archive');need(used[0]+len(b)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.synthetic(705,tick);seconds=time.monotonic()-before;forecast=16*seconds+170;save('profile.json',dict(trial,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<240,'measured four-unit sequential admission');gc.collect()
   card=read(ROOT/CARD);shared=shared_bind(card,tick);save('input-bindings.json',dict(card=card,inventories=shared['rows']))
   for unit in range(1,5):
    prefix=f'chart-{unit}/';chart,dg,pg,td,dom,E,metrics,native,records,metric,events,binding=bind_unit(card,unit,shared,tick);save(prefix+'input-bindings.json',binding)
    data,proof=H.localize(chart,dg,pg,tick,unit);save(prefix+'localization-480.json',proof);del chart,dg,pg,proof;gc.collect();tick()
    for k in ('center','whole'):exact_cover(dom['full705'][k],data[k],'fresh own480 '+k)
    for k,field in [('segment','segment'),('error','error'),('predictor_tangent','pprime')]:exact_cover(td[k],data[field],'actual '+k)
    b=R.exact(F(1,256));plus=[data['error'][i]/b+b*data['pprime'][i]for i in range(705)];minus=[data['error'][i]/b-b*data['pprime'][i]for i in range(705)];exact_cover(td['plus'],plus,'actual plus');exact_cover(td['minus'],minus,'actual minus')
    tangent,tproof=H.correlated_tangent(data,events,tick);save(prefix+'tangent-480.json',tproof);del events,tproof;gc.collect();tick();exact_cover(dom['full705']['tangent'],tangent,'actual full705 tangent')
    save(prefix+'domain-coverage.json',dict(passed=True,coordinates=705,own_intersections=6,exact_rational_containment=True,domains=['center','whole','segment','error','predictor tangent','plus','minus','true tangent']))
    proof=H.reduction(data['center'][:-1],data['whole'][:-1],tangent[:-1],E,metrics,native,records,metric,data['halfwidth'],data['whole'][-1],tick);save(prefix+'complement-480.json',proof);tick()
    result['rows'].append(dict(unit=unit,passed=True,parameter_endpoints=geometry(unit)[1],primitive_period=18,complement_dimension=11,own_localization=True,own_correlated_tangent=True,domain_coverage=True,complementary_modulus_upper=proof['complementary_modulus_upper']));save(prefix+'outcome.json',result['rows'][-1]);del data,tangent,td,dom,E,metrics,native,records,metric,proof,binding;gc.collect()
   need(len(result['rows'])==4,'all fixed units pass');result.update(passed=True,status='independently_qualified_four_successor705_complements',parameter_endpoints=['63/4096','319/4096'],scope='Four independently connected primitive18 charts, all11 complementary modes strictly interior. Critical pair neutral; shared native analytic acquisitions/Arb. No later chart spectrum, signed NS coefficient, stable side or complete window.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
