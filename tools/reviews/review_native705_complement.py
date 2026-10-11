"""Single saved-only independent first705 tangent and complementary audit."""
import gc,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_native704_complement as B
import review_native705_complement_math as H
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need,R,M,U=H.need,H.R,H.M,H.U
entry,read,inventory,historical,bundle,exact_cover,equal_balls,save_bound=B.entry,B.read,B.inventory,B.historical,B.bundle,B.exact_cover,B.equal_balls,B.save_bound
OUT=Path('/data2/neural-shrimp-data/audits/native705-complement-independent/attempt001/run')
CARD='docs/plans/native705-complement-independent-inputs.json';THREADS=B.THREADS;NATIVE=B.NATIVE
PATH_FILES=['src/neural_shrimp/'+n+'.py'for n in('rigorous_eigenpair_locus','rigorous_eigenpair_path','rigorous_unit_pair','rigorous_unit_pair_pal')]
FILES=[CARD,'docs/plans/native705-complement-independent.md','tools/reviews/review_native705_complement.py','tools/reviews/launch_native705_complement_review.py','tests/test_native705_complement_driver.py','tests/test_native705_complement_review.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))


def native_inventory(root,count):
 need({p.name for p in(root/'native').iterdir()if p.is_file()}=={f'{i:03d}-{k}.json'for i in range(1,count+1)for k in('input','return')},'all attempted native input/return files')

def path_event(inp,out,index):
 group=(index-1)//36;phase=((index-1)%36)//2;direction='q'if index%2 else'r';stage=['profile_plus','profile_minus','target_plus','target_minus'][group]
 need(1<=index<=144 and inp['stage']==out['stage']==stage and inp['phase']==phase and inp['direction']==direction,'all144 actual ordered path calls')
 need(len(inp['inputs'])==15 and len(out['returns'])==13,'complete native path acquisition')
 return dict(phase=phase,direction=direction,d=ball_json(arb(0)),inputs=inp['inputs'],returns=out['returns'])

def native_return(inp,out,domains,kind,phase):
 need(out['kind']==kind and out['phase']==phase,'literal derivative output kind/phase');B.acquired_input(inp,domains,kind,phase)
 a,ap=R.matrix(out['A']),R.matrix(out['Aprime']);need(a.nrows()==a.ncols()==ap.nrows()==ap.ncols()==13 and H.finite(a)and H.finite(ap),'full finite native derivative return')
 if kind=='center':need(all(v.is_zero()for row in ap.tolist()for v in row),'zero center tangent derivative')
 return a,ap

def restored(v):return [ball_json(R.original_ball(x))for x in v]

def bind(card,tick):
 rows=[];original=card['inputs']['original'];curve=original['curve'];cache={}
 specs=[('producer',curve['producer'],2),('independent',curve['independent'],0),('connection',curve['connection'],0),('localization',original['localization'],0),('basis',original['frozen_basis'],2),('physical',original['physical_origin'],0),('failed',card['inputs']['failed'],2),('tangent',card['inputs']['tangent'],0),('complement',card['complement'],0)]
 for label,spec,code in specs:
  spec=dict(spec,source=card['inherited_sources'][label])if label in card['inherited_sources']else spec
  root,rt,out,rr=bundle(spec,tick,code);rows+=rr;cache[label]=(root,rt,out)
 pr,prt,po=cache['producer'];ir,irt,io=cache['independent'];cr,crt,co=cache['connection'];lr,lrt,lo=cache['localization'];fr,frt,fo=cache['failed'];tr,trt,to=cache['tangent'];qr,qrt,qo=cache['complement'];br,brt,bo=cache['basis'];phys,physrt,physout=cache['physical']
 need(not po['passed']and po['strict_inclusions']==705 and po['ticks']==253 and po['gates']==dict(anchor_identified=False,primitive_period=True,realified_frame_independent=True,uniform_full_native_unit_pair_locus=True,whole_angle_positive=True),'preserved original coarse-anchor failure with strict entire curve')
 need(io['passed']and io['uniform_primitive_positive_unit_pair']and io['inclusions']==705 and io['native_calls']==0 and co['passed']and co['connected']and co['anchor_coordinates']==705 and co['native_calls']==0,'independent entire primitive705 curve and exact same-root connection')
 ib=read(ir/'run/input-binding.json');cb=read(cr/'run/input-binding.json');cert=pr/'run/certificate.json.gz'
 need(ib['root']==str(pr)and ib['pins']=={k:curve['producer'][k]for k in('source','outer','worker')}and cb['prior705']==ib and ib['certificate']==entry(cert),'literal independent original root identity')
 directions=sorted((pr/'run/events').glob('*-direction_return.json.gz'));paths=sorted((pr/'run/events').glob('*-pal_path_phase_return.json.gz'));need(len(directions)==144 and len(paths)==108,'complete original144+108 acquisition inventory');allfiles={str(p.relative_to(pr/'run')):entry(p)for p in directions+paths};need(ib['events']==allfiles,'exact independently audited native acquisitions')
 up=read(ir/'run/native705-384.json');need(up['inclusions']==705 and up['gates']['uniform_root']and up['gates']['primitive_period']and up['gates']['frame_independent']and up['gates']['positive_angle'],'actual prior independent uniform arithmetic');del up
 lb=read(lr/'run/bindings.json');loc=read(lr/'run/localization.json');need(lo['passed']and lo['status']=='conditional_full705_localization_completed'and lo['native_calls']==0 and lb['card']=={k:{a:b for a,b in v.items()if a!='source' or k=='producer'}for k,v in curve.items()}and entry(lr/'run/localization.json')['sha256']==original['localization']['proof'],'same conditional root localization')
 oldcurve={k:{a:b for a,b in v.items()if a!='source' or k=='producer'}for k,v in curve.items()}
 need(not fo['passed']and fo['status']=='completed_correlated_bound_unresolved'and fo['native_calls']==72 and not bo['passed']and bo['status']=='completed_sufficient_bound_unresolved'and bo['native_calls']==0,'retained failed complement and frozen basis')
 # The inherited input card is byte-exact; added independent source pins live
 # in this audit card separately and do not rewrite the producer's card.
 tb=read(tr/'run/bindings.json');qb=read(qr/'run/bindings.json');fb=read(fr/'run/bindings.json')
 need(tb['card']['curve']==oldcurve and tb['card']['localization']==original['localization']and tb['card']['failed_complement']==card['inputs']['failed'],'same-root tangent binding')
 need(qb['card']==card['inputs']and fb['card']==original and qb['new_target_calls']==qb['shared_target_center_calls']==18,'exact refined quotient and original failed center bindings')
 need(to['passed']and to['status']=='conditional_correlated705_tangent_completed'and to['native_calls']==144 and not to['complementary_stability_evaluated']and entry(tr/'run/tangent.json')['sha256']==card['inputs']['tangent']['proof'],'complete literal correlated tangent proof')
 need(qo['passed']and qo['status']=='first705_refined_complement_attracting'and qo['native_calls']==54 and qo['complement_dimension']==11 and entry(qr/'run/complement.json')['sha256']==card['complement']['proof'],'complete literal refined quotient proof')
 for rt in [trt,qrt,frt]:need(rt['precision']==384 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'actual native settings')
 inp=read(pr/'run/inputs.json')['map_inputs'];need(inp==read(phys/'run/inputs.json')['map_inputs']==tb['map_inputs']==fb['map_inputs']==qb['unchanged_root']['map_inputs'],'same complete native model/data/both rates')
 for name in NATIVE:
  need(all(rt['sources'][name]==prt['sources'][name]for rt in [trt,qrt,frt,physrt]),'identical physical derivative sources '+name)
 for name in PATH_FILES:need(trt['sources'][name]==prt['sources'][name],'identical analytic path source '+name)
 need(qb['unchanged_root']['card']==original and qb['tangent']==card['inputs']['tangent']and qb['unchanged_root']['localization']==original['localization'],'all unchanged actual root domains')
 domains=read(qr/'run/actual-domain-inputs.json');td=read(tr/'run/actual-domain-inputs.json');tproof=read(tr/'run/tangent.json');need(domains['exact_endpoints']==td['exact_parameter_endpoints']==['-1/4096','63/4096']and domains['halfwidth']=='1/128'and td['scale']=='1/256','whole same gauge and balanced scale')
 with ctx.workprec(384):
  full={'center':restored(loc['center_box']),'whole':restored(loc['whole_box']),'tangent':restored([v[0]for v in tproof['tangent_box']])}
 for k,v in full.items():equal_balls(domains['full705_localized_domains'][k],v,'all705 actual '+k);equal_balls(domains[k],v[:-1],'drop only actual angle from '+k)
 need(all(len(v)==705 for v in full.values()),'complete state/modes/rates/angle before projection');del loc,tproof
 prop=read(br/'run/events/001-proposal.json');bb=read(br/'run/bindings.json');need(prop['strategy']=='periodic_lyapunov_balanced_orthogonal'and bb['card']['producer']==original['physical_origin']and qb['unchanged_root']['fixed_proposals']==entry(br/'run/events/001-proposal.json'),'literal fixed same-map basis proposals')
 E=[R.matrix(v)for v in domains['complements']];metrics=[R.matrix(v)for v in domains['metrics']];need(len(E)==len(metrics)==18,'all18 complementary frames')
 for j in range(18):need(E[j]==R.matrix(prop['phase_complements'][j])and metrics[j]==arb_mat([[R.exact(F.from_float(x))for x in row]for row in prop['upper_cholesky_factors'][j]]),'exact unchanged dyadic frame/metric proposals')
 for root,count in [(tr,144),(qr,54),(fr,72)]:native_inventory(root/'run',count)
 events=[]
 for i in range(1,145):
  event=path_event(read(tr/f'run/native/{i:03d}-input.json'),read(tr/f'run/native/{i:03d}-return.json'),i)
  if i>72:events.append(event)
 native=[];selected={}
 for i in range(1,73):
  a=read(fr/f'run/native/{i:03d}-input.json');v=read(fr/f'run/native/{i:03d}-return.json');kind='synthetic'if i<=36 else('center'if i%2 else'whole_tangent');phase=i-1 if i<=36 else(i-37)//2;need(a['kind']==v['kind']==kind and a['phase']==v['phase']==phase,'all retained failed-bound acquisition order')
 for i in range(1,55):
  a=read(qr/f'run/native/{i:03d}-input.json');v=read(qr/f'run/native/{i:03d}-return.json');kind='synthetic'if i<=36 else'whole_tangent';j=i-1 if i<=36 else i-37;need(a['kind']==v['kind']==kind and a['phase']==v['phase']==j,'all54 actual quotient acquisition order')
  if i>36:
   ip=fr/f'run/native/{37+2*j:03d}-input.json';vp=fr/f'run/native/{37+2*j:03d}-return.json';A0,_=native_return(read(ip),read(vp),domains,'center',j);Aw,Ap=native_return(a,v,domains,kind,j);native.append((A0,Aw,Ap));selected[str(ip.relative_to(fr/'run'))]=entry(ip);selected[str(vp.relative_to(fr/'run'))]=entry(vp)
 need(selected==qb['retained_center_derivatives'],'all18 exact shared center acquisitions')
 records=[read(qr/f'run/events/{j+2:03d}-phase.json')for j in range(18)];metric=read(qr/'run/events/020-metric.json');chart=read(cert)
 keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','curvature','weights','coordinate_radii','preconditioner','whole_box','normal','origin'}
 for k in list(chart):
  if k not in keep:del chart[k]
 dg=[read(p)for p in directions];pg=[read(p)for p in paths];gc.collect();tick()
 return chart,dg,pg,td,domains,E,metrics,native,records,metric,events,{'card':card,'inventories':rows,'certificate':entry(cert),'all_original_acquisitions':allfiles,'shared_center_acquisitions':selected,'native_source':{n:prt['sources'][n]for n in NATIVE+PATH_FILES}}


def synthetic(n,tick):
 need(type(n)is int and 1<=n<=705,'bounded full arithmetic synthetic');small=arb((1,-16));C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)]);D=M.ident(n)-C*J;tick();w=[arb(1)]*n;k=H.infnorm(D);v=arb_mat([[arb((1,-80))]for _ in range(n)]);initial=arb_mat([[arb(0,arb((1,-10)))]for _ in range(n)])
 for sign in(1,-1,2):need(H.finite(H.H.enclosure(D,-C*(v*sign),w,k,initial,iterations=6)),'full synthetic six-intersection enclosure');tick()
 return {'passed':True,'dimension':n,'intersections':18,'neumann':ball_json(k)}


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10')
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];result={'passed':False,'native_calls':0,'status':'incomplete'}
 def save(n,v,closing=False):save_bound(OUT,n,v,used,closing)
 def tick():need(not stop[0]and time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'180s/4GiB cap')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,180)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'sources':{n:entry(ROOT/n)for n in FILES},'precision':480,'native_calls':0,'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(THREADS,'1')})
  for n in FILES:
   b=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(b).hexdigest(),'exact committed archive source');need(used[0]+len(b)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();trial=synthetic(705,tick);seconds=time.monotonic()-before;forecast=4*seconds+122;save('profile.json',dict(trial,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<180,'measured full705 saved arithmetic admission');gc.collect()
   card=read(ROOT/CARD);need(card['audit']==dict(output=str(OUT),precision=480,seconds=180,rss_bytes=4*1024**3,as_bytes=6*1024**3,output_bytes=64*1024**2,closing_reserve_bytes=2*1024**2,own_intersections=6,forecast_multiplier=4,forecast_allowance_seconds=122),'literal fixed audit settings')
   chart,dg,pg,td,dom,E,metrics,native,records,metric,events,binding=bind(card,tick);save('input-bindings.json',binding)
   data,proof=H.localize(chart,dg,pg,tick);save('localization-480.json',proof);del chart,dg,pg;gc.collect();tick()
   for key in('center','whole'):exact_cover(dom['full705_localized_domains'][key],data[key],'fresh own480 actual '+key)
   for key,field in [('segment','segment'),('error','error'),('predictor_tangent','pprime')]:exact_cover(td[key],data[field],'fresh own480 acquired '+key)
   b=R.exact(F(1,256));plus=[data['error'][i]/b+b*data['pprime'][i]for i in range(705)];minus=[data['error'][i]/b-b*data['pprime'][i]for i in range(705)]
   exact_cover(td['plus'],plus,'fresh own balanced plus');exact_cover(td['minus'],minus,'fresh own balanced minus')
   tangent,tproof=H.correlated_tangent(data,events,tick);save('tangent-480.json',tproof);del events;gc.collect();tick();exact_cover(dom['full705_localized_domains']['tangent'],tangent,'all705 fresh actual tangent')
   save('domain-coverage.json',dict(passed=True,coordinates=705,own_intersections=6,exact_rational_containment=True,domains=['center','whole','predictor-root segment','error','predictor tangent','plus','minus','true tangent']))
   proof=H.reduction(data['center'][:-1],data['whole'][:-1],tangent[:-1],E,metrics,native,records,metric,data['halfwidth'],data['whole'][-1],tick);save('complement-480.json',proof);tick()
   result.update(passed=True,status='independently_qualified_first705_complement',parameter_endpoints=['-1/4096','63/4096'],primitive_period=18,complement_dimension=11,own_localization=True,own_correlated_tangent=True,domain_coverage=True,complementary_modulus_upper=proof['complementary_modulus_upper'],scope=proof['scope'])
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
