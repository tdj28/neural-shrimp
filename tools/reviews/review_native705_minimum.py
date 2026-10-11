"""Fixed two-root independent continuation and qualitative hidden-rate minimum."""
import gc,json,os,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import review_cubic705_prefix as B
import review_native705_minimum_math as A
from neural_shrimp.rigorous_orbits import ball_json,vector_json
H,R,M,U,need=B.H,B.R,B.M,B.U,B.need
read,entry,inventory,historical=B.read,B.entry,B.inventory,B.historical
OUT=Path('/data2/neural-shrimp-data/audits/native705-minimum-independent/attempt001/run')
CARD='docs/plans/native705-minimum-independent-inputs.json'
FILES=[CARD,'docs/plans/native705-minimum-independent.md','tools/reviews/review_native705_minimum.py','tools/launch_native705_minimum_review.py','tests/test_native705_minimum_math.py','tests/test_native705_minimum_driver.py','tests/test_cubic705_review.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
NATIVE=B.OLD.NATIVE+['src/neural_shrimp/'+n+'.py'for n in('rigorous_gradient_path','rigorous_cubic_unit_pair','rigorous_cubic_gradient_path')]

def settings():return dict(precision=480,seconds=900,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=2048*1024**2,closing_reserve=4*1024**2,charts=2,native_calls=0)
def source_owner(spec,w):
 root=Path(spec['root']);rows=[inventory(root,spec['outer'],w.tick),inventory(root/'run',spec['worker'],w.tick)];rt=historical(root/'run',spec['source']);la=read(root/'launcher-result.json');owner=read(root/'owned-process.json')
 need(la['exit_code']==0 and not la['owned_live_pids']and not Path('/proc',str(owner.get('group',owner.get('group_pid')))).exists(),'stopped exact owned successful worker')
 need(rt['threads']==dict.fromkeys(B.THREADS,'1')and rt['nice']>=10,'same actual CPU1nice10 runtime');return root,rt,read(root/'run/outcome.json'),rows

def worker_contract(root,rt,out,source,attempt,mode,model,native):
 pc=read(root/'input-card.json');inp=read(root/'run/inputs.json');need(pc['source']==source and pc['attempt']==rt['attempt']==out['attempt']==attempt and pc['mode']==mode and pc['kernel']=='fresh-cubic705-maintained-v1','literal retained worker identity')
 need(pc['settings']==rt['settings']==inp['settings']and pc['map_inputs']==inp['map_inputs']==model and inp['card']==entry(root/'input-card.json')and pc['anchor']==inp['anchor'],'same entire map/input/scales/anchor')
 for n in NATIVE:need(rt['sources'][n]==native[n],'exact native derivative source '+n)
 need(out['passed']and out['status']=='uniform705_unjoined'and not out['connected']and out['ticks']==331 and out['strict_inclusions']==705 and out['events']==B.EXPECTED,'complete original full unjoined root acquisition')
 events=sorted((root/'run/events').glob('*.json.gz'));need(len(events)==sum(B.EXPECTED.values())and[int(p.name.split('-')[0])for p in events]==list(range(len(events))),'complete ordered original events')
 for k,num in B.EXPECTED.items():need(len(list((root/'run/events').glob('*-'+k+'.json.gz')))==num,'literal original event types')
 return pc

def bind(card,w):
 need(card['schema']=='independent-root5-successor-minimum-v1'and card['settings']==settings(),'fixed exact source/card')
 prior,prt,po,rows=B.OLD.bundle(card['prior'],w.tick);ep=read(prior/'run/connected-endpoint.json');ib=read(prior/'run/input-binding.json');proof=read(prior/'run/chart003/independent-proof.json')
 need(po['passed']and po['connected']and po['connected_count']==3 and po['connected_right']=='63/512'and po['native_calls']==0 and prt['precision']==480 and ib['selected_attempts']==[1,2,4],'independent original complete root4 authority')
 need(entry(prior/'run/connected-endpoint.json')==card['prior']['endpoint']and entry(prior/'run/chart003/independent-proof.json')==card['prior']['proof']and ep['chart']==proof['chart']and ep['endpoint']==proof['endpoints'][1],'literal independently derived endpoint')
 need(proof['passed']and all(proof['gates'].values())and len(proof['root']['coordinate_inclusions'])==705 and all(proof['root']['coordinate_inclusions'])and proof['anchor']['coordinates']==705,'strict entire independent root and known-root join')
 maintain=card['maintained'];need(ib['card']['producer']['root']==maintain['root']and ib['card']['producer']['source']==maintain['source']and ib['card']['producer']['outer']==maintain['outer'],'actual independently audited historical source')
 rows.append(inventory(Path(maintain['root']),maintain['outer'],w.tick));mo=read(Path(maintain['root'])/'outcome.json');need(mo['status']=='mathematical_stop'and mo['connected_count']==3 and mo['attempted']==5 and not mo['owned_live_pids'],'preserved complete old failure/prefix')
 old,ort,oo,rr=source_owner(card['old'],w);rows+=rr;native={n:ort['sources'][n]for n in NATIVE};model=read(old/'input-card.json')['map_inputs'];oldpc=worker_contract(old,ort,oo,maintain['source'],4,'same_gauge',model,native)
 need(ib['card']['producer']['workers'][3]['worker']==card['old']['worker']and proof['chart']['parameter_endpoints']==['47/512','63/512']and proof['chart']['normal']==oldpc['normal']and proof['chart']['origin']==oldpc['origin'],'exact old4 acquired root geometry')
 fifth,frt,fo,rr=source_owner(card['root5'],w);rows+=rr;fpc=worker_contract(fifth,frt,fo,maintain['source'],5,'rotated_gauge',model,native)
 need(fpc['parent']['root']==str(old)and fpc['parent']['worker']==card['old']['worker']and fpc['parent']['outer']==card['old']['outer'],'literal root4 to retained root5 genealogy')
 fb,brt,bo,rr=source_owner(card['fallback'],w);rows+=rr;need(bo['passed']and bo['native_calls']==72 and bo['saved_replay_calls']==144 and bo['events']==72 and bo['strict_inclusions']==705,'complete stopped fallback producer history')
 for n in NATIVE:need(brt['sources'][n]==native[n],'same fallback native map '+n)
 pi=read(fb/'run/point-input.json');need(entry(fb/'run/point-input.json')==card['fallback']['point_input']and pi['model']==model and pi['normal']==oldpc['normal']and pi['origin']==oldpc['origin']and pi['parameter']=='63/512','same full fixed fresh point residual map/gauge')
 binding=read(fb/'run/binding.json');need(binding['old_worker']['sha256']==card['old']['worker']and binding['new_worker']['sha256']==card['root5']['worker'],'actual fixed fallback old/new native proofs')
 events=sorted((fb/'run/events').glob('*.json.gz'));need(len(events)==72 and [p.name for p in events]==[f'{i:03d}-direction_return.json.gz'for i in range(72)],'complete36profile36target residual acquisition order')
 nextroot,nrt,no,rr=source_owner(card['successor'],w);rows+=rr;npc=worker_contract(nextroot,nrt,no,card['successor']['source'],1,'same_gauge',model,native)
 need(npc['parent']['root']==str(fifth)and npc['parent']['worker']==card['root5']['worker']and npc['parent']['outer']==card['root5']['outer']and npc['parent']['authority']==dict(root=str(fb),source=card['fallback']['source'],outer=card['fallback']['outer'],worker=card['fallback']['worker'],endpoint=card['fallback']['endpoint']['sha256']),'complete retained-root/fallback source genealogy')
 need(npc['normal']==fpc['normal']and npc['origin']==fpc['origin'],'literal successor affine gauge');need(oldpc['settings']['parameter']=='55/512'and fpc['settings']['parameter']=='31/2048'and npc['settings']['parameter']=='47/1024','fixed exact three root positions')
 w.save('input-binding.json',dict(card=card,inventories=rows,native_source=native,model=model,scope='Exact full independently proved old4 root and all retained failed/successful histories. Running producer parent not opened or required. Producer derived proof/endpoint-operator math is not audit authority.'))
 return old/'run',oldpc,fifth/'run',fpc,nextroot/'run',npc,fb/'run',pi,ep,po

def geometry(run,pc):
 ds=sorted((run/'events').glob('*-cubic_correlated_domain.json.gz'));ps=sorted((run/'events').glob('*-cubic_predictor_return.json.gz'));need(len(ds)==len(ps)==1,'unique full actual domain/proposal');dom=read(ds[0]);proposal=read(ps[0]);data=B.geometry(dom,proposal,pc['settings'],pc,allowed_widths=(-6,));C0=B.exact_inverse(proposal['preconditioner_exact']);del proposal;gc.collect();return data,C0,dom

def rebuild_J(run,data,w):
 c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=data
 z=H.jacobian_coefficients(c,s,a,b,error,arb(0),B.acquisitions(run,'gradient_path',0),13,18,normal,w.tick);v=H.jacobian_coefficients(c,s,a,b,error,d,B.acquisitions(run,'gradient_path',1),13,18,normal,w.tick);return [z[0],z[1],v[2]]

def fallback(run,pc,point_run,pi,known,w):
 data,C0,dom=geometry(run,pc);c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=data
 J=rebuild_J(run,data,w);C,D,q=A.point_operator(C0,J,h,weights,w.tick);del C0,J;gc.collect();w.stage='known-endpoint';w.matrix(w.stage+'/C.json.gz',C);w.matrix(w.stage+'/D.json.gz',D)
 predictor=[c[i]+s[i]*h+a[i]*h*h/2+b[i]*h*h*h for i in range(705)];need(pi['original_error']==dom['error']and len(pi['point'])==705 and len(pi['predictor'])==705,'literal actual original error and complete finite point')
 for v,x in zip(pi['predictor'],predictor,strict=True):need(M.point(v)==M.point(ball_json(x)),'same exact endpoint predictor')
 point=[R.exact(M.point(v))for v in pi['point']];A.translated_domain(dom['error'],predictor,[known['endpoint'],pi['point']]);returns=[read(point_run/f'events/{i:03d}-direction_return.json.gz')for i in range(36,72)]
 G,unused=H.P.rebuild(point,returns,13,18,normal,origin,R.exact(F('63/512')));del unused;w.matrix(w.stage+'/independent-residual.json.gz',G);proof=A.localize(C,D,G,point,known['endpoint'],weights,w.tick);w.save(w.stage+'/localization.json',proof);del C,D,G;gc.collect();return dict(chart=known['chart'],endpoint=proof['box'])

def same_gauge_parameter(previous,pc):
 need(previous['chart']['normal']==pc['normal']and previous['chart']['origin']==pc['origin'],'literal same affine gauge before exact scalar membership')
 value=F(previous['chart']['parameter_endpoints'][1]);center=F(pc['settings']['parameter']);left=center-F(1,64);right=center+F(1,64)
 need(left<=value<=right and value-left==F(1,2048),'literal planned closed overlap and exact known scalar')
 return str(value)

def tube(run,pc,previous,w,index,rotated):
 data,C0,dom=geometry(run,pc);c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=data;sc=R.exact(F(pc['settings']['parameter']));w.stage=f'chart{index:03d}'
 if rotated:
  need(previous['chart']['normal']!=pc['normal']or previous['chart']['origin']!=pc['origin'],'actual changed gauge');parameter=None
 else:
  parameter=same_gauge_parameter(previous,pc)
 join=H.cubic_membership(previous['endpoint'],c,s,a,b,normal,origin,sc,h,radii,known_parameter=parameter);w.save(w.stage+'/known-root-membership.json',join);need(join['passed']and join['coordinates']==705,'all705 strict known-root projected/exact join')
 J=rebuild_J(run,data,w)
 for k,v in enumerate(J):w.matrix(w.stage+f'/jacobian_coefficient_{k}.json.gz',v)
 cc=H.residual_coefficients(c,s,a,b,arb(0),B.acquisitions(run,'cubic_path',0),13,18,normal,origin,sc);wc=H.residual_coefficients(c,s,a,b,d,B.acquisitions(run,'cubic_path',1),13,18,normal,origin,sc);res=[arb_mat([[cc[i,k]if k<4 else wc[i,4]]for i in range(705)])for k in range(5)]
 for k,v in enumerate(res):w.matrix(w.stage+f'/residual_coefficient_{k}.json.gz',v)
 proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[39*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];frame=[i for i in range(13)if not whole[26+i].contains(0)]
 gates=dict(uniform_root=proof['passed'],known_root=join['passed'],primitive18=all(v['coordinates']for v in proper),independent_frame=bool(frame),positive_rates_angle=all(x>0 for x in whole[-3:]));ends=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  x=sign*h;ends.append(vector_json([c[i]+s[i]*x+a[i]*x*x/2+b[i]*x*x*x+R.original_ball(corr[i])for i in range(705)]))
 scf=F(pc['settings']['parameter']);chart=dict(shooting_dimension=705,normal=pc['normal'],origin=pc['origin'],parameter_center=str(scf),parameter_endpoints=list(map(str,[scf-F(1,64),scf+F(1,64)])),settings=dict(halfwidth_exponent=-6,radius_exponent=-10));passed=all(gates.values())
 w.save(w.stage+'/independent-proof.json',dict(passed=passed,gates=gates,root=proof,anchor=join,proper_divisors=proper,frame_nonzero_coordinates=frame,chart=chart,endpoints=ends));need(passed,'all strict independent root/primitive/frame/rate gates')
 return dict(chart=chart,endpoint=ends[1]),dict(passed=passed,strict_inclusions=sum(proof['coordinate_inclusions']),anchor_coordinates=join['coordinates'],contraction=proof['contraction'],rotated=rotated,parameter_endpoints=chart['parameter_endpoints'])

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in B.THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);B.OUT=OUT;w=B.Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,900);result=dict(passed=False,connected=False,status='incomplete',native_calls=0,charts=[])
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads=dict.fromkeys(B.THREADS,'1')))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   card=read(ROOT/CARD);need(card['settings']==settings(),'fixed bounds');B.synthetic(w);old,opc,fifth,fpc,nxt,npc,point_run,pi,previous,priorout=bind(card,w);need(2*priorout['elapsed_seconds']+90<900,'actual prior complete3-root measured cost floor');left=previous['endpoint'];previous=fallback(old,opc,point_run,pi,previous,w);w.save('localized-old-endpoint.json',previous)
   previous,row=tube(fifth,fpc,previous,w,1,True);result['charts'].append(row);middle=previous['endpoint'];w.save('connected-endpoint-001.json',previous);gc.collect();previous,row=tube(nxt,npc,previous,w,2,False);result['charts'].append(row);w.save('connected-endpoint.json',previous)
   minimum=A.interior_minimum([left,middle,previous['endpoint']],True);w.save('hidden-rate-minimum.json',minimum);result.update(passed=True,connected=True,status='independently_joined_two_roots_with_interior_rate_minimum',minimum=minimum,physical_endpoints=[r[-3:]for r in [left,middle,previous['endpoint']]],scope='Independent480 full native acquired-domain cubic705 roots and full-state rotated/same-gauge joins. Shared source-authenticated native derivatives and Arb. Exact strict endpoint ordering proves at least one interior hidden-rate minimum only; no uniqueness/nondegeneracy, complementary spectrum or complete-window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
