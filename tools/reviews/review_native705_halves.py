"""Fixed saved-only independent complement replay on two halves of one known705 root."""
import gc,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_native705_complement as A
import review_native705_halves_math as Q
H,B=Q.H,A.B
need,R,M=Q.need,Q.R,Q.M
entry,read,inventory,historical,bundle,exact_cover,equal_balls,save_bound=A.entry,A.read,A.inventory,A.historical,A.bundle,A.exact_cover,A.equal_balls,A.save_bound
THREADS,NATIVE,PATH_FILES=A.THREADS,A.NATIVE,A.PATH_FILES
OUT=Path('/data2/neural-shrimp-data/audits/native705-halves-independent/attempt001/run')
CARD='docs/plans/native705-halves-independent-inputs.json'
FILES=sorted(set(A.FILES+[CARD,'docs/plans/native705-halves-independent.md','tools/reviews/review_native705_halves.py','tools/reviews/review_native705_halves_math.py','tools/reviews/launch_native705_halves_review.py','tests/test_native705_halves_review.py','tests/test_native705_halves_driver.py']))


def settings():return dict(output=str(OUT),units=[1,2],precision=480,seconds=240,rss_bytes=4*1024**3,as_bytes=6*1024**3,output_bytes=64*1024**2,closing_reserve_bytes=2*1024**2,own_intersections=6,forecast_multiplier=8,forecast_allowance_seconds=150)


def complete_pair(row,source,exit_code=0):
 need(row['source_commit']==source and row['passed']==(exit_code==0)and not row['owned_live_pids']and len(row['workers'])==2,'complete fixed two-worker result')
 for i,v in enumerate(row['workers'],1):need(v['unit']==i and v['exit_code']==exit_code and not v['owned_live_pids']and v['worker_pid_absent']and not Path('/proc',str(v['group_pid'])).exists(),'each exact owned worker ended')


def closed_union(rows):
 need(len(rows)==2 and all(v['passed']and v['unit']==i for i,v in enumerate(rows,1)),'both exact half results')
 ends=[[F(v)for v in row['parameter_endpoints']]for row in rows]
 need(ends==[Q.geometry(1)[1],Q.geometry(2)[1]]and ends[0][1]==ends[1][0]and [ends[0][0],ends[1][1]]==[F(319,4096),F(447,4096)],'closed no-gap same-gauge union')
 return list(map(str,[ends[0][0],ends[1][1]]))


def shared_bind(card,tick):
 need(card['schema']=='independent-two-half705-complement-v1'and card['audit']==settings()and card['inputs']['units']==[1,2],'fixed exact two-half input card')
 old=card['inputs']['original'];pr=Path(old['producer']['root']);qr=Path(card['complement']['root']);run=pr/'chart001/run';rows=[inventory(pr,old['producer']['outer'],tick),inventory(run,old['producer']['worker'],tick),inventory(qr,card['complement']['outer'],tick)];rt=historical(run,old['producer']['source']);po=read(pr/'outcome.json');out=read(run/'outcome.json')
 need(po['commit']==old['producer']['source']and not po['passed']and po['connected_count']==1 and not po['owned_live_pids']and len(po['workers'])==4 and [v['unit']for v in po['workers']]==[1,2,3,4]and [v['exit_code']for v in po['workers']]==[0,2,2,2]and all(not v['owned_live_pids']for v in po['workers']),'retain original one-qualified plus three failed charts')
 need(out['passed']and out['unit']==1 and out['ticks']==253 and out['strict_inclusions']==705 and out['events']['direction_return']==144 and out['events']['pal_path_phase_return']==108,'full known-root original history')
 complete_pair(read(qr/'launcher-result.json'),card['complement']['source'])
 ir,irt,io,rr=bundle(old['independent'],tick);rows+=rr;ib=read(ir/'run/input-binding.json')
 need(io['passed']and io['connected']and io['connected_count']==1 and io['native_calls']==0 and io['connected_parameter_endpoints']==['319/4096','447/4096'],'independently proved same complete primitive root graph')
 need(ib['card']==old['independent_input_card']and ib['card']['root']==str(pr)and ib['card']['outer']==old['producer']['outer']and ib['card']['source']==old['producer']['source']and len(ib['charts'])==1,'same exact independent root/genealogy card')
 cert=run/'certificate.json.gz';need(ib['charts']==[dict(root=str(run),manifest=entry(run/'manifest.json'),certificate=entry(cert))],'literal independently proved chart')
 up=ir/'run/chart-1/uniform.json';jp=ir/'run/chart-1/join.json';u=read(up);j=read(jp)
 need(u['inclusions']==705 and all(v for k,v in u['gates'].items()if k!='anchor_identified')and j['passed']and j['coordinates']==705 and j['parameter']=='319/4096','independent full705 root and exact prior-root join')
 previous=card['prior_connected'];ar,art,ao,rr=bundle(previous,tick);rows+=rr
 need(ib['prior']==str(ar)and ib['prior_manifest']==previous['outer']==ib['card']['prior']['outer']and ao['passed']and ao['connected']and ao['connected_count']==4 and ao['native_calls']==0,'same independently connected predecessor through319/4096')
 fr,frt,fo,rr=bundle(old['frozen_basis'],tick,2);rows+=rr;prop=read(fr/'run/events/001-proposal.json');fb=read(fr/'run/bindings.json');need(not fo['passed']and fo['native_calls']==0 and fo['status']=='completed_sufficient_bound_unresolved'and prop['strategy']=='periodic_lyapunov_balanced_orthogonal'and fb['card']['producer']==old['physical_origin'],'same original fixed E/B proposals')
 phys,prt,physout,rr=bundle(old['physical_origin'],tick);rows+=rr;need(physout['passed']and entry(phys/'run/certificate.json.gz')['sha256']==old['physical_origin']['certificate'],'literal original physical root')
 physical=read(run/'inputs.json')['map_inputs'];need(physical==read(phys/'run/inputs.json')['map_inputs'],'same actual model/data/both rates')
 # Both failed attempts remain immutable inputs; neither supplies a positive bound.
 fail=card['inputs']['failed_wide'];froot=Path(fail['root']);rows+=[inventory(froot,fail['outer'],tick),inventory(froot/'chart001/run',fail['worker'],tick)];historical(froot/'chart001/run',fail['source']);fout=read(froot/'chart001/run/outcome.json')
 need(not fout['passed']and fout['status']=='completed_correlated_bound_unresolved'and fout['native_calls']==216 and read(froot/'chart001/run/bindings.json')['card']==old and not read(froot/'launcher-result.json')['owned_live_pids'],'retained whole-chart bound failure')
 failed=card['inputs']['failed_profile'];froot=Path(failed['root']);rows.append(inventory(froot,failed['outer'],tick));complete_pair(read(froot/'launcher-result.json'),failed['source'],2)
 for i,pin in enumerate(failed['workers'],1):
  rr=froot/f'chart{i:03d}/run';rows.append(inventory(rr,pin,tick));historical(rr,failed['source']);fout=read(rr/'outcome.json');pf=read(rr/'profile.json');need(not fout['passed']and fout['native_calls']==180 and 'storage admission'in fout['error']and pf['target_arrays_unopened']and not(rr/'bindings.json').exists(),'original profile-only storage rejection')
 directions=sorted((run/'events').glob('*-direction_return.json.gz'));paths=sorted((run/'events').glob('*-pal_path_phase_return.json.gz'));need(len(directions)==144 and len(paths)==108,'full original252 derivative returns')
 chart=read(cert);keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','curvature','weights','coordinate_radii','normal','origin'}
 for k in list(chart):
  if k not in keep:del chart[k]
 pathrows=[read(p)for p in paths];gc.collect();tick()
 return dict(pr=run,qr=qr,rt=rt,ir=ir,uniform=entry(up),join=entry(jp),cert=entry(cert),prop=prop,physical=physical,physical_runtime=prt,rows=rows,chart=chart,paths=pathrows)


def exact_direction(a,b):
 need(a['phase']==b['phase']and a['direction']==b['direction'],'literal direction identity')
 equal_balls(a['input'],b['input'],'actual full direction input')
 for k in ('value','block','rates'):
  need(len(a[k])==len(b[k]),'all direction rows')
  for x,y in zip(a[k],b[k],strict=True):equal_balls(x,y,'actual reconstructed '+k)


def acquisition(root,index):
 need(type(index)is int and 1<=index<=360,'fixed360 acquired pairs only')
 a=read(root/f'native/{index:03d}-input.json');b=read(root/f'native/{index:03d}-return.json');j=(index-1)%180
 if j<72:
  stage=('profile_'if index<=180 else'target_')+('center'if j<36 else'whole');need(a['stage']==b['stage']==stage,'full Jacobian acquisition order');r=Q.mixed_direction(a,b);need(r['phase']==(j%36)//2 and r['direction']==('q'if j%2==0 else'r'),'ordered raw q/r phase');return 'mixed',r
 if j<144:
  virtual=index-72 if index<=180 else index-180-72+72
  return 'path',A.path_event(a,b,virtual)
 kind='synthetic'if index<=180 else('center'if index%2 else'whole_tangent');phase=index-145 if index<=180 else(index-325)//2
 need(a['kind']==b['kind']==kind and a['phase']==b['phase']==phase,'all native derivative acquisition order');return 'native',(a,b)


def bind_unit(card,unit,shared,tick):
 _,ends=Q.geometry(unit);ends=list(map(str,ends));root=shared['qr']/f'chart{unit:03d}/run';rows=[inventory(root,card['complement']['workers'][unit-1],tick)];rt=historical(root,card['complement']['source']);out=read(root/'outcome.json');binding=read(root/'bindings.json');old=binding['original']
 need(out['passed']and out['status']=='whole_chart_complement_attracting'and out['unit']==unit and out['native_calls']==360 and out['complement_dimension']==11 and out['parameter_endpoints']==ends,'literal complete two-half complement unit')
 need(rt['precision']==384 and rt['native_call_cap']==360 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'actual native settings')
 need(binding['card']==card['inputs']and binding['unit']==unit and binding['parameter_endpoints']==ends and old['card']==card['inputs']['original']and old['original_run']==str(shared['pr'])and old['original_certificate']==shared['cert']and old['independent_uniform']==shared['uniform']and old['independent_join']==shared['join']and old['map_inputs']==shared['physical'],'exact same-root restriction/derivative binding')
 for name in NATIVE:need(rt['sources'][name]==shared['rt']['sources'][name]==shared['physical_runtime']['sources'][name],'unchanged native source '+name)
 for name in PATH_FILES:need(rt['sources'][name]==shared['rt']['sources'][name],'unchanged actual PathJet/native source '+name)
 profile=read(root/'profile.json');need(profile['passed']and profile['native_calls']==180 and profile['target_arrays_unopened']and profile['forecast_seconds']<400 and profile['forecast_bytes']<384*1024**2,'producer complete measured admission')
 dom=read(root/'quotient-domain-inputs.json');td=read(root/'tangent-domain-inputs.json');loc=read(root/'localization.json');tan=read(root/'tangent.json');need(dom['parameter_endpoints']==td['parameter_endpoints']==ends and dom['halfwidth']=='1/128'and td['scale']=='1/256','same actual closed-half gauge')
 with ctx.workprec(384):full={'center':A.restored(loc['center_box']),'whole':A.restored(loc['whole_box']),'tangent':A.restored([v[0]for v in tan['tangent_box']])}
 for k,v in full.items():need(len(v)==705,'all705 actual root/tangent coordinates');equal_balls(dom['full705'][k],v,'actual restored '+k);equal_balls(dom[k],v[:-1],'drop only angle '+k)
 E=[R.matrix(v)for v in dom['complements']];metrics=[R.matrix(v)for v in dom['metrics']];need(len(E)==len(metrics)==18,'all18 frozen frames')
 for j in range(18):need(E[j]==R.matrix(shared['prop']['phase_complements'][j])and metrics[j]==arb_mat([[R.exact(F.from_float(x))for x in row]for row in shared['prop']['upper_cholesky_factors'][j]]),'exact E/B finite proposals')
 A.native_inventory(root,360);fresh=[];paths=[];native=[]
 for i in range(1,361):
  kind,got=acquisition(root,i)
  if kind=='mixed':
   stage=('profile_'if i<=180 else'target_')+('center'if(i-1)%180<36 else'whole');exact_direction(got,read(root/f'assembly/{stage}-{got["phase"]:02d}-{got["direction"]}.json'))
   if i>180:fresh.append(got)
  elif kind=='path':
   if i>180:paths.append(got)
  elif i>180:
   a,b=got;k='center'if i%2 else'whole_tangent';j=(i-325)//2;mat,derivative=A.native_return(a,b,dom,k,j)
   if k=='center':A0=mat
   else:native.append((A0,mat,derivative))
  tick()
 need(len(fresh)==len(paths)==72 and len(native)==18,'all target72mixed72path36quotient returns')
 stage_domains={k:read(root/f'assembly/target_{k}-domain.json')for k in('center','whole')}
 for d in stage_domains.values():
  need(len(d['flat'])==len(d['normal'])==len(d['origin'])==705,'complete fresh assembly domain')
  for key in ('normal','origin'):equal_balls(d[key],Q.vector_json([R.exact(F(v))for v in shared['chart'][key]]),'literal original full gauge '+key)
  need(M.point(d['parameter'])==Q.geometry(unit)[0],'literal restricted center parameter')
  equal_balls(d['normal'],td['normal'],'same exact full gauge normal');equal_balls(d['origin'],td['origin'],'same exact full gauge origin');equal_balls([d['parameter']],[td['parameter']],'same exact center parameter')
 C=R.matrix(read(root/'assembly/target-preconditioner.json.gz')['C']);records=[read(root/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(root/'events/020-metric.json');gc.collect();tick()
 return fresh,paths,native,td,dom,E,metrics,C,records,metric,stage_domains,dict(unit=unit,root=str(root),source=card['complement']['source'],worker=card['complement']['workers'][unit-1],inventories=rows,parameter_endpoints=ends,complete_acquisition_pairs=360,original_certificate=shared['cert'],independent_uniform=shared['uniform'],independent_join=shared['join'],same_root=True)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();used=[0];stopped=[False];result=dict(passed=False,status='incomplete',native_calls=0,rows=[])
 def save(n,v,closing=False):save_bound(OUT,n,v,used,closing)
 def tick():need(not stopped[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s4GiB cap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(used[0]+len(b)<62*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.synthetic(705,tick);seconds=time.monotonic()-before;forecast=8*seconds+150;save('profile.json',dict(trial,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<240,'full705 measured saved-reducer admission');gc.collect();card=read(ROOT/CARD);shared=shared_bind(card,tick);save('input-binding.json',dict(card=card,inventories=shared['rows']))
   for unit in(1,2):
    prefix=f'half-{unit}/';fresh,paths,native,td,dom,E,metrics,C,records,metric,stage,binding=bind_unit(card,unit,shared,tick);save(prefix+'bindings.json',binding)
    data,proof=Q.localize(shared['chart'],shared['paths'],fresh,C,unit,tick);save(prefix+'localization-480.json',proof);del fresh,C,proof;gc.collect();tick()
    for key,field in [('center','fresh_center'),('whole','fresh_domain')]:exact_cover(stage[key]['flat'],data[field],'all705 actual fresh-J acquisition '+key)
    for key in('center','whole'):exact_cover(dom['full705'][key],data[key],'own480 root '+key)
    for key,field in [('segment','segment'),('error','error'),('predictor_tangent','pprime')]:exact_cover(td[key],data[field],'actual '+key)
    exact_cover(stage['whole']['flat'],data['segment'],'actual full705 mean-value segment');b=R.exact(F(1,256));plus=[data['error'][i]/b+b*data['pprime'][i]for i in range(705)];minus=[data['error'][i]/b-b*data['pprime'][i]for i in range(705)];exact_cover(td['plus'],plus,'actual plus');exact_cover(td['minus'],minus,'actual minus')
    tangent,tproof=H.correlated_tangent(data,paths,tick);save(prefix+'tangent-480.json',tproof);del paths,tproof;gc.collect();tick();exact_cover(dom['full705']['tangent'],tangent,'actual full705 true tangent');save(prefix+'domain-coverage.json',dict(passed=True,coordinates=705,own_intersections=6,exact_serialized_endpoint_containment=True))
    proof=H.reduction(data['center'][:-1],data['whole'][:-1],tangent[:-1],E,metrics,native,records,metric,data['halfwidth'],data['whole'][-1],tick);save(prefix+'complement-480.json',proof);row=dict(passed=True,unit=unit,parameter_endpoints=list(map(str,Q.geometry(unit)[1])),primitive_period=18,complement_dimension=11,same_root=True,complementary_modulus_upper=proof['complementary_modulus_upper']);result['rows'].append(row);save(prefix+'outcome.json',row);del data,tangent,td,dom,E,metrics,native,records,metric,proof,stage;gc.collect();tick()
   union=closed_union(result['rows']);save('same-root-union.json',dict(passed=True,parameter_endpoints=union,shared_endpoint='383/4096',same_exact_gauge=True,root_authority=card['inputs']['original']['independent'],coordinates=705));result.update(passed=True,status='independently_qualified_same_root_two_half705_complements',parameter_endpoints=union,scope='Both closed restrictions of the same independently connected primitive18 root graph; all11 complementary moduli strictly interior uniformly. Critical unit pair neutral. Shared source-authenticated native acquisitions and Arb; independent recentering, full705 assembly/localization/tangent and quotient/metric arithmetic. No later-chart, signed NS coefficient, stable-side or full-window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
