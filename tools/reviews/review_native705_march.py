"""One bounded saved-only replay of the fixed48 accepted705 complementary cells."""
import gc,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
import flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_native705_halves as S
import review_native705_march_math as Q
A,H,R,M,need=S.A,S.H,S.R,S.M,S.need
entry,read,historical,exact_cover,equal_balls=S.entry,S.read,S.historical,S.exact_cover,S.equal_balls
acquisition,exact_direction=S.acquisition,S.exact_direction
NATIVE,PATH_FILES,THREADS=S.NATIVE,S.PATH_FILES,S.THREADS
OUT=Path('/data2/neural-shrimp-data/audits/native705-march-independent/attempt001/run')
CARD='docs/plans/native705-march-independent-inputs.json'
FILES=sorted(set(S.FILES+[CARD,'docs/plans/native705-march-independent.md','tools/reviews/review_native705_march_math.py','tools/reviews/review_native705_march.py','tools/reviews/launch_native705_march_review.py','tests/test_native705_march_review.py','docs/reviews/native705-halves-independent.json']))
_CACHE={};_HISTORY={}

def settings():return dict(precision=480,worker_seconds=1800,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=256*1024**2,closing_reserve=4*1024**2,workers=1,expected_leaves=48,original_ordinals=list(range(7,31)))
def inventory(root,pin,tick):
 key=(str(root),pin)
 if key not in _CACHE:_CACHE[key]=S.inventory(root,pin,tick)
 return _CACHE[key]
def archive(run,source):
 rt=read(run/'runtime.json');need(rt.get('commit',rt.get('source_commit'))==source and not Path('/proc',str(rt['pid'])).exists(),'exact stopped historical worker')
 key=(source,json.dumps(rt['sources'],sort_keys=True))
 if key not in _HISTORY:_HISTORY[key]=historical(run,source)['sources']
 paths={str(p.relative_to(run/'source')):p for p in(run/'source').rglob('*')if p.is_file()};need(set(paths)==set(rt['sources']),'complete covered executed source archive')
 # Complete parent inventory authenticates every byte; compare the archive
 # table against this worker manifest before sharing a git-source lookup.
 manifest=read(run/'manifest.json')['files']
 for name,pin in rt['sources'].items():need(entry(run/'source'/name)==manifest['source/'+name]==pin==_HISTORY[key][name],'same actual archived source bytes')
 return rt

def bundle(spec,tick,code=0):
 root=Path(spec['root']);rows=[inventory(root,spec['outer'],tick),inventory(root/'run',spec['worker'],tick)];rt=archive(root/'run',spec['source']);la=read(root/'launcher-result.json');need(la.get('commit',la.get('source_commit'))==spec['source']and la['exit_code']==code and not la['owned_live_pids'],'complete owned prior execution')
 for k in('launcher_pid','owned_group_pid'):
  if k in la:need(not Path('/proc',str(la[k])).exists(),'prior owner absent')
 return root,rt,read(root/'run/outcome.json'),rows

def canonical_assignment(row,packet,index):
 old=index['charts'][row['ordinal']-7];need(old['ordinal']==row['ordinal']and 7<=row['ordinal']<=30,'fixed original ordinal');Q.geometry(dict(parameter_center=old['settings']['parameter'],parameter_endpoints=old['parameter_endpoints']),row['cell'])
 need(row['root']==str(Path(packet['root'])/f"chart{row['attempt']:03d}")and type(row['attempt'])is int,'fixed selected worker path');return old

def validate_packets(card,tick):
 need(card['schema']=='fixed48-saved-independent-complement-v1'and card['settings']==settings()and len(card['packets'])==2,'fixed exact audit card');accepted=[];inventories=[];base=None
 for pi,packet in enumerate(card['packets']):
  root=Path(packet['root']);inventories.append(inventory(root,packet['outer'],tick));index=read(root/'fixed-index.json');need(entry(root/'fixed-index.json')==packet['fixed_index_identity']and index==packet['fixed_index'],'exact complete producer input index');done=read(root/'launcher-result.json')
  count=(48,24)[pi];need(done['source_commit']==packet['source']and done['launched']==len(done['attempts'])==len(packet['rows'])==count and not done['owned_live_pids']and done['failure']is None,'all fixed completed attempts including negatives')
  need([r['ordinal']for r in index['charts']]==list(range(7,31)),'all original24 fixed roots')
  common={k:index[k]for k in('charts','frozen_basis','physical_origin','retained_schema_failure')}
  if base is None:base=common
  else:need(common==base,'same original roots/proposals across resume')
  need(len({v['attempt']for v in packet['rows']})==count,'unique permanent original attempt numbers')
  decisions={v['assignment']['attempt']:v for v in done['attempts']}
  for row in packet['rows']:
   old=canonical_assignment(row,packet,index);outer=Path(row['root']);run=outer/'run';need(entry(outer/'manifest.json')==row['outer']and entry(run/'manifest.json')==row['worker']and entry(run/'outcome.json')==row['outcome'],'literal covered selected/negative unit artifacts')
   assignment=read(outer/'attempt-card.json');decision=read(outer/'decision.json');out=read(run/'outcome.json');la=read(outer/'launcher-result.json');rt=archive(run,packet['source']);expected=decisions[row['attempt']];need(rt['sources'][packet['source_index_path']]==packet['source_index']and read(run/'source'/packet['source_index_path'])==index,'exact actual executed input card versus reserialized parent index')
   need(assignment==expected['assignment']==out['assignment']and assignment['source_commit']==packet['source']and assignment['ordinal']==row['ordinal']and assignment['cell']==row['cell']and assignment['root']==str(run)and assignment['input_index']==packet['source_index'],'exact actual source/cell assignment')
   need(la['exit_code']==(0 if row['passed']else 2)and la['worker_pid_absent']and not la['owned_live_pids']and not Path('/proc',str(la['group_pid'])).exists(),'actual completed owned leaf/parent')
   need(row['action']==decision['action']==expected['action']and row['passed']==out['passed']and out['native_calls']==360 and out['profile_native_calls']==out['target_native_calls']==180 and out['unit']==row['attempt'],'all360 actual attempts retained')
   need(row['status']==out['status']and row['target_native_calls']==180 and row['native_calls']==360,'literal complete retained outcome status')
   if row['passed']:
    need(row['action']=='accept'and out['metric_passed']and out['complement_dimension']==11 and out['status']=='whole_chart_complement_attracting','complete fixed accepted leaf');accepted.append(dict(packet=pi,row=row))
   else:need(row['action']in('split','depth_limit')and out['status']=='completed_correlated_bound_unresolved'and out['metric_passed']is False,'complete retained negative bound only')
   tick()
 need(len(accepted)==48,'exact48 accepted leaves');accepted.sort(key=lambda v:(v['row']['ordinal'],F(v['row']['cell']['endpoints'][0])));Q.closed_unions([dict(passed=True,same_root=True,complement_dimension=11,ordinal=v['row']['ordinal'],cell=v['row']['cell'])for v in accepted],base['charts'])
 # The initial schema failure remains an authenticated non-scientific failure.
 fail=base['retained_schema_failure'];fr=Path(fail['root']);inventories.append(inventory(fr,fail['outer'],tick));fo=read(fr/'launcher-result.json');need(not fo['passed']and fo['launched']==4 and fo['complete_charts']==0 and not fo['owned_live_pids'],'retained initial schema failure')
 for i,pin in enumerate(fail['workers'],1):
  run=fr/f'chart{i:03d}/run';need(entry(run/'manifest.json')['sha256']==pin,'literal failed worker');archive(run,fail['source']);o=read(run/'outcome.json');need(o['native_calls']==o['profile_native_calls']==180 and o['target_native_calls']==0 and o['error']=="KeyError: 'whole_angle_positive'",'retained zero-target failure')
 return accepted,base,inventories

def shared_bind(base,ordinal,tick):
 old=base['charts'][ordinal-7];run=Path(old['producer_root']);pr=run.parents[1];unit=old['unit'];need(unit==(ordinal-7)%8+1,'fixed batch-local root');rows=[inventory(pr,old['producer_outer'],tick),inventory(run,old['producer_manifest']['sha256'],tick)];rt=archive(run,old['producer_source']);po=read(pr/'outcome.json');out=read(run/'outcome.json')
 need(po['commit']==old['producer_source']and po['passed']and po['connected_count']==8 and not po['owned_live_pids']and len(po['workers'])==8 and all(v['exit_code']==0 and not v['owned_live_pids']for v in po['workers']),'complete original8 root batch')
 need(out['passed']and out['unit']==unit and out['ticks']==253 and out['strict_inclusions']==705 and out['events']['direction_return']==144 and out['events']['pal_path_phase_return']==108 and out['gates']['whole_angle_positive'],'complete original quadratic native proof')
 spec=old['independent'];ind=dict(root=spec['root'],source=spec['source'],outer=spec['outer']['sha256'],worker=spec['worker']['sha256']);ir,irt,io,rr=bundle(ind,tick);rows+=rr;ib=read(ir/'run/input-binding.json');ic=ib['card']
 need(entry(ir/'run/input-binding.json')==spec['input_binding']and io['passed']and io['connected']and io['connected_count']==8 and io['native_calls']==0,'complete independent original graph authority')
 need(ic['root']==str(pr)and ic['source']==old['producer_source']and ic['outer']==old['producer_outer']and ic['qualified_prefix']==ic['expected_workers']==8 and len(ic['charts'])==len(ib['charts'])==8,'exact independently audited original producer scope')
 cert=run/'certificate.json.gz';need(entry(cert)==old['certificate']and ib['charts'][unit-1]==dict(root=str(run),manifest=old['producer_manifest'],certificate=old['certificate'])and ic['charts'][unit-1]==dict(path=f'chart{unit:03d}/run',unit=unit,worker=old['producer_manifest']['sha256']),'exact independent full certificate')
 up=ir/f'run/chart-{unit}/uniform.json';jp=ir/f'run/chart-{unit}/join.json';u=read(up);j=read(jp);need(entry(up)==spec['uniform']and entry(jp)==spec['join']and u['inclusions']==705 and all(v for k,v in u['gates'].items()if k!='anchor_identified')and j['passed']and j['coordinates']==705,'full705 independent uniform root and known-root join')
 need(old['same_gauge']==(unit!=1)and io['proofs'][unit-1]['same_gauge']==old['same_gauge']and io['proofs'][unit-1]['parameter_endpoints']==old['parameter_endpoints'],'retain real genealogy/gauge changes')
 need(j.get('parameter')==old['parameter_endpoints'][0]if unit>1 else j.get('parameter_inside')is True,'actual scalar or projected full705 join')
 fr,frt,fo,rr=bundle(base['frozen_basis'],tick,2);rows+=rr;prop=read(fr/'run/events/001-proposal.json');fb=read(fr/'run/bindings.json');need(not fo['passed']and fo['native_calls']==0 and fo['status']=='completed_sufficient_bound_unresolved'and prop['strategy']=='periodic_lyapunov_balanced_orthogonal'and fb['card']['producer']==base['physical_origin'],'same failed-bound proposal source')
 phys,prt,physout,rr=bundle(base['physical_origin'],tick);rows+=rr;physical=read(run/'inputs.json')['map_inputs'];need(physout['passed']and physical==read(phys/'run/inputs.json')['map_inputs'],'same literal native model/data/both rates')
 chart=read(cert);need(chart['parameter_center']==old['settings']['parameter']and chart['parameter_endpoints']==old['parameter_endpoints']and chart['settings']=={k:old['settings'][k]for k in('precision','newton_steps','halfwidth_exponent','radius_exponent')},'exact original chart settings')
 need(all(chart['gates'][k]for k in('uniform_full_native_unit_pair_locus','primitive_period','realified_frame_independent'))and u['gates']['positive_angle'],'original and independent positive/primitive/frame root gates')
 need({k:hashlib.sha256(json.dumps(chart[k],sort_keys=True,separators=(',',':')).encode()).hexdigest()for k in('normal','origin','weights')}==old['geometry_hashes'],'original complete gauge geometry')
 keep={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','center','slope','curvature','weights','coordinate_radii','normal','origin'}
 for k in list(chart):
  if k not in keep:del chart[k]
 directions=sorted((run/'events').glob('*-direction_return.json.gz'));paths=sorted((run/'events').glob('*-pal_path_phase_return.json.gz'));need(len(directions)==144 and len(paths)==108,'all original252 native returns');pathrows=[read(p)for p in paths];gc.collect();tick()
 return dict(pr=run,rt=rt,ir=ir,uniform=entry(up),join=entry(jp),cert=entry(cert),prop=prop,physical=physical,physical_runtime=prt,rows=rows,chart=chart,paths=pathrows,old=old,independent_card=ic)


def bind_unit(card,item,shared,tick):
 packet=card['packets'][item['packet']];row=item['row'];cell=row['cell'];unit=row['attempt'];ends=cell['endpoints'];root=Path(row['root'])/'run';rt=archive(root,packet['source']);out=read(root/'outcome.json');binding=read(root/'bindings.json');old=binding['original'];job=read(root.parent/'attempt-card.json')
 need(out['passed']and out['unit']==unit and out['native_calls']==360 and out['complement_dimension']==11 and out['parameter_endpoints']==ends,'literal complete accepted cell')
 need(rt['precision']==384 and rt['native_call_cap']==360 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'actual native settings')
 need(binding['assignment']==job and binding['assignment_artifact']==entry(root.parent/'attempt-card.json')and binding['cell']==cell and binding['parameter_endpoints']==ends and old['ordinal']==row['ordinal']and old['index']==shared['old']and old['card']==packet['fixed_index']and old['original_run']==str(shared['pr'])and old['independent_uniform']==shared['uniform']and old['independent_join']==shared['join']and old['map_inputs']==shared['physical']and old['independent_input_card']==shared['independent_card']and old['same_root'],'exact same-root cell/derivative binding')
 for name in NATIVE:need(rt['sources'][name]==shared['rt']['sources'][name]==shared['physical_runtime']['sources'][name],'unchanged native source '+name)
 for name in PATH_FILES:need(rt['sources'][name]==shared['rt']['sources'][name],'unchanged actual PathJet/native source '+name)
 profile=read(root/'profile.json');need(profile['passed']and profile['native_calls']==180 and profile['target_arrays_unopened']and profile['forecast_seconds']<400 and profile['forecast_bytes']<384*1024**2,'producer complete measured admission')
 dom=read(root/'quotient-domain-inputs.json');td=read(root/'tangent-domain-inputs.json');loc=read(root/'localization.json');tan=read(root/'tangent.json');need(dom['parameter_endpoints']==td['parameter_endpoints']==ends and dom['halfwidth']==cell['halfwidth']and td['scale']=='1/256','same actual closed-half gauge')
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
  need(M.point(d['parameter'])==F(cell['center']),'literal restricted center parameter')
  equal_balls(d['normal'],td['normal'],'same exact full gauge normal');equal_balls(d['origin'],td['origin'],'same exact full gauge origin');equal_balls([d['parameter']],[td['parameter']],'same exact center parameter')
 C=R.matrix(read(root/'assembly/target-preconditioner.json.gz')['C']);records=[read(root/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(root/'events/020-metric.json');gc.collect();tick()
 return fresh,paths,native,td,dom,E,metrics,C,records,metric,stage_domains,dict(ordinal=row['ordinal'],unit=unit,root=str(root),source=packet['source'],worker=row['worker'],parameter_endpoints=ends,cell=cell,complete_acquisition_pairs=360,original_certificate=shared['cert'],independent_uniform=shared['uniform'],independent_join=shared['join'],same_root=True)


def main():
 resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();used=[0];stopped=[False];result=dict(passed=False,status='incomplete',native_calls=0,rows=[])
 def save(name,value,closing=False):
  raw=S.B.canonical_bytes(value);need(used[0]+len(raw)<(256 if closing else 252)*1024**2,'256MiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used[0]+=len(raw)
 def tick():need(not stopped[0]and time.monotonic()-start<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'1800s4GiB cap')
 signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stopped.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,1800)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,nice=10,threads=dict.fromkeys(THREADS,'1')))
  for n in FILES:
   b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed source');need(used[0]+len(b)<252*1024**2,'source reserve');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
  with ctx.workprec(480):
   before=time.monotonic();trial=A.synthetic(705,tick);seconds=time.monotonic()-before;card=read(ROOT/CARD);need(card['settings']==settings(),'fixed audit caps');prior,_,priorout,rr=bundle(card['prior_measured'],tick);need(priorout['passed']and len(priorout['rows'])==2 and priorout['native_calls']==0,'actual complete previous same-math measurement')
   forecast=max(8*seconds+1440,36*priorout['elapsed_seconds']+90)
   # Storage uses the entire immutable prior two-cell inventory, not manifest length.
   raw=36*sum(v['bytes']for v in read(prior/'manifest.json')['files'].values())+8*1024**2
   rss=1.5*priorout['peak_rss_kib']+128*1024;profile=dict(trial,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,forecast_rss_kib=rss,target_arrays_unopened=True);profile['passed']=bool(trial['passed']and forecast<1800 and raw<252*1024**2 and rss<4*1024**2);save('profile.json',profile);need(profile['passed'],'measured full48 saved-reducer admission');gc.collect();accepted,base,invs=validate_packets(card,tick);save('input-binding.json',dict(card=card,inventories=invs+rr));shared=None
   for index,item in enumerate(accepted,1):
    row=item['row'];ordinal=row['ordinal'];cell=row['cell'];prefix=f'cell-{index:03d}/'
    if shared is None or shared['old']['ordinal']!=ordinal:
     shared=None;gc.collect();shared=shared_bind(base,ordinal,tick);save(f'original-{ordinal}/binding.json',dict(index=shared['old'],inventories=shared['rows'],uniform=shared['uniform'],join=shared['join'],certificate=shared['cert']))
    fresh,paths,native,td,dom,E,metrics,C,records,metric,stage,binding=bind_unit(card,item,shared,tick);save(prefix+'bindings.json',binding)
    data,proof=Q.localize(shared['chart'],shared['paths'],fresh,C,cell,tick);save(prefix+'localization-480.json',proof);del fresh,C,proof;gc.collect();tick()
    for key,field in [('center','fresh_center'),('whole','fresh_domain')]:exact_cover(stage[key]['flat'],data[field],'all705 actual fresh-J acquisition '+key)
    for key in('center','whole'):exact_cover(dom['full705'][key],data[key],'own480 root '+key)
    for key,field in [('segment','segment'),('error','error'),('predictor_tangent','pprime')]:exact_cover(td[key],data[field],'actual '+key)
    exact_cover(stage['whole']['flat'],data['segment'],'actual full705 mean-value segment');b=R.exact(F(1,256));plus=[data['error'][i]/b+b*data['pprime'][i]for i in range(705)];minus=[data['error'][i]/b-b*data['pprime'][i]for i in range(705)];exact_cover(td['plus'],plus,'actual plus');exact_cover(td['minus'],minus,'actual minus')
    tangent,tproof=H.correlated_tangent(data,paths,tick);save(prefix+'tangent-480.json',tproof);del paths,tproof;gc.collect();tick();exact_cover(dom['full705']['tangent'],tangent,'actual full705 true tangent');save(prefix+'domain-coverage.json',dict(passed=True,coordinates=705,own_intersections=6,exact_serialized_endpoint_containment=True))
    proof=H.reduction(data['center'][:-1],data['whole'][:-1],tangent[:-1],E,metrics,native,records,metric,data['halfwidth'],data['whole'][-1],tick);save(prefix+'complement-480.json',proof);answer=dict(passed=True,ordinal=ordinal,cell=cell,producer_root=row['root'],primitive_period=18,complement_dimension=11,same_root=True,complementary_modulus_upper=proof['complementary_modulus_upper']);result['rows'].append(answer);save(prefix+'outcome.json',answer);del data,tangent,td,dom,E,metrics,native,records,metric,proof,stage;gc.collect();tick()
   unions=Q.closed_unions(result['rows'],base['charts']);save('same-root-unions.json',dict(passed=True,charts=unions,scope='Exact closed unions within each original chart gauge. Across rotations, retain prior independently proved full705 joins; no scalar interpolation across gauges.'));result.update(passed=True,status='independently_qualified_all48_leaves_on24_roots',original_charts=24,accepted_leaves=48,scope='All11 complementary multipliers strictly interior on the exact original24 independently connected primitive18 charts7–30. Neutral critical pair unchanged. Shared native acquisitions and Arb; independent full705 localization/tangent/quotient arithmetic. No signed NS criticality or complete-window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
