"""Fixed18 saved cubic charts and strict known-junction membership; zero native."""
import gc,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,ctx
import review_maintained705_third as T
B,H,R,M,A,need,read,entry,inventory,historical=T.B,T.H,T.R,T.M,T.A,T.need,T.read,T.entry,T.inventory,T.historical
OUT=Path('/data2/neural-shrimp-data/audits/native705-junction-chain-independent/attempt001/run');T.OUT=OUT
CARD='docs/plans/native705-maintained004-independent-inputs.json'
SELECTED=[1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,19,20]
FILES=sorted(set([CARD,'docs/plans/native705-maintained004-independent.md','tools/reviews/review_maintained705_junction.py','tools/launch_maintained705_junction_review.py','tests/test_maintained705_junction_review.py']+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_file()and Path(m.__file__).resolve().is_relative_to(ROOT)]))


def selection(card):
 need(card['status']=='reviewed'and card['schema']=='independent-maintained004-junction-v1'and card['selected_attempts']==SELECTED and card['failed_attempts']==[10,18],'fixed18 plus two retained failures')
 need(isinstance(card['anchor'],dict)and set(card['anchor'])=={'root','source','outer','worker','endpoint','proof'},'complete finished independent predecessor required')
 rows=card['producer']['workers'];need([v['attempt']for v in rows]==list(range(1,21))and[v['attempt']for v in rows if v['joined']]==SELECTED,'all20 actual histories')
 need(all(v['root_passed']==v['joined']and v['mode']==('same_gauge'if v['attempt']==1 else'rotated_gauge')and v['settings']['halfwidth_exponent']==(-3 if v['attempt']<=10 else -4 if v['attempt']<=18 else -5)for v in rows),'actual rotations and adaptive widths retained');return rows


def bind(card,w):
 specs=selection(card);anchor,rt,ao,inv=B.OLD.bundle(card['anchor'],w.tick);ep=read(anchor/'run/connected-endpoint.json');ib=read(anchor/'run/input-binding.json');pr=read(anchor/'run/chart024/independent-proof.json')
 need(ao['passed']and ao['connected']and ao['connected_count']==24 and ao['native_calls']==0 and rt['precision']==480,'all24 preceding independently connected authority')
 need(entry(anchor/'run/connected-endpoint.json')==card['anchor']['endpoint']and entry(anchor/'run/chart024/independent-proof.json')==card['anchor']['proof']and ep['chart']==pr['chart']and ep['endpoint']==pr['endpoints'][1],'literal independently computed preceding endpoint')
 need(pr['passed']and all(pr['gates'].values())and all(pr['root']['coordinate_inclusions'])and pr['anchor']['passed']and pr['anchor']['coordinates']==705 and ep['chart']['parameter_endpoints'][1]=='511/2048','entire exact preceding root/gauge')
 p=card['producer'];root=Path(p['root']);inv.append(inventory(root,p['outer'],w.tick));po=read(root/'outcome.json');start=read(root/'launcher-start.json');need(po['source']==start['source']==p['source']and po['passed']and po['status']=='known_junction_connected'and po['attempted']==po['launched_attempts']==20 and po['connected_count']==18 and not po['owned_live_pids']and po['fallback_count']==0 and not Path('/proc',str(start['pid'])).exists(),'complete20-attempt stopped-at-junction packet')
 indexed=[];prior=None
 for v,old in zip(specs,po['attempts'],strict=True):
  i=v['attempt'];base=root/f'attempt{i:03d}';run=base/'run';inv.append(inventory(base,v['outer'],w.tick));inv.append(inventory(run,v['worker'],w.tick));rt=historical(run,p['source']);pc=read(base/'input-card.json');inp=read(run/'inputs.json');o=read(run/'outcome.json');own=read(base/'owned-process.json');la=read(base/'launcher-result.json')
  need(old['attempt']==i and old['worker']['sha256']==v['worker']and old['outer_manifest']['sha256']==v['outer']and old['joined']==v['joined']and old['root_passed']==v['root_passed'],'entire literal accepted/failed order')
  need(la['exit_code']==(0 if v['joined']else 2)and not la['owned_live_pids']and not Path('/proc',str(own['group'])).exists(),'complete owned attempted chart')
  need(entry(base/'input-card.json')==v['card']and pc['source']==p['source']and pc['attempt']==rt['attempt']==o['attempt']==i and pc['mode']==v['mode']and pc['kernel']=='fresh-cubic705-maintained-v1','exact source/card/mode')
  need(pc['settings']==rt['settings']==inp['settings']==v['settings']and pc['map_inputs']==inp['map_inputs']==ib['model']and pc['parent']==inp['parent']==v['parent']and pc['anchor']==inp['anchor']and inp['card']==v['card'],'complete native physical/domain association')
  for n in A.NATIVE:need(rt['sources'][n]==ib['native_source'][n],'unchanged actual derivative source '+n)
  need(rt['threads']==dict.fromkeys(B.THREADS,'1')and rt['nice']>=10 and o['passed']==v['root_passed']and o['ticks']==331 and o['events']==B.EXPECTED and not o['connected'],'all actual331-phase acquisitions retained')
  ev=sorted((run/'events').glob('*.json.gz'));need(len(ev)==sum(B.EXPECTED.values())and[int(f.name.split('-')[0])for f in ev]==list(range(len(ev))),'all original event indices')
  for k,num in B.EXPECTED.items():need(len(list((run/'events').glob('*-'+k+'.json.gz')))==num,'actual phase/event kind counts')
  if i==1:
   inherited=ib['card']['producer'];last=inherited['workers'][-1];expected=dict(kind='accepted-maintained-endpoint',root=str(Path(inherited['root'])/'attempt024'),source=inherited['source'],outer=last['outer'],worker=last['worker'],endpoint=last['endpoint']['sha256'])
   need(pc['parent']==expected and ep['chart']['normal']==pc['normal']and ep['chart']['origin']==pc['origin'],'literal initial independently proved predecessor')
  else:need(pc['parent']==dict(kind='accepted-maintained-endpoint',root=str(root/f"attempt{prior['attempt']:03d}"),source=p['source'],outer=prior['outer'],worker=prior['worker'],endpoint=prior['endpoint']['sha256']),'last accepted parent, never failed chart')
  if v['joined']:
   need(o['status']=='uniform705_unjoined'and o['strict_inclusions']==705 and entry(base/'connected-endpoint.json')==v['endpoint'],'whole original positive root/endpoint');prior=v;indexed.append((run,pc))
 need([pc['attempt']for _,pc in indexed]==SELECTED,'fixed selected18 queue');w.save('input-binding.json',dict(card=card,inventories=inv,model=ib['model'],native_source=ib['native_source'],all_attempts_retained=20,derived_producer_proofs_unparsed=True));return indexed,ep,ib


def same_equations(lift,curve,physical):
 need(physical==curve['model'],'literal identical known-junction and continued training model')
 need(lift['native_source']and all(v==curve['native_source'][n]for n,v in lift['native_source'].items()),'same actual junction/continuation native derivative sources')


def known_lift(card,w,curve_binding):
 x=card['lift'];root=Path(x['root']);inventory(root,x['outer']['sha256'],w.tick);inventory(root/'run',x['worker']['sha256'],w.tick);rt=read(root/'run/runtime.json');o=read(root/'run/outcome.json');la=read(root/'launcher-result.json')
 need(rt['source_commit']==la['source_commit']==x['source']and o==x['outcome']and o['passed']and o['coordinates']==705 and o['native_calls']==0 and la['exit_code']==0 and not la['owned_live_pids']and la['worker_pid_absent'],'completed independently exact-rational junction lift')
 for pid in[rt['pid'],la['launcher_pid'],la['owned_group_pid']]:need(not Path('/proc',str(pid)).exists(),'actual independent owner absent')
 for n,v in rt['files'].items():need(entry(root/'run/source'/n)==v and (root/'run/source'/n).read_bytes()==subprocess.check_output(['git','show',x['source']+':'+n],cwd=ROOT),'literal independent lift archive')
 binding=read(root/'run/input-binding.json');need(binding==x['binding'],'literal independent known-lift binding')
 producer=Path(binding['producer']);inventory(producer,binding['producer_pins'][0],w.tick);inventory(producer/'run',binding['producer_pins'][1],w.tick);physical=read(producer/'run/input-binding.json')['map_inputs'];same_equations(binding,curve_binding,physical)
 z=read(root/'run/independent705.json');need(entry(root/'run/independent705.json')==x['independent705']and z['passed']and z['coordinates']==705,'full exact lifted known junction');return z['box']


def membership(data,pc,box):
 c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=data
 proof=H.cubic_membership(box,c,s,a,b,normal,origin,R.exact(F(pc['settings']['parameter'])),h,radii,known_parameter=None);need(proof['passed']and proof['parameter_inside']and proof['coordinates']==705,'strict independent whole-known-junction membership');return proof


def junction(run,pc,box):
 data,_,_=T.geometry(run,pc);return membership(data,pc,box)


def synthetic(w,card):
 C,J,res,h,rho=B.synthetic_fixture();before=time.monotonic();used=w.used;pr=H.polynomial_root(C,J,res,[arb(1)]*705,rho,h,[arb(0,rho)]*705,w.tick,w.emit);need(pr['passed'],'full705 degree5 unrelated profile');elapsed=time.monotonic()-before;stored=w.used-used
 ar=Path(card['anchor']['root']);need(entry(ar/'manifest.json')['sha256']==card['anchor']['outer'],'literal completed actual24 timing');o=read(ar/'run/outcome.json');mi=read(ar/'manifest.json')['files'];need(mi['run/outcome.json']==entry(ar/'run/outcome.json')and o['passed']and o['connected_count']==24 and o['native_calls']==0,'actual independently measured24')
 seconds=max(elapsed+18*(3*elapsed+75),2*18*o['elapsed_seconds']/24+90+elapsed);raw=max(55*stored+128*1024**2,2*18*sum(v['bytes']for v in mi.values())/24+128*1024**2);peak=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3,2*o['peak_rss_kib']*1024+256*1024**2);report=dict(passed=seconds<7200 and raw<14*1024**3 and peak<6*1024**3,seconds=elapsed,stored_bytes=stored,forecast_seconds=seconds,forecast_bytes=raw,forecast_peak_bytes=peak,target_arrays_unopened=True);w.save('profile.json',report);need(report['passed'],'full fixed18 measured admission');del C,J,res;gc.collect()


def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in B.THREADS),'cleanCPU1nice10source');OUT.mkdir(parents=True,exist_ok=False);w=T.Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,7200);result=dict(passed=False,connected=False,native_calls=0,charts=[])
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(B.THREADS,'1')))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'literal own source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);w.used+=len(data)
  with ctx.workprec(480):
   card=read(ROOT/CARD);selection(card);synthetic(w,card);runs,previous,curve_binding=bind(card,w);box=known_lift(card,w,curve_binding)
   for i,(run,pc)in enumerate(runs,1):
    previous,row=T.tube(run,pc,previous,w,i,pc['mode']=='rotated_gauge');row['attempt']=pc['attempt'];result['charts'].append(row);w.save(f'connected-endpoint-{i:03d}.json',previous);gc.collect();w.tick()
   need(len(result['charts'])==18 and runs[-1][1]['attempt']==20,'all18 independently connected charts');membership=junction(*runs[-1],box);w.save('junction-membership.json',membership);w.save('connected-endpoint.json',previous);result.update(passed=True,connected=True,connected_count=18,junction_connected=True,membership_coordinates=705,status='independently_connected_to_same_unit_pair_flip_junction',scope='Independent full705 root and ordered joins from maintained003 endpoint through18 charts to same independently certified unit-pair/simple-flip junction. Both failed attempts retained. This is global connection of this critical branch; complementary stability along every later chart and other window boundaries are separate proofs.')
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
