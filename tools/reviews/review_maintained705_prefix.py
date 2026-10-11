"""Independent480 fixed maintained705 prefix; no native acquisitions."""
import gc,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import ctx
import review_cubic705_prefix as B
H=B.H;OLD=B.OLD;need=B.need;read=B.read;entry=B.entry;inventory=B.inventory;historical=B.historical;THREADS=B.THREADS
OUT=Path('/data2/neural-shrimp-data/audits/native705-maintained-independent/attempt001/run');CARD='docs/plans/native705-maintained-independent-inputs.json'
FILES=[CARD,'docs/plans/native705-maintained-independent.md','tools/reviews/review_maintained705_prefix.py','tools/launch_maintained705_review.py','tests/test_maintained705_review.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def selection(card):
 need(card['selected_attempts']==[1,2,4] and [x['attempt']for x in card['producer']['workers']]==[1,2,3,4,5],'fixed three accepted charts and all five preserved workers')
 return card['selected_attempts']

def bind(card,w):
 selected=selection(card);ar,rt,ao,rows=OLD.bundle(card['anchor'],w.tick);ep=read(ar/'run/connected-endpoint.json');ib=read(ar/'run/input-binding.json');proof=read(ar/'run/new-root/independent-proof.json')
 need(ao['passed']and ao['connected']and ao['strict_inclusions']==ao['anchor_coordinates']==705 and ao['primitive_period']==18 and ao['native_calls']==0 and rt['precision']==480,'independent complete rotated root and endpoint join')
 need(proof['passed']and ep['chart']==proof['chart']and ep['endpoint']==proof['endpoints'][1]and ep['chart']['parameter_endpoints']==['-1/2048','63/2048']and entry(ar/'run/connected-endpoint.json')['sha256']==card['anchor']['endpoint'],'literal independently proved endpoint')
 spec=card['producer'];root=Path(spec['root']);rows.append(inventory(root,spec['outer'],w.tick));out=read(root/'outcome.json');start=read(root/'launcher-start.json')
 need(out['source']==start['source']==spec['source']and out['status']=='mathematical_stop'and not out['passed']and out['connected_count']==3 and out['attempted']==out['launched_attempts']==5 and not out['owned_live_pids'],'complete maintained prefix and failed suffix')
 need(not Path('/proc',str(start['pid'])).exists(),'producer launcher exited')
 need([x['attempt']for x in out['attempts']]==[1,2,3,4,5]and [x['root_passed']for x in out['attempts']]==[True,True,False,True,True]and [x['joined']for x in out['attempts']]==[True,True,False,True,False],'preserved exact successful/failed ordering')
 old=Path(ib['card']['new_producer']['root']);rows.append(inventory(old,ib['card']['new_producer']['outer'],w.tick));oldcard=read(old/'attempt002/input-card.json');indexed=[]
 for v in spec['workers']:
  i=v['attempt'];base=root/f'attempt{i:03d}';run=base/'run';rows.append(inventory(run,v['worker'],w.tick));rt=historical(run,spec['source']);pc=read(base/'input-card.json');inp=read(run/'inputs.json');po=read(run/'outcome.json');owner=read(base/'owned-process.json');la=read(base/'launcher-result.json')
  need(not la['owned_live_pids']and la['exit_code']==(2 if i==3 else 0)and not Path('/proc',str(owner['group'])).exists(),'every original owned worker exited')
  need(rt['attempt']==pc['attempt']==po['attempt']==i and pc['source']==spec['source']and pc['kernel']=='fresh-cubic705-maintained-v1'and rt['settings']==pc['settings']==inp['settings']and inp['card']==entry(base/'input-card.json'),'exact actual worker source/settings/input')
  need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10 and pc['map_inputs']==inp['map_inputs']==oldcard['map_inputs'],'same entire model and single-thread runtime')
  need(pc['anchor']==inp['anchor'],'original anchor declaration preserved')
  for n in OLD.NATIVE:need(rt['sources'][n]==ib['source_native'][n],'source-identical native map '+n)
  need(po['ticks']==331 and po['events']==B.EXPECTED,'all actual native acquisitions even failed attempts')
  events=sorted((run/'events').glob('*.json.gz'));need(len(events)==sum(B.EXPECTED.values())and[int(p.name.split('-')[0])for p in events]==list(range(len(events))),'contiguous full event stream')
  for k,num in B.EXPECTED.items():need(len(list((run/'events').glob('*-'+k+'.json.gz')))==num,'all original native event kinds')
  if i in selected:
   need(po['passed']and po['status']=='uniform705_unjoined'and not po['connected']and po['strict_inclusions']==705 and pc['mode']=='same_gauge','retained exact full root acquisition')
   need(pc['normal']==ep['chart']['normal']and pc['origin']==ep['chart']['origin']and pc['anchor']['normal']==pc['normal']and pc['anchor']['origin']==pc['origin'],'same literal independent affine gauge')
   indexed.append((run,pc))
 w.save('input-binding.json',dict(card=card,inventories=rows,source_native=ib['source_native'],selected_attempts=selected,all_attempts_retained=5,producer_proofs_unparsed=True));return indexed,ep

def main():
 resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);B.OUT=OUT;w=B.Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,900);result=dict(passed=False,connected=False,native_calls=0,status='incomplete',charts=[])
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0)))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'own exact source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
  with ctx.workprec(480):
   card=read(ROOT/CARD);need(card['audit']==dict(output=str(OUT),precision=480,seconds=900,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=2048*1024**2,charts=3),'fixed bounded resource card');B.synthetic(w);runs,previous=bind(card,w)
   for unit,(run,pc)in enumerate(runs,1):
    need(pc['anchor']['exact_parameter']==previous['chart']['parameter_endpoints'][1],'actual prior exact scalar identity')
    previous,rec=B.check_one(run,pc['settings'],pc,previous,w,unit,allowed_widths=(-6,));rec['attempt']=pc['attempt'];result['charts'].append(rec);need(rec['passed'],'all independent full root/join/divisor/frame gates');w.save(f'connected-endpoint-{unit:03d}.json',previous);w.tick();gc.collect()
   need(previous['chart']['parameter_endpoints'][1]=='63/512'and len(result['charts'])==3,'complete fixed connected prefix');w.save('connected-endpoint.json',previous);result.update(passed=True,connected=True,status='independently_connected_maintained705_prefix',connected_count=3,connected_right='63/512',physical_right=previous['endpoint'][-3:],scope='Independent480 full705 acquired-domain polynomial root reductions and exact inherited same-gauge joins. Shared native acquisitions and Arb. No complementary spectrum, rate turn or complete window claim.')
 except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
